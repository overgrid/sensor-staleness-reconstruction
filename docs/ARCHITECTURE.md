# Architecture

Detailed breakdown of every module: what it does, why it's shaped the way
it is, and what it deliberately does NOT do yet. Read this alongside the
docstrings in each file — this document explains the *reasoning* behind
decisions; the docstrings explain the *mechanics*.

## Related work

Before investing further in Kafka/automation, a survey of existing
projects tackling similar problems was done — see
[`RELATED_WORK.md`](RELATED_WORK.md). Short version: no existing
open-source project combines stuck-detection + Chronos-based
reconstruction + honest baseline validation + this specific GraphQL/MLflow
integration. The closest single match is
[PyPOTS](https://github.com/WenjieDu/PyPOTS), a general-purpose imputation
library — worth knowing about, not a replacement for this pipeline.

## Data flow, end to end

```
CSV file (data_source.py)
    → pandas.Series, one per sensor
        → detection.py: find stuck periods
            → reconstruction.py: reconstruct each one (using chronos_model.py)
                → storage.py: write results
        → synthetic_injection.py: validate accuracy (using reconstruction.py)
            → tracking.py: log results to MLflow

offline_job.py orchestrates all of the above.
cli.py exposes it as the `staleness offline` command.
```

Every module below is independently unit-tested with a fake Chronos
pipeline (a stand-in object that returns predictable values), so tests run
in seconds without needing the real ~100MB model. A handful of tests
marked `real_chronos` do use the actual model — run those deliberately,
not as part of routine testing.

---

## `detection.py`

**What it does:** finds runs of exactly-repeated sensor values longer than
a configurable threshold (`min_stuck_hours`).

**Key function:** `find_stuck_periods(series, min_stuck_hours, ffill_limit)`
→ `list[StuckPeriod]`

**Design notes:**
- Ported from an existing, already-validated script — the core grouping
  logic (`change`/`cumsum()` trick) is unchanged from the original.
- Split cleanly from plotting: `plot_stuck_periods()` is a separate,
  optional function for notebook/interactive use. The automated pipeline
  never calls it — it can't, since nothing is watching a screen.
- `ffill_limit` (default 1) forward-fills isolated nulls before grouping,
  so a single missing reading doesn't fracture one real stuck run into two
  shorter runs that individually fall under the threshold and go
  undetected. This mirrors a known issue with single-outlier readings
  doing the same thing — same root cause, different trigger.

**Known limitation, not yet fixed:** a single *outlier* reading (a real,
different value) interrupting an otherwise-stuck run still splits it into
two runs. The null-handling fix above doesn't cover this case — it would
need a tolerance/skip-one-outlier rule, which is a judgment call not yet
made.

---

## `chronos_model.py`

**What it does:** loads and caches the Chronos forecasting model. The only
module that imports `chronos` directly.

**Key function:** `get_chronos_pipeline(model_name, device)` → a loaded
pipeline object, cached in memory and reused across calls.

**Design notes:**
- Checks the Hugging Face cache before loading, logging clearly whether it
  downloaded fresh or loaded from cache — useful for distinguishing "slow
  because first run" from "slow because something's wrong" in production
  logs.
- Thread-safe (a lock guards the cache), since a real deployment might
  call this from multiple places concurrently (e.g. an eventual online
  consumer alongside the offline job).
- Deliberately separate from `reconstruction.py` — both the offline job
  and any future online consumer need the same loaded model without each
  managing their own instance.

---

## `reconstruction.py`

**What it does:** the actual forecasting logic. Built in layers, each
independently testable:

| Function | Purpose |
|---|---|
| `forecast_forward` | Single-call forward prediction (≤~64 steps) |
| `forecast_forward_chunked` | Forward prediction for longer gaps — chunks calls, feeding each chunk's output back as context for the next |
| `forecast_backward` | Predicts backward from real data *after* a gap, by reversing the sequence, forecasting forward on that, then reversing the result back |
| `blend_bidirectional` | Combines forward + backward forecasts, trusting each more near its own side of the gap |
| `feather_edges` | Smooths the seam where reconstruction meets real data on either side |
| `reconstruct_stale_window` | Orchestrates all of the above for one `StuckPeriod` |

**Design notes:**
- `reconstruct_stale_window()` decides forward-only vs. bidirectional
  automatically, based on whether real data exists after the gap — not
  from a flag on `StuckPeriod` (which doesn't currently distinguish
  open-ended runs).
- Forward and backward reconstructions are explicitly **realigned onto the
  real observed gap timestamps** before blending, rather than trusting
  each direction's independently-computed cadence math. This was a real
  bug found on real data: irregular spacing near a gap boundary (common in
  real sensor exports) can cause the two directions to compute slightly
  different timestamp sequences, which used to break blending outright.
- The `np.ascontiguousarray()` call in `forecast_backward` fixes a real
  crash: reversing a numpy array creates a negative-stride view, which
  torch tensors reject.

**Known limitation:** chunked forecasts for very long gaps partly rely on
their own earlier predictions as context for later chunks — error can
compound the further into a long gap this goes. Not corrected for; how
much it matters in practice is what `synthetic_injection.py` measures.

---

## `data_source.py`

**What it does:** loads sensor data from the wide-format CSV export into
the `pandas.Series` shape everything else expects.

**Key functions:**
- `load_series_from_csv(file_path, column)` → one sensor's data
- `find_matching_columns(file_path, attribute_keywords)` → locate columns
  by attribute name substring (e.g. `"temperature"`, `"humidity"`) without
  hardcoding equipment IDs

**Design notes:** deliberately a stand-in for a live GraphQL client
against Overgrid's schema (`Point.series(startDate, endDate, every, fn)`
→ `[Measurement]`) — same output shape, so swapping to live fetching later
only means changing this one file. Two things carried over from the real
schema on purpose: timestamps get parsed from strings (the API returns
strings too), and nulls are preserved as `NaN` rather than dropped (the
API's `Measurement.value` is nullable).

---

---

## `graphql_source.py`

**What it does:** live counterpart to `data_source.py` — fetches sensor
data directly from Overgrid's GraphQL API instead of reading a CSV.

**Key functions:**
- `build_client(endpoint, token)` → a `gql` client, token from
  `OVERGRID_TOKEN` env var by default
- `fetch_points(client, alias, equipment_id, attribute, start_date,
  end_date, every, fn)` → `list[PointData]`
- `fetch_recent_points(...)` → convenience wrapper for "last N days"

**Design notes:**
- Ported from an existing, already-validated reference script
  (`dataset_builder.py`), trimmed to just what this pipeline needs — no
  CSV writing, no wide-pivot table.
- Pivots on `point.id` — the only field the schema guarantees unique —
  rather than `equipment_id`. This fixes a real limitation from CSV-only
  testing: CLI usage had been passing an equipment_id as `--point-id`,
  since the CSV export doesn't carry real per-attribute Point IDs. Each
  `PointData` now carries the genuine, unique `point_id` for its sensor.
- Doesn't hardcode any `.env` file path (the reference script's
  `/etc/overgrid/.env` is specific to `host241`) — `OVERGRID_TOKEN` must
  be set in the environment however fits wherever this runs.
- Same output shape as `data_source.py` (`pandas.Series`, real UTC
  timestamps, clean attribute name) — but **not yet wired into
  `offline_job.py`/`cli.py`**. The module exists and is tested; nothing
  calls it yet.

---

## `synthetic_injection.py`

**What it does:** measures whether reconstruction is actually accurate,
since real stuck windows have no ground truth to check against.

**Key functions:**
- `inject_synthetic_gap(series, gap_length_points)` → picks a random real
  stretch, returns a fake `StuckPeriod` pointing at it plus the real
  (hidden) values it covers
- `compute_naive_baselines(series, period, gap_index)` → forward-fill and
  linear interpolation for the same window
- `score_reconstruction(...)` → MAE/RMSE/MAPE for all three methods, plus
  `beats_forward_fill`/`beats_linear_interp` booleans
- `run_gap_trial(...)` → ties all of the above together for one trial

**Design notes:**
- Doesn't need to hide or freeze any data — `reconstruct_stale_window()`
  only ever reads context strictly outside a period's start/end, so
  pointing a fake `StuckPeriod` at real data is a genuinely blind test.
- Retries with a different random window if the chosen one (or its
  immediate boundary values) contains a null — otherwise a single missing
  real-world reading silently turns every metric into `NaN`. This was
  found via real data: a randomly-chosen 200-point window landed on one of
  the dataset's actual nulls.
- Scoring is intentionally honest — `beats_linear_interp` can and does
  come back `False`. The point is to know when Chronos isn't earning its
  complexity, not to assume it always does.

**What the validation has actually shown so far** (10 trials/gap length,
temperature + humidity, real 30-day dataset): linear interpolation wins at
short (~35min) and medium (~5hr) gaps; Chronos has the lowest average
error of all three methods at the longest gap tested (~33hr), for both
sensors. See MLflow experiment `chronos-staleness-reconstruction` for the
full numbers.

---

## `storage.py`

**What it does:** writes reconstructed measurements somewhere.

**Key concepts:**
- `MeasurementSink` — abstract interface, the one deliberate swap point in
  the whole pipeline
- `LocalJSONLSink` — real, working implementation; appends newline-JSON to
  disk
- `GraphQLSink` — a stub, not yet implemented
- `ImputedMeasurement` — the record format: `point_id`, `sensor`,
  `timestamp`, `raw_value`, `imputed_value`, `imputation_method`,
  `imputation_confidence`, `imputation_model_version`
- `reconstruction_to_measurements(...)` — converts a `Reconstruction`
  (from `reconstruction.py`) into a list of `ImputedMeasurement`

**Design notes:**
- `raw_value` and `imputed_value` are always kept as separate fields —
  never overwrite a real reading, anywhere in this pipeline.
- `sensor` exists specifically so records sharing the same `point_id`
  (e.g. an equipment ID used as a stand-in during CSV testing, since the
  CSV doesn't carry real per-attribute Point IDs) stay distinguishable.
  Found necessary after running temperature and humidity into the same
  output file and realizing they were otherwise indistinguishable.
- `GraphQLSink` stays a stub because the real schema
  (`schema_1.gqls`) has no imputation-aware fields, and its only write
  path (`WritablePoint.write()`) would silently overwrite the real
  reading with no way to tell real from imputed afterward — that breaks
  the "never overwrite raw data" principle. Becomes real once a schema
  extension exists, or a separate shadow point/attribute is agreed on.

---

## `tracking.py`

**What it does:** the only module that talks to MLflow directly.

**Key concepts:**
- `Tracker` — abstract interface (`run`, `log_params`, `log_metrics`, `log_artifact`)
- `MLflowTracker` — real implementation
- `NoOpTracker` — records nothing; used in tests and whenever
  `mlflow_enabled=False`

**Design notes:** centralizing MLflow access means consistent run
naming/tagging (git commit sha, sensor name) everywhere, and means tests
never need a live MLflow server to pass.

---

## `offline_job.py`

**What it does:** orchestrates everything above into one runnable job.

**Key functions:**
- `run_validation(pipeline, series, tracker)` — runs synthetic-gap
  validation across standard gap lengths (7/60/200 points, matching the
  real gap-length distribution seen in the original dataset), **averaging
  across trials before logging to MLflow**
- `run_offline_job(...)` — loads data, validates, detects, reconstructs,
  writes, returns the count of measurements written

**Design notes:**
- `VALIDATION_TRIALS_PER_LENGTH = 10` — bumped up from an initial 3 after
  the smaller sample gave noisy, unstable win-rate numbers (e.g. `0/3` vs
  `1/3` from a single trial flipping).
- Trials are averaged *in Python* before being logged to MLflow, not
  logged individually. Logging each trial under the same metric key with
  no distinguishing step would let MLflow silently keep only the last
  trial's value — a real bug that was caught and fixed.
- Gap lengths too long for the available data are skipped with a warning,
  not a crash — expected behavior on short datasets, not an error.
- Deliberately thin: no real logic lives here, only wiring, so a future
  scheduler/CI job/cron entry can all call `run_offline_job()` identically.

---

## `cli.py`

**What it does:** the `staleness offline` command (installed via
`pyproject.toml`'s `[project.scripts]`).

**Design notes:** has an intentionally empty `@app.callback()`. Without
it, `typer` collapses into a single-command CLI when there's only one
`@app.command()` registered, silently dropping the subcommand name
(`staleness --column ...` instead of `staleness offline --column ...`).
This keeps `offline` (and future commands like `validate`/`online`)
explicit.

---

## What's genuinely not built yet

- **Live GraphQL fetching** — `data_source.py` is CSV-only. A reference
  script exists (not yet reviewed/integrated).
- **Schema-driven `storage.py`** — extracting sink structure dynamically
  from the GraphQL schema, contingent on the live-fetching work above.
- **Real-time / Kafka processing** — the "online, forward-only, low-latency"
  half of the original hybrid design. Nothing exists here yet, not even a
  stub — `PROVISIONAL` confidence is defined in `storage.py` but nothing
  currently produces it.
- **Automation/scheduling** — deliberately deferred. Options discussed but
  not built: plain cron on the server, or a self-hosted GitHub Actions
  runner.
- **Method selection by gap length** — validation data suggests linear
  interpolation may be preferable to Chronos at short/medium gaps, but
  `reconstruct_stale_window()` currently always uses Chronos regardless of
  gap length. Deliberately not changed yet, pending a decision.
- **MLflow authentication without a manual SSH tunnel** — MLflow is
  currently only reachable by tunneling directly to its port, bypassing a
  JupyterHub proxy that gates the public URL behind browser-session auth
  (see README). That's fine for interactive use, but not viable for
  anything automated (cron, CI) — nothing will be sitting there manually
  opening a tunnel at 2am. Needs a real fix: either a service
  account/token MLflow itself accepts (bypassing the JupyterHub proxy
  legitimately), or the proxy's auth scheme investigated for a
  script-friendly path in. Flagged to revisit after the Kafka work.
- **A third validation baseline from PyPOTS** — see `RELATED_WORK.md`.
  `synthetic_injection.py` currently only compares against forward-fill
  and linear interpolation; a simple classical PyPOTS method could
  strengthen the validation story further. Not started.
