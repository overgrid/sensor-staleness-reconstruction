# Sensor Staleness Reconstruction

Detects stuck or frozen sensor readings (temperature and humidity, extendable
to any Overgrid attribute) and reconstructs plausible values for them using
Amazon's Chronos-Bolt-Small forecasting model. Every reconstruction is
checked against synthetic ground truth before it is trusted. This was the
earlier, research phase of the staleness work: the production tool that
writes fault flags back to the platform is
[`stale-data-detector`](https://github.com/overgrid/stale-data-detector).

## Why

Overgrid sensors occasionally freeze and report the same value for hours.
Left alone, this quietly corrupts anything trained on the data. This
pipeline finds those runs and fills them with a value estimated from the
sensor's own recent behaviour, rather than dropping the data or leaving it
wrong, and it checks its own accuracy before trusting any result.

## Status

Core pipeline is built, tested, and validated against real data.

**Built and working:**

- stuck-period detection, ported and extended from earlier prototype work
- Chronos-based reconstruction: forward, chunked, backward, bidirectional
  blend, edge feathering
- a blind validation harness, scoring against real hidden data with MAE,
  RMSE, and MAPE
- live GraphQL data fetching and schema introspection
- a Kafka-based real-time simulation: a producer replays real sensor data
  at accelerated speed (60x by default, so a 10-minute cadence arrives every
  10 seconds), and an online consumer runs detection and reconstruction on
  it incrementally
- a FastAPI + WebSocket dashboard showing the live stream with reconstructed
  windows shaded
- MLflow tracking, with a pluggable, no-op tracker so tests run without a
  server
- a thread-safe Chronos model cache with Hugging Face cache detection
- an installable package with a `staleness` CLI
- a deployment runbook for the company server: Docker, a dedicated service
  account, systemd units, and an nginx location behind the existing TLS
  certificate
- 121 unit tests across 15 test files

**Deliberately not built:** a GraphQL write path for reconstructed values.
The only write path available on the platform would have overwritten real
raw readings, so the sink is left as a stub until the schema supports a
separate, safe write for imputed values. Raw and imputed values are always
kept in separate fields for exactly this reason, see Storage below.

## How it works

| Module | Purpose |
|---|---|
| `detection.py` | Finds runs of repeated values lasting 15 minutes or longer (configurable). Handles isolated null readings so one missing sample cannot split a real fault into two shorter, undetected halves. |
| `chronos_model.py` | Loads and caches the Chronos forecasting model, thread-safe, Hugging Face cache aware. |
| `reconstruction.py` | Fills a stuck run: forward prediction, backward prediction, a blend of both, and edge smoothing. |
| `synthetic_injection.py` | Validates accuracy by hiding real data as fake gaps and scoring the reconstruction against the hidden ground truth. |
| `storage.py` | Writes reconstructed values through a `MeasurementSink` interface, keeping raw and imputed values in separate fields. |
| `tracking.py` | Logs every run to MLflow through a pluggable `Tracker` interface. |
| `offline_job.py` | Orchestrates the pipeline end to end: load, validate, detect, reconstruct, write. |
| `data_source.py` | Reads a sensor series from a CSV export. |
| `graphql_source.py` | Reads a sensor series live from Overgrid's GraphQL API. |
| `cli.py` | Exposes the pipeline as the `staleness offline` command. |

### Reconstruction

Chronos is a general-purpose forecasting model, not built for gap-filling,
so reconstruction works in layers rather than a single forward guess:

- **Forward prediction** covers gaps up to about 64 steps, Chronos-Bolt's
  native horizon. Longer gaps use chunked forward prediction, feeding each
  chunk's output back in as context for the next.
- **Backward prediction** reverses the data after the gap, forecasts
  forward on the reversed series, then reverses the result back.
- **Bidirectional blending** combines the forward and backward predictions,
  trusting each one more near the side it came from.
- **Edge feathering** smooths the join where a reconstructed section meets
  real data, so there is no visible seam.
- Both directions are realigned onto the real observed gap timestamps
  before blending, so the result stays correct when the surrounding data is
  irregularly spaced.
- When a fault is still ongoing, there is no data after it yet, so the
  pipeline automatically falls back to forward-only prediction instead of a
  blend.

### Validation

Before any real stuck run is reconstructed, the method is checked on data
where the true answer is already known. Random real stretches are hidden as
synthetic gaps, reconstructed blind, and scored against the hidden truth.
Each gap length runs 10 trials, at three lengths chosen to match what is
actually seen in the real stuck-period data: 7, 60, and 200 points.

**Results (MAE, lower is better), from the MLflow experiment
`chronos-staleness-reconstruction`:**

**Temperature (C)**

| Gap | Chronos | Forward-fill | Linear interpolation |
|---|---|---|---|
| 7 pts | 0.056 | 0.067 | 0.055 |
| 60 pts | 0.163 | 0.236 | 0.144 |
| 200 pts | 0.533 | 0.646 | 0.564 |

**Humidity (%RH)**

| Gap | Chronos | Forward-fill | Linear interpolation |
|---|---|---|---|
| 7 pts | 0.167 | 0.319 | 0.130 |
| 60 pts | 1.239 | 1.637 | 0.898 |
| 200 pts | 3.424 | 5.547 | 3.922 |

**What this means:** Chronos is not the best method at every gap length.
At the longest gaps tested (200 points), it was the most accurate method
for both sensors, 17.6 percent lower error than forward-fill on temperature
and 38.3 percent lower on humidity, and 5.6 and 12.7 percent lower than
linear interpolation respectively. At the short and medium gaps, plain
linear interpolation won, by up to 27.5 percent on humidity at 60 points.
Over those shorter timescales the signal is close enough to linear that the
model adds cost without adding accuracy.

**Recommendation, evidence based:** use linear interpolation for short and
medium gaps, and reserve Chronos for long ones. Full detail in
[`docs/VALIDATION_FINDINGS.md`](docs/VALIDATION_FINDINGS.md).

**Two measurement bugs found and fixed during validation:**

1. MLflow was silently keeping only the last of every 3 logged trials,
   because each trial was logged under the same metric key. An early run
   showed Chronos losing every comparison, which turned out to be this
   artefact rather than a real result. Fixed by averaging trials in Python
   before logging, and the trial count was raised to 10 for a stable read.
2. A single null value in a validation window turned every metric into
   NaN, for the model and both baselines. Fixed by resampling validation
   windows until they were null-free.

## Storage

Reconstructed values are never allowed to overwrite a real reading. Every
write keeps `raw_value` and `imputed_value` as separate fields, along with
the method used, a confidence score, and the model version, so a
reconstructed point is always distinguishable from a real one downstream.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running tests

```bash
python -m pytest tests/ -v -k "not real_chronos"
```

The `real_chronos` tests are excluded by default since they download the
actual model (about 100 MB, first run only) and take noticeably longer:

```bash
python -m pytest tests/ -v -k real_chronos
```

## Usage

```bash
staleness offline \
  --column "ecbc3d63b0e4__Air_Temperature_Sensor__aht_temperature" \
  --point-id "ecbc3d63b0e4" \
  --csv-path "data/ecbc3d63b0e4_last_30_days_mean_ecbc3d63b0e4_wide.csv" \
  --mlflow-uri "http://localhost:5000/services/mlflow"
```

This detects every stuck period in the given CSV column, validates
reconstruction accuracy through synthetic gap injection (logged to MLflow),
reconstructs each real stuck period, and writes the results to
`data/reconstructed_measurements.jsonl`.

Useful flags:

- `--skip-mlflow`, disable MLflow logging, for example if no server is
  reachable
- `--skip-validation`, skip synthetic gap validation and just reconstruct
- `--min-stuck-hours`, override the default stuck-detection threshold
- `--sink-path`, where reconstructed measurements get written

Run `staleness offline --help` for the full list.

## MLflow

If the MLflow server sits behind a proxy that requires login, as
JupyterHub-hosted MLflow instances typically do, tunnel directly to the
underlying MLflow process instead of going through the proxy:

```bash
ssh -L 5000:localhost:5000 <user>@<server>
```

Then point `--mlflow-uri` at `http://localhost:5000<static-prefix>`,
checking the MLflow server's own startup command for its `--static-prefix`,
if any, for example `/services/mlflow`.

## Data

The offline CLI reads sensor data from a wide-format CSV export by default,
one column per sensor, one row per timestamp, through `data_source.py`.
Live fetching from Overgrid's GraphQL API is built and tested separately in
`graphql_source.py`, including schema introspection. Both produce the same
`pandas.Series` shape, so pointing `offline_job.py` at one instead of the
other is a small, contained change.

## Deployment

A runbook for running this on the company server is included: Docker, a
dedicated service account, systemd units for the long-running pieces, and
an nginx location proxied behind the existing TLS certificate.
