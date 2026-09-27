# Sensor Staleness Reconstruction

Detects stuck or frozen sensor readings (temperature, humidity, extendable to any
Overgrid attribute) and reconstructs plausible values for them using the Chronos
Bolt Small forecasting model. Every reconstruction is checked against synthetic
ground truth before it is trusted.

## Why

Overgrid sensors occasionally report the same value for hours at a time. Left in
place, these stuck runs quietly corrupt anything trained on the data. This
pipeline finds those runs and fills them with a value estimated from the sensor's
own recent behaviour, rather than dropping the data or leaving it wrong.

## Status

Core pipeline is built, tested, and validated against real data.

**Built and working:** stuck period detection, Chronos based reconstruction
(forward, chunked, backward, bidirectional blend, edge feathering), synthetic
gap accuracy validation, local JSONL storage, MLflow tracking, and a CLI
(`staleness offline`) that runs the whole thing end to end. A live GraphQL
fetch module (`graphql_source.py`) is also built and unit tested standalone.

**Not yet integrated:** the GraphQL module is not yet wired into the
end to end pipeline, which currently reads from CSV via `data_source.py`.
Also not yet built: a GraphQL storage backend, real time or Kafka based
processing, and any scheduling or automation.

## How it works

| Module | Purpose |
|---|---|
| `detection.py` | Finds stuck runs: readings that repeat identically for longer than a threshold. |
| `chronos_model.py` | Loads and caches the Chronos forecasting model, with thread safe access. |
| `reconstruction.py` | Fills a stuck run: forward prediction, backward prediction, a blend of both, and edge smoothing. |
| `synthetic_injection.py` | Validates accuracy by hiding real data as fake gaps and scoring the reconstruction against it. |
| `storage.py` | Writes reconstructed values through a `MeasurementSink` interface, keeping the raw and imputed values separate. |
| `tracking.py` | Logs every run to MLflow through a `Tracker` interface. |
| `offline_job.py` | Orchestrates the pipeline end to end: load, validate, detect, reconstruct, write. |
| `data_source.py` | Reads a sensor series from a CSV export. |
| `graphql_source.py` | Reads a sensor series from Overgrid's live GraphQL API. Tested, not yet wired into `offline_job.py`. |
| `cli.py` | Exposes the pipeline as the `staleness offline` command. |

### Detection

Consecutive identical readings are grouped into runs. A short forward fill
(`ffill_limit`, default 1) bridges isolated missing values, so a stuck run does
not fragment into several shorter runs that would fall below the detection
threshold individually. Anything held for longer than `--min-stuck-hours`
(default 0.25 hours) is flagged.

**Known limitation:** a single genuine outlier reading in the middle of an
otherwise stuck run still splits it into two shorter runs, since there is no
tolerance for a lone interruption yet.

### Reconstruction

Reconstruction works in layers, not a single forward guess:

- **Forward prediction** covers gaps up to about 64 steps. Longer gaps are
  filled with chunked forward prediction, feeding each chunk's output back in
  as context for the next.
- **Backward prediction** reverses the data after the gap, forecasts forward
  on the reversed series, then reverses the result back.
- **Blending** combines the forward and backward predictions, trusting each
  one more near the side it came from.
- **Edge feathering** smooths the join where a reconstructed section meets
  real data, so there is no visible seam.
- Both directions are realigned onto the real observed gap timestamps before
  blending, to stay correct when the surrounding data is irregularly spaced.

### Validation

Before any real stuck run is reconstructed, the method is checked on data
where the true answer is already known. Random real stretches are hidden as
synthetic gaps, reconstructed, and scored against the hidden ground truth with
MAE, RMSE, and MAPE. Each gap length runs 10 trials, at three lengths chosen to
match what is actually seen in production: about 35 minutes, about 5 hours,
and about 33 hours.

**Result:** Chronos does not win outright. Plain linear interpolation is more
accurate at the short and medium gap lengths tested; Chronos only pulls ahead
at the longest gaps, about 33 hours. This is logged, not hidden, since it
directly shapes which method should run at which gap length going forward.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
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

- `--skip-mlflow`, disable MLflow logging (for example, no server reachable)
- `--skip-validation`, skip synthetic gap validation and just reconstruct
- `--min-stuck-hours`, override the default 0.25 hour detection threshold
- `--sink-path`, where reconstructed measurements get written

Run `staleness offline --help` for the full list.

## Testing

```bash
python -m pytest tests/ -v -k "not real_chronos"
```

The `real_chronos` tests are excluded by default since they download the
actual model (about 100MB, first run only) and take noticeably longer:

```bash
python -m pytest tests/ -v -k real_chronos
```

## MLflow

If the MLflow server sits behind a proxy that requires login, as
JupyterHub hosted MLflow instances typically do, tunnel directly to the
underlying MLflow process instead of going through the proxy:

```bash
ssh -L 5000:localhost:5000 <user>@<server>
```

Then point `--mlflow-uri` at `http://localhost:5000<static-prefix>`, checking
the MLflow server's own startup command for its `--static-prefix`, if any, for
example `/services/mlflow`.

## Data

Real sensor data currently comes from a wide format CSV export, one column per
sensor, one row per timestamp. This is a deliberate stand in for the live
GraphQL client, which already exists and is tested in `graphql_source.py`.
Switching the pipeline over only means pointing `offline_job.py` at that
module instead of `data_source.py`, since both produce the same
`pandas.Series` shape.
