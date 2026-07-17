# Sensor Staleness Reconstruction

Detects stuck/frozen sensor readings (temperature, humidity — extendable
to any Overgrid attribute) and reconstructs plausible values using the
Chronos Bolt Small forecasting model, with every reconstruction validated
against real synthetic ground truth before being trusted.

## Status

Core pipeline is built, tested, and validated against real data. See
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for a detailed breakdown of
every module. In short:

**Built and working:** stuck-period detection, Chronos-based reconstruction
(forward, chunked, backward, bidirectional blend, edge feathering),
synthetic-gap accuracy validation, local JSONL storage, MLflow tracking,
and a CLI (`staleness offline`) that runs the whole thing end to end.

**Not yet built:** live GraphQL data fetching (currently CSV-based), a
GraphQL storage backend, real-time/Kafka processing, and any
automation/scheduling.

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
actual model (~100MB, first run only) and take noticeably longer. Run them
deliberately when you want to sanity-check against the real model:

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
reconstruction accuracy via synthetic gap injection (logged to MLflow),
reconstructs each real stuck period, and writes the results to
`data/reconstructed_measurements.jsonl`.

Useful flags:
- `--skip-mlflow` — disable MLflow logging entirely (e.g. no server reachable)
- `--skip-validation` — skip synthetic-gap validation, just reconstruct
- `--min-stuck-hours` — override the default 0.25hr stuck-detection threshold
- `--sink-path` — where reconstructed measurements get written

Run `staleness offline --help` for the full list.

## MLflow

If your MLflow server sits behind a proxy that requires login (as
JupyterHub-hosted MLflow instances typically do), tunnel directly to the
underlying MLflow process instead of going through the proxy:

```bash
ssh -L 5000:localhost:5000 <user>@<server>
```

then point `--mlflow-uri` at `http://localhost:5000<static-prefix>` (check
the MLflow server's actual startup command for its `--static-prefix`, if
any — e.g. `/services/mlflow`).

## Data

Real sensor data currently comes from a wide-format CSV export (one column
per sensor, one row per timestamp) — see `data_source.py`. This is a
deliberate stand-in for a live GraphQL client against Overgrid's schema;
switching to live fetching only requires changing `data_source.py`, not
anything downstream, since both produce the same `pandas.Series` shape.
