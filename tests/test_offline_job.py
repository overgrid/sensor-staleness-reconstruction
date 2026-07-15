import pandas as pd
import pytest
import torch

from staleness_pipeline import offline_job
from staleness_pipeline.data_source import load_series_from_csv
from staleness_pipeline.tracking import NoOpTracker


class FakePipeline:
    def predict_quantiles(self, inputs, prediction_length, quantile_levels):
        last_value = inputs[-1].item()
        mean = torch.full((1, prediction_length), last_value)
        quantiles = torch.zeros((1, prediction_length, len(quantile_levels)))
        return quantiles, mean


class RecordingTracker(NoOpTracker):
    """Same as NoOpTracker (does nothing real), but remembers what was
    logged, so tests can check run_validation actually logged something."""

    def __init__(self):
        self.logged_metrics: list[dict] = []

    def log_metrics(self, metrics: dict) -> None:
        self.logged_metrics.append(metrics)


@pytest.fixture
def sample_csv(tmp_path):
    n = 60
    values = [20.0 + (i % 5) * 0.1 for i in range(n)]
    for i in range(30, 36):  # a real stuck run: 6 points = 1 hour at 10-min cadence
        values[i] = 21.0
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="10min", tz="UTC"),
            "sensor__aht_temperature": values,
        }
    )
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    return path


def test_run_offline_job_writes_reconstructed_measurements(tmp_path, sample_csv, monkeypatch):
    monkeypatch.setattr(offline_job, "get_chronos_pipeline", lambda: FakePipeline())

    sink_path = tmp_path / "out.jsonl"
    count = offline_job.run_offline_job(
        csv_path=str(sample_csv),
        column="sensor__aht_temperature",
        point_id="pt-1",
        min_stuck_hours=0.5,
        local_sink_path=str(sink_path),
        mlflow_enabled=False,
        run_validation_first=False,  # kept out here — tested separately below
    )

    assert count == 6  # the 6-point stuck run we planted
    assert sink_path.exists()
    lines = sink_path.read_text().splitlines()
    assert len(lines) == 6


def test_run_offline_job_with_no_stuck_periods_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(offline_job, "get_chronos_pipeline", lambda: FakePipeline())

    # No repeated values anywhere — nothing should be flagged as stuck.
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=20, freq="10min", tz="UTC"),
            "sensor__aht_temperature": [20.0 + i * 0.1 for i in range(20)],
        }
    )
    csv_path = tmp_path / "clean.csv"
    df.to_csv(csv_path, index=False)
    sink_path = tmp_path / "out.jsonl"

    count = offline_job.run_offline_job(
        csv_path=str(csv_path),
        column="sensor__aht_temperature",
        point_id="pt-1",
        local_sink_path=str(sink_path),
        mlflow_enabled=False,
        run_validation_first=False,
    )

    assert count == 0


def test_run_validation_logs_metrics_and_skips_gaps_too_long_for_the_data(sample_csv):
    series = load_series_from_csv(str(sample_csv), column="sensor__aht_temperature")
    tracker = RecordingTracker()

    # Series is only 60 points; gap lengths of 60 and 200 need more room
    # than that (gap + 20 points of required context) and should be
    # skipped with a warning rather than crash. Only gap=7 should succeed.
    offline_job.run_validation(FakePipeline(), series, tracker)

    assert len(tracker.logged_metrics) > 0
    logged_keys = set()
    for metrics in tracker.logged_metrics:
        logged_keys.update(metrics.keys())
    assert any("gap7" in key for key in logged_keys)
    assert not any("gap200" in key for key in logged_keys)
