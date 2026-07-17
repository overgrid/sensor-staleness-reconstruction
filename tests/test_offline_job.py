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


def test_run_validation_averages_trials_instead_of_only_keeping_the_last_one(sample_csv, monkeypatch):
    # Regression test: logging each trial under the same metric key with
    # no step counter used to let MLflow silently overwrite trial 1 and 2
    # with trial 3's value. This proves run_validation now averages
    # in-process, so every metrics dict logged reflects all trials, not
    # just whichever ran last.
    series = load_series_from_csv(str(sample_csv), column="sensor__aht_temperature")
    tracker = RecordingTracker()

    # Force run_gap_trial to return distinct, known MAE values per trial
    # (0.0, 10.0, 20.0, ...) so the expected average is unambiguous —
    # references VALIDATION_TRIALS_PER_LENGTH directly so this test stays
    # correct even if that constant changes again later.
    from staleness_pipeline.synthetic_injection import GapTrialResult

    n_trials = offline_job.VALIDATION_TRIALS_PER_LENGTH
    call_count = {"n": 0}

    def fake_run_gap_trial(pipeline, series, gap_length_points, trial_index, rng_seed=None, min_context_points=10):
        mae = float(call_count["n"] * 10)
        call_count["n"] += 1
        return GapTrialResult(
            gap_length_points=gap_length_points,
            trial_index=trial_index,
            chronos_mae=mae,
            chronos_rmse=mae,
            chronos_mape=mae,
            forward_fill_mae=99.0,
            forward_fill_rmse=99.0,
            forward_fill_mape=99.0,
            linear_interp_mae=99.0,
            linear_interp_rmse=99.0,
            linear_interp_mape=99.0,
            beats_forward_fill=True,
            beats_linear_interp=True,
        )

    monkeypatch.setattr(offline_job, "run_gap_trial", fake_run_gap_trial)
    offline_job.run_validation(FakePipeline(), series, tracker)

    # Only one metrics dict should be logged per gap length (not one per
    # trial), and it should reflect the AVERAGE of 0, 10, 20, ... (n-1)*10.
    gap7_metrics = [m for m in tracker.logged_metrics if any("gap7" in k for k in m)]
    assert len(gap7_metrics) == 1
    expected_mean = sum(i * 10 for i in range(n_trials)) / n_trials
    assert gap7_metrics[0]["chronos_mae_gap7"] == pytest.approx(expected_mean)
    assert gap7_metrics[0]["num_trials_gap7"] == float(n_trials)