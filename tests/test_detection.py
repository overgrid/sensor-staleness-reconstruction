import pandas as pd
import pytest

from staleness_pipeline.detection import find_stuck_periods, stuck_periods_to_dataframe


def make_series(values, freq="10min", name="test_sensor"):
    index = pd.date_range("2026-01-01", periods=len(values), freq=freq, tz="UTC")
    return pd.Series(values, index=index, name=name)


def test_no_stuck_period_when_everything_changes():
    series = make_series([1.0, 2.0, 3.0, 4.0])
    periods = find_stuck_periods(series, min_stuck_hours=0.25)
    assert periods == []


def test_detects_a_stuck_run_above_threshold():
    # 21.0 repeats for 4 readings at 10-minute spacing = 30 minutes stuck.
    series = make_series([20.0, 21.0, 21.0, 21.0, 21.0, 22.0])
    periods = find_stuck_periods(series, min_stuck_hours=0.25)  # 15 min threshold

    assert len(periods) == 1
    p = periods[0]
    assert p.stuck_value == 21.0
    assert p.duration_hours == 0.5
    assert p.sensor == "test_sensor"


def test_ignores_runs_shorter_than_threshold():
    # Same repeated value, but only 10 minutes — below a 30-minute threshold.
    series = make_series([20.0, 21.0, 21.0, 22.0])
    periods = find_stuck_periods(series, min_stuck_hours=0.5)
    assert periods == []


def test_multiple_stuck_runs_detected_separately():
    series = make_series([1.0, 5.0, 5.0, 5.0, 2.0, 9.0, 9.0, 9.0, 3.0])
    periods = find_stuck_periods(series, min_stuck_hours=0.25)
    assert len(periods) == 2
    assert periods[0].stuck_value == 5.0
    assert periods[1].stuck_value == 9.0


def test_empty_series_returns_empty_list():
    series = pd.Series([], dtype=float, name="test_sensor")
    assert find_stuck_periods(series) == []


def test_isolated_null_does_not_fracture_a_stuck_run():
    # 21.0, 21.0, NaN, 21.0, 21.0 at 10-min cadence = 40 minutes total.
    # Without null-handling, this splits into two 20-min runs, each under
    # a 30-min threshold, and neither gets detected. With ffill_limit=1,
    # it's correctly seen as one continuous 40-minute stuck run.
    series = make_series([20.0, 21.0, 21.0, float("nan"), 21.0, 21.0, 22.0])
    periods = find_stuck_periods(series, min_stuck_hours=0.5, ffill_limit=1)

    assert len(periods) == 1
    assert periods[0].duration_hours == pytest.approx(0.6667, abs=0.01)  # 40 minutes


def test_null_handling_can_be_disabled():
    series = make_series([20.0, 21.0, 21.0, float("nan"), 21.0, 21.0, 22.0])
    periods = find_stuck_periods(series, min_stuck_hours=0.5, ffill_limit=0)
    # Without ffill, both halves are 20 minutes each — under the 30-min
    # threshold — so nothing gets flagged.
    assert periods == []


def test_dataframe_conversion_formats_dates_as_strings():
    series = make_series([21.0, 21.0, 21.0, 21.0])
    periods = find_stuck_periods(series, min_stuck_hours=0.25)
    df = stuck_periods_to_dataframe(periods)

    assert list(df.columns) == [
        "Sensor/Metric", "Stuck Value", "Start Time", "End Time", "Duration (Hours)"
    ]
    assert isinstance(df.iloc[0]["Start Time"], str)