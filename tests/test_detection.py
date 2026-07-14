import pandas as pd

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


def test_dataframe_conversion_formats_dates_as_strings():
    series = make_series([21.0, 21.0, 21.0, 21.0])
    periods = find_stuck_periods(series, min_stuck_hours=0.25)
    df = stuck_periods_to_dataframe(periods)

    assert list(df.columns) == [
        "Sensor/Metric", "Stuck Value", "Start Time", "End Time", "Duration (Hours)"
    ]
    assert isinstance(df.iloc[0]["Start Time"], str)