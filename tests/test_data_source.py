import pandas as pd
import pytest

from staleness_pipeline.data_source import find_matching_columns, load_series_from_csv


@pytest.fixture
def sample_csv(tmp_path):
    """A tiny wide-format CSV matching your real naming pattern:
    <equipment_id>__<equipment_type>__<attribute>."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="10min", tz="UTC"),
            "ecbc3d63b0e4__Air_Temperature_Sensor__aht_temperature": [20.0, 20.5, None, 21.0],
            "ecbc3d63b0e4__Air_Temperature_Sensor__aht_humidity": [55.0, 55.2, 55.1, 55.3],
        }
    )
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    return path


def test_load_series_returns_clean_attribute_name(sample_csv):
    series = load_series_from_csv(
        str(sample_csv), column="ecbc3d63b0e4__Air_Temperature_Sensor__aht_temperature"
    )
    assert series.name == "aht_temperature"


def test_load_series_parses_timestamps_and_sorts(sample_csv):
    series = load_series_from_csv(
        str(sample_csv), column="ecbc3d63b0e4__Air_Temperature_Sensor__aht_temperature"
    )
    assert isinstance(series.index, pd.DatetimeIndex)
    assert series.index.is_monotonic_increasing
    assert series.index.tz is not None  # UTC-aware, not naive


def test_load_series_preserves_nulls_as_nan(sample_csv):
    series = load_series_from_csv(
        str(sample_csv), column="ecbc3d63b0e4__Air_Temperature_Sensor__aht_temperature"
    )
    assert series.isna().sum() == 1


def test_load_series_raises_clear_error_for_unknown_column(sample_csv):
    with pytest.raises(ValueError, match="not found"):
        load_series_from_csv(str(sample_csv), column="does_not_exist")


def test_find_matching_columns_finds_both_temperature_and_humidity(sample_csv):
    matches = find_matching_columns(str(sample_csv), attribute_keywords=["temperature", "humidity"])
    assert len(matches) == 2
    assert any("temperature" in c for c in matches)
    assert any("humidity" in c for c in matches)


def test_find_matching_columns_excludes_the_timestamp_column(sample_csv):
    matches = find_matching_columns(str(sample_csv), attribute_keywords=["timestamp"])
    assert matches == []
