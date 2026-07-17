import pandas as pd
import pytest

from staleness_pipeline.graphql_source import build_client, fetch_points


class FakeGraphQLClient:
    """Stands in for gql.Client — returns a canned response shaped exactly
    like a real Overgrid API reply, so tests never make a real network
    call and never need a real token."""

    def __init__(self, response):
        self._response = response

    def execute(self, query, variable_values=None):
        return self._response


def make_response():
    return {
        "projects": [
            {
                "id": "proj-1",
                "alias": "MyHome",
                "equipment": [
                    {
                        "id": "ecbc3d63b0e4",
                        "type": "Air_Temperature_Sensor",
                        "points": [
                            {
                                "id": "point-temp-123",
                                "type": "Air_Temperature_Sensor",
                                "attribute": "aht_temperature",
                                "series": [
                                    {"timestamp": "2026-01-01T00:00:00", "value": 20.0},
                                    {"timestamp": "2026-01-01T00:10:00", "value": None},
                                    {"timestamp": "2026-01-01T00:20:00", "value": 20.5},
                                ],
                            },
                            {
                                "id": "point-empty-456",
                                "type": "Some_Other_Sensor",
                                "attribute": "unused_attribute",
                                "series": [],
                            },
                        ],
                    }
                ],
            }
        ]
    }


def test_fetch_points_skips_points_with_no_series_data():
    client = FakeGraphQLClient(make_response())
    points = fetch_points(
        client,
        alias="MyHome",
        equipment_id="ecbc3d63b0e4",
        attribute="aht_temperature",
        start_date=pd.Timestamp("2026-01-01"),
        end_date=pd.Timestamp("2026-01-02"),
    )
    assert len(points) == 1  # the empty-series point is skipped, not an error


def test_fetch_points_carries_the_real_point_id():
    client = FakeGraphQLClient(make_response())
    points = fetch_points(
        client,
        alias="MyHome",
        equipment_id="ecbc3d63b0e4",
        attribute="aht_temperature",
        start_date=pd.Timestamp("2026-01-01"),
        end_date=pd.Timestamp("2026-01-02"),
    )
    # This is the real per-attribute Point.id — not the equipment_id
    # stand-in used during CSV-only testing.
    assert points[0].point_id == "point-temp-123"
    assert points[0].equipment_id == "ecbc3d63b0e4"


def test_fetch_points_preserves_nulls_as_nan():
    client = FakeGraphQLClient(make_response())
    points = fetch_points(
        client,
        alias="MyHome",
        equipment_id="ecbc3d63b0e4",
        attribute="aht_temperature",
        start_date=pd.Timestamp("2026-01-01"),
        end_date=pd.Timestamp("2026-01-02"),
    )
    assert points[0].series.isna().sum() == 1


def test_fetch_points_series_is_sorted_and_named():
    client = FakeGraphQLClient(make_response())
    points = fetch_points(
        client,
        alias="MyHome",
        equipment_id="ecbc3d63b0e4",
        attribute="aht_temperature",
        start_date=pd.Timestamp("2026-01-01"),
        end_date=pd.Timestamp("2026-01-02"),
    )
    series = points[0].series
    assert series.name == "aht_temperature"
    assert series.index.is_monotonic_increasing
    assert isinstance(series.index, pd.DatetimeIndex)


def test_build_client_raises_clear_error_when_no_token(monkeypatch):
    monkeypatch.delenv("OVERGRID_TOKEN", raising=False)
    with pytest.raises(ValueError, match="token"):
        build_client()


def test_build_client_uses_env_token_without_raising(monkeypatch):
    monkeypatch.setenv("OVERGRID_TOKEN", "fake-token-123")
    client = build_client()
    assert client is not None
