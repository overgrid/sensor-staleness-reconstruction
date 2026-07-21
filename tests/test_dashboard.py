"""Integration tests for dashboard.py.

Mocks out exactly the two things that would otherwise require real
infrastructure (a Kafka broker, a downloaded Chronos model) — everything
else (routing, the WebSocket contract, the REST snapshot endpoints) is
exercised for real via FastAPI's TestClient, same testing philosophy as
the rest of this project: fake the expensive boundary, test the real
wiring around it.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from staleness_pipeline import dashboard
from staleness_pipeline.live_store import UpdateEvent


class FakePipeline:
    def predict_quantiles(self, inputs, prediction_length, quantile_levels):
        import torch

        last_value = inputs[-1].item()
        mean = torch.full((1, prediction_length), last_value)
        quantiles = torch.zeros((1, prediction_length, len(quantile_levels)))
        return quantiles, mean


@pytest.fixture(autouse=True)
def no_real_infra(monkeypatch):
    """Every test in this file gets a Chronos pipeline that needs no
    download, and a Kafka consumer loop that's a no-op — the background
    thread in dashboard.py's lifespan starts and exits immediately
    instead of blocking on a real broker connection."""
    monkeypatch.setattr(dashboard, "get_chronos_pipeline", lambda: FakePipeline())
    monkeypatch.setattr(dashboard, "_kafka_consumer_loop", lambda loop: None)
    yield


def test_index_serves_html():
    with TestClient(dashboard.app) as client:
        resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Staleness pipeline" in resp.text


def test_list_series_empty_before_any_data():
    with TestClient(dashboard.app) as client:
        resp = client.get("/api/series")
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_series_unknown_sensor_returns_empty_shape():
    with TestClient(dashboard.app) as client:
        resp = client.get("/api/series/no-such-point/no-such-sensor")
    assert resp.status_code == 200
    assert resp.json() == {"timestamps": [], "values": []}


def test_get_series_returns_ingested_data():
    with TestClient(dashboard.app) as client:
        dashboard.store.ingest(
            {"point_id": "p1", "sensor": "aht_temperature", "timestamp": "2026-01-01T00:00:00+00:00", "value": 21.0}
        )
        resp = client.get("/api/series/p1/aht_temperature")

    body = resp.json()
    assert body["timestamps"] == ["2026-01-01T00:00:00+00:00"]
    assert body["values"] == [21.0]


def test_list_series_reflects_ingested_sensors():
    with TestClient(dashboard.app) as client:
        dashboard.store.ingest(
            {"point_id": "p1", "sensor": "aht_temperature", "timestamp": "2026-01-01T00:00:00+00:00", "value": 21.0}
        )
        dashboard.store.ingest(
            {"point_id": "p1", "sensor": "aht_humidity", "timestamp": "2026-01-01T00:00:00+00:00", "value": 55.0}
        )
        resp = client.get("/api/series")

    pairs = {(row["point_id"], row["sensor"]) for row in resp.json()}
    assert pairs == {("p1", "aht_temperature"), ("p1", "aht_humidity")}


def test_websocket_receives_broadcast_point_message():
    with TestClient(dashboard.app) as client:
        with client.websocket_connect("/ws") as ws:
            import asyncio

            asyncio.run(
                dashboard.manager.broadcast(
                    {"type": "point", "point_id": "p1", "sensor": "aht_temperature", "timestamp": "t", "value": 1.0}
                )
            )
            msg = ws.receive_json()

    assert msg == {"type": "point", "point_id": "p1", "sensor": "aht_temperature", "timestamp": "t", "value": 1.0}


def test_event_to_messages_includes_point_and_reconstruction():
    from dataclasses import dataclass

    @dataclass
    class FakeReconstruction:
        sensor: str
        start_time: pd.Timestamp
        end_time: pd.Timestamp
        values: pd.Series
        method: str

    from staleness_pipeline.detection import StuckPeriod

    period = StuckPeriod(
        sensor="aht_temperature",
        stuck_value=20.0,
        start_time=pd.Timestamp("2026-01-01T00:00:00+00:00"),
        end_time=pd.Timestamp("2026-01-01T00:30:00+00:00"),
        duration_hours=0.5,
    )
    recon = FakeReconstruction(
        sensor="aht_temperature",
        start_time=period.start_time,
        end_time=period.end_time,
        values=pd.Series(
            [20.1, 20.2],
            index=pd.DatetimeIndex(["2026-01-01T00:10:00+00:00", "2026-01-01T00:20:00+00:00"]),
            name="aht_temperature",
        ),
        method="chronos-bolt-small-bidirectional",
    )
    event = UpdateEvent(
        point_id="p1",
        sensor="aht_temperature",
        timestamp=pd.Timestamp("2026-01-01T00:30:00+00:00"),
        value=20.0,
        new_periods=[period],
        new_reconstructions=[recon],
    )

    messages = dashboard.event_to_messages(event)

    assert messages[0]["type"] == "point"
    assert messages[1]["type"] == "reconstruction"
    assert messages[1]["confidence"] == "RECONCILED"
    assert len(messages[1]["values"]) == 2


def test_event_to_messages_marks_open_ended_as_provisional():
    from dataclasses import dataclass

    from staleness_pipeline.detection import StuckPeriod

    @dataclass
    class FakeReconstruction:
        sensor: str
        start_time: pd.Timestamp
        end_time: pd.Timestamp
        values: pd.Series
        method: str

    period = StuckPeriod(
        sensor="aht_temperature",
        stuck_value=20.0,
        start_time=pd.Timestamp("2026-01-01T00:00:00+00:00"),
        end_time=pd.Timestamp("2026-01-01T00:30:00+00:00"),
        duration_hours=0.5,
    )
    recon = FakeReconstruction(
        sensor="aht_temperature",
        start_time=period.start_time,
        end_time=period.end_time,
        values=pd.Series([20.1], index=pd.DatetimeIndex(["2026-01-01T00:10:00+00:00"]), name="aht_temperature"),
        method="chronos-bolt-small-forward-only",
    )
    event = UpdateEvent(
        point_id="p1",
        sensor="aht_temperature",
        timestamp=period.end_time,
        value=20.0,
        new_periods=[period],
        new_reconstructions=[recon],
    )

    messages = dashboard.event_to_messages(event)

    assert messages[1]["confidence"] == "PROVISIONAL"