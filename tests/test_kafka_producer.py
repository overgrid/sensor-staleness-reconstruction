"""Tests for kafka_producer.py.

Same philosophy as the rest of the project's tests (e.g.
test_chronos_model.py's fake pipeline): a FakeKafkaProducer stands in for
the real thing so these run in milliseconds, with no real broker and no
real wall-clock waiting required.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from staleness_pipeline import kafka_producer


class FakeKafkaProducer:
    """Records every send() call instead of talking to a real broker."""

    def __init__(self):
        self.sent = []
        self.flushed = False
        self.closed = False

    def send(self, topic, key=None, value=None):
        self.sent.append({"topic": topic, "key": key, "value": value})

    def flush(self):
        self.flushed = True

    def close(self):
        self.closed = True


def make_series(name="aht_temperature", start="2026-01-01T00:00:00Z", periods=4, freq="10min", values=None):
    index = pd.date_range(start=start, periods=periods, freq=freq, tz="UTC")
    if values is None:
        values = [20.0 + 0.5 * i for i in range(periods)]
    return pd.Series(values, index=index, name=name)


def test_make_message_formats_real_value():
    ts = pd.Timestamp("2026-01-01T00:00:00", tz="UTC")
    msg = kafka_producer.make_message("point-1", "aht_temperature", ts, 21.5)

    assert msg == {
        "point_id": "point-1",
        "sensor": "aht_temperature",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "value": 21.5,
    }


def test_make_message_naive_timestamp_gets_localized_to_utc():
    ts = pd.Timestamp("2026-01-01T00:00:00")  # no tzinfo
    msg = kafka_producer.make_message("point-1", "aht_temperature", ts, 1.0)

    assert msg["timestamp"] == "2026-01-01T00:00:00+00:00"


def test_make_message_nan_value_becomes_json_null_not_dropped():
    ts = pd.Timestamp("2026-01-01T00:00:00", tz="UTC")
    msg = kafka_producer.make_message("point-1", "aht_temperature", ts, float("nan"))

    assert msg["value"] is None


def test_make_message_none_value_stays_none():
    ts = pd.Timestamp("2026-01-01T00:00:00", tz="UTC")
    msg = kafka_producer.make_message("point-1", "aht_temperature", ts, None)

    assert msg["value"] is None


def test_stream_series_to_kafka_sends_one_message_per_point():
    producer = FakeKafkaProducer()
    series = make_series(periods=4)

    messages = list(
        kafka_producer.stream_series_to_kafka(
            producer, series, point_id="point-1", topic="sensor-readings",
            sleep_fn=lambda s: None, remap_to_now=False,
        )
    )

    assert len(messages) == 4
    assert len(producer.sent) == 4
    assert [m["value"] for m in messages] == [20.0, 20.5, 21.0, 21.5]
    assert all(sent["key"] == "point-1" for sent in producer.sent)
    assert all(sent["topic"] == "sensor-readings" for sent in producer.sent)
    assert producer.flushed is True


def test_stream_series_to_kafka_preserves_message_order():
    producer = FakeKafkaProducer()
    series = make_series(periods=5)

    list(
        kafka_producer.stream_series_to_kafka(
            producer, series, point_id="point-1", sleep_fn=lambda s: None, remap_to_now=False,
        )
    )

    sent_values = [s["value"]["value"] for s in producer.sent]
    assert sent_values == sorted(sent_values)


def test_stream_series_to_kafka_sleeps_real_delta_scaled_by_speed_multiplier():
    producer = FakeKafkaProducer()
    series = make_series(periods=3, freq="10min")  # 600s real gaps
    sleeps = []

    list(
        kafka_producer.stream_series_to_kafka(
            producer, series, point_id="point-1", speed_multiplier=60.0,
            sleep_fn=lambda s: sleeps.append(s), remap_to_now=False,
        )
    )

    # 3 points -> 2 gaps, each 600s real / 60x speed = 10s simulated sleep
    assert len(sleeps) == 2
    assert all(math.isclose(s, 10.0) for s in sleeps)


def test_stream_series_to_kafka_handles_irregular_real_cadence():
    """Real data isn't perfectly evenly spaced (see reconstruction.py bug
    #3) — the replay must sleep based on the ACTUAL gap between each pair
    of points, not an assumed fixed cadence."""
    producer = FakeKafkaProducer()
    index = pd.DatetimeIndex(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:10:00Z",  # 600s gap
            "2026-01-01T00:25:00Z",  # 900s gap (irregular!)
        ]
    )
    series = pd.Series([1.0, 2.0, 3.0], index=index, name="aht_temperature")
    sleeps = []

    list(
        kafka_producer.stream_series_to_kafka(
            producer, series, point_id="point-1", speed_multiplier=60.0,
            sleep_fn=lambda s: sleeps.append(s), remap_to_now=False,
        )
    )

    assert math.isclose(sleeps[0], 10.0)
    assert math.isclose(sleeps[1], 15.0)


def test_stream_series_to_kafka_remaps_timestamps_to_now_by_default():
    producer = FakeKafkaProducer()
    series = make_series(start="2020-01-01T00:00:00Z", periods=2, freq="10min")
    fixed_now = pd.Timestamp("2026-07-21T12:00:00", tz="UTC")

    messages = list(
        kafka_producer.stream_series_to_kafka(
            producer, series, point_id="point-1",
            sleep_fn=lambda s: None, now_fn=lambda: fixed_now,
        )
    )

    first_ts = pd.Timestamp(messages[0]["timestamp"])
    second_ts = pd.Timestamp(messages[1]["timestamp"])
    assert first_ts == fixed_now
    # original cadence (10 minutes) must be preserved exactly under remap
    assert (second_ts - first_ts) == pd.Timedelta(minutes=10)


def test_stream_series_to_kafka_remap_false_keeps_original_timestamps():
    producer = FakeKafkaProducer()
    series = make_series(start="2020-01-01T00:00:00Z", periods=2, freq="10min")

    messages = list(
        kafka_producer.stream_series_to_kafka(
            producer, series, point_id="point-1",
            sleep_fn=lambda s: None, remap_to_now=False,
        )
    )

    assert messages[0]["timestamp"] == "2020-01-01T00:00:00+00:00"


class _StopTest(Exception):
    """Sentinel used to break out of an intentionally-infinite loop=True
    replay in tests. Plain StopIteration can't be used here — PEP 479
    converts a StopIteration raised inside a generator into a
    RuntimeError, which would make this test assert the wrong thing."""


def test_stream_series_to_kafka_loop_stops_after_two_passes_when_flagged():
    producer = FakeKafkaProducer()
    series = make_series(periods=2)
    call_count = {"n": 0}

    def now_fn():
        call_count["n"] += 1
        if call_count["n"] > 2:
            raise _StopTest()
        return pd.Timestamp("2026-07-21T12:00:00", tz="UTC")

    gen = kafka_producer.stream_series_to_kafka(
        producer, series, point_id="point-1", loop=True,
        sleep_fn=lambda s: None, now_fn=now_fn,
    )

    with pytest.raises(_StopTest):
        list(gen)

    # 2 full passes of a 2-point series completed before the 3rd now_fn() call blew up
    assert len(producer.sent) == 4


def test_stream_series_to_kafka_empty_series_sends_nothing():
    producer = FakeKafkaProducer()
    empty = pd.Series([], dtype=float, name="aht_temperature")

    messages = list(
        kafka_producer.stream_series_to_kafka(producer, empty, point_id="point-1", sleep_fn=lambda s: None)
    )

    assert messages == []
    assert producer.sent == []


def test_stream_csv_to_kafka_streams_multiple_columns_concurrently(tmp_path, monkeypatch):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(
        "timestamp,pointA__Sensor__aht_temperature,pointA__Sensor__aht_humidity\n"
        "2026-01-01T00:00:00Z,20.0,50.0\n"
        "2026-01-01T00:10:00Z,20.5,51.0\n"
    )

    producer = FakeKafkaProducer()
    monkeypatch.setattr(kafka_producer, "build_producer", lambda bootstrap_servers: producer)

    kafka_producer.stream_csv_to_kafka(
        csv_path=str(csv_path),
        columns=["pointA__Sensor__aht_temperature", "pointA__Sensor__aht_humidity"],
        point_id="pointA",
        speed_multiplier=1_000_000.0,  # keep the (real, un-mocked) sleep negligible
    )

    sensors_sent = {sent["value"]["sensor"] for sent in producer.sent}
    assert sensors_sent == {"aht_temperature", "aht_humidity"}
    assert len(producer.sent) == 4  # 2 points x 2 columns
    assert producer.flushed is True
    assert producer.closed is True


def test_generate_synthetic_series_is_a_clearly_marked_stub():
    with pytest.raises(NotImplementedError):
        kafka_producer.generate_synthetic_series()
