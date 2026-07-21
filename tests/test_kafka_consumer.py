"""Tests for kafka_consumer.py.

FakeRecord/fake iterable stand in for a real KafkaConsumer's stream of
ConsumerRecord objects — real ones expose .value (and more), which is all
consume_messages() relies on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from staleness_pipeline import kafka_consumer


@dataclass
class FakeRecord:
    value: Any


def make_message(sensor="aht_temperature", value=21.0, point_id="point-1", ts="2026-01-01T00:00:00+00:00"):
    return {"point_id": point_id, "sensor": sensor, "timestamp": ts, "value": value}


def test_consume_messages_yields_parsed_dicts_in_order():
    records = [FakeRecord(make_message(value=1.0)), FakeRecord(make_message(value=2.0))]

    messages = list(kafka_consumer.consume_messages(records))

    assert [m["value"] for m in messages] == [1.0, 2.0]


def test_consume_messages_respects_max_messages():
    records = [FakeRecord(make_message(value=float(i))) for i in range(10)]

    messages = list(kafka_consumer.consume_messages(records, max_messages=3))

    assert len(messages) == 3
    assert [m["value"] for m in messages] == [0.0, 1.0, 2.0]


def test_consume_messages_calls_handler_for_each_message():
    records = [FakeRecord(make_message(value=1.0)), FakeRecord(make_message(value=2.0))]
    seen = []

    list(kafka_consumer.consume_messages(records, handler=lambda m: seen.append(m["value"])))

    assert seen == [1.0, 2.0]


def test_consume_messages_handles_empty_stream():
    messages = list(kafka_consumer.consume_messages([]))

    assert messages == []


def test_consume_messages_max_messages_zero_yields_nothing():
    records = [FakeRecord(make_message())]

    messages = list(kafka_consumer.consume_messages(records, max_messages=0))

    assert messages == []
