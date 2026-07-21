"""Reads the simulated live sensor feed back off Kafka.

Deliberately minimal for now — just enough to smoke-test that
kafka_producer.py's messages arrive correctly, and to give the next stage
(real-time detection + reconstruction + live visualization) a tested
starting point to build on, rather than starting that stage from zero.

Mirrors kafka_producer.py's message schema exactly:
    {"point_id": str, "sensor": str, "timestamp": str (ISO 8601 UTC), "value": float | None}
"""

from __future__ import annotations

import json
import logging
from typing import Callable, Iterator, Optional

logger = logging.getLogger(__name__)

DEFAULT_TOPIC = "sensor-readings"
DEFAULT_BOOTSTRAP_SERVERS = "localhost:9092"
DEFAULT_GROUP_ID = "staleness-pipeline"


def build_consumer(
    topic: str = DEFAULT_TOPIC,
    bootstrap_servers: str = DEFAULT_BOOTSTRAP_SERVERS,
    group_id: str = DEFAULT_GROUP_ID,
    auto_offset_reset: str = "latest",
):
    """Return a configured KafkaConsumer. Imported lazily, same reasoning
    as build_producer() in kafka_producer.py — no kafka-python or broker
    required just to import this module.

    auto_offset_reset="latest" (the default) means a freshly-started
    consumer only sees NEW messages, matching "live feed" semantics. Pass
    "earliest" for smoke tests/demos that want to replay everything
    already sitting in the topic.
    """
    from kafka import KafkaConsumer

    return KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        auto_offset_reset=auto_offset_reset,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        key_deserializer=lambda k: k.decode("utf-8") if k is not None else None,
    )


def consume_messages(
    consumer,
    max_messages: Optional[int] = None,
    handler: Optional[Callable[[dict], None]] = None,
) -> Iterator[dict]:
    """Iterate over incoming messages as plain dicts.

    Args:
        consumer: a real KafkaConsumer (iterable of records with .value),
            or any iterable of objects exposing .value for tests.
        max_messages: stop after this many messages instead of blocking
            forever — used by smoke tests/demos, omit for a real
            long-running consumer.
        handler: optional callback invoked with each message dict as it
            arrives, before it's yielded — a convenient hook for the
            future real-time detection pipeline without needing to
            restructure this loop.

    Yields:
        One parsed message dict per Kafka record, in the same shape
        kafka_producer.make_message() produces.
    """
    if max_messages is not None and max_messages <= 0:
        return

    count = 0
    for record in consumer:
        message = record.value
        if handler is not None:
            handler(message)
        yield message

        count += 1
        if max_messages is not None and count >= max_messages:
            break
