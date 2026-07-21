"""Simulates a live sensor feed by replaying real CSV data onto Kafka.

This is deliberately its own module, mirroring the split already used
elsewhere in this package (chronos_model.py is the only module that
imports chronos; this is the only module that imports kafka directly).
The online/live pipeline (detection + reconstruction running against a
Kafka stream) will consume from the same topic this module produces to —
that's the next stage, built on top of this one.

Two things this module is careful about, both driven by real lessons
already learned in reconstruction.py:

  - Real sensor timestamps are NOT perfectly evenly spaced (see
    reconstruction.py bug #3). The replay sleeps based on the ACTUAL
    delta between consecutive real points (scaled by speed_multiplier),
    not an assumed fixed cadence — so the simulated stream reproduces the
    same irregularities a live detector will eventually have to handle.
  - Measurement.value is nullable (see data_source.py). NaN values in the
    source series are sent as JSON null, not silently coerced to 0 or
    dropped, since the whole point of this pipeline is handling missing/
    stuck data honestly.

Message schema (one JSON object per Kafka message):
    {
        "point_id": str,
        "sensor": str,             # e.g. "aht_temperature"
        "timestamp": str,          # ISO 8601, UTC
        "value": float | null,
    }

Synthetic data generation (producing readings that never existed in any
CSV, to stress-test longer/rarer gap patterns than the real dataset
happens to contain) is intentionally NOT built yet — see
generate_synthetic_series() below and the project's open items list.
This module only replays real data for now.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Callable, Iterable, Iterator

import pandas as pd

from staleness_pipeline.data_source import load_series_from_csv

logger = logging.getLogger(__name__)

DEFAULT_TOPIC = "sensor-readings"
DEFAULT_BOOTSTRAP_SERVERS = "localhost:9092"


def build_producer(bootstrap_servers: str = DEFAULT_BOOTSTRAP_SERVERS):
    """Return a configured KafkaProducer. Imported lazily so the rest of
    the package (and its tests) don't require kafka-python or a running
    broker just to import this module."""
    from kafka import KafkaProducer

    return KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k if isinstance(k, bytes) else str(k).encode("utf-8"),
    )


def make_message(point_id: str, sensor: str, timestamp: pd.Timestamp, value: float | None) -> dict:
    """Build one wire-format message. Pure function — no Kafka, no I/O —
    so it's trivially unit-testable and reusable by the future consumer
    side for round-trip tests.

    Args:
        value: the raw reading, or NaN/None for a missing point. Always
            serialized as JSON null, never silently dropped or zeroed.
    """
    ts = timestamp if timestamp.tzinfo is not None else timestamp.tz_localize("UTC")
    clean_value = None if value is None or pd.isna(value) else float(value)
    return {
        "point_id": point_id,
        "sensor": sensor,
        "timestamp": ts.tz_convert("UTC").isoformat(),
        "value": clean_value,
    }


def stream_series_to_kafka(
    producer,
    series: pd.Series,
    point_id: str,
    topic: str = DEFAULT_TOPIC,
    speed_multiplier: float = 60.0,
    loop: bool = False,
    remap_to_now: bool = True,
    sleep_fn: Callable[[float], None] = time.sleep,
    now_fn: Callable[[], pd.Timestamp] = lambda: pd.Timestamp.now(tz="UTC"),
) -> Iterator[dict]:
    """Replay one already-loaded sensor series onto a Kafka topic.

    Sleeps between sends so wall-clock gaps mirror the REAL gaps between
    consecutive points in `series`, divided by `speed_multiplier` (e.g.
    speed_multiplier=60 turns a real 10-minute cadence into a 10-second
    wait). This is a generator — it yields every message dict as it's
    sent, both so tests can inspect exactly what went out without needing
    a real broker to read back, and so a future CLI/demo can print
    progress live.

    Args:
        producer: a Kafka producer exposing .send(topic, key=..., value=...)
            — a real KafkaProducer from build_producer(), or a fake with
            the same interface for tests.
        series: full sensor series (as returned by data_source.py),
            indexed by real UTC timestamp.
        point_id: the Overgrid point this belongs to.
        speed_multiplier: how much faster than real time to replay.
        loop: if True, keep replaying indefinitely (Ctrl+C to stop) — the
            way to simulate a continuously live feed rather than a single
            finite replay.
        remap_to_now: if True (default), shift all timestamps by a fixed
            offset so the first point lands at "now" — preserving every
            real relative gap (including the irregular ones) exactly,
            while giving downstream consumers sensible current
            timestamps instead of years-old historical ones. Each loop
            iteration recomputes the offset from the current "now", so a
            looped replay always looks fresh rather than jumping back in
            time. Set False to send the original historical timestamps
            as-is.
        sleep_fn / now_fn: injection points for tests — real code should
            never need to pass these.
    """
    if series.empty:
        return

    while True:
        offset = (now_fn() - series.index[0]) if remap_to_now else pd.Timedelta(0)
        n = len(series)

        for i in range(n):
            ts = series.index[i] + offset
            value = series.iloc[i]
            message = make_message(point_id, str(series.name), ts, value)

            producer.send(topic, key=point_id, value=message)
            yield message

            if i < n - 1:
                real_delta_seconds = (series.index[i + 1] - series.index[i]).total_seconds()
                sleep_fn(max(real_delta_seconds / speed_multiplier, 0))

        flush = getattr(producer, "flush", None)
        if flush is not None:
            flush()

        if not loop:
            break


def stream_csv_to_kafka(
    csv_path: str,
    columns: Iterable[str],
    point_id: str,
    topic: str = DEFAULT_TOPIC,
    bootstrap_servers: str = DEFAULT_BOOTSTRAP_SERVERS,
    speed_multiplier: float = 60.0,
    loop: bool = False,
    remap_to_now: bool = True,
) -> None:
    """Load one or more CSV columns and stream them onto Kafka concurrently.

    Mirrors run_offline_job_live()'s handling of multiple attributes per
    point: one real sensor per column (e.g. temperature and humidity),
    streamed independently and concurrently on their own real cadence, all
    onto the same topic (partitioned by point_id as the message key so a
    future consumer can keep each point's ordering intact).

    Blocks until all series finish (or forever if loop=True — the caller,
    e.g. the CLI, is expected to run this in a process the user can
    interrupt with Ctrl+C).
    """
    producer = build_producer(bootstrap_servers)
    threads = []

    for column in columns:
        series = load_series_from_csv(csv_path, column=column)
        logger.info("Loaded %d points for column=%s (sensor=%s)", len(series), column, series.name)

        def _run(s: pd.Series = series) -> None:
            for _ in stream_series_to_kafka(
                producer,
                s,
                point_id=point_id,
                topic=topic,
                speed_multiplier=speed_multiplier,
                loop=loop,
                remap_to_now=remap_to_now,
            ):
                pass

        threads.append(threading.Thread(target=_run, daemon=True))

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    producer.flush()
    producer.close()


def generate_synthetic_series(*args, **kwargs):
    """NOT YET IMPLEMENTED — synthetic sensor data generation.

    Intent (see project open items): produce sensor-shaped data that never
    existed in any real CSV, so validation and the live pipeline can be
    stress-tested against gap lengths, cadences, and stuck-run patterns
    the real ~30-day dataset doesn't happen to contain. Deliberately
    deferred until CSV replay via Kafka is confirmed working end to end —
    same "prove the simpler thing first" approach used for every other
    module in this project (see chronos_model.py's fake-pipeline tests,
    synthetic_injection.py's real-data-based fake gaps, etc.).

    When built, this should return a pd.Series with the same shape
    data_source.load_series_from_csv() produces (DatetimeIndex, named
    after the sensor) so it's a drop-in alternative to
    load_series_from_csv() wherever a series is needed — including as a
    direct input to stream_series_to_kafka() above.
    """
    raise NotImplementedError(
        "Synthetic data generation is planned but not built yet — "
        "use stream_csv_to_kafka() / stream_series_to_kafka() with real "
        "CSV data for now."
    )
