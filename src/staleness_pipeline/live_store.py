"""In-memory rolling state for the live/online pipeline.

This is the part of the online consumer that doesn't know or care about
Kafka, FastAPI, WebSockets, or Chronos specifics — just: given messages
arriving one at a time, maintain a rolling per-sensor buffer, run the
SAME find_stuck_periods() the offline job already uses, and call an
injected reconstruction function once a period becomes worth
reconstructing. Kept framework-agnostic and dependency-light on purpose
so it's testable without a real broker, a real model, or a real web
server — the same separation already used elsewhere in this project
(tracking.py's Tracker abstraction, storage.py's MeasurementSink).

This is also where storage.py's ImputationConfidence.PROVISIONAL
actually starts getting produced — see the project's open items: it was
defined from the start but nothing produced it until now. A period with
no real data after it yet (still actively stuck, in a live feed) gets
reconstructed forward-only and is PROVISIONAL; once real data resumes
after it, later ingestion re-reconstructs it bidirectionally and it
becomes RECONCILED — matching reconstruct_stale_window()'s own forward-
only vs. bidirectional split in reconstruction.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

import pandas as pd

from staleness_pipeline.detection import StuckPeriod, find_stuck_periods

if TYPE_CHECKING:
    # Only needed for type hints below — importing reconstruction.py for
    # real would pull in torch at runtime, which this module has no other
    # reason to need. `from __future__ import annotations` (above) makes
    # every annotation in this file a lazy string, so this guard is safe.
    from staleness_pipeline.reconstruction import Reconstruction

SeriesKey = tuple[str, str]  # (point_id, sensor)


@dataclass
class UpdateEvent:
    """What changed after ingesting one message — enough for a caller
    (the WebSocket layer) to broadcast an incremental update without
    re-sending the whole buffer every time."""

    point_id: str
    sensor: str
    timestamp: pd.Timestamp
    value: Optional[float]
    new_periods: list[StuckPeriod] = field(default_factory=list)
    new_reconstructions: list[Reconstruction] = field(default_factory=list)


class LiveSeriesStore:
    """Rolling per-(point_id, sensor) buffer + incremental stuck-period
    detection + reconstruction, fed one Kafka message at a time.

    Args:
        reconstruct_fn: called as reconstruct_fn(series, period) ->
            Reconstruction — same signature as
            reconstruction.reconstruct_stale_window() with its `pipeline`
            argument already bound (e.g. via functools.partial). Real
            code binds a real Chronos pipeline; tests bind a fake one
            with no model weights required, same pattern already used in
            test_reconstruction.py / test_offline_job.py.
        max_points: rolling buffer cap per sensor — bounds memory for an
            indefinitely-running live consumer (kafka_producer.py's
            loop=True keeps sending forever). Oldest points are dropped
            first.
        min_stuck_hours: same meaning/default as detection.py.
        reconstruct_open_ended: if True, also reconstruct a stuck run
            that hasn't resolved yet (no real data after it in the
            buffer) — forward-only, PROVISIONAL. If False, only
            reconstructs periods that already have real data on both
            sides.
        min_context_points: minimum real (non-null) points required
            before a period's start before attempting to reconstruct it
            at all — mirrors reconstruct_stale_window()'s own "no real
            data before period.start_time" guard, just checked earlier so
            we can skip cleanly instead of relying on it to raise.
    """

    def __init__(
        self,
        reconstruct_fn: Callable[[pd.Series, StuckPeriod], Reconstruction],
        max_points: int = 2000,
        min_stuck_hours: float = 0.25,
        reconstruct_open_ended: bool = True,
        min_context_points: int = 5,
    ):
        self._reconstruct_fn = reconstruct_fn
        self._max_points = max_points
        self._min_stuck_hours = min_stuck_hours
        self._reconstruct_open_ended = reconstruct_open_ended
        self._min_context_points = min_context_points

        self._series: dict[SeriesKey, pd.Series] = {}
        # Which periods (by start_time) are already RECONCILED — a
        # resolved period is done for good; an open-ended one is
        # deliberately NOT added here, since more of it may still arrive
        # and each new point should be able to extend/refresh it.
        self._reconciled_starts: dict[SeriesKey, set] = {}

    def ingest(self, message: dict) -> UpdateEvent:
        """Add one Kafka message (kafka_consumer.py's dict shape — keys
        point_id, sensor, timestamp, value) to the right buffer, run
        detection, reconstruct any newly-eligible period, and return what
        changed."""
        point_id = message["point_id"]
        sensor = message["sensor"]
        ts = pd.Timestamp(message["timestamp"])
        value = message["value"]
        key = (point_id, sensor)

        existing = self._series.get(key)
        new_point = pd.Series([value], index=pd.DatetimeIndex([ts]), name=sensor)
        if existing is None:
            series = new_point
        else:
            series = pd.concat([existing, new_point])
            # A duplicate/out-of-order timestamp (possible on a real
            # feed) keeps the newest value rather than growing forever.
            series = series[~series.index.duplicated(keep="last")].sort_index()

        if len(series) > self._max_points:
            series = series.iloc[-self._max_points :]
        self._series[key] = series

        event = UpdateEvent(point_id=point_id, sensor=sensor, timestamp=ts, value=value)

        if series.dropna().empty:
            return event

        periods = find_stuck_periods(series, min_stuck_hours=self._min_stuck_hours)
        reconciled = self._reconciled_starts.setdefault(key, set())

        for period in periods:
            if period.start_time in reconciled:
                continue

            context_before = series.loc[series.index < period.start_time].dropna()
            if len(context_before) < self._min_context_points:
                continue  # not enough real history yet to forecast from

            is_open_ended = series.loc[series.index > period.end_time].dropna().empty
            if is_open_ended and not self._reconstruct_open_ended:
                continue

            try:
                reconstruction = self._reconstruct_fn(series, period)
            except ValueError:
                # Same "not enough context" case reconstruct_stale_window()
                # itself guards against — skip this period this round,
                # try again as more real data arrives.
                continue

            event.new_periods.append(period)
            event.new_reconstructions.append(reconstruction)
            if not is_open_ended:
                reconciled.add(period.start_time)

        return event

    def get_series(self, key: SeriesKey) -> Optional[pd.Series]:
        return self._series.get(key)

    def keys(self) -> list[SeriesKey]:
        return list(self._series.keys())
