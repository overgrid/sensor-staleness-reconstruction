"""Tests for live_store.py.

FakeReconstructFn stands in for reconstruction.reconstruct_stale_window()
with its pipeline already bound — same "fake the model, test the wiring"
philosophy as test_reconstruction.py / test_offline_job.py, just one
level up: nothing here needs torch or a real Chronos model.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from staleness_pipeline.detection import StuckPeriod
from staleness_pipeline.live_store import LiveSeriesStore


@dataclass
class FakeReconstruction:
    """Duck-typed stand-in for reconstruction.Reconstruction — same
    fields, no torch import required. live_store.py never does an
    isinstance check on this, only attribute access, so this is a valid
    substitute for testing purposes."""

    sensor: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    values: pd.Series
    method: str


class RecordingReconstructFn:
    """Records every call and returns a canned FakeReconstruction, tagging
    method name by whether context_after existed — mirrors
    reconstruct_stale_window()'s own forward-only vs. bidirectional
    naming so LiveSeriesStore's PROVISIONAL/RECONCILED logic (which reads
    real data presence, not the method string) can be tested independently."""

    def __init__(self):
        self.calls: list[tuple[pd.Series, StuckPeriod]] = []

    def __call__(self, series: pd.Series, period: StuckPeriod) -> FakeReconstruction:
        self.calls.append((series, period))
        gap_index = series.index[(series.index >= period.start_time) & (series.index <= period.end_time)]
        has_after = not series.loc[series.index > period.end_time].dropna().empty
        method = "chronos-bolt-small-bidirectional" if has_after else "chronos-bolt-small-forward-only"
        return FakeReconstruction(
            sensor=str(series.name),
            start_time=period.start_time,
            end_time=period.end_time,
            values=pd.Series([99.0] * len(gap_index), index=gap_index, name=series.name),
            method=method,
        )


def msg(point_id="point-1", sensor="aht_temperature", ts="2026-01-01T00:00:00+00:00", value=20.0):
    return {"point_id": point_id, "sensor": sensor, "timestamp": ts, "value": value}


def ts_seq(start="2026-01-01T00:00:00Z", periods=20, freq="10min"):
    return pd.date_range(start=start, periods=periods, freq=freq, tz="UTC")


def test_ingest_single_point_creates_series():
    store = LiveSeriesStore(reconstruct_fn=RecordingReconstructFn())

    store.ingest(msg(value=21.0))

    series = store.get_series(("point-1", "aht_temperature"))
    assert list(series.to_numpy()) == [21.0]


def test_ingest_no_reconstruction_call_when_nothing_stuck_yet():
    fn = RecordingReconstructFn()
    store = LiveSeriesStore(reconstruct_fn=fn, min_stuck_hours=0.25)
    times = ts_seq(periods=5)

    for t, v in zip(times, [20.0, 20.5, 21.0, 21.5, 22.0]):
        store.ingest(msg(ts=t.isoformat(), value=v))

    assert fn.calls == []


def test_resolved_stuck_period_gets_reconstructed_and_finalized_as_reconciled():
    fn = RecordingReconstructFn()
    store = LiveSeriesStore(reconstruct_fn=fn, min_stuck_hours=0.25, min_context_points=3)
    times = ts_seq(periods=10)
    # 3 real points, then stuck at 20.0 for 4 points (30 min, above the
    # 0.25h threshold), then 3 real points resolving it.
    values = [20.0, 20.2, 20.4, 20.0, 20.0, 20.0, 20.0, 21.0, 21.2, 21.4]

    events = [store.ingest(msg(ts=t.isoformat(), value=v)) for t, v in zip(times, values)]

    # The still-open run legitimately refreshes more than once as it
    # grows (see test_open_ended_period_gets_refreshed_as_it_grows) — the
    # thing that matters here is that the LAST reconstruction, once real
    # data resumes after it, is bidirectional/RECONCILED, not that it
    # only ever fired once.
    reconstructed_events = [e for e in events if e.new_reconstructions]
    assert len(reconstructed_events) >= 1
    final_reconstruction = reconstructed_events[-1].new_reconstructions[0]
    assert final_reconstruction.method == "chronos-bolt-small-bidirectional"


def test_resolved_period_not_reconstructed_again_after_more_data_arrives():
    fn = RecordingReconstructFn()
    store = LiveSeriesStore(reconstruct_fn=fn, min_stuck_hours=0.25, min_context_points=3)
    times = ts_seq(periods=12)
    values = [20.0, 20.2, 20.4, 20.0, 20.0, 20.0, 20.0, 21.0, 21.2, 21.4, 21.6, 21.8]

    for t, v in zip(times, values):
        store.ingest(msg(ts=t.isoformat(), value=v))

    calls_after_resolution = len(fn.calls)

    # Feed three more real, non-stuck points — the already-RECONCILED
    # period must not be reconstructed again.
    more_times = pd.date_range(times[-1] + pd.Timedelta(minutes=10), periods=3, freq="10min", tz="UTC")
    for t, v in zip(more_times, [22.0, 22.2, 22.4]):
        store.ingest(msg(ts=t.isoformat(), value=v))

    assert len(fn.calls) == calls_after_resolution


def test_open_ended_period_reconstructed_as_provisional_by_default():
    fn = RecordingReconstructFn()
    store = LiveSeriesStore(reconstruct_fn=fn, min_stuck_hours=0.25, min_context_points=3)
    times = ts_seq(periods=7)
    # 3 real points, then stuck for 4 points with NO resolution yet.
    values = [20.0, 20.2, 20.4, 20.0, 20.0, 20.0, 20.0]

    events = [store.ingest(msg(ts=t.isoformat(), value=v)) for t, v in zip(times, values)]

    reconstructed = [e for e in events if e.new_reconstructions]
    assert len(reconstructed) >= 1
    # still open-ended by the end of this window (never resolved) — every
    # reconstruction produced must be forward-only/PROVISIONAL.
    assert all(
        e.new_reconstructions[0].method == "chronos-bolt-small-forward-only" for e in reconstructed
    )


def test_open_ended_period_skipped_when_reconstruct_open_ended_false():
    fn = RecordingReconstructFn()
    store = LiveSeriesStore(
        reconstruct_fn=fn, min_stuck_hours=0.25, min_context_points=3, reconstruct_open_ended=False,
    )
    times = ts_seq(periods=7)
    values = [20.0, 20.2, 20.4, 20.0, 20.0, 20.0, 20.0]

    for t, v in zip(times, values):
        store.ingest(msg(ts=t.isoformat(), value=v))

    assert fn.calls == []


def test_open_ended_period_gets_refreshed_as_it_grows():
    """An ongoing stuck run (still open-ended) should be able to trigger
    a fresh reconstruction as it extends — unlike a resolved one, it's
    deliberately not marked 'done' forever."""
    fn = RecordingReconstructFn()
    store = LiveSeriesStore(reconstruct_fn=fn, min_stuck_hours=0.25, min_context_points=3)
    times = ts_seq(periods=9)
    # stuck run keeps extending, never resolves within this window
    values = [20.0, 20.2, 20.4] + [20.0] * 6

    for t, v in zip(times, values):
        store.ingest(msg(ts=t.isoformat(), value=v))

    # Called more than once as the still-open run kept growing.
    assert len(fn.calls) >= 2


def test_insufficient_context_points_skips_reconstruction():
    fn = RecordingReconstructFn()
    store = LiveSeriesStore(reconstruct_fn=fn, min_stuck_hours=0.25, min_context_points=10)
    times = ts_seq(periods=7)
    values = [20.0, 20.2, 20.4, 20.0, 20.0, 20.0, 20.0]

    for t, v in zip(times, values):
        store.ingest(msg(ts=t.isoformat(), value=v))

    assert fn.calls == []  # only 3 real context points, need 10


def test_value_error_from_reconstruct_fn_is_swallowed_not_raised():
    def raising_fn(series, period):
        raise ValueError("no real data before period.start_time")

    store = LiveSeriesStore(reconstruct_fn=raising_fn, min_stuck_hours=0.25, min_context_points=3)
    times = ts_seq(periods=7)
    values = [20.0, 20.2, 20.4, 20.0, 20.0, 20.0, 20.0]

    events = [store.ingest(msg(ts=t.isoformat(), value=v)) for t, v in zip(times, values)]

    assert all(e.new_reconstructions == [] for e in events)  # no crash, just skipped


def test_max_points_trims_oldest_first():
    store = LiveSeriesStore(reconstruct_fn=RecordingReconstructFn(), max_points=5)
    times = ts_seq(periods=8)

    for i, t in enumerate(times):
        store.ingest(msg(ts=t.isoformat(), value=float(i)))

    series = store.get_series(("point-1", "aht_temperature"))
    assert len(series) == 5
    assert list(series.to_numpy()) == [3.0, 4.0, 5.0, 6.0, 7.0]


def test_duplicate_timestamp_keeps_latest_value():
    store = LiveSeriesStore(reconstruct_fn=RecordingReconstructFn())
    ts = "2026-01-01T00:00:00+00:00"

    store.ingest(msg(ts=ts, value=1.0))
    store.ingest(msg(ts=ts, value=2.0))

    series = store.get_series(("point-1", "aht_temperature"))
    assert len(series) == 1
    assert series.iloc[0] == 2.0


def test_multiple_sensors_tracked_independently():
    store = LiveSeriesStore(reconstruct_fn=RecordingReconstructFn())
    ts = "2026-01-01T00:00:00+00:00"

    store.ingest(msg(sensor="aht_temperature", ts=ts, value=20.0))
    store.ingest(msg(sensor="aht_humidity", ts=ts, value=55.0))

    assert set(store.keys()) == {("point-1", "aht_temperature"), ("point-1", "aht_humidity")}


def test_nan_value_does_not_crash_ingestion():
    store = LiveSeriesStore(reconstruct_fn=RecordingReconstructFn())

    event = store.ingest(msg(value=None))

    assert event.value is None
    series = store.get_series(("point-1", "aht_temperature"))
    assert pd.isna(series.iloc[0])
