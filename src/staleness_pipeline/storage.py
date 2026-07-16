"""Where reconstructed measurements get written.

MeasurementSink is the one deliberate swap point in the whole pipeline.
Today the only real implementation is LocalJSONLSink. GraphQLSink stays a
stub — schema_1.gqls has no imputation-aware fields (no isImputed,
imputedValue, etc.), and its only write path, WritablePoint.write(), would
silently overwrite the real reading with no way to tell them apart
afterward — that breaks the "never overwrite raw data" rule this whole
project is built around. GraphQLSink becomes real once that's resolved
(a real schema extension, or a separate shadow point/attribute) — not
before.

Every record keeps raw_value and imputed_value as separate fields, always.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import pandas as pd

from staleness_pipeline.reconstruction import Reconstruction


class ImputationConfidence(str, Enum):
    """Set by the caller based on how the reconstruction was produced —
    never assumed just because Chronos was involved. See project notes:
    confidence should be earned (validated via synthetic_injection.py),
    not assumed."""

    PROVISIONAL = "PROVISIONAL"  # online, forward-only, run may still be active
    RECONCILED = "RECONCILED"    # offline, bidirectional, run has resolved


@dataclass
class ImputedMeasurement:
    point_id: str
    sensor: str                   # e.g. "aht_temperature" — distinguishes records that
                                   # share a point_id (e.g. equipment ID used as a CSV-testing
                                   # stand-in for a real per-attribute Point.id)
    timestamp: datetime
    raw_value: float | None       # None if there's no real reading at all for this timestamp
    imputed_value: float
    imputation_method: str        # e.g. "chronos-bolt-small-bidirectional"
    imputation_confidence: ImputationConfidence
    imputation_model_version: str | None = None

    def to_json_dict(self) -> dict:
        d = asdict(self)
        d["timestamp"] = self.timestamp.astimezone(timezone.utc).isoformat()
        d["imputation_confidence"] = self.imputation_confidence.value
        return d


def reconstruction_to_measurements(
    reconstruction: Reconstruction,
    point_id: str,
    confidence: ImputationConfidence,
    raw_values: pd.Series | None = None,
    model_version: str | None = None,
) -> list[ImputedMeasurement]:
    """Converts a Reconstruction (reconstruction.py's output) into sink-ready
    records. This is the seam between "we computed a reconstruction" and
    "we're about to write it somewhere" — reconstruction.py never needs to
    know about ImputedMeasurement, and storage.py never needs to know
    about forecasting.

    Args:
        reconstruction: output of reconstruct_stale_window().
        point_id: the Overgrid point this belongs to (see schema_1.gqls —
            Point.id is what identifies a sensor there).
        confidence: PROVISIONAL or RECONCILED — the caller's job to decide,
            based on which pipeline (online/offline) produced this.
        raw_values: the original (stuck/repeated) readings at these
            timestamps, if you want them preserved for reference. None
            means raw_value will be recorded as None for every point.
        model_version: optional, for traceability.
    """
    measurements = []
    for ts, imputed_value in reconstruction.values.items():
        raw_value = float(raw_values.loc[ts]) if raw_values is not None and ts in raw_values.index else None
        measurements.append(
            ImputedMeasurement(
                point_id=point_id,
                sensor=reconstruction.sensor,
                timestamp=ts,
                raw_value=raw_value,
                imputed_value=float(imputed_value),
                imputation_method=reconstruction.method,
                imputation_confidence=confidence,
                imputation_model_version=model_version,
            )
        )
    return measurements


class MeasurementSink(ABC):
    """Write path for reconstructed measurements. Implementations must be
    safe to call repeatedly for the same (point_id, timestamp) — an offline
    RECONCILED record superseding an earlier PROVISIONAL one is expected
    behavior, not an error."""

    @abstractmethod
    def write(self, measurements: list[ImputedMeasurement]) -> None: ...


class LocalJSONLSink(MeasurementSink):
    """Appends newline-delimited JSON to disk. No upsert-by-timestamp logic
    yet — every write appends, and a downstream reader is responsible for
    taking the latest record per (point_id, timestamp) if it cares. That's
    an acceptable simplification for a local file; a real sink (including
    GraphQLSink, eventually) would enforce this at write time instead."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, measurements: list[ImputedMeasurement]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            for m in measurements:
                f.write(json.dumps(m.to_json_dict()) + "\n")


class GraphQLSink(MeasurementSink):
    """Stub. See module docstring — schema_1.gqls doesn't currently support
    this safely. Deliberately not built yet."""

    def __init__(self, endpoint: str, token: str):
        self.endpoint = endpoint
        self.token = token

    def write(self, measurements: list[ImputedMeasurement]) -> None:
        raise NotImplementedError(
            "GraphQLSink is a stub — schema_1.gqls has no imputation-aware "
            "write path yet. Use LocalJSONLSink until that's resolved."
        )


def get_sink(backend: str, local_path: str, endpoint: str | None = None, token: str | None = None) -> MeasurementSink:
    """Factory keyed off a config value, so callers never instantiate a
    concrete sink class directly — one less thing to change when the
    eventual swap happens."""
    if backend == "local_jsonl":
        return LocalJSONLSink(local_path)
    if backend == "graphql":
        if not endpoint or not token:
            raise ValueError("GraphQLSink requires endpoint and token")
        return GraphQLSink(endpoint, token)
    raise ValueError(f"Unknown sink backend: {backend!r}")