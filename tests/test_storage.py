import json

import pandas as pd
import pytest

from staleness_pipeline.reconstruction import Reconstruction
from staleness_pipeline.storage import (
    GraphQLSink,
    ImputationConfidence,
    LocalJSONLSink,
    get_sink,
    reconstruction_to_measurements,
)


def make_reconstruction():
    index = pd.date_range("2026-01-01", periods=3, freq="10min", tz="UTC")
    values = pd.Series([21.0, 21.2, 21.4], index=index, name="aht_temperature")
    return Reconstruction(
        sensor="aht_temperature",
        start_time=index[0],
        end_time=index[-1],
        values=values,
        method="chronos-bolt-small-bidirectional",
    )


def test_reconstruction_to_measurements_basic_shape():
    reconstruction = make_reconstruction()
    measurements = reconstruction_to_measurements(
        reconstruction, point_id="pt-123", confidence=ImputationConfidence.RECONCILED
    )

    assert len(measurements) == 3
    assert measurements[0].point_id == "pt-123"
    assert measurements[0].sensor == "aht_temperature"
    assert measurements[0].imputation_confidence == ImputationConfidence.RECONCILED
    assert measurements[0].imputation_method == "chronos-bolt-small-bidirectional"
    assert measurements[0].raw_value is None


def test_records_from_different_sensors_sharing_a_point_id_stay_distinguishable():
    # This is exactly the scenario that surfaced the bug: temperature and
    # humidity reconstructed with the same point_id (an equipment ID
    # standing in for a real per-attribute Point.id during CSV testing).
    index = pd.date_range("2026-01-01", periods=2, freq="10min", tz="UTC")
    temp_reconstruction = Reconstruction(
        sensor="aht_temperature",
        start_time=index[0],
        end_time=index[-1],
        values=pd.Series([21.0, 21.2], index=index),
        method="chronos-bolt-small-bidirectional",
    )
    humidity_reconstruction = Reconstruction(
        sensor="aht_humidity",
        start_time=index[0],
        end_time=index[-1],
        values=pd.Series([55.0, 55.5], index=index),
        method="chronos-bolt-small-bidirectional",
    )

    temp_measurements = reconstruction_to_measurements(
        temp_reconstruction, point_id="ecbc3d63b0e4", confidence=ImputationConfidence.RECONCILED
    )
    humidity_measurements = reconstruction_to_measurements(
        humidity_reconstruction, point_id="ecbc3d63b0e4", confidence=ImputationConfidence.RECONCILED
    )

    # Same point_id, same timestamps — sensor is the only thing that
    # distinguishes them, and it must actually do so.
    assert temp_measurements[0].point_id == humidity_measurements[0].point_id
    assert temp_measurements[0].timestamp == humidity_measurements[0].timestamp
    assert temp_measurements[0].sensor != humidity_measurements[0].sensor
    assert temp_measurements[0].sensor == "aht_temperature"
    assert humidity_measurements[0].sensor == "aht_humidity"


def test_reconstruction_to_measurements_includes_raw_values_when_given():
    reconstruction = make_reconstruction()
    raw_values = pd.Series([21.0, 21.0, 21.0], index=reconstruction.values.index)

    measurements = reconstruction_to_measurements(
        reconstruction, point_id="pt-123", confidence=ImputationConfidence.RECONCILED, raw_values=raw_values
    )

    assert all(m.raw_value == 21.0 for m in measurements)
    # imputed_value should be untouched, still the reconstructed value
    assert measurements[1].imputed_value == pytest.approx(21.2)


def test_local_jsonl_sink_writes_one_line_per_record(tmp_path):
    reconstruction = make_reconstruction()
    measurements = reconstruction_to_measurements(reconstruction, "pt-123", ImputationConfidence.RECONCILED)

    sink = LocalJSONLSink(tmp_path / "out.jsonl")
    sink.write(measurements)

    lines = (tmp_path / "out.jsonl").read_text().splitlines()
    assert len(lines) == 3
    record = json.loads(lines[0])
    assert record["imputed_value"] == pytest.approx(21.0)
    assert record["imputation_confidence"] == "RECONCILED"


def test_local_jsonl_sink_appends_across_calls(tmp_path):
    reconstruction = make_reconstruction()
    measurements = reconstruction_to_measurements(reconstruction, "pt-123", ImputationConfidence.RECONCILED)

    sink = LocalJSONLSink(tmp_path / "out.jsonl")
    sink.write(measurements)
    sink.write(measurements)
    assert len((tmp_path / "out.jsonl").read_text().splitlines()) == 6


def test_graphql_sink_raises_not_implemented():
    sink = GraphQLSink(endpoint="https://legacy.overgrid.eu/graphql", token="fake")
    with pytest.raises(NotImplementedError):
        sink.write([])


def test_get_sink_factory_local(tmp_path):
    sink = get_sink("local_jsonl", str(tmp_path / "out.jsonl"))
    assert isinstance(sink, LocalJSONLSink)


def test_get_sink_factory_unknown_backend_raises():
    with pytest.raises(ValueError):
        get_sink("not_a_real_backend", "unused")