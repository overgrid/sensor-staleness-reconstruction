import pandas as pd
import pytest
import torch

from staleness_pipeline.reconstruction import (
    blend_bidirectional,
    feather_edges,
    forecast_backward,
    forecast_forward,
    forecast_forward_chunked,
)


class FakePipeline:
    """Stands in for a real Chronos pipeline. Just repeats the last context
    value for every future step — we don't care about forecast quality
    here, only that forecast_forward() wires timestamps/shapes correctly."""

    def predict_quantiles(self, inputs, prediction_length, quantile_levels):
        last_value = inputs[-1].item()
        mean = torch.full((1, prediction_length), last_value)
        quantiles = torch.zeros((1, prediction_length, len(quantile_levels)))
        return quantiles, mean


def make_context(values, freq="10min", name="test_sensor"):
    index = pd.date_range("2026-01-01", periods=len(values), freq=freq, tz="UTC")
    return pd.Series(values, index=index, name=name)


def test_forecast_returns_correct_number_of_steps():
    context = make_context([20.0, 20.5, 21.0])
    result = forecast_forward(FakePipeline(), context, steps=5)
    assert len(result) == 5


def test_forecast_continues_at_the_same_cadence():
    context = make_context([20.0, 20.5, 21.0], freq="10min")
    result = forecast_forward(FakePipeline(), context, steps=3)

    # First predicted timestamp should be exactly one cadence after the
    # last context timestamp — not the same timestamp, not a gap.
    expected_first = context.index[-1] + pd.Timedelta(minutes=10)
    assert result.index[0] == expected_first
    # And each subsequent step should be another 10 minutes apart.
    assert (result.index[1] - result.index[0]) == pd.Timedelta(minutes=10)


def test_forecast_preserves_series_name():
    context = make_context([20.0, 20.5, 21.0], name="aht_temperature")
    result = forecast_forward(FakePipeline(), context, steps=2)
    assert result.name == "aht_temperature"


def test_zero_steps_returns_empty_series():
    context = make_context([20.0, 20.5, 21.0])
    result = forecast_forward(FakePipeline(), context, steps=0)
    assert len(result) == 0


def test_chunked_forecast_covers_gaps_longer_than_max_chunk_size():
    context = make_context([20.0, 20.5, 21.0])
    # 150 steps with a 64-step chunk limit means 3 calls: 64 + 64 + 22.
    result = forecast_forward_chunked(FakePipeline(), context, total_steps=150, max_chunk_size=64)
    assert len(result) == 150


def test_chunked_forecast_has_no_gap_or_overlap_at_chunk_boundaries():
    context = make_context([20.0, 20.5, 21.0], freq="10min")
    result = forecast_forward_chunked(FakePipeline(), context, total_steps=150, max_chunk_size=64)

    # Every consecutive pair of timestamps should be exactly one cadence
    # apart — including right across a chunk boundary (index 63 -> 64).
    diffs = result.index.to_series().diff().dropna()
    assert (diffs == pd.Timedelta(minutes=10)).all()


def test_chunked_forecast_matches_single_call_when_under_the_limit():
    context = make_context([20.0, 20.5, 21.0])
    chunked = forecast_forward_chunked(FakePipeline(), context, total_steps=10, max_chunk_size=64)
    single = forecast_forward(FakePipeline(), context, steps=10)
    assert list(chunked.index) == list(single.index)
    assert list(chunked.values) == list(single.values)


def test_chunked_forecast_zero_steps_returns_empty():
    context = make_context([20.0, 20.5, 21.0])
    result = forecast_forward_chunked(FakePipeline(), context, total_steps=0)
    assert len(result) == 0


def test_backward_forecast_ends_right_before_context_starts():
    # context_after starts at 2026-01-01 02:00; steps=5 at 10min cadence
    # should land exactly at 01:10, 01:20, ..., 01:50, 02:00 is NOT included.
    context_after = pd.Series(
        [30.0, 30.5, 31.0],
        index=pd.date_range("2026-01-01 02:00", periods=3, freq="10min", tz="UTC"),
        name="test_sensor",
    )
    result = forecast_backward(FakePipeline(), context_after, steps=5)

    assert len(result) == 5
    assert result.index[-1] == pd.Timestamp("2026-01-01 01:50", tz="UTC")
    assert result.index[0] == pd.Timestamp("2026-01-01 01:10", tz="UTC")
    # index should be strictly increasing (chronological), not reversed
    assert result.index.is_monotonic_increasing


def test_backward_forecast_value_comes_from_the_real_data_after_the_gap():
    # FakePipeline always repeats the last value it's given as context.
    # After reversing, the "last" value fed in is context_after's FIRST
    # real point (30.0) — so the backward forecast should be constant 30.0.
    context_after = pd.Series(
        [30.0, 31.0, 32.0],
        index=pd.date_range("2026-01-01 02:00", periods=3, freq="10min", tz="UTC"),
        name="test_sensor",
    )
    result = forecast_backward(FakePipeline(), context_after, steps=4)
    assert (result == 30.0).all()


def test_backward_forecast_handles_gaps_longer_than_chunk_size():
    context_after = pd.Series(
        [30.0, 31.0],
        index=pd.date_range("2026-01-01 12:00", periods=2, freq="10min", tz="UTC"),
        name="test_sensor",
    )
    result = forecast_backward(FakePipeline(), context_after, steps=100, max_chunk_size=64)
    assert len(result) == 100


def test_blend_favors_forward_at_the_start_and_backward_at_the_end():
    index = pd.date_range("2026-01-01", periods=5, freq="10min", tz="UTC")
    forward = pd.Series([20.0] * 5, index=index, name="test_sensor")
    backward = pd.Series([30.0] * 5, index=index, name="test_sensor")

    blended = blend_bidirectional(forward, backward)

    assert blended.iloc[0] == pytest.approx(20.0)   # fully forward at the start
    assert blended.iloc[-1] == pytest.approx(30.0)   # fully backward at the end
    assert blended.iloc[2] == pytest.approx(25.0)     # roughly halfway in between


def test_blend_raises_on_mismatched_lengths():
    index_a = pd.date_range("2026-01-01", periods=5, freq="10min", tz="UTC")
    index_b = pd.date_range("2026-01-01", periods=3, freq="10min", tz="UTC")
    forward = pd.Series([20.0] * 5, index=index_a)
    backward = pd.Series([30.0] * 3, index=index_b)

    with pytest.raises(ValueError):
        blend_bidirectional(forward, backward)


def test_feather_pulls_first_point_exactly_to_real_value_before():
    index = pd.date_range("2026-01-01", periods=10, freq="10min", tz="UTC")
    reconstructed = pd.Series([25.0] * 10, index=index)

    result = feather_edges(reconstructed, real_value_before=20.0, real_value_after=None, feather_points=3)

    assert result.iloc[0] == pytest.approx(20.0)


def test_feather_pulls_last_point_exactly_to_real_value_after():
    index = pd.date_range("2026-01-01", periods=10, freq="10min", tz="UTC")
    reconstructed = pd.Series([25.0] * 10, index=index)

    result = feather_edges(reconstructed, real_value_before=None, real_value_after=30.0, feather_points=3)

    assert result.iloc[-1] == pytest.approx(30.0)


def test_feather_leaves_the_middle_of_the_window_unchanged():
    index = pd.date_range("2026-01-01", periods=10, freq="10min", tz="UTC")
    reconstructed = pd.Series([25.0] * 10, index=index)

    result = feather_edges(reconstructed, real_value_before=20.0, real_value_after=30.0, feather_points=3)

    # Points far from either edge (index 4-5, well outside feather_points=3
    # from either end) should be untouched.
    assert result.iloc[4] == pytest.approx(25.0)
    assert result.iloc[5] == pytest.approx(25.0)


def test_feather_correction_tapers_off_moving_away_from_the_edge():
    index = pd.date_range("2026-01-01", periods=10, freq="10min", tz="UTC")
    reconstructed = pd.Series([25.0] * 10, index=index)

    result = feather_edges(reconstructed, real_value_before=20.0, real_value_after=None, feather_points=3)

    # The correction should shrink as we move away from the start —
    # each successive point should be closer to the original 25.0.
    correction_0 = abs(result.iloc[0] - 25.0)
    correction_1 = abs(result.iloc[1] - 25.0)
    correction_2 = abs(result.iloc[2] - 25.0)
    assert correction_0 > correction_1 > correction_2


def test_feather_with_no_real_values_returns_unchanged():
    index = pd.date_range("2026-01-01", periods=5, freq="10min", tz="UTC")
    reconstructed = pd.Series([25.0] * 5, index=index)

    result = feather_edges(reconstructed, real_value_before=None, real_value_after=None)

    assert list(result.values) == list(reconstructed.values)


@pytest.mark.slow
def test_forecast_with_real_chronos_model():
    """Not run by default (pytest -m "not slow" to skip, or just don't
    pass -m at all and it'll still run — see note below on marking this
    properly once we add pytest config). Downloads real model weights on
    first run. Run manually with:
        PYTHONPATH=src python -m pytest tests/test_reconstruction.py -v -k real_chronos
    """
    from staleness_pipeline.chronos_model import get_chronos_pipeline

    pipeline = get_chronos_pipeline()
    context = make_context([20.0, 20.2, 20.1, 20.3, 20.5, 20.4])
    result = forecast_forward(pipeline, context, steps=6)

    assert len(result) == 6
    # Sanity check, not a strict accuracy claim: predictions should be in
    # a plausible range near the context values, not wildly off.
    assert result.min() > 0
    assert result.max() < 100