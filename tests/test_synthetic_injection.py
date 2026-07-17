import numpy as np
import pandas as pd
import pytest
import torch

from staleness_pipeline.synthetic_injection import (
    compute_naive_baselines,
    inject_synthetic_gap,
    run_gap_trial,
    score_reconstruction,
)


class FakePipeline:
    """Repeats the last context value for every future step — same fake
    used in test_reconstruction.py."""

    def predict_quantiles(self, inputs, prediction_length, quantile_levels):
        last_value = inputs[-1].item()
        mean = torch.full((1, prediction_length), last_value)
        quantiles = torch.zeros((1, prediction_length, len(quantile_levels)))
        return quantiles, mean


def make_series(values, freq="10min", name="test_sensor"):
    index = pd.date_range("2026-01-01", periods=len(values), freq=freq, tz="UTC")
    return pd.Series(values, index=index, name=name)


def test_inject_synthetic_gap_returns_correct_length():
    series = make_series([float(i) for i in range(50)])
    period, hidden = inject_synthetic_gap(series, gap_length_points=5, min_context_points=10, rng_seed=1)
    assert len(hidden) == 5
    assert period.start_time == hidden.index[0]
    assert period.end_time == hidden.index[-1]


def test_inject_synthetic_gap_leaves_context_on_both_sides():
    series = make_series([float(i) for i in range(50)])
    period, hidden = inject_synthetic_gap(series, gap_length_points=5, min_context_points=10, rng_seed=1)

    context_before = series.loc[series.index < period.start_time]
    context_after = series.loc[series.index > period.end_time]
    assert len(context_before) >= 10
    assert len(context_after) >= 10


def test_inject_synthetic_gap_raises_when_series_too_short():
    series = make_series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        inject_synthetic_gap(series, gap_length_points=5, min_context_points=10)


def test_inject_synthetic_gap_is_reproducible_with_same_seed():
    series = make_series([float(i) for i in range(50)])
    period_a, _ = inject_synthetic_gap(series, gap_length_points=5, rng_seed=42)
    period_b, _ = inject_synthetic_gap(series, gap_length_points=5, rng_seed=42)
    assert period_a.start_time == period_b.start_time


def test_inject_synthetic_gap_never_returns_a_window_containing_nulls():
    # A null sits right where the first several random draws would likely
    # land (deterministic given rng_seed=1) — proves the retry logic
    # actually skips bad windows instead of returning one with a NaN in it.
    values = [float(i) for i in range(50)]
    values[25] = float("nan")
    series = make_series(values)

    for seed in range(20):  # try several seeds, not just one lucky draw
        period, hidden = inject_synthetic_gap(series, gap_length_points=5, min_context_points=10, rng_seed=seed)
        assert not hidden.isna().any()


def test_inject_synthetic_gap_avoids_null_boundary_values_too():
    # A null sits just outside where a gap would land, but exactly where a
    # baseline's boundary value (last real point before the gap) would be
    # read from — this must ALSO be avoided, or forward_fill/linear_interp
    # baselines end up NaN even though the gap itself looks clean.
    values = [float(i) for i in range(50)]
    values[19] = float("nan")  # would be context_before.iloc[-1] for a gap starting at 20
    series = make_series(values)

    for seed in range(20):
        period, hidden = inject_synthetic_gap(series, gap_length_points=5, min_context_points=10, rng_seed=seed)
        boundary_before = series.loc[series.index < period.start_time].iloc[-1]
        assert not pd.isna(boundary_before)


def test_inject_synthetic_gap_raises_clear_error_when_no_clean_window_exists():
    # Nulls everywhere — no window of this length can possibly be clean.
    values = [float(i) if i % 2 == 0 else float("nan") for i in range(50)]
    series = make_series(values)

    with pytest.raises(ValueError, match="null"):
        inject_synthetic_gap(series, gap_length_points=5, min_context_points=10, rng_seed=1, max_attempts=10)


def test_forward_fill_baseline_repeats_the_last_real_value():
    series = make_series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    period, hidden = inject_synthetic_gap(series, gap_length_points=2, min_context_points=2, rng_seed=0)
    forward_fill, _ = compute_naive_baselines(series, period, hidden.index)

    context_before = series.loc[series.index < period.start_time]
    assert (forward_fill == context_before.iloc[-1]).all()


def test_linear_interp_baseline_is_monotonic_between_endpoints():
    series = make_series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
    period, hidden = inject_synthetic_gap(series, gap_length_points=3, min_context_points=2, rng_seed=0)
    _, linear_interp = compute_naive_baselines(series, period, hidden.index)

    values = linear_interp.to_numpy()
    assert all(values[i] < values[i + 1] for i in range(len(values) - 1))


def test_score_reconstruction_computes_lower_error_for_the_better_method():
    index = pd.date_range("2026-01-01", periods=3, freq="10min", tz="UTC")
    hidden_true = pd.Series([10.0, 10.0, 10.0], index=index)
    good_reconstruction = pd.Series([10.0, 10.0, 10.0], index=index)  # perfect
    forward_fill = pd.Series([5.0, 5.0, 5.0], index=index)  # way off
    linear_interp = pd.Series([9.0, 9.0, 9.0], index=index)  # close but not perfect

    result = score_reconstruction(good_reconstruction, hidden_true, forward_fill, linear_interp, 3, 0)

    assert result.chronos_mae == pytest.approx(0.0)
    assert result.beats_forward_fill is True
    assert result.beats_linear_interp is True


def test_score_reconstruction_honestly_reports_a_loss_to_a_baseline():
    # A reconstruction that's WORSE than linear interpolation should
    # honestly report beats_linear_interp=False, not be forced positive.
    index = pd.date_range("2026-01-01", periods=3, freq="10min", tz="UTC")
    hidden_true = pd.Series([10.0, 20.0, 30.0], index=index)
    bad_reconstruction = pd.Series([10.0, 10.0, 10.0], index=index)  # flat, misses the trend
    forward_fill = pd.Series([5.0, 5.0, 5.0], index=index)
    linear_interp = pd.Series([10.0, 20.0, 30.0], index=index)  # perfectly tracks a linear trend

    result = score_reconstruction(bad_reconstruction, hidden_true, forward_fill, linear_interp, 3, 0)

    assert result.beats_linear_interp is False


def test_run_gap_trial_end_to_end_with_fake_pipeline():
    series = make_series([20.0 + (i % 3) * 0.1 for i in range(60)])
    result = run_gap_trial(FakePipeline(), series, gap_length_points=5, trial_index=0, rng_seed=1)

    assert result.gap_length_points == 5
    assert isinstance(result.chronos_mae, float)
    assert isinstance(result.beats_forward_fill, bool)