"""Synthetic gap injection + accuracy scoring.

Real stuck windows have no ground truth — we never know what the sensor
"should have" read. The only way to measure accuracy is to pick a stretch
of data we DO know, pretend it's a stuck window, reconstruct it blind, and
compare against what was actually there.

Key simplification: reconstruct_stale_window() only ever reads context
strictly BEFORE a period's start_time and strictly AFTER its end_time —
never the real values inside the period. So faking a gap doesn't require
hiding or freezing anything in the series; we just point a StuckPeriod at
a real stretch and reconstruct it as if we didn't already know the answer.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

from staleness_pipeline.detection import StuckPeriod
from staleness_pipeline.reconstruction import reconstruct_stale_window


def inject_synthetic_gap(
    series: pd.Series,
    gap_length_points: int,
    min_context_points: int = 10,
    rng_seed: int | None = None,
    max_attempts: int = 50,
) -> tuple[StuckPeriod, pd.Series]:
    """Pick a random real stretch of `gap_length_points` and pretend it's a
    stuck window that needs reconstructing.

    Args:
        series: real sensor data, indexed by timestamp. May contain nulls
            elsewhere (real Overgrid data does) — but the chosen window and
            its immediate boundary values must be null-free, or every
            downstream metric silently becomes NaN (one missing point out
            of 200 poisons the whole trial's MAE/RMSE/MAPE, for Chronos AND
            both baselines — this isn't specific to the model).
        gap_length_points: how many points the fake gap should span.
        min_context_points: how much real data must exist on each side —
            reconstruction needs real context to work with, and a fair
            test shouldn't pick a window right at the very edge of the
            series where there's barely any.
        rng_seed: for reproducible trials (score_reconstruction across
            trials needs to compare the same fake gaps run to run).
        max_attempts: how many random windows to try before giving up if
            the data is too full of nulls to find a clean one.

    Returns:
        (synthetic_period, hidden_true_values) — the StuckPeriod to feed
        into reconstruct_stale_window(), and the real values it covers,
        to compare the reconstruction against afterward.
    """
    n = len(series)
    if n < gap_length_points + 2 * min_context_points:
        raise ValueError(
            f"Series too short ({n} points) for a {gap_length_points}-point gap "
            f"with {min_context_points} points of context required on each side."
        )

    rng = random.Random(rng_seed)
    earliest_start = min_context_points
    latest_start = n - gap_length_points - min_context_points

    for _ in range(max_attempts):
        start_idx = rng.randint(earliest_start, latest_start)
        end_idx = start_idx + gap_length_points - 1

        hidden_true_values = series.iloc[start_idx : end_idx + 1]
        boundary_before = series.iloc[start_idx - 1]
        boundary_after = series.iloc[end_idx + 1] if end_idx + 1 < n else None

        if hidden_true_values.isna().any():
            continue
        if pd.isna(boundary_before):
            continue
        if boundary_after is not None and pd.isna(boundary_after):
            continue

        duration = series.index[end_idx] - series.index[start_idx]
        period = StuckPeriod(
            sensor=str(series.name),
            stuck_value=float("nan"),  # meaningless here — this isn't a real stuck run
            start_time=series.index[start_idx],
            end_time=series.index[end_idx],
            duration_hours=round(duration.total_seconds() / 3600, 2),
        )
        return period, hidden_true_values

    raise ValueError(
        f"Could not find a {gap_length_points}-point window without nulls after "
        f"{max_attempts} attempts — this sensor's data may have too many missing "
        "readings for validation at this gap length."
    )


def compute_naive_baselines(
    series: pd.Series, period: StuckPeriod, gap_index: pd.DatetimeIndex
) -> tuple[pd.Series, pd.Series]:
    """The two baselines Chronos actually needs to beat — forward-fill
    (repeat the last real value) and linear interpolation (straight line
    between the real value before and after the gap)."""
    context_before = series.loc[series.index < period.start_time]
    context_after = series.loc[series.index > period.end_time]

    last_before = context_before.iloc[-1]
    first_after = context_after.iloc[0] if not context_after.empty else last_before

    forward_fill = pd.Series([last_before] * len(gap_index), index=gap_index, name=series.name)

    # linspace across n+2 points, then drop the two real endpoints —
    # we only want the values strictly inside the gap.
    linear_values = np.linspace(last_before, first_after, len(gap_index) + 2)[1:-1]
    linear_interp = pd.Series(linear_values, index=gap_index, name=series.name)

    return forward_fill, linear_interp


@dataclass
class GapTrialResult:
    gap_length_points: int
    trial_index: int
    chronos_mae: float
    chronos_rmse: float
    chronos_mape: float
    forward_fill_mae: float
    forward_fill_rmse: float
    forward_fill_mape: float
    linear_interp_mae: float
    linear_interp_rmse: float
    linear_interp_mape: float
    beats_forward_fill: bool
    beats_linear_interp: bool


def _mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))


def _rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def _mape(pred: np.ndarray, true: np.ndarray) -> float:
    nonzero = true != 0
    if not nonzero.any():
        return float("nan")
    return float(np.mean(np.abs((pred[nonzero] - true[nonzero]) / true[nonzero])) * 100)


def score_reconstruction(
    reconstructed: pd.Series,
    hidden_true_values: pd.Series,
    forward_fill: pd.Series,
    linear_interp: pd.Series,
    gap_length_points: int,
    trial_index: int,
) -> GapTrialResult:
    """Compare a reconstruction against the hidden ground truth AND both
    naive baselines. beats_forward_fill/beats_linear_interp are the real
    bar — a reconstruction that's merely "not wrong" but loses to linear
    interpolation isn't earning RECONCILED confidence."""
    true_arr = hidden_true_values.to_numpy()

    chronos_mae = _mae(reconstructed.to_numpy(), true_arr)
    ff_mae = _mae(forward_fill.to_numpy(), true_arr)
    li_mae = _mae(linear_interp.to_numpy(), true_arr)

    return GapTrialResult(
        gap_length_points=gap_length_points,
        trial_index=trial_index,
        chronos_mae=chronos_mae,
        chronos_rmse=_rmse(reconstructed.to_numpy(), true_arr),
        chronos_mape=_mape(reconstructed.to_numpy(), true_arr),
        forward_fill_mae=ff_mae,
        forward_fill_rmse=_rmse(forward_fill.to_numpy(), true_arr),
        forward_fill_mape=_mape(forward_fill.to_numpy(), true_arr),
        linear_interp_mae=li_mae,
        linear_interp_rmse=_rmse(linear_interp.to_numpy(), true_arr),
        linear_interp_mape=_mape(linear_interp.to_numpy(), true_arr),
        beats_forward_fill=chronos_mae < ff_mae,
        beats_linear_interp=chronos_mae < li_mae,
    )


def run_gap_trial(
    pipeline,
    series: pd.Series,
    gap_length_points: int,
    trial_index: int,
    rng_seed: int | None = None,
    min_context_points: int = 10,
) -> GapTrialResult:
    """One full trial, end to end: fake a gap, reconstruct it, score it
    against both baselines. This is the single function the eventual
    offline validation job will call in a loop across gap lengths/trials."""
    period, hidden_true_values = inject_synthetic_gap(
        series, gap_length_points, min_context_points, rng_seed
    )
    reconstruction = reconstruct_stale_window(pipeline, series, period)
    forward_fill, linear_interp = compute_naive_baselines(series, period, reconstruction.values.index)

    return score_reconstruction(
        reconstruction.values, hidden_true_values, forward_fill, linear_interp, gap_length_points, trial_index
    )