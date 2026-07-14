"""Chronos-based reconstruction of stuck windows.

STAGE 1 (this version): forward-only forecasting for a single short gap.
No chunking (works up to Chronos' ~64-step recommended horizon), no
backward pass, no blending. We'll add those next, once this piece is
proven to work.

Chronos itself has no idea about timestamps or real-world time — it just
sees a sequence of numbers ("context") and extrapolates more numbers. All
the timestamp/frequency bookkeeping below is our responsibility, not the
model's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def forecast_forward(pipeline, context: pd.Series, steps: int) -> pd.Series:
    """Predict `steps` future points continuing on from `context`.

    Args:
        pipeline: a loaded Chronos pipeline — get one from
            staleness_pipeline.chronos_model.get_chronos_pipeline(), which
            handles caching/downloading. Any object exposing the same
            predict_quantiles(inputs=..., prediction_length=...,
            quantile_levels=...) interface works too — tests pass in a
            fake one so they don't need real model weights.
        context: real values immediately before the gap, indexed by
            timestamp, evenly spaced. Only the values matter to Chronos;
            we use the index just to figure out timestamps for the output.
        steps: how many future points to predict. Keep this <= ~64 for
            now — chunking longer gaps is the next stage.

    Returns:
        A Series of predicted values, indexed by the timestamps that
        continue on from `context` at the same cadence.
    """
    if steps <= 0:
        return pd.Series([], dtype=float, name=context.name)

    context_tensor = torch.tensor(context.to_numpy(), dtype=torch.float32)

    _, mean = pipeline.predict_quantiles(
        inputs=context_tensor,
        prediction_length=steps,
        quantile_levels=[0.1, 0.5, 0.9],
    )
    predicted_values = mean[0].numpy()  # mean shape is [batch=1, steps]; we only ever pass one series

    # Continue at the same cadence as the context. Using the gap between
    # the last two context points, rather than assuming a fixed "10
    # minutes", keeps this correct for any sensor's reporting interval.
    cadence = context.index[-1] - context.index[-2]
    future_index = pd.date_range(start=context.index[-1] + cadence, periods=steps, freq=cadence)

    return pd.Series(predicted_values, index=future_index, name=context.name)


def forecast_forward_chunked(
    pipeline,
    context: pd.Series,
    total_steps: int,
    max_chunk_size: int = 64,
    max_context_length: int = 512,
) -> pd.Series:
    """Predict `total_steps` future points, for gaps longer than Chronos'
    single-call limit.

    Forecasts one chunk of at most `max_chunk_size` steps, appends that
    chunk's own predictions onto the context, then forecasts the next
    chunk — repeating until `total_steps` is covered.

    Worth understanding: from the second chunk onward, part of the
    "context" is our own earlier prediction, not real data. Error can
    compound the further into a long gap this goes — later chunks are
    inherently less trustworthy than the first one. This function doesn't
    try to correct for that; synthetic_injection.py's validation is what
    tells us how much it actually matters at the gap lengths we see.

    Args:
        pipeline: a loaded Chronos pipeline.
        context: real values immediately before the gap.
        total_steps: total number of points to reconstruct — can be
            larger than max_chunk_size.
        max_chunk_size: steps per individual Chronos call. Defaults to
            Chronos' recommended ~64.
        max_context_length: caps how much history gets fed into each
            call. Without this, context grows every chunk and slows down
            later calls for no accuracy benefit — Chronos is zero-shot and
            doesn't reliably improve with arbitrarily long context.

    Returns:
        A single Series covering all `total_steps`, indexed continuously
        from the end of `context`.
    """
    if total_steps <= 0:
        return pd.Series([], dtype=float, name=context.name)

    running_context = context
    chunks: list[pd.Series] = []
    remaining = total_steps

    while remaining > 0:
        chunk_size = min(remaining, max_chunk_size)

        # Trim context so it doesn't grow unbounded across many chunks.
        trimmed_context = running_context.iloc[-max_context_length:]
        chunk = forecast_forward(pipeline, trimmed_context, chunk_size)

        chunks.append(chunk)
        running_context = pd.concat([running_context, chunk])
        remaining -= chunk_size

    return pd.concat(chunks)


def forecast_backward(
    pipeline,
    context_after: pd.Series,
    steps: int,
    max_chunk_size: int = 64,
    max_context_length: int = 512,
) -> pd.Series:
    """Predict `steps` points immediately BEFORE context_after, by
    forecasting backward in time — used when a stuck window has real data
    on both sides, which is the common case (see project notes: a run
    only ends when the value changes, so most stuck runs aren't open-ended).

    Args:
        context_after: real values immediately after the gap, in normal
            chronological order. Needs at least 2 points to establish
            cadence.
        steps: how many points before context_after to predict.

    Returns:
        A Series of predicted values ending exactly one cadence-interval
        before context_after begins, in normal chronological order.
    """
    if steps <= 0:
        return pd.Series([], dtype=float, name=context_after.name)

    cadence = context_after.index[1] - context_after.index[0]

    # Reversing a numpy array creates a "negative stride" view. Torch
    # tensors require positive strides, so handing this straight to
    # torch.tensor() crashes with a confusing error — ascontiguousarray()
    # forces a real, positively-strided copy first.
    reversed_values = np.ascontiguousarray(context_after.to_numpy()[::-1])

    # forecast_forward_chunked needs a Series with a real, evenly-spaced
    # DatetimeIndex to work out cadence and build output timestamps — but
    # for this reversed sequence the actual calendar dates are meaningless
    # (we throw this index away below and compute the real one separately).
    # Any fixed start date works, as long as the spacing matches the real cadence.
    placeholder_index = pd.date_range("2000-01-01", periods=len(reversed_values), freq=cadence)
    reversed_context = pd.Series(reversed_values, index=placeholder_index, name=context_after.name)

    reversed_forecast = forecast_forward_chunked(
        pipeline, reversed_context, steps, max_chunk_size, max_context_length
    )

    # reversed_forecast[0] corresponds to the real-time point immediately
    # before context_after starts; reversed_forecast[-1] is the earliest
    # predicted point. Reverse back into normal chronological order.
    chronological_values = reversed_forecast.to_numpy()[::-1]

    real_index = pd.date_range(end=context_after.index[0] - cadence, periods=steps, freq=cadence)
    return pd.Series(chronological_values, index=real_index, name=context_after.name)


def blend_bidirectional(forward: pd.Series, backward: pd.Series) -> pd.Series:
    """Blend a forward forecast and a backward forecast covering the same
    gap. Trusts the forward pass more near the start of the gap (closer to
    its own real context, before the gap) and the backward pass more near
    the end (closer to its real context, after the gap) — linearly in
    between, rather than switching abruptly at the midpoint.

    Args:
        forward: output of forecast_forward/forecast_forward_chunked,
            covering the gap starting right after the pre-gap context.
        backward: output of forecast_backward, covering the same gap,
            ending right before the post-gap context.

    Returns:
        A single Series over the same index, blending both.
    """
    if len(forward) != len(backward):
        raise ValueError(f"forward ({len(forward)} points) and backward ({len(backward)} points) must be the same length")
    if len(forward) > 0 and not forward.index.equals(backward.index):
        raise ValueError("forward and backward forecasts must cover the exact same timestamps")

    n = len(forward)
    if n == 0:
        return forward

    # 1.0 (fully forward) at the first point, down to 0.0 (fully backward)
    # at the last point.
    forward_weight = np.linspace(1.0, 0.0, n)
    blended_values = forward.to_numpy() * forward_weight + backward.to_numpy() * (1 - forward_weight)

    return pd.Series(blended_values, index=forward.index, name=forward.name)