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
