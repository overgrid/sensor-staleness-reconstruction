import pandas as pd
import pytest
import torch

from staleness_pipeline.reconstruction import forecast_forward


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
