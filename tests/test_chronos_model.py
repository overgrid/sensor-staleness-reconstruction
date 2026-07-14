import sys
import types

from staleness_pipeline import chronos_model


class FakePipeline:
    def __init__(self, name):
        self.name = name


class FakeBaseChronosPipeline:
    """Stands in for chronos.BaseChronosPipeline. Counts how many times
    from_pretrained was actually called, so tests can prove caching works
    without downloading a real model."""

    load_count = 0

    @classmethod
    def from_pretrained(cls, model_name, device_map, torch_dtype):
        cls.load_count += 1
        return FakePipeline(model_name)


def install_fake_chronos_module(monkeypatch):
    """Injects a fake 'chronos' module into sys.modules, so
    `from chronos import BaseChronosPipeline` inside chronos_model.py picks
    up our fake instead of needing the real chronos-forecasting package
    installed."""
    fake_module = types.ModuleType("chronos")
    fake_module.BaseChronosPipeline = FakeBaseChronosPipeline
    monkeypatch.setitem(sys.modules, "chronos", fake_module)


def test_same_model_and_device_loads_only_once(monkeypatch):
    chronos_model.clear_pipeline_cache()
    FakeBaseChronosPipeline.load_count = 0
    install_fake_chronos_module(monkeypatch)
    monkeypatch.setattr(chronos_model, "_is_model_cached", lambda name: True)

    first = chronos_model.get_chronos_pipeline("fake-model", "cpu")
    second = chronos_model.get_chronos_pipeline("fake-model", "cpu")

    assert first is second
    assert FakeBaseChronosPipeline.load_count == 1


def test_different_model_names_load_separately(monkeypatch):
    chronos_model.clear_pipeline_cache()
    FakeBaseChronosPipeline.load_count = 0
    install_fake_chronos_module(monkeypatch)
    monkeypatch.setattr(chronos_model, "_is_model_cached", lambda name: True)

    chronos_model.get_chronos_pipeline("model-a", "cpu")
    chronos_model.get_chronos_pipeline("model-b", "cpu")

    assert FakeBaseChronosPipeline.load_count == 2


def test_clear_pipeline_cache_forces_a_reload(monkeypatch):
    chronos_model.clear_pipeline_cache()
    FakeBaseChronosPipeline.load_count = 0
    install_fake_chronos_module(monkeypatch)
    monkeypatch.setattr(chronos_model, "_is_model_cached", lambda name: True)

    chronos_model.get_chronos_pipeline("fake-model", "cpu")
    chronos_model.clear_pipeline_cache()
    chronos_model.get_chronos_pipeline("fake-model", "cpu")

    assert FakeBaseChronosPipeline.load_count == 2
