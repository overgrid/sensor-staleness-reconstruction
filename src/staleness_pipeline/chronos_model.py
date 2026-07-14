"""Loads and caches the Chronos forecasting model.

This is deliberately its own module, separate from reconstruction.py:
both the offline job and the online consumer need the *same* loaded
model, and neither should pay the cost of re-checking the Hugging Face
cache or reloading weights into memory every time they need to forecast.
Call get_chronos_pipeline() from wherever you need it — the first caller
in the process loads it (downloading first if the cache is empty); every
call after that, anywhere in the app, gets the same object back instantly.
"""

from __future__ import annotations

import logging
import threading

import torch

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "amazon/chronos-bolt-small"
DEFAULT_DEVICE = "cpu"

# Keyed by (model_name, device) so asking for a different model or device
# loads a separate copy, rather than silently reusing the wrong one.
_pipeline_cache: dict[tuple[str, str], object] = {}
_cache_lock = threading.Lock()


def _is_model_cached(model_name: str) -> bool:
    """Check the local Hugging Face cache without downloading anything.
    Used only to decide what to log — from_pretrained() below will do the
    real cache check/download itself either way."""
    try:
        from huggingface_hub import scan_cache_dir

        cache_info = scan_cache_dir()
        return any(repo.repo_id == model_name for repo in cache_info.repos)
    except Exception:
        # If the cache scan itself fails for any reason, we don't want
        # that to block loading the model — just log less precisely.
        return False


def get_chronos_pipeline(model_name: str = DEFAULT_MODEL_NAME, device: str = DEFAULT_DEVICE):
    """Return a loaded Chronos pipeline, downloading weights only if they
    aren't already cached, and reusing the loaded object across calls.

    Safe to call from multiple threads/places in the app — only the first
    caller for a given (model_name, device) actually loads anything.
    """
    key = (model_name, device)

    if key in _pipeline_cache:
        return _pipeline_cache[key]

    with _cache_lock:
        # Another thread may have finished loading while we were waiting
        # for the lock — check again before doing real work.
        if key in _pipeline_cache:
            return _pipeline_cache[key]

        from chronos import BaseChronosPipeline

        if _is_model_cached(model_name):
            logger.info("Loading %s from local cache (no download needed)", model_name)
        else:
            logger.info("%s not cached locally — downloading now (first run only, ~100MB)", model_name)

        pipeline = BaseChronosPipeline.from_pretrained(
            model_name, device_map=device, torch_dtype=torch.float32
        )

        _pipeline_cache[key] = pipeline
        logger.info("%s ready on device=%s", model_name, device)
        return pipeline


def clear_pipeline_cache() -> None:
    """Drop the in-memory cache of loaded pipeline objects. Does NOT touch
    the on-disk Hugging Face cache — the weights stay downloaded. Mainly
    useful for tests, or if you deliberately want to force a reload."""
    with _cache_lock:
        _pipeline_cache.clear()


if __name__ == "__main__":
    # Quick manual check: `python chronos_model.py` loads the model and
    # confirms it worked, without needing pytest or the rest of the app.
    logging.basicConfig(level=logging.INFO)
    pipeline = get_chronos_pipeline()
    print(f"Loaded pipeline: {pipeline}")
