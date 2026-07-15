"""MLflow tracking, centralized.

Nothing else in the package should import mlflow directly. That keeps two
things possible:
  1. Consistent run naming/tags everywhere (git commit sha, sensor,
     gap length) instead of each script reinventing it slightly differently.
  2. Swapping in NoOpTracker for unit tests and dry runs, so pytest never
     needs a live MLflow server to pass.
"""

from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def _git_commit_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


class Tracker(ABC):
    @abstractmethod
    @contextmanager
    def run(self, run_name: str, tags: dict[str, Any] | None = None) -> Iterator[None]: ...

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None: ...

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float]) -> None: ...

    @abstractmethod
    def log_artifact(self, path: str | Path) -> None: ...


class MLflowTracker(Tracker):
    def __init__(self, tracking_uri: str, experiment_name: str):
        import mlflow  # imported lazily — this module shouldn't require mlflow to be installed just to import it

        self._mlflow = mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    @contextmanager
    def run(self, run_name: str, tags: dict[str, Any] | None = None) -> Iterator[None]:
        all_tags = {"git_commit": _git_commit_sha(), **(tags or {})}
        with self._mlflow.start_run(run_name=run_name, tags=all_tags):
            yield

    def log_params(self, params: dict[str, Any]) -> None:
        self._mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self._mlflow.log_metrics(metrics)

    def log_artifact(self, path: str | Path) -> None:
        self._mlflow.log_artifact(str(path))


class NoOpTracker(Tracker):
    """Used in tests and any dry-run mode. Records nothing, talks to nothing."""

    @contextmanager
    def run(self, run_name: str, tags: dict[str, Any] | None = None) -> Iterator[None]:
        yield

    def log_params(self, params: dict[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: dict[str, float]) -> None:
        pass

    def log_artifact(self, path: str | Path) -> None:
        pass


def get_tracker(tracking_uri: str, experiment_name: str, enabled: bool = True) -> Tracker:
    if not enabled:
        return NoOpTracker()
    return MLflowTracker(tracking_uri, experiment_name)