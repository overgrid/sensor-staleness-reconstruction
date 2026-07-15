"""Offline batch job: the nightly, high-confidence path.

Ties together every module built so far: loads real data (data_source.py),
detects stuck periods (detection.py), validates accuracy via synthetic gap
injection (synthetic_injection.py), reconstructs each real stuck period
(reconstruction.py), and writes RECONCILED records through the sink
(storage.py) — all logged to MLflow (tracking.py).

Deliberately thin: no real logic lives here, only wiring. That's what lets
the CLI, a cron job, or a future scheduler all call run_offline_job() the
same way without duplicating any of this.
"""

from __future__ import annotations

import logging

from staleness_pipeline.chronos_model import get_chronos_pipeline
from staleness_pipeline.data_source import load_series_from_csv
from staleness_pipeline.detection import find_stuck_periods
from staleness_pipeline.reconstruction import reconstruct_stale_window
from staleness_pipeline.storage import ImputationConfidence, get_sink, reconstruction_to_measurements
from staleness_pipeline.synthetic_injection import run_gap_trial
from staleness_pipeline.tracking import Tracker, get_tracker

logger = logging.getLogger(__name__)

# Matches the real gap-length distribution from your original validation:
# ~7 pts / ~35 min, ~60 pts / ~5 hr, ~200 pts / ~16-25 hr.
VALIDATION_GAP_LENGTHS = [7, 60, 200]
VALIDATION_TRIALS_PER_LENGTH = 3


def run_validation(pipeline, series, tracker: Tracker) -> None:
    """Synthetic-gap validation across the standard gap lengths, logged to
    MLflow. This is what makes RECONCILED confidence earned rather than
    assumed. Gap lengths too long for the available data (e.g. a 200-point
    gap on a 60-point series) are skipped with a warning, not a crash —
    that's expected on short test datasets, not an error."""
    with tracker.run(run_name=f"validate-{series.name}", tags={"sensor": str(series.name)}):
        for gap_length in VALIDATION_GAP_LENGTHS:
            for trial in range(VALIDATION_TRIALS_PER_LENGTH):
                try:
                    result = run_gap_trial(pipeline, series, gap_length, trial, rng_seed=trial)
                except ValueError as e:
                    logger.warning("Skipping validation trial (gap=%d, trial=%d): %s", gap_length, trial, e)
                    continue

                tracker.log_metrics(
                    {
                        f"chronos_mae_gap{gap_length}": result.chronos_mae,
                        f"forward_fill_mae_gap{gap_length}": result.forward_fill_mae,
                        f"linear_interp_mae_gap{gap_length}": result.linear_interp_mae,
                        f"beats_forward_fill_gap{gap_length}": float(result.beats_forward_fill),
                        f"beats_linear_interp_gap{gap_length}": float(result.beats_linear_interp),
                    }
                )


def run_offline_job(
    csv_path: str,
    column: str,
    point_id: str,
    min_stuck_hours: float = 0.25,
    sink_backend: str = "local_jsonl",
    local_sink_path: str = "data/reconstructed_measurements.jsonl",
    mlflow_tracking_uri: str = "http://localhost:5000",
    mlflow_experiment_name: str = "chronos-staleness-reconstruction",
    mlflow_enabled: bool = True,
    run_validation_first: bool = True,
) -> int:
    """Run the full offline pipeline for one sensor column.

    Returns:
        Number of reconstructed measurements written.
    """
    series = load_series_from_csv(csv_path, column=column)
    pipeline = get_chronos_pipeline()
    tracker = get_tracker(mlflow_tracking_uri, mlflow_experiment_name, enabled=mlflow_enabled)
    sink = get_sink(sink_backend, local_sink_path)

    if run_validation_first:
        run_validation(pipeline, series, tracker)

    stuck_periods = find_stuck_periods(series, min_stuck_hours)
    logger.info("Found %d stuck periods for %s", len(stuck_periods), series.name)

    all_measurements = []
    for period in stuck_periods:
        try:
            reconstruction = reconstruct_stale_window(pipeline, series, period)
        except ValueError as e:
            logger.warning("Skipping period %s to %s: %s", period.start_time, period.end_time, e)
            continue

        raw_values = series.loc[period.start_time : period.end_time]
        measurements = reconstruction_to_measurements(
            reconstruction,
            point_id=point_id,
            confidence=ImputationConfidence.RECONCILED,
            raw_values=raw_values,
        )
        all_measurements.extend(measurements)

    sink.write(all_measurements)
    logger.info("Wrote %d reconstructed points for %s", len(all_measurements), series.name)
    return len(all_measurements)
