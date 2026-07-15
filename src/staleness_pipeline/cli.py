"""Command-line entry point.

Installed as the `staleness` command (see pyproject.toml [project.scripts]).
This is the one thing you, cron, or eventually a CI runner needs to know
how to call — never Python paths or module internals.
"""

from __future__ import annotations

import logging

import typer

from staleness_pipeline.offline_job import run_offline_job

app = typer.Typer(help="Staleness detection & reconstruction pipeline.")


@app.callback()
def _callback() -> None:
    """Empty on purpose. Without this, Typer collapses into a
    single-command CLI when there's only one @app.command() registered,
    dropping the subcommand name — `staleness --column ...` instead of
    `staleness offline --column ...`. This keeps `offline` (and future
    commands like `validate`/`online`) explicit and consistent."""


@app.command()
def offline(
    csv_path: str = typer.Option("data/ecbc3d63b0e4_last_30_days_mean_ecbc3d63b0e4_wide.csv", help="Path to the wide-format CSV."),
    column: str = typer.Option(
        ..., help="Exact CSV column to process, e.g. ecbc3d63b0e4__Air_Temperature_Sensor__aht_temperature"
    ),
    point_id: str = typer.Option(..., help="Overgrid point_id this sensor corresponds to."),
    min_stuck_hours: float = typer.Option(0.25, help="Minimum duration to flag as stuck."),
    sink_path: str = typer.Option("data/reconstructed_measurements.jsonl"),
    mlflow_uri: str = typer.Option("http://localhost:5000"),
    mlflow_experiment: str = typer.Option("chronos-staleness-reconstruction"),
    skip_mlflow: bool = typer.Option(False, help="Disable MLflow logging (e.g. no server running)."),
    skip_validation: bool = typer.Option(False, help="Skip synthetic-gap validation, just reconstruct."),
) -> None:
    """Run the nightly offline job: detect, validate, reconstruct, write."""
    logging.basicConfig(level=logging.INFO)
    count = run_offline_job(
        csv_path=csv_path,
        column=column,
        point_id=point_id,
        min_stuck_hours=min_stuck_hours,
        local_sink_path=sink_path,
        mlflow_tracking_uri=mlflow_uri,
        mlflow_experiment_name=mlflow_experiment,
        mlflow_enabled=not skip_mlflow,
        run_validation_first=not skip_validation,
    )
    typer.echo(f"Done — wrote {count} reconstructed measurements to {sink_path}")


if __name__ == "__main__":
    app()