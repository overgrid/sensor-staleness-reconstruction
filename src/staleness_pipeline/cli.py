"""Command-line entry point.

Installed as the `staleness` command (see pyproject.toml [project.scripts]).
This is the one thing you, cron, or eventually a CI runner needs to know
how to call — never Python paths or module internals.
"""

from __future__ import annotations

import logging

import typer

from staleness_pipeline.offline_job import run_offline_job, run_offline_job_live

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
    csv_path: str = typer.Option("data/hum_temp_wide.csv", help="Path to the wide-format CSV."),
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


@app.command(name="offline-live")
def offline_live(
    alias: str = typer.Option(..., help="Overgrid project alias, e.g. 'MyHome'."),
    equipment_id: str = typer.Option(None, help="Filter to one piece of equipment, or omit for all."),
    attribute: str = typer.Option(
        ..., help="Comma-separated attribute(s), e.g. 'aht_temperature,aht_humidity'."
    ),
    days_back: int = typer.Option(30, help="How many days of history to fetch."),
    every: str = typer.Option("10m", help="Resampling interval, passed to Point.series()."),
    fn: str = typer.Option("mean", help="Aggregation function, passed to Point.series()."),
    min_stuck_hours: float = typer.Option(0.25, help="Minimum duration to flag as stuck."),
    sink_path: str = typer.Option("data/reconstructed_measurements.jsonl"),
    mlflow_uri: str = typer.Option("http://localhost:5000"),
    mlflow_experiment: str = typer.Option("chronos-staleness-reconstruction"),
    skip_mlflow: bool = typer.Option(False, help="Disable MLflow logging (e.g. no server running)."),
    skip_validation: bool = typer.Option(False, help="Skip synthetic-gap validation, just reconstruct."),
) -> None:
    """Run the offline job against LIVE Overgrid data via GraphQL, instead
    of a CSV export. Real point_ids are discovered automatically from the
    API — no --point-id needed. Requires OVERGRID_TOKEN in the environment."""
    logging.basicConfig(level=logging.INFO)
    results = run_offline_job_live(
        alias=alias,
        equipment_id=equipment_id,
        attribute=attribute,
        days_back=days_back,
        every=every,
        fn=fn,
        min_stuck_hours=min_stuck_hours,
        local_sink_path=sink_path,
        mlflow_tracking_uri=mlflow_uri,
        mlflow_experiment_name=mlflow_experiment,
        mlflow_enabled=not skip_mlflow,
        run_validation_first=not skip_validation,
    )

    if not results:
        typer.echo("No points found with data for the given filters.")
        return

    for point_id, count in results.items():
        typer.echo(f"  {point_id}: wrote {count} reconstructed measurements")
    typer.echo(
        f"Done — processed {len(results)} point(s), "
        f"{sum(results.values())} total measurements written to {sink_path}"
    )


if __name__ == "__main__":
    app()