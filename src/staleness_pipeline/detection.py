"""Stuck-period detection.

Finds runs where a sensor value repeats exactly for longer than a
threshold. Adapted from the original find_stuck_periods_with_plot() —
same core grouping trick (change/cumsum), but split so detection and
plotting are two separate functions. That split matters: an automated job
can call find_stuck_periods() every night with nothing watching the
screen, but it can never call plt.show(). Plotting only makes sense when a
human is looking, so it stays a separate, optional function.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class StuckPeriod:
    sensor: str
    stuck_value: float
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    duration_hours: float


def find_stuck_periods(
    series: pd.Series,
    min_stuck_hours: float = 0.25,
    ffill_limit: int = 1,
) -> list[StuckPeriod]:
    """Detect runs of exactly-repeated values longer than min_stuck_hours.

    Args:
        series: sensor readings, indexed by timestamp (a DatetimeIndex).
            Must already be sorted chronologically — call
            series.sort_index() first if you're not sure. May contain
            NaN/null values (real APIs return nulls for missing readings).
        min_stuck_hours: minimum run duration to flag as stuck.
        ffill_limit: real data can have isolated missing readings (a null
            instead of a repeated value). Without handling this, a single
            null in the middle of an otherwise-stuck run splits it into two
            shorter runs that might each fall under min_stuck_hours and go
            undetected entirely — the same underlying problem as a one-off
            outlier reading interrupting a stuck run. Forward-filling small
            gaps (default: 1 point) before grouping fixes this for isolated
            nulls, without pretending we have real data for longer missing
            stretches. Set to 0 to disable.

    Returns:
        List of StuckPeriod, in chronological order. Empty list if nothing
        was stuck long enough to flag.
    """
    if series.empty:
        return []

    # Grouping is computed on a forward-filled copy (so isolated nulls
    # don't break up a real stuck run), but stuck_value/reporting below
    # still reads from the ORIGINAL series — we're only using the filled
    # version to decide where group boundaries fall.
    grouping_series = series.ffill(limit=ffill_limit) if ffill_limit > 0 else series

    # Same trick as the original script: `changed` is True at every point
    # the value differs from the one before it. cumsum() over that turns
    # each run of identical values into its own group id.
    changed = grouping_series.ne(grouping_series.shift())
    group_id = changed.cumsum()

    periods: list[StuckPeriod] = []
    for _, group in series.groupby(group_id):
        if len(group) <= 1:
            continue  # a single reading isn't a "run" of anything

        start_time = group.index[0]
        end_time = group.index[-1]
        duration = end_time - start_time

        if duration >= pd.to_timedelta(min_stuck_hours, unit="h"):
            periods.append(
                StuckPeriod(
                    sensor=str(series.name),
                    stuck_value=group.iloc[0],
                    start_time=start_time,
                    end_time=end_time,
                    duration_hours=round(duration.total_seconds() / 3600, 2),
                )
            )

    return periods


def stuck_periods_to_dataframe(periods: list[StuckPeriod]) -> pd.DataFrame:
    """Convert to a DataFrame for display in a notebook — the same
    human-readable table the original script printed. Not used by the
    automated job, only for interactive inspection."""
    if not periods:
        return pd.DataFrame(
            columns=["Sensor/Metric", "Stuck Value", "Start Time", "End Time", "Duration (Hours)"]
        )
    return pd.DataFrame(
        [
            {
                "Sensor/Metric": p.sensor,
                "Stuck Value": p.stuck_value,
                "Start Time": p.start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "End Time": p.end_time.strftime("%Y-%m-%d %H:%M:%S"),
                "Duration (Hours)": p.duration_hours,
            }
            for p in periods
        ]
    )


def plot_stuck_periods(series: pd.Series, periods: list[StuckPeriod]):
    """Chart the raw series with stuck windows shaded red. Interactive use
    only (notebook, manual debugging) — never called from the automated
    offline job. Imports matplotlib lazily so the rest of the package
    doesn't require it just to run detection."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 6))
    plt.plot(series.index, series.values, label="Sensor value", color="#1f77b4", linewidth=1.5)

    for p in periods:
        plt.axvspan(p.start_time, p.end_time, color="red", alpha=0.3, label="Stuck period")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.title(f"Sensor data & detected stuck periods ({series.name})", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    return plt.gcf()