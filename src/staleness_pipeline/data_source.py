"""Loads sensor data for the pipeline to work on.

Right now this reads from the wide-format CSV already exported from
Overgrid. It's deliberately a stand-in for a live GraphQL client against
schema_1.gqls (Point.series(startDate, endDate, every, fn) -> [Measurement]):
same output shape — one pd.Series per sensor, indexed by real UTC
timestamps, named after the clean attribute — so swapping to live calls
later only means changing this file, not detection.py or reconstruction.py.

Two things about the real schema worth remembering here:
  - Measurement.timestamp is a String, not a native datetime — needs
    parsing either way (done below for the CSV; the live client will need
    the same pd.to_datetime() step).
  - Measurement.value is nullable. The CSV can have the same thing (a
    blank cell -> NaN after pd.read_csv). detection.py's ffill_limit
    handles isolated nulls; longer real gaps are a separate concern this
    module doesn't try to solve.
"""

from __future__ import annotations

import pandas as pd


def load_series_from_csv(
    file_path: str,
    column: str,
    time_column: str = "timestamp",
) -> pd.Series:
    """Load one sensor's readings from a wide-format CSV.

    Args:
        file_path: path to the CSV (one row per timestamp, one column per
            sensor).
        column: exact column name to load, e.g.
            "ecbc3d63b0e4__Air_Temperature_Sensor__aht_temperature".
        time_column: name of the timestamp column.

    Returns:
        A pd.Series indexed by UTC timestamp, sorted chronologically,
        named after the cleaned-up attribute (e.g. "aht_temperature"
        instead of the full column label) — matches what Point.attribute
        gives us from the live API.
    """
    df = pd.read_csv(file_path)

    if column not in df.columns:
        available = ", ".join(c for c in df.columns if c != time_column)
        raise ValueError(f"Column {column!r} not found in {file_path}. Available columns: {available}")

    df[time_column] = pd.to_datetime(df[time_column], utc=True)
    df = df.sort_values(time_column)

    clean_name = column.split("__")[-1]  # e.g. "aht_temperature"
    return pd.Series(
        df[column].to_numpy(),
        index=pd.DatetimeIndex(df[time_column]),
        name=clean_name,
    )


def find_matching_columns(
    file_path: str,
    attribute_keywords: list[str],
    time_column: str = "timestamp",
) -> list[str]:
    """Find CSV columns whose attribute (the part after the last '__')
    contains any of the given keywords, case-insensitive.

    Example: find_matching_columns(path, ["temperature", "humidity"])
    matches both "..._aht_temperature" and "..._aht_humidity" columns
    without needing to know the exact equipment ID prefix ahead of time.
    """
    columns = pd.read_csv(file_path, nrows=0).columns
    matches = []
    for col in columns:
        if col == time_column:
            continue
        attribute = col.split("__")[-1].lower()
        if any(keyword.lower() in attribute for keyword in attribute_keywords):
            matches.append(col)
    return matches
