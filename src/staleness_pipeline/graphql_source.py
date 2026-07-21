"""Live data fetching from Overgrid's GraphQL API.

This is the live counterpart to data_source.py's CSV loader — both
produce the same shape (pandas.Series, real UTC timestamps, clean
attribute name), so callers can swap one for the other without touching
detection.py or reconstruction.py.

Ported from an existing, already-validated reference script
(dataset_builder.py), trimmed to just what this pipeline needs: no CSV
writing, no multi-point wide-pivot table — just series per point, ready
to feed straight into detection/reconstruction.

One thing this fixes for free, worth calling out explicitly: CLI usage
against the CSV export has been passing an equipment_id as --point-id,
since the CSV doesn't carry real per-attribute Point IDs. Live fetching
pivots on point.id — the only field the schema guarantees is unique — so
each PointData below carries the REAL point_id for its sensor.

Credentials: pass the bearer token via the OVERGRID_TOKEN environment
variable — this module never hardcodes it or loads a specific .env path,
since that path is machine-specific (e.g. /etc/overgrid/.env on host241).
Set it up however fits your shell/environment before running.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd
from gql import Client, gql
from gql.transport.exceptions import TransportQueryError
from gql.transport.requests import RequestsHTTPTransport

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "https://legacy.overgrid.eu/graphql"

# NOTE: equipment(flat: true, ...) has been observed to fail intermittently
# with a server-side INTERNAL_ERROR. Confirmed NOT deterministic: the exact
# same query with no code changes failed 3 times, then succeeded on a later
# retry -- and flat:true genuinely matters (it surfaces equipment nested
# under locations that a non-flat query misses entirely: 13 vs 10 devices
# on one real project). So this stays in, with retry logic below handling
# the flakiness, rather than being removed and silently under-reporting
# real equipment.
QUERY = gql(
    """
    query Projects(
        $alias: String
        $equipmentId: String
        $attribute: String
        $startDate: String!
        $endDate: String!
        $every: String!
        $fn: String!
    ) {
        projects(alias: $alias) {
            id
            alias
            equipment(flat: true, id: $equipmentId) {
                id
                type
                points(attribute: $attribute) {
                    id
                    type
                    attribute
                    series(
                        startDate: $startDate
                        endDate: $endDate
                        every: $every
                        fn: $fn
                    ) {
                        timestamp
                        value
                    }
                }
            }
        }
    }
    """
)


@dataclass
class PointData:
    """One sensor's data plus the metadata needed to identify it later.

    point_id is the REAL, schema-guaranteed-unique identifier — this is
    what should be used for storage.py's point_id going forward, instead
    of the equipment_id stand-in used during CSV-only testing.
    """

    point_id: str
    equipment_id: str
    equipment_type: str
    point_type: str
    attribute: str
    series: pd.Series  # indexed by UTC timestamp, named after `attribute`


def build_client(endpoint: str = DEFAULT_ENDPOINT, token: str | None = None) -> Client:
    """Create a GraphQL client. token defaults to the OVERGRID_TOKEN
    environment variable if not passed explicitly — never hardcode it."""
    token = token or os.environ.get("OVERGRID_TOKEN")
    if not token:
        raise ValueError(
            "No Overgrid token provided — pass token= explicitly or set the "
            "OVERGRID_TOKEN environment variable."
        )
    transport = RequestsHTTPTransport(
        url=endpoint,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        verify=True,
    )
    return Client(transport=transport, fetch_schema_from_transport=False)


def execute_with_retry(
    client: Client,
    query,
    variables: dict,
    max_retries: int = 3,
    backoff_seconds: float = 2.0,
):
    """Execute a GraphQL query, retrying on TransportQueryError.

    equipment(flat: true, ...) has been observed to fail intermittently
    with a server-side INTERNAL_ERROR — the same query with no changes at
    all has both failed and succeeded across separate runs. Retrying with
    a short backoff is the honest response to a flaky dependency, rather
    than avoiding whatever argument happened to be involved in one failed
    attempt.
    """
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return client.execute(query, variable_values=variables)
        except TransportQueryError as e:
            last_error = e
            if attempt < max_retries:
                logger.warning(
                    "GraphQL query failed (attempt %d/%d): %s — retrying in %.0fs",
                    attempt, max_retries, e, backoff_seconds,
                )
                time.sleep(backoff_seconds)
    raise last_error


def fetch_points(
    client: Client,
    alias: str,
    equipment_id: str | None,
    attribute: str | None,
    start_date: datetime,
    end_date: datetime,
    every: str = "10m",
    fn: str = "mean",
) -> list[PointData]:
    """Fetch one or more points' series data in a single request.

    Args:
        client: from build_client() — or any object exposing the same
            execute(query, variable_values=...) interface; tests pass in
            a fake one so they never make a real network call.
        alias: project alias (e.g. "MyHome").
        equipment_id: filter to one piece of equipment, or None for all.
        attribute: comma-separated attribute names (e.g.
            "aht_temperature,aht_humidity") to pull several sensors in one
            request, or None for every attribute.
        start_date / end_date: real datetimes — converted to the string
            format the API expects internally.
        every / fn: resampling interval and aggregation function, passed
            straight through to Point.series(...).

    Returns:
        One PointData per point that had any series data at all — points
        with an empty series are silently skipped (real Overgrid data can
        include points with no readings in a given window; that's
        expected, not an error).
    """
    variables = {
        "alias": alias,
        "equipmentId": equipment_id,
        "attribute": attribute,
        "startDate": start_date.strftime("%Y-%m-%dT%H:%M:%S"),
        "endDate": end_date.strftime("%Y-%m-%dT%H:%M:%S"),
        "every": every,
        "fn": fn,
    }
    result = execute_with_retry(client, QUERY, variables)

    points_data: list[PointData] = []
    for project in result["projects"]:
        for equip in project["equipment"]:
            for point in equip["points"]:
                raw_series = point["series"]
                if not raw_series:
                    continue

                timestamps = pd.to_datetime([s["timestamp"] for s in raw_series], utc=True)
                # Measurement.value is nullable on the real schema — None
                # becomes NaN here, same as data_source.py's CSV path.
                values = [s["value"] for s in raw_series]
                series = (
                    pd.Series(values, index=timestamps, name=point["attribute"], dtype=float)
                    .sort_index()
                )

                points_data.append(
                    PointData(
                        point_id=point["id"],
                        equipment_id=equip["id"],
                        equipment_type=equip["type"],
                        point_type=point["type"],
                        attribute=point["attribute"],
                        series=series,
                    )
                )
    return points_data


def fetch_recent_points(
    client: Client,
    alias: str,
    equipment_id: str | None,
    attribute: str | None,
    days_back: int = 30,
    every: str = "10m",
    fn: str = "mean",
) -> list[PointData]:
    """Convenience wrapper: fetch the last `days_back` days, computed from now."""
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days_back)
    return fetch_points(client, alias, equipment_id, attribute, start_dt, end_dt, every, fn)