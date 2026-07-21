"""Checks what the LIVE Overgrid GraphQL schema actually supports, rather
than trusting a static snapshot (schema_1.gqls) that may be out of date.

This exists specifically to answer one question before touching
storage.py further: does the schema now have any imputation-aware field
or mutation (something like recordImputedMeasurements, isImputed,
imputedValue, imputationConfidence)? If yes, GraphQLSink can become real.
If no — confirmed via this tool, not assumed from an old file — it stays
a stub, honestly.

Important: Overgrid's Mutation type is a thin root — Mutation.projects(id)
returns a mutable object, and the real write operations (WritablePoint's
write()) are nested several levels deep inside that (MutProject ->
equipment -> points -> WritablePoint.write()). A shallow check of only
top-level Mutation fields would completely miss this — an earlier version
of this module made exactly that mistake. This version searches every
type in the schema, not just the Mutation root.
"""

from __future__ import annotations

from gql import gql

from staleness_pipeline.graphql_source import execute_with_retry

# Standard GraphQL introspection query, trimmed to just what we need:
# every type's name/kind, and every field's name/description/args. We
# don't need the full `ofType` wrapping chain (that's for building/calling
# a query correctly) -- just enough to search names and descriptions.
FULL_INTROSPECTION_QUERY = gql(
    """
    query IntrospectionQuery {
        __schema {
            types {
                name
                kind
                description
                fields {
                    name
                    description
                    args {
                        name
                        description
                    }
                }
            }
        }
    }
    """
)

# Terms that would indicate imputation-aware support if they showed up
# anywhere in a type/field/arg name or description — case-insensitive
# substring match, deliberately broad so nothing gets missed by guessing
# one exact name.
IMPUTATION_KEYWORDS = [
    "imput", "reconcil", "provisional", "stale", "confidence", "reconstruct",
]


def list_all_types(client) -> list[dict]:
    """Every type in the schema, with its fields and their args. Read-only
    — this never calls any mutation, just describes what exists."""
    result = execute_with_retry(client, FULL_INTROSPECTION_QUERY, {})
    return result["__schema"]["types"]


def find_imputation_aware_fields(client) -> list[dict]:
    """Search every field (and its args) across the WHOLE schema — every
    type, not just Mutation's top level — for imputation-related keywords.

    Returns a list of {"type": ..., "field": ..., "description": ...} for
    every match. Empty list means: as of this check, nothing in the schema
    supports writing or representing imputed values distinctly from real
    ones, anywhere.
    """
    matches = []
    for t in list_all_types(client):
        for f in t.get("fields") or []:
            haystack = f"{t['name']} {f['name']} {f.get('description') or ''}".lower()
            arg_haystack = " ".join(
                f"{a['name']} {a.get('description') or ''}" for a in (f.get("args") or [])
            ).lower()
            if any(k in haystack or k in arg_haystack for k in IMPUTATION_KEYWORDS):
                matches.append({"type": t["name"], "field": f["name"], "description": f.get("description")})
    return matches