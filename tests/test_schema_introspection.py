from staleness_pipeline.schema_introspection import (
    find_imputation_aware_fields,
    list_all_types,
)


class FakeIntrospectionClient:
    def __init__(self, types):
        self._types = types

    def execute(self, query, variable_values=None):
        return {"__schema": {"types": self._types}}


def make_overgrid_shaped_schema():
    """Mimics the real shape: Mutation only has one thin top-level field
    (projects), and the actual write operation is nested inside a
    different type entirely (WritablePoint) — exactly the structure that
    made the shallow, top-level-only version of this module miss
    everything."""
    return [
        {
            "name": "Mutation",
            "kind": "OBJECT",
            "description": None,
            "fields": [
                {"name": "projects", "description": None, "args": []},
            ],
        },
        {
            "name": "MutProject",
            "kind": "OBJECT",
            "description": None,
            "fields": [
                {"name": "equipment", "description": None, "args": []},
            ],
        },
        {
            "name": "WritablePoint",
            "kind": "OBJECT",
            "description": None,
            "fields": [
                {
                    "name": "write",
                    "description": "Write a raw value to this point.",
                    "args": [{"name": "value", "description": None}],
                },
            ],
        },
    ]


def test_list_all_types_returns_every_type_not_just_mutation():
    types = make_overgrid_shaped_schema()
    client = FakeIntrospectionClient(types)
    result = list_all_types(client)
    assert {t["name"] for t in result} == {"Mutation", "MutProject", "WritablePoint"}


def test_find_imputation_aware_fields_searches_nested_types_not_just_mutation_root():
    # Confirms the real schema (no imputation support at all currently)
    # correctly returns nothing, searched thoroughly rather than shallowly.
    types = make_overgrid_shaped_schema()
    client = FakeIntrospectionClient(types)
    matches = find_imputation_aware_fields(client)
    assert matches == []


def test_find_imputation_aware_fields_finds_a_match_nested_deep_in_the_schema():
    # If a future schema update adds imputation support on some nested
    # type (not Mutation's top level), this must still find it — this is
    # exactly the case the earlier, shallower version of this module
    # would have missed entirely.
    types = make_overgrid_shaped_schema()
    types.append(
        {
            "name": "WritablePoint",
            "kind": "OBJECT",
            "description": None,
            "fields": [
                {
                    "name": "recordImputedMeasurements",
                    "description": "Records reconstructed values with RECONCILED confidence.",
                    "args": [],
                }
            ],
        }
    )
    client = FakeIntrospectionClient(types)
    matches = find_imputation_aware_fields(client)
    assert len(matches) == 1
    assert matches[0]["field"] == "recordImputedMeasurements"


def test_find_imputation_aware_fields_matches_on_arg_names_too():
    types = [
        {
            "name": "WritablePoint",
            "kind": "OBJECT",
            "description": None,
            "fields": [
                {
                    "name": "write",
                    "description": None,
                    "args": [{"name": "imputationConfidence", "description": None}],
                }
            ],
        }
    ]
    client = FakeIntrospectionClient(types)
    matches = find_imputation_aware_fields(client)
    assert len(matches) == 1
    assert matches[0]["field"] == "write"