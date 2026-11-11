"""Coverage uses verified typed components without relaxing the witness or field identity."""

import copy

import pytest

from kyb.witness.coverage import coverage

CLAIM = {
    "claim_id": "c1",
    "type": "owns",
    "attribute": "shareholding",
    "block_id": "b1",
    "subject": "Valeria Ortega Méndez",
    "value": {"pct": 25, "shares": 1250},
}
FACT = {
    "block_id": "b1",
    "kind": "ownership",
    "subject": "Valeria Ortega Méndez",
    "attribute": "%",
    "value": "25%",
}


def test_structured_holding_covers_each_typed_column():
    count = {**FACT, "attribute": "acciones", "value": "1.250"}
    assert [x["status"] for x in coverage([FACT, count], [CLAIM])] == ["matched", "matched"]
    # The caller supplies only verified claims. An unverified/absent claim covers nothing.
    assert coverage([FACT], [])[0]["status"] == "uncovered"


@pytest.mark.parametrize("key,value,text", [("pct", 25, "25%"), ("shares", 1250, "1.250 acciones")])
def test_single_field_typed_values_keep_their_component(key, value, text):
    claim = {**CLAIM, "attribute": key, "value": {key: value}}
    fact = {**FACT, "attribute": key, "value": text}
    assert coverage([fact], [claim])[0]["status"] == "matched"


@pytest.mark.parametrize(
    "change,status",
    [
        ({"block_id": "b2"}, "uncovered"),
        ({"subject": "Lucas Navarro Solís"}, "uncovered"),
        ({"attribute": "par_value", "value": "25"}, "uncovered"),
        ({"attribute": "unknown", "value": "25"}, "uncovered"),
        ({"attribute": "Acciones Serie B", "value": "1.250"}, "uncovered"),
        ({"value": "30%"}, "conflict"),
        ({"attribute": "shares", "value": "25"}, "conflict"),
    ],
)
def test_component_matching_preserves_identity_and_units(change, status):
    assert coverage([{**FACT, **change}], [CLAIM])[0]["status"] == status


def test_corroborating_component_does_not_hide_a_conflicting_claim():
    bad = copy.deepcopy(CLAIM)
    bad["claim_id"], bad["value"]["pct"] = "c2", 30
    result = coverage([FACT], [CLAIM, bad])
    assert result[0]["status"] == "matched"
    assert result[1]["status"] == "conflict" and result[1]["claims"] == ["c2"]


def test_same_number_in_another_component_does_not_cover_a_percentage():
    assert coverage([FACT], [{**CLAIM, "value": {"shares": 25}}])[0]["status"] == "uncovered"


def test_distinct_date_attributes_are_not_interchangeable():
    claim = {**CLAIM, "attribute": "issue_date", "value": "2026-08-19"}
    fact = {**FACT, "attribute": "date_of_birth", "value": "2026-08-19"}
    assert coverage([fact], [claim])[0]["status"] == "uncovered"


@pytest.mark.parametrize("series", ["A", "B"])
@pytest.mark.parametrize("layout_value", ["25 acciones Serie A", "25 acciones", "25"])
def test_series_qualified_components_are_not_projected_to_unqualified_counts(series, layout_value):
    claim = {**CLAIM, "value": {"shares": 25, "series": series}}
    fact = {**FACT, "attribute": "shares", "value": layout_value}
    assert coverage([fact], [claim])[0]["status"] == "uncovered"


@pytest.mark.parametrize("layout_value", ["25 acciones Serie A", "25 acciones Serie B", "25 A"])
def test_a_plain_count_does_not_establish_a_qualified_layout_count(layout_value):
    claim = {**CLAIM, "value": {"shares": 25}}
    fact = {**FACT, "attribute": "shares", "value": layout_value}
    assert coverage([fact], [claim])[0]["status"] == "uncovered"


def test_review_series_mismatch_can_bind_but_cannot_cover_another_series():
    from kyb.witness.bind import binding

    claim = {**CLAIM, "value": {"shares": 25, "series": "B"}}
    reread = {
        "values": [
            {
                "subject": claim["subject"],
                "attribute": "shareholding",
                "value": "25 acciones Serie B",
            }
        ]
    }
    assert binding(claim, reread, "Valeria Ortega Méndez 25 acciones Serie B")
    fact = {**FACT, "attribute": "shares", "value": "25 acciones Serie A"}
    assert coverage([fact], [claim])[0]["status"] == "uncovered"
