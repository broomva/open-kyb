"""Adjudication controls beside the immutable original matrix, not replacement labels."""

import copy
import dataclasses

import pytest

from kyb.contracts.b4 import FactSet
from kyb.evals import loop_eval
from kyb.facts.derive import DocInput, fact_set_hash
from kyb.loop.models import LoopBudget
from kyb.loop.planner import plan_cycle
from kyb.util.hashing import sha256_json


def _fixture():
    return next(f for f in loop_eval.FIXTURES if f.case_id == "V02-fresh-run-closes-gap")


def _current(variant):
    data = copy.deepcopy(_fixture().after)
    evidence = next(e for e in data["evidence"] if e["evidence_id"] == "c-t1-juris")
    if variant != "original":
        evidence["field"] = "entity[e-t1].jurisdiction"
        evidence["excerpt"] = "Covered Holder S.A. is incorporated in Mexico."
    if variant == "wrong_country":
        evidence["excerpt"] = "Covered Holder S.A. is incorporated in Belize."
    if variant == "wrong_entity":
        evidence["field"] = "entity[e-app].jurisdiction"
    if variant == "unverified":
        evidence["verified"] = False
    if variant == "wrong_document":
        evidence["doc"] = "01"
    facts = FactSet.model_validate(data)
    facts = facts.model_copy(update={"fact_set_hash": fact_set_hash(facts)})
    claim = {
        "claim_id": "c-t1-juris",
        "type": "identity",
        "subject": "Covered Holder S.A.",
        "attribute": "jurisdiction",
        "value": "Mexico",
    }
    docs = [
        loop_eval._doc_input("01"),
        DocInput(
            doc="02",
            document_name="02_declaracion_jurada_ubo.pdf",
            claims=[claim],
            verdicts={"c-t1-juris": True},
            assurance="W2",
            isolated=True,
        ),
    ]
    return facts, docs


def _prior():
    return plan_cycle(
        FactSet.model_validate(_fixture().facts),
        [loop_eval._doc_input("01"), loop_eval._doc_input("02")],
        budget=LoopBudget(),
    )


def test_original_unqualified_evidence_still_does_not_close_the_gap():
    facts, docs = _current("original")
    plan = plan_cycle(facts, docs, budget=LoopBudget(), previous=_prior())
    assert plan.state == "no_progress"
    assert plan.metrics["verified_assertion_delta"] == 0
    assert any(g.kind == "jurisdiction" and "e-t1" in g.entity_ids for g in plan.gaps)


def test_field_bound_jurisdiction_and_its_claim_close_the_same_gap():
    """A valid control must exercise progress instead of only testing refusal."""
    facts, docs = _current("valid")
    before = sha256_json(facts.model_dump(mode="json"))
    plan = plan_cycle(facts, docs, budget=LoopBudget(), previous=_prior())
    assert sha256_json(facts.model_dump(mode="json")) == before
    assert plan.state == "ready"
    assert plan.metrics["verified_assertion_delta"] == 1
    assert not any(g.kind == "jurisdiction" and "e-t1" in g.entity_ids for g in plan.gaps)


@pytest.mark.parametrize(
    "variant", ["wrong_country", "wrong_entity", "unverified", "wrong_document"]
)
def test_nearby_invalid_evidence_never_clears_jurisdiction(variant):
    facts, docs = _current(variant)
    plan = plan_cycle(facts, docs, budget=LoopBudget(), previous=_prior())
    assert plan.state != "ready"
    assert any(g.kind == "jurisdiction" and "e-t1" in g.entity_ids for g in plan.gaps)


def test_adjudication_does_not_mutate_original_fixtures_or_expectations():
    before = sha256_json([dataclasses.asdict(f) for f in loop_eval.FIXTURES])
    for variant in [
        "original",
        "valid",
        "wrong_country",
        "wrong_entity",
        "unverified",
        "wrong_document",
    ]:
        facts, docs = _current(variant)
        plan_cycle(facts, docs, budget=LoopBudget(), previous=_prior())
    assert sha256_json([dataclasses.asdict(f) for f in loop_eval.FIXTURES]) == before
