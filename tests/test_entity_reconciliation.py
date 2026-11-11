from test_evidence_loop import sample

from kyb.contracts.b4 import Entity, EvidenceItem, EvidenceRef
from kyb.loop.reconcile import reconcile


def test_names_do_not_promote_and_false_evidence_cannot_promote():
    facts, _ = sample()
    a = Entity(
        entity_id="a",
        kind="company",
        names=["Acme"],
        identifiers={"reg_no": ["123"]},
        jurisdiction=["US"],
        evidence=[EvidenceRef(claim_id="a1", doc="01", page=1, verified=False)],
    )
    b = Entity(
        entity_id="b",
        kind="company",
        names=["Acme"],
        identifiers={"reg_no": ["123"]},
        jurisdiction=["US"],
        evidence=[],
    )
    evidence = [
        EvidenceItem(
            evidence_id="a1",
            doc="01",
            document_name="a.pdf",
            page=1,
            bbox=(1, 1, 2, 2),
            excerpt="123",
            verified=True,
            field="entity[a].identifiers.reg_no",
        )
    ]
    candidate = reconcile(facts.model_copy(update={"entities": [a, b], "evidence": evidence}))[0]
    assert candidate.state == "unresolved"
    assert candidate.applied_to_factset is False


def _company(eid, country, identifier="123"):
    refs = [
        EvidenceRef(claim_id=f"{eid}-id", doc="01", page=1, verified=True),
        EvidenceRef(claim_id=f"{eid}-jur", doc="01", page=1, verified=True),
    ]
    entity = Entity(
        entity_id=eid,
        kind="company",
        names=["Acme"],
        identifiers={"reg_no": [identifier]},
        jurisdiction=[country],
        evidence=refs,
    )
    items = [
        EvidenceItem(
            evidence_id=f"{eid}-id",
            doc="01",
            document_name="a.pdf",
            page=1,
            bbox=(1, 1, 2, 2),
            excerpt=f"registration {identifier}",
            verified=True,
            field=f"entity[{eid}].identifiers.reg_no",
        ),
        EvidenceItem(
            evidence_id=f"{eid}-jur",
            doc="01",
            document_name="a.pdf",
            page=1,
            bbox=(1, 1, 2, 2),
            excerpt=country,
            verified=True,
            field=f"entity[{eid}].jurisdiction",
        ),
    ]
    return entity, items


def test_equal_identifiers_and_explicit_jurisdiction_and_cross_country():
    facts, _ = sample()
    a, ae = _company("a", "US")
    b, be = _company("b", "US")
    assert (
        reconcile(facts.model_copy(update={"entities": [a, b], "evidence": ae + be}))[0].state
        == "supported"
    )
    c, ce = _company("c", "MX")
    candidates = reconcile(
        facts.model_copy(update={"entities": [a, b, c], "evidence": ae + be + ce})
    )
    assert {c.state for c in candidates} == {"unresolved", "contradictory"}
    assert all(not c.applied_to_factset for c in candidates)


def test_identifier_substring_does_not_forge_support():
    facts, _ = sample()
    a, ae = _company("a", "US")
    b, be = _company("b", "US")
    ae[0] = ae[0].model_copy(update={"excerpt": "registration 1234"})
    assert (
        reconcile(facts.model_copy(update={"entities": [a, b], "evidence": ae + be}))[0].state
        == "unresolved"
    )
