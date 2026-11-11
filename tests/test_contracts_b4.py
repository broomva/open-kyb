"""The B4 contracts (spec) import, validate a minimal example, and refuse what they must."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from kyb.contracts import b4

REF = {"claim_id": "c-B03-02", "doc": "03", "page": 1, "verified": True}
UNKNOWN = {"state": "unknown"}


def _facts() -> b4.FactSet:
    return b4.FactSet(
        case="B", run="r1", as_of="2026-10-25", fact_set_hash="sha256:x", applicant="e-mch",
        entities=[b4.Entity(entity_id="e-mch", kind="company", names=["Iberia Comercial"],
                            jurisdiction=["MX"], incorporated=["2020-04-12"], evidence=[REF])],
        holdings=[], roles=[], documents=[], conflicts=[],
        operations=b4.Operations(distinct_operating_address=UNKNOWN,
                                 bank_reference_or_financials={"state": "absent"},
                                 employees_or_website=UNKNOWN),
        evidence=[b4.EvidenceItem(evidence_id="c-B03-02", doc="03", document_name="03_x.pdf",
                                  page=1, bbox=(1, 2, 3, 4), excerpt="Iberia Comercial",
                                  verified=True)],
        verification=[b4.DocVerification(doc="03", assurance="W2", isolated=True)],
    )  # fmt: skip


def _decision() -> dict:
    return {"decision_id": "d-1", "entity_id": "e-mch", "entity_name": "Iberia Comercial",
            "rules_version": "2026.11-r1", "status": "MANUAL_REVIEW", "risk_score": 70,
            "violations": [], "citations": [], "rationale": "r", "audit_trail": [],
            "emitted_at": datetime(2026, 10, 27, tzinfo=UTC)}  # fmt: skip


def test_a_minimal_flow_validates():
    facts = _facts()
    assert facts.contracts_version == b4.CONTRACTS_VERSION
    c = b4.citation("es-iberia-software", facts.evidence[0], "entity.name")
    assert c.document_id == "es-iberia-software/03_x.pdf" and c.locator.startswith("p.1 bbox")
    rr = b4.RuleResult(rule_id="JUR-02", severity="SCORE", fired="yes", score_delta=25,
                       evidence=[b4.EvidenceRef(**REF)], description="PA holder ≥ 25%")  # fmt: skip
    j = b4.JudgeOutcome(judge_id="J-SUPPORT", target="p-1", verdict="veto", effect="unknown",
                        model="jev-1.14.0")  # fmt: skip
    rec = b4.DecisionRecord(decision_id="d-1", case="B", run="r1", fingerprint="f",
                            kyb_decision=_decision(), rule_results=[rr],
                            judge_outcomes=[j])  # fmt: skip
    assert rec.signature_state == "pending_signature"
    entry = b4.WatchlistEntry(entry_id="PEP-0101", list_kind="pep", name="A. B. R.",
                              in_office=False, left_office="2023-03-27")  # fmt: skip
    inst = b4.PredicateInstance(instance_id="sp-c1", predicate="same_party", kind="model",
                                args={"predicate": "same_party", "candidate_id": "c1",
                                      "entity": facts.entities[0], "entry": entry},
                                evidence_universe=["c-B03-02"])  # fmt: skip
    assert isinstance(inst.args, b4.SamePartyArgs)


def test_contracts_refuse_unknown_fields_and_bad_values():
    with pytest.raises(ValidationError):
        b4.EvidenceRef(**REF, note="x")
    with pytest.raises(ValidationError):
        b4.RuleResult(rule_id="SCR-01", severity="MAYBE", fired="yes", description="")
    with pytest.raises(ValidationError):  # a conflict needs two values (spec)
        b4.Conflict(conflict_id="x", subject="e", attribute="ownership_pct",
                    values=[b4.ConflictValue(value="50", evidence=[])])  # fmt: skip
    with pytest.raises(ValidationError):
        b4.PredicateAnswer(instance_id="p", value="maybe", confidence=0.5, evidence=[],
                           rationale="")  # fmt: skip
    with pytest.raises(ValidationError):  # an OPS-01 indicator is present / absent / unknown
        b4.TriFact(state="probably")


def test_a_decision_record_needs_a_valid_kyb_decision():
    with pytest.raises(ValidationError):
        b4.DecisionRecord(decision_id="d", case="B", run="r1", fingerprint="f", kyb_decision={},
                          rule_results=[], judge_outcomes=[])  # fmt: skip
    bad = {**_decision(), "status": "APPROVE"}
    with pytest.raises(ValidationError):
        b4.DecisionRecord(decision_id="d", case="B", run="r1", fingerprint="f", kyb_decision=bad,
                          rule_results=[], judge_outcomes=[])  # fmt: skip


def test_comparable_profiles_ignore_rule_order():
    assert b4.comparable_key("mx", ["DOC-01", "JUR-02"]) == b4.comparable_key(
        "MX", ["JUR-02", "DOC-01"]
    )


def test_model_predicates_need_their_own_typed_args():
    doc = b4.Document(doc="05", document_name="lease.pdf", doc_type="lease", signed=True,
                      signed_on="2026-01-25", evidence=[])  # fmt: skip
    ok = b4.PredicateInstance(instance_id="p1", predicate="doc_is_proof_of_address", kind="model",
                              args={"predicate": "doc_is_proof_of_address", "document": doc},
                              evidence_universe=[])  # fmt: skip
    assert isinstance(ok.args, b4.ProofOfAddressArgs) and ok.args.document.signed_on == "2026-01-25"
    with pytest.raises(ValidationError):  # empty args for a model predicate
        b4.PredicateInstance(instance_id="p2", predicate="doc_is_proof_of_address", kind="model",
                             args={}, evidence_universe=[])  # fmt: skip
    with pytest.raises(ValidationError):  # another predicate's args
        b4.PredicateInstance(instance_id="p3", predicate="doc_is_proof_of_address", kind="model",
                             args={"predicate": "amendment_effective", "holding_ids": [],
                                   "holdings": []}, evidence_universe=[])  # fmt: skip
    with pytest.raises(ValidationError):  # a model predicate dispatched as code
        b4.PredicateInstance(instance_id="p4", predicate="same_party", kind="code", args={},
                             evidence_universe=[])  # fmt: skip
    code = b4.PredicateInstance(instance_id="p5", predicate="doc_fresh", kind="code",
                                args={"doc": "05", "max_months": 6},
                                evidence_universe=[])  # fmt: skip
    assert code.args == {"doc": "05", "max_months": 6}


def test_entities_carry_regions_for_r2_r3():
    e = b4.Entity(entity_id="e-x", kind="company", names=["X"], jurisdiction=["AE"],
                  regions=["DMCC free zone"], evidence=[])  # fmt: skip
    assert e.regions == ["DMCC free zone"]
    assert b4.Entity(entity_id="e-y", kind="person", names=["Y"], evidence=[]).regions == []
