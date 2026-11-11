"""Recorded decisions for the L5 tests: what L4's `emit_decision` writes, built by hand.

L5 is built against fixture `DecisionRecord`s (b4-v1 is frozen); the real writer is L4's. These go
through the same receipted-write path, so a signature test reads them exactly as it would L4's.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from kyb.audit.chain import workspace_action
from kyb.contracts.b4 import (
    DECISION_REL,
    FACTS_REL,
    DecisionRecord,
    Entity,
    EvidenceRef,
    FactSet,
    JudgeOutcome,
    Operations,
    RuleResult,
    TriFact,
)
from kyb.contracts.kyb_decision_schema import KYBDecision
from kyb.signing.ledger import write_record
from kyb.util.hashing import sha256_json
from kyb.workspace.init import init_workspace

REPO = Path(__file__).resolve().parents[1]
POLICY = REPO / "examples" / "policy" / "compliance_rules.md"
AS_OF = "2026-10-25"


def workspace(tmp: Path) -> Path:
    root = tmp / "ws"
    init_workspace(root, tenant="kyb-demo", policy_src=POLICY)
    return root


def rule(rule_id: str, fired: str = "yes", severity: str = "REVIEW") -> RuleResult:
    return RuleResult(rule_id=rule_id, severity=severity, fired=fired,
                      description=f"{rule_id} {fired}")  # fmt: skip


def facts(case: str, run: str, jurisdiction: list[str]) -> FactSet:
    ref = EvidenceRef(claim_id="c-1", doc="01", page=1, verified=True)
    unknown = TriFact(state="unknown")
    body = FactSet.model_construct(
        case=case, run=run, as_of=AS_OF, fact_set_hash="", applicant="e-acme",
        entities=[Entity(entity_id="e-acme", kind="company", names=["ACME"],
                         jurisdiction=jurisdiction, evidence=[ref])],
        holdings=[], roles=[], documents=[], conflicts=[], evidence=[], verification=[],
        operations=Operations(distinct_operating_address=unknown,
                              bank_reference_or_financials=unknown,
                              employees_or_website=unknown),
    )  # fmt: skip
    digest = sha256_json(body.model_dump(mode="json", exclude={"fact_set_hash"}))
    return FactSet.model_validate(body.model_dump(mode="json") | {"fact_set_hash": digest})


def decision(case: str, run: str = "r1", *, status: str = "MANUAL_REVIEW", rules=None,
             emitted: str = "2026-10-25T12:00:00Z", name: str = "ACME", seed: str = "",
             vetoes=(), violations=(), considered=()) -> DecisionRecord:  # fmt: skip
    rules = rules if rules is not None else [rule("DOC-01")]
    decision_id = "dec-" + sha256_json([case, run, status, emitted, seed])[7:23]
    kyb = KYBDecision(
        decision_id=decision_id, entity_id=f"{case}-entity", entity_name=name,
        rules_version="2026.11-r1+sha256:test", status=status, risk_score=10,
        violations=list(violations), citations=[], rationale="fixture", audit_trail=[],
        officer_overrides_considered=list(considered),
        emitted_at=datetime.fromisoformat(emitted).astimezone(UTC),
    )  # fmt: skip
    return DecisionRecord(
        decision_id=decision_id, case=case, run=run, fingerprint="sha256:fixture",
        kyb_decision=kyb, rule_results=rules,
        judge_outcomes=[JudgeOutcome(judge_id=j, target="decision", verdict="veto",
                                     effect="manual_review", reason=f"{j} fixture")
                        for j in vetoes],
    )  # fmt: skip


def record(root: Path, rec: DecisionRecord) -> str:
    """Write a decision (and the run's facts, once) the way L4 receipts them."""
    with workspace_action(root):
        facts_rel = FACTS_REL.format(case=rec.case, run=rec.run)
        from kyb.workspace.catalog import Catalog

        if Catalog(root).get(facts_rel) is None:
            fs = facts(rec.case, rec.run, ["MX"])
            write_record(root, facts_rel, fs.model_dump(mode="json"), action="fixture_facts",
                         tier="derived", as_of=AS_OF, inputs=[], detail={})  # fmt: skip
        rel = DECISION_REL.format(decision_id=rec.decision_id)
        write_record(root, rel, rec.model_dump(mode="json"), action="emit_decision",
                     tier="decided", as_of=AS_OF, inputs=[], detail={})  # fmt: skip
    return rec.decision_id
