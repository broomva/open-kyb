"""Pure decision assembly. All variable state is supplied by the action boundary."""

from __future__ import annotations

from datetime import datetime

from kyb.contracts import b4
from kyb.contracts.kyb_decision_schema import KYBDecision, Violation
from kyb.judges.core import APPROVAL_ONLY, JUDGES_VERSION, decision_judges
from kyb.policy import compose, evaluate, version_id
from kyb.util.hashing import sha256_json


def decision_record(
    facts: b4.FactSet,
    screening: b4.ScreeningResult,
    policy: b4.PolicyModel,
    answers: list[b4.PredicateAnswer],
    judges: list[b4.JudgeOutcome],
    corrections: list[b4.CorrectionRecord],
    *,
    provenance: dict,
    emitted_at: datetime,
    audit_trail: list[dict],
) -> b4.DecisionRecord:
    evaluation = evaluate(facts, screening, policy, answers)
    rules = evaluation.rule_results
    base = compose(rules, policy)
    if base.status != "APPROVED":
        # spec/spec: approval assurance guards preserve independently established code triggers.
        judges = [
            j.model_copy(
                update={
                    "verdict": "pass",
                    "effect": "none",
                    "reason": f"{j.reason}; approval-only guard not required for {base.status}",
                }
            )
            if j.judge_id in APPROVAL_ONLY and j.verdict == "veto"
            else j
            for j in judges
        ]
    evidence = {e.evidence_id: e for e in facts.evidence if e.verified}
    citations = [b4.citation(facts.case, e) for _, e in sorted(evidence.items())]
    violations = []
    for rule in rules:
        if rule.fired != "yes":
            continue
        refs = sorted({r.claim_id for r in rule.evidence if r.verified})
        violations.append(
            Violation(
                rule_id=rule.rule_id,
                severity=rule.severity,
                score_delta=rule.score_delta,
                description=rule.description,
                evidence=[
                    b4.citation(facts.case, evidence[eid]) for eid in refs if eid in evidence
                ],
            )
        )
    applicant = next(e for e in facts.entities if e.entity_id == facts.applicant)
    correction_ids = sorted(c.correction_id for c in corrections)
    decision = KYBDecision(
        decision_id="pending",
        entity_id=facts.applicant,
        entity_name=sorted(applicant.names)[0],
        rules_version=version_id(policy),
        status=base.status,
        risk_score=base.risk_score,
        violations=violations,
        citations=citations,
        rationale="; ".join(base.reasons)[:2000],
        officer_overrides_considered=correction_ids,
        audit_trail=audit_trail,
        emitted_at=emitted_at,
    )
    judges = [*judges, *decision_judges(decision, rules, policy)]
    composition = compose(rules, policy, judges)
    fingerprint = sha256_json(
        {
            "facts": facts.fact_set_hash,
            "screening": screening.model_dump(mode="json"),
            "policy": policy.content_hash,
            "as_of": facts.as_of,
            "answers": [
                a.model_dump(mode="json") for a in sorted(answers, key=lambda a: a.instance_id)
            ],
            "judges_version": JUDGES_VERSION,
            "judges": [j.model_dump(mode="json") for j in judges],
            "corrections": [c.model_dump(mode="json") for c in corrections],
            "provenance": provenance,
        }
    )
    decision_id = "d-" + fingerprint.removeprefix("sha256:")
    rationale = list(composition.reasons)
    if correction_ids:
        rationale.append(
            "Comparable officer corrections considered as context: "
            + ", ".join(correction_ids)
            + "; policy unchanged"
        )
    decision = decision.model_copy(
        update={
            "decision_id": decision_id,
            "status": composition.status,
            "risk_score": composition.risk_score,
            "rationale": "; ".join(rationale)[:2000],
        }
    )
    return b4.DecisionRecord(
        decision_id=decision_id,
        case=facts.case,
        run=facts.run,
        fingerprint=fingerprint,
        corrections_considered=correction_ids,
        kyb_decision=decision,
        rule_results=rules,
        judge_outcomes=judges,
        signature_state="pending_signature",
    )
