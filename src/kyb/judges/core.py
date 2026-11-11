"""Exact judges: no model clients, clocks, or filesystem access."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from kyb.contracts.b4 import (
    FactSet,
    JudgeOutcome,
    PolicyModel,
    PredicateAnswer,
    PredicateInstance,
    RuleResult,
    WorkOrder,
)
from kyb.contracts.kyb_decision_schema import KYBDecision
from kyb.policy import compose
from kyb.policy.compose import GUARD_ID
from kyb.policy.rules import claim_rules, fact_rules
from kyb.predicates.lifecycle import lifecycle_reasons

# A conservative input guard, pinned in the fingerprint through JUDGES_VERSION. This is not
# a policy score threshold and never creates an approval or rejection.
MIN_CONFIDENCE = 0.8
JUDGES_VERSION = "exact-judges-v4-approval-inputs-support-confidence-0.8-witness-anomaly"
APPROVAL_ASSURANCE = "J-APPROVAL-ASSURANCE"
FIXTURE_INPUT = "J-FIXTURE-INPUT"
ANOMALY = "J-ANOMALY"
APPROVAL_ONLY = frozenset({APPROVAL_ASSURANCE, FIXTURE_INPUT, ANOMALY, "j-support-unavailable"})
SELF_REFERENCE = "self_referential_verification"
COVERAGE_GAP = "coverage_incomplete"  # the finding that accompanies `uncovered` facts

# spec: the decision profile is chosen at decide time and recorded on every decision it touches.
# `strict` is the default and the only behaviour before spec. `approve-measured` scopes the two
# fact-gap vetoes (failed claims, uncovered facts) to the gaps a rule reads; see `fact_judges`.
STRICT = "strict"
APPROVE_MEASURED = "approve-measured"
PROFILES = (STRICT, APPROVE_MEASURED)
PROFILE_JUDGE = "J-PROFILE"
WARNING = "warning (approve-measured): "


def outcome(judge: str, target: str, reason: str = "", *, unknown: bool = False) -> JudgeOutcome:
    return JudgeOutcome(
        judge_id=judge,
        target=target,
        verdict="veto" if reason else "pass",
        effect=("unknown" if unknown else "manual_review") if reason else "none",
        reason=reason,
    )


def _refs(node: Any) -> Iterable[dict]:
    if isinstance(node, dict):
        if {"claim_id", "doc", "page", "verified"} <= node.keys():
            yield node
        for value in node.values():
            yield from _refs(value)
    elif isinstance(node, list):
        for value in node:
            yield from _refs(value)


def live_rules(rules: Iterable[RuleResult], policy: PolicyModel) -> frozenset[str]:
    """The rules (and their stable kinds) whose value can change the status or the score: a REJECT
    or REVIEW trigger, a rule with non-zero points, or one the policy does not define (it reads
    `unknown`). The guard is not a rule."""
    specs = {spec.rule_id: spec for spec in policy.rules}
    out: set[str] = set()
    for r in rules:
        if r.rule_id == GUARD_ID:
            continue
        spec = specs.get(r.rule_id)
        kind = (spec.kind or spec.rule_id) if spec else ""
        if (
            spec is None
            or kind in ("conflict_multiplier", "document_age_penalty")
            or spec.severity != "SCORE"
            or spec.score_delta != 0
        ):
            out.add(r.rule_id)
            if kind:
                out.add(kind)
    return frozenset(out)


def fact_judges(facts: FactSet, live: frozenset[str] | None = None) -> list[JudgeOutcome]:
    """J-EVIDENCE and J-VERIFICATION over the FactSet.

    Strict (`live is None`): any failed claim and any uncovered material fact vetoes approval.
    approve-measured (`live` = `live_rules`): a failed claim or an uncovered fact vetoes only when
    a live rule reads it (policy/rules.py GAP_RULES; an unclassified one feeds every rule, so it
    counts). One that no live rule reads becomes a warning (a passing outcome that says so). The
    other checks are unchanged under both: evidence integrity, reader isolation, document
    coverage of the verification, findings other than `coverage_incomplete`, and W1."""

    def feeds(rule_ids: tuple[str, ...]) -> bool:
        return live is None or not live or "*" in rule_ids or bool(live & set(rule_ids))

    evidence = {e.evidence_id: e for e in facts.evidence}
    errors = []
    if len(evidence) != len(facts.evidence):
        errors.append("duplicate evidence ids")
    for ref in _refs(facts.model_dump(mode="json", exclude={"unverified"})):
        e = evidence.get(ref["claim_id"])
        if (
            not e
            or not e.verified
            or not ref["verified"]
            or (e.doc, e.page) != (ref["doc"], ref["page"])
        ):
            errors.append(f"{ref['claim_id']}: unresolved, unverified or mismatched evidence")
    for e in facts.evidence:
        x0, y0, x1, y1 = e.bbox
        if e.verified and (not e.excerpt.strip() or x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0):
            errors.append(f"{e.evidence_id}: empty excerpt or invalid crop")
    out = [outcome("J-EVIDENCE", "facts", "; ".join(sorted(set(errors))))]
    docs = {d.doc for d in facts.documents}
    coverage = {v.doc for v in facts.verification}
    out.append(
        outcome(
            "J-VERIFICATION",
            "facts",
            "missing or duplicate document verification"
            if docs != coverage or len(coverage) != len(facts.verification)
            else "",
        )
    )
    for v in sorted(facts.verification, key=lambda v: v.doc):
        problems = []
        assurance = []
        unread = [u for u in v.uncovered if not feeds(fact_rules(u.kind))]
        read = len(v.uncovered) - len(unread)
        if v.assurance != "W2" or SELF_REFERENCE in v.findings:
            assurance.append("W1/self-referential evidence cannot support approval")
        if not v.isolated:
            problems.append("reader not isolated")
        if v.uncovered and read:
            problems.append("uncovered material facts")
        other_findings = sorted(set(v.findings) - {SELF_REFERENCE})
        if v.uncovered and not read:
            other_findings = [f for f in other_findings if f != COVERAGE_GAP]
        if other_findings:
            problems.append("verification findings: " + ", ".join(other_findings))
        if problems or not unread:
            out.append(outcome("J-VERIFICATION", v.doc, "; ".join(problems)))
        else:
            kinds = ", ".join(sorted({u.kind for u in unread}))
            out.append(_warning("J-VERIFICATION", v.doc, f"{len(unread)} uncovered ({kinds})"))
        out.append(outcome(APPROVAL_ASSURANCE, v.doc, "; ".join(assurance)))
    if facts.unverified:
        fields = {e.evidence_id: e.field for e in facts.evidence}
        failed = [r for r in facts.unverified if feeds(claim_rules(fields.get(r.claim_id) or ""))]
        if failed or live is None:
            out.append(outcome("J-EVIDENCE", "unverified", "claims failed verification"))
        else:
            out.append(_warning("J-EVIDENCE", "unverified", f"{len(facts.unverified)} failed"))
    return out


def _warning(judge: str, target: str, what: str) -> JudgeOutcome:
    return JudgeOutcome(
        judge_id=judge,
        target=target,
        verdict="pass",
        effect="none",
        reason=f"{WARNING}{what}; no live rule reads them",
    )


def profile_outcome(profile: str, judges: Iterable[JudgeOutcome]) -> JudgeOutcome:
    """The record that a decision was made under a non-default profile (never a veto)."""
    warnings = sum(1 for j in judges if j.reason.startswith(WARNING))
    return JudgeOutcome(
        judge_id=PROFILE_JUDGE,
        target="decision",
        verdict="pass",
        effect="none",
        reason=f"decision profile {profile}; {warnings} gap warnings",
    )


def witness_judges(now: dict[str, dict | None]) -> list[JudgeOutcome]:
    """Approval-only caps from each document's witness state as of now (L8 `assurance_now`).

    J-ANOMALY: any recorded envelope anomaly, including attempts
    that later healed, prevents approval. A missing witness record is unknown, never clean.
    `None` means no witness record exists for the document."""
    out = []
    for doc, w in sorted(now.items()):
        if w is None:
            out.append(outcome(ANOMALY, doc, f"{doc}: no witness record; anomalies unknown"))
            continue
        count = int(w["anomalies"]["count"])
        severities = ", ".join(f"{k}={v}" for k, v in sorted(w["anomalies"]["severities"].items()))
        out.append(
            outcome(
                ANOMALY,
                doc,
                f"{doc}: {count} witness envelope anomalies ({severities}); cannot support approval"
                if count
                else "",
            )
        )
        assurance = []
        if w["assurance"] != "W2":
            assurance.append(f"witness assurance now {w['assurance']}")
        if w["self_referential"]:
            assurance.append("self-referential witness (spec stratum B)")
        out.append(outcome(APPROVAL_ASSURANCE, f"witness/{doc}", "; ".join(assurance)))
    return out


def predicate_judges(
    facts: FactSet,
    work: WorkOrder,
    answers: Iterable[PredicateAnswer],
    *,
    who: dict | None,
    events: list[dict],
    sources: dict | None = None,
    support: Callable[[PredicateInstance, PredicateAnswer, list], JudgeOutcome] | None = None,
) -> tuple[list[PredicateAnswer], list[JudgeOutcome]]:
    instances = {i.instance_id: i for i in work.instances}
    evidence = {e.evidence_id: e for e in facts.evidence}
    answers = list(answers)
    by_id = {a.instance_id: a for a in answers}
    shape = (
        ""
        if len(by_id) == len(answers) and set(by_id) <= set(instances)
        else ("duplicate or foreign predicate answers")
    )
    outcomes = [outcome("J-SCHEMA", "predicates", shape)]
    sources = sources or {}
    if work.instances and not sources:
        reasons = lifecycle_reasons(
            events,
            who or {},
            f"kyb-predicates {work.work_order_id}",
            {"predicate_work_order", "propose_predicates"},
        )
        outcomes.append(outcome("J-PRED-ISOLATION", work.work_order_id, "; ".join(reasons)))
    checked = []
    for iid, instance in sorted(instances.items()):
        if sources:
            source = sources.get(iid, {})
            reasons = lifecycle_reasons(
                events,
                source.get("who") or {},
                f"kyb-predicates {source.get('work_order_id', work.work_order_id)}",
                {"predicate_work_order", "propose_predicates"},
            )
            outcomes.append(outcome("J-PRED-ISOLATION", iid, "; ".join(reasons)))
        a = by_id.get(iid)
        if a is None:
            a = PredicateAnswer(
                instance_id=iid,
                value="UNKNOWN",
                confidence=0,
                evidence=[],
                rationale="predicate not answered",
            )
        reasons = []
        if a.value != "UNKNOWN" and not a.evidence:
            reasons.append("answer has no cited evidence")
        if a.value != "UNKNOWN" and a.confidence < MIN_CONFIDENCE:
            reasons.append("low confidence")
        for eid in a.evidence:
            if (
                eid not in instance.evidence_universe
                or eid not in evidence
                or not evidence[eid].verified
            ):
                reasons.append(f"{eid}: outside verified evidence universe")
        outcomes.append(outcome("J-EVIDENCE", iid, "; ".join(reasons), unknown=True))
        if shape or reasons:
            a = a.model_copy(update={"value": "UNKNOWN"})
        if support is None:
            # spec: preserve the rule explanations on all-MR goldens; unchecked answers can
            # never be the foundation of an APPROVED, regardless of their value.
            if a.value != "UNKNOWN":
                outcomes.append(
                    outcome(
                        "j-support-unavailable",
                        iid,
                        "J-SUPPORT unavailable; model answer unchecked",
                    )
                )
            else:
                outcomes.append(outcome("J-SUPPORT", iid, "UNKNOWN answer", unknown=True))
        else:
            verdict = support(instance, a, [evidence[eid] for eid in a.evidence if eid in evidence])
            if verdict.judge_id != "J-SUPPORT" or verdict.target != iid:
                verdict = outcome("J-SUPPORT", iid, "invalid support judge response", unknown=True)
            outcomes.append(verdict)
            if verdict.verdict == "veto":
                a = a.model_copy(update={"value": "UNKNOWN"})
        checked.append(a)
    return checked, outcomes


def decision_judges(
    decision: dict | KYBDecision, rules: list[RuleResult], policy: PolicyModel
) -> list[JudgeOutcome]:
    try:
        d = KYBDecision.model_validate(decision)
    except ValueError as exc:
        return [outcome("J-SCHEMA", "decision", str(exc))]
    expected = compose(rules, policy)
    score = outcome(
        "J-ARITH",
        "decision",
        "score differs from the policy sum/cap" if d.risk_score != expected.risk_score else "",
    )
    precedence = outcome(
        "J-PRECEDENCE",
        "decision",
        "status differs from policy composition" if d.status != expected.status else "",
    )
    return [outcome("J-SCHEMA", "decision"), score, precedence]
