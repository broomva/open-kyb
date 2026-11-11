"""Deterministic finite readiness diagnostics, never a policy decision."""

from typing import Any

from kyb.contracts.b4 import FactSet
from kyb.facts.derive import DocInput
from kyb.loop.models import Gap, LoopBudget, LoopPlan
from kyb.loop.reconcile import _refs, reconcile
from kyb.util.hashing import sha256_json


def plan_cycle(
    facts: FactSet,
    documents: list[DocInput],
    *,
    budget: LoopBudget,
    previous: LoopPlan | None = None,
    usage: dict[str, Any] | None = None,
) -> LoopPlan:
    gaps: list[Gap] = []
    semantic: set[str] = set()

    def add(
        kind: str,
        entities: list[str],
        docs: list[str],
        field: str,
        repairable: bool,
        reason: str,
        evidence: list[str],
        key: Any,
    ) -> None:
        digest = sha256_json([kind, key])
        semantic.add(digest)
        gaps.append(
            Gap(
                gap_id=digest,
                kind=kind,
                entity_ids=sorted(set(entities)),
                doc_ids=sorted(set(docs)),
                field=field,
                repairable=repairable,
                reason=reason,
                evidence_ids=sorted(set(evidence)),
            )
        )

    names = {e.entity_id: sorted(n.casefold().strip() for n in e.names) for e in facts.entities}
    for v in facts.verification:
        for u in v.uncovered:
            add(
                "uncovered",
                [],
                [v.doc],
                u.attribute or u.kind,
                not (
                    "explicit_nondisclosure" in v.findings
                    and "controller" in (u.attribute or u.kind).casefold()
                ),
                "Reader: revisit the uncovered material field in this document.",
                [],
                [v.doc, u.kind, u.subject.casefold(), u.attribute, u.value.casefold()],
            )
        for finding in v.findings:
            add(
                "verification_finding",
                [],
                [v.doc],
                finding,
                False,
                "Verifier: inspect the recorded finding.",
                [],
                [v.doc, finding],
            )
        if not v.isolated:
            add(
                "assurance",
                [],
                [v.doc],
                "isolation",
                False,
                "Verifier: reader isolation is not established.",
                [],
                [v.doc, "isolation"],
            )
    refused = {(ref.doc, ref.claim_id) for ref in facts.unverified}
    for d in documents:
        for claim in d.claims:
            cid = str(claim.get("claim_id", ""))
            if not d.verdicts.get(cid, False) or (d.doc, cid) in refused:
                add(
                    "failed_claim",
                    [],
                    [d.doc],
                    str(claim.get("attribute") or claim.get("type") or "unknown"),
                    True,
                    "Reader and verifier: revisit the failed field.",
                    [cid] if cid else [],
                    [
                        d.doc,
                        claim.get("subject"),
                        claim.get("attribute"),
                        claim.get("type"),
                        claim.get("value"),
                    ],
                )
    for d in documents:
        for claim in d.claims:
            text = (
                str(claim.get("value", "")).casefold()
                + " "
                + str(claim.get("excerpt", "")).casefold()
            )
            if (
                d.verdicts.get(str(claim.get("claim_id", "")), False)
                and any(word in text for word in ("controller", "controlador", "controlante"))
                and any(
                    word in text
                    for word in (
                        "disclosed on request",
                        "disclose on request",
                        "divulgará a solicitud",
                        "revelará a solicitud",
                    )
                )
            ):
                add(
                    "nondisclosure",
                    [],
                    [d.doc],
                    "controller",
                    False,
                    "Reviewer: the controller is explicitly undisclosed; do not invent one.",
                    [str(claim.get("claim_id", ""))],
                    [d.doc, "controller_nondisclosure"],
                )
    for c in facts.conflicts:
        refs = [r for v in c.values for r in v.evidence] + c.context
        add(
            "conflict",
            [c.subject],
            [r.doc for r in refs],
            c.attribute,
            False,
            "Reviewer: conflicting records remain unresolved.",
            [r.claim_id for r in refs],
            [names.get(c.subject, []), c.attribute, sorted(v.value for v in c.values)],
        )
    holders = {h.owner for h in facts.holdings}
    for e in facts.entities:
        if (
            e.entity_id in holders
            and e.kind == "company"
            and not any(_refs(facts, e, "jurisdiction", j) for j in e.jurisdiction)
        ):
            add(
                "jurisdiction",
                [e.entity_id],
                [r.doc for r in e.evidence],
                "jurisdiction",
                True,
                "Reader: find an explicit incorporation jurisdiction; do not infer it from a name.",
                [r.claim_id for r in e.evidence],
                names.get(e.entity_id, []),
            )
    candidates = reconcile(facts)
    for c in candidates:
        if c.state != "supported":
            add(
                "identity",
                [c.left, c.right],
                [],
                "identity",
                False,
                "Reviewer: identity remains unresolved in the policy facts.",
                c.evidence_ids,
                sorted([names.get(c.left, []), names.get(c.right, [])]),
            )
    supported_refs = {(e.doc, e.evidence_id) for e in facts.evidence if e.verified and e.field}
    verified = sorted(
        {
            sha256_json(
                [
                    c.get("subject"),
                    c.get("type"),
                    c.get("attribute"),
                    c.get("value"),
                    c.get("qualifiers"),
                ]
            )
            for d in documents
            for c in d.claims
            if d.verdicts.get(str(c.get("claim_id", "")), False)
            and (d.doc, str(c.get("claim_id", ""))) not in refused
            and (d.doc, str(c.get("claim_id", ""))) in supported_refs
        }
    )
    fingerprint = sha256_json(
        {"facts": facts.fact_set_hash, "documents": [d.doc for d in documents]}
    )
    attempt = previous.attempt + 1 if previous else 0
    usage = usage or {}

    def measured(metric: str) -> float | int | None:
        value = usage.get(metric)
        if isinstance(value, dict):
            value = value.get("value") if value.get("status") == "measured" else None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            return None
        return value

    state = "repair" if gaps else "ready"
    if not documents or not facts.verification or any(not v.isolated for v in facts.verification):
        state = "blocked"
    elif gaps and (
        attempt >= budget.max_attempts
        or budget.max_elapsed_seconds == 0
        or (
            measured("elapsed_seconds") is not None
            and measured("elapsed_seconds") >= budget.max_elapsed_seconds
        )
    ):
        state = "budget_exhausted"
    elif gaps and any(
        getattr(budget, limit) is not None and measured(metric) is None
        for limit, metric in [
            ("max_tokens", "tokens"),
            ("max_provider_calls", "provider_calls"),
            ("max_cost_usd", "cost_usd"),
        ]
    ):
        state = "blocked"
    elif gaps and any(
        getattr(budget, limit) is not None
        and measured(metric) is not None
        and measured(metric) >= getattr(budget, limit)
        for limit, metric in [
            ("max_tokens", "tokens"),
            ("max_provider_calls", "provider_calls"),
            ("max_cost_usd", "cost_usd"),
        ]
    ):
        state = "budget_exhausted"
    elif (
        previous
        and set(previous.metrics.get("semantic_gap_keys", [])) == semantic
        and set(previous.metrics.get("verified_identities", [])) == set(verified)
    ):
        state = "no_progress"
    elif gaps and not any(g.repairable for g in gaps):
        state = "manual_review"
    gaps.sort(key=lambda g: (g.gap_id, g.doc_ids, g.entity_ids))
    actions = (
        [{"doc_ids": g.doc_ids, "field": g.field} for g in gaps if g.repairable]
        if state == "repair"
        else []
    )
    metrics = {
        "stop_reason": (
            "usage_unknown"
            if state == "blocked"
            and documents
            and facts.verification
            and all(v.isolated for v in facts.verification)
            else state
            if state != "repair"
            else None
        ),
        "semantic_gap_keys": sorted(semantic),
        "verified_identities": verified,
        "verified_assertions": len(verified),
        "verified_assertion_delta": (
            len(verified) - previous.metrics.get("verified_assertions", 0) if previous else None
        ),
        "verified_rule_input_fields": len(
            {e.field for e in facts.evidence if e.verified and e.field}
        ),
        "verification_refusals": len(refused),
        "known_new_false_accepts": None,
        "unresolved_gaps": len(semantic),
        "unresolved_gap_delta": (
            len(semantic) - previous.metrics.get("unresolved_gaps", 0) if previous else None
        ),
        "repairable_gaps": len({g.gap_id for g in gaps if g.repairable}),
        "candidate_states": {
            s: sum(c.state == s for c in candidates)
            for s in ("supported", "unresolved", "contradictory")
        },
        "holding_statuses": {
            s: sum(h.status == s for h in facts.holdings)
            for s in ("current", "pending", "struck", "historical")
        },
        "usage": {
            "tokens": usage.get("tokens"),
            "provider_calls": usage.get("provider_calls"),
            "cost_usd": usage.get("cost_usd"),
            "elapsed_seconds": usage.get("elapsed_seconds"),
            "provenance": usage.get("provenance", "unknown"),
        },
        "budget": budget.model_dump(mode="json"),
        "cost_per_verified_input_gain": None,
    }
    base = dict(
        version="evidence-loop-v1",
        case=facts.case,
        source_run=facts.run,
        input_fingerprint=fingerprint,
        fact_set_hash=facts.fact_set_hash,
        previous_plan=previous.plan_id if previous else None,
        attempt=attempt,
        state=state,
        gaps=gaps,
        candidates=candidates,
        next_actions=actions,
        metrics=metrics,
    )
    plan_id = sha256_json(
        {
            **base,
            "gaps": [g.model_dump(mode="json") for g in gaps],
            "candidates": [c.model_dump(mode="json") for c in candidates],
        }
    )
    return LoopPlan(plan_id=plan_id.removeprefix("sha256:"), **base)
