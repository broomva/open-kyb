"""`review_queue`: every unsigned recommendation, the riskiest to the review first (spec).

All statuses appear: MANUAL_REVIEW first, then REJECTED, then APPROVED, and oldest first within
each. This is the queue open-kyb asked for, not an inbox: an approval waits for its signature too.
A decision record that does not load is listed first as `unreadable`, never dropped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kyb.contracts.b4 import DecisionRecord
from kyb.signing.decisions import decision_paths, load_decision, signed_ids
from kyb.util.fsutil import safe_id

RANK = {"MANUAL_REVIEW": 0, "REJECTED": 1, "APPROVED": 2}


def entry(record: DecisionRecord, *, latest: str) -> dict[str, Any]:
    d = record.kyb_decision
    return {
        "decision_id": record.decision_id, "case": record.case, "run": record.run,
        "entity_name": d.entity_name, "status": d.status, "risk_score": d.risk_score,
        "emitted_at": d.emitted_at.isoformat(),
        "rules_version": d.rules_version,
        "fired_rules": [{"rule_id": r.rule_id, "severity": r.severity,
                         "description": r.description}
                        for r in record.rule_results if r.fired == "yes"],
        "unknown_rules": [r.rule_id for r in record.rule_results if r.fired == "unknown"],
        # the findings an officer weighs: coverage, integrity, isolation and agent-context caps
        "findings": [{"judge_id": j.judge_id, "target": j.target, "effect": j.effect,
                      "reason": j.reason}
                     for j in record.judge_outcomes if j.verdict == "veto"],
        "officer_overrides_considered": d.officer_overrides_considered,
        "signature_state": "pending_signature",
        # a newer decision of the same case exists: this one is history, not the live ask
        "superseded_by": None if latest == record.decision_id else latest,
    }  # fmt: skip


def _order(e: dict[str, Any]) -> tuple[int, str, str]:
    return RANK[e["status"]], e["emitted_at"], e["decision_id"]


def review_queue(
    root: Path, case: str | None = None, *, include_archived: bool = False
) -> dict[str, Any]:
    """Unsigned recommendations (optionally one case's), in review order."""
    if case is not None:
        case = safe_id(case, "case id")
    from kyb.workspace.cases import case_states

    archived = {cid for cid, state in case_states(root).items() if state.state == "archived"}
    signed = signed_ids(root)
    records: list[DecisionRecord] = []
    unreadable: list[dict[str, str]] = []
    for rel in decision_paths(root):
        decision_id = Path(rel).stem
        try:
            records.append(load_decision(root, decision_id))
        except (LookupError, PermissionError, ValueError) as exc:
            if decision_id not in signed:
                unreadable.append({"decision": rel, "error": f"{type(exc).__name__}: {exc}"})
    newest: dict[str, tuple[str, str]] = {}
    for r in records:
        key = (r.kyb_decision.emitted_at.isoformat(), r.decision_id)
        newest[r.case] = max(newest.get(r.case, key), key)
    rows = [
        entry(r, latest=newest[r.case][1])
        for r in records
        if r.decision_id not in signed
        and (case is None or r.case == case)
        and (include_archived or r.case not in archived)
    ]
    rows.sort(key=_order)
    in_scope = {
        r.decision_id
        for r in records
        if (case is None or r.case == case) and (include_archived or r.case not in archived)
    }
    return {
        "queue": rows,
        "unsigned": len(rows),
        "signed": len(signed & in_scope),
        "unreadable": unreadable,
    }
