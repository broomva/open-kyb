"""`sign_decision`: an enrolled officer assents to a recommendation or overrides it (spec).

Officer only: the call carries the officer's id and token, checked here, and the hook denies the
agent's attempt before it gets this far. Code never decides *for* the officer. An override sets
a different status, needs a written rationale, and is recorded as a `CorrectionRecord`
that L4 later surfaces on comparable profiles. The correction informs; it never edits the
policy or the original decision (invariant 6). The signature is written last and is keyed by the
decision id, so a decision is signed exactly once and nothing counts as signed before it exists.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from kyb.contracts.b4 import CORRECTION_REL, CorrectionRecord, FactSet
from kyb.signing.decisions import load_decision, signature, signature_rel
from kyb.signing.ledger import read_record, unreceipted, write_record
from kyb.signing.officers import authenticate
from kyb.util.fsutil import safe_id
from kyb.util.hashing import sha256_json
from kyb.workspace.catalog import Catalog

Verdict = Literal["assent", "override"]
APPROVED_VIA = ("terminal", "harness-ask")
STATUSES = ("APPROVED", "REJECTED", "MANUAL_REVIEW")
UNKNOWN_JURISDICTION = "UNKNOWN"  # recorded, so the override exists; it matches no applicant


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jurisdictions(root: Path, case: str, run: str) -> list[str]:
    """The applicant's incorporation jurisdictions as the decision's facts record them: the keys
    L4's retrieval looks corrections up by."""
    rel = f"gold/facts/{case}/{run}.json"
    if Catalog(root).get(rel) is None:
        return [UNKNOWN_JURISDICTION]
    facts = FactSet.model_validate(read_record(root, rel))
    applicant = next((e for e in facts.entities if e.entity_id == facts.applicant), None)
    found = sorted({j.upper() for j in applicant.jurisdiction}) if applicant else []
    return found or [UNKNOWN_JURISDICTION]


def _correction_id(decision_id: str, officer: str, to_status: str, rationale: str,
                   jurisdiction: str) -> str:  # fmt: skip
    digest = sha256_json([decision_id, officer, to_status, rationale, jurisdiction])
    return "co-" + hashlib.sha256(digest.encode()).hexdigest()[:16]


def _record_override(root: Path, decision, officer: str, to_status: str, rationale: str,
                     as_of: str, inputs: list[str]) -> list[str]:  # fmt: skip
    fired = sorted({r.rule_id for r in decision.rule_results if r.fired == "yes"})
    ids = []
    for jur in _jurisdictions(root, decision.case, decision.run):
        cid = _correction_id(decision.decision_id, officer, to_status, rationale, jur)
        rel = CORRECTION_REL.format(correction_id=cid)
        # an interrupted earlier try keeps its timestamp, so the retry publishes the same bytes
        prior = unreceipted(root, rel) or (
            read_record(root, rel) if Catalog(root).get(rel) is not None else None
        )
        record = CorrectionRecord(
            correction_id=cid, decision_id=decision.decision_id, officer=officer,
            from_status=decision.kyb_decision.status, to_status=to_status,
            rationale=rationale, jurisdiction=jur, fired_rules=fired,
            recorded_at=prior["recorded_at"] if prior else _now(),
        )  # fmt: skip
        write_record(root, rel, record.model_dump(mode="json"), action="record_override",
                     tier="decided", as_of=as_of, inputs=inputs,
                     detail={"decision_id": decision.decision_id, "officer": officer,
                             "to_status": to_status})  # fmt: skip
        ids.append(cid)
    return ids


def sign_decision(root: Path, *, decision_id: str, verdict: Verdict, officer: str,
                  officer_token: str | None, status: str | None = None, rationale: str = "",
                  approved_via: str = "terminal") -> dict[str, Any]:  # fmt: skip
    """Sign one recommendation. Call under `workspace_action`. Raises PermissionError for a bad
    token or an already-signed decision, LookupError for an unknown decision, ValueError for a
    malformed verdict."""
    authenticate(root, officer, officer_token)  # first: nothing else is revealed to a stranger
    decision_id = safe_id(decision_id, "decision id")
    if approved_via not in APPROVED_VIA:
        raise ValueError(f"approved_via must be one of {', '.join(APPROVED_VIA)}")
    if verdict not in ("assent", "override"):
        raise ValueError(f"verdict must be 'assent' or 'override' (got {verdict!r})")
    decision = load_decision(root, decision_id)
    if (done := signature(root, decision_id)) is not None:
        raise PermissionError(f"{decision_id} is already signed by {done['officer']} "
                              f"({done['verdict']}); a signature is written once")  # fmt: skip
    current = decision.kyb_decision.status
    rationale = rationale.strip()
    if verdict == "assent":
        if status not in (None, current):
            raise ValueError(f"assent signs the recommendation as it stands ({current}); "
                             "to change the status, override")  # fmt: skip
        to_status = current
    else:
        if status not in STATUSES:
            raise ValueError(f"an override needs a status: one of {', '.join(STATUSES)}")
        if status == current:
            raise ValueError(f"the recommendation is already {current}: assent instead")
        if not rationale:
            raise ValueError("an override needs a written rationale")
        to_status = status
    catalog = Catalog(root)
    dec_rel = f"ledger/decisions/{decision_id}.json"
    as_of = catalog.get(dec_rel)["as_of"]
    inputs = [catalog.ref(dec_rel)]
    corrections = (_record_override(root, decision, officer, to_status, rationale, as_of, inputs)
                   if verdict == "override" else [])  # fmt: skip
    rel = signature_rel(decision_id)
    prior = unreceipted(root, rel)
    record = {
        "decision_id": decision_id, "case": decision.case, "run": decision.run,
        "officer": officer, "verdict": verdict, "from_status": current, "to_status": to_status,
        "rationale": rationale, "correction_ids": corrections, "approved_via": approved_via,
        "signed_at": prior["signed_at"] if prior else _now(),
    }  # fmt: skip
    write_record(root, rel, record, action="sign_decision", tier="decided", as_of=as_of,
                 inputs=[*inputs, *(catalog.ref(CORRECTION_REL.format(correction_id=c))
                                    for c in corrections)],
                 detail={"decision_id": decision_id, "officer": officer, "verdict": verdict,
                         "to_status": to_status, "approved_via": approved_via})  # fmt: skip
    return {**record, "signature": rel, "signature_state": "signed"}
