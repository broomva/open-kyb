"""Suite 1 (contract and invariants): the records validate, composition is arithmetic, and the
audit chain verifies and detects every kind of tampering.

Decisions are emitted by L4 (`kyb decide`, and the decision assembly for the synthetic variants),
so every emitted `DecisionRecord.kyb_decision` is validated against `KYBDecision`, and a real
decision's `audit_trail` is checked to be the workspace chain it was emitted from. The b4 records
the decision is built from, the one evidence → `Citation` mapping, and the chain are checked
beside them.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from kyb.audit.chain import AuditLog, verify_workspace, write_checkpoint
from kyb.contracts import b4
from kyb.contracts.kyb_decision_schema import Citation
from kyb.evals.decisions import all_cases, base_policy, decide
from kyb.evals.scorecard import Case, Suite, check


def _records() -> list[Case]:
    out = []
    for cid, _, p in all_cases():
        again = b4.FactSet.model_validate_json(p.facts.model_dump_json())
        ok = again == p.facts and all(
            Citation.model_validate(b4.citation(p.facts.case, e).model_dump())
            for e in p.facts.evidence
        )
        refs = {e.evidence_id for e in p.facts.evidence}
        used = {r.claim_id for h in p.facts.holdings for r in h.evidence}
        out.append(
            check(
                f"factset-roundtrip:{cid}",
                ok and used <= refs,
                "FactSet does not round-trip, or a ref does not resolve",
            )
        )
    return out


def _arithmetic() -> list[Case]:
    pol = base_policy()
    cap, threshold = int(pol.params["score_cap"]), int(pol.params["auto_approve_max_score"])
    bad = []
    for cid, _, p in all_cases():
        ev, comp = decide(p)
        raw = sum(r.score_delta for r in ev.rule_results if r.fired == "yes")
        fired_sev = {r.severity for r in ev.rule_results if r.fired == "yes"}
        unknown = any(r.fired == "unknown" for r in ev.rule_results)
        want = (
            "REJECTED"
            if "REJECT" in fired_sev
            else "MANUAL_REVIEW"
            if "REVIEW" in fired_sev or unknown or min(raw, cap) > threshold
            else "APPROVED"
        )
        if comp.uncapped_score != raw or comp.risk_score != min(raw, cap) or comp.status != want:
            bad.append(cid)
        if any(r.fired != "yes" and r.score_delta for r in ev.rule_results):
            bad.append(f"{cid}: points on an unfired rule")
    return [
        check(
            "precedence-and-score-arithmetic", not bad, f"disagree: {bad}", cases=len(all_cases())
        )
    ]


def _tamper(raw: bytes) -> dict[str, bytes]:
    lines = raw.decode().splitlines(keepends=True)
    ev = json.loads(lines[3])
    edited = lines.copy()
    edited[3] = json.dumps({**ev, "detail": {**ev["detail"], "n": 999}}, sort_keys=True) + "\n"
    rehashed = lines.copy()  # an edit that also recomputes its own hash breaks the next link
    from kyb.audit.chain import event_hash

    forged = {**ev, "action": "decision.emitted"}
    forged["hash"] = event_hash({k: v for k, v in forged.items() if k != "hash"})
    rehashed[3] = json.dumps(forged, sort_keys=True) + "\n"
    return {
        "edit-a-field": "".join(edited).encode(),
        "edit-and-rehash": "".join(rehashed).encode(),
        "delete-a-record": "".join(lines[:3] + lines[4:]).encode(),
        "reorder-two-records": "".join(lines[:3] + [lines[4], lines[3]] + lines[5:]).encode(),
        "insert-a-copy": "".join(lines[:4] + [lines[3]] + lines[4:]).encode(),
    }


def _audit() -> list[Case]:
    """A genuine chain, then one fresh attack fixture per forgery. No ledger is ever rewritten:
    each forged byte sequence is written once, as a new file in its own new workspace, beside
    a copy of the genuine checkpoints (an attacker who edits the log cannot edit those)."""
    out = []
    with tempfile.TemporaryDirectory(prefix="kyb-eval-audit-") as tmp:
        genuine = Path(tmp) / "genuine"
        log = AuditLog(genuine / "ledger" / "audit.jsonl")
        for i in range(8):
            log.append(
                "eval",
                "tool.call",
                input_hash=f"sha256:{i:064x}",
                output_hash=f"sha256:{i + 1:064x}",
                detail={"n": i},
            )
        write_checkpoint(genuine)
        out.append(check("chain-verifies", verify_workspace(genuine, at_boundary=True).ok))
        raw = log.path.read_bytes()
        lines = raw.decode().splitlines(keepends=True)
        forged = {**_tamper(raw), "truncate-tail": "".join(lines[:-2]).encode()}
        # the control: the genuine bytes in a fixture built the same way must verify, so a
        # detection below is the forgery's, not the fixture's
        for name, body in {"control-copy": raw, **forged}.items():
            attack = Path(tmp) / f"attack-{name}"
            shutil.copytree(genuine / "ledger" / "checkpoints", attack / "ledger" / "checkpoints")
            (attack / "ledger" / "audit.jsonl").write_bytes(body)
            ok = verify_workspace(attack).ok
            if name == "control-copy":
                out.append(check("control-copy-verifies", ok, "a faithful copy does not verify"))
            else:
                out.append(check(f"tamper-detected:{name}", not ok, "tampering not detected"))
        out.append(
            check(
                "genuine-untouched",
                log.path.read_bytes() == raw and verify_workspace(genuine, at_boundary=True).ok,
                "the genuine ledger changed during the suite",
            )
        )
    return out


def suite() -> Suite:
    s = Suite("1", "Contract & invariants", True, threshold="100%")
    s.cases += _records()
    s.cases += _arithmetic()
    s.cases += _audit()
    s.cases += _emitted()
    return s


def _emitted() -> list[Case]:
    from kyb.audit.chain import chain_events
    from kyb.contracts.kyb_decision_schema import KYBDecision
    from kyb.evals import agentpath as ap
    from kyb.evals.cases import variants

    records = {f"golden-{k}": d.record for k, d in ap.golden_decided().items()}
    records |= {f"e2e-{k}": d.record for k, d in ap.derived_decided().items()}
    records |= {v["id"]: ap.synthetic_decision(v, p, "healthy")[0] for v, p in variants()}
    bad = []
    for cid, rec in records.items():
        try:
            KYBDecision.model_validate(rec.kyb_decision.model_dump(mode="json"))
            b4.DecisionRecord.model_validate_json(rec.model_dump_json())
        except ValueError as exc:
            bad.append(f"{cid}: {str(exc)[:80]}")
    out = [
        check(
            "decisions-validate-KYBDecision",
            not bad,
            f"invalid: {bad}",
            decisions=len(records),
        )
    ]
    broken = []
    for letter, d in {
        **ap.golden_decided(),
        **{f"e2e-{k}": v for k, v in ap.derived_decided().items()},
    }.items():
        trail = [e.model_dump(mode="json") for e in d.record.kyb_decision.audit_trail]
        events = chain_events(d.ws)
        if (
            not trail
            or events[: len(trail)] != trail
            or not verify_workspace(d.ws, at_boundary=True).ok
        ):
            broken.append(letter)
    out.append(
        check(
            "decision-audit-trail-verifies",
            not broken,
            f"the audit_trail is not the verifying workspace chain for: {broken}",
            decisions=len(ap.golden_decided()) + len(ap.derived_decided()),
        )
    )
    return out
