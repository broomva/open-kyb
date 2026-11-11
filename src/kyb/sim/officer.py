"""Officers: who signs a decision.

- `SimOfficer` signs the gold label, with a configurable error rate, override rationale and latency.
- `JevOfficer` is the `full-auto-jev` experiment profile (spec): a Jev model resolves every
  MANUAL_REVIEW to APPROVE or REJECT, through the same recorded/replayed System One port as
  J-SUPPORT (`kyb.witness.jev.JevPort`). Offline runs use replay fixtures and make no call. Every
  Jev signature is marked `approved_via: "jev-auto"`.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict

from kyb.sim.decide import Decision, Status
from kyb.synth.case import CaseSpec
from kyb.synth.label import Label
from kyb.witness.jev import PROMPT_VERSION, JevPort, JevUnavailable

QUESTION_ID = "approve_is_correct"
STATUSES: tuple[Status, ...] = ("APPROVED", "REJECTED", "MANUAL_REVIEW")


class Signature(BaseModel):
    model_config = ConfigDict(frozen=True)
    status: Status  # MANUAL_REVIEW from a Jev officer means "unresolved: no recording / no answer"
    signer: Literal["officer", "jev"]
    approved_via: str  # "officer" | "jev-auto"
    latency_h: float = 0.0
    rationale: str | None = None  # an override's written rationale (policy §12)
    model: str | None = None
    request_hash: str | None = None
    replayed: bool | None = None
    probability: float | None = None  # Jev's Noul P(approve), as returned


class Officer(Protocol):
    name: str
    full_auto: bool  # True: signs every case without the autonomy dial (the full-auto-jev profile)

    def review(self, case: CaseSpec, proposal: Decision, gold: Label, key: str) -> Signature: ...


class SimOfficer:
    """The gold label with probability 1 - error_rate; otherwise a different status."""

    name = "sim-officer"
    full_auto = False

    def __init__(self, error_rate: float = 0.0, latency_h: float = 0.5, seed: int = 0) -> None:
        self.error_rate, self.latency_h, self.seed = error_rate, latency_h, seed

    def review(self, case: CaseSpec, proposal: Decision, gold: Label, key: str) -> Signature:
        rng = random.Random(f"officer/{self.seed}/{key}")
        status: Status = gold.status
        if rng.random() < self.error_rate:
            status = rng.choice([s for s in STATUSES if s != gold.status])
        why = None
        if status != proposal.status:
            why = (
                f"Officer correction: {proposal.status} → {status}; rules "
                f"{','.join(proposal.fired_rules) or 'none'}"
            )
        return Signature(
            status=status,
            signer="officer",
            approved_via="officer",
            latency_h=self.latency_h,
            rationale=why,
        )


def jev_body(case: CaseSpec, proposal: Decision) -> dict[str, Any]:
    """The System One request for one MANUAL_REVIEW: no names, only the case id and the decision.

    A decision that carries reasons or cited evidence (every L4 or live decision) also sends them
    with the policy text, and asks the live question: should this application be approved? A bare
    decision (the phase-1 fixtures) keeps its original, smaller request."""
    state: dict[str, Any] = {
        "case_id": case.case_id,
        "proposed_status": proposal.status,
        "risk_score": proposal.score,
        "fired_rules": list(proposal.fired_rules),
        "policy": "2026.11-r1",
    }
    if not (proposal.reasons or proposal.evidence):
        return {
            "model": "jev-latest",
            "state": state,
            "questions": {
                QUESTION_ID: {
                    "type": "noul",
                    "instructions": "Resolve this KYB MANUAL_REVIEW. Is APPROVE the correct "
                    "outcome under the cited policy rules?",
                    "criteria": {"true": "approve", "false": "reject"},
                }
            },
        }
    from kyb.sim.l4 import policy_text

    state["reasons"] = list(proposal.reasons)
    state["cited_evidence"] = list(proposal.evidence)
    state["policy_text"] = policy_text()
    return {
        "model": "jev-latest",
        "state": state,
        "questions": {
            QUESTION_ID: {
                "type": "noul",
                "instructions": "A KYB review ended in MANUAL_REVIEW. Using the compliance policy "
                "in `policy_text`, and the fired rules, reasons and cited evidence of the decision "
                "in this state, should this application be approved?",
                "criteria": {
                    "true": "The policy permits approval: no fired rule or reason requires "
                    "rejection or leaves a material question open.",
                    "false": "A fired rule, a conflict or an unresolved question means the "
                    "application should not be approved.",
                },
            }
        },
    }


class JevOfficer:
    """full-auto-jev. `port` is a JevPort; use mode="replay" offline (zero calls)."""

    name = "jev-officer"
    full_auto = True
    APPROVED_VIA = "jev-auto"

    def __init__(self, port: JevPort, threshold: float = 0.5) -> None:
        self.port, self.threshold = port, threshold

    def review(self, case: CaseSpec, proposal: Decision, gold: Label, key: str) -> Signature:
        if proposal.status != "MANUAL_REVIEW":  # nothing to resolve: Jev countersigns the proposal
            return Signature(status=proposal.status, signer="jev", approved_via=self.APPROVED_VIA)
        try:
            resp = self.port.ask(jev_body(case, proposal))
        except JevUnavailable as exc:
            return Signature(
                status="MANUAL_REVIEW",
                signer="jev",
                approved_via=self.APPROVED_VIA,
                rationale=f"jev unavailable: {exc}",
                request_hash=exc.request_hash,
            )
        raw = resp.answers.get(QUESTION_ID) or {}
        p = raw.get("noul") if isinstance(raw, dict) else None
        if isinstance(p, bool) or not isinstance(p, int | float):
            return Signature(
                status="MANUAL_REVIEW",
                signer="jev",
                approved_via=self.APPROVED_VIA,
                rationale="jev answer unreadable",
                model=resp.model,
                request_hash=resp.request_hash,
                replayed=resp.replayed,
            )
        status: Status = "APPROVED" if float(p) >= self.threshold else "REJECTED"
        return Signature(
            status=status,
            signer="jev",
            approved_via=self.APPROVED_VIA,
            model=resp.model,
            request_hash=resp.request_hash,
            replayed=resp.replayed,
            probability=float(p),
        )


class ScriptedJev:
    """A recording ADAPTER for fixtures, not a model: answers each question from the gold label with
    `error_rate`, so replay fixtures can be written offline. Its model id says so."""

    MODEL = "jev-fixture-1.0"

    def __init__(self, gold: dict[str, str], error_rate: float = 0.1, seed: int = 0) -> None:
        self.gold, self.error_rate, self.seed = gold, error_rate, seed
        self.calls = 0

    def ask(self, body: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        cid = body["state"]["case_id"]
        h = int.from_bytes(hashlib.sha256(f"{self.seed}|{cid}".encode()).digest()[:4], "big")
        wrong = h / 2**32 < self.error_rate
        approve = (self.gold[cid] == "APPROVED") != wrong
        return {
            "model": self.MODEL,
            "usage": {"fixture": True},
            "answers": {QUESTION_ID: {"type": "noul", "noul": 0.9 if approve else 0.1}},
        }


def record_fixture(
    root: Path,
    decisions: dict[str, tuple[CaseSpec, Decision]],
    gold: dict[str, str],
    error_rate: float = 0.1,
    seed: int = 0,
) -> int:
    """Write replay recordings for every MANUAL_REVIEW decision, offline, with `ScriptedJev`."""
    adapter = ScriptedJev(gold, error_rate, seed)
    port = JevPort(root, adapter, mode="auto")
    for case, dec in decisions.values():
        if dec.status == "MANUAL_REVIEW":
            port.ask(jev_body(case, dec))
    return adapter.calls


__all__ = [
    "JevOfficer",
    "Officer",
    "PROMPT_VERSION",
    "ScriptedJev",
    "Signature",
    "SimOfficer",
    "jev_body",
    "record_fixture",
]
