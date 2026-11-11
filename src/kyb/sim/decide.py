"""Deciders: `decide(case) -> Decision(status, score, fired_rules)`.

- `LabelDecider` is the *oracle* (the label engine itself). It exists to test the harness, never
  to measure a decision system: any metric it produces is trivially perfect on false approvals.
- `FixtureDecider` feeds the case's facts into the real policy engine on main (`kyb.policy`).
- `L4Decider` (kyb.sim.l4) runs L4's real judges and `decision_record` on the case's facts, offline.
- `LiveDecider` (kyb.sim.live) replays recorded end-to-end pipeline runs on rendered packets.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from kyb.contracts import b4
from kyb.policy import compose, evaluate, load_policy, model_instances
from kyb.screening.screen import screen_factset
from kyb.synth.case import CaseSpec
from kyb.synth.facts import to_factset
from kyb.synth.label import label, watchlists

ROOT = Path(__file__).resolve().parents[3]
POLICY_PATH = ROOT / "examples/policy/compliance_rules.md"

Status = Literal["APPROVED", "REJECTED", "MANUAL_REVIEW"]


class Decision(BaseModel):
    model_config = ConfigDict(frozen=True)
    status: Status
    score: int
    fired_rules: tuple[str, ...]  # sorted rule ids (policy ids, e.g. "JUR-02", "MOD-01")
    # what an officer (or a Jev officer) is shown; empty for the label and fixture deciders
    reasons: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()  # cited evidence, "doc p.N: excerpt" (synthetic: the claim id)


class Decider(Protocol):
    name: str

    def decide(self, case: CaseSpec) -> Decision: ...


class LabelDecider:
    """ORACLE: returns the gold label. For harness tests only."""

    name = "label-oracle"

    def decide(self, case: CaseSpec) -> Decision:
        lab = label(case)
        return Decision(status=lab.status, score=lab.score, fired_rules=tuple(lab.rules))


@lru_cache(maxsize=1)
def _policy() -> b4.PolicyModel:
    return load_policy(POLICY_PATH)


def _answers(
    fs: b4.FactSet, sr: b4.ScreeningResult, pol: b4.PolicyModel
) -> list[b4.PredicateAnswer]:
    """Deterministic stand-ins for the model predicates, from the facts alone (no model call):
    `same_party` is true only for a corroborated, unconflicted candidate; a natural person who
    directs the business is true when one holds a control role or a resolved stake; a document
    typed proof_of_address is a proof of address."""
    cands = {c.candidate_id: c for c in sr.candidates}
    persons = {e.entity_id for e in fs.entities if e.kind == "person"}
    directs = any(r.holder in persons for r in fs.roles) or any(
        h.owner in persons for h in fs.holdings
    )
    out = []
    for inst in model_instances(fs, sr, pol):
        if inst.predicate == "same_party":
            ident = cands[inst.instance_id.removeprefix("sp-")].identifiers
            value = bool(ident.corroborating) and not ident.conflicting
        elif inst.predicate == "natural_person_directs_business":
            value = directs
        else:
            value = False
        out.append(
            b4.PredicateAnswer(
                instance_id=inst.instance_id,
                value=value,
                confidence=1.0,
                evidence=[],
                rationale="fixture answer from the case facts",
            )
        )
    return out


class FixtureDecider:
    """The real policy engine on the case's facts: CaseSpec → FactSet → screening → evaluate →
    compose. Model predicates get deterministic fixture answers (`_answers`); judges are not run."""

    name = "fixture-policy-engine"

    def decide(self, case: CaseSpec) -> Decision:
        fs, pol = to_factset(case), _policy()
        sr = screen_factset(
            fs,
            watchlists(),
            watchlist_version=watchlists()["version"],
            threshold=float(pol.params["name_similarity_threshold"]),
        )
        ev = evaluate(fs, sr, pol, _answers(fs, sr, pol))
        comp = compose(ev.rule_results, pol)
        return Decision(
            status=comp.status, score=comp.risk_score, fired_rules=tuple(sorted(comp.fired))
        )


class FaultyDecider:
    """STRESS HARNESS: wraps a decider and, with probability `rate` per case (seeded by the case
    id), approves a case the wrapped decider did not approve. It exists to exercise demotion and
    false-approval accounting; it models "a decider that sometimes errs", not any real one."""

    def __init__(self, base: Decider, rate: float = 0.15, seed: int = 0) -> None:
        self.base, self.rate, self.seed = base, rate, seed
        self.name = f"faulty({base.name},{rate})"

    def decide(self, case: CaseSpec) -> Decision:
        d = self.base.decide(case)
        h = int.from_bytes(
            hashlib.sha256(f"{self.seed}|{case.case_id}".encode()).digest()[:4], "big"
        )
        if d.status != "APPROVED" and h / 2**32 < self.rate:
            return d.model_copy(update={"status": "APPROVED"})
        return d
