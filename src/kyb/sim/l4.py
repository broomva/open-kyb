"""`L4Decider`: L4's real judges and decision assembly on a synthetic case, offline.

The case spec becomes b4-v1 inputs the way L4's own synthetic-variant test does it
(tests/test_decide.py): a FactSet whose every document is verified W2 and isolated, a
ScreeningResult from the real screener, and predicate answers that cite their evidence universe.
Then the real `predicate_judges` (healthy J-SUPPORT), `fact_judges` and `decision_record` (compose
plus decision judges) decide. No model call, no workspace.

Provenance is synthetic, and the decision says so (`provenance["inputs"]`). What this does NOT
cover: the reader and witness (the facts are stated by the spec, not read from a packet), the
witness-anomaly and fixture-input approval caps (they need a workspace), and a live J-SUPPORT (it
passes). Isolation is a property of the live harness; `lifecycle_reasons` is stubbed to "attested"
here and nowhere else.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from datetime import UTC, datetime
from functools import lru_cache
from unittest import mock

from kyb.contracts import b4
from kyb.decide.core import decision_record
from kyb.judges.core import (
    STRICT,
    fact_judges,
    live_rules,
    outcome,
    predicate_judges,
    profile_outcome,
)
from kyb.policy import evaluate, model_instances
from kyb.policy.rules import GAP_RULES
from kyb.screening.screen import screen_factset
from kyb.sim.decide import POLICY_PATH, Decision, _answers, _policy
from kyb.synth.case import CaseSpec
from kyb.synth.facts import to_factset
from kyb.synth.label import watchlists

PROVENANCE = {
    "inputs": "synthetic case spec: claims verified by construction, not read from a packet",
    "reader": "none",
    "witness": "none",
}
EMITTED_AT = datetime(2026, 10, 26, tzinfo=UTC)
MAX_EVIDENCE = 12


def verified_inputs(case: CaseSpec) -> tuple[b4.FactSet, b4.ScreeningResult, b4.PolicyModel]:
    pol = _policy()
    fs = to_factset(case)
    fs = fs.model_copy(
        update={
            "verification": [
                b4.DocVerification(doc=d.doc, assurance="W2", isolated=True) for d in fs.documents
            ]
        }
    )
    sr = screen_factset(
        fs,
        watchlists(),
        watchlist_version=watchlists()["version"],
        threshold=float(pol.params["name_similarity_threshold"]),
    )
    return fs, sr, pol


def _work_order(
    fs: b4.FactSet, instances: list[b4.PredicateInstance], pol: b4.PolicyModel
) -> b4.WorkOrder:
    need = {eid for i in instances for eid in i.evidence_universe}
    return b4.WorkOrder(
        work_order_id=f"wo-{fs.case}",
        case=fs.case,
        run=fs.run,
        fact_set_hash=fs.fact_set_hash,
        rules_version="2026.11-r1",
        catalog_version="synthetic",
        instances=instances,
        evidence=[e for e in fs.evidence if e.evidence_id in need],
        policy_text={},
    )


def record_for(
    case: CaseSpec,
    *,
    profile: str = STRICT,
    gaps: Callable[[b4.FactSet], b4.FactSet] | None = None,
) -> b4.DecisionRecord:
    """`gaps` perturbs the verified FactSet (a reader that missed or failed claims, which the
    spec-built facts never do); `profile` is the spec decision profile."""
    fs, sr, pol = verified_inputs(case)
    if gaps is not None:
        fs = gaps(fs)
    instances = model_instances(fs, sr, pol)
    universe = {i.instance_id: i.evidence_universe for i in instances}
    answers = [
        a.model_copy(update={"evidence": universe[a.instance_id]}) for a in _answers(fs, sr, pol)
    ]
    work = _work_order(fs, instances, pol)
    sources = {i.instance_id: {"who": {}, "work_order_id": work.work_order_id} for i in instances}
    with mock.patch("kyb.judges.core.lifecycle_reasons", return_value=[]):
        checked, judges = predicate_judges(
            fs,
            work,
            answers,
            who=None,
            events=[],
            sources=sources,
            support=lambda i, a, e: outcome("J-SUPPORT", i.instance_id),
        )
    live = None
    if profile != STRICT:
        live = live_rules(evaluate(fs, sr, pol, checked).rule_results, pol)
    judges = [*judges, *fact_judges(fs, live)]
    if profile != STRICT:
        judges.append(profile_outcome(profile, judges))
    return decision_record(
        fs,
        sr,
        pol,
        checked,
        judges,
        [],
        provenance=PROVENANCE if profile == STRICT else {**PROVENANCE, "decision_profile": profile},
        emitted_at=EMITTED_AT,
        audit_trail=[],
    )


MATERIAL_KINDS = tuple(sorted(GAP_RULES))


def gap_perturbation(
    share: float, seed: int = 11
) -> Callable[[CaseSpec], Callable[[b4.FactSet], b4.FactSet]]:
    """A deterministic reader-gap model: a `share` of cases get one gap, half an uncovered
    material fact (a random kind, on a random document) and half a failed claim."""

    def for_case(case: CaseSpec) -> Callable[[b4.FactSet], b4.FactSet]:
        rng = random.Random(f"{seed}:{case.case_id}")
        hit, uncovered, kind = rng.random() < share, rng.random() < 0.5, rng.choice(MATERIAL_KINDS)

        def apply(fs: b4.FactSet) -> b4.FactSet:
            if not hit:
                return fs
            doc = sorted(d.doc for d in fs.documents)[0]
            if uncovered:
                gap = b4.UncoveredFact(
                    block_id="b-0", kind=kind, subject=fs.applicant, attribute="x", value="v"
                )
                ver = [
                    v.model_copy(update={"uncovered": [gap], "findings": ["coverage_incomplete"]})
                    if v.doc == doc
                    else v
                    for v in fs.verification
                ]
                return fs.model_copy(update={"verification": ver})
            ref = b4.EvidenceRef(claim_id="c-failed-0", doc=doc, page=1, verified=False)
            return fs.model_copy(update={"unverified": [*fs.unverified, ref]})

        return apply

    return for_case


_ENTITY_ID = re.compile(r"\be-[a-z0-9][a-z0-9-]*")


def scrub(text: str) -> str:
    """Entity ids are slugs of names (`e-<company-name>`): a description must not carry them."""
    return _ENTITY_ID.sub("<entity>", text)


def decision_from(record: b4.DecisionRecord, *, names: bool = False) -> Decision:
    """The part of a DecisionRecord a simulation (or an officer) needs. `names=False` (synthetic
    cases, whose companies are real registry names) scrubs entity-id slugs; the sample's own
    fictional packets A, B and C keep them."""
    clean = (lambda t: t) if names else scrub
    fired = [r for r in record.rule_results if r.fired == "yes"]
    cites = []
    for r in fired:
        for ref in r.evidence:
            cites.append(f"{r.rule_id} {ref.claim_id} (doc {ref.doc} p.{ref.page})")
    reasons = [
        clean(f"{r.rule_id} [{r.severity}] +{r.score_delta}: {r.description}") for r in fired
    ]
    reasons += [clean(c.strip()) for c in record.kyb_decision.rationale.split(";") if c.strip()]
    return Decision(
        status=record.kyb_decision.status,
        score=record.kyb_decision.risk_score,
        fired_rules=tuple(sorted(r.rule_id for r in fired)),
        reasons=tuple(dict.fromkeys(reasons)),
        evidence=tuple(dict.fromkeys(cites))[:MAX_EVIDENCE],
    )


class L4Decider:
    """L4's judges + decision assembly on the case's facts (offline). See the module docstring."""

    name = "l4-decide-offline"

    def __init__(self, profile: str = STRICT, gaps: Callable | None = None) -> None:
        self.profile, self.gaps = profile, gaps
        self._cache: dict[str, Decision] = {}

    def decide(self, case: CaseSpec) -> Decision:
        if case.case_id not in self._cache:
            perturb = self.gaps(case) if self.gaps else None
            self._cache[case.case_id] = decision_from(
                record_for(case, profile=self.profile, gaps=perturb)
            )
        return self._cache[case.case_id]


@lru_cache(maxsize=1)
def policy_text() -> str:
    return POLICY_PATH.read_text(encoding="utf-8")
