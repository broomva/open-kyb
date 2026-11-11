"""Fixture-first runner (spec). Fixtures are explicit CLI inputs, never a live fallback.

Each policy/work-order revision is immutable. The run anchor pins facts and screening;
content-addressed siblings permit a policy edit without replacing the original work order.
"""

from __future__ import annotations

from pathlib import Path

from kyb.contracts.b4 import (
    FACTS_REL,
    PREDICATES_REL,
    SCREENING_REL,
    AmendmentEffectiveArgs,
    FactSet,
    PredicateInstance,
    ScreeningResult,
    WorkOrder,
)
from kyb.policy import evaluate, model_instances, version_id
from kyb.policy.store import record_policy
from kyb.run.records import read_record, write_record
from kyb.util.fsutil import safe_id
from kyb.util.hashing import canonical_json, sha256_json
from kyb.workspace.catalog import Catalog
from kyb.workspace.runs import safe_run_id

CATALOG_VERSION = "predicates-v1"


def facts_for_run(root: Path, case: str, run: str, fixture: Path | None = None) -> FactSet:
    if fixture is None:
        from kyb.facts.action import derive_facts, load_facts

        derive_facts(root, case, run)
        return load_facts(root, case, run)
    facts = FactSet.model_validate_json(fixture.read_bytes())
    if (facts.case, facts.run) != (case, run):
        raise ValueError("fixture facts target another case/run")
    if sha256_json(facts.model_dump(mode="json", exclude={"fact_set_hash"})) != facts.fact_set_hash:
        raise ValueError("fixture fact_set_hash differs from its content")
    write_record(
        root,
        FACTS_REL.format(case=case, run=run),
        facts.model_dump(mode="json"),
        action="fixture_facts",
        tier="derived",
        as_of=facts.as_of,
    )
    return facts


def screen_for_run(root: Path, facts: FactSet, fixture: Path | None = None) -> ScreeningResult:
    if fixture is None:
        from kyb.screening.action import default_watchlist, screen_case

        watchlist = default_watchlist(root)
        if watchlist is None:
            raise LookupError("policy/watchlists.json is missing")
        return screen_case(root, facts.case, facts.run, watchlist)[0]
    screening = ScreeningResult.model_validate_json(fixture.read_bytes())
    if (screening.case, screening.run, screening.fact_set_hash) != (
        facts.case,
        facts.run,
        facts.fact_set_hash,
    ):
        raise ValueError("fixture screening targets different facts")
    write_record(
        root,
        SCREENING_REL.format(case=facts.case, run=facts.run),
        screening.model_dump(mode="json"),
        action="fixture_screen",
        tier="derived",
        as_of=facts.as_of,
        inputs=[Catalog(root).ref(FACTS_REL.format(case=facts.case, run=facts.run))],
    )
    return screening


def amendment_claims(root: Path, facts: FactSet) -> frozenset[str]:
    """spec: flags belong to cited claims, not the frozen Holding contract."""
    verified = {e.evidence_id for e in facts.evidence if e.verified}
    marked = set()
    for doc in facts.documents:
        rel = f"silver/claims/{facts.case}/{facts.run}/{doc.doc}.json"
        if not Catalog(root).get(rel):
            continue
        for claim in read_record(root, rel)["claims"]:
            flags = {str(f).lower().replace("_", "-") for f in claim.get("flags", [])}
            if claim["claim_id"] in verified and flags & {
                "handwriting",
                "handwritten",
                "struck-through",
                "pending",
            }:
                marked.add(claim["claim_id"])
    return frozenset(marked)


def build_work_order(
    facts: FactSet,
    screening: ScreeningResult,
    policy,
    *,
    flagged_claims: frozenset[str] = frozenset(),
) -> WorkOrder:
    # L3 supplies the same ids consumed by evaluate(), including sp-<candidate_id>.
    instances = model_instances(facts, screening, policy)
    pairs = {
        (h.owner, h.owned)
        for h in facts.holdings
        if h.status in ("struck", "pending")
        or any(r.claim_id in flagged_claims for r in h.evidence)
    }
    # Include the competing/current observation of each affected holding as context too.
    amended = sorted(
        [h for h in facts.holdings if (h.owner, h.owned) in pairs], key=lambda h: h.holding_id
    )
    if amended:
        instances.append(
            PredicateInstance(
                instance_id="amendments",
                predicate="amendment_effective",
                kind="model",
                args=AmendmentEffectiveArgs(
                    holding_ids=[h.holding_id for h in amended], holdings=amended
                ),
                evidence_universe=sorted(
                    {r.claim_id for h in amended for r in h.evidence if r.verified}
                ),
            )
        )
    instances.sort(key=lambda i: i.instance_id)
    ids = {eid for i in instances for eid in i.evidence_universe}
    evidence = sorted(
        [e for e in facts.evidence if e.evidence_id in ids and e.verified],
        key=lambda e: e.evidence_id,
    )
    if ids != {e.evidence_id for e in evidence}:
        raise ValueError("work order evidence universe does not resolve to verified evidence")
    fields = dict(
        case=facts.case,
        run=facts.run,
        fact_set_hash=facts.fact_set_hash,
        rules_version=version_id(policy),
        catalog_version=CATALOG_VERSION,
        instances=instances,
        evidence=evidence,
        policy_text={r.rule_id: r.text for r in policy.rules},
    )
    work = WorkOrder(work_order_id="pending", **fields)
    return work.model_copy(
        update={
            "work_order_id": "wo-"
            + sha256_json(work.model_dump(mode="json", exclude={"work_order_id"})).removeprefix(
                "sha256:"
            )
        }
    )


def work_order_rel(work_order_id: str) -> str:
    return f"gold/predicates/work-orders/{safe_id(work_order_id, 'work order id')}.json"


def load_work_order(root: Path, work_order_id: str) -> WorkOrder:
    # JSON-mode validation preserves typed Decimal args in b4's typed-args | code-dict union.
    return WorkOrder.model_validate_json(
        canonical_json(read_record(root, work_order_rel(work_order_id))["work_order"])
    )


def run_case(
    root: Path,
    case: str,
    run: str,
    *,
    facts_fixture: Path | None = None,
    screening_fixture: Path | None = None,
) -> dict:
    case, run = safe_id(case, "case id"), safe_run_id(run)
    if (facts_fixture is None) != (screening_fixture is None):
        raise ValueError("both facts and screening fixtures are required together")
    facts = facts_for_run(root, case, run, facts_fixture)
    screening = screen_for_run(root, facts, screening_fixture)
    policy_rel, policy = record_policy(root, as_of=facts.as_of)
    evaluation = evaluate(facts, screening, policy)
    work = build_work_order(facts, screening, policy, flagged_claims=amendment_claims(root, facts))
    catalog = Catalog(root)
    inputs = [catalog.ref(rel.format(case=case, run=run)) for rel in (FACTS_REL, SCREENING_REL)]
    anchor = PREDICATES_REL.format(case=case, run=run)
    write_record(
        root,
        anchor,
        {"case": case, "run": run, "fact_set_hash": facts.fact_set_hash},
        action="eval_predicates",
        tier="derived",
        as_of=facts.as_of,
        inputs=inputs,
    )
    rel = work_order_rel(work.work_order_id)
    write_record(
        root,
        rel,
        {
            "work_order": work.model_dump(mode="json"),
            "evaluation": evaluation.model_dump(mode="json"),
        },
        action="eval_predicates",
        tier="derived",
        as_of=facts.as_of,
        inputs=[*inputs, catalog.ref(policy_rel)],
    )
    return {
        "case": case,
        "run": run,
        "work_order_id": work.work_order_id,
        "recorded": rel,
        "instances": len(work.instances),
        "launch": f"kyb-predicates {work.work_order_id}",
    }
