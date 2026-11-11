"""Checked workspace actions. Writes require the caller's workspace action lock."""

from __future__ import annotations

import fnmatch
import json
from datetime import UTC, datetime
from pathlib import Path

from kyb.audit.chain import AuditLog, chain_events
from kyb.facts.action import POLICY_REL, facts_rel, load_documents, load_facts
from kyb.loop.models import LoopBudget, LoopPlan
from kyb.loop.planner import plan_cycle
from kyb.ontology import workspace_ontology
from kyb.run.records import read_record, write_record
from kyb.util.fsutil import contained, safe_id
from kyb.util.hashing import sha256_bytes, sha256_json
from kyb.workspace.catalog import Catalog
from kyb.workspace.runs import list_runs, run_meta, safe_run_id, start_run


def _require(root: Path, name: str, rel: str | None = None) -> None:
    ontology = workspace_ontology(root)
    if rel is None:
        if name not in ontology.reads:
            raise PermissionError(f"workspace ontology needs upgrade for {name}")
        return
    action = ontology.actions.get(name)
    if (
        action is None
        or action.tier_out not in {"claimed", "derived"}
        or not any(fnmatch.fnmatchcase(rel, pattern) for pattern in action.writes)
    ):
        raise PermissionError(f"workspace ontology needs upgrade for {name}")


def _rel(case: str, plan_id: str) -> str:
    case = safe_id(case, "case id")
    if len(plan_id) != 64 or any(c not in "abcdef9876543210" for c in plan_id):
        raise ValueError("invalid plan id")
    return f"ledger/loops/{case}/{plan_id}.json"


def _allocation(case: str, plan_id: str) -> str:
    return _rel(case, plan_id)[:-5] + ".repair.json"


def _plan(root: Path, case: str, plan_id: str) -> LoopPlan:
    p = LoopPlan.model_validate(read_record(root, _rel(case, plan_id)))
    if (
        p.plan_id != plan_id
        or p.case != case
        or sha256_json(p.model_dump(mode="json", exclude={"plan_id"})).removeprefix("sha256:")
        != plan_id
    ):
        raise PermissionError("plan content or identity does not match")
    return p


def _chain(root: Path, case: str, plan: LoopPlan) -> list[LoopPlan]:
    seen = set()
    out = []
    p = plan
    while True:
        if p.plan_id in seen:
            raise PermissionError("cyclic loop lineage")
        seen.add(p.plan_id)
        source_facts = load_facts(root, case, p.source_run)
        if source_facts.fact_set_hash != p.fact_set_hash:
            raise PermissionError("predecessor facts changed")
        lineage = p.metrics.get("lineage", {})
        if (
            lineage.get("catalog_refs") != _inputs(root, case, p.source_run)
            or lineage.get("policy_hash") != sha256_bytes(contained(root, POLICY_REL).read_bytes())
            or lineage.get("as_of") != run_meta(root, case, p.source_run)["as_of"]
        ):
            raise PermissionError("predecessor lineage changed")
        out.append(p)
        if p.previous_plan is None:
            if p.attempt != 0:
                raise PermissionError("invalid initial attempt")
            break
        prev = _plan(root, case, p.previous_plan)
        allocation = read_record(root, _allocation(case, prev.plan_id))
        expected_allocation = {
            "version": "evidence-loop-v1",
            "case": case,
            "plan_id": prev.plan_id,
            "run": p.source_run,
            "source_run": prev.source_run,
            "as_of": prev.metrics["lineage"]["as_of"],
        }
        if allocation != expected_allocation or p.attempt != prev.attempt + 1:
            raise PermissionError("predecessor allocation or attempt mismatch")
        if run_meta(root, case, p.source_run) != {
            "run_id": p.source_run,
            "case": case,
            "as_of": expected_allocation["as_of"],
            "reason": f"evidence-loop:{prev.plan_id}",
        }:
            raise PermissionError("predecessor run mismatch")
        if p.metrics.get("budget") != prev.metrics.get("budget"):
            raise PermissionError("budget changed across lineage")
        p = prev
    return out


def _inputs(root: Path, case: str, run: str) -> list[str]:
    catalog = Catalog(root)
    rel = facts_rel(case, run)
    entry = catalog.get(rel)
    if entry is None:
        raise LookupError("facts not recorded")
    refs = [catalog.ref(rel), *entry["inputs"], catalog.ref(f"ledger/runs/{case}/{run}.json")]
    return sorted(set(refs))


def _current(
    root: Path, case: str, run: str, budget: LoopBudget, previous: LoopPlan | None = None
) -> LoopPlan:
    facts = load_facts(root, case, run)
    docs = load_documents(root, case, run)
    p = plan_cycle(facts, docs, budget=budget, previous=previous)
    data = p.model_dump(mode="json", exclude={"plan_id"})
    data["metrics"]["lineage"] = {
        "catalog_refs": _inputs(root, case, run),
        "policy_hash": sha256_bytes(contained(root, POLICY_REL).read_bytes()),
        "as_of": run_meta(root, case, run)["as_of"],
    }
    data["input_fingerprint"] = sha256_json(data["metrics"]["lineage"])
    return LoopPlan(plan_id=sha256_json(data).removeprefix("sha256:"), **data)


def inspect_loop(root: Path, case: str, run: str) -> dict:
    case, run = safe_id(case, "case id"), safe_run_id(run)
    _require(root, "inspect_loop")
    return _current(root, case, run, LoopBudget()).model_dump(mode="json")


def preview_loop_plan(
    root: Path,
    case: str,
    run: str,
    *,
    budget: LoopBudget | None = None,
    previous_plan: str | None = None,
) -> dict:
    """The exact admissible plan, including predecessor/budget checks, without a write."""
    case, run = safe_id(case, "case id"), safe_run_id(run)
    # A repair run already names its predecessor. Omitting the optional argument
    # must not turn it into a fresh budget or reset its attempt counter.
    reason = run_meta(root, case, run).get("reason", "")
    if reason.startswith("evidence-loop:"):
        allocated_from = reason.removeprefix("evidence-loop:")
        if previous_plan is not None and previous_plan != allocated_from:
            raise PermissionError("predecessor does not match repair run")
        previous_plan = allocated_from
    previous = _plan(root, case, previous_plan) if previous_plan else None
    if previous:
        _chain(root, case, previous)
        allocation = read_record(root, _allocation(case, previous.plan_id))
        if allocation.get("run") != run:
            raise PermissionError("source run does not follow predecessor allocation")
        inherited = LoopBudget.model_validate(previous.metrics["budget"])
        if budget is not None and budget != inherited:
            raise PermissionError("budget cannot change across attempts")
        budget = inherited
    p = _current(root, case, run, budget or LoopBudget(), previous)
    _require(root, "plan_loop", _rel(case, p.plan_id))
    return p.model_dump(mode="json")


def record_loop_plan(
    root: Path,
    case: str,
    run: str,
    *,
    budget: LoopBudget | None = None,
    previous_plan: str | None = None,
) -> dict:
    p = LoopPlan.model_validate(
        preview_loop_plan(root, case, run, budget=budget, previous_plan=previous_plan)
    )
    refs = _inputs(root, case, run)
    if p.previous_plan:
        refs += [
            Catalog(root).ref(_rel(case, p.previous_plan)),
            Catalog(root).ref(_allocation(case, p.previous_plan)),
        ]
    write_record(
        root,
        _rel(case, p.plan_id),
        p.model_dump(mode="json"),
        action="plan_loop",
        tier="derived",
        as_of=run_meta(root, case, run)["as_of"],
        inputs=refs,
    )
    return p.model_dump(mode="json")


def _elapsed(root: Path, case: str, initial: LoopPlan) -> float:
    rel = _rel(case, initial.plan_id)
    events = [
        e
        for e in chain_events(root)
        if e["action"] == "tool.call"
        and e["detail"].get("action") == "plan_loop"
        and e["detail"].get("record") == rel
    ]
    if len(events) != 1:
        raise PermissionError("initial plan audit receipt missing or ambiguous")
    return max(
        0.0,
        (
            datetime.now(UTC)
            - datetime.fromisoformat(events[0]["timestamp"].replace("Z", "+00:00"))
        ).total_seconds(),
    )


def start_loop_repair(root: Path, case: str, plan_id: str) -> dict:
    case = safe_id(case, "case id")
    p = _plan(root, case, plan_id)
    chain = _chain(root, case, p)
    budget = LoopBudget.model_validate(p.metrics["budget"])
    previous = chain[1] if len(chain) > 1 else None
    now = _current(root, case, p.source_run, budget, previous)
    if now.model_dump(mode="json") != p.model_dump(mode="json"):
        raise PermissionError("plan inputs or policy changed")
    allocation_rel = _allocation(case, plan_id)
    _require(root, "start_loop_repair", allocation_rel)
    catalog = Catalog(root)
    existing = catalog.get(allocation_rel)
    pending_path = contained(root, allocation_rel)
    if existing:
        allocation = read_record(root, allocation_rel)
        run = safe_run_id(allocation["run"])
    elif pending_path.exists():
        allocation = json.loads(pending_path.read_bytes())
        run = safe_run_id(allocation["run"])
        expected = {
            "version": "evidence-loop-v1",
            "case": case,
            "plan_id": plan_id,
            "run": run,
            "source_run": p.source_run,
            "as_of": p.metrics["lineage"]["as_of"],
        }
        if allocation != expected:
            raise PermissionError("interrupted allocation mismatch")
        existing_run = contained(root, f"ledger/runs/{case}/{run}.json")
        if existing_run.exists():
            expected_run = {
                "run_id": run,
                "case": case,
                "as_of": allocation["as_of"],
                "reason": f"evidence-loop:{plan_id}",
            }
            if run_meta(root, case, run) != expected_run:
                raise PermissionError("allocated run conflicts with an existing run")
        else:
            runs = list_runs(root, case)
            next_run = f"r{int(runs[-1][1:]) + 1 if runs else 1}"
            if next_run != run:
                raise PermissionError("allocated run no longer available")
        write_record(
            root,
            allocation_rel,
            allocation,
            action="start_loop_repair",
            tier="derived",
            as_of=allocation["as_of"],
            inputs=[catalog.ref(_rel(case, plan_id))],
        )
    else:
        if p.state != "repair":
            raise PermissionError(f"plan is terminal: {p.state}")
        if _elapsed(root, case, chain[-1]) >= budget.max_elapsed_seconds:
            raise PermissionError("elapsed budget exhausted")
        runs = list_runs(root, case)
        # Checked under the caller's workspace lock: a newer attempt (another repair or a fresh
        # review) supersedes this plan, and allocating from its source would fork the history.
        # An existing or interrupted allocation above stays recoverable.
        if runs and runs[-1] != p.source_run:
            raise PermissionError("plan source is no longer the latest attempt")
        run = f"r{int(runs[-1][1:]) + 1 if runs else 1}"
        allocation = {
            "version": "evidence-loop-v1",
            "case": case,
            "plan_id": plan_id,
            "run": run,
            "source_run": p.source_run,
            "as_of": p.metrics["lineage"]["as_of"],
        }
        write_record(
            root,
            allocation_rel,
            allocation,
            action="start_loop_repair",
            tier="derived",
            as_of=allocation["as_of"],
            inputs=[catalog.ref(_rel(case, plan_id))],
        )
    if allocation != {
        "version": "evidence-loop-v1",
        "case": case,
        "plan_id": plan_id,
        "run": run,
        "source_run": p.source_run,
        "as_of": p.metrics["lineage"]["as_of"],
    }:
        raise PermissionError("allocation mismatch")
    reason = f"evidence-loop:{plan_id}"
    run_rel = f"ledger/runs/{case}/{run}.json"
    path = contained(root, run_rel)
    if not path.exists():
        runs = list_runs(root, case)
        next_run = f"r{int(runs[-1][1:]) + 1 if runs else 1}"
        if next_run != run:
            raise PermissionError("allocated run no longer available")
        if start_run(root, case, as_of=allocation["as_of"], reason=reason) != run:
            raise PermissionError("unexpected allocated run")
    else:
        meta = run_meta(root, case, run)
        expected = {"run_id": run, "case": case, "as_of": allocation["as_of"], "reason": reason}
        if meta != expected:
            raise PermissionError("allocated run conflicts with an existing run")
        catalog.register(
            run_rel,
            tier="none",
            produced_by={"action": "start_run", "reason": reason},
            as_of=allocation["as_of"],
        )
        if not any(
            e["action"] == "run.started"
            and e["detail"].get("case") == case
            and e["detail"].get("run") == run
            for e in chain_events(root)
        ):
            AuditLog(root / "ledger/audit.jsonl").append(
                "runs",
                "run.started",
                input_hash=sha256_json({"case": case, "as_of": allocation["as_of"]}),
                output_hash=sha256_json(meta),
                detail={"case": case, "run": run, "reason": reason},
            )
    return {**allocation, "reread_all_documents": True}


def get_repair_targets(root: Path, case: str, run: str, doc: str) -> list[dict[str, str]]:
    """Return only controlled, document-scoped tags after validating repair lineage."""
    from kyb.read.plan import _documents

    case, run, doc = safe_id(case, "case id"), safe_run_id(run), safe_id(doc, "doc")
    if doc not in dict(_documents(root, case)):
        raise LookupError("document is not in the case manifest")
    meta = run_meta(root, case, run)
    reason = meta.get("reason", "")
    catalog = Catalog(root)
    run_rel = f"ledger/runs/{case}/{run}.json"
    run_entry = catalog.get(run_rel)
    if (
        not run_entry
        or sha256_bytes(contained(root, run_rel).read_bytes()) != run_entry["content_hash"]
    ):
        raise PermissionError("run receipt missing or changed")
    directory = contained(root, f"ledger/loops/{case}")
    allocations = []
    if directory.exists():
        for path in directory.glob("*.repair.json"):
            rel = path.relative_to(root).as_posix()
            if catalog.get(rel) is None:
                # An incomplete allocation cannot be safely treated as a normal run.
                raise PermissionError("uncatalogued repair allocation")
            record = read_record(root, rel)
            if record.get("run") == run:
                allocations.append((rel, record))
    if not reason.startswith("evidence-loop:"):
        if allocations:
            raise PermissionError("conflicting repair allocation")
        return []
    plan_id = reason.removeprefix("evidence-loop:")
    expected_rel = _allocation(case, plan_id)
    if len(allocations) != 1 or allocations[0][0] != expected_rel:
        raise PermissionError("missing or conflicting repair allocation")
    allocation = allocations[0][1]
    plan = _plan(root, case, plan_id)
    chain = _chain(root, case, plan)
    expected = {
        "version": "evidence-loop-v1",
        "case": case,
        "plan_id": plan_id,
        "run": run,
        "source_run": plan.source_run,
        "as_of": plan.metrics["lineage"]["as_of"],
    }
    if allocation != expected or meta != {
        "run_id": run,
        "case": case,
        "as_of": expected["as_of"],
        "reason": reason,
    }:
        raise PermissionError("repair allocation does not match run")
    for rel, action_name in [
        (expected_rel, "start_loop_repair"),
        (_rel(case, plan_id), "plan_loop"),
    ]:
        entry = catalog.get(rel)
        if not entry or entry.get("produced_by", {}).get("action") != action_name:
            raise PermissionError("invalid repair receipt")
        matches = [
            e
            for e in chain_events(root)
            if e["action"] == "tool.call"
            and e["detail"].get("action") == action_name
            and e["detail"].get("record") == rel
            and e["output_hash"] == entry["content_hash"]
        ]
        if len(matches) != 1:
            raise PermissionError("missing or ambiguous repair audit receipt")
    previous = chain[1] if len(chain) > 1 else None
    now = _current(
        root, case, plan.source_run, LoopBudget.model_validate(plan.metrics["budget"]), previous
    )
    if now != plan or plan.state != "repair":
        raise PermissionError("repair source or plan changed")
    field_map = {
        "jurisdiction": "jurisdiction",
        "controller": "controller_disclosure",
        "registered_ownership": "registered_ownership",
        "document_date": "document_date",
        "address": "address",
        "officer_role": "officer_role",
    }
    kind_map = {
        "jurisdiction": "missing_field",
        "uncovered": "uncovered",
        "failed_claim": "verification_failed",
        "identity": "identity_unresolved",
        "conflict": "contradiction",
    }
    rows = {
        (field_map.get(g.field, "material_facts"), kind_map.get(g.kind, "uncovered"))
        for g in plan.gaps
        if g.repairable and doc in g.doc_ids
    }
    return [{"field": field, "kind": kind} for field, kind in sorted(rows)]
