"""Bounded offline checks of loop planning and workspace allocation."""

from dataclasses import replace
from pathlib import Path

import pytest

from kyb.contracts.b4 import DocVerification, Entity, EvidenceItem, FactSet, Operations, TriFact
from kyb.facts.derive import DocInput
from kyb.loop import action
from kyb.loop.models import LoopBudget
from kyb.loop.planner import plan_cycle
from kyb.workspace.catalog import Catalog
from kyb.workspace.runs import list_runs, start_run


def sample():
    entity = Entity(entity_id="e-a", kind="company", names=["Acme"], evidence=[])
    unknown = TriFact(state="unknown")
    facts = FactSet(
        case="case",
        run="r1",
        as_of="2026-01-01",
        fact_set_hash="a" * 64,
        applicant="e-a",
        entities=[entity],
        holdings=[],
        roles=[],
        documents=[],
        conflicts=[],
        operations=Operations(
            distinct_operating_address=unknown,
            bank_reference_or_financials=unknown,
            employees_or_website=unknown,
        ),
        evidence=[],
        verification=[DocVerification(doc="01", assurance="W2", isolated=True)],
    )
    claim = {
        "claim_id": "c-1-a",
        "subject": "Acme",
        "type": "identity",
        "attribute": "reg_no",
        "value": "1",
        "excerpt": "1",
    }
    doc = DocInput(
        doc="01",
        document_name="a.pdf",
        claims=[claim],
        verdicts={"c-1-a": False},
        assurance="W2",
        isolated=True,
    )
    return facts, [doc]


def test_duplicate_claim_is_not_progress_and_new_verified_input_is():
    facts, docs = sample()
    first = plan_cycle(facts, docs, budget=LoopBudget())
    duplicate = {**docs[0].claims[0], "claim_id": "c-1-b", "excerpt": "broader 1"}
    same = [
        replace(
            docs[0], claims=[*docs[0].claims, duplicate], verdicts={"c-1-a": False, "c-1-b": False}
        )
    ]
    second = plan_cycle(facts, same, budget=LoopBudget(), previous=first)
    assert second.state == "no_progress"
    new = {
        "claim_id": "c-1-c",
        "subject": "Acme",
        "type": "identity",
        "attribute": "incorporated",
        "value": "2020",
        "excerpt": "2020",
    }
    changed = [
        replace(
            same[0], claims=[*same[0].claims, new], verdicts={**same[0].verdicts, "c-1-c": True}
        )
    ]
    evidence = EvidenceItem(
        evidence_id="c-1-c",
        doc="01",
        document_name="a.pdf",
        page=1,
        bbox=(1, 1, 2, 2),
        excerpt="2020",
        verified=True,
        field="entity[e-a].incorporated",
    )
    changed_facts = facts.model_copy(update={"evidence": [evidence]})
    assert plan_cycle(changed_facts, changed, budget=LoopBudget(), previous=first).state == "repair"


def test_budget_unknown_and_zero_attempts():
    facts, docs = sample()
    assert plan_cycle(facts, docs, budget=LoopBudget(max_tokens=1)).state == "blocked"
    assert plan_cycle(facts, docs, budget=LoopBudget(max_attempts=0)).state == "budget_exhausted"


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch):
    facts, docs = sample()
    root = tmp_path
    (root / "policy").mkdir()
    (root / "policy/compliance_rules.md").write_text("policy")
    run = start_run(root, "case", as_of="2026-01-01", reason="test")
    assert run == "r1"
    path = root / "gold/facts/case/r1.json"
    path.parent.mkdir(parents=True)
    path.write_text("facts")
    Catalog(root).register(
        "gold/facts/case/r1.json",
        tier="derived",
        as_of="2026-01-01",
        produced_by={"action": "test"},
    )
    monkeypatch.setattr(action, "_require", lambda *_: None)
    monkeypatch.setattr(action, "load_facts", lambda *_: facts)
    monkeypatch.setattr(action, "load_documents", lambda *_: docs)
    return root


def test_allocation_retry_and_stale_refusal(workspace, monkeypatch):
    root = workspace
    p = action.record_loop_plan(root, "case", "r1")
    assert p == action.record_loop_plan(root, "case", "r1")
    real = action.start_run

    def interrupted(*args, **kwargs):
        real(*args, **kwargs)
        raise OSError("interruption after start")

    monkeypatch.setattr(action, "start_run", interrupted)
    with pytest.raises(OSError):
        action.start_loop_repair(root, "case", p["plan_id"])
    assert list_runs(root, "case") == ["r1", "r2"]
    monkeypatch.setattr(action, "start_run", real)
    result = action.start_loop_repair(root, "case", p["plan_id"])
    before = (root / "ledger/audit.jsonl").read_bytes()
    assert action.start_loop_repair(root, "case", p["plan_id"]) == result
    assert (root / "ledger/audit.jsonl").read_bytes() == before
    assert list_runs(root, "case") == ["r1", "r2"]
    (root / "policy/compliance_rules.md").write_text("changed")
    with pytest.raises(PermissionError, match="changed"):
        action.start_loop_repair(root, "case", p["plan_id"])
    assert (root / "ledger/audit.jsonl").read_bytes() == before
    with pytest.raises((ValueError, LookupError)):
        action.start_loop_repair(root, "../case", p["plan_id"])


def test_refusals_leave_audit_unchanged(workspace, monkeypatch):
    root = workspace
    p = action.record_loop_plan(root, "case", "r1")
    before = (root / "ledger/audit.jsonl").read_bytes()
    monkeypatch.setattr(
        action, "load_facts", lambda *_: (_ for _ in ()).throw(PermissionError("assurance revoked"))
    )
    with pytest.raises(PermissionError, match="assurance revoked"):
        action.start_loop_repair(root, "case", p["plan_id"])
    assert (root / "ledger/audit.jsonl").read_bytes() == before
    assert list_runs(root, "case") == ["r1"]


def test_interrupted_run_audit_finishes_once(workspace, monkeypatch):
    from kyb.audit.chain import AuditLog

    root = workspace
    p = action.record_loop_plan(root, "case", "r1")
    real = AuditLog.append

    def interrupted(self, component, event, *args, **kwargs):
        if event == "run.started":
            raise OSError("after run publication")
        return real(self, component, event, *args, **kwargs)

    monkeypatch.setattr(AuditLog, "append", interrupted)
    with pytest.raises(OSError):
        action.start_loop_repair(root, "case", p["plan_id"])
    monkeypatch.setattr(AuditLog, "append", real)
    assert action.start_loop_repair(root, "case", p["plan_id"])["run"] == "r2"
    before = (root / "ledger/audit.jsonl").read_bytes()
    action.start_loop_repair(root, "case", p["plan_id"])
    assert (root / "ledger/audit.jsonl").read_bytes() == before
    assert list_runs(root, "case") == ["r1", "r2"]


def test_nondisclosure_and_pending_are_not_promoted():
    from kyb.contracts.b4 import Holding

    facts, docs = sample()
    claim = {
        "claim_id": "c-1-n",
        "subject": "Acme",
        "type": "control",
        "attribute": "controller",
        "value": "controller will be disclosed on request",
        "excerpt": "controller will be disclosed on request",
    }
    docs = [replace(docs[0], claims=[claim], verdicts={"c-1-n": True})]
    facts = facts.model_copy(
        update={
            "holdings": [
                Holding(holding_id="h1", owner="e-a", owned="e-a", status="pending", evidence=[])
            ]
        }
    )
    plan = plan_cycle(facts, docs, budget=LoopBudget())
    assert any(g.kind == "nondisclosure" and not g.repairable for g in plan.gaps)
    assert plan.metrics["holding_statuses"]["pending"] == 1


def test_tampered_plan_and_cross_case_refuse_without_writes(workspace):
    root = workspace
    p = action.record_loop_plan(root, "case", "r1")
    before = (root / "ledger/audit.jsonl").read_bytes()
    with pytest.raises(LookupError):
        action.start_loop_repair(root, "other", p["plan_id"])
    path = root / action._rel("case", p["plan_id"])
    path.write_bytes(path.read_bytes().replace(b'"attempt":0', b'"attempt":1'))
    with pytest.raises(PermissionError):
        action.start_loop_repair(root, "case", p["plan_id"])
    assert (root / "ledger/audit.jsonl").read_bytes() == before
    assert list_runs(root, "case") == ["r1"]


def test_old_ontology_refused(tmp_path):
    from kyb.ontology import template_bytes

    (tmp_path / "ontology").mkdir()
    import re
    import tomllib

    legacy = tomllib.loads(template_bytes().decode())
    assert "plan_loop" in legacy["actions"]
    old = template_bytes().decode().splitlines()
    old = [
        line
        for line in old
        if not re.match(r"^(plan_loop|start_loop_repair|inspect_loop)\s*=", line)
    ]
    assert "plan_loop" not in tomllib.loads("\n".join(old))["actions"]
    (tmp_path / "ontology/kyb.ontology.toml").write_text("\n".join(old) + "\n")
    with pytest.raises(PermissionError, match="upgrade"):
        action._require(tmp_path, "plan_loop", "ledger/loops/case/plan.json")


def test_nested_usage_and_terminal_precedence():
    facts, docs = sample()
    first = plan_cycle(facts, docs, budget=LoopBudget())
    unknown = {"tokens": {"status": "unknown", "provenance": "synthetic"}}
    assert (
        plan_cycle(facts, docs, budget=LoopBudget(max_tokens=1), usage=unknown).state == "blocked"
    )
    measured = {"tokens": {"status": "measured", "value": 2, "provenance": "synthetic"}}
    assert (
        plan_cycle(
            facts, docs, budget=LoopBudget(max_tokens=1), previous=first, usage=measured
        ).state
        == "budget_exhausted"
    )
    revoked = facts.model_copy(
        update={
            "verification": [
                facts.verification[0].model_copy(update={"isolated": False, "assurance": "W1"})
            ]
        }
    )
    assert plan_cycle(revoked, docs, budget=LoopBudget()).state == "blocked"


def test_repair_targets_scoped_and_reject_tampered_allocation(workspace, monkeypatch):
    import kyb.read.plan as read_plan_module

    root = workspace
    monkeypatch.setattr(read_plan_module, "_documents", lambda *_: [("01", {}), ("02", {})])
    assert action.get_repair_targets(root, "case", "r1", "01") == []
    plan = action.record_loop_plan(root, "case", "r1")
    run = action.start_loop_repair(root, "case", plan["plan_id"])["run"]
    assert action.get_repair_targets(root, "case", run, "01") == [
        {"field": "material_facts", "kind": "verification_failed"}
    ]
    assert action.get_repair_targets(root, "case", run, "02") == []
    with pytest.raises(LookupError):
        action.get_repair_targets(root, "case", run, "03")
    path = root / action._allocation("case", plan["plan_id"])
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(PermissionError):
        action.get_repair_targets(root, "case", run, "01")


def test_repair_run_inherits_predecessor_when_argument_is_omitted(workspace, monkeypatch):
    root = workspace
    first = action.record_loop_plan(root, "case", "r1", budget=LoopBudget(max_attempts=1))
    action.start_loop_repair(root, "case", first["plan_id"])
    facts, _ = sample()
    monkeypatch.setattr(
        action, "load_facts", lambda root, case, run: facts.model_copy(update={"run": run})
    )
    rel = "gold/facts/case/r2.json"
    path = root / rel
    path.write_text("second run facts")
    Catalog(root).register(rel, tier="derived", as_of="2026-01-01", produced_by={"action": "test"})
    second = action.record_loop_plan(root, "case", "r2")
    assert second["previous_plan"] == first["plan_id"]
    assert second["attempt"] == 1 and second["state"] == "budget_exhausted"
    with pytest.raises(PermissionError, match="budget cannot change"):
        action.record_loop_plan(root, "case", "r2", budget=LoopBudget(max_attempts=99))
    with pytest.raises(PermissionError, match="predecessor does not match"):
        action.record_loop_plan(root, "case", "r2", previous_plan="0" * 64)
