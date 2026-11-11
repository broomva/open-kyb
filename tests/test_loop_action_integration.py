"""Real PDF/catalog/verification/action boundaries; only the inference adapter is a double."""

import json
import shutil
from pathlib import Path

import pytest
from test_witness import CASE, FakeWitness, _attested_reader, _claim, crop
from typer.testing import CliRunner

from kyb.audit.chain import verify_workspace, workspace_action
from kyb.cli import app
from kyb.docintel.ingest import ingest_packet
from kyb.docintel.pages import build_pages
from kyb.facts.action import derive_facts, load_facts
from kyb.loop import action
from kyb.read.plan import read_plan
from kyb.util.hashing import sha256_bytes
from kyb.witness.layout import layout_case
from kyb.witness.port import Witness
from kyb.witness.verify import verify_case
from kyb.workspace import guard
from kyb.workspace.init import init_workspace
from kyb.workspace.runs import list_runs, start_run

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "examples"


@pytest.fixture
def real_workspace(tmp_path):
    base = tmp_path
    packet = base / "packet"
    packet.mkdir()
    shutil.copyfile(ASSETS / "packets" / CASE / "05_acta_junta_SCAN.pdf", packet / "05.pdf")
    root = base / "ws"
    init_workspace(root, tenant="loop-test", policy_src=ASSETS / "policy/compliance_rules.md")
    with workspace_action(root):
        ingest_packet(root, packet, as_of="2026-10-25", case=CASE)
        start_run(root, CASE, as_of="2026-10-25", reason="test")
        build_pages(root, CASE, "r1")
        layout_case(root, CASE, "r1", Witness(root, FakeWitness()))
    plans = {e["doc"]: e for e in read_plan(root, CASE, "r1")}
    payload = {
        "case": CASE,
        "run": "r1",
        "doc": "05",
        "doc_sha256": plans["05"]["doc_sha256"],
        "claims": [
            _claim(
                "c-B05-03",
                "Inversiones Altiplano, S.A.",
                {"pct": 25},
                "Inversiones Altiplano, S.A. 25%",
                [1710, 760, 2300, 835],
            )
        ],
    }
    _attested_reader(root, plans, payload)
    witness = FakeWitness(
        [crop("Inversiones Altiplano, S.A. 25%", "Inversiones Altiplano, S.A.", "25%")],
        model="gemini-3.1-pro-preview",
    )
    with workspace_action(root):
        verify_case(root, CASE, "r1", Witness(root, witness))
        facts, _, _ = derive_facts(root, CASE, "r1")
    assert facts.verification[0].assurance == "W2" and facts.verification[0].isolated
    assert any(e.verified for e in facts.evidence)
    assert load_facts(root, CASE, "r1") == facts
    assert verify_workspace(root).ok
    return root


def snapshot(root):
    return {
        p.relative_to(root).as_posix(): sha256_bytes(p.read_bytes())
        for p in root.rglob("*")
        if p.is_file() and not p.name.endswith(".lock")
    }


def allocate(root):
    with workspace_action(root):
        plan = action.record_loop_plan(root, CASE, "r1")
        assert plan["state"] == "repair"
        allocation = action.start_loop_repair(root, CASE, plan["plan_id"])
    return plan, allocation


def test_real_actions_are_idempotent_and_append_only(real_workspace):
    root = real_workspace
    prefix = (root / "ledger/audit.jsonl").read_bytes()
    plan, allocation = allocate(root)
    assert allocation["run"] == "r2" and allocation["reread_all_documents"]
    assert (root / "ledger/audit.jsonl").read_bytes().startswith(prefix)
    before = snapshot(root)
    with workspace_action(root):
        assert action.record_loop_plan(root, CASE, "r1") == plan
        assert action.start_loop_repair(root, CASE, plan["plan_id"]) == allocation
    assert snapshot(root) == before
    assert list_runs(root, CASE) == ["r1", "r2"]
    assert not (root / f"silver/claims/{CASE}/r2").exists()
    assert verify_workspace(root).ok


def test_real_targets_are_controlled_and_read_only(real_workspace):
    root = real_workspace
    _, allocation = allocate(root)
    before = snapshot(root)
    assert action.get_repair_targets(root, CASE, "r1", "05") == []
    targets = action.get_repair_targets(root, CASE, allocation["run"], "05")
    assert targets
    for row in targets:
        assert set(row) == {"field", "kind"}
        assert row["field"] in {
            "jurisdiction",
            "registered_ownership",
            "controller_disclosure",
            "document_date",
            "address",
            "officer_role",
            "material_facts",
        }
        assert row["kind"] in {
            "missing_field",
            "uncovered",
            "verification_failed",
            "identity_unresolved",
            "contradiction",
        }
    encoded = json.dumps(targets)
    assert all(s not in encoded for s in ["Ortega", "Altiplano", "25%", "c-B05", "Reader:"])
    assert snapshot(root) == before


@pytest.mark.parametrize("foreign", ["case", "doc"])
def test_real_targets_refuse_foreign_scope_without_writes(real_workspace, foreign):
    root = real_workspace
    allocate(root)
    before = snapshot(root)
    with pytest.raises((LookupError, PermissionError, FileNotFoundError)):
        action.get_repair_targets(
            root,
            "foreign" if foreign == "case" else CASE,
            "r2",
            "foreign" if foreign == "doc" else "05",
        )
    assert snapshot(root) == before


@pytest.mark.parametrize("record", ["plan", "allocation"])
def test_real_targets_refuse_changed_record_without_writes(real_workspace, record):
    root = real_workspace
    plan, _ = allocate(root)
    rel = (
        action._rel(CASE, plan["plan_id"])
        if record == "plan"
        else action._allocation(CASE, plan["plan_id"])
    )
    path = root / rel
    data = json.loads(path.read_bytes())
    data["source_run"] = "r999"
    path.write_text(json.dumps(data))
    before = snapshot(root)
    with pytest.raises(PermissionError):
        action.get_repair_targets(root, CASE, "r2", "05")
    assert snapshot(root) == before


def test_real_targets_refuse_revoked_assurance(real_workspace):
    root = real_workspace
    allocate(root)
    token = guard.HOOK_SOURCE.set("workspace")
    try:
        guard.append_hook(
            {
                "cwd": str(root),
                "session_id": "s1",
                "tool_name": "Read",
                "tool_input": {},
                "tool_use_id": "late-registration",
            }
        )
    finally:
        guard.HOOK_SOURCE.reset(token)
    before = snapshot(root)
    with pytest.raises(PermissionError):
        action.get_repair_targets(root, CASE, "r2", "05")
    assert snapshot(root) == before


def test_real_interrupted_allocation_recovers_once(real_workspace, monkeypatch):
    root = real_workspace
    with workspace_action(root):
        plan = action.record_loop_plan(root, CASE, "r1")
    real = action.start_run

    def interrupted(*args, **kwargs):
        real(*args, **kwargs)
        raise OSError("fault after published run")

    monkeypatch.setattr(action, "start_run", interrupted)
    with pytest.raises(OSError), workspace_action(root):
        action.start_loop_repair(root, CASE, plan["plan_id"])
    monkeypatch.setattr(action, "start_run", real)
    with workspace_action(root):
        assert action.start_loop_repair(root, CASE, plan["plan_id"])["run"] == "r2"
    assert list_runs(root, CASE) == ["r1", "r2"]
    assert verify_workspace(root).ok


def test_real_cli_mcp_read_plan_targets_agree(real_workspace, monkeypatch):
    from kyb import mcp_server

    root = real_workspace
    allocate(root)
    with workspace_action(root):
        build_pages(root, CASE, "r2")
        layout_case(root, CASE, "r2", Witness(root, FakeWitness()))
    monkeypatch.setenv("KYB_WORKSPACE", str(root))
    before = snapshot(root)
    result = CliRunner().invoke(app, ["-w", str(root), "read", "plan", CASE, "--run", "r2"])
    assert result.exit_code == 0, result.exception
    cli = json.loads(result.stdout)
    mcp = mcp_server.read_plan(CASE, "r2")
    assert cli[0]["repair_targets"] == mcp[0]["repair_targets"]
    assert cli[0]["launch_prompt"] == mcp[0]["launch_prompt"] == f"kyb-reader {CASE} r2 05"
    assert snapshot(root) == before


def test_superseded_plan_cannot_allocate_from_a_stale_source(real_workspace):
    """A newer attempt created after the plan supersedes it; repair must not fork from r1."""
    root = real_workspace
    with workspace_action(root):
        plan = action.record_loop_plan(root, CASE, "r1")
        start_run(root, CASE, as_of="2026-10-25", reason="independent review")
    before = snapshot(root)
    with pytest.raises(PermissionError, match="no longer the latest"), workspace_action(root):
        action.start_loop_repair(root, CASE, plan["plan_id"])
    assert list_runs(root, CASE) == ["r1", "r2"]
    assert snapshot(root) == before
    assert verify_workspace(root).ok
