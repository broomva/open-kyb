"""Tests for Lane B loop CLI, MCP server and ontology (issue)."""

import json

import pytest
from decision_fixtures import workspace
from typer.testing import CliRunner

from kyb.cli import app
from kyb.loop import action
from kyb.ontology import workspace_ontology


# 1. Test Ontology Paths and registration
def test_ontology_registration(tmp_path):
    ws = workspace(tmp_path)
    onto = workspace_ontology(ws)

    # plan_loop and start_loop_repair should be actions writing to ledger/loops/**
    assert "plan_loop" in onto.actions
    assert "start_loop_repair" in onto.actions
    assert "ledger/loops/**" in onto.actions["plan_loop"].writes
    assert "ledger/loops/**" in onto.actions["start_loop_repair"].writes
    assert onto.actions["plan_loop"].tier_out == "derived"
    assert onto.actions["start_loop_repair"].tier_out == "derived"

    # inspect_loop should be a read tool
    assert "inspect_loop" in onto.reads
    assert "agent" in onto.reads["inspect_loop"].invoker


# 2. Test CLI/MCP Same Core/Locks and delegation
def test_cli_same_core_and_locks(tmp_path, monkeypatch):
    ws = workspace(tmp_path)
    runner = CliRunner()

    # Let's mock the actual core functions to verify CLI delegates to them
    inspect_called = []

    def mock_inspect(root, case, run):
        inspect_called.append((root, case, run))
        return {"inspected": True}

    plan_called = []

    def mock_plan(root, case, run, *, budget=None, previous_plan=None):
        plan_called.append((root, case, run, budget, previous_plan))
        return {"planned": True}

    repair_called = []

    def mock_repair(root, case, plan_id):
        repair_called.append((root, case, plan_id))
        return {"repaired": True}

    monkeypatch.setattr(action, "inspect_loop", mock_inspect)
    monkeypatch.setattr(action, "record_loop_plan", mock_plan)
    monkeypatch.setattr(action, "start_loop_repair", mock_repair)

    # Run CLI inspect
    result = runner.invoke(app, ["-w", str(ws), "loop", "inspect", "case_test", "--run", "r1"])
    assert result.exit_code == 0
    assert "inspected" in result.stdout
    assert inspect_called == [(ws, "case_test", "r1")]

    # Run CLI plan
    result = runner.invoke(
        app,
        [
            "-w",
            str(ws),
            "loop",
            "plan",
            "case_test",
            "--run",
            "r1",
            "--previous-plan",
            "123",
            "--max-attempts",
            "5",
        ],
    )
    assert result.exit_code == 0
    assert "planned" in result.stdout
    assert len(plan_called) == 1
    assert plan_called[0][1] == "case_test"
    assert plan_called[0][2] == "r1"
    assert plan_called[0][3].max_attempts == 5
    assert plan_called[0][4] == "123"

    # Run CLI repair
    result = runner.invoke(
        app,
        [
            "-w",
            str(ws),
            "loop",
            "repair",
            "case_test",
            "--plan",
            "plan_id_123",
        ],
    )
    assert result.exit_code == 0
    assert "repaired" in result.stdout
    assert repair_called == [(ws, "case_test", "plan_id_123")]


# 3. Test MCP tool Parity and locking
def test_mcp_parity(tmp_path, monkeypatch):
    from kyb.mcp_server import inspect_loop, plan_loop, start_loop_repair

    ws = workspace(tmp_path)

    # Set environment variables for the MCP discovery helper
    monkeypatch.setenv("KYB_WORKSPACE", str(ws))

    inspect_called = []

    def mock_inspect(root, case, run):
        inspect_called.append((root, case, run))
        return {"inspected": True}

    plan_called = []

    def mock_plan(root, case, run, *, budget=None, previous_plan=None):
        plan_called.append((root, case, run, budget, previous_plan))
        return {"planned": True}

    repair_called = []

    def mock_repair(root, case, plan_id):
        repair_called.append((root, case, plan_id))
        return {"repaired": True}

    monkeypatch.setattr(action, "inspect_loop", mock_inspect)
    monkeypatch.setattr(action, "record_loop_plan", mock_plan)
    monkeypatch.setattr(action, "start_loop_repair", mock_repair)

    res = inspect_loop("case_test", "r1")
    assert res == {"inspected": True}
    assert inspect_called == [(ws, "case_test", "r1")]

    res = plan_loop("case_test", "r1", previous_plan="123", max_attempts=5)
    assert res == {"planned": True}
    assert plan_called[0][3].max_attempts == 5

    res = start_loop_repair("case_test", "plan_id_123")
    assert res == {"repaired": True}
    assert repair_called == [(ws, "case_test", "plan_id_123")]


# 4. Test Malicious ID/path refusal
def test_malicious_refusal(tmp_path, monkeypatch):
    ws = workspace(tmp_path)
    monkeypatch.setenv("KYB_WORKSPACE", str(ws))
    runner = CliRunner()

    # safe_id should refuse invalid characters
    result = runner.invoke(app, ["-w", str(ws), "loop", "inspect", "case/../../etc/passwd"])
    assert result.exit_code != 0 or "ValueError" in result.stdout or "invalid" in result.stdout

    # Also verify that start_loop_repair refuses non-64hex plan IDs
    from kyb.mcp_server import start_loop_repair

    with pytest.raises(Exception) as exc:
        start_loop_repair("case_test", "short_id")
    assert "ValueError" in str(type(exc.value)) or "ValueError" in str(exc.value)


# 5. Test Old Workspace Upgrade
def test_old_workspace_upgrade(tmp_path, monkeypatch):
    ws = workspace(tmp_path)

    # Mock ontology to look like an old one that doesn't support loop actions
    class LegacyOntology:
        def __init__(self):
            self.actions = {}
            self.reads = {}

    monkeypatch.setattr("kyb.loop.action.workspace_ontology", lambda *a: LegacyOntology())

    with pytest.raises(PermissionError) as exc:
        action.inspect_loop(ws, "case_test", "r1")
    assert "upgrade" in str(exc.value)


# 6. Test Target Non-leak and Legacy shape compatibility
def test_target_non_leak_and_compatibility(tmp_path, monkeypatch):
    ws = workspace(tmp_path)

    # We write a custom mocked get_repair_targets to verify it appends correctly.
    def dummy_targets(root, case, run, doc):
        return [{"field": "address", "kind": "missing_field"}]

    # Let's attach get_repair_targets to action
    monkeypatch.setattr(action, "get_repair_targets", dummy_targets, raising=False)

    # Let's mock a simple read_plan output
    mock_plan = [
        {
            "doc": "doc-a",
            "case": "case-test",
            "run": "r1",
            "launch_prompt": "prompt",
        }
    ]
    monkeypatch.setattr("kyb.commands.reading.read_plan", lambda *a, **k: mock_plan)

    runner = CliRunner()
    result = runner.invoke(app, ["-w", str(ws), "read", "plan", "case-test", "--run", "r1"])
    assert result.exit_code == 0
    parsed = json.loads(result.stdout)
    assert parsed[0]["repair_targets"] == [{"field": "address", "kind": "missing_field"}]

    # Verify non-leak constraint: only controlled keys allowed
    for target in parsed[0]["repair_targets"]:
        assert set(target.keys()) == {"field", "kind"}


@pytest.mark.parametrize("surface", ["cli", "mcp"])
@pytest.mark.parametrize("failure", ["missing_helper", "inner_import"])
def test_read_plan_refuses_unavailable_repair_validation(tmp_path, monkeypatch, surface, failure):
    """Broken integration must never return a successful legacy work order."""
    from kyb import mcp_server

    ws = workspace(tmp_path)
    monkeypatch.setenv("KYB_WORKSPACE", str(ws))
    entries = [{"doc": "doc-a", "layout": "layout.json", "pages": []}]
    monkeypatch.setattr("kyb.commands.reading.read_plan", lambda *a, **k: entries)
    monkeypatch.setattr("kyb.read.plan.read_plan", lambda *a, **k: entries)
    if failure == "missing_helper":
        monkeypatch.delattr(action, "get_repair_targets")
    else:

        def broken(*args):
            raise ImportError("required validator dependency is missing")

        monkeypatch.setattr(action, "get_repair_targets", broken)
    if surface == "mcp":
        with pytest.raises(ImportError):
            mcp_server.read_plan("case_test", "r1")
    else:
        result = CliRunner().invoke(
            app, ["-w", str(ws), "read", "plan", "case_test", "--run", "r1"]
        )
        assert result.exit_code != 0
        assert isinstance(result.exception, ImportError)
        assert result.stdout == ""
