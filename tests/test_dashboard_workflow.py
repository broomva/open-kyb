"""Caseload, scope, CLI parity and serialized API contracts; all inference is doubled."""

import asyncio
import json
from pathlib import Path

import pytest
from dashboard_sample import CASE, build_sample
from typer.testing import CliRunner

from kyb.audit.chain import verify_workspace
from kyb.cli import app
from kyb.dashboard import data
from kyb.dashboard.contracts import DashboardSnapshot
from kyb.mcp_server import server
from kyb.workspace.runs import list_runs


@pytest.fixture
def sample(tmp_path):
    return build_sample(tmp_path)


def test_workspace_lists_many_cases_and_preserves_decision_signature_distinction(sample):
    result = data.snapshot(sample)
    rows = {c["case"]: c for c in result["cases"]}
    assert len(rows) == 4
    assert rows["DEMO_Cedar"]["process"]["state"] == "not_started"
    assert rows["DEMO_Solana"]["process"]["state"] == "review_complete"
    assert rows["DEMO_Solana"]["recommendation"]["status"] == "REJECTED"
    assert rows["DEMO_Solana"]["recommendation"]["signature_state"] == "pending_signature"
    assert rows["DEMO_Harbor"]["process"]["state"] == "signed"
    assert rows["DEMO_Harbor"]["recommendation"]["status"] == "MANUAL_REVIEW"
    events = result["timeline"]["events"]
    assert result["timeline"]["chain_verified"] and events
    assert any(e["case"] == "DEMO_Harbor" and e["action"] == "sign_decision" for e in events)
    assert any(e["case"] == "DEMO_Solana" and e["action"] == "emit_decision" for e in events)
    assert events == sorted(events, key=lambda e: e["seq"], reverse=True)


def test_case_questions_are_pinned_and_do_not_allocate_or_append(sample):
    snap = data.snapshot(sample, CASE, "r1")
    prefix = (sample / "ledger/audit.jsonl").read_bytes()
    args = {
        "case": CASE,
        "run": "r1",
        "workspace_id": snap["workspace_context"]["id"],
        "snapshot_id": snap["selected"]["snapshot_id"],
        "kind": "question",
        "question": "Which evidence is missing?",
        "case_state_hash": None,
    }
    request = data.intent(sample, **args)
    assert request["authorization"] == "question_only"
    assert request["case"] == CASE and request["run"] == "r1"
    assert json.loads(request["message"].splitlines()[1]) == args
    assert request["message"].count("record_answer") == 1
    assert (sample / "ledger/audit.jsonl").read_bytes() == prefix
    assert list_runs(sample, CASE) == ["r1"]


def test_wrong_workspace_or_case_cannot_reuse_a_request(sample, tmp_path):
    snap = data.snapshot(sample, CASE, "r1")
    other = tmp_path / "other"
    other.mkdir()
    import shutil

    root = other / "officer-workspace"
    shutil.copytree(sample, root)
    # Same basename, same case/run and same evidence bytes: different workspace identity.
    assert data.workspace_context(sample).label == data.workspace_context(root).label
    assert data.workspace_context(sample).id != data.workspace_context(root).id
    with pytest.raises(PermissionError, match="another workspace"):
        data.prepare(
            root,
            CASE,
            "r1",
            snap["selected"]["snapshot_id"],
            workspace_id=snap["workspace_context"]["id"],
        )
    with pytest.raises(PermissionError):
        data.intent(
            sample,
            "DEMO_Solana",
            "r1",
            snap["workspace_context"]["id"],
            "question",
            "Why?",
            snap["selected"]["snapshot_id"],
        )
    assert list_runs(root, CASE) == ["r1"] and verify_workspace(root).ok


@pytest.mark.parametrize("name", ["ask", "record_answer", "plan_loop", "start_loop_repair"])
def test_operative_mcp_tools_reject_foreign_workspace_before_work(sample, monkeypatch, name):
    from kyb import mcp_server

    monkeypatch.setattr(mcp_server, "_root", lambda: sample)
    args = {"case": CASE, "workspace_id": "sha256:foreign"}
    if name in {"ask", "record_answer"}:
        args["question"] = "Which evidence is missing?"
    if name == "record_answer":
        args["statements"] = []
    if name == "start_loop_repair":
        args["plan_id"] = "a" * 64
    before = (sample / "ledger/audit.jsonl").read_bytes()
    with pytest.raises(PermissionError, match="another workspace"):
        getattr(mcp_server, name)(**args)
    assert (sample / "ledger/audit.jsonl").read_bytes() == before
    assert list_runs(sample, CASE) == ["r1"]


def test_cli_snapshot_question_and_retry_use_the_same_typed_core(sample):
    runner = CliRunner()
    common = ["-w", str(sample), "dashboard"]
    result = runner.invoke(app, [*common, "snapshot", "--case", CASE, "--run", "r1"])
    assert result.exit_code == 0, result.output
    snap = DashboardSnapshot.model_validate_json(result.output).model_dump(mode="json")
    result = runner.invoke(
        app,
        [
            *common,
            "intent",
            CASE,
            "--run",
            "r1",
            "--workspace-id",
            snap["workspace_context"]["id"],
            "--kind",
            "question",
            "--question",
            "Why?",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["authorization"] == "question_only"
    result = runner.invoke(
        app,
        [
            *common,
            "request-retry",
            CASE,
            "--run",
            "r1",
            "--snapshot",
            snap["selected"]["snapshot_id"],
        ],
    )
    assert result.exit_code == 0, result.output
    request = json.loads(result.output)
    assert request["request_id"] != request["plan_id"]
    assert request["allocation"] is None and request["valid"]
    result = runner.invoke(
        app,
        [*common, "handoff", CASE, request["plan_id"], "--workspace-id", request["workspace_id"]],
    )
    assert result.exit_code == 0 and json.loads(result.output) == request
    assert list_runs(sample, CASE) == ["r1"]


def test_mcp_public_outputs_have_named_schemas_and_strict_limits(sample, monkeypatch):
    monkeypatch.setenv("KYB_WORKSPACE", str(sample))

    async def check():
        from mcp.server.mcpserver.exceptions import ToolError

        tools = {t.name: t for t in await server.list_tools()}
        schema = tools["dashboard_snapshot"].output_schema
        assert "workspace_context" in schema["properties"]
        assert "timeline" in schema["properties"]
        assert "$ref" in schema["properties"]["cases"]["items"]
        snap = data.snapshot(sample, CASE, "r1")
        with pytest.raises(ToolError, match="valid integer"):
            await server.call_tool(
                "prepare_dashboard_action",
                {
                    "case": CASE,
                    "run": "r1",
                    "snapshot_id": snap["selected"]["snapshot_id"],
                    "max_attempts": True,
                },
            )
        assert list_runs(sample, CASE) == ["r1"]

    asyncio.run(check())


def test_corrupt_activity_is_unavailable_instead_of_a_green_timeline(sample):
    # Fresh fixture, append-only even in adversarial testing.
    with (sample / "ledger/audit.jsonl").open("ab") as stream:
        stream.write(b'{"bad":"event"}\n')
    result = data.snapshot(sample)
    assert not result["timeline"]["chain_verified"]
    assert not result["timeline"]["events"]


def test_generated_types_match_api_models():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts/dashboard_contracts.py"
    spec = importlib.util.spec_from_file_location("dashboard_contracts", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.TARGET.read_text() == module.generate()


def test_expired_retry_cannot_be_reused_or_silently_extend_its_budget(sample, monkeypatch):
    from datetime import UTC, datetime, timedelta

    pin = data.detail(sample, CASE, "r1")["snapshot_id"]
    request = data.prepare(sample, CASE, "r1", pin)
    before = (sample / "ledger/audit.jsonl").read_bytes()
    after_expiry = datetime.fromisoformat(request["expires_at"]) + timedelta(seconds=1)

    class Later(datetime):
        @classmethod
        def now(cls, tz=UTC):
            return after_expiry.astimezone(tz)

    monkeypatch.setattr(data, "datetime", Later)
    assert data.handoff(sample, CASE, request["plan_id"])["state"] == "expired"
    assert not data.handoff(sample, CASE, request["plan_id"])["valid"]
    with pytest.raises(PermissionError, match="expired"):
        data.prepare(sample, CASE, "r1", pin)
    assert (sample / "ledger/audit.jsonl").read_bytes() == before
    assert list_runs(sample, CASE) == ["r1"]
