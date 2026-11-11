"""Archive is queue membership, never cancellation, deletion or a KYB outcome change."""

import asyncio
import json

import pytest
from dashboard_sample import build_sample
from typer.testing import CliRunner

from kyb.audit.chain import verify_workspace
from kyb.cli import app
from kyb.dashboard import data
from kyb.mcp_server import server
from kyb.signing.queue import review_queue
from kyb.workspace.cases import case_states, set_archived
from kyb.workspace.runs import list_runs

CASE = "DEMO_Solana"


@pytest.fixture
def workspace(tmp_path):
    return build_sample(tmp_path)


def preserved_files(root):
    return {
        str(p.relative_to(root)): p.read_bytes()
        for name in (
            "bronze",
            "silver",
            "gold",
            "policy",
            "ledger/decisions",
            "ledger/signatures",
            "ledger/runs",
        )
        for p in (root / name).rglob("*")
        if p.is_file()
    }


def test_archive_and_restore_preserve_evidence_outcomes_and_append_history(workspace):
    before = preserved_files(workspace)
    audit = workspace / "ledger/audit.jsonl"
    prefix = audit.read_bytes()
    assert CASE in {row["case"] for row in review_queue(workspace)["queue"]}
    state = set_archived(workspace, CASE, archived=True, expected_event_hash=None)
    assert state.state == "archived" and state.event_hash
    assert audit.read_bytes().startswith(prefix)
    assert preserved_files(workspace) == before
    assert list_runs(workspace, CASE) == ["r1"]
    assert CASE not in {row["case"] for row in review_queue(workspace)["queue"]}
    assert CASE in {row["case"] for row in review_queue(workspace, include_archived=True)["queue"]}
    snap = data.snapshot(workspace, CASE, "r1")
    row = next(row for row in snap["cases"] if row["case"] == CASE)
    assert row["lifecycle"]["state"] == "archived"
    assert row["recommendation"]["status"] == "REJECTED"
    assert row["recommendation"]["signature_state"] == "pending_signature"
    assert snap["selected"]["recommendations"]
    assert snap["timeline"]["events"][0]["action"] == "case.archived"
    prefix = audit.read_bytes()
    assert set_archived(workspace, CASE, archived=True, expected_event_hash=None) == state
    assert audit.read_bytes() == prefix  # duplicate delivery is an idempotent read
    restored = set_archived(workspace, CASE, archived=False, expected_event_hash=state.event_hash)
    assert restored.state == "active" and restored.event_hash != state.event_hash
    assert audit.read_bytes().startswith(prefix)
    assert CASE in {row["case"] for row in review_queue(workspace)["queue"]}
    assert preserved_files(workspace) == before and verify_workspace(workspace).ok
    # An old Archive answer cannot re-archive after another officer restored the case.
    prefix = audit.read_bytes()
    with pytest.raises(PermissionError, match="changed"):
        set_archived(workspace, CASE, archived=True, expected_event_hash=None)
    assert audit.read_bytes() == prefix


def test_archive_intent_is_pinned_native_confirmation_only_and_supports_no_run(workspace):
    snap = data.snapshot(workspace)
    workspace_id = snap["workspace_context"]["id"]
    prefix = (workspace / "ledger/audit.jsonl").read_bytes()
    intent = data.intent(workspace, "DEMO_Cedar", None, workspace_id, "archive_case")
    assert intent["authorization"] == "native_question_required"
    assert "Archive case / Cancel" in intent["message"]
    assert "work already in progress" in intent["message"].lower()
    assert (workspace / "ledger/audit.jsonl").read_bytes() == prefix
    assert list_runs(workspace, "DEMO_Cedar") == []
    archived = data.change_case_membership(
        workspace, "DEMO_Cedar", workspace_id, None, archived=True
    )
    with pytest.raises(PermissionError):
        data.intent(workspace, "DEMO_Cedar", None, workspace_id, "restore_case")
    restore = data.intent(
        workspace,
        "DEMO_Cedar",
        None,
        workspace_id,
        "restore_case",
        case_state_hash=archived.lifecycle.event_hash,
    )
    assert "Restore case / Cancel" in restore["message"]
    prefix = (workspace / "ledger/audit.jsonl").read_bytes()
    with pytest.raises(PermissionError, match="another workspace"):
        data.change_case_membership(workspace, "DEMO_Cedar", "foreign", None, archived=True)
    assert (workspace / "ledger/audit.jsonl").read_bytes() == prefix


def test_archive_cli_and_mcp_share_the_same_receipt_and_queue(workspace, monkeypatch):
    monkeypatch.setenv("KYB_WORKSPACE", str(workspace))
    workspace_id = data.workspace_context(workspace).id
    runner = CliRunner()
    common = ["-w", str(workspace)]
    result = runner.invoke(
        app, [*common, "dashboard", "archive", CASE, "--workspace-id", workspace_id]
    )
    assert result.exit_code == 0, result.output
    receipt = json.loads(result.output)
    assert receipt["case"] == CASE and receipt["lifecycle"]["state"] == "archived"
    for extra, expected in (([], False), (["--include-archived"], True)):
        result = runner.invoke(app, [*common, "queue", *extra])
        assert result.exit_code == 0, result.output
        assert (CASE in {r["case"] for r in json.loads(result.output)["queue"]}) == expected

    async def check():
        tools = {tool.name: tool for tool in await server.list_tools()}
        assert {"archive_case", "restore_case"} <= tools.keys()
        assert not tools["archive_case"].annotations.read_only_hint
        assert "expected_event_hash" in tools["archive_case"].input_schema["required"]
        prefix = (workspace / "ledger/audit.jsonl").read_bytes()
        intent = await server.call_tool(
            "dashboard_intent",
            {
                "case": CASE,
                "workspace_id": workspace_id,
                "kind": "restore_case",
                "case_state_hash": receipt["lifecycle"]["event_hash"],
            },
        )
        assert intent.structured_content["authorization"] == "native_question_required"
        assert (workspace / "ledger/audit.jsonl").read_bytes() == prefix
        result = await server.call_tool(
            "restore_case",
            {
                "case": CASE,
                "workspace_id": workspace_id,
                "expected_event_hash": receipt["lifecycle"]["event_hash"],
            },
        )
        assert not result.is_error
        assert result.structured_content["case"] == CASE
        assert result.structured_content["workspace_id"] == workspace_id
        assert result.structured_content["lifecycle"]["state"] == "active"

    asyncio.run(check())
    assert case_states(workspace)[CASE].state == "active" and verify_workspace(workspace).ok


def test_archive_refuses_a_corrupt_chain_without_hiding_the_case(workspace, monkeypatch):
    with (workspace / "ledger/audit.jsonl").open("ab") as stream:
        stream.write(b'{"bad":"event"}\n')
    before = (workspace / "ledger/audit.jsonl").read_bytes()
    with pytest.raises(RuntimeError, match="does not verify"):
        set_archived(workspace, CASE, archived=True, expected_event_hash=None)
    monkeypatch.setenv("KYB_WORKSPACE", str(workspace))

    async def check_refusal():
        from mcp.server.mcpserver.exceptions import ToolError

        with pytest.raises(ToolError, match="does not verify"):
            await server.call_tool(
                "archive_case",
                {
                    "case": CASE,
                    "workspace_id": data.workspace_context(workspace).id,
                    "expected_event_hash": None,
                },
            )

    asyncio.run(check_refusal())
    snapshot = data.snapshot(workspace)
    assert (
        next(row for row in snapshot["cases"] if row["case"] == CASE)["lifecycle"]["state"]
        == "unavailable"
    )
    assert (workspace / "ledger/audit.jsonl").read_bytes() == before
