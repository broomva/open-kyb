"""The kyb MCP server and plugin (spec B3): tools over the same core, found from the repo root."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path

import pytest

from kyb import mcp_server as mcp
from kyb.witness.port import Witness

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "examples"
CASE = "es-iberia-software"
pytestmark = pytest.mark.skipif(shutil.which("pdftotext") is None, reason="poppler missing")

LAYOUT = {"pages": [{"page": 1, "blocks": [
    {"block_id": "b5", "kind": "paragraph", "box_2d": [209, 98, 270, 914], "refers_to": []}]}],
    "material_facts": []}  # fmt: skip


class Fake:
    model, prompt_version = "fake-witness-1", "witness-v1"

    def call(self, kind, images, spec):
        if kind == "layout":
            return LAYOUT, {}
        return {"crops": [{"crop_id": c, "readable": True, "transcription": "V. Ortega Méndez 25%",
                           "reread": {"subject": "V. Ortega Méndez", "value": "25%"}}
                          for c in spec["crop_ids"]]}, {}  # fmt: skip


@pytest.fixture
def project(tmp_path_factory):
    """A project directory with one workspace under workspaces/, as the officer would have it."""
    import os

    proj = tmp_path_factory.mktemp("proj")
    old = Path.cwd()
    os.chdir(proj)
    mp = pytest.MonkeyPatch()
    mp.delenv("KYB_WORKSPACE", raising=False)
    mp.setattr("kyb.witness.port.make_witness", lambda root, mode: Witness(root, Fake()))
    try:
        mcp.init_workspace(policy=str(ASSETS / "policy/compliance_rules.md"))
        mcp.ingest_packet(str(ASSETS / "packets" / CASE), "2026-10-25")
        yield proj
    finally:
        mp.undo()
        os.chdir(old)


def test_the_server_lists_the_workflow_tools():
    names = {t.name for t in asyncio.run(mcp.server.list_tools())}
    assert {"review_status", "init_workspace", "ingest_packet", "pages", "layout_pages",
            "read_plan", "render_crop", "read_document", "verify_quotes",
            "audit_verify"} <= names  # fmt: skip


def test_the_workflow_runs_through_the_tools(project, monkeypatch):
    monkeypatch.chdir(project)
    monkeypatch.setattr("kyb.witness.port.make_witness", lambda root, mode: Witness(root, Fake()))
    assert mcp.review_status(CASE)["next"] == "pages"
    elsewhere = project.parent / "elsewhere"
    elsewhere.mkdir(exist_ok=True)
    monkeypatch.chdir(elsewhere)  # no workspace in reach: the next step is init
    assert mcp.review_status(CASE)["next"] == "init"
    monkeypatch.chdir(project)
    assert mcp.pages(CASE)["pages"] == 5
    assert mcp.review_status(CASE)["next"] == "layout"
    assert len(mcp.layout_pages(CASE)["laid_out"]) == 5
    status = mcp.review_status(CASE)
    assert status["next"] == "read" and status["launch"][4] == f"kyb-reader {CASE} r1 05"
    (entry,) = mcp.read_plan(CASE, "r1", "05")
    crop = mcp.render_crop(CASE, "r1", "05", 1, [1710, 760, 2300, 835], 600)
    assert Path(crop["crop"]).is_absolute()
    assert Path(crop["crop"]).read_bytes()[:4] == b"\x89PNG"
    claim = {"claim_id": "c-B05-06", "type": "owns", "subject": "V. Ortega Méndez",
             "value": {"pct": 25}, "excerpt": "V. Ortega Méndez 25%", "page": 1,
             "bbox": [1710, 760, 2300, 835], "block_id": "05-b5", "source": "pixels",
             "confidence": 0.9}  # fmt: skip
    bad = mcp.read_document(CASE, "r1", "05", entry["doc_sha256"], [{**claim, "page": 9}])
    assert "rejected" in bad  # structured errors, nothing recorded
    ok = mcp.read_document(CASE, "r1", "05", entry["doc_sha256"], [claim])
    assert ok["context"] == "unattested"  # no hook ran in this test
    for d in ("01", "02", "03", "04"):
        sha = mcp.read_plan(CASE, "r1", d)[0]["doc_sha256"]
        mcp.read_document(CASE, "r1", d, sha, [])
    summary = mcp.verify_quotes(CASE)
    assert summary["documents"]["05"]["verified"] == 1
    assert mcp.review_status(CASE)["next"] == "decide"
    assert mcp.audit_verify()["ok"] is True


def test_a_call_through_the_mcp_layer_returns_structured_content(project, monkeypatch):
    monkeypatch.chdir(project)
    result = asyncio.run(mcp.server.call_tool("audit_verify", {}))
    payload = result.structured_content or json.loads(result.content[0].text)
    assert payload["ok"] is True


def test_the_plugin_is_wired_to_the_same_tools():
    plugin = REPO / "plugin"
    reader = (plugin / "agents/kyb-reader.md").read_text()
    for tool in ("read_plan", "render_crop", "read_document"):
        assert f"mcp__plugin_kyb_kyb__{tool}" in reader
    hooks = json.loads((plugin / "hooks/hooks.json").read_text())["hooks"]
    assert {"PreToolUse", "SessionStart", "SubagentStart", "SubagentStop", "PostToolUse",
            "Stop"} <= set(hooks)  # fmt: skip
    assert all(h.get("onFailure") == "block" for h in hooks["PreToolUse"][0]["hooks"])
    same = (REPO / ".claude/skills/kyb-read-packet/SKILL.md").read_text()
    assert (plugin / "skills/kyb-read-packet/SKILL.md").read_text() == same


def test_a_reader_in_the_project_directory_can_read_what_read_plan_returns(project, monkeypatch):
    """Dogfood regression: the session's cwd is the project, not the workspace. read_plan must
    hand readers paths the guard admits, and a workspace-relative path must stay refused."""
    from kyb.workspace import guard

    monkeypatch.chdir(project)
    monkeypatch.setattr("kyb.witness.port.make_witness", lambda root, mode: Witness(root, Fake()))
    mcp.pages(CASE)
    mcp.layout_pages(CASE)

    def hook(fn, **event):
        return fn({"cwd": str(project), "session_id": "s1", **event})

    def tool(name, tin, agent_id="", agent_type="", tool_use_id="t1"):
        return hook(guard.pre_tool_use, tool_name=name, tool_input=tin, agent_id=agent_id,
                    agent_type=agent_type, tool_use_id=tool_use_id)  # fmt: skip

    launch = {"subagent_type": "kyb:kyb-reader", "prompt": f"kyb-reader {CASE} r1 05"}
    assert tool("Agent", launch, tool_use_id="L1")[0] == 0
    assert hook(guard.subagent_start, agent_id="a1", agent_type="kyb:kyb-reader")[0] == 0
    bind = {"case": CASE, "run": "r1", "doc": "05"}
    assert tool("mcp__plugin_kyb_kyb__read_plan", bind, "a1", "kyb:kyb-reader", "rp")[0] == 0
    (entry,) = mcp.read_plan(CASE, "r1", "05")
    crop = mcp.render_crop(CASE, "r1", "05", 1, [1710, 760, 2300, 835], 600)
    for path in (entry["pages"][0]["png"], entry["pages"][0]["json"], entry["layout"],
                 crop["crop"]):  # fmt: skip
        assert Path(path).is_absolute() and Path(path).exists()
        assert tool("Read", {"file_path": path}, "a1", "kyb:kyb-reader")[0] == 0, path
    rel = f"silver/pages/{CASE}/r1/05/p1.png"
    assert tool("Read", {"file_path": rel}, "a1", "kyb:kyb-reader")[0] == 2
    other = entry["pages"][0]["png"].replace("/05/", "/04/")
    assert tool("Read", {"file_path": other}, "a1", "kyb:kyb-reader")[0] == 2


def test_review_round_1_findings(project, monkeypatch, tmp_path):
    """PR #13 cross-review: discovery from a subdirectory, fail closed when ambiguous, crops on
    the bound document only, one workspace per project, refusals keep their message, plugin
    hooks stand down inside a workspace with its own, and a stale verification is not `decide`."""
    from kyb.workspace import guard

    monkeypatch.chdir(project)
    ws = project / "workspaces/kyb-demo"
    sub = project / "assets"
    sub.mkdir(exist_ok=True)
    assert guard.find_root(sub.resolve()) == ws.resolve()  # from a project subdirectory

    # a refused precondition reaches the agent with its message, not a masked failure
    from mcp.server.mcpserver.exceptions import ToolError

    with pytest.raises(ToolError, match="no run yet"):
        asyncio.run(mcp.server.call_tool("read_plan", {"case": "nope"}))

    # a second workspace in the same project is refused (it would make discovery ambiguous),
    # also when two inits race: one wins under the project lock, the other is refused
    from concurrent.futures import ThreadPoolExecutor

    with pytest.raises(FileExistsError):
        mcp.init_workspace(tenant="other", policy=str(ASSETS / "policy/compliance_rules.md"))
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    monkeypatch.chdir(fresh)
    with pytest.raises(FileNotFoundError, match="policy"):  # a wrong path: say so, early
        mcp.init_workspace(policy="missing/compliance_rules.md")
    assert not (fresh / "workspaces").exists()

    def init(t):
        try:
            return mcp.init_workspace(tenant=t, policy=str(ASSETS / "policy/compliance_rules.md"))
        except FileExistsError:
            return None

    with ThreadPoolExecutor(2) as pool:
        made = [r for r in pool.map(init, ("a", "b")) if r]
    assert len(made) == 1 and guard.discover(fresh)[0] is not None
    # a packet-only project with no policy argument gets the shipped copy of open-kyb's policy
    bare = tmp_path / "bare"
    bare.mkdir()
    monkeypatch.chdir(bare)
    made = mcp.init_workspace()
    ws = Path(made["workspace"])
    for name in ("compliance_rules.md", "watchlists.json"):  # screening needs the list too
        assert (ws / "policy" / name).read_bytes() == (ASSETS / "policy" / name).read_bytes()
    from kyb.screening.action import default_watchlist

    assert made["watchlist_at_init"] and default_watchlist(ws) is not None
    monkeypatch.chdir(project)

    # a reader bound to 05 may not render another document's crop
    mcp.pages(CASE)
    mcp.layout_pages(CASE)

    def tool(name, tin, agent_id="", agent_type="", tool_use_id="t1", cwd=project):
        event = {
            "cwd": str(cwd),
            "session_id": "s9",
            "tool_name": name,
            "tool_input": tin,
            "agent_id": agent_id,
            "agent_type": agent_type,
            "tool_use_id": tool_use_id,
        }
        return guard.pre_tool_use(event)

    launch = {"subagent_type": "kyb:kyb-reader", "prompt": f"kyb-reader {CASE} r1 05"}
    assert tool("Agent", launch, tool_use_id="L9")[0] == 0
    guard.subagent_start({"cwd": str(project), "session_id": "s9", "agent_id": "a9",
                          "agent_type": "kyb:kyb-reader"})  # fmt: skip
    bind = {"case": CASE, "run": "r1", "doc": "05"}
    assert tool("mcp__plugin_kyb_kyb__read_plan", bind, "a9", "kyb:kyb-reader", "rp9")[0] == 0
    crop = {**bind, "page": 1, "bbox": [0, 0, 10, 10]}
    assert tool("mcp__plugin_kyb_kyb__render_crop", crop, "a9", "kyb:kyb-reader")[0] == 0
    assert tool("mcp__plugin_kyb_kyb__render_crop", {**crop, "doc": "04"}, "a9",
                "kyb:kyb-reader")[0] == 2  # fmt: skip

    # with two workspaces in reach nothing resolves, and a reader fails closed instead of open
    two = tmp_path / "two"
    for t in ("a", "b"):
        (two / "workspaces" / t).mkdir(parents=True)
        (two / "workspaces" / t / "template.lock").write_text("{}")
    assert guard.discover(two)[0] is None and "ambiguous" in guard.discover(two)[1]
    assert tool("Read", {"file_path": "/etc/hosts"}, "a9", "kyb:kyb-reader", cwd=two)[0] == 2
    assert tool("Read", {"file_path": "/etc/hosts"}, cwd=two)[0] == 0  # not a kyb agent


def test_status_refuses_decide_on_a_stale_verification(project, monkeypatch):
    monkeypatch.chdir(project)
    mcp.pages(CASE)
    mcp.layout_pages(CASE)
    for d in ("01", "02", "03", "04", "05"):
        mcp.read_document(CASE, "r1", d, mcp.read_plan(CASE, "r1", d)[0]["doc_sha256"], [])
    mcp.verify_quotes(CASE)
    assert mcp.review_status(CASE)["next"] == "decide"
    ws = project / "workspaces/kyb-demo"
    (ws / f"silver/verified_claims/{CASE}/r1.json").unlink()
    status = mcp.review_status(CASE)
    assert status["next"] == "stale" and "receipt" in status["detail"]


def test_discovery_never_climbs_above_its_root(tmp_path, monkeypatch):
    """PR #14 review: a fresh project nested under a directory with its own workspaces/ must not
    resolve that outer workspace when the harness pins KYB_DISCOVERY_ROOT to the project."""
    from kyb.workspace import guard

    outer = tmp_path / "home"
    (outer / "workspaces" / "old").mkdir(parents=True)
    (outer / "workspaces" / "old" / "template.lock").write_text("{}")
    fresh = outer / "runs" / "r1" / "project"
    fresh.mkdir(parents=True)
    assert guard.discover(fresh)[0] == (outer / "workspaces" / "old").resolve()  # unbounded
    monkeypatch.setenv("KYB_DISCOVERY_ROOT", str(fresh))
    assert guard.discover(fresh) == (None, "none")


def test_default_init_works_from_a_zip_imported_package(tmp_path):
    """resources.files() need not be a filesystem path: a zip-imported kyb still initializes."""
    import subprocess
    import sys
    import zipfile

    archive = tmp_path / "kyb.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for f in (REPO / "src").rglob("*"):
            if f.is_file() and "__pycache__" not in f.parts:
                z.write(f, f.relative_to(REPO / "src"))
    project = tmp_path / "proj"
    project.mkdir()
    probe = (
        "import sys; sys.path.insert(0, sys.argv[1]); import kyb;"
        "assert kyb.__file__.startswith(sys.argv[1]), kyb.__file__;"
        "from kyb.mcp_server import init_workspace; r = init_workspace();"
        "print(r['watchlist_at_init'] is not None)"
    )
    env = {k: v for k, v in os.environ.items() if k != "KYB_WORKSPACE"}
    out = subprocess.run([sys.executable, "-c", probe, str(archive)], cwd=project,
                         env={**env, "CLAUDE_PROJECT_DIR": str(project)},
                         capture_output=True, text=True, check=True)  # fmt: skip
    assert out.stdout.strip() == "True"
    assert (project / "workspaces/kyb-demo/policy/watchlists.json").is_file()


def test_decide_reply_leaves_the_audit_trail_in_the_record(project, monkeypatch):
    """The ledger inlined in a decision outgrew a host's tool-output limit (168k characters on
    packet B); the reply drops it and points at the recorded decision, which keeps it."""
    from types import SimpleNamespace

    trail = [{"seq": 0}] * 3
    record = {
        "decision_id": "d-1",
        "kyb_decision": {"status": "MANUAL_REVIEW", "audit_trail": trail},
    }
    made = SimpleNamespace(model_dump=lambda mode: record)
    monkeypatch.setattr("kyb.decide.action.decide_case", lambda root, case, run: made)
    monkeypatch.setattr(mcp, "_run", lambda root, case, run: "r1")
    out = mcp.decide(CASE)
    assert (
        "audit_trail" not in out["kyb_decision"] and out["kyb_decision"]["audit_trail_events"] == 3
    )
    assert out["record"].endswith("ledger/decisions/d-1.json")
