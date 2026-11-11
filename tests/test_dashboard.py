"""Real workspace admission + MCP protocol + browser boundary regressions (offline)."""

import asyncio
import json
import os
import shutil
import subprocess
import sys
import threading
from http.client import HTTPConnection
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_loop_action_integration import CASE, snapshot
from test_loop_action_integration import real_workspace as workspace_fixture

from kyb.audit.chain import verify_workspace, workspace_action
from kyb.dashboard import data
from kyb.dashboard.browser import DashboardServer, app_html
from kyb.dashboard.mcp import MIME, URI, register
from kyb.loop.action import preview_loop_plan, record_loop_plan, start_loop_repair
from kyb.loop.models import LoopBudget
from kyb.workspace.runs import list_runs


@pytest.fixture
def real_workspace(tmp_path):
    return workspace_fixture.__wrapped__(tmp_path)


def test_inspection_pins_receipted_run_without_writes(real_workspace):
    root = real_workspace
    before = snapshot(root)
    s = data.snapshot(root, CASE, "r1")
    d = s["selected"]
    assert d["readiness"] == "repair" and d["agents"][0]["assurance"] == "W2"
    assert d["agents"][0]["accuracy"] is None
    assert d["loop"]["metrics"]["usage"]["cost_usd"] is None
    assert snapshot(root) == before
    with workspace_action(root):
        plan = record_loop_plan(root, CASE, "r1")
        start_loop_repair(root, CASE, plan["plan_id"])
    d = data.snapshot(root, CASE, "r1")["selected"]
    assert d["historical"] and d["run"] == "r1"
    assert d["plans"][0]["allocation"]["run"] == "r2"
    with pytest.raises(PermissionError, match="stale"):
        data.prepare(root, CASE, "r1", d["snapshot_id"])


def test_prepare_is_plan_only_idempotent_and_has_no_browser_authority(real_workspace):
    root = real_workspace
    pin = data.detail(root, CASE, "r1")["snapshot_id"]
    before = snapshot(root)
    for attempts in (False, 1.5, -1, 0):
        with pytest.raises((ValueError, PermissionError)):
            data.prepare(root, CASE, "r1", pin, attempts)
        assert snapshot(root) == before
    out = data.prepare(root, CASE, "r1", pin, 2, 1200)
    assert out["authorization"] == "native_question_required" and out["valid"]
    assert out["allocation"] is None and list_runs(root, CASE) == ["r1"]
    assert out["plan"]["metrics"]["budget"]["max_elapsed_seconds"] == 1200
    before = snapshot(root)
    assert data.prepare(root, CASE, "r1", pin, 2, 1200) == out
    assert data.handoff(root, CASE, out["plan_id"]) == out
    assert snapshot(root) == before and verify_workspace(root).ok


@pytest.mark.parametrize("target", ["policy/compliance_rules.md", "gold/facts/{case}/r1.json"])
def test_changed_inputs_refuse_staging_and_late_handoff(real_workspace, target):
    root = real_workspace
    d = data.detail(root, CASE, "r1")
    out = data.prepare(root, CASE, "r1", d["snapshot_id"])
    path = root / target.format(case=CASE)
    path.write_bytes(path.read_bytes() + b" ")  # fresh adversarial fixture, never rewrite audit
    before = snapshot(root)
    if target.startswith("gold/"):
        with pytest.raises(PermissionError, match="differs from receipt"):
            data.handoff(root, CASE, out["plan_id"])
    else:
        assert not data.handoff(root, CASE, out["plan_id"])["valid"]
    with pytest.raises(PermissionError):
        data.prepare(root, CASE, "r1", d["snapshot_id"])
    assert snapshot(root) == before and list_runs(root, CASE) == ["r1"]


@pytest.mark.parametrize(
    "case,run", [("../foreign", "r1"), (CASE, "../r1"), ("foreign", "r1"), (CASE, "r999")]
)
def test_foreign_context_is_not_a_filesystem_browser(real_workspace, case, run):
    with pytest.raises((ValueError, LookupError)):
        data.snapshot(real_workspace, case, run)


def test_preview_uses_the_exact_write_plan_without_writes(real_workspace):
    root = real_workspace
    before = snapshot(root)
    p = preview_loop_plan(root, CASE, "r1", budget=LoopBudget(max_elapsed_seconds=1200))
    assert snapshot(root) == before
    with workspace_action(root):
        assert record_loop_plan(root, CASE, "r1", budget=LoopBudget(max_elapsed_seconds=1200)) == p


def test_local_browser_rejects_origin_host_traversal_and_execution(real_workspace):
    root = real_workspace
    server = DashboardServer(root)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    conn = HTTPConnection("127.0.0.1", server.server_port)

    def request(method, path, body=None, headers=None):
        conn.request(method, path, body, headers or {})
        response = conn.getresponse()
        return response.status, response.read(), dict(response.getheaders())

    try:
        status, body, headers = request("GET", f"/{server.token}/")
        assert status == 200 and b"open-kyb" in body and b"/* DASHBOARD" not in body
        assert headers["Cache-Control"] == "no-store"
        assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
        assert request("GET", "/snapshot")[0] == 400
        assert request("GET", f"/{server.token}/snapshot", headers={"Host": "evil.test"})[0] == 400
        assert (
            request("GET", f"/{server.token}/snapshot", headers={"Origin": "https://evil.test"})[0]
            == 400
        )
        args = {
            "case": CASE,
            "run": "r1",
            "snapshot_id": data.detail(root, CASE, "r1")["snapshot_id"],
        }
        assert (
            request(
                "POST",
                f"/{server.token}/prepare",
                json.dumps(args),
                {"Content-Type": "application/json", "Origin": "https://evil.test"},
            )[0]
            == 400
        )
        before = snapshot(root)
        assert (
            request(
                "POST",
                f"/{server.token}/execute",
                '{"approved":true}',
                {"Content-Type": "application/json", "Origin": server.origin},
            )[0]
            == 404
        )
        assert snapshot(root) == before
        for endpoint in ("archive", "restore"):
            assert (
                request(
                    "POST",
                    f"/{server.token}/{endpoint}",
                    '{"confirmed":true}',
                    {"Content-Type": "application/json", "Origin": server.origin},
                )[0]
                == 404
            )
        for query in ("run=r1", "case=x&case=y", "workspace_path=/tmp/foreign"):
            assert request("GET", f"/{server.token}/snapshot?{query}")[0] == 400
        for extra in ({"max_attempts": True}, {"approved": True}, {"max_attempts": "2"}):
            assert (
                request(
                    "POST",
                    f"/{server.token}/prepare",
                    json.dumps({**args, **extra}),
                    {"Content-Type": "application/json", "Origin": server.origin},
                )[0]
                == 400
            )
        assert snapshot(root) == before
        status, body, _ = request(
            "POST",
            f"/{server.token}/prepare",
            json.dumps(args),
            {"Content-Type": "application/json", "Origin": server.origin},
        )
        assert status == 200 and json.loads(body)["authorization"] == "native_question_required"
        assert list_runs(root, CASE) == ["r1"]
    finally:
        conn.close()
        server.shutdown()
        server.server_close()


def test_registered_resource_and_negotiated_opening(real_workspace):
    from mcp.server.apps import Apps
    from mcp.server.mcpserver import Context, MCPServer
    from mcp.types import ClientCapabilities

    server = MCPServer("dashboard-test", extensions=[Apps()])
    register(server, lambda: real_workspace)

    async def run():
        tools = {t.name: t for t in await server.list_tools()}
        assert tools["open_dashboard"].meta["ui"]["resourceUri"] == URI
        resource = next(r for r in await server.list_resources() if str(r.uri) == URI)
        assert resource.mime_type == MIME
        content = await server.read_resource(URI)
        assert "open-kyb" in content[0].content
        # Exercise the SDK's injected Context, with real client capability models.
        fn = server._tool_manager.get_tool("open_dashboard").fn
        before = snapshot(real_workspace)
        for supported in (False, True):
            caps = ClientCapabilities(
                extensions={"io.modelcontextprotocol/ui": {"mimeTypes": [MIME]}}
                if supported
                else {}
            )
            ctx = Context(
                request_context=SimpleNamespace(session=SimpleNamespace(client_capabilities=caps))
            )
            out = fn(ctx, CASE, "r1", "auto").model_dump(mode="json")
            assert out["presentation"] == ("inline" if supported else "browser_link")
            assert out["browser_url"].startswith("http://127.0.0.1:")
            # compact for a host's tool-output limit (Claude Code shows structured content): no
            # evidence; the embedded app reads the pinned snapshot itself
            assert out["version"] == "kyb-dashboard-open-v1" and "selected" not in out
            assert (out["case"], out["run"]) == (CASE, "r1") and len(json.dumps(out)) < 20_000
            assert out["browser_url"].startswith("http://127.0.0.1:")
        assert snapshot(real_workspace) == before

    asyncio.run(run())


@pytest.mark.parametrize("name", ["open_dashboard", "dashboard_snapshot"])
def test_no_workspace_reaches_the_agent_with_its_message(name):
    """A new user's first dashboard call, before init: the refusal must say what to do next."""
    from mcp.server.apps import Apps
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError

    def none():
        raise RuntimeError("no kyb workspace resolves (none): call init_workspace first")

    server = MCPServer("dashboard-test", extensions=[Apps()])
    register(server, none)
    with pytest.raises(ToolError, match="call init_workspace first"):
        asyncio.run(server.call_tool(name, {}))


def test_ui_has_no_execution_tool_and_preserves_offline_materials():
    html = app_html()
    assert 'data-view="overview"' in html and "data-theme=light" in html
    assert "frame-ancestors" not in html  # Browser sets headers; MCP host owns its sandbox.
    assert "start_loop_repair" not in html and "sign_decision" not in html
    assert "https://" not in html and "/* INTER_FONT */" not in html


def test_actual_stdio_client_discovers_app_and_falls_back_without_host_support(tmp_path):
    from pathlib import Path

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    from kyb.workspace.init import init_workspace

    repo = Path(__file__).resolve().parents[1]
    root = tmp_path / "ws"
    init_workspace(
        root,
        tenant="dashboard-protocol-test",
        policy_src=repo / "examples/policy/compliance_rules.md",
    )

    async def run():
        for supported in (False, True):
            params = StdioServerParameters(
                command=sys.executable,
                args=["-m", "kyb.mcp_server"],
                cwd=str(repo),
                env={**os.environ, "KYB_WORKSPACE": str(root)},
            )
            ext = {"io.modelcontextprotocol/ui": {"mimeTypes": [MIME]}} if supported else {}
            async with (
                stdio_client(params) as (read, write),
                ClientSession(read, write, extensions=ext) as session,
            ):
                init = await session.discover() if supported else await session.initialize()
                assert "kyb-dashboard" in init.instructions
                assert "open_dashboard" in init.instructions
                assert "intent, not assent" in init.instructions
                if supported:
                    assert "io.modelcontextprotocol/ui" in init.capabilities.extensions
                result = await session.call_tool("open_dashboard", {"surface": "auto"})
                assert not result.is_error
                out = result.structured_content
                assert out["presentation"] == ("inline" if supported else "browser_link")
                resource = await session.read_resource(URI)
                assert resource.contents[0].mime_type == MIME
                assert "open-kyb" in resource.contents[0].text
                refused = await session.call_tool(
                    "dashboard_snapshot", {"case": "../foreign", "run": "r1"}
                )
                assert refused.is_error

    asyncio.run(run())


def test_apps_visibility_exposes_only_inspection_and_preparation_to_the_app():
    """An Apps host forwards an app's tools/call by advertised visibility, not our JS allowlist."""
    from kyb.mcp_server import server

    tools = {t.name: (t.meta or {}).get("ui") or {} for t in asyncio.run(server.list_tools())}
    app = {name for name, ui in tools.items() if "app" in ui.get("visibility", ["model", "app"])}
    assert app == {
        "dashboard_snapshot",
        "prepare_dashboard_action",
        "dashboard_handoff",
        "dashboard_intent",
    }
    for name in ("start_loop_repair", "sign_decision", "archive_case", "restore_case"):
        assert tools[name]["visibility"] == ["model"]
    assert tools["open_dashboard"] == {
        "resourceUri": "ui://kyb/orchestration-dashboard",
        "visibility": ["model"],
    }


HOST_PROBE = r"""
const vm = require('vm'), fs = require('fs');
const sent = [], listeners = {};
const stub = () => new Proxy(function () {}, {
  get: (t, k) => (k in t ? t[k] : (t[k] = stub())),
  set: (t, k, v) => ((t[k] = v), true),
  apply: () => stub(),
});
const nodes = {};
const g = {
  console, URLSearchParams, structuredClone, setImmediate,
  setTimeout: () => 0, clearTimeout: () => {}, requestAnimationFrame: () => 0,
  matchMedia: () => ({ matches: false, addEventListener() {} }),
  ResizeObserver: class { observe() {} },
  location: { search: '', href: 'http://host.invalid/' }, history: { replaceState() {} },
  HTMLElement: class {}, Element: class {}, HTMLInputElement: class {},
  HTMLTextAreaElement: class {}, HTMLButtonElement: class {}, HTMLSelectElement: class {},
  parent: { postMessage: (m) => sent.push(m) },
  addEventListener: (type, fn) => { listeners[type] = fn; },
  document: {
    getElementById: (id) => (id === 'preview-data' ? null : nodes[id] || (nodes[id] = stub())),
    querySelector: () => stub(), querySelectorAll: () => [], addEventListener() {},
    documentElement: { dataset: {}, classList: { add() {} }, style: { setProperty() {} } },
  },
};
g.window = g;
vm.runInNewContext(fs.readFileSync(process.argv[2], 'utf8'), g);
const deliver = (data) => listeners.message({ source: g.parent, data });
(async () => {
  const init = sent.find((m) => m.method === 'ui/initialize');
  deliver({ jsonrpc: '2.0', id: init.id, method: 'ping' });
  await new Promise((r) => setImmediate(r));
  const caps = JSON.parse(process.argv[4] || '{}'), early = process.argv[5] === 'early';
  const extra = JSON.parse(process.argv[3] || '[]');
  if (early) for (const m of extra) deliver(m);  // a host may send the result before init returns
  deliver({ jsonrpc: '2.0', id: init.id, result: { hostCapabilities: caps, hostContext: {} } });
  await new Promise((r) => setImmediate(r));
  if (!early) for (const m of extra) deliver(m);
  await new Promise((r) => setImmediate(r));
  const ui = { browserHidden: nodes.browser && nodes.browser.hidden,
               connection: nodes.connection && nodes.connection.innerHTML };
  console.log(JSON.stringify({ init: init.id, sent, ui }));
})();
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_host_request_with_a_colliding_id_does_not_answer_the_app(tmp_path):
    """A host `ping` reusing the app's pending id is answered; it never resolves initialization."""
    probe = tmp_path / "probe.js"
    probe.write_text(HOST_PROBE)
    js = Path(__file__).resolve().parents[1] / "src/kyb/dashboard/assets/dashboard.js"
    out = subprocess.run(
        ["node", str(probe), str(js)], capture_output=True, text=True, timeout=30, check=True
    )
    result = json.loads(out.stdout.strip().splitlines()[-1])
    sent = result["sent"]
    assert {"jsonrpc": "2.0", "id": result["init"], "result": {}} in sent
    assert any(m.get("method") == "ui/notifications/initialized" for m in sent)


def _open_in_host(tmp_path, caps, order="late"):
    probe = tmp_path / "probe.js"
    probe.write_text(HOST_PROBE)
    js = Path(__file__).resolve().parents[1] / "src/kyb/dashboard/assets/dashboard.js"
    opened = {"version": "kyb-dashboard-open-v1", "case": CASE, "run": "r1",
              "browser_url": "http://127.0.0.1:1/token/"}  # fmt: skip
    note = [{"jsonrpc": "2.0", "method": "ui/notifications/tool-result",
             "params": {"structuredContent": opened}}]  # fmt: skip
    out = subprocess.run(["node", str(probe), str(js), json.dumps(note), json.dumps(caps), order],
                         capture_output=True, text=True, timeout=30, check=True)  # fmt: skip
    result = json.loads(out.stdout.strip().splitlines()[-1])
    calls = [m["params"] for m in result["sent"] if m.get("method") == "tools/call"]
    return calls, result["ui"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize("order", ["late", "early"])
def test_compact_open_result_makes_the_app_read_its_pinned_snapshot(tmp_path, order):
    """open_dashboard's reply is compact; an MCP Apps host proxies the app's own snapshot read,
    also when the result arrives before initialization completes."""
    calls, _ = _open_in_host(tmp_path, {"serverTools": {}}, order)
    assert calls and calls[-1]["name"] == "dashboard_snapshot"
    assert calls[-1]["arguments"]["case"] == CASE and calls[-1]["arguments"]["run"] == "r1"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize("links", [True, False])
def test_compact_open_without_a_tool_proxy_offers_the_browser_alternative(tmp_path, links):
    """No serverTools: no record can load inline, so the panel says so and keeps the browser link
    (as a button when the host can open links)."""
    calls, ui = _open_in_host(tmp_path, {"openLinks": {}} if links else {})
    assert calls == []
    assert "Tool connection unavailable" in ui["connection"]
    assert ui["browserHidden"] is (not links)
    assert ("browser alternative" if links else "browser link your agent shared") in ui[
        "connection"
    ]
