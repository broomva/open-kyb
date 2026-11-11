"""`make dogfood` (scripts/dogfood.py): the harness run is confined to its own project, and a run
that did not finish never passes the gate (PR #14 review)."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("dogfood", REPO / "scripts" / "dogfood.py")
dogfood = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dogfood)


def _forwarded(cmd: list[str]) -> list[str]:
    value = next(c for c in cmd if c.startswith("mcp_servers.kyb.env_vars="))
    return json.loads(value.split("=", 1)[1])


def test_codex_forwards_the_discovery_bound_through_its_env_filter(tmp_path, monkeypatch):
    """Codex starts MCP servers with a filtered environment: KYB_DISCOVERY_ROOT must be among
    the forwarded names, or the server's discovery climbs to an enclosing workspace."""
    outer = tmp_path / "home"
    (outer / "workspaces" / "old").mkdir(parents=True)
    (outer / "workspaces" / "old" / "template.lock").write_text("{}")
    project = outer / "runs" / "r1" / "project"
    project.mkdir(parents=True)
    monkeypatch.setenv("KYB_WORKSPACE", str(outer / "workspaces" / "old"))  # stripped
    env = dogfood._env(project)
    assert "KYB_WORKSPACE" not in env and env["KYB_DISCOVERY_ROOT"] == str(project.resolve())
    names = _forwarded(dogfood._codex_cmd(tmp_path, "p"))
    assert "KYB_DISCOVERY_ROOT" in names
    # what the MCP server sees under Codex's filter: only the forwarded names (and PATH)
    server_env = {k: env[k] for k in names if k in env} | {"PATH": os.environ["PATH"]}
    probe = ("import sys; from pathlib import Path; from kyb.workspace.guard import discover;"
             "print(discover(Path(sys.argv[1]).resolve()))")  # fmt: skip
    out = subprocess.run([sys.executable, "-c", probe, str(project)], env=server_env,
                         capture_output=True, text=True, check=True)  # fmt: skip
    assert out.stdout.strip() == "(None, 'none')"


def test_a_harness_run_that_did_not_finish_fails_the_gate():
    ws = {"workspace": "workspaces/kyb-demo", "documents": {}, "next": "decide",
          "audit_ok": True}  # fmt: skip
    cut_short = {"exit": 1, "subtype": "error_max_turns"}
    fails = dogfood._gate("claude", 0, ws, cut_short)
    assert any("exited 1" in f for f in fails) and any("did not finish" in f for f in fails)
    assert dogfood._gate("codex", 0, ws, {"exit": None, "error": "FileNotFoundError: codex"})


def test_decided_gate_requires_a_valid_pending_signature_record():
    ws = {"workspace": "workspaces/kyb-demo", "documents": {}, "next": "decided", "audit_ok": True}
    agent = {"exit": 0}
    assert dogfood._gate("codex", 0, ws, agent)
    for state in ("signed", None):
        assert dogfood._gate(
            "codex", 0, {**ws, "decision": {"valid": True, "signature_state": state}}, agent
        )
    assert (
        dogfood._gate(
            "codex",
            0,
            {
                **ws,
                "decision": {
                    "valid": True,
                    "signature_state": "pending_signature",
                    "fixture_inputs": [],
                },
            },
            agent,
        )
        == []
    )
    assert dogfood._gate("codex", 0, {**ws, "next": "decide"}, agent) == []


def test_decided_fixture_inputs_fail_the_live_gate():
    for fixture in ("gold/facts/c/r1.json", "gold/screening/c/r1.json"):
        ws = {
            "workspace": "workspaces/kyb-demo",
            "documents": {},
            "next": "decided",
            "audit_ok": True,
            "decision": {
                "valid": True,
                "signature_state": "pending_signature",
                "fixture_inputs": [fixture],
            },
        }
        assert any("fixture-produced" in f for f in dogfood._gate("codex", 0, ws, {"exit": 0}))


def test_decided_union_keeps_l8_stratum_gate_and_l4_decision_checks():
    """After the L8 merge, a decided stratum-B run passes only with both halves of the gate."""
    from test_witness_strata import gate_document

    decision = {"valid": True, "signature_state": "pending_signature", "fixture_inputs": []}
    ws = {"workspace": "ws", "next": "decided", "audit_ok": True, "stratum": "B",
          "documents": {"01": gate_document()}, "decision": decision}  # fmt: skip
    agent = {"exit": 0, "subtype": "success"}
    assert dogfood._gate("claude", 1, ws, agent) == []
    # L8's stratum evidence still fails a decided run.
    for change in ({"assurance": "W2"}, {"self_referential": False}, {"stratum": "A"}):
        broken = {**ws, "documents": {"01": {**gate_document(), **change}}}
        fails = dogfood._gate("claude", 1, broken, agent)
        assert any("stratum" in f or "not W1" in f for f in fails)
    # L4's decision checks still fail a stratum-valid run.
    for bad in ({**decision, "fixture_inputs": ["gold/facts/c/r1.json"]},
                {**decision, "signature_state": "signed"}, None):  # fmt: skip
        assert dogfood._gate("claude", 1, {**ws, "decision": bad}, agent)
    assert "KYB_WITNESS_STRATUM" in dogfood.CODEX_FORWARDED_ENV
    assert "KYB_WITNESS_A_BACKENDS" in dogfood.CODEX_FORWARDED_ENV
    assert "GEMINI_API_KEY" not in dogfood.CODEX_FORWARDED_ENV
    assert {"predicate_work_order", "propose_predicates", "decide"} <= set(dogfood.KYB_TOOLS)


def test_precheck_passes_when_gcloud_present_and_gemini_api_key_unset(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("KYB_WITNESS_STRATUM", "A")
    monkeypatch.setattr(
        dogfood.shutil, "which", lambda cmd: "/usr/bin/gcloud" if cmd == "gcloud" else None
    )
    assert dogfood._precheck() is None
    assert dogfood._precheck({"KYB_WITNESS_STRATUM": "A"}) is None


def test_precheck_errors_when_stratum_a_and_gcloud_missing(monkeypatch):
    monkeypatch.setenv("KYB_WITNESS_STRATUM", "A")
    monkeypatch.setattr(dogfood.shutil, "which", lambda cmd: None)
    expected_error = (
        "gcloud not found: the stratum-A witness uses the approved Vertex route "
        "(application-default credentials)"
    )
    assert dogfood._precheck() == expected_error
    assert dogfood._precheck({"KYB_WITNESS_STRATUM": "A"}) == expected_error


def test_precheck_passes_when_stratum_b_and_gcloud_missing(monkeypatch):
    monkeypatch.setenv("KYB_WITNESS_STRATUM", "B")
    monkeypatch.setattr(dogfood.shutil, "which", lambda cmd: None)
    assert dogfood._precheck() is None
    assert dogfood._precheck({"KYB_WITNESS_STRATUM": "B"}) is None
    assert dogfood._precheck({}) is None
