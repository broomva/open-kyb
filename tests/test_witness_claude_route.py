"""Claude locked-down headless route: recorded probe streams (scrubbed), offline only."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_witness_strata import SPEC

from kyb.witness import adapters
from kyb.witness.errors import WitnessRefused

FIX = Path(__file__).parent / "fixtures" / "claude_cli"
PROBE = Path(__file__).parents[1] / "tests/fixtures/claude_cli/route-probe.jsonl"
SCHEMA = json.loads((FIX / "schema.json").read_text())


def events(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def run(monkeypatch, evs, model):
    def fake(cmd, **kwargs):
        return SimpleNamespace(
            returncode=0, stdout="\n".join(map(json.dumps, evs)).encode(), stderr=b""
        )

    monkeypatch.setattr(adapters.subprocess, "run", fake)
    w = adapters.ClaudeWitness(model)
    return w, w.run("p", [b"png"], SCHEMA)


def test_command_is_locked_down_and_never_bare():
    cmd = adapters.ClaudeWitness("claude-sonnet-4-5").command(SCHEMA)
    assert "--bare" not in cmd
    for flag in ("--restricted", "--safe-mode", "--strict-mcp-config", "--no-session-persistence"):
        assert flag in cmd
    assert cmd[cmd.index("--tools") + 1] == "" and cmd[cmd.index("--setting-sources") + 1] == ""
    assert cmd[cmd.index("--input-format") + 1] == cmd[cmd.index("--output-format") + 1]


def test_recorded_live_probe_is_accepted_and_served_model_is_message_model(monkeypatch):
    evs = events(PROBE)
    init = next(e for e in evs if e.get("subtype") == "init")
    assert init["tools"] == ["StructuredOutput"] and init["mcp_servers"] == []
    w, (data, _) = run(monkeypatch, evs, "claude-haiku-4-5")
    assert data["canary_contents"] == "UNREADABLE"
    assert w.last_provenance["reported_model"] == "claude-haiku-4-5-20251001"


def test_haiku_side_call_in_model_usage_never_becomes_recorded_model(monkeypatch):
    evs = events(FIX / "p2.stream.jsonl")
    assert "claude-haiku-5-5" in next(e for e in evs if e["type"] == "result")["modelUsage"]
    # This recorded stream also shows the model attempting denied Read/Bash calls: refused.
    w = adapters.ClaudeWitness("claude-sonnet-4-5")
    monkeypatch.setattr(
        adapters.subprocess,
        "run",
        lambda cmd, **kw: SimpleNamespace(
            returncode=0, stdout="\n".join(map(json.dumps, evs)).encode(), stderr=b""
        ),
    )
    with pytest.raises(WitnessRefused):
        w.run("p", [b"png"], SCHEMA)
    assert w.last_provenance["reported_models"] == ["claude-sonnet-4-5-20250929"]
    assert w.last_provenance["reported_model"] == "claude-sonnet-4-5-20250929"


@pytest.mark.parametrize("tools", [["Read"], ["StructuredOutput", "Bash"], ["mcp__x__y"]])
def test_init_tools_beyond_structured_output_refuse(monkeypatch, tools):
    evs = events(PROBE)
    next(e for e in evs if e.get("subtype") == "init")["tools"] = tools
    with pytest.raises(WitnessRefused):
        run(monkeypatch, evs, "claude-haiku-4-5")


def test_empty_tools_list_is_accepted(monkeypatch):
    evs = events(PROBE)
    next(e for e in evs if e.get("subtype") == "init")["tools"] = []
    run(monkeypatch, evs, "claude-haiku-4-5")


def test_missing_assistant_model_refuses(monkeypatch):
    evs = [e for e in events(PROBE) if e["type"] != "assistant"]
    with pytest.raises(WitnessRefused):
        run(monkeypatch, evs, "claude-haiku-4-5")


def test_served_model_not_matching_request_refuses(monkeypatch):
    with pytest.raises(WitnessRefused):
        run(monkeypatch, events(PROBE), "claude-sonnet-4-5")


def test_modelusage_cannot_substitute_for_a_missing_message_model(monkeypatch):
    evs = events(PROBE)
    for e in evs:
        if e["type"] == "assistant":
            e["message"]["model"] = None
    with pytest.raises(WitnessRefused):
        run(monkeypatch, evs, "claude-haiku-4-5")


def test_codex_and_agy_stay_fail_closed():
    for cls in (adapters.CodexWitness, adapters.AgyWitness):
        w = cls("x")
        with pytest.raises(WitnessRefused):
            w.call("transcribe", [b"c"], SPEC)
        assert w.last_provenance["reported_model"] is None
