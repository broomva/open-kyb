"""P0-c substrate (spec, spec): Python 3.15 lazy imports, the Typer CLI, the `kyb-hook` fast
lane, and the TOML ontology."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from kyb import hook, ontology
from kyb.audit.chain import AuditLog, verify_workspace
from kyb.util.hashing import sha256_bytes, sha256_file
from kyb.workspace.init import init_workspace

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "examples"
PLUGIN = REPO / "plugin"


def _modules() -> list[str]:
    import pkgutil

    import kyb

    return sorted(m.name for m in pkgutil.walk_packages(kyb.__path__, "kyb."))


# --- lazy imports: nothing fails late ---------------------------------------------------------


def test_every_module_imports_with_lazy_imports_disabled():
    """Importing a module does not load its lazy imports, so a broken one would fail only at first
    use. With the lazy-imports filter refusing every lazy import, each one runs eagerly here."""
    code = (
        "import importlib, sys\n"
        "sys.set_lazy_imports_filter(lambda *a: False)\n"
        f"for m in {_modules()!r}:\n"
        "    importlib.import_module(m)\n"
        "assert 'pydantic' in sys.modules  # really eager: the chain's lazy pydantic loaded\n"
    )
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO)
    assert p.returncode == 0, p.stderr


def _lazy_names() -> dict[str, list[str]]:
    """Every name a `lazy` statement binds, by module, read from the source itself."""
    import ast
    import importlib.util

    out: dict[str, list[str]] = {}
    for name in _modules():
        origin = importlib.util.find_spec(name).origin
        if not origin or not origin.endswith(".py"):
            continue
        for node in ast.parse(Path(origin).read_text("utf-8")).body:
            if isinstance(node, (ast.Import, ast.ImportFrom)) and node.is_lazy:
                bound = [a.asname or a.name.split(".")[0] for a in node.names]
                out.setdefault(name, []).extend(bound)
    return out


def test_every_lazy_name_resolves():
    """In a fresh process, every name bound by a `lazy` statement resolves: a proxy still pending
    is resolved, and one already reified was resolved on first use."""
    names = _lazy_names()
    assert sum(map(len, names.values())) >= 40  # the hook path, the CLI groups, the MCP server
    code = (
        "import importlib, types\n"
        "def sweep(names):  # locals: a global lookup of a proxy would reify it (PEP 810)\n"
        "    for module, bound in names.items():\n"
        "        mod = importlib.import_module(module)\n"
        "        for attr in bound:\n"
        "            value = vars(mod)[attr]  # vars() does not reify\n"
        "            if isinstance(value, types.LazyImportType):\n"
        "                value = value.resolve()\n"
        "            assert value is not None and getattr(mod, attr) is value, (module, attr)\n"
        f"sweep({names!r})\n"
    )
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO)
    assert p.returncode == 0, p.stderr


def _imported(argv: list[str], event: dict, env: dict | None = None) -> tuple[int, set[str]]:
    """Run a hook process for real and return its exit code and every module it imported."""
    p = subprocess.run([sys.executable, "-X", "importtime", *argv], input=json.dumps(event),
                       capture_output=True, text=True,
                       env={**os.environ, **(env or {})})  # fmt: skip
    mods = {line.split("|")[-1].strip() for line in p.stderr.splitlines()
            if line.startswith("import time:")}  # fmt: skip
    return p.returncode, mods


@pytest.fixture
def ws(tmp_path) -> Path:
    root = tmp_path / "ws"
    init_workspace(root, tenant="t", policy_src=ASSETS / "policy/compliance_rules.md")
    return root


@pytest.mark.parametrize("tool", ["Bash", "Read", "Write", "mcp__kyb__review_status"])
def test_hook_pre_on_a_non_submit_tool_never_loads_pydantic(ws, tool):
    event = {"tool_name": tool, "tool_input": {"command": "ls", "file_path": "cases/A/notes/n.md"},
             "cwd": str(ws), "session_id": "s", "tool_use_id": "t1"}  # fmt: skip
    code, mods = _imported(["-m", "kyb.hook", "pre", "--source", "workspace"], event,
                           {"CLAUDE_PROJECT_DIR": str(ws)})  # fmt: skip
    assert code == 0
    assert "kyb.workspace.guard" in mods  # the hook really ran
    assert "pydantic" not in mods and "typer" not in mods


def test_hook_post_records_without_pydantic(ws):
    """Recording a tool call appends with hashing and canonical JSON only (spec)."""
    event = {"tool_name": "Bash", "tool_input": {"command": "ls"}, "tool_response": {},
             "cwd": str(ws), "session_id": "s", "tool_use_id": "t1"}  # fmt: skip
    code, mods = _imported(["-m", "kyb.hook", "post", "--source", "workspace"], event,
                           {"CLAUDE_PROJECT_DIR": str(ws)})  # fmt: skip
    assert code == 0 and "pydantic" not in mods
    calls = [e for e in _events(ws) if e["action"] == "tool.call"]
    assert [c["detail"] for c in calls] == [
        {"tool": "Bash", "session": "s", "tool_use_id": "t1", "hook_source": "workspace"}
    ]
    assert verify_workspace(ws, at_boundary=True).ok


def _events(ws: Path) -> list[dict]:
    raw = (ws / "ledger/audit.jsonl").read_text("utf-8")
    return [json.loads(line) for line in raw.splitlines()]


# --- the chain's contract gate is unchanged ----------------------------------------------------

_GOOD = {"seq": 0, "timestamp": "2026-10-25T15:00:00.000000Z", "component": "c", "action": "a",
         "input_hash": "i", "output_hash": "o", "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0,
         "detail": {}, "prev_hash": "GENESIS", "hash": "h"}  # fmt: skip


@pytest.mark.parametrize("change", [
    {}, {"seq": "0"}, {"seq": True}, {"seq": 1.0}, {"seq": 10**30}, {"tokens_in": None},
    {"cost_usd": "0.1"}, {"cost_usd": 1}, {"cost_usd": float("nan")}, {"cost_usd": True},
    {"cost_usd": 10**400}, {"cost_usd": 10**308}, {"cost_usd": 2**1024}, {"cost_usd": 1e308},
    {"cost_usd": float("inf")}, {"seq": 10**400}, {"tokens_out": -(10**400)},
    {"detail": []}, {"detail": {"k": "\ud800"}}, {"component": 5}, {"component": "\ud800"},
    {"hash": None}, {"extra": 1}, {"timestamp": "2026-10-25T15:00:00Z"},
    {"timestamp": "2026-10-25T24:00:00.000000Z"}, {"timestamp": "2026-02-30T15:00:00.000000Z"},
    {"timestamp": "2026-10-25T15:00:00.000000+00:00"}, {"timestamp": 1761490800},
    {"timestamp": "2026-10-25T15:00:00.000000"}, {"timestamp": "２０２６-10-26T15:00:00.000000Z"},
    {"timestamp": "0000-01-01T00:00:00.000000Z"},
])  # fmt: skip
def test_the_fast_contract_gate_agrees_with_the_model(change):
    """`check_event` admits a record without pydantic only when the contract model would too, and
    leaves every other verdict to the model; `canonical_ts` agrees with the model's parse."""
    from kyb.audit import chain
    from kyb.contracts.kyb_decision_schema import AuditEvent

    ev = {**_GOOD, **change}
    try:
        AuditEvent.model_validate(ev)
        model_ok = True
    except ValueError:
        model_ok = False
    try:
        chain.check_event(ev)
        gate_ok = True
    except ValueError:
        gate_ok = False
    assert gate_ok == model_ok
    if chain._plainly_valid(ev):
        assert model_ok
        dt = chain._datetime_adapter().validate_python(ev["timestamp"])
        assert chain.canonical_ts(ev["timestamp"]) == dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def test_append_still_returns_the_contract_model(tmp_path):
    from kyb.contracts.kyb_decision_schema import AuditEvent

    ev = AuditLog(tmp_path / "a.jsonl").append("c", "a", input_hash="i", output_hash="o")
    assert isinstance(ev, AuditEvent) and ev.seq == 0
    with pytest.raises(ValueError):
        AuditLog(tmp_path / "b.jsonl").append("c", "a", input_hash="i", output_hash="o",
                                              tokens_in="many")  # fmt: skip
    assert not (tmp_path / "b.jsonl").exists() or not (tmp_path / "b.jsonl").read_text()


# --- kyb-hook: every generated hook command parses and dispatches ------------------------------

# the harness event each registration answers, and the kyb-hook event it must name
HARNESS_EVENTS = {"PreToolUse": "pre", "PostToolUse": "post", "SessionStart": "session-start",
                  "SubagentStart": "subagent-start", "SubagentStop": "subagent-stop",
                  "Stop": "stop", "SessionEnd": "ingest-session"}  # fmt: skip


def _commands(hooks: dict) -> dict[str, str]:
    out = {}
    for name, regs in hooks.items():
        [reg] = regs
        [h] = reg["hooks"]
        out[name] = h["command"]
    return out


def _payload(name: str, ws: Path) -> dict:
    base = {"cwd": str(ws), "session_id": f"s-{name}"}
    return base | {
        "PreToolUse": {"tool_name": "Bash", "tool_input": {"command": "ls"}, "tool_use_id": "t1"},
        "PostToolUse": {"tool_name": "Bash", "tool_input": {"command": "ls"}, "tool_use_id": "t1",
                        "tool_response": {}},
        "SessionStart": {"model": "claude-opus-5-5", "source": "startup"},
        "SubagentStart": {"agent_id": "a1", "agent_type": "general-purpose"},
        "SubagentStop": {"agent_id": "a1", "agent_type": "general-purpose"},
        "Stop": {},
        "SessionEnd": {},
    }[name]  # fmt: skip


# what each event leaves in the chain (pre on a plain tool and stop record nothing)
RECORDED = {"PostToolUse": "tool.call", "SessionStart": "harness.session_start",
            "SubagentStart": "harness.subagent_start",
            "SubagentStop": "harness.subagent_stop"}  # fmt: skip


def _dispatch(ws: Path, commands: dict[str, str], source: str, env: dict) -> None:
    assert set(commands) == set(HARNESS_EVENTS)
    for name, command in commands.items():
        argv = shlex.split(command)
        assert hook.parse(argv[-3:]) == (HARNESS_EVENTS[name], source), command
        before = len(_events(ws))
        p = subprocess.run(argv, input=json.dumps(_payload(name, ws)), capture_output=True,
                           text=True, env=env, cwd=ws)  # fmt: skip
        assert p.returncode == 0, (command, p.stderr)
        new = _events(ws)[before:]
        if name in RECORDED:
            assert [e["action"] for e in new] == [RECORDED[name]], command
            assert new[0]["detail"]["hook_source"] == source
        else:
            assert new == []
    assert verify_workspace(ws, at_boundary=True).ok


def test_the_workspace_hooks_parse_and_dispatch(ws):
    settings = json.loads((ws / ".claude/settings.json").read_text())
    commands = _commands(settings["hooks"])
    for command in commands.values():  # the interpreter that ran init, the fast lane
        assert shlex.split(command)[:3] == [sys.executable, "-m", "kyb.hook"]
    _dispatch(ws, commands, "workspace", {"PATH": "/usr/bin:/bin"})  # no venv on PATH


def test_the_plugin_hooks_parse_and_dispatch(ws):
    if shutil.which("uv") is None:
        pytest.skip("uv is not installed")
    raw = json.loads((PLUGIN / "hooks/hooks.json").read_text())["hooks"]
    commands = {k: v.replace("${CLAUDE_PLUGIN_ROOT}", str(PLUGIN))
                for k, v in _commands(raw).items()}  # fmt: skip
    for command in commands.values():
        assert shlex.split(command)[0] == str(PLUGIN / "bin/kyb-hook")
    env = {**os.environ, "KYB_DEV_REPO": str(REPO), "CLAUDE_PROJECT_DIR": str(ws)}
    env.pop("KYB_WORKSPACE", None)
    _dispatch(ws, commands, "plugin", env)


@pytest.mark.parametrize("argv", [
    [], ["pre"], ["pre", "--source"], ["pre", "--source", "forged"], ["pre", "--source="],
    ["bogus", "--source", "plugin"], ["pre", "post", "--source", "plugin"],
    ["pre", "--source", "plugin", "--verbose"],
])  # fmt: skip
def test_kyb_hook_refuses_anything_else_with_exit_2(argv, capsys):
    assert hook.main(argv) == 2
    assert "usage: kyb-hook" in capsys.readouterr().err


def test_kyb_hook_accepts_both_spellings():
    assert hook.parse(["pre", "--source", "plugin"]) == ("pre", "plugin")
    assert hook.parse(["--source=workspace", "subagent-stop"]) == ("subagent-stop", "workspace")


def test_a_hook_that_denies_still_exits_2_on_both_entry_points(ws):
    event = json.dumps({"tool_name": "Write", "tool_input": {"file_path": "gold/x.json"},
                        "cwd": str(ws)})  # fmt: skip
    for argv in (["-m", "kyb.hook", "pre", "--source", "workspace"],
                 ["-m", "kyb.cli", "guard", "pre", "--source", "workspace"]):  # fmt: skip
        p = subprocess.run([sys.executable, *argv], input=event, capture_output=True, text=True)
        assert p.returncode == 2 and "agents may write only" in p.stderr


# --- the plugin wrappers bootstrap Python 3.15 -------------------------------------------------


def _older_python() -> str | None:
    p = subprocess.run(["uv", "python", "find", "3.12"], capture_output=True, text=True)
    return "3.12" if p.returncode == 0 else None


@pytest.mark.parametrize("path", ["dev", "uvx"])
def test_the_plugin_wrappers_bootstrap_315_under_an_older_default(path, tmp_path):
    """`uvx` ignores the checkout's .python-version; the wrappers pass `--python 3.15`, so a
    machine whose default interpreter is older still runs kyb (whose `lazy` syntax needs 3.15)."""
    if shutil.which("uv") is None or _older_python() is None:
        pytest.skip("needs uv and a Python 3.12 it can find")
    env = {**os.environ, "UV_PYTHON": "3.12"}
    env.pop("KYB_WORKSPACE", None)
    env |= {"KYB_DEV_REPO": str(REPO)} if path == "dev" else {"KYB_SOURCE": str(REPO)}
    if path == "uvx":
        env.pop("KYB_DEV_REPO", None)
    for wrapper, args in (("kyb-hook", ["stop", "--source", "plugin"]), ("kyb", ["--help"]),
                          ("kyb-mcp", ["--help"])):  # fmt: skip
        p = subprocess.run([str(PLUGIN / "bin" / wrapper), *args], input="{}", text=True,
                           capture_output=True, env=env, cwd=tmp_path, timeout=300)  # fmt: skip
        assert p.returncode == 0, (wrapper, p.stderr)
    # the control: without --python, the same default cannot run kyb at all
    p = subprocess.run(["uvx", "--quiet", "--from", str(REPO), "kyb-hook", "stop", "--source",
                        "plugin"], input="{}", text=True, capture_output=True, env=env,
                       cwd=tmp_path, timeout=300)  # fmt: skip
    assert p.returncode != 0


def test_the_plugin_validates():
    if shutil.which("claude") is None:
        pytest.skip("Claude Code is not installed")
    p = subprocess.run(["claude", "plugin", "validate", str(PLUGIN)], capture_output=True,
                       text=True, timeout=120)  # fmt: skip
    assert p.returncode == 0, p.stdout + p.stderr


# --- the Typer CLI keeps every command, option and exit code -----------------------------------


def _click_tree() -> dict:
    """The Typer CLI's structure, in the form `tests/snapshots/cli_tree.json` was taken from the
    argparse CLI it replaced: positionals, flags, VALUE or FLAG, required, choices, default."""
    import typer

    from kyb.cli import app

    # Typer's own completion options are new and deliberate (spec), not a changed command
    typer_own = {"--install-completion", "--show-completion"}

    def arg(p) -> str | None:
        if p.param_type_name == "argument":
            return f"<{p.name}>"
        if typer_own & set(p.opts):
            return None
        s = "/".join(sorted(p.opts)) + (" FLAG" if p.is_flag else " VALUE")
        s += " required" if p.required else ""
        if choices := getattr(p.type, "choices", None):
            s += " choices=" + "|".join(sorted(choices))
        if not p.is_flag and p.default is not None:
            s += f" default={p.default}"
        return s

    out = {}

    def walk(cmd, path):
        args = [a for a in map(arg, cmd.params) if a is not None]
        sub = getattr(cmd, "commands", None)
        out[" ".join(path)] = {"help": cmd.help or "", "args": args, "group": bool(sub)}
        for name, child in (sub or {}).items():
            walk(child, [*path, name])

    walk(typer.main.get_command(app), ["kyb"])
    return out


def test_the_cli_keeps_every_command_and_option():
    snapshot = json.loads((REPO / "tests/snapshots/cli_tree.json").read_text())
    assert _click_tree() == snapshot


@pytest.mark.parametrize("argv", [[], ["audit"], ["bogus"], ["review", "B"], ["claims", "submit",
                         "B", "--doc", "d"], ["claims", "verify", "B", "--witness", "nope"],
                         ["guard", "pre", "--source", "forged"], ["init", "x"]])  # fmt: skip
def test_usage_errors_still_exit_2(argv):
    from kyb.cli import main

    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2


def test_command_exit_codes_are_unchanged(ws, tmp_path, capsys, monkeypatch):
    import io

    from kyb.cli import main

    with pytest.raises(SystemExit) as exc:  # a refusal message: Python prints it, exit 1
        main(["-w", str(tmp_path / "nowhere"), "audit", "verify"])
    assert "no workspace" in str(exc.value.code)
    assert main(["-w", str(ws), "audit", "verify"]) == 0
    assert main(["-w", str(ws), "review", "B", "--status"]) == 0
    with pytest.raises(SystemExit) as exc:  # help exits 0, as argparse's did
        main(["-h"])
    assert exc.value.code == 0 and "open-kyb KYB reviewer" in capsys.readouterr().out
    deny = json.dumps({"tool_name": "Write", "tool_input": {"file_path": "gold/x"},
                       "cwd": str(ws)})  # fmt: skip
    monkeypatch.setattr("sys.stdin", io.StringIO(deny))
    assert main(["guard", "pre", "--source", "workspace"]) == 2  # a hook's code is returned
    AuditLog(ws / "ledger/audit.jsonl").append("x", "y", input_hash="i", output_hash="o")
    assert main(["-w", str(ws), "audit", "verify"]) == 1


# --- the ontology is TOML with a pydantic schema -----------------------------------------------


def test_the_template_validates_and_its_json_schema_is_committed():
    onto = ontology.template()
    assert onto.version == "0.1.0" and len(onto.actions) == 23 and len(onto.reads) == 12
    assert onto.links["evidenced_by"].from_ == "*"
    committed = json.loads((REPO / "src/kyb/workspace/template/kyb.ontology.schema.json")
                           .read_text())  # fmt: skip
    assert committed == ontology.json_schema()  # regenerate: python -m kyb.ontology schema


def test_the_toml_is_the_yaml_it_replaced():
    """The same content as the 0.1.0 YAML: compared through the model, on the fields both have."""
    legacy = (REPO / "tests/fixtures/kyb.ontology-0.1.0.yaml").read_bytes()
    assert sha256_bytes(legacy) == ontology.LEGACY_SHA256
    text = legacy.decode()
    added_since = {"plan_loop", "start_loop_repair", "inspect_loop", "propose_policy",
                   "apply_policy"}  # fmt: skip
    for name in (*ontology.template().actions, *ontology.template().reads):
        if name in added_since:
            continue
        assert f"\n  {name}:" in text


@pytest.mark.parametrize("edit, why", [
    (lambda d: d["actions"]["pages"].update(writes=["policy/x"]), "outside the declared layers"),
    (lambda d: d["actions"]["pages"].update(invoker=["robot"]), "invoker"),
    (lambda d: d["actions"]["pages"].update(colour="red"), "colour"),
    (lambda d: d["actions"]["run_judges"].update(writes=["gold/proposals/**"]), "proposer"),
    (lambda d: d["actions"]["run_judges"].update(writes=["gold/**"]), "proposer"),
    (lambda d: d["actions"]["emit_decision"].update(writes=["gold/*/p1.json"]), "proposer"),
    (lambda d: d["actions"]["pages"].update(writes=["silver/../../policy/x.md"]), "plain"),
    (lambda d: d["actions"]["pages"].update(writes=["silver/./pages/**"]), "plain"),
    (lambda d: d["actions"]["pages"].update(writes=["silver//pages"]), "plain"),
    (lambda d: d["actions"]["pages"].update(writes=["silver\\..\\policy"]), "plain"),
    (lambda d: d["actions"]["pages"].update(writes=["/etc/passwd"]), "plain"),
    (lambda d: d["reads"].update(pages={"invoker": ["agent"], "transport": ["mcp"]}), "both"),
    (lambda d: d.update(tiers=["untrusted", "trusted"]), "tiers"),
])  # fmt: skip
def test_the_schema_refuses_a_bad_ontology(edit, why):
    data = tomllib.loads(ontology.template_bytes().decode())
    edit(data)
    with pytest.raises(ValueError, match=why):
        ontology.Ontology.model_validate(data)


def test_a_legacy_yaml_workspace_stays_readable(ws):
    """Workspaces from template 0.1.0 keep `ontology/kyb.ontology.yaml`, pinned in template.lock:
    kyb never parses YAML, recognises those bytes as the same ontology, and keeps working."""
    legacy = (REPO / "tests/fixtures/kyb.ontology-0.1.0.yaml").read_bytes()
    (ws / ontology.WORKSPACE_PATH).unlink()
    (ws / ontology.LEGACY_PATH).write_bytes(legacy)
    lock = json.loads((ws / "template.lock").read_text())
    lock.pop("ontology_path")
    lock |= {"version": "0.1.0", "ontology": sha256_bytes(legacy)}
    (ws / "template.lock").write_text(json.dumps(lock, indent=2) + "\n")

    assert ontology.workspace_ontology(ws) == ontology.template()
    assert sha256_file(ws / ontology.LEGACY_PATH) == lock["ontology"]  # the pin still holds
    event = {"tool_name": "Bash", "tool_input": {}, "tool_response": {}, "cwd": str(ws),
             "session_id": "s", "tool_use_id": "t1"}  # fmt: skip
    code, _ = _imported(["-m", "kyb.hook", "post", "--source", "workspace"], event)
    assert code == 0 and verify_workspace(ws, at_boundary=True).ok

    (ws / ontology.LEGACY_PATH).write_bytes(legacy + b"# edited\n")
    with pytest.raises(ValueError, match="not the 0.1.0 template"):
        ontology.workspace_ontology(ws)


@pytest.mark.parametrize("a, b, overlap", [
    ("gold/**", "gold/proposals/**", True), ("gold/proposals/**", "gold/judgements/**", False),
    ("**/current", "gold/facts/x", True), ("gold/*/p1.json", "gold/proposals/p1.json", True),
    ("gold/pro*/x", "gold/judgements/x", False), ("gold/a", "gold/a/**", True),
    ("gold/a", "gold/a/b", False), ("ledger/decisions/**", "gold/predicates/**", False),
])  # fmt: skip
def test_write_patterns_overlap_conservatively(a, b, overlap):
    assert ontology.may_overlap(a, b) is overlap and ontology.may_overlap(b, a) is overlap


# --- every migrated command dispatches in normal mode ------------------------------------------


def test_every_command_dispatches_with_its_options(tmp_path, monkeypatch, capsys):
    """Each Typer command forwards its arguments and options to the core and returns its exit
    code: real runs where they are cheap (init, ingest, pages, review, audit), spies elsewhere."""
    from kyb.cli import main
    from kyb.read.submit import SubmissionRejected
    from kyb.witness import port

    root = tmp_path / "ws"
    assert main(["init", str(root), "--tenant", "t", "--policy",
                 str(ASSETS / "policy/compliance_rules.md")]) == 0  # fmt: skip
    assert json.loads(capsys.readouterr().out)["tenant"] == "t"
    w = ["-w", str(root)]
    assert main([*w, "ingest", str(ASSETS / "packets/cl-andes-agro"), "--as-of",
                 "2026-10-25", "--case", "A"]) == 0  # fmt: skip
    assert "ingested A:" in capsys.readouterr().out
    assert main([*w, "pages", "A"]) == 0  # starts r1
    assert capsys.readouterr().out.startswith("A r1: ")
    assert main([*w, "review", "A", "--status"]) == 0
    assert json.loads(capsys.readouterr().out)["next"]

    calls: dict[str, tuple] = {}

    def spy(name, result):
        def fn(*args, **kwargs):
            calls[name] = (args, kwargs)
            if isinstance(result, Exception):
                raise result
            return result

        return fn

    monkeypatch.setattr(port, "make_witness", lambda ws, mode: ("witness", mode))
    monkeypatch.setattr("kyb.commands.reading.read_plan", spy("plan", [{"doc": "d"}]))
    # This dispatch spy returns a synthetic doc; actual repair admission has separate tests.
    monkeypatch.setattr("kyb.loop.action.get_repair_targets", lambda *a, **kw: [])
    monkeypatch.setattr("kyb.witness.layout.layout_case", spy("layout", ["d"]))
    monkeypatch.setattr("kyb.witness.verify.verify_case", spy("verify", {"verified": 1}))
    ws = root.resolve()

    assert main([*w, "read", "plan", "A", "--doc", "01"]) == 0
    assert calls["plan"][0] == (ws, "A", "r1", "01")  # the latest run, since --run was omitted
    assert main([*w, "layout", "A", "--run", "r1", "--witness", "replay"]) == 0
    assert calls["layout"][0] == (ws, "A", "r1", ("witness", "replay"))
    assert main([*w, "claims", "verify", "A", "--witness", "live"]) == 0
    assert calls["verify"][0] == (ws, "A", "r1", ("witness", "live"))
    capsys.readouterr()

    body = tmp_path / "claims.json"
    body.write_text(json.dumps({"doc_sha256": "sha256:x", "claims": [{"c": 1}]}))
    monkeypatch.setattr("kyb.commands.reading.submit_claims", spy("submit", {"accepted": 1}))
    assert main([*w, "claims", "submit", "A", "--doc", "01", "--file", str(body)]) == 0
    assert calls["submit"][0][1] == {"case": "A", "run": "r1", "doc": "01",
                                     "doc_sha256": "sha256:x", "claims": [{"c": 1}]}  # fmt: skip
    assert json.loads(capsys.readouterr().out) == {"accepted": 1}
    monkeypatch.setattr("kyb.commands.reading.submit_claims",
                        spy("submit", SubmissionRejected(["bad claim"])))  # fmt: skip
    assert main([*w, "claims", "submit", "A", "--doc", "01", "--file", str(body)]) == 1
    assert json.loads(capsys.readouterr().out) == {"rejected": ["bad claim"]}
    assert main([*w, "audit", "verify"]) == 0
