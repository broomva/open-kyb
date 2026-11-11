"""propose_policy / apply_policy: the checked, human-approved way an agent changes policy.

Fresh workspaces only (invariant 5): every test inits its own and never touches another's log.
"""

import json
from pathlib import Path

import pytest

from kyb.audit.chain import chain_events, verify_workspace, workspace_action
from kyb.policy.change import CONFIRMATION_FIELD, PolicyRefused, apply_policy, propose_policy
from kyb.util.hashing import sha256_bytes, sha256_json
from kyb.workspace import guard
from kyb.workspace.init import init_workspace

POLICY = Path(__file__).resolve().parents[1] / "examples/policy/compliance_rules.md"
TEXT = POLICY.read_text("utf-8")
PEP_NOW, PEP_FORMER = "within the last **3 years**", "more than 3 years ago"
THREE_YEARS = TEXT.replace(PEP_NOW, "within the last **5 years**").replace(
    PEP_FORMER, "more than 5 years ago"
)

R2_REVIEW = TEXT.replace("2. Embargoed Jurisdictions — **REJECT**",
                         "2. Embargoed Jurisdictions — **REVIEW**")  # fmt: skip
R3_ONE = TEXT.replace("3. Monitored Jurisdictions — **SCORE +20**",
                      "3. Monitored Jurisdictions — **SCORE +1**")  # fmt: skip


@pytest.fixture
def ws(tmp_path):
    root = tmp_path / "ws"
    init_workspace(root, tenant="t", policy_src=POLICY)
    return root


def live_hash(root: Path) -> str:
    return sha256_bytes((root / "policy/compliance_rules.md").read_bytes())


def propose(root: Path, text: str) -> dict:
    with workspace_action(root):
        return propose_policy(root, text)


def event(tool, tool_input, root, **extra):
    """A Claude Code PreToolUse event (the fields the gate recognises it by)."""
    return {"tool_name": tool, "tool_input": tool_input, "cwd": str(root), "tool_use_id": "tu-p1",
            "session_id": "s-1", "hook_event_name": "PreToolUse", "permission_mode": "default",
            **extra}  # fmt: skip


def harness_ask(root: Path, monkeypatch, proposal_id: str, expected: str, *,
                started: bool = True, **extra) -> tuple:  # fmt: skip
    """What the hook does for an apply_policy call in Claude Code. `started`: the harness's
    SessionStart hook saw the session first, as it does for a real one."""
    monkeypatch.setenv("KYB_WORKSPACE", str(root))
    session = extra.get("session_id", "s-1")
    if started:
        guard.session_start({"session_id": session, "cwd": str(root), "model": "m"})
    tin = {"proposal_id": proposal_id, "expected_current_hash": expected}
    return guard.pre_tool_use(event("mcp__plugin_kyb_kyb__apply_policy", tin, root, **extra))


def approved(root: Path, monkeypatch, proposal_id: str, expected: str, **extra) -> str:
    """The token the harness delivers when the human approves the hook's prompt."""
    out = harness_ask(root, monkeypatch, proposal_id, expected, **extra)[2]["hookSpecificOutput"]
    assert out["permissionDecision"] == "ask"
    return out["updatedInput"][CONFIRMATION_FIELD]


def apply(root: Path, proposal_id: str, expected: str, **kw) -> dict:
    with workspace_action(root):
        return apply_policy(root, proposal_id, expected, **kw)


def test_a_valid_edit_proposes_with_its_diff_and_applies(ws, monkeypatch):
    before = live_hash(ws)
    out = propose(ws, THREE_YEARS)
    assert out["parameter_diff"] == ["pep_lookback_years 3 → 5"]
    assert out["changed_rules"] == ["PEP-01", "PEP-02"] and out["changed_sections"] == ["pep"]
    assert out["current_hash"] == before
    assert out["proposed_hash"] == sha256_bytes(THREE_YEARS.encode())
    assert len(json.dumps(out)) < 8_000  # compact: well under a host's tool-output limit
    assert live_hash(ws) == before  # proposing never changes the live policy
    record = json.loads((ws / out["record"]).read_text())
    assert record["param_diff"] == {"pep_lookback_years": {"old": 3, "new": 5}}
    assert "-**PEP-01" in record["text_diff"] or "PEP-01" in record["text_diff"]
    assert propose(ws, THREE_YEARS) == out  # content-addressed: re-proposing is idempotent

    asked = harness_ask(ws, monkeypatch, out["proposal_id"], before)[2]["hookSpecificOutput"]
    assert asked["permissionDecisionReason"].endswith(
        ": pep_lookback_years 3 → 5; rule text changed: PEP-01, PEP-02")  # fmt: skip
    token = asked["updatedInput"][CONFIRMATION_FIELD]
    applied = apply(ws, out["proposal_id"], before, confirmation=token)
    assert applied["new_hash"] == out["proposed_hash"] == live_hash(ws)
    assert (ws / "policy/compliance_rules.md").read_text("utf-8") == THREE_YEARS
    assert applied["confirmation"]["via"] == "harness-ask"
    assert applied["confirmation"]["tool_use_id"] == "tu-p1"
    assert token not in json.dumps(applied) and token not in (ws / "ledger/audit.jsonl").read_text()
    # the previous raw text is kept, content-addressed
    assert (ws / applied["previous_text"]).read_bytes() == POLICY.read_bytes()
    assert applied["previous_text"].endswith(before.removeprefix("sha256:") + ".md")
    done = [e for e in chain_events(ws) if e["action"] == "policy.apply"][-1]["detail"]
    assert (done["old_hash"], done["new_hash"], done["proposal_id"]) == (
        before, out["proposed_hash"], out["proposal_id"])  # fmt: skip
    assert verify_workspace(ws, at_boundary=True).ok


@pytest.mark.parametrize(
    "edit, reason",
    [
        (lambda t: t.replace(PEP_NOW, "within the last **5 years**"),
         "pep: the PEP look-back is stated inconsistently: ['3', '5']"),
        (lambda t: t + "\n## 14. R-11 · Crypto custody — **REVIEW**\nAny crypto custody.\n",
         "unrecognised section"),
        (lambda t: t.replace("`risk_score ≤ 25`", "a risk score of at most 25"),
         "auto-approve threshold could not be read"),
    ],
    ids=["pep-one-place", "r-11", "s11-reworded"],
)  # fmt: skip
def test_an_unrepresentable_text_is_refused_with_its_reasons(ws, edit, reason):
    before = live_hash(ws)
    with pytest.raises(PolicyRefused) as exc:
        propose(ws, edit(TEXT))
    assert any(reason in r for r in exc.value.reasons) and reason in str(exc.value)
    assert live_hash(ws) == before and not (ws / "ledger/policy/proposals").exists()


def test_a_refusal_lists_its_reasons_but_stays_bounded():
    many = [f"policy: unrecognised section 'x{i}'" for i in range(500)]
    message = str(PolicyRefused(many))
    assert "x0'" in message and "and 460 more" in message and len(message) < 4_000


def test_a_stale_hash_and_a_stale_proposal_are_refused(ws, monkeypatch):
    from kyb.signing import officers

    with workspace_action(ws):
        ana = officers.enroll(ws, "ana")
    before = live_hash(ws)
    out = propose(ws, THREE_YEARS)
    other = propose(ws, R2_REVIEW)
    # the hook does not even ask for a hash the proposal was not made against
    code, why, *_ = harness_ask(ws, monkeypatch, out["proposal_id"], "sha256:" + "0" * 64)
    assert code == 2 and "stale" in why
    token = approved(ws, monkeypatch, out["proposal_id"], before)
    apply(ws, other["proposal_id"], before, officer="ana", officer_token=ana["token"])
    with pytest.raises(PermissionError, match="stale"):  # the live policy moved meanwhile
        apply(ws, out["proposal_id"], before, confirmation=token)
    with pytest.raises(PermissionError, match="human's approval"):  # a failed call spent it
        apply(ws, out["proposal_id"], before, confirmation=token)
    assert live_hash(ws) == other["proposed_hash"]
    assert verify_workspace(ws, at_boundary=True).ok


def test_no_approval_no_apply(ws, monkeypatch):
    """No token, a guessed token, a denied prompt's leftover record, a token for another input,
    or a used token: each is refused, and the live policy stays as it was."""
    before = live_hash(ws)
    out = propose(ws, THREE_YEARS)
    pid = out["proposal_id"]
    with pytest.raises(PermissionError, match="human's approval"):
        apply(ws, pid, before)  # no hook asked: no harness, no apply
    harness_ask(ws, monkeypatch, pid, before)  # the human denies: the token never arrives
    assert [e for e in chain_events(ws) if e["action"] == "harness.policy_ask"]
    for guess in ("", "0" * 32, "kyb_confirmation"):
        with pytest.raises(PermissionError, match="human's approval"):
            apply(ws, pid, before, confirmation=guess)
    second = propose(ws, R2_REVIEW)["proposal_id"]
    other = approved(ws, monkeypatch, second, before)  # approved for another proposal
    with pytest.raises(PermissionError, match="human's approval"):
        apply(ws, pid, before, confirmation=other)
    assert live_hash(ws) == before
    token = approved(ws, monkeypatch, pid, before)
    apply(ws, pid, before, confirmation=token)
    back = propose(ws, TEXT)  # a used token covers nothing else
    with pytest.raises(PermissionError, match="human's approval"):
        apply(ws, back["proposal_id"], out["proposed_hash"], confirmation=token)
    assert live_hash(ws) == out["proposed_hash"]


def test_an_enrolled_officer_applies_with_their_token(ws):
    from kyb.signing import officers

    with workspace_action(ws):
        ana = officers.enroll(ws, "ana")
    before = live_hash(ws)
    out = propose(ws, THREE_YEARS)
    with pytest.raises(PermissionError):
        apply(ws, out["proposal_id"], before, officer="ana", officer_token="guess")
    applied = apply(ws, out["proposal_id"], before, officer="ana", officer_token=ana["token"])
    assert applied["confirmation"] == {"via": "officer", "officer": "ana"}
    assert live_hash(ws) == out["proposed_hash"]


@pytest.mark.parametrize(
    "extra",
    [
        {"turn_id": "t-1"},
        {"permission_mode": "bypassPermissions"},
        {"permission_mode": None},
        {"permission_mode": "auto"},
    ],
    ids=["codex", "bypass", "no-mode", "auto"],
)
def test_the_hook_denies_apply_where_no_prompt_can_be_shown(ws, monkeypatch, extra):
    out = propose(ws, THREE_YEARS)
    code, why, *_ = harness_ask(ws, monkeypatch, out["proposal_id"], live_hash(ws), **extra)
    assert code == 2 and "fails closed" in why
    assert not [e for e in chain_events(ws) if e["action"] == "harness.policy_ask"]


APPLY = "kyb policy apply pp-abcdef9876543210 --expected-current-hash sha256:" + "a" * 64


@pytest.mark.parametrize(
    "command",
    ["grep -rn 'kyb policy apply' docs", "uv run pytest -k apply_policy",
     "git commit -m 'feat: kyb policy apply flow'", "grep -rn kyb-hook src",
     "kyb policy propose --file cases/c/proposals/policy.md"],
)  # fmt: skip
def test_other_shell_commands_pass_the_hook_and_carry_no_token(ws, monkeypatch, command):
    monkeypatch.setenv("KYB_WORKSPACE", str(ws))
    assert guard.pre_tool_use(event("Bash", {"command": command}, ws)) == (0, "")
    assert not [e for e in chain_events(ws) if e["action"] == "harness.policy_ask"]


def test_an_officer_applies_from_their_own_terminal(ws, monkeypatch, capsys):
    from kyb.cli import main
    from kyb.policy import commands
    from kyb.signing import officers

    with workspace_action(ws):
        ana = officers.enroll(ws, "ana")
    before = live_hash(ws)
    out = propose(ws, R2_REVIEW)
    monkeypatch.setattr(commands, "_terminal_token", lambda officer: ana["token"])
    argv = ["-w", str(ws), "policy", "apply", out["proposal_id"], "--expected-current-hash",
            before, "--officer", "ana"]  # fmt: skip
    assert main(argv) == 0
    captured = capsys.readouterr()
    assert "JUR-01 REJECT → REVIEW" in captured.err  # the officer sees what they approve
    assert json.loads(captured.out)["confirmation"] == {"via": "officer", "officer": "ana"}
    assert live_hash(ws) == out["proposed_hash"]


def test_malformed_apply_input_is_denied_and_direct_writes_stay_denied(ws, monkeypatch):
    monkeypatch.setenv("KYB_WORKSPACE", str(ws))
    good = {"proposal_id": "pp-abcdef9876543210", "expected_current_hash": "sha256:" + "a" * 64}
    bad = {**good, "expected_current_hash": "x"}
    assert guard.pre_tool_use(event("mcp__kyb__apply_policy", bad, ws))[0] == 2
    # a value already in kyb_confirmation (the agent's, or a second hook registration's) is
    # replaced by a fresh token, never trusted
    out = propose(ws, THREE_YEARS)
    preset = {"proposal_id": out["proposal_id"], "expected_current_hash": live_hash(ws),
              CONFIRMATION_FIELD: "self"}  # fmt: skip
    hso = guard.pre_tool_use(event("mcp__kyb__apply_policy", preset, ws))[2]["hookSpecificOutput"]
    assert hso["permissionDecision"] == "ask" and hso["updatedInput"][CONFIRMATION_FIELD] != "self"
    assert guard.pre_tool_use(event("mcp__other__apply_policy", {"x": 1}, ws)) == (0, "")
    target = str(ws / "policy/compliance_rules.md")
    code, why, *_ = guard.pre_tool_use(event("Write", {"file_path": target, "content": "x"}, ws))
    assert code == 2 and "policy/compliance_rules.md" in why
    code, why, *_ = guard.pre_tool_use(event("Bash", {"command": f"cp /tmp/x {target}"}, ws))
    assert code == 2 and "policy" in why


def test_decide_after_apply_demands_a_rerun_under_the_new_rules_version(tmp_path, monkeypatch):
    from test_decide import fixture_run, submit_fixture

    from kyb.decide.action import decide_case, decision_status
    from kyb.predicates.submit import load_answers
    from kyb.run.action import load_work_order, run_case

    monkeypatch.setattr("kyb.decide.action.support_checker", lambda *a, **kw: None)
    root, packet, work = fixture_run(tmp_path, "C")
    submit_fixture(root, packet, work)
    case, run = packet.facts.case, packet.facts.run
    with workspace_action(root):
        first = decide_case(root, case, run)
    before = live_hash(root)
    out = propose(root, THREE_YEARS)
    apply(root, out["proposal_id"], before,
          confirmation=approved(root, monkeypatch, out["proposal_id"], before))  # fmt: skip
    # the old decision is not re-used: the new hash means a new work order that no run made yet,
    # so decide refuses and the status sends the agent back to `kyb run`
    assert decision_status(root, case, run)["stage"] == "run"
    with (
        workspace_action(root),
        pytest.raises(LookupError, match="work-orders/.*no catalog receipt"),
    ):
        decide_case(root, case, run)
    with workspace_action(root):
        result = run_case(root, case, run, facts_fixture=tmp_path / "C-facts.json",
                          screening_fixture=tmp_path / "C-screening.json")  # fmt: skip
    rerun = load_work_order(root, result["work_order_id"])
    assert rerun.rules_version.endswith(out["proposed_hash"])
    if decision_status(root, case, run)["stage"] == "predicates":
        # answers whose evidence is unchanged carry over; only the pending ones are re-asked
        cached = load_answers(root, rerun.work_order_id) or {}
        pending = [i for i in rerun.instances if i.instance_id not in cached.get("sources", {})]
        submit_fixture(root, packet, rerun.model_copy(update={"instances": pending}))
    assert decision_status(root, case, run)["stage"] == "decide"
    with workspace_action(root):
        second = decide_case(root, case, run)
    assert second.kyb_decision.rules_version.endswith(out["proposed_hash"])
    assert first.kyb_decision.rules_version.endswith(before)
    assert second.decision_id != first.decision_id
    assert verify_workspace(root, at_boundary=True).ok


def test_a_failed_audit_append_leaves_the_live_policy_unchanged(ws, monkeypatch):
    from kyb.audit.chain import AuditLog

    before = live_hash(ws)
    out = propose(ws, THREE_YEARS)
    token = approved(ws, monkeypatch, out["proposal_id"], before)
    real = AuditLog.append_record

    def failing(self, component, action, **kw):
        if action == "policy.apply":
            raise OSError("disk full")
        return real(self, component, action, **kw)

    monkeypatch.setattr(AuditLog, "append_record", failing)
    with pytest.raises(OSError, match="disk full"):
        apply(ws, out["proposal_id"], before, confirmation=token)
    assert live_hash(ws) == before
    assert not [e for e in chain_events(ws) if e["action"] == "policy.apply"]
    assert verify_workspace(ws, at_boundary=True).ok


# --- cross-family review of PR #80 (Gemini, VERDICT: CHANGES): one regression per finding ------

SCRATCH_EVENT = "cases/c/scratch/ev.json"


@pytest.mark.parametrize(
    "command",
    [f"kyb guard pre < {SCRATCH_EVENT}", f"kyb -w ws guard pre --source plugin < {SCRATCH_EVENT}",
     f"kyb-hook pre --source workspace < {SCRATCH_EVENT}",
     f"uv run python -m kyb.hook pre --source plugin < {SCRATCH_EVENT}",
     f"python3 -c 'import kyb.workspace.guard as g' < {SCRATCH_EVENT}",
     f"sh -c 'kyb-hook pre --source plugin' < {SCRATCH_EVENT}",
     "kyb audit append-hook --source plugin < x.json", "kyb gu\\ard pre < x.json"],
)  # fmt: skip
def test_f1_the_agent_cannot_run_the_hook_entry_points(ws, monkeypatch, command):
    """F1: the agent forged a PreToolUse event and ran the hook itself to mint its own token."""
    monkeypatch.setenv("KYB_WORKSPACE", str(ws))
    code, why, *_ = guard.pre_tool_use(event("Bash", {"command": command}, ws))
    assert code == 2 and "hook" in why


@pytest.mark.parametrize(
    "command",
    [APPLY, APPLY + " --confirmation 0123", 'kyb policy "apply" pp-abcdef9876543210',
     "kyb policy ap\\ply pp-abcdef9876543210", "uv run kyb -w ws policy apply x --officer ana",
     "sh -c 'kyb policy apply pp-abcdef9876543210'"],
)  # fmt: skip
def test_f1_policy_apply_is_never_run_from_the_agents_shell(ws, monkeypatch, command):
    monkeypatch.setenv("KYB_WORKSPACE", str(ws))
    code, why, *_ = guard.pre_tool_use(event("Bash", {"command": command}, ws))
    assert code == 2 and "apply_policy" in why
    assert not [e for e in chain_events(ws) if e["action"] == "harness.policy_ask"]


def test_f1_a_token_from_a_session_the_harness_never_started_is_refused(ws, monkeypatch):
    """F1: a forged event names a session no SessionStart hook recorded."""
    before = live_hash(ws)
    out = propose(ws, THREE_YEARS)
    token = approved(ws, monkeypatch, out["proposal_id"], before, started=False,
                     session_id="s-forged")  # fmt: skip
    with pytest.raises(PermissionError, match="session"):
        apply(ws, out["proposal_id"], before, confirmation=token)
    assert live_hash(ws) == before


def test_f1_a_token_from_a_doubly_registered_session_is_refused(ws, monkeypatch):
    from kyb.workspace.guard import HOOK_SOURCE

    before = live_hash(ws)
    out = propose(ws, THREE_YEARS)
    for source in ("plugin", "workspace"):  # both registrations delivered hooks in s-1
        reset = HOOK_SOURCE.set(source)
        try:
            guard.session_start({"session_id": "s-1", "cwd": str(ws), "model": "m"})
        finally:
            HOOK_SOURCE.reset(reset)
    token = approved(ws, monkeypatch, out["proposal_id"], before)
    with pytest.raises(PermissionError, match="registered more than once"):
        apply(ws, out["proposal_id"], before, confirmation=token)


def test_f1_the_cli_apply_is_the_officers_only(ws, capsys):
    from kyb.cli import main

    out = propose(ws, THREE_YEARS)
    argv = ["-w", str(ws), "policy", "apply", out["proposal_id"], "--expected-current-hash",
            live_hash(ws), "--confirmation", "0" * 32]  # fmt: skip
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2  # usage: no --confirmation option, --officer is required
    assert live_hash(ws) != out["proposed_hash"]


def test_f1_and_f3_workspace_settings_deny_the_bypasses(ws):
    deny = json.loads((ws / ".claude/settings.json").read_text())["permissions"]["deny"]
    for rule in ("Bash(kyb guard *)", "Bash(kyb-hook *)", "Bash(kyb audit append-hook *)",
                 "Bash(kyb policy apply *)", "Write(policy/**)", "Edit(policy/**)"):  # fmt: skip
        assert rule in deny


@pytest.mark.parametrize(
    "text, line",
    [(R2_REVIEW, "JUR-01 REJECT → REVIEW"), (R3_ONE, "JUR-02 SCORE +20 → SCORE +1")],
    ids=["r2-severity", "r3-points"],
)
def test_f2_severity_and_points_changes_are_shown(ws, monkeypatch, text, line):
    """F2: a severity or points edit produced `parameter_diff: []`: 'no parameter change'."""
    out = propose(ws, text)
    assert out["rule_diff"] == [line]
    record = json.loads((ws / out["record"]).read_text())
    assert record["rule_diff"]
    asked = harness_ask(ws, monkeypatch, out["proposal_id"], live_hash(ws))
    reason = asked[2]["hookSpecificOutput"]["permissionDecisionReason"]
    assert line in reason and "no parameter change" not in reason


def test_f2_a_severity_change_bundled_with_a_param_edit_is_not_hidden(ws):
    both = R2_REVIEW.replace(PEP_NOW, "within the last **5 years**").replace(
        PEP_FORMER, "more than 5 years ago")  # fmt: skip
    out = propose(ws, both)
    assert out["parameter_diff"] == ["pep_lookback_years 3 → 5"]
    assert out["rule_diff"] == ["JUR-01 REJECT → REVIEW"]


@pytest.mark.parametrize(
    "command",
    ['cp /tmp/x pol"icy"/compliance_rules.md', "cd policy && cp ../x compliance_rules.md",
     "cd policy; tee compliance_rules.md < /tmp/x", "echo x >'policy/compliance_rules.md'",
     "mv /tmp/x ./policy/compliance_rules.md", "sed -i s/5/3/ 'policy/compliance_rules.md'",
     "cd cases/../policy && rm compliance_rules.md"],
)  # fmt: skip
def test_f3_quoted_or_relative_writes_to_policy_are_denied(ws, command):
    assert guard.check_bash(command, ws, ws) is not None


@pytest.mark.parametrize(
    "command",
    ["cat policy/compliance_rules.md", "diff policy/compliance_rules.md cases/c/scratch/p.md",
     "cd policy && grep -n years compliance_rules.md"],
)  # fmt: skip
def test_f3_reading_the_policy_into_scratch_stays_allowed(ws, command):
    assert guard.check_bash(command, ws, ws) is None


def test_f3_python_reaching_the_policy_internals_is_denied(ws, monkeypatch):
    monkeypatch.setenv("KYB_WORKSPACE", str(ws))
    command = "uv run python -c 'from kyb.policy.change import apply_policy as a'"
    code, why, *_ = guard.pre_tool_use(event("Bash", {"command": command}, ws))
    assert code == 2


def test_f3_an_unapproved_live_policy_is_refused_before_run_and_decide(tmp_path, monkeypatch):
    """F3: a policy file replaced outside apply_policy was used as if approved."""
    from test_decide import fixture_run, submit_fixture

    from kyb.decide.action import decide_case
    from kyb.run.action import run_case

    monkeypatch.setattr("kyb.decide.action.support_checker", lambda *a, **kw: None)
    root, packet, work = fixture_run(tmp_path, "C")
    submit_fixture(root, packet, work)
    case, run = packet.facts.case, packet.facts.run
    (root / "policy/compliance_rules.md").write_text(THREE_YEARS, "utf-8")  # no apply
    with workspace_action(root), pytest.raises(PermissionError, match="not the approved policy"):
        decide_case(root, case, run)
    with workspace_action(root), pytest.raises(PermissionError, match="not the approved policy"):
        run_case(root, case, run, facts_fixture=tmp_path / "C-facts.json",
                 screening_fixture=tmp_path / "C-screening.json")  # fmt: skip


def test_f4_a_country_listed_in_several_clauses_is_not_dropped():
    """F4: `_codes` kept only the last region entry for a repeated country code."""
    from kyb.policy.parse import parse_policy

    split = TEXT.replace(
        "or the Crimea/Donetsk/Luhansk/Zaporizhzhia/Kherson regions of `UA`.",
        "the Crimea region of `UA`, the Donetsk region of `UA`, or the Luhansk region of `UA`.",
    )
    model = parse_policy(split.encode("utf-8"))
    assert any("`UA`" in r and "more than once" in r for r in model.unrepresentable)


def test_f5_a_threshold_change_tells_the_agent_to_start_a_fresh_run(tmp_path, monkeypatch):
    """F5: after a name_similarity_threshold change, re-screening the run refused vaguely."""
    from kyb.evals.cases import build, golden
    from kyb.run.action import facts_for_run
    from kyb.screening.action import screen_case
    from kyb.workspace.runs import start_run

    packet = build(golden()["A"])
    fs = packet.facts.model_copy(update={"fact_set_hash": sha256_json(
        packet.facts.model_dump(mode="json", exclude={"fact_set_hash"}))})  # fmt: skip
    root = tmp_path / "ws"
    init_workspace(root, tenant="t", policy_src=POLICY)
    facts_file = tmp_path / "facts.json"
    facts_file.write_text(fs.model_dump_json())
    watchlist = POLICY.parent / "watchlists.json"
    # fixture facts carry no derivation receipt; the refusal under test is the re-screen's
    monkeypatch.setattr("kyb.screening.action.load_facts", lambda *_: fs)
    with workspace_action(root):
        assert start_run(root, fs.case, as_of=fs.as_of, reason="test") == fs.run
        facts_for_run(root, fs.case, fs.run, facts_file)
        screen_case(root, fs.case, fs.run, watchlist)
    lower = TEXT.replace("score ≥ 0.88", "score ≥ 0.70")
    out = propose(root, lower)
    applied = apply(root, out["proposal_id"], out["current_hash"],
                    confirmation=approved(root, monkeypatch, out["proposal_id"],
                                          out["current_hash"]))  # fmt: skip
    assert "new_run" in applied["next"]
    with workspace_action(root), pytest.raises(PermissionError) as exc:
        screen_case(root, fs.case, fs.run, watchlist)
    assert "pages(case, new_run=True)" in str(exc.value) and "--new-run" in str(exc.value)


def test_f6_the_prompt_reads_a_verified_proposal_for_this_hash(ws, monkeypatch):
    """F6: the prompt rendered the proposal file raw and for any expected hash."""
    out = propose(ws, THREE_YEARS)
    stale = "sha256:" + "1" * 64
    code, why, *_ = harness_ask(ws, monkeypatch, out["proposal_id"], stale)
    assert code == 2 and "stale" in why
    record = ws / out["record"]
    tampered = record.read_text().replace('"new":5', '"new":50')
    record.chmod(0o644)
    record.write_text(tampered)
    code, why, *_ = harness_ask(ws, monkeypatch, out["proposal_id"], live_hash(ws))
    assert code == 2 and "receipt" in why


# --- independent review of the fixes (fresh-context, VERDICT: CHANGES): one regression each -----


@pytest.mark.parametrize(
    "command",
    ["kyb init . --tenant t --policy /tmp/evil.md --force",
     "uv run kyb init ws --policy cases/c/scratch/p.md --tenant t --force",
     "python3 -c 'from kyb.workspace.init import init_workspace'"],
)  # fmt: skip
def test_r1_the_agent_cannot_reinstall_the_policy_by_reinitialising(ws, monkeypatch, command):
    """A forced re-init made any text the approved policy (provenance trusts `kyb init`)."""
    monkeypatch.setenv("KYB_WORKSPACE", str(ws))
    code, why, *_ = guard.pre_tool_use(event("Bash", {"command": command}, ws))
    assert code == 2


def test_r7_a_new_run_is_allocated_only_when_the_latest_consumed_its_inputs(ws):
    from kyb.cli import main
    from kyb.contracts.b4 import FACTS_REL
    from kyb.run.records import write_record
    from kyb.workspace.runs import list_runs

    case = "cl-andes-agro"
    packet = POLICY.parent.parent / "packets" / case
    assert main(["-w", str(ws), "ingest", str(packet), "--as-of", "2026-10-25"]) == 0
    assert main(["-w", str(ws), "pages", case]) == 0
    assert main(["-w", str(ws), "pages", case, "--new-run"]) == 0  # r1 consumed nothing yet
    assert list_runs(ws, case) == ["r1"]
    with workspace_action(ws):  # r1's facts are derived: its inputs are consumed
        write_record(ws, FACTS_REL.format(case=case, run="r1"), {"x": 1}, action="test",
                     tier="derived", as_of="2026-10-25")  # fmt: skip
    assert main(["-w", str(ws), "pages", case, "--new-run"]) == 0
    assert main(["-w", str(ws), "pages", case, "--new-run"]) == 0  # a retry reuses r2
    assert list_runs(ws, case) == ["r1", "r2"]


# --- final pass of the independent review (VERDICT: CHANGES): A, B and the cheap leftovers -----


def test_ra_a_workspace_with_a_chain_is_never_reinitialised_without_force(ws, tmp_path):
    """`rm template.l*` then a plain `kyb init` installed an unapproved policy."""
    (ws / "template.lock").unlink()  # however it went: the audit chain is still there
    evil = tmp_path / "evil.md"
    evil.write_text(THREE_YEARS, "utf-8")
    with pytest.raises(FileExistsError, match="audit chain"):
        init_workspace(ws, tenant="t", policy_src=evil)
    assert (ws / "policy/compliance_rules.md").read_text("utf-8") == TEXT  # untouched


@pytest.mark.parametrize(
    "command", ["rm template.l*", "rm -f templ?te.lock", "mv ledger/aud*.jsonl /tmp/"]
)
def test_ra_a_glob_does_not_hide_a_protected_file(ws, command):
    assert guard.check_bash(command, ws, ws) is not None


@pytest.mark.parametrize(
    "command",
    ["timeout 60 kyb init . --tenant t --policy /tmp/e.md --force",
     "env -u FOO kyb init . --tenant t --policy /tmp/e.md --force",
     "nice -n 5 kyb guard pre < x.json", "env -S 'kyb guard pre' < x.json",
     "sudo -u me kyb-hook pre", "stdbuf -o L kyb guard pre",
     "echo 'kyb guard pre' | bash -s", "find . -exec sh -c 'kyb guard pre' +",
     "find . -exec timeout 9 kyb guard pre +",
     "python3 - <<'EOF'\nfrom kyb.workspace import guard\nEOF"],
)  # fmt: skip
def test_rb_wrappers_with_values_and_leftovers_are_seen_through(ws, monkeypatch, command):
    monkeypatch.setenv("KYB_WORKSPACE", str(ws))
    assert guard.pre_tool_use(event("Bash", {"command": command}, ws))[0] == 2


# --- last independent pass (VERDICT: CHANGES): re-init is an officer's act; more shell forms -----


def test_rc_a_forced_reinit_that_changes_the_policy_needs_an_officer(ws, tmp_path):
    """`arch -arm64 kyb init . --policy evil.md --force` installed an unapproved policy: no
    wrapper list is complete, so the action itself requires the officer."""
    from kyb.policy.provenance import approved_policy
    from kyb.signing import officers

    evil = tmp_path / "evil" / "compliance_rules.md"
    evil.parent.mkdir()
    evil.write_text(THREE_YEARS, "utf-8")
    with pytest.raises(PermissionError, match="officer"):
        init_workspace(ws, tenant="t", policy_src=evil, force=True)
    assert approved_policy(ws) == TEXT.encode("utf-8")
    with workspace_action(ws):
        ana = officers.enroll(ws, "ana")
    with pytest.raises(PermissionError):
        init_workspace(ws, tenant="t", policy_src=evil, force=True, officer="ana",
                       officer_token="guess")  # fmt: skip
    init_workspace(ws, tenant="t", policy_src=evil, force=True, officer="ana",
                   officer_token=ana["token"])  # fmt: skip
    assert approved_policy(ws) == THREE_YEARS.encode("utf-8")
    init = [e for e in chain_events(ws) if e["action"] == "workspace.init"][-1]["detail"]
    assert init["policy_change_approved_by"] == "ana"


def test_rc_a_forced_reinit_that_keeps_the_policy_needs_no_officer(ws):
    init_workspace(ws, tenant="t", policy_src=POLICY, force=True)
    assert live_hash(ws) == sha256_bytes(POLICY.read_bytes())


@pytest.mark.parametrize(
    "command",
    ["find . -name template.lock -delete",
     r"find . -maxdepth 2 \( -name template.lock -o -name audit.jsonl \) -delete",
     "find . -name template.lock -exec rm {} +", "mv .. /tmp/x", "rm -rf ../ws"],
)  # fmt: skip
def test_rc_find_deletes_and_moving_the_workspace_are_writes(ws, command):
    assert guard.check_bash(command, ws, ws) is not None


@pytest.mark.parametrize(
    "command",
    ["arch -arm64 kyb guard pre", "setsid kyb guard pre", "script -q /dev/null kyb guard pre",
     "uv tool run --from . kyb guard pre"],
)  # fmt: skip
def test_rc_more_wrappers_are_seen_through(ws, monkeypatch, command):
    monkeypatch.setenv("KYB_WORKSPACE", str(ws))
    assert guard.pre_tool_use(event("Bash", {"command": command}, ws))[0] == 2


def test_rd_hiding_the_chain_and_lock_does_not_make_a_workspace_new(ws, tmp_path):
    """`gzip -r ledger template.lock; kyb init . --policy evil.md` installed an unapproved policy:
    any directory holding workspace data is an existing workspace."""
    evil = tmp_path / "evil" / "compliance_rules.md"
    evil.parent.mkdir()
    evil.write_text(THREE_YEARS, "utf-8")
    hidden = tmp_path / "hidden"
    hidden.mkdir()
    for rel in ("ledger", "template.lock"):  # what `gzip -r` amounts to: the files are gone
        (ws / rel).rename(hidden / rel.replace("/", "_"))
    with pytest.raises(FileExistsError, match="workspace data"):
        init_workspace(ws, tenant="t", policy_src=evil)
    with pytest.raises(PermissionError, match="officer"):
        init_workspace(ws, tenant="t", policy_src=evil, force=True)
    assert (ws / "policy/compliance_rules.md").read_text("utf-8") == TEXT


def test_rd_a_fresh_directory_still_initialises(tmp_path):
    root = tmp_path / "new"
    init_workspace(root, tenant="t", policy_src=POLICY)
    assert (root / "template.lock").exists()


@pytest.mark.parametrize(
    "command", ["gzip -r ledger template.lock", "zip -m /tmp/x.zip ledger/audit.jsonl",
                "trash ledger/audit.jsonl template.lock", "bzip2 catalog.json"]
)  # fmt: skip
def test_rd_archivers_and_trash_are_writes(ws, command):
    assert guard.check_bash(command, ws, ws) is not None
