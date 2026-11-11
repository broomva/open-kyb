"""L5 sign-off (issue): the review queue, officer-only `sign_decision`, overrides as comparable
corrections, and the hook that denies the agent's attempt (spec, agent-driven-review §4b)."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
from decision_fixtures import decision, record, rule, workspace

from kyb.audit.chain import verify_workspace, workspace_action
from kyb.contracts.b4 import CorrectionRecord, comparable_key
from kyb.signing import officers
from kyb.signing.queue import review_queue
from kyb.signing.sign import sign_decision
from kyb.workspace import guard
from kyb.workspace.catalog import Catalog


@pytest.fixture
def ws(tmp_path):
    return workspace(tmp_path)


@pytest.fixture
def officer(ws):
    with workspace_action(ws):
        return officers.enroll(ws, "ana")


def sign(ws, officer, decision_id, verdict="assent", **kw):
    with workspace_action(ws):
        return sign_decision(ws, decision_id=decision_id, verdict=verdict, officer="ana",
                             officer_token=officer["token"], **kw)  # fmt: skip


def comparable(ws, jurisdiction, fired):
    """L4's retrieval (`kyb.corrections.comparable` on lane/4-decide), as a spec."""
    key, out = comparable_key(jurisdiction, fired), []
    for rel in sorted(Catalog(ws).entries()):
        if rel.startswith("ledger/corrections/"):
            rec = CorrectionRecord.model_validate_json((ws / rel).read_text())
            if comparable_key(rec.jurisdiction, rec.fired_rules) == key:
                out.append(rec)
    return out


# --- the review queue ----------------------------------------------------------------------------


def test_the_queue_puts_manual_review_first_then_rejected_then_approved_oldest_first(ws):
    rows = [("A", "APPROVED", "2026-10-25T09:00:00Z"), ("B", "REJECTED", "2026-10-25T08:00:00Z"),
            ("C", "MANUAL_REVIEW", "2026-10-25T11:00:00Z"),
            ("D", "MANUAL_REVIEW", "2026-10-25T10:00:00Z"),
            ("E", "APPROVED", "2026-10-25T07:00:00Z")]  # fmt: skip
    ids = {c: record(ws, decision(c, status=s, emitted=t)) for c, s, t in rows}
    got = [e["decision_id"] for e in review_queue(ws)["queue"]]
    assert got == [ids[c] for c in "DCBEA"]


def test_an_entry_carries_its_rules_findings_and_state(ws):
    rec = decision("A", rules=[rule("DOC-01"), rule("DOC-02", "unknown"), rule("JUR-01", "no")],
                   vetoes=["J-READER-ISOLATION"])  # fmt: skip
    record(ws, rec)
    (e,) = review_queue(ws)["queue"]
    assert [r["rule_id"] for r in e["fired_rules"]] == ["DOC-01"]
    assert e["unknown_rules"] == ["DOC-02"]
    assert [f["judge_id"] for f in e["findings"]] == ["J-READER-ISOLATION"]
    assert e["signature_state"] == "pending_signature"
    assert e["superseded_by"] is None


def test_an_older_decision_of_the_same_case_says_it_was_superseded(ws):
    old = record(ws, decision("A", emitted="2026-10-25T09:00:00Z"))
    new = record(ws, decision("A", run="r2", emitted="2026-10-25T10:00:00Z", seed="2"))
    by_id = {e["decision_id"]: e for e in review_queue(ws)["queue"]}
    assert by_id[old]["superseded_by"] == new and by_id[new]["superseded_by"] is None


def test_the_queue_can_be_narrowed_to_a_case(ws):
    record(ws, decision("A"))
    keep = record(ws, decision("B"))
    assert [e["decision_id"] for e in review_queue(ws, "B")["queue"]] == [keep]


def test_an_unreadable_decision_is_listed_never_dropped(ws):
    rec = decision("A")
    record(ws, rec)
    (ws / "ledger" / "decisions" / f"{rec.decision_id}.json").write_text("{}\n")
    out = review_queue(ws)
    assert out["queue"] == [] and len(out["unreadable"]) == 1
    assert out["unreadable"][0]["decision"].endswith(f"{rec.decision_id}.json")


# --- officers -------------------------------------------------------------------------------------


def test_the_ledger_keeps_only_the_token_hash(ws, officer):
    stored = (ws / "ledger" / "officers" / "ana.json").read_text()
    assert officer["token"] not in stored and officers.token_digest(officer["token"]) in stored


def test_after_the_first_officer_enrolment_needs_an_enrolled_officers_token(ws, officer):
    with workspace_action(ws), pytest.raises(PermissionError):
        officers.enroll(ws, "mallory")  # an agent enrolling itself
    with workspace_action(ws), pytest.raises(PermissionError):
        officers.enroll(ws, "mallory", by="ana", by_token="guess")
    with workspace_action(ws):
        second = officers.enroll(ws, "ben", by="ana", by_token=officer["token"])
    assert officers.enrolled(ws) == ["ana", "ben"] and second["token"] != officer["token"]
    with workspace_action(ws), pytest.raises(FileExistsError):
        officers.enroll(ws, "ben", by="ana", by_token=officer["token"])


# --- sign_decision --------------------------------------------------------------------------------


def test_an_assent_is_a_chained_signature_and_leaves_the_decision_untouched(ws, officer):
    did = record(ws, decision("A", status="APPROVED"))
    before = (ws / "ledger" / "decisions" / f"{did}.json").read_bytes()
    out = sign(ws, officer, did)
    assert out["signature_state"] == "signed" and out["to_status"] == "APPROVED"
    assert out["correction_ids"] == []
    assert (ws / "ledger" / "decisions" / f"{did}.json").read_bytes() == before
    assert review_queue(ws) == {"queue": [], "unsigned": 0, "signed": 1, "unreadable": []}
    assert verify_workspace(ws, at_boundary=True).ok
    audit = [json.loads(line) for line in (ws / "ledger" / "audit.jsonl").read_text().split("\n")
             if line]  # fmt: skip
    signed = [e for e in audit if e["detail"].get("action") == "sign_decision"]
    assert len(signed) == 1 and signed[0]["detail"]["officer"] == "ana"


@pytest.mark.parametrize("token", [None, "", "wrong", "x" * 43])
def test_a_missing_or_wrong_token_signs_nothing(ws, officer, token):
    did = record(ws, decision("A"))
    with workspace_action(ws), pytest.raises(PermissionError, match="officer_token"):
        sign_decision(ws, decision_id=did, verdict="assent", officer="ana", officer_token=token)
    assert Catalog(ws).get(f"ledger/signatures/{did}.json") is None


def test_an_unknown_officer_gets_the_same_refusal_as_a_wrong_token(ws, officer):
    did = record(ws, decision("A"))
    with workspace_action(ws), pytest.raises(PermissionError) as unknown:
        sign_decision(ws, decision_id=did, verdict="assent", officer="nobody",
                      officer_token=officer["token"])  # fmt: skip
    with workspace_action(ws), pytest.raises(PermissionError) as wrong:
        sign_decision(ws, decision_id=did, verdict="assent", officer="ana", officer_token="x")
    assert str(unknown.value) == str(wrong.value)


def test_a_decision_is_signed_once(ws, officer):
    did = record(ws, decision("A"))
    sign(ws, officer, did)
    with pytest.raises(PermissionError, match="already signed"):
        sign(ws, officer, did)
    with pytest.raises(PermissionError, match="already signed"):
        sign(ws, officer, did, "override", status="REJECTED", rationale="changed my mind")


def test_an_unknown_decision_is_refused(ws, officer):
    with pytest.raises(LookupError):
        sign(ws, officer, "dec-nope")


@pytest.mark.parametrize(
    ("verdict", "status", "rationale"),
    [("assent", "REJECTED", ""),  # assent cannot change the status
     ("override", None, "why"),  # an override names a status
     ("override", "MANUAL_REVIEW", "why"),  # the same status is an assent
     ("override", "APPROVED", "   "),  # and needs a written rationale
     ("sign", None, "")],
)  # fmt: skip
def test_malformed_verdicts_are_refused_and_record_nothing(ws, officer, verdict, status, rationale):
    did = record(ws, decision("A"))
    with pytest.raises(ValueError):
        sign(ws, officer, did, verdict, status=status, rationale=rationale)
    assert Catalog(ws).get(f"ledger/signatures/{did}.json") is None
    assert not [r for r in Catalog(ws).entries() if r.startswith("ledger/corrections/")]


def test_an_override_is_a_chained_correction_that_becomes_a_comparable_profile(ws, officer):
    did = record(ws, decision("B", rules=[rule("DOC-01"), rule("GOV-01"), rule("JUR-01", "no")]))
    out = sign(ws, officer, did, "override", status="APPROVED",
               rationale="  ownership resolved by the notarial deed  ")  # fmt: skip
    assert out["to_status"] == "APPROVED" and len(out["correction_ids"]) == 1
    (rec,) = comparable(
        ws, "mx", ["GOV-01", "DOC-01"]
    )  # same jurisdiction and fired rules, any order
    assert rec.correction_id == out["correction_ids"][0]
    assert (rec.officer, rec.decision_id) == ("ana", did)
    assert (rec.from_status, rec.to_status) == ("MANUAL_REVIEW", "APPROVED")
    assert rec.rationale == "ownership resolved by the notarial deed"
    assert rec.fired_rules == ["DOC-01", "GOV-01"] and rec.jurisdiction == "MX"
    assert (
        comparable(ws, "MX", ["DOC-01"]) == [] and comparable(ws, "CO", ["DOC-01", "GOV-01"]) == []
    )
    cat = Catalog(ws).get(f"ledger/corrections/{rec.correction_id}.json")
    assert cat["tier"] == "decided" and cat["inputs"]
    assert verify_workspace(ws, at_boundary=True).ok
    sig = json.loads((ws / "ledger" / "signatures" / f"{did}.json").read_text())
    assert sig["correction_ids"] == [rec.correction_id] and sig["verdict"] == "override"


def test_an_assent_records_no_correction(ws, officer):
    sign(ws, officer, record(ws, decision("B")))
    assert not [r for r in Catalog(ws).entries() if r.startswith("ledger/corrections/")]


def test_a_signing_that_fails_midway_resumes_without_a_second_correction(ws, officer, monkeypatch):
    from kyb.signing import sign as sign_module

    did = record(ws, decision("B"))
    real = sign_module.write_record

    def flaky(root, rel, *a, **kw):
        if rel.startswith("ledger/signatures/"):
            raise OSError("disk full")
        return real(root, rel, *a, **kw)

    monkeypatch.setattr(sign_module, "write_record", flaky)
    with pytest.raises(OSError):
        sign(ws, officer, did, "override", status="REJECTED", rationale="sanctions confirmed")
    assert Catalog(ws).get(f"ledger/signatures/{did}.json") is None  # not signed yet
    monkeypatch.setattr(sign_module, "write_record", real)
    out = sign(ws, officer, did, "override", status="REJECTED", rationale="sanctions confirmed")
    corrections = [r for r in Catalog(ws).entries() if r.startswith("ledger/corrections/")]
    assert len(corrections) == 1 and out["signature_state"] == "signed"
    assert verify_workspace(ws, at_boundary=True).ok


# --- the agent signs only on the officer's approvals: the hook asks, or denies -----------------


def event(tool, tool_input, ws, **extra):
    """A Claude Code PreToolUse event (the fields gate.py recognises it by)."""
    return {"tool_name": tool, "tool_input": tool_input, "cwd": str(ws), "tool_use_id": "tu-1",
            "session_id": "s-1", "hook_event_name": "PreToolUse", "permission_mode": "default",
            **extra}  # fmt: skip


def asked(out):
    hso = out[2]["hookSpecificOutput"]
    assert out[0] == 0 and hso["hookEventName"] == "PreToolUse"
    return hso["permissionDecision"], hso["permissionDecisionReason"]


@pytest.mark.parametrize(
    "tool", ["sign_decision", "mcp__kyb__sign_decision", "mcp__plugin_kyb_kyb__sign_decision"]
)
def test_a_sign_decision_call_in_claude_code_asks_the_officer(ws, monkeypatch, tool):
    monkeypatch.setenv("KYB_WORKSPACE", str(ws))
    out = guard.pre_tool_use(event(tool, {"decision_id": "d-1", "verdict": "assent",
                                          "officer": "ana"}, ws))  # fmt: skip
    decision, why = asked(out)
    assert decision == "ask" and "sign decision d-1" in why and "ana" in why


@pytest.mark.parametrize(
    "command",
    [
        "kyb sign dec-1 assent --officer ana",
        "uv run kyb -w ws sign dec-1 assent --officer ana",
        "python3 -m kyb sign dec-1 assent --officer ana",
    ],
)
def test_a_kyb_sign_command_in_claude_code_asks_the_officer(ws, monkeypatch, command):
    monkeypatch.setenv("KYB_WORKSPACE", str(ws))
    decision, why = asked(guard.pre_tool_use(event("Bash", {"command": command}, ws)))
    assert decision == "ask" and "dec-1" in why


@pytest.mark.parametrize(
    "extra",
    [
        {"hook_event_name": "BeforeTool"},  # not Claude Code's event name
        {"hook_event_name": None},
        {"turn_id": "t-1", "model": "gpt"},  # a Codex event
        {"permission_mode": None},
        {"permission_mode": "bypassPermissions"},  # the prompt would not be shown
        {"tool_use_id": ""},
        {"session_id": ""},
    ],
)
@pytest.mark.parametrize("tool, tool_input", [
    ("mcp__kyb__sign_decision", {"decision_id": "d", "verdict": "assent", "officer": "ana"}),
    ("Bash", {"command": "kyb sign d assent --officer ana"}),
])  # fmt: skip
def test_a_harness_that_cannot_ask_fails_closed(ws, monkeypatch, tool, tool_input, extra):
    monkeypatch.setenv("KYB_WORKSPACE", str(ws))
    code, why, *_ = guard.pre_tool_use({**event(tool, tool_input, ws), **extra})
    assert code == 2 and "fails closed" in why


@pytest.mark.parametrize(
    ("tool", "tool_input"),
    [
        ("record_override", {"decision_id": "d"}),
        ("mcp__kyb__record_override", {"decision_id": "d"}),
        ("enroll_officer", {"officer": "mallory"}),
        ("mcp__kyb__enroll_officer", {"officer": "mallory"}),
        ("mcp__kyb__sign_decision", {"decision_id": "d", "verdict": "override", "officer": "a"}),
        ("mcp__kyb__sign_decision", {"decision_id": "d", "verdict": "assent", "officer": "a",
                                     "officer_token": "t"}),
        ("mcp__kyb__sign_decision", {"decision_id": "d", "officer": "a"}),
    ],
)  # fmt: skip
def test_the_hook_still_denies_overrides_enrolment_and_a_token_in_the_call(
    ws, monkeypatch, tool, tool_input
):
    monkeypatch.setenv("KYB_WORKSPACE", str(ws))
    code, why, *_ = guard.pre_tool_use(event(tool, tool_input, ws))
    assert code == 2 and ("officer" in why or "override" in why)


@pytest.mark.parametrize(
    "command",
    [
        "uv run kyb -w ws sign dec-1 override --officer ana --status APPROVED --rationale ok",
        "kyb override dec-1 --officer ana --status APPROVED --rationale ok",
        "KYB_OFFICER_TOKEN=abc kyb sign dec-1 assent --officer ana",
        "kyb sign dec-1 assent --officer ana --officer-token abc",
        "kyb sign dec-1 assent --officer-token X",
        "kyb sign dec-1 assent --officer ana; echo hi",
        "kyb sign dec-1 assent --officer ana && ls",
        "echo $KYB_OFFICER_TOKEN",
        "env | grep -i kyb_officer_token",
        "kyb officer enroll mallory",
        "kyb officer enroll mallory --by ana",
        "python -c 'from kyb.signing.sign import sign_decision'",
    ],
)
def test_the_hook_still_denies_these_bash_commands(ws, monkeypatch, command):
    monkeypatch.setenv("KYB_WORKSPACE", str(ws))
    code, why, *_ = guard.pre_tool_use(event("Bash", {"command": command}, ws))
    assert code == 2 and ("officer" in why or "override" in why)


@pytest.mark.parametrize(("tool", "tool_input"), [
    ("Bash", {"command": "kyb queue"}),
    ("Bash", {"command": "uv run kyb -w ws ask B 'who owns it?'"}),
    ("Bash", {"command": "echo design review of the signing flow"}),
    ("mcp__kyb__review_queue", {}),
    ("mcp__kyb__ask", {"question": "who owns B?"}),
    ("Read", {"file_path": "/tmp/x"}),
])  # fmt: skip
def test_the_hook_leaves_reading_the_queue_and_asking_alone(ws, monkeypatch, tool, tool_input):
    monkeypatch.setenv("KYB_WORKSPACE", str(ws))
    assert guard.pre_tool_use(event(tool, tool_input, ws))[0] == 0


def test_the_hook_entry_point_prints_ask_json_for_a_sign_and_exits_0(ws):
    """The shipped path: `kyb-hook pre` reads the event on stdin; "ask" is JSON on stdout."""
    sign_input = {"decision_id": "d", "verdict": "assent", "officer": "ana"}
    ev = json.dumps(event("mcp__kyb__sign_decision", sign_input, ws))
    run = subprocess.run([sys.executable, "-m", "kyb.hook", "pre", "--source", "plugin"],
                         input=ev, capture_output=True, text=True, env={"KYB_WORKSPACE": str(ws),
                         "PATH": "/usr/bin:/bin"}, check=False)  # fmt: skip
    assert run.returncode == 0
    hso = json.loads(run.stdout)["hookSpecificOutput"]
    assert hso["permissionDecision"] == "ask"
    assert "sign decision d" in hso["permissionDecisionReason"]


def test_the_hook_entry_point_exits_2_for_an_override(ws):
    ev = json.dumps(event("mcp__kyb__record_override", {"decision_id": "d"}, ws))
    run = subprocess.run([sys.executable, "-m", "kyb.hook", "pre", "--source", "plugin"],
                         input=ev, capture_output=True, text=True, env={"KYB_WORKSPACE": str(ws),
                         "PATH": "/usr/bin:/bin"}, check=False)  # fmt: skip
    assert run.returncode == 2 and "officer-only" in run.stderr


def test_without_the_hook_the_action_still_demands_an_enrolled_officers_token(ws, officer):
    """A harness with no hooks (Codex) gains no right to sign: the token is the second denial."""
    did = record(ws, decision("A"))
    with workspace_action(ws), pytest.raises(PermissionError):
        sign_decision(ws, decision_id=did, verdict="assent", officer="agent", officer_token=None)


# --- the surfaces ---------------------------------------------------------------------------


def test_the_cli_queues_and_signs_with_the_env_token(ws, officer, capsys, monkeypatch):
    from kyb.cli import main

    did = record(ws, decision("A"))
    assert main(["-w", str(ws), "queue"]) == 0
    assert json.loads(capsys.readouterr().out)["queue"][0]["decision_id"] == did
    monkeypatch.setenv("KYB_OFFICER_TOKEN", "wrong")
    with pytest.raises(SystemExit, match="officer_token"):
        main(["-w", str(ws), "sign", did, "assent", "--officer", "ana"])
    monkeypatch.setenv("KYB_OFFICER_TOKEN", officer["token"])
    assert main(["-w", str(ws), "sign", did, "assent", "--officer", "ana"]) == 0
    assert json.loads(capsys.readouterr().out)["signature_state"] == "signed"


def test_the_mcp_tools_list_and_refuse_a_stranger(ws, officer, monkeypatch):
    import asyncio

    from kyb import mcp_server as mcp

    names = {t.name for t in asyncio.run(mcp.server.list_tools())}
    assert {"review_queue", "sign_decision"} <= names
    monkeypatch.setenv("KYB_WORKSPACE", str(ws))
    did = record(ws, decision("A"))
    assert mcp.review_queue()["queue"][0]["decision_id"] == did
    monkeypatch.delenv("KYB_OFFICER_TOKEN", raising=False)
    with pytest.raises(PermissionError):  # no token in the server's environment: nothing signs
        mcp.sign_decision(did, "assent", "ana")
    monkeypatch.setenv("KYB_OFFICER_TOKEN", "not-the-token")
    with pytest.raises(PermissionError):
        mcp.sign_decision(did, "assent", "ana")
    monkeypatch.setenv("KYB_OFFICER_TOKEN", officer["token"])
    out = mcp.sign_decision(did, "assent", "ana")
    assert out["signature_state"] == "signed" and out["approved_via"] == "harness-ask"


# --- review round 1 -----------------------------------------------------------------------------


def test_the_signed_count_is_the_cases_own(ws, officer):
    sign(ws, officer, record(ws, decision("A")))
    record(ws, decision("B"))
    assert review_queue(ws, "B")["signed"] == 0 and review_queue(ws, "B")["unsigned"] == 1
    assert review_queue(ws, "A")["signed"] == 1 and review_queue(ws)["signed"] == 1


def test_an_override_on_facts_without_the_applicant_still_records(ws, officer):
    from decision_fixtures import facts

    from kyb.contracts.b4 import FACTS_REL
    from kyb.signing.ledger import write_record

    fs = facts("D", "r1", ["MX"]).model_dump(mode="json")
    fs.update(entities=[], applicant="e-gone")
    fs["fact_set_hash"] = __import__("kyb.util.hashing", fromlist=["x"]).sha256_json(
        {k: v for k, v in fs.items() if k != "fact_set_hash"})  # fmt: skip
    with workspace_action(ws):
        write_record(ws, FACTS_REL.format(case="D", run="r1"), fs, action="fixture_facts",
                     tier="derived", as_of="2026-10-25", inputs=[], detail={})  # fmt: skip
    did = record(ws, decision("D"))
    out = sign(ws, officer, did, "override", status="REJECTED", rationale="fixture")
    (cid,) = out["correction_ids"]
    rec = json.loads((ws / "ledger" / "corrections" / f"{cid}.json").read_text())
    assert rec["jurisdiction"] == "UNKNOWN"


# --- provenance: how a signature was approved -----------------------------------------------------


def _audit_detail(ws, did):
    audit = [json.loads(line) for line in (ws / "ledger" / "audit.jsonl").read_text().splitlines()]
    return next(
        e["detail"]
        for e in audit
        if e["detail"].get("decision_id") == did and e["detail"].get("action") == "sign_decision"
    )


def test_a_harness_approved_signature_records_it_in_the_record_and_the_audit_event(ws, officer):
    did = record(ws, decision("A"))
    with workspace_action(ws):
        out = sign_decision(ws, decision_id=did, verdict="assent", officer="ana",
                            officer_token=officer["token"], approved_via="harness-ask")  # fmt: skip
    assert out["approved_via"] == "harness-ask"
    assert (
        json.loads((ws / "ledger" / "signatures" / f"{did}.json").read_text())["approved_via"]
        == "harness-ask"
    )
    assert _audit_detail(ws, did)["approved_via"] == "harness-ask"
    assert verify_workspace(ws, at_boundary=True).ok


def test_a_terminal_signature_says_terminal_and_a_bad_origin_is_refused(ws, officer):
    did = record(ws, decision("A"))
    assert sign(ws, officer, did)["approved_via"] == "terminal"
    other = record(ws, decision("B"))
    with workspace_action(ws), pytest.raises(ValueError, match="approved_via"):
        sign_decision(ws, decision_id=other, verdict="assent", officer="ana",
                      officer_token=officer["token"], approved_via="agent")  # fmt: skip


def test_the_cli_in_a_claude_code_shell_records_harness_ask(ws, officer, capsys, monkeypatch):
    from kyb.cli import main

    did = record(ws, decision("A"))
    monkeypatch.setenv("KYB_OFFICER_TOKEN", officer["token"])
    monkeypatch.setenv("CLAUDECODE", "1")
    assert main(["-w", str(ws), "sign", did, "assent", "--officer", "ana"]) == 0
    assert json.loads(capsys.readouterr().out)["approved_via"] == "harness-ask"
    other = record(ws, decision("B"))
    monkeypatch.delenv("CLAUDECODE")
    assert main(["-w", str(ws), "sign", other, "assent", "--officer", "ana"]) == 0
    assert json.loads(capsys.readouterr().out)["approved_via"] == "terminal"


# R3: only the exact assent call can be asked; ids reach the officer's prompt verbatim
_CC = {"hook_event_name": "PreToolUse", "tool_use_id": "tu-1", "session_id": "s-1",
       "permission_mode": "default"}  # fmt: skip


@pytest.mark.parametrize(
    "tool_input",
    [
        {"decision_id": "d-1", "verdict": "assent", "officer": "ana", "credential": "secret"},
        {"decision_id": "d-1", "verdict": "assent", "officer": "ana", "extra": 1},
        {"decision_id": "d-1\n[note]", "verdict": "assent", "officer": "ana"},
        {"decision_id": "d-1", "verdict": "assent", "officer": "ana x"},
        {"decision_id": "d 1", "verdict": "assent", "officer": "ana"},
        {"decision_id": ["d-1"], "verdict": "assent", "officer": "ana"},
        {"verdict": "assent", "officer": "ana"},
        {"decision_id": "d-1", "verdict": "assent", "officer": "ana", "status": "APPROVED"},
        {"decision_id": "d-1", "verdict": "assent", "officer": "ana", "rationale": "x" * 2001},
    ],
)
def test_sign_decision_with_anything_but_the_plain_call_is_denied(tool_input):
    from kyb.signing import gate

    assert gate.check("sign_decision", tool_input, _CC)[0] == gate.DENY


def test_the_plain_sign_decision_call_with_a_rationale_is_asked():
    from kyb.signing import gate

    call = {"decision_id": "dec-B-0001", "verdict": "assent", "officer": "ana",
            "rationale": "reviewed\nwith the team"}  # fmt: skip
    decision, reason = gate.check("sign_decision", call, _CC)
    assert decision == gate.ASK and "\n" not in reason


@pytest.mark.parametrize(
    "command",
    [
        "kyb sign d-1 assent --officer ana\nkyb override d-1",
        "kyb\nsign d-1 assent --officer ana",
        "kyb sign d-1\nassent --officer ana",
        "kyb -w $(pwd) sign d-1 assent --officer ana",
        "kyb sign d-1 assent --officer ana ",
    ],
)
def test_a_multi_line_or_substituted_sign_command_is_denied(command):
    from kyb.signing import gate

    assert gate.check("Bash", {"command": command}, _CC)[0] == gate.DENY
