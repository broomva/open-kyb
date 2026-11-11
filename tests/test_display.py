"""The display check (issue): `kyb harness ingest-session` reads what a session showed, from the
harness's transcript, and flags factual text about an entity that is not a rendered answer."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
from test_copilot import answer, fact_for, stmt

from kyb.audit.chain import chain_events, verify_workspace, workspace_action
from kyb.copilot.display import FLAG, ingest_session

SESSION = "sess-1"
TOOL = "mcp__plugin_kyb_kyb__record_answer"


def line(kind, content, **extra):
    return json.dumps({"type": kind, "message": {"role": kind, "content": content}, **extra})


def prompt(text="who owns it?"):
    return line("user", text)


def say(text):
    return line("assistant", [{"type": "text", "text": text}])


def called(use_id, result):
    return [
        line("assistant", [{"type": "tool_use", "id": use_id, "name": TOOL, "input": {}}]),
        line("user", [{"type": "tool_result", "tool_use_id": use_id,
                       "content": [{"type": "text", "text": json.dumps(result)}]}]),
    ]  # fmt: skip


def transcript(tmp_path, *lines, name="t.jsonl"):
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n")
    return path


def ingest(ws, path):
    with workspace_action(ws):
        return ingest_session(ws, SESSION, str(path))


def flags(ws):
    return [e for e in chain_events(ws) if e["action"] == FLAG]


@pytest.fixture
def ws(qa_ws):
    return qa_ws


@pytest.fixture
def shown(ws):
    f = fact_for(ws, "owns", "STRAIT")
    out = answer(ws, [stmt(f)], "who owns it?")
    return out, f


def test_a_rendered_answer_shown_verbatim_is_clean(ws, shown, tmp_path):
    out, _ = shown
    t = transcript(tmp_path, prompt(), *called("u1", out), say(out["text"]))
    assert ingest(ws, t) == {"messages": 1, "unverified_answer": 0}
    assert flags(ws) == []


def test_a_rendered_answer_shown_with_characters_cut_from_its_middle_is_flagged(
    ws, shown, tmp_path
):
    # issue: the agent copied a long, repetitive decision answer and dropped 48 characters
    # from the middle. That is a truncation by the agent, which the check must flag (a message
    # is clean only if the recorded text is in it byte for byte), not a false positive.
    out, _ = shown
    text = out["text"]
    mid = len(text) // 2
    cut = text[:mid] + text[mid + 6 :]
    assert cut != text and len(text) - len(cut) == 6
    t = transcript(tmp_path, prompt(), *called("u1", out), say(cut))
    assert ingest(ws, t) == {"messages": 1, "unverified_answer": 1}
    assert flags(ws)[0]["detail"]["answers"] == [out["answer_id"]]


def test_extra_factual_text_beyond_the_rendered_answer_is_flagged(ws, shown, tmp_path):
    out, _ = shown
    msg = out["text"] + "\nSovereign Trust is also under investigation in Belize."
    t = transcript(tmp_path, prompt(), *called("u1", out), say(msg))
    assert ingest(ws, t) == {"messages": 1, "unverified_answer": 1}
    (flag,) = flags(ws)
    assert flag["detail"]["reason"].startswith("factual text about an entity beyond")
    assert flag["detail"]["answers"] == [out["answer_id"]] and flag["detail"]["sha256"]


def test_a_paraphrase_of_the_rendered_answer_is_flagged(ws, shown, tmp_path):
    out, f = shown
    paraphrase = f"{f.subjects[0]} holds the whole company, I believe."
    t = transcript(tmp_path, prompt(), *called("u1", out), say(paraphrase))
    assert ingest(ws, t)["unverified_answer"] == 1


def test_a_statement_about_an_entity_with_no_answer_this_turn_is_flagged(ws, shown, tmp_path):
    out, _ = shown
    # the answer was recorded in an earlier turn; this turn answers from memory
    t = transcript(
        tmp_path,
        prompt("q1"),
        *called("u1", out),
        say(out["text"]),
        prompt("and the agent?"),
        say("Pacific Trustees Ltd is the agent."),
    )
    assert ingest(ws, t) == {"messages": 2, "unverified_answer": 1}
    assert "no rendered answer in this turn" in flags(ws)[0]["detail"]["reason"]


def test_a_rejected_answer_does_not_license_text(ws, tmp_path):
    f = fact_for(ws, "owns", "STRAIT")
    bad = answer(ws, [stmt(f, value="49% of STRAIT CAPITAL PTE. LTD.")], "who owns it?")
    assert not bad["accepted"]
    t = transcript(
        tmp_path,
        prompt(),
        *called("u1", bad),
        say("Sovereign Trust owns 49% of STRAIT CAPITAL PTE. LTD.."),
    )
    assert ingest(ws, t)["unverified_answer"] == 1


def test_talk_that_names_no_entity_is_not_an_answer(ws, tmp_path):
    t = transcript(tmp_path, prompt("hi"), say("Hello. Ask me about a case."))
    assert ingest(ws, t) == {"messages": 1, "unverified_answer": 0}


def test_a_forged_answer_id_licenses_nothing(ws, tmp_path):
    forged = {"answer_id": "a-000000000000", "accepted": True, "text": "Sovereign Trust owns it"}
    t = transcript(tmp_path, prompt(), *called("u1", forged), say("Sovereign Trust owns it"))
    assert ingest(ws, t)["unverified_answer"] == 1


def test_the_cursor_reads_each_message_once_and_resumes_mid_turn(ws, shown, tmp_path):
    out, _ = shown
    head = [prompt(), *called("u1", out)]
    path = transcript(tmp_path, *head)
    assert ingest(ws, path) == {"messages": 0, "unverified_answer": 0}
    with path.open("a") as fh:  # the turn goes on: the answer recorded before the cursor counts
        fh.write(say(out["text"]) + "\n")
    assert ingest(ws, path) == {"messages": 1, "unverified_answer": 0}
    assert ingest(ws, path) == {"messages": 0, "unverified_answer": 0}  # nothing new
    with path.open("a") as fh:
        fh.write(say("Sovereign Trust also owns a yacht.") + "\n")
    assert ingest(ws, path) == {"messages": 1, "unverified_answer": 0 + 1}
    assert len(flags(ws)) == 1 and verify_workspace(ws, at_boundary=True).ok


def test_a_half_written_last_line_waits_for_the_next_hook(ws, tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text(prompt() + "\n" + say("Sovereign Trust is great")[:20])
    assert ingest(ws, path)["messages"] == 0
    path.write_text(prompt() + "\n" + say("Sovereign Trust is great") + "\n")
    assert ingest(ws, path) == {"messages": 1, "unverified_answer": 1}


def test_an_unreadable_transcript_is_nothing_not_a_crash(ws, tmp_path):
    assert ingest(ws, tmp_path / "missing.jsonl") == {"messages": 0, "unverified_answer": 0}


def test_the_hook_flags_but_never_blocks(ws, shown, tmp_path):
    out, _ = shown
    t = transcript(tmp_path, prompt(), say("Sovereign Trust is under investigation."))
    event = json.dumps({"cwd": str(ws), "session_id": SESSION, "transcript_path": str(t)})
    run = subprocess.run(
        [sys.executable, "-m", "kyb.hook", "ingest-session", "--source", "workspace"],
        input=event,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "KYB_WORKSPACE": str(ws)},
        check=False,
    )
    assert run.returncode == 0 and "1 unverified answer" in run.stderr
    assert len(flags(ws)) == 1


def test_the_cli_exits_1_when_something_is_flagged(ws, tmp_path, capsys):
    from kyb.cli import main

    t = transcript(tmp_path, prompt(), say("Sovereign Trust is under investigation."))
    assert (
        main(
            [
                "-w",
                str(ws),
                "harness",
                "ingest-session",
                "--transcript",
                str(t),
                "--session",
                SESSION,
            ]
        )
        == 1
    )
    assert json.loads(capsys.readouterr().out)["unverified_answer"] == 1


def test_a_short_primary_name_is_still_watched(ws, tmp_path):
    from decision_fixtures import decision, record

    record(ws, decision("sg-strait-capital", name="Jev", seed="short"))
    t = transcript(tmp_path, prompt(), say("Jev has active sanctions listings."))
    assert ingest(ws, t)["unverified_answer"] == 1


def test_an_explanation_says_what_the_officer_decided_and_why(ws):
    from decision_fixtures import decision, record

    from kyb.signing import officers
    from kyb.signing.sign import sign_decision

    did = record(ws, decision("sg-strait-capital"))
    with workspace_action(ws):
        tok = officers.enroll(ws, "ana")["token"]
        sign_decision(ws, decision_id=did, verdict="override", officer="ana", officer_token=tok,
                      status="APPROVED", rationale="notarial deed\nverified")  # fmt: skip
    text = answer(ws, [{"type": "explain", "decision_id": did}])["text"]
    assert "overrode the recommendation (MANUAL_REVIEW) to APPROVED: notarial deed verified" in text


def test_a_call_and_its_result_on_either_side_of_a_hook_still_license_the_answer(
    ws, shown, tmp_path
):
    out, _ = shown
    call, result = called("u1", out)
    path = transcript(tmp_path, prompt(), call)
    assert ingest(ws, path)["messages"] == 0  # a hook fires between the call and its result
    with path.open("a") as fh:
        fh.write(result + "\n" + say(out["text"]) + "\n")
    assert ingest(ws, path) == {"messages": 1, "unverified_answer": 0}
