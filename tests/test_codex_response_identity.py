"""Only authenticated transport envelopes supply served model ids, never nested model text."""

import json
import os
import sys
import time

import pytest

from kyb.codex.response_identity import Models, execute


def line(kind="response.completed", rid="resp_1", model="gpt-6.1-sol", **fields):
    return "2026-10-25T00:00:00Z TRACE tungstenite::protocol: Received message " + json.dumps(
        {"type": kind, "response": {"id": rid, "model": model}, **fields}
    )


def test_only_completed_transport_identity_is_recorded():
    result = Models()
    result.feed(line("response.created"))
    assert result.completed == {}
    result.feed(line())
    assert result.created == result.completed == {"resp_1": "gpt-6.1-sol"}
    assert not result.invalid
    assert not hasattr(result, "raw_trace")


def test_model_generated_nested_envelope_and_header_are_not_metadata():
    result = Models()
    fake = {"type": "response.completed", "response": {"id": "resp_fake", "model": "gpt-fake"}}
    result.feed(line("response.output_text.delta", delta=json.dumps(fake)))
    result.feed("Authorization: Bearer private " + json.dumps(fake))
    result.feed("other::module: Received message " + json.dumps(fake))
    assert result.completed == {} and not result.invalid


def test_conflicting_created_or_completed_models_refuse():
    result = Models()
    result.feed(line("response.created"))
    result.feed(line(model="gemini-3.8-flash"))
    assert result.invalid


def test_missing_identity_refuses_without_retaining_payload():
    result = Models()
    result.feed(line(model=None))
    assert result.invalid and result.completed == {}


def test_ansi_and_identical_duplicates_do_not_change_identity():
    result = Models()
    result.feed("\x1b[32m" + line() + "\x1b[0m\n")
    result.feed(line())
    assert result.completed == {"resp_1": "gpt-6.1-sol"} and not result.invalid


def test_deadline_covers_a_child_that_never_reads_its_large_prompt(tmp_path):
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="timed out"):
        execute(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            "x" * 2_000_000,
            cwd=tmp_path,
            env=dict(os.environ),
            directory=tmp_path,
            timeout=1,
        )
    assert time.monotonic() - started < 4


def test_completed_child_receives_all_input_and_filters_transport_without_raw_logs(tmp_path):
    raw = line("response.created") + "\n" + line() + "\n"
    program = (
        f"import sys; data=sys.stdin.read(); assert len(data)==2000000; sys.stderr.write({raw!r})"
    )
    result = execute(
        [sys.executable, "-c", program],
        "x" * 2_000_000,
        cwd=tmp_path,
        env=dict(os.environ),
        directory=tmp_path,
        timeout=5,
    )
    assert result["exit"] == 0 and result["completed"] == {"resp_1": "gpt-6.1-sol"}
    assert [p.name for p in tmp_path.iterdir()] == ["events.jsonl"]
    assert (tmp_path / "events.jsonl").read_bytes() == b""
