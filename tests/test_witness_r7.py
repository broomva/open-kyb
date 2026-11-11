"""R7 root-cause refactor (issue): allow-listed outcome parsing, one shared terminal
consistency check, exact served model. Offline mocks only; every repro has a control."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
from test_witness_outcomes import gemini_route, rewrite
from test_witness_strata import (
    SPEC,
    SUB,
    claude_stream_case,
    no_answer_stream,
    parser_route,
)
from test_witness_vertex import Io, envelope, http_error

from kyb.util.hashing import sha256_json
from kyb.witness import adapters
from kyb.witness.claude_stream import ClaudeStream
from kyb.witness.errors import WitnessAnomalous, WitnessRefused, WitnessUnavailable
from kyb.witness.port import Witness

PROBE = Path(__file__).parents[1] / "tests/fixtures/claude_cli/route-probe.jsonl"
FIX = Path(__file__).parent / "fixtures" / "claude_cli"


def with_frame(frame, base=None):
    events = [json.loads(x) for x in (base or no_answer_stream()).splitlines()]
    events.insert(2, frame)
    return "\n".join(map(json.dumps, events))


def both_modes_refuse_claude(tmp_path, monkeypatch, output, rc=1):
    router, calls, downstream = parser_route(tmp_path, monkeypatch, output, rc)
    for mode in ("auto", "replay"):
        router.mode = mode
        with pytest.raises(WitnessRefused):
            router.for_document(SUB)["A"].ask("transcribe", [b"crop"], SPEC)
    assert len(calls) == 1 and not downstream


def both_modes_refuse_gemini(tmp_path, monkeypatch, response):
    router, calls, downstream = gemini_route(tmp_path, monkeypatch, response)
    for mode in ("auto", "replay"):
        router.mode = mode
        with pytest.raises(WitnessRefused):
            router.for_document(SUB)["A"].ask("transcribe", [b"crop"], SPEC)
    assert len(calls) == 1 and not downstream


def both_modes_anomalous(router, calls, downstream, *, claude_calls):
    """Unknown (non-error) shapes: recorded, never verified, bounded self-heal, replay 0 calls."""
    for mode in ("auto", "replay"):
        router.mode = mode
        route = router.for_document(SUB)["A"]
        route.ask("transcribe", [b"crop"], SPEC)
        assert [a["status"] for a in route.attempts] == ["anomalous", "anomalous", "complete"]
        assert route.attempts[0]["anomalies"] and route.attempts[0]["attempt"] == 0
        assert route.attempts[1]["attempt"] == 1
    assert len(calls) == claude_calls


def both_modes_anomalous_claude(tmp_path, monkeypatch, output, rc=1):
    router, calls, downstream = parser_route(tmp_path, monkeypatch, output, rc)
    both_modes_anomalous(router, calls, downstream, claude_calls=2)
    assert sum(len(d.calls) for d in downstream) == 1


def both_modes_anomalous_gemini(tmp_path, monkeypatch, response):
    router, calls, downstream = gemini_route(tmp_path, monkeypatch, response)
    both_modes_anomalous(router, calls, downstream, claude_calls=2)


# 1. R7 repro: Claude system frames --------------------------------------------------------


@pytest.mark.parametrize(
    "frame",
    [
        {"type": "system", "subtype": "status", "error_status": 401},
        {"type": "system", "subtype": "status", "message": "Authentication failed"},
        {"type": "system", "subtype": "hook_response", "message": "boom"},
        {"type": "system", "subtype": "hook_response", "error_status": 403},
        {"type": "system", "subtype": "hook_started", "errorMessage": "x"},
        {"type": "system", "subtype": "init", "Errors": ["x"], "tools": ["StructuredOutput"]},
        {"type": "system", "subtype": "status", "status": "authentication_failed"},
        {"type": "system", "subtype": "api_retry", "error_status": 401},
        {"type": "system", "subtype": "api_retry", "error": "authentication_failed"},
        {"type": "system", "subtype": "api_retry", "error_status": 429, "message": "m"},
        {"type": "system", "subtype": "status", "nested": {"deep": [{"ErrorCode": 7}]}},
    ],
)
def test_claude_unlisted_system_frame_refuses_in_auto_and_replay(tmp_path, monkeypatch, frame):
    both_modes_refuse_claude(tmp_path, monkeypatch, with_frame(frame))


@pytest.mark.parametrize(
    "frame",
    [{"type": "system", "subtype": "never_seen_before"}, {"type": "mystery_frame"}],
)
def test_claude_unknown_non_error_frame_is_an_anomaly_in_auto_and_replay(
    tmp_path, monkeypatch, frame
):
    both_modes_anomalous_claude(tmp_path, monkeypatch, with_frame(frame))


@pytest.mark.parametrize("status", [429, 500, 503, 504, 529])
def test_control_api_retry_transient_status_still_falls_back(tmp_path, monkeypatch, status):
    frame = {"type": "system", "subtype": "api_retry", "error_status": status, "attempt": 1}
    router, calls, downstream = parser_route(tmp_path, monkeypatch, with_frame(frame), 1)
    for mode in ("auto", "replay"):
        router.mode = mode
        route = router.for_document(SUB)["A"]
        assert route.ask("transcribe", [b"crop"], SPEC).model == "gemini-3.1-pro-preview"
        assert [a["status"] for a in route.attempts] == ["unavailable", "complete"]


@pytest.mark.parametrize(
    "frame",
    [
        {"type": "system", "subtype": "status", "status": "requesting"},
        {"type": "system", "subtype": "status", "status": "compacting"},
        {"type": "system", "subtype": "status", "status": None},
        {"type": "system", "subtype": "thinking_tokens", "estimated_tokens": 3,
         "estimated_tokens_delta": 3},
        {"type": "system", "subtype": "hook_started", "hook_id": "h", "hook_name": "n"},
        {"type": "system", "subtype": "hook_response", "hook_id": "h", "exit_code": 0,
         "outcome": "success", "output": ""},
    ],
)  # fmt: skip
def test_control_benign_system_frames_still_fall_back(tmp_path, monkeypatch, frame):
    router, calls, downstream = parser_route(tmp_path, monkeypatch, with_frame(frame), 1)
    route = router.for_document(SUB)["A"]
    assert route.ask("transcribe", [b"crop"], SPEC).model == "gemini-3.1-pro-preview"


# 2. R7 repro: Gemini/Vertex nested and PascalCase error keys -------------------------------

CAND_KEYS = ["errorMessage", "error_message", "errorCode", "error_code", "error_status",
             "error_details", "error", "errors"]  # fmt: skip
ROOT_KEYS = ["ErrorMessage", "Errors", "ErrorCode", "ErrorDetails", "ErrorStatus"]


def with_candidate(**extra):
    out = envelope()
    out["candidates"][0].update(extra)
    return out


@pytest.mark.parametrize("key", CAND_KEYS)
def test_gemini_candidate_error_keys_refuse(tmp_path, monkeypatch, key):
    both_modes_refuse_gemini(tmp_path, monkeypatch, with_candidate(**{key: "boom"}))
    with pytest.raises(WitnessRefused):
        Witness(tmp_path / "v", Io(with_candidate(**{key: "boom"})).witness()).ask(
            "transcribe", [b"crop"], SPEC
        )


@pytest.mark.parametrize("key", ROOT_KEYS)
def test_gemini_pascal_case_root_error_keys_refuse(tmp_path, monkeypatch, key):
    out = {**envelope(), key: "boom"}
    both_modes_refuse_gemini(tmp_path, monkeypatch, out)
    with pytest.raises(WitnessRefused):
        Witness(tmp_path / "v", Io(out).witness()).ask("transcribe", [b"crop"], SPEC)


def test_gemini_tool_evidence_candidate_key_refuses(tmp_path, monkeypatch):
    both_modes_refuse_gemini(tmp_path, monkeypatch, with_candidate(groundingMetadata={}))


@pytest.mark.parametrize("extra", [{"unknownKey": 1}, {"logprobs": 1}])
def test_gemini_unknown_candidate_key_is_an_anomaly(tmp_path, monkeypatch, extra):
    both_modes_anomalous_gemini(tmp_path, monkeypatch, with_candidate(**extra))


def test_control_gemini_known_candidate_keys_and_empty_errors_complete(tmp_path):
    out = with_candidate(index=0, safetyRatings=[], avgLogprobs=-0.1, tokenCount=3,
                         citationMetadata={}, finishMessage="STOP")  # fmt: skip
    assert Witness(tmp_path, Io(out).witness()).ask("transcribe", [b"crop"], SPEC)
    out["errors"], out["ErrorMessage"] = [], ""  # empty unknown keys are still anomalies
    with pytest.raises(WitnessAnomalous):
        Witness(tmp_path / "v", Io(out).witness()).ask("transcribe", [b"crop"], SPEC)


def test_control_gemini_http_503_unavailable_still_falls_back(tmp_path, monkeypatch):
    with pytest.raises(WitnessUnavailable):
        Witness(tmp_path, Io(error=http_error(503, "UNAVAILABLE")).witness()).ask(
            "transcribe", [b"crop"], SPEC
        )


# 3. R7 repro: symmetric terminal consistency ----------------------------------------------


def test_claude_unavailable_replay_refuses_retained_answer_hash(tmp_path, monkeypatch):
    router, calls, downstream = parser_route(tmp_path, monkeypatch, no_answer_stream(), 1)
    route = router.for_document(SUB)["A"]
    assert route.ask("transcribe", [b"crop"], SPEC).model == "gemini-3.1-pro-preview"
    paths = list((tmp_path / "silver/witness/raw").glob("*/failed/0.json"))
    assert paths
    for path in paths:
        rec = json.loads(path.read_text())
        rec["provenance"]["final_output_hash"] = sha256_json({"crops": []})
        rewrite(path, rec)
    router.mode = "replay"
    with pytest.raises(WitnessRefused):
        router.for_document(SUB)["A"].ask("transcribe", [b"crop"], SPEC)


@pytest.mark.parametrize(
    ("field", "value"),
    [("final_output_hash", "h"), ("schema_text_hash", "h"), ("reported_model", "claude-opus-5-5"),
     ("schema_checks", [{"valid": True}]), ("reported_models", ["claude-opus-5-5"])],
)  # fmt: skip
def test_shared_check_rejects_answer_evidence_on_any_provider(field, value):
    from kyb.witness.evidence import require_no_answer_evidence

    require_no_answer_evidence({}, "x")
    with pytest.raises(WitnessRefused):
        require_no_answer_evidence({field: value}, "x")


# 4. R7 repro: served model ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("requested", "served", "ok"),
    [
        ("claude-haiku-4-5", "claude-haiku-4-5", True),
        ("claude-haiku-4-5", "claude-haiku-4-5-20251001", True),
        ("claude-haiku-4-5", "claude-haiku-4-5-evil", False),
        ("claude-haiku-4-5", "claude-haiku-4-5-2025100", False),
        ("claude-haiku-4-5", "claude-haiku-4-5-20251001x", False),
        ("claude-opus-5", "claude-opus-5-5", False),
        ("claude-opus-5", "claude-opus-5-20260101", True),
    ],
)
def test_served_model_is_exact_or_dated_snapshot(requested, served, ok):
    assert adapters.served_model_matches(requested, served) is ok
    stream = ClaudeStream({}, lambda v, s: None, True)
    stream.models.add(served)
    adapters.ClaudeWitness(requested).check_served_model(stream)
    assert ("served_model_mismatch" in stream.refusals) is (not ok)


def test_served_model_evil_suffix_refuses_end_to_end(tmp_path, monkeypatch):
    output, rc = claude_stream_case()
    output = output.replace("claude-opus-5-5", "claude-opus-5-5-evil")
    both_modes_refuse_claude(tmp_path, monkeypatch, output, rc)


# 5. Controls: real recorded positives are not falsely refused ------------------------------


@pytest.mark.parametrize("path", [PROBE, FIX / "p2.stream.jsonl", FIX / "p3.restricted.jsonl"])
def test_recorded_streams_hit_no_allow_list_refusal(path):
    stream = ClaudeStream({}, lambda v, s: None, True)
    stream.consume(path.read_bytes())
    assert not [r for r in stream.refusals if "unlisted" in r or "error_key" in r], stream.refusals


def test_control_synthetic_claude_positive_still_complete(tmp_path, monkeypatch):
    router, calls, downstream = parser_route(tmp_path, monkeypatch, *claude_stream_case())
    for mode in ("auto", "replay"):
        router.mode = mode
        router.for_document(SUB)["A"].ask("transcribe", [b"crop"], SPEC)
    assert not downstream


# 6. Generative: error-ish key, random casing, random depth ---------------------------------

STEMS = ["error", "errors", "error_message", "errorMessage", "error_code", "ErrorDetails",
         "error-status", "error_x", "errorstatus"]  # fmt: skip
SEEDS = list(range(24))


def cased(rng, key):
    return "".join(c.upper() if rng.random() < 0.5 else c.lower() for c in key)


def nest(rng, depth, key, value):
    node = {cased(rng, key): value}
    for _ in range(depth):
        node = rng.choice([{"x": node}, {"k": [{"y": 1}, node]}])
    return node


@pytest.mark.parametrize("seed", SEEDS)
def test_generated_error_key_in_any_claude_system_frame_refuses(tmp_path, monkeypatch, seed):
    rng = random.Random(seed)
    key, depth = rng.choice(STEMS), rng.randint(0, 4)
    value = rng.choice(["boom", 401, ["x"], {"a": 1}, True])
    frame = {
        "type": "system",
        "subtype": rng.choice(["status", "thinking_tokens", "hook_started", "api_retry"]),
    }
    if depth == 0:
        frame[cased(rng, key)] = value
    else:
        frame["nested"] = nest(rng, depth - 1, key, value)
    both_modes_refuse_claude(tmp_path, monkeypatch, with_frame(frame))


@pytest.mark.parametrize("seed", SEEDS)
def test_generated_error_key_inside_init_containers_refuses(tmp_path, monkeypatch, seed):
    rng = random.Random(seed)
    events = [json.loads(x) for x in no_answer_stream().splitlines()]
    holder = rng.choice(["plugins", "agents", "capabilities"])
    events[0][holder] = [nest(rng, rng.randint(0, 4), rng.choice(STEMS), "boom")]
    both_modes_refuse_claude(tmp_path, monkeypatch, "\n".join(map(json.dumps, events)))


@pytest.mark.parametrize("seed", SEEDS)
def test_generated_unknown_frame_subtype_is_an_anomaly(tmp_path, monkeypatch, seed):
    rng = random.Random(seed)
    sub = "".join(rng.choice("abcdefgh_") for _ in range(rng.randint(3, 12))) + "_x"
    both_modes_anomalous_claude(
        tmp_path, monkeypatch, with_frame({"type": "system", "subtype": sub})
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_generated_error_key_in_any_gemini_location_refuses(tmp_path, monkeypatch, seed):
    rng = random.Random(seed)
    out = envelope()
    out["usageMetadata"] = {"promptTokenCount": 1}
    out["promptFeedback"] = {}
    key = rng.choice(STEMS)
    spot = rng.choice(["root", "candidate", "usage", "feedback", "content"])
    holder = {
        "root": out,
        "candidate": out["candidates"][0],
        "usage": out["usageMetadata"],
        "feedback": out["promptFeedback"],
        "content": out["candidates"][0]["content"],
    }[spot]
    holder["injected"] = nest(rng, rng.randint(0, 4), key, rng.choice(["boom", 7, ["x"], {"a": 1}]))
    if rng.random() < 0.4:
        holder.pop("injected")
        holder[cased(rng, key)] = "boom"
    both_modes_refuse_gemini(tmp_path, monkeypatch, out)
    with pytest.raises(WitnessRefused):
        Witness(tmp_path / "vertex", Io(out).witness()).ask("transcribe", [b"crop"], SPEC)


def test_control_generated_inputs_without_error_keys_complete(tmp_path):
    out = envelope()
    out["usageMetadata"] = {"promptTokenCount": 1}
    out["candidates"][0]["index"] = 0
    assert Witness(tmp_path, Io(out).witness()).ask("transcribe", [b"crop"], SPEC)
