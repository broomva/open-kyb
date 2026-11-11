"""R6 repairs (issue): system-frame errors, explicit Gemini error metadata, contradictory
no-answer evidence. Offline mocks only; each regression has a transient/positive control."""

from __future__ import annotations

import json

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
from kyb.witness.errors import WitnessAnomalous, WitnessRefused, WitnessUnavailable
from kyb.witness.port import Witness


def with_system_frame(frame):
    events = [json.loads(x) for x in no_answer_stream().splitlines()]
    events.insert(2, frame)
    return "\n".join(map(json.dumps, events))


# 1. Claude system frames -----------------------------------------------------------------


@pytest.mark.parametrize(
    "frame",
    [
        {"type": "system", "subtype": "api_retry", "error_status": 401},
        {"type": "system", "subtype": "api_retry", "error_status": 400},
        {"type": "system", "subtype": "api_retry", "status": 403},
        {
            "type": "system",
            "subtype": "api_retry",
            "message": "Invalid JSON schema private-sentinel",
        },
    ],
)
def test_system_frame_error_diagnostic_prevents_fallback_in_auto_and_replay(
    tmp_path, monkeypatch, frame
):
    router, calls, downstream = parser_route(tmp_path, monkeypatch, with_system_frame(frame), 1)
    for mode in ("auto", "replay"):
        router.mode = mode
        with pytest.raises(WitnessRefused):
            router.for_document(SUB)["A"].ask("transcribe", [b"crop"], SPEC)
    assert len(calls) == 1 and not downstream


@pytest.mark.parametrize(
    "frame",
    [
        None,
        {"type": "system", "subtype": "api_retry", "error_status": 529},
        {"type": "system", "subtype": "status", "status": "requesting"},
    ],
)
def test_control_genuine_transient_system_frames_still_fall_back(tmp_path, monkeypatch, frame):
    out = no_answer_stream() if frame is None else with_system_frame(frame)
    router, calls, downstream = parser_route(tmp_path, monkeypatch, out, 1)
    for mode in ("auto", "replay"):
        router.mode = mode
        route = router.for_document(SUB)["A"]
        assert route.ask("transcribe", [b"crop"], SPEC).model == "gemini-3.1-pro-preview"
        assert [a["status"] for a in route.attempts] == ["unavailable", "complete"]
    assert len(calls) == 1 and sum(len(d.calls) for d in downstream) == 1


def test_control_unmodified_claude_positive_unchanged(tmp_path, monkeypatch):
    router, calls, downstream = parser_route(tmp_path, monkeypatch, *claude_stream_case())
    for mode in ("auto", "replay"):
        router.mode = mode
        assert router.for_document(SUB)["A"].ask("transcribe", [b"crop"], SPEC).model == (
            "claude-opus-5-5"
        )
    assert len(calls) == 1 and not downstream


# 2. Gemini / Vertex explicit error metadata ------------------------------------------------

EXPLICIT = [
    {"errors": ["Invalid JSON schema private-sentinel"]},
    {"error_message": "Authentication failed private-sentinel"},
    {"error_code": 401},
    {"errorMessage": "boom"},
]


@pytest.mark.parametrize("extra", EXPLICIT)
def test_vertex_explicit_error_metadata_refuses(tmp_path, extra):
    with pytest.raises(WitnessRefused):
        Witness(tmp_path, Io({**envelope(), **extra}).witness()).ask("transcribe", [b"crop"], SPEC)


@pytest.mark.parametrize("extra", EXPLICIT)
def test_gemini_api_explicit_error_metadata_refuses_in_auto_and_replay(
    tmp_path, monkeypatch, extra
):
    router, calls, downstream = gemini_route(tmp_path, monkeypatch, {**envelope(), **extra})
    for mode in ("auto", "replay"):
        router.mode = mode
        with pytest.raises(WitnessRefused):
            router.for_document(SUB)["A"].ask("transcribe", [b"crop"], SPEC)
    assert len(calls) == 1 and not downstream


def test_control_empty_error_metadata_and_plain_positive_still_complete(tmp_path):
    assert Witness(tmp_path, Io().witness()).ask("transcribe", [b"crop"], SPEC)
    plain = tmp_path / "empty"
    out = {**envelope(), "errors": [], "error_message": ""}
    with pytest.raises(WitnessAnomalous):  # unknown (empty) keys are recorded, never verified
        Witness(plain, Io(out).witness()).ask("transcribe", [b"crop"], SPEC)


# 3. Contradictory no-answer evidence -------------------------------------------------------


@pytest.mark.parametrize("field", ["final_output_hash", "schema_text_hash"])
def test_unavailable_replay_refuses_retained_answer_hash(tmp_path, field):
    with pytest.raises(WitnessUnavailable):
        Witness(tmp_path, Io(error=http_error(503, "UNAVAILABLE")).witness()).ask(
            "transcribe", [b"crop"], SPEC
        )
    path = next((tmp_path / "silver/witness/raw").glob("*/failed/0.json"))
    rec = json.loads(path.read_text())
    rec["provenance"][field] = sha256_json({"crops": []})
    rewrite(path, rec)
    with pytest.raises(WitnessRefused):
        Witness(tmp_path, Io().witness(), mode="replay").ask("transcribe", [b"crop"], SPEC)


def test_control_unmodified_unavailable_replay_still_unavailable(tmp_path):
    with pytest.raises(WitnessUnavailable):
        Witness(tmp_path, Io(error=http_error(503, "UNAVAILABLE")).witness()).ask(
            "transcribe", [b"crop"], SPEC
        )
    with pytest.raises(WitnessUnavailable):
        Witness(tmp_path, Io().witness(), mode="replay").ask("transcribe", [b"crop"], SPEC)
