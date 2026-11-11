"""Actual adapter/port/router outcome contracts; offline fakes are not confinement proof."""

from __future__ import annotations

import io
import json
import urllib.error

import pytest
from test_witness_strata import DATA, SPEC, SUB, Transport, attest

from kyb.util.hashing import sha256_json
from kyb.witness import adapters, gemini
from kyb.witness.errors import WitnessRefused
from kyb.witness.port import Witness
from kyb.witness.strata import Config, RoutedWitness


def envelope(reason="STOP", data=None):
    return {
        "modelVersion": "gemini-3.1-pro-preview",
        "candidates": [
            {
                "finishReason": reason,
                "content": {"parts": [{"text": json.dumps(DATA if data is None else data)}]},
            }
        ],
    }


def gemini_route(root, monkeypatch, output=None, error=None):
    attest(root, "gpt-6.1-sol")
    calls, downstream = [], []

    def open_request(*args, **kwargs):
        calls.append(1)
        if error is not None:
            raise error
        raw = output if isinstance(output, bytes) else json.dumps(output).encode()
        return io.BytesIO(raw)

    def factory(backend):
        if backend.startswith("gemini-api:"):
            return gemini.GeminiWitness(
                model="gemini-3.1-pro-preview", retries=0, offline_transport=open_request
            )
        candidate = Transport(backend)
        downstream.append(candidate)
        return candidate

    monkeypatch.setattr(gemini.time, "sleep", lambda _: None)
    return (
        RoutedWitness(
            root,
            config=Config("A", ("gemini-api:gemini-3.1-pro-preview", "claude-p:claude-opus-5-5")),
            factory=factory,
        ),
        calls,
        downstream,
    )


def assert_terminal(router, calls, downstream):
    for mode in ("auto", "replay"):
        router.mode = mode
        with pytest.raises(WitnessRefused):
            router.for_document(SUB)["A"].ask("transcribe", [b"crop"], SPEC)
    assert len(calls) == 1 and not downstream


@pytest.mark.parametrize(
    "reason", ["SAFETY", "MAX_TOKENS", "MALFORMED_FUNCTION_CALL", "UNKNOWN_FUTURE_REASON"]
)
def test_r3_exact_gemini_terminal_reason_is_not_erased_by_schema(tmp_path, monkeypatch, reason):
    assert_terminal(*gemini_route(tmp_path, monkeypatch, envelope(reason)))


def test_r3_exact_gemini_http_schema_is_terminal(tmp_path, monkeypatch):
    body = {
        "error": {
            "code": 400,
            "status": "INVALID_ARGUMENT",
            "message": "Invalid JSON schema private-sentinel",
        }
    }
    error = urllib.error.HTTPError(
        "https://offline.invalid",
        400,
        "private-sentinel",
        {},
        io.BytesIO(json.dumps(body).encode()),
    )
    router, calls, downstream = gemini_route(tmp_path, monkeypatch, error=error)
    assert_terminal(router, calls, downstream)
    raw = next((tmp_path / "silver/witness/raw").glob("*/failed/0.json")).read_text()
    assert "private-sentinel" not in raw


def test_r3_exact_contradictory_failure_type_cannot_promote_capability_refusal(tmp_path):
    attest(tmp_path, "claude-opus-5-5")
    with pytest.raises(WitnessRefused):
        Witness(tmp_path, adapters.CodexWitness("gpt-6.1-sol")).ask("transcribe", [b"crop"], SPEC)
    path = next((tmp_path / "silver/witness/raw").glob("*/failed/0.json"))
    record = json.loads(path.read_text())
    record["failure_kind"] = "unavailable"
    record.pop("content_hash")
    record["content_hash"] = sha256_json(record)
    path.write_text(json.dumps(record))
    downstream = []

    def factory(backend):
        if backend.startswith("codex-exec:"):
            return adapters.CodexWitness("gpt-6.1-sol")
        candidate = Transport(backend)
        downstream.append(candidate)
        return candidate

    router = RoutedWitness(
        tmp_path,
        config=Config("A", ("codex-exec:gpt-6.1-sol", "gemini-api:gemini-3.1-pro-preview")),
        factory=factory,
    )
    for mode in ("auto", "replay"):
        router.mode = mode
        with pytest.raises(WitnessRefused):
            router.for_document(SUB)["A"].ask("transcribe", [b"crop"], SPEC)
    assert not downstream


@pytest.mark.parametrize(
    "reason",
    [
        None,
        "",
        False,
        [],
        "FINISH_REASON_UNSPECIFIED",
        "RECITATION",
        "OTHER",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "UNEXPECTED_TOOL_CALL",
        "CONTINUATION",
    ],
)
def test_gemini_only_affirmative_stop_can_complete(tmp_path, monkeypatch, reason):
    assert_terminal(*gemini_route(tmp_path, monkeypatch, envelope(reason)))


@pytest.mark.parametrize(
    "defect",
    [
        "missing_finish",
        "missing_model",
        "unknown_model",
        "schema",
        "bad_json",
        "extra_candidate",
        "no_candidates",
        "candidate_not_object",
        "blocked_prompt",
        "unknown_block",
        "blocked_rating",
        "invalid_blocked",
        "feedback_invalid",
        "ratings_invalid",
        "function_call",
        "two_answers",
        "thought_only",
        "usage_invalid",
        "json_duplicate",
        "malformed_response",
    ],
)
@pytest.mark.parametrize("status", [200, 429])
def test_gemini_emitted_defects_dominate_http_status(tmp_path, monkeypatch, defect, status):
    out = envelope()
    candidate = out["candidates"][0]
    if defect == "missing_finish":
        candidate.pop("finishReason")
    elif defect == "missing_model":
        out.pop("modelVersion")
    elif defect == "unknown_model":
        out["modelVersion"] = "unknown"
    elif defect == "schema":
        candidate["content"]["parts"][0]["text"] = "{}"
    elif defect == "bad_json":
        candidate["content"]["parts"][0]["text"] = "{"
    elif defect == "extra_candidate":
        out["candidates"].append({"finishReason": "SAFETY"})
    elif defect == "no_candidates":
        out["candidates"] = []
    elif defect == "candidate_not_object":
        out["candidates"] = [False]
    elif defect in {"blocked_prompt", "unknown_block"}:
        out["promptFeedback"] = {
            "blockReason": "SAFETY" if defect == "blocked_prompt" else "UNKNOWN"
        }
    elif defect in {"blocked_rating", "invalid_blocked"}:
        candidate["safetyRatings"] = [{"blocked": True if defect == "blocked_rating" else "false"}]
    elif defect == "feedback_invalid":
        out["promptFeedback"] = []
    elif defect == "ratings_invalid":
        candidate["safetyRatings"] = {}
    elif defect == "function_call":
        candidate["content"]["parts"].insert(
            0, {"functionCall": {"name": "Read", "args": {"path": "private-sentinel"}}}
        )
    elif defect == "two_answers":
        candidate["content"]["parts"].append({"text": "ignored-private-sentinel"})
    elif defect == "thought_only":
        candidate["content"]["parts"][0]["thought"] = True
    elif defect == "usage_invalid":
        out["usageMetadata"] = {"promptTokenCount": -1}
    raw = json.dumps(out).encode()
    if defect == "json_duplicate":
        raw = raw[:-1] + b',"modelVersion":"gemini-other"}'
    if defect == "malformed_response":
        raw += b'\n{"type":'
    error = None
    if status != 200:
        error = urllib.error.HTTPError(
            "https://offline.invalid", status, "private-sentinel", {}, io.BytesIO(raw)
        )
    router, calls, downstream = gemini_route(tmp_path, monkeypatch, raw, error)
    assert_terminal(router, calls, downstream)
    for p in (tmp_path / "silver/witness/raw").rglob("*.json"):
        assert "private-sentinel" not in p.read_text()


def http_error(http_status, provider_status, **extra):
    error = {
        "code": http_status,
        "status": provider_status,
        "message": "temporarily unavailable",
        **extra,
    }
    return urllib.error.HTTPError(
        "https://offline.invalid",
        http_status,
        "redacted",
        {},
        io.BytesIO(json.dumps({"error": error}).encode()),
    )


@pytest.mark.parametrize(
    ("status", "provider_status"),
    [
        (429, "RESOURCE_EXHAUSTED"),
        (500, "INTERNAL"),
        (503, "UNAVAILABLE"),
        (504, "DEADLINE_EXCEEDED"),
    ],
)
def test_explicit_transient_http_availability_can_fall_back_and_replay(
    tmp_path, monkeypatch, status, provider_status
):
    router, calls, downstream = gemini_route(
        tmp_path, monkeypatch, error=http_error(status, provider_status)
    )
    for mode in ("auto", "replay"):
        router.mode = mode
        route = router.for_document(SUB)["A"]
        reply = route.ask("transcribe", [b"crop"], SPEC)
        assert reply.model == "claude-opus-5-5" and reply.replayed == (mode == "replay")
        assert [a["status"] for a in route.attempts] == ["unavailable", "complete"]
    assert len(calls) == 1 and sum(len(d.calls) for d in downstream) == 1


@pytest.mark.parametrize(
    ("status", "provider_status"),
    [
        (400, "INVALID_ARGUMENT"),
        (401, "UNAUTHENTICATED"),
        (402, "FAILED_PRECONDITION"),
        (403, "PERMISSION_DENIED"),
        (404, "NOT_FOUND"),
        (409, "ABORTED"),
        (418, "UNKNOWN"),
        (500, "UNKNOWN"),
        (503, "INVALID_ARGUMENT"),
        (429, "SAFETY"),
    ],
)
def test_schema_config_auth_unknown_http_failures_never_fall_back(
    tmp_path, monkeypatch, status, provider_status
):
    assert_terminal(*gemini_route(tmp_path, monkeypatch, error=http_error(status, provider_status)))


@pytest.mark.parametrize(
    "extra",
    [
        {"message": "Invalid JSON schema private-sentinel"},
        {"message": "safety blocked private-sentinel"},
        {"details": [{"reason": "SAFETY"}]},
        {"code": 400},
        {"status": None},
        {"details": "private-sentinel"},
        {"blocked": True},
    ],
)
def test_explicit_refusal_or_unknown_error_metadata_overrides_transient(
    tmp_path, monkeypatch, extra
):
    router, calls, downstream = gemini_route(
        tmp_path, monkeypatch, error=http_error(429, "RESOURCE_EXHAUSTED", **extra)
    )
    assert_terminal(router, calls, downstream)
    raw = next((tmp_path / "silver/witness/raw").glob("*/failed/0.json")).read_text()
    assert "private-sentinel" not in raw


@pytest.mark.parametrize("error", [TimeoutError(), urllib.error.URLError(ConnectionRefusedError())])
def test_genuine_no_answer_network_failure_falls_back(tmp_path, monkeypatch, error):
    router, calls, downstream = gemini_route(tmp_path, monkeypatch, error=error)
    for mode in ("auto", "replay"):
        router.mode = mode
        assert (
            router.for_document(SUB)["A"].ask("transcribe", [b"crop"], SPEC).model
            == "claude-opus-5-5"
        )
    assert len(calls) == 1 and sum(len(d.calls) for d in downstream) == 1


@pytest.mark.parametrize(
    "error",
    [urllib.error.URLError("unknown private-sentinel"), OSError("unknown private-sentinel")],
)
def test_generic_network_error_is_terminal(tmp_path, monkeypatch, error):
    assert_terminal(*gemini_route(tmp_path, monkeypatch, error=error))


def test_gemini_stop_complete_model_schema_and_replay_positive(tmp_path, monkeypatch):
    router, calls, downstream = gemini_route(tmp_path, monkeypatch, envelope())
    for mode in ("auto", "replay"):
        router.mode = mode
        reply = router.for_document(SUB)["A"].ask("transcribe", [b"crop"], SPEC)
        assert reply.model == "gemini-3.1-pro-preview" and reply.data == DATA
        assert reply.provenance["normal_terminal"] and reply.provenance["schema_valid"]
        assert reply.provenance["candidates"][0]["finish_reason"] == "STOP"
    assert len(calls) == 1 and not downstream


def rewrite(path, record):
    record.pop("content_hash", None)
    record["content_hash"] = sha256_json(record)
    path.write_text(json.dumps(record))


@pytest.mark.parametrize("provider", ["gemini", "claude", "codex", "agy"])
@pytest.mark.parametrize(
    "defect",
    [
        "type_conflict",
        "kind_conflict",
        "type_missing",
        "kind_missing",
        "status_missing",
        "classification_missing",
        "classification_unknown",
        "normal_missing",
        "normal_true",
        "transport_missing",
        "validation_missing",
        "evidence_missing",
        "provenance_status",
        "complete_plus_failure",
    ],
)
def test_all_adapter_failure_metadata_must_agree_even_with_recomputed_hash(
    tmp_path, monkeypatch, provider, defect
):
    from test_witness_strata import no_answer_stream, parser_route

    if provider == "gemini":
        router, calls, downstream = gemini_route(
            tmp_path, monkeypatch, error=http_error(503, "UNAVAILABLE")
        )
    elif provider == "claude":
        router, calls, downstream = parser_route(tmp_path, monkeypatch, no_answer_stream(), 1)
    else:
        attest(tmp_path, "claude-opus-5-5")
        calls, downstream = [], []
        backend = "codex-exec:gpt-6.1-sol" if provider == "codex" else "agy:gemini-3.8-flash"

        def factory(name):
            if name == backend:
                candidate = (
                    adapters.CodexWitness("gpt-6.1-sol")
                    if provider == "codex"
                    else adapters.AgyWitness("gemini-3.8-flash")
                )
                return candidate
            candidate = Transport(name)
            downstream.append(candidate)
            return candidate

        router = RoutedWitness(
            tmp_path,
            config=Config("A", (backend, "gemini-api:gemini-3.1-pro-preview")),
            factory=factory,
        )
    try:
        router.for_document(SUB)["A"].ask("transcribe", [b"crop"], SPEC)
    except WitnessRefused:
        assert provider in {"codex", "agy"}
    initial_calls, initial_downstream = len(calls), sum(len(d.calls) for d in downstream)
    path = next((tmp_path / "silver/witness/raw").glob("*/failed/0.json"))
    record = json.loads(path.read_text())
    if defect == "type_conflict":
        record["failure_type"] = (
            "WitnessRefused" if record["failure_kind"] == "unavailable" else "WitnessUnavailable"
        )
    elif defect == "kind_conflict":
        record["failure_kind"] = (
            "refused" if record["failure_kind"] == "unavailable" else "unavailable"
        )
    elif defect == "type_missing":
        record.pop("failure_type")
    elif defect == "kind_missing":
        record.pop("failure_kind")
    elif defect == "status_missing":
        record.pop("status")
    elif defect == "classification_missing":
        record["provenance"].pop("classification")
    elif defect == "classification_unknown":
        record["provenance"]["classification"] = "unknown"
    elif defect == "normal_missing":
        record["provenance"].pop("normal_terminal")
    elif defect == "normal_true":
        record["provenance"]["normal_terminal"] = True
    elif defect == "transport_missing":
        record["provenance"].pop("transport")
    elif defect == "validation_missing":
        record["provenance"].pop("validation_version")
    elif defect == "evidence_missing":
        record["provenance"].pop("refusal_evidence")
    elif defect == "provenance_status":
        record["provenance"]["status"] = "complete"
    elif defect == "complete_plus_failure":
        record |= {"status": "complete", "data": DATA}
        record["provenance"] |= {"classification": "complete", "normal_terminal": True}
        path = path.parent.parent / "0.json"
    rewrite(path, record)
    for mode in ("auto", "replay"):
        router.mode = mode
        with pytest.raises(WitnessRefused):
            router.for_document(SUB)["A"].ask("transcribe", [b"crop"], SPEC)
    assert (
        len(calls) == initial_calls and sum(len(d.calls) for d in downstream) == initial_downstream
    )


@pytest.mark.parametrize(
    "defect",
    [
        "missing_finish",
        "safety_finish",
        "schema_false",
        "missing_normal",
        "status_refused",
        "failure_type",
        "blocked_metadata",
        "unknown_terminal_field",
    ],
)
def test_successful_gemini_replay_requires_consistent_affirmative_evidence(
    tmp_path, monkeypatch, defect
):
    router, calls, downstream = gemini_route(tmp_path, monkeypatch, envelope())
    router.for_document(SUB)["A"].ask("transcribe", [b"crop"], SPEC)
    path = next((tmp_path / "silver/witness/raw").glob("*/0.json"))
    record = json.loads(path.read_text())
    prov = record["provenance"]
    if defect == "missing_finish":
        prov["candidates"][0].pop("finish_reason")
    elif defect == "safety_finish":
        prov["candidates"][0]["finish_reason"] = "SAFETY"
    elif defect == "schema_false":
        prov["schema_valid"] = False
    elif defect == "missing_normal":
        prov.pop("normal_terminal")
    elif defect == "status_refused":
        prov["classification"] = "refused"
    elif defect == "failure_type":
        record["failure_type"] = "WitnessRefused"
    elif defect == "blocked_metadata":
        prov["terminal_evidence"].append({"field": "blockReason", "value": "SAFETY"})
    else:
        prov["terminal_evidence"].append({"field": "unknown", "value": "STOP"})
    rewrite(path, record)
    for mode in ("auto", "replay"):
        router.mode = mode
        with pytest.raises(WitnessRefused):
            router.for_document(SUB)["A"].ask("transcribe", [b"crop"], SPEC)
    assert len(calls) == 1 and not downstream


@pytest.mark.parametrize("error_code", ["authentication_failed", "billing_error", "unknown_error"])
def test_claude_config_auth_unknown_refusals_cannot_use_transient_status(
    tmp_path, monkeypatch, error_code
):
    from test_witness_strata import no_answer_stream, parser_route

    assert_terminal(
        *parser_route(tmp_path, monkeypatch, no_answer_stream(code=error_code, status=429), 1)
    )


def test_direct_gemini_billing_route_refuses_without_provider_or_fallback(tmp_path, monkeypatch):
    router, calls, downstream = gemini_route(tmp_path, monkeypatch, envelope())
    original = router.factory

    def factory(backend):
        candidate = original(backend)
        if isinstance(candidate, gemini.GeminiWitness):
            candidate.offline_transport = None
        return candidate

    router.factory = factory
    for mode in ("auto", "replay"):
        router.mode = mode
        with pytest.raises(WitnessRefused):
            router.for_document(SUB)["A"].ask("transcribe", [b"crop"], SPEC)
    assert not calls and not downstream


@pytest.mark.parametrize("reason", ["STOP", "SAFETY"])
def test_actual_pdf_verification_requires_normal_gemini_terminal(tmp_path, monkeypatch, reason):
    import copy
    import shutil
    from pathlib import Path

    from kyb.witness import verify

    repository = Path(__file__).resolve().parents[1]
    source = repository / "tests/fixtures/witness_b05"
    monkeypatch.chdir(tmp_path)
    for part in ("claims", "layout", "pages"):
        shutil.copytree(source / "silver" / part, tmp_path / "silver" / part)
    path = tmp_path / "silver/claims/es-iberia-software/r1/05.json"
    sub = json.loads(path.read_text())
    sub["claims"] = [next(c for c in sub["claims"] if c["claim_id"] == "c-B05-08")]
    sub["produced_by"] = SUB["produced_by"]
    path.write_text(json.dumps(sub))
    data = copy.deepcopy(DATA)
    data["crops"][0]["transcription"] = "Lucas Navarro Solís 68%"
    data["crops"][0]["reread"]["values"][0]["subject"] = "Lucas Navarro Solís"
    data["crops"][0]["reread"]["values"][0]["value"] = "68%"
    data["crops"].append(
        {"crop_id": "k2", "transcription": "pendiente", "readable": True, "reread": {"values": []}}
    )
    router, calls, downstream = gemini_route(tmp_path, monkeypatch, envelope(reason, data))
    pdf = next((repository / "examples/packets/es-iberia-software").glob("05*.pdf"))
    for mode in ("auto", "replay"):
        router.mode = mode
        rec = verify.verify_document(tmp_path, "es-iberia-software", "r1", "05", router, pdf)
        assert rec["results"][0]["verified"] == (reason == "STOP")
        assert bool(rec.get("routing_refused")) == (reason != "STOP")
        if reason == "SAFETY":
            assert rec["assurance"] != "W2"
            assert any(f["kind"] == "witness_routing_refused" for f in rec["findings"])
    assert len(calls) == 1 and not downstream


@pytest.mark.parametrize("defect", ["api_status", "model_error", "provider_error"])
def test_claude_success_payload_cannot_erase_error_metadata(tmp_path, monkeypatch, defect):
    from test_witness_strata import claude_stream_case, parser_route

    output, _ = claude_stream_case()
    events = [json.loads(line) for line in output.splitlines()]
    if defect == "api_status":
        events[-1]["api_error_status"] = 429
    elif defect == "model_error":
        events[-1]["terminal_reason"] = "model_error"
    else:
        events[1]["error"] = "rate_limit"
    assert_terminal(*parser_route(tmp_path, monkeypatch, "\n".join(map(json.dumps, events)), 0))


def test_partial_response_read_failure_is_terminal_and_keeps_observed_stop(tmp_path, monkeypatch):
    router, calls, downstream = gemini_route(tmp_path, monkeypatch, envelope())

    class PartialResponse(io.BytesIO):
        status = 200

        def read(self, *args):
            error = urllib.error.URLError(ConnectionResetError())
            error.partial = json.dumps(envelope("SAFETY")).encode()
            raise error

    def open_partial(*args, **kwargs):
        calls.append(1)
        return PartialResponse()

    original = router.factory

    def partial_factory(backend):
        candidate = original(backend)
        if isinstance(candidate, gemini.GeminiWitness):
            candidate.offline_transport = open_partial
        return candidate

    router.factory = partial_factory
    assert_terminal(router, calls, downstream)
    record = json.loads(next((tmp_path / "silver/witness/raw").glob("*/failed/0.json")).read_text())
    assert {"field": "finishReason", "value": "SAFETY"} in record["provenance"]["terminal_evidence"]


@pytest.mark.parametrize("value", [[], {}, None])
def test_non_scalar_failure_markers_are_typed_terminal(tmp_path, monkeypatch, value):
    router, calls, downstream = gemini_route(
        tmp_path, monkeypatch, error=http_error(503, "UNAVAILABLE")
    )
    router.for_document(SUB)["A"].ask("transcribe", [b"crop"], SPEC)
    path = next((tmp_path / "silver/witness/raw").glob("*/failed/0.json"))
    record = json.loads(path.read_text())
    record["failure_kind"] = value
    rewrite(path, record)
    count = sum(len(d.calls) for d in downstream)
    for mode in ("auto", "replay"):
        router.mode = mode
        with pytest.raises(WitnessRefused):
            router.for_document(SUB)["A"].ask("transcribe", [b"crop"], SPEC)
    assert len(calls) == 1 and sum(len(d.calls) for d in downstream) == count


@pytest.mark.parametrize("defect", ["duplicate_type", "contradictory_provider_status"])
def test_integrity_cannot_replace_unambiguous_provider_and_type_evidence(
    tmp_path, monkeypatch, defect
):
    router, calls, downstream = gemini_route(
        tmp_path, monkeypatch, error=http_error(503, "UNAVAILABLE")
    )
    router.for_document(SUB)["A"].ask("transcribe", [b"crop"], SPEC)
    path = next((tmp_path / "silver/witness/raw").glob("*/failed/0.json"))
    record = json.loads(path.read_text())
    if defect == "duplicate_type":
        body = path.read_text().replace(
            '"failure_type": "WitnessUnavailable"',
            '"failure_type": "WitnessRefused", "failure_type": "WitnessUnavailable"',
        )
        path.write_text(body)
    else:
        record["provenance"]["error_status"]["status"] = "INVALID_ARGUMENT"
        rewrite(path, record)
    count = sum(len(d.calls) for d in downstream)
    for mode in ("auto", "replay"):
        router.mode = mode
        with pytest.raises(WitnessRefused):
            router.for_document(SUB)["A"].ask("transcribe", [b"crop"], SPEC)
    assert len(calls) == 1 and sum(len(d.calls) for d in downstream) == count
