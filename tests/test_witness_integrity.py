"""Contradictions cannot authorize fallback or verification; offline contract proof only."""

from __future__ import annotations

import json

import pytest
from test_witness_outcomes import assert_terminal, envelope, gemini_route, rewrite
from test_witness_strata import SPEC, SUB, no_answer_stream, parser_route

from kyb.witness.errors import WitnessRefused


@pytest.mark.parametrize(
    "errors",
    [
        ["Invalid JSON schema private-sentinel"],
        ["Authentication failed private-sentinel"],
        ["unsupported CLI option private-sentinel"],
        ["unknown terminal failure private-sentinel"],
        {"type": "authentication_failed"},
    ],
)
def test_r4_final_diagnostics_override_earlier_transient(tmp_path, monkeypatch, errors):
    events = [json.loads(line) for line in no_answer_stream().splitlines()]
    events[-1]["errors"] = errors
    router, calls, downstream = parser_route(
        tmp_path, monkeypatch, "\n".join(map(json.dumps, events)), 1
    )
    assert_terminal(router, calls, downstream)
    record = next((tmp_path / "silver/witness/raw").glob("*/failed/0.json")).read_text()
    assert "private-sentinel" not in record


@pytest.mark.parametrize("status", [400, 401, 403, 418])
def test_r4_final_status_overrides_earlier_transient(tmp_path, monkeypatch, status):
    events = [json.loads(line) for line in no_answer_stream().splitlines()]
    events[-1]["api_error_status"] = status
    assert_terminal(*parser_route(tmp_path, monkeypatch, "\n".join(map(json.dumps, events)), 1))


@pytest.mark.parametrize(
    "defect",
    ["prompt_block", "candidate_block", "error_status", "missing_model", "missing_schema_hash"],
)
def test_r4_all_retained_cache_evidence_is_reconciled(tmp_path, monkeypatch, defect):
    router, calls, downstream = gemini_route(tmp_path, monkeypatch, envelope())
    router.for_document(SUB)["A"].ask("transcribe", [b"crop"], SPEC)
    path = next((tmp_path / "silver/witness/raw").glob("*/0.json"))
    record = json.loads(path.read_text())
    if defect == "prompt_block":
        record["provenance"]["prompt_block_reason"] = "SAFETY"
    elif defect == "candidate_block":
        record["provenance"]["candidates"][0]["safety"] = [
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "probability": "HIGH", "blocked": True}
        ]
    elif defect == "error_status":
        record["provenance"]["error_status"] = {"code": 403, "status": "PERMISSION_DENIED"}
    elif defect == "missing_model":
        record.pop("model")
    else:
        record["provenance"].pop("schema_hash")
    rewrite(path, record)
    assert_terminal(router, calls, downstream)


@pytest.mark.parametrize("earlier", ["rate_limit", "server_error"])
@pytest.mark.parametrize(
    "later",
    [
        ["rate_limit", "unknown private-sentinel"],
        ["server_error", "Invalid JSON schema private-sentinel"],
        ["rate_limit", "server_error"],
        None,
        "rate_limit",
        [429],
        [False],
        [[]],
    ],
)
def test_all_final_diagnostics_must_consistently_indicate_availability(
    tmp_path, monkeypatch, earlier, later
):
    events = [json.loads(line) for line in no_answer_stream(code=earlier).splitlines()]
    events[-1]["errors"] = later
    assert_terminal(*parser_route(tmp_path, monkeypatch, "\n".join(map(json.dumps, events)), 1))


@pytest.mark.parametrize(
    ("earlier", "status", "subtype"),
    [
        ("rate_limit", 500, "success"),
        ("server_error", 429, "success"),
        ("rate_limit", 429, "error_during_execution"),
        ("server_error", 503, "error_during_execution"),
        ("rate_limit", None, "error_max_turns"),
    ],
)
def test_transient_codes_cannot_override_conflicting_final_status_or_shape(
    tmp_path, monkeypatch, earlier, status, subtype
):
    events = [json.loads(line) for line in no_answer_stream(code=earlier).splitlines()]
    events[-1]["subtype"] = subtype
    if status is not None:
        events[-1]["api_error_status"] = status
    assert_terminal(*parser_route(tmp_path, monkeypatch, "\n".join(map(json.dumps, events)), 1))


@pytest.mark.parametrize(
    ("code", "status"),
    [
        ("rate_limit", None),
        ("server_error", None),
        ("rate_limit", 429),
        ("server_error", 500),
        ("server_error", 529),
    ],
)
def test_genuinely_consistent_claude_availability_and_zero_call_replay(
    tmp_path, monkeypatch, code, status
):
    events = [json.loads(line) for line in no_answer_stream(code=code, status=status).splitlines()]
    if status is None:
        events[-1]["errors"] = [code, code]
    else:
        events[-1]["result"] = code
    router, calls, downstream = parser_route(
        tmp_path, monkeypatch, "\n".join(map(json.dumps, events)), 1
    )
    for mode in ("auto", "replay"):
        router.mode = mode
        route = router.for_document(SUB)["A"]
        assert route.ask("transcribe", [b"crop"], SPEC).model == "gemini-3.1-pro-preview"
        assert [x["status"] for x in route.attempts] == ["unavailable", "complete"]
    assert len(calls) == 1 and sum(len(d.calls) for d in downstream) == 1


@pytest.mark.parametrize(
    ("field", "mutation"),
    [
        (field, mutation)
        for field in (
            "request",
            "kind",
            "model",
            "requested_model",
            "prompt_version",
            "usage",
            "provenance",
            "status",
            "record_version",
            "validation_version",
            "data",
            "content_hash",
        )
        for mutation in ("delete", "none", "list", "dict", "bool")
        if (field, mutation) != ("usage", "dict")
    ],
)
def test_complete_cache_required_field_matrix_refuses_before_use(
    tmp_path, monkeypatch, field, mutation
):
    router, calls, downstream = gemini_route(tmp_path, monkeypatch, envelope())
    router.for_document(SUB)["A"].ask("transcribe", [b"crop"], SPEC)
    path = next((tmp_path / "silver/witness/raw").glob("*/0.json"))
    record = json.loads(path.read_text())
    if mutation == "delete":
        record.pop(field)
    else:
        record[field] = {"none": None, "list": [], "dict": {}, "bool": True}[mutation]
    if field == "content_hash":
        path.write_text(json.dumps(record))
    else:
        rewrite(path, record)
    assert_terminal(router, calls, downstream)


@pytest.mark.parametrize(
    "field",
    [
        "reported_model",
        "schema_hash",
        "schema_valid",
        "data_hash",
        "candidates",
        "terminal_evidence",
        "prompt_block_reason",
        "prompt_safety",
        "error_status",
        "error_diagnostics",
        "tools",
        "family",
        "normal_terminal",
        "classification",
        "validation_version",
    ],
)
def test_missing_retained_gemini_fields_never_authorizes_fallback(tmp_path, monkeypatch, field):
    router, calls, downstream = gemini_route(tmp_path, monkeypatch, envelope())
    router.for_document(SUB)["A"].ask("transcribe", [b"crop"], SPEC)
    path = next((tmp_path / "silver/witness/raw").glob("*/0.json"))
    record = json.loads(path.read_text())
    record["provenance"].pop(field)
    rewrite(path, record)
    assert_terminal(router, calls, downstream)


@pytest.mark.parametrize(
    "error",
    [
        LookupError("private-sentinel"),
        KeyError("private-sentinel"),
        IndexError("private-sentinel"),
        ValueError("private-sentinel"),
    ],
)
def test_router_never_falls_back_on_lookup_or_parser_errors(tmp_path, monkeypatch, error):
    router, calls, downstream = gemini_route(tmp_path, monkeypatch, envelope())
    reached = []

    def factory(backend):
        reached.append(backend)
        raise error

    router.factory = factory
    for mode in ("auto", "replay"):
        router.mode = mode
        with pytest.raises(WitnessRefused):
            router.for_document(SUB)["A"].ask("transcribe", [b"crop"], SPEC)
    assert (
        len(reached) == 2
        and set(reached) == {"gemini-api:gemini-3.1-pro-preview"}
        and not calls
        and not downstream
    )


@pytest.mark.parametrize("where", ["assistant_text", "system", "result_text"])
def test_other_observed_terminal_diagnostics_cannot_be_silently_discarded(
    tmp_path, monkeypatch, where
):
    events = [json.loads(line) for line in no_answer_stream().splitlines()]
    if where == "assistant_text":
        events[1]["message"]["content"] = [
            {"type": "text", "text": "Invalid JSON schema private-sentinel"}
        ]
    elif where == "system":
        events.insert(
            1, {"type": "system", "subtype": "error", "error": "unknown private-sentinel"}
        )
    else:
        events[-1] |= {
            "subtype": "success",
            "api_error_status": 429,
            "result": "unknown private-sentinel",
        }
    router, calls, downstream = parser_route(
        tmp_path, monkeypatch, "\n".join(map(json.dumps, events)), 1
    )
    assert_terminal(router, calls, downstream)
    assert (
        "private-sentinel"
        not in next((tmp_path / "silver/witness/raw").glob("*/failed/0.json")).read_text()
    )


@pytest.mark.parametrize(
    "defect",
    [
        "positive",
        "prompt_block",
        "candidate_block",
        "prompt_safety",
        "error_status",
        "missing_model",
        "missing_schema_hash",
    ],
)
def test_actual_pdf_cache_contradictions_remain_unverified(tmp_path, monkeypatch, defect):
    import copy
    import shutil
    from pathlib import Path

    from test_witness_strata import DATA

    from kyb.witness import verify

    repository = Path(__file__).resolve().parents[1]
    source = repository / "tests/fixtures/witness_b05"
    monkeypatch.chdir(tmp_path)
    for part in ("claims", "layout", "pages"):
        shutil.copytree(source / "silver" / part, tmp_path / "silver" / part)
    path = tmp_path / "silver/claims/es-iberia-software/r1/05.json"
    sub = json.loads(path.read_text())
    sub["claims"] = [c for c in sub["claims"] if c["claim_id"] == "c-B05-08"]
    sub["produced_by"] = SUB["produced_by"]
    path.write_text(json.dumps(sub))
    data = copy.deepcopy(DATA)
    data["crops"][0]["transcription"] = "Lucas Navarro Solís 68%"
    data["crops"][0]["reread"]["values"][0]["subject"] = "Lucas Navarro Solís"
    data["crops"][0]["reread"]["values"][0]["value"] = "68%"
    data["crops"].append(
        {"crop_id": "k2", "transcription": "pendiente", "readable": True, "reread": {"values": []}}
    )
    router, calls, downstream = gemini_route(tmp_path, monkeypatch, envelope(data=data))
    pdf = next((repository / "examples/packets/es-iberia-software").glob("05*.pdf"))
    initial = verify.verify_document(tmp_path, "es-iberia-software", "r1", "05", router, pdf)
    assert initial["results"][0]["verified"] and initial["assurance"] == "W2"
    raw = next((tmp_path / "silver/witness/raw").glob("*/0.json"))
    record = json.loads(raw.read_text())
    prov = record["provenance"]
    rating = {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "probability": "HIGH", "blocked": True}
    if defect == "prompt_block":
        prov["prompt_block_reason"] = "SAFETY"
    elif defect == "candidate_block":
        prov["candidates"][0]["safety"] = [rating]
    elif defect == "prompt_safety":
        prov["prompt_safety"] = [rating]
    elif defect == "error_status":
        prov["error_status"] = {"code": 403, "status": "PERMISSION_DENIED"}
    elif defect == "missing_model":
        record.pop("model")
    elif defect == "missing_schema_hash":
        prov.pop("schema_hash")
    rewrite(raw, record)
    for mode in ("auto", "replay"):
        router.mode = mode
        result = verify.verify_document(tmp_path, "es-iberia-software", "r1", "05", router, pdf)
        assert result["results"][0]["verified"] == (defect == "positive")
        assert bool(result.get("routing_refused")) == (defect != "positive")
        if defect != "positive":
            assert result["assurance"] != "W2"
            assert any(f["kind"] == "witness_routing_refused" for f in result["findings"])
    assert len(calls) == 1 and not downstream


@pytest.mark.parametrize(
    "defect",
    [
        "top_status",
        "diagnostic_unknown",
        "missing_errors",
        "errors_shape",
        "model",
        "schema_count",
        "tools",
        "directory",
    ],
)
def test_claude_replay_rechecks_all_retained_diagnostics_and_provenance(
    tmp_path, monkeypatch, defect
):
    router, calls, downstream = parser_route(tmp_path, monkeypatch, no_answer_stream(), 1)
    router.for_document(SUB)["A"].ask("transcribe", [b"crop"], SPEC)
    initial_downstream = sum(len(d.calls) for d in downstream)
    path = next((tmp_path / "silver/witness/raw").glob("*/failed/0.json"))
    record = json.loads(path.read_text())
    prov = record["provenance"]
    if defect == "top_status":
        prov["api_error_status"] = 401
    elif defect == "diagnostic_unknown":
        prov["result_evidence"][0]["errors"] = [
            {"category": "unknown", "hash": prov["response_hash"]}
        ]
    elif defect == "missing_errors":
        prov["result_evidence"][0].pop("errors")
    elif defect == "errors_shape":
        prov["result_evidence"][0]["errors"] = "rate_limit"
    elif defect == "model":
        prov["reported_model"] = "claude-opus-5-5"
    elif defect == "schema_count":
        prov["schema_values_checked"] = 1
    elif defect == "tools":
        prov["tools"] = ["Read"]
    else:
        prov["workdir"]["after"] = ["outside.txt"]
    rewrite(path, record)
    for mode in ("auto", "replay"):
        router.mode = mode
        with pytest.raises(WitnessRefused):
            router.for_document(SUB)["A"].ask("transcribe", [b"crop"], SPEC)
    assert len(calls) == 1 and sum(len(d.calls) for d in downstream) == initial_downstream


@pytest.mark.parametrize(
    "message",
    [
        "unknown private-sentinel",
        "Invalid JSON schema private-sentinel",
        "authentication failed private-sentinel",
    ],
)
def test_gemini_unknown_or_refused_diagnostic_dominates_transient_status(
    tmp_path, monkeypatch, message
):
    from test_witness_outcomes import http_error

    router, calls, downstream = gemini_route(
        tmp_path, monkeypatch, error=http_error(503, "UNAVAILABLE", message=message)
    )
    assert_terminal(router, calls, downstream)
    assert (
        "private-sentinel"
        not in next((tmp_path / "silver/witness/raw").glob("*/failed/0.json")).read_text()
    )


@pytest.mark.parametrize("where", ["finish_message", "quota_description"])
def test_every_observed_gemini_terminal_diagnostic_is_inspected(tmp_path, monkeypatch, where):
    from test_witness_outcomes import http_error

    out = envelope()
    out["candidates"][0]["finishMessage"] = "unknown terminal failure private-sentinel"
    error = (
        http_error(
            429,
            "RESOURCE_EXHAUSTED",
            details=[
                {
                    "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                    "violations": [{"description": "unknown terminal failure private-sentinel"}],
                }
            ],
        )
        if where == "quota_description"
        else None
    )
    router, calls, downstream = gemini_route(tmp_path, monkeypatch, out, error)
    assert_terminal(router, calls, downstream)


@pytest.mark.parametrize("where", ["finish_message", "retry_detail", "quota_detail"])
def test_affirmative_normal_and_consistent_structured_transient_controls(
    tmp_path, monkeypatch, where
):
    from test_witness_outcomes import http_error

    out = envelope()
    error = None
    if where == "finish_message":
        out["candidates"][0]["finishMessage"] = "STOP"
    elif where == "retry_detail":
        error = http_error(
            503,
            "UNAVAILABLE",
            details=[{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "1.5s"}],
        )
    else:
        error = http_error(
            429,
            "RESOURCE_EXHAUSTED",
            details=[
                {
                    "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                    "violations": [{"description": "rate_limit"}],
                }
            ],
        )
    router, calls, downstream = gemini_route(tmp_path, monkeypatch, out, error)
    for mode in ("auto", "replay"):
        router.mode = mode
        answer = router.for_document(SUB)["A"].ask("transcribe", [b"crop"], SPEC)
        assert answer.model == (
            "gemini-3.1-pro-preview" if where == "finish_message" else "claude-opus-5-5"
        )
    assert len(calls) == 1 and sum(len(d.calls) for d in downstream) == (
        0 if where == "finish_message" else 1
    )


@pytest.mark.parametrize("defect", ["missing", "unknown"])
def test_finish_diagnostic_replay_evidence_is_mandatory_and_consistent(
    tmp_path, monkeypatch, defect
):
    router, calls, downstream = gemini_route(tmp_path, monkeypatch, envelope())
    router.for_document(SUB)["A"].ask("transcribe", [b"crop"], SPEC)
    path = next((tmp_path / "silver/witness/raw").glob("*/0.json"))
    record = json.loads(path.read_text())
    if defect == "missing":
        record["provenance"].pop("finish_diagnostics")
    else:
        record["provenance"]["finish_diagnostics"] = [
            {"category": "unknown", "hash": record["provenance"]["response_hash"]}
        ]
    rewrite(path, record)
    assert_terminal(router, calls, downstream)
