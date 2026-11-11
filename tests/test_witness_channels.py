"""All-channel and final-output consistency; mocked adapters do not prove live confinement."""

from __future__ import annotations

import copy
import json

import pytest
from test_witness_outcomes import assert_terminal, rewrite
from test_witness_strata import DATA, SPEC, SUB, claude_stream_case, no_answer_stream, parser_route


@pytest.mark.parametrize(
    "diagnostic",
    [
        "Invalid JSON schema private-sentinel",
        "Authentication failed private-sentinel",
        "unknown terminal failure private-sentinel",
    ],
)
def test_r5_stderr_diagnostic_cannot_disappear_beside_transient_stdout(
    tmp_path, monkeypatch, diagnostic
):
    router, calls, downstream = parser_route(
        tmp_path, monkeypatch, no_answer_stream(), 1, diagnostic=diagnostic
    )
    assert_terminal(router, calls, downstream)
    assert (
        "private-sentinel"
        not in next((tmp_path / "silver/witness/raw").glob("*/failed/0.json")).read_text()
    )


def test_r5_cached_claude_payload_must_match_final_schema_evidence(tmp_path, monkeypatch):
    router, calls, downstream = parser_route(tmp_path, monkeypatch, *claude_stream_case())
    router.for_document(SUB)["A"].ask("transcribe", [b"crop"], SPEC)
    path = next((tmp_path / "silver/witness/raw").glob("*/0.json"))
    rec = json.loads(path.read_text())
    rec["data"]["crops"][0]["transcription"] = "changed private-sentinel"
    rewrite(path, rec)
    assert_terminal(router, calls, downstream)


@pytest.mark.parametrize("stdout_kind", ["empty", "transient", "complete", "truncated"])
@pytest.mark.parametrize(
    "stderr_kind",
    [
        "empty",
        "whitespace",
        "plain_schema",
        "plain_auth",
        "plain_unknown",
        "plain_transient",
        "structured_schema",
        "structured_auth",
        "structured_unknown",
        "structured_transient",
        "conflicting_status",
        "response_frame",
    ],
)
@pytest.mark.parametrize("exit_kind", [0, 1, "timeout"])
def test_all_stdout_stderr_exit_combinations_fail_closed_or_consistently_fallback(
    tmp_path, monkeypatch, stdout_kind, stderr_kind, exit_kind
):
    from kyb.witness.errors import WitnessRefused

    out = (
        ""
        if stdout_kind == "empty"
        else no_answer_stream()
        if stdout_kind in ("transient", "truncated")
        else claude_stream_case()[0]
    )
    if stdout_kind == "truncated":
        out += '\n{"type":'
    stderr = {
        "empty": "",
        "whitespace": " \n\t ",
        "plain_schema": "Invalid JSON schema private-sentinel",
        "plain_auth": "Authentication failed private-sentinel",
        "plain_unknown": "unknown private-sentinel",
        "plain_transient": "rate_limit",
        "structured_schema": json.dumps(
            {"error": {"code": 400, "message": "Invalid JSON schema private-sentinel"}}
        ),
        "structured_auth": json.dumps(
            {"error": {"type": "authentication_error", "message": "private-sentinel"}}
        ),
        "structured_unknown": json.dumps(
            {"error": {"type": "rate_limit_error", "message": "unknown private-sentinel"}}
        ),
        "structured_transient": json.dumps(
            {"type": "error", "error": {"type": "rate_limit_error", "message": "rate_limit"}}
        ),
        "conflicting_status": json.dumps({"error": {"code": 401, "message": "rate_limit"}}),
        "response_frame": json.dumps(
            {
                "type": "assistant",
                "message": {
                    "model": "claude-opus-5-5",
                    "content": [
                        {"type": "tool_use", "name": "Read", "input": {"path": "private-sentinel"}}
                    ],
                },
            }
        ),
    }[stderr_kind]
    clean = stderr_kind in ("empty", "whitespace")
    transient = stderr_kind in ("plain_transient", "structured_transient")
    fallback = (
        stdout_kind == "empty"
        and exit_kind in (1, "timeout")
        and (clean or transient)
        or stdout_kind == "transient"
        and exit_kind in (0, 1)
        and (clean or transient)
    )
    complete = stdout_kind == "complete" and exit_kind == 0 and clean
    router, calls, downstream = parser_route(
        tmp_path, monkeypatch, out, exit_kind, diagnostic=stderr
    )
    for mode in ("auto", "replay"):
        router.mode = mode
        if fallback or complete:
            route = router.for_document(SUB)["A"]
            response = route.ask("transcribe", [b"crop"], SPEC)
            assert response.model == ("gemini-3.1-pro-preview" if fallback else "claude-opus-5-5")
            assert [a["status"] for a in route.attempts] == (
                ["unavailable", "complete"] if fallback else ["complete"]
            )
        else:
            with pytest.raises(WitnessRefused):
                router.for_document(SUB)["A"].ask("transcribe", [b"crop"], SPEC)
    assert len(calls) == 1 and sum(len(d.calls) for d in downstream) == int(fallback)
    for path in (tmp_path / "silver/witness/raw").rglob("*.json"):
        assert "private-sentinel" not in path.read_text()


@pytest.mark.parametrize("where", ["assistant", "nested", "rate_status", "overage_status"])
def test_r5_adjacent_observed_error_fields_and_status_enums_are_not_ignored(
    tmp_path, monkeypatch, where
):
    events = [json.loads(line) for line in no_answer_stream().splitlines()]
    if where == "assistant":
        events[1]["error_message"] = "Invalid JSON schema private-sentinel"
    elif where == "nested":
        events[1]["message"]["error_message"] = "Authentication failed private-sentinel"
    else:
        events.insert(
            2,
            {
                "type": "rate_limit_event",
                "rate_limit_info": {"status": "unknown_private_sentinel"}
                if where == "rate_status"
                else {"status": "allowed", "overage_status": "unknown_private_sentinel"},
            },
        )
    router, calls, downstream = parser_route(
        tmp_path, monkeypatch, "\n".join(map(json.dumps, events)), 1
    )
    if where in ("rate_status", "overage_status"):
        # an unknown enum value is an envelope anomaly (recorded with its value), not a refusal
        from test_witness_r7 import both_modes_anomalous

        both_modes_anomalous(router, calls, downstream, claude_calls=2)
        return
    assert_terminal(router, calls, downstream)
    assert (
        "private-sentinel"
        not in next((tmp_path / "silver/witness/raw").glob("*/failed/0.json")).read_text()
    )


@pytest.mark.parametrize("provider", ["claude", "gemini"])
@pytest.mark.parametrize(
    "mutation",
    [
        "transcription",
        "value",
        "annotations",
        "struck",
        "readability",
        "missing_final_hash",
        "invalid_final_hash",
        "conflicting_final_hash",
        "conflicting_schema_hash",
    ],
)
def test_final_output_identity_binds_every_material_field_in_auto_and_replay(
    tmp_path, monkeypatch, provider, mutation
):
    from test_witness_outcomes import envelope, gemini_route

    from kyb.util.hashing import sha256_json

    router, calls, downstream = (
        parser_route(tmp_path, monkeypatch, *claude_stream_case())
        if provider == "claude"
        else gemini_route(tmp_path, monkeypatch, envelope())
    )
    router.for_document(SUB)["A"].ask("transcribe", [b"crop"], SPEC)
    path = next((tmp_path / "silver/witness/raw").glob("*/0.json"))
    record = json.loads(path.read_text())
    prov = record["provenance"]
    crop = record["data"]["crops"][0]
    if mutation == "transcription":
        crop["transcription"] = "Irene 10%"
    elif mutation == "value":
        crop["reread"]["values"][0]["value"] = "10%"
    elif mutation == "annotations":
        crop["reread"]["annotations"] = ["pending transfer"]
    elif mutation == "struck":
        crop["reread"]["values"][0]["struck"] = True
        crop["reread"]["struck"] = ["25%"]
    elif mutation == "readability":
        crop["readable"] = False
    elif mutation == "missing_final_hash":
        prov.pop("final_output_hash")
    elif mutation == "invalid_final_hash":
        prov["final_output_hash"] = []
    elif mutation == "conflicting_final_hash":
        prov["final_output_hash"] = sha256_json({"wrong": "final"})
    elif provider == "claude":
        next(c for c in prov["schema_checks"] if c["source"] == "final")["value_hash"] = (
            sha256_json({"wrong": "schema"})
        )
    else:
        prov["schema_hash"] = sha256_json({"wrong": "schema"})
    rewrite(path, record)
    assert_terminal(router, calls, downstream)


def test_multiple_distinct_valid_schema_attempts_cannot_substitute_for_the_final_output(
    tmp_path, monkeypatch
):
    from kyb.util.hashing import sha256_json

    earlier = copy.deepcopy(DATA)
    earlier["crops"][0]["transcription"] = "Irene 10%"
    earlier["crops"][0]["reread"]["values"][0]["value"] = "10%"
    events = [json.loads(line) for line in claude_stream_case()[0].splitlines()]
    events[1]["message"]["content"] = [
        {"type": "tool_use", "id": "schema1", "name": "StructuredOutput", "input": earlier}
    ]
    router, calls, downstream = parser_route(
        tmp_path, monkeypatch, "\n".join(map(json.dumps, events)), 0
    )
    response = router.for_document(SUB)["A"].ask("transcribe", [b"crop"], SPEC)
    assert response.data == DATA
    path = next((tmp_path / "silver/witness/raw").glob("*/0.json"))
    record = json.loads(path.read_text())
    assert {c["source"] for c in record["provenance"]["schema_checks"]} == {"tool", "final"}
    assert sha256_json(earlier) in [c["value_hash"] for c in record["provenance"]["schema_checks"]]
    record["data"] = earlier
    rewrite(path, record)
    assert_terminal(router, calls, downstream)


def pdf_case(root, monkeypatch):
    """Real Poppler/render/match/bind input; only provider transports will be mocked."""
    import shutil
    from pathlib import Path

    repository = Path(__file__).resolve().parents[1]
    source = repository / "tests/fixtures/witness_b05"
    monkeypatch.chdir(root)
    for part in ("claims", "layout", "pages"):
        shutil.copytree(source / "silver" / part, root / "silver" / part)
    path = root / "silver/claims/es-iberia-software/r1/05.json"
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
    pdf = next((repository / "examples/packets/es-iberia-software").glob("05*.pdf"))
    return data, pdf


def pdf_router(root, monkeypatch, data, *, provider="claude", transient=False, stderr=""):
    import io
    from types import SimpleNamespace

    from test_witness_outcomes import envelope
    from test_witness_strata import attest

    from kyb.witness import adapters, gemini
    from kyb.witness.strata import Config, RoutedWitness

    attest(root, "gpt-6.1-sol")
    calls = {"claude": 0, "gemini": 0}
    original_run = adapters.subprocess.run
    events = [json.loads(line) for line in claude_stream_case()[0].splitlines()]
    events[-1]["structured_output"] = data
    output = no_answer_stream() if transient else "\n".join(map(json.dumps, events))

    def process(command, **kwargs):
        if command[0] != "claude":
            return original_run(command, **kwargs)
        calls["claude"] += 1
        return SimpleNamespace(returncode=int(transient), stdout=output, stderr=stderr)

    def offline_response(body, **kwargs):
        calls["gemini"] += 1
        return io.BytesIO(json.dumps(envelope(data=data)).encode())

    def factory(backend):
        if backend.startswith("claude-p:"):
            return adapters.ClaudeWitness("claude-opus-5-5")
        return gemini.GeminiWitness(retries=0, offline_transport=offline_response)

    monkeypatch.setattr(adapters.subprocess, "run", process)
    backends = ("claude-p:claude-opus-5-5", "gemini-api:gemini-3.1-pro-preview")
    return RoutedWitness(
        root,
        config=Config("A", backends if provider == "claude" else backends[::-1]),
        factory=factory,
    ), calls


@pytest.mark.parametrize(
    "stderr",
    [
        "",
        "rate_limit",
        "Invalid JSON schema private-sentinel",
        "Authentication failed private-sentinel",
        "unknown private-sentinel",
    ],
)
def test_actual_pdf_stderr_refusals_dominate_transient_stdout(tmp_path, monkeypatch, stderr):
    from kyb.witness import verify

    data, pdf = pdf_case(tmp_path, monkeypatch)
    router, calls = pdf_router(tmp_path, monkeypatch, data, transient=True, stderr=stderr)
    permitted = stderr in ("", "rate_limit")
    for mode in ("auto", "replay"):
        router.mode = mode
        result = verify.verify_document(tmp_path, "es-iberia-software", "r1", "05", router, pdf)
        assert result["results"][0]["verified"] is permitted
        assert bool(result.get("routing_refused")) is not permitted
        assert (result["assurance"] == "W2") is permitted
    assert calls == {"claude": 1, "gemini": int(permitted)}
    for path in (tmp_path / "silver/witness/raw").rglob("*.json"):
        assert "private-sentinel" not in path.read_text()


@pytest.mark.parametrize("provider", ["claude", "gemini"])
@pytest.mark.parametrize("answer", ["ten", "twentyfive", "mutated"])
def test_actual_pdf_final_hash_rejects_changed_answer_without_rejecting_original(
    tmp_path, monkeypatch, provider, answer
):
    from kyb.witness import verify

    data, pdf = pdf_case(tmp_path, monkeypatch)
    final = copy.deepcopy(data)
    if answer != "twentyfive":
        data["crops"][0]["transcription"] = "Lucas Navarro Solís 10%"
        data["crops"][0]["reread"]["values"][0]["value"] = "10%"
    router, calls = pdf_router(tmp_path, monkeypatch, data, provider=provider)
    initial = verify.verify_document(tmp_path, "es-iberia-software", "r1", "05", router, pdf)
    assert initial["results"][0]["verified"] == (answer == "twentyfive")
    assert initial["assurance"] == "W2" and not initial.get("routing_refused")
    if answer == "mutated":
        raw = next((tmp_path / "silver/witness/raw").glob("*/0.json"))
        record = json.loads(raw.read_text())
        record["data"] = final
        rewrite(raw, record)
    for mode in ("auto", "replay"):
        router.mode = mode
        result = verify.verify_document(tmp_path, "es-iberia-software", "r1", "05", router, pdf)
        assert result["results"][0]["verified"] == (answer == "twentyfive")
        assert bool(result.get("routing_refused")) == (answer == "mutated")
        assert (result["assurance"] == "W2") == (answer != "mutated")
    assert calls == {"claude": int(provider == "claude"), "gemini": int(provider == "gemini")}


@pytest.mark.parametrize("explicit_key", [None, "offline-test-only"])
def test_default_direct_gemini_route_never_reads_credentials_or_opens_network(
    tmp_path, monkeypatch, explicit_key
):
    import os
    import urllib.request

    from test_witness_strata import attest

    from kyb.witness import gemini
    from kyb.witness.errors import WitnessRefused
    from kyb.witness.strata import Config, RoutedWitness

    # A process-local dummy inherited value must not enable the prohibited route.
    monkeypatch.setenv("GEMINI_API_KEY", "offline-test-only")
    original_getitem = type(os.environ).__getitem__

    def guarded_getitem(environment, name):
        assert name != "GEMINI_API_KEY", "credentials must not be inspected"
        return original_getitem(environment, name)

    def forbidden_network(*args, **kwargs):
        pytest.fail("direct Gemini network is prohibited")

    monkeypatch.setattr(type(os.environ), "__getitem__", guarded_getitem)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden_network)
    attest(tmp_path, "gpt-6.1-sol")
    made = []

    def factory(backend):
        assert backend.startswith("gemini-api:"), "billing refusal cannot permit fallback"
        candidate = gemini.GeminiWitness(api_key=explicit_key)
        made.append(candidate)
        return candidate

    router = RoutedWitness(
        tmp_path,
        config=Config("A", ("gemini-api:gemini-3.1-pro-preview", "claude-p:claude-opus-5-5")),
        factory=factory,
    )
    for mode in ("auto", "replay"):
        router.mode = mode
        with pytest.raises(WitnessRefused):
            router.for_document(SUB)["A"].ask("transcribe", [b"crop"], SPEC)
    record = json.loads(next((tmp_path / "silver/witness/raw").glob("*/failed/0.json")).read_text())
    assert record["provenance"]["refusal_evidence"] == ["direct_api_billing_prohibited"]
    assert record["failure_type"] == "WitnessRefused" and made
    assert "offline-test-only" not in json.dumps(record)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_channels",
        "missing_stderr",
        "invalid_channels",
        "erased_diagnostics",
        "impossible_count",
        "conflicting_hash",
    ],
)
def test_cached_channel_evidence_cannot_omit_or_contradict_observed_stderr(
    tmp_path, monkeypatch, mutation
):
    router, calls, downstream = parser_route(
        tmp_path, monkeypatch, no_answer_stream(), 1, diagnostic="rate_limit"
    )
    router.for_document(SUB)["A"].ask("transcribe", [b"crop"], SPEC)
    path = next((tmp_path / "silver/witness/raw").glob("*/failed/0.json"))
    record = json.loads(path.read_text())
    prov = record["provenance"]
    if mutation == "missing_channels":
        prov.pop("channels")
    elif mutation == "missing_stderr":
        prov["channels"].pop("stderr")
    elif mutation == "invalid_channels":
        prov["channels"] = []
    elif mutation == "erased_diagnostics":
        prov["channels"]["stderr"]["diagnostics"] = []
    elif mutation == "impossible_count":
        prov["channels"]["stderr"]["nonblank_lines"] = 1000
    else:
        prov["channels"]["stderr"]["hash"] = "0" * 64
    rewrite(path, record)
    count = sum(len(d.calls) for d in downstream)
    from kyb.witness.errors import WitnessRefused

    for mode in ("auto", "replay"):
        router.mode = mode
        with pytest.raises(WitnessRefused):
            router.for_document(SUB)["A"].ask("transcribe", [b"crop"], SPEC)
    assert len(calls) == 1 and sum(len(d.calls) for d in downstream) == count
