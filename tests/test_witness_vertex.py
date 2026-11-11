"""Approved GCP Vertex transport: zero-network mocks only. Not live-route proof."""

from __future__ import annotations

import copy
import io
import json
import urllib.error
from pathlib import Path

import pytest
from test_witness_strata import DATA, SPEC, SUB, Transport, attest

from kyb.util.hashing import sha256_json
from kyb.witness import gemini, verify, vertex
from kyb.witness.errors import WitnessRefused, WitnessUnavailable
from kyb.witness.gemini import GeminiWitness
from kyb.witness.port import Witness
from kyb.witness.strata import Config, RoutedWitness, adapter
from kyb.witness.vertex import VertexWitness

TOKEN = "ya29.PRIVATE-TOKEN-SENTINEL-9876543210"
BACKEND = "vertex-gcp:gemini-3.1-pro-preview"


def envelope(reason="STOP", data=None, model="gemini-3.1-pro-preview"):
    return {
        "modelVersion": model,
        "candidates": [
            {
                "finishReason": reason,
                "content": {"parts": [{"text": json.dumps(DATA if data is None else data)}]},
            }
        ],
    }


class Io:
    """Mock token source and urlopen. Records every request; opens no socket."""

    def __init__(self, output=None, error=None, status=200):
        self.output, self.error, self.status = (
            envelope() if output is None else output,
            error,
            status,
        )
        self.requests, self.token_calls = [], 0

    def token(self):
        self.token_calls += 1
        return TOKEN

    def urlopen(self, req, timeout):
        self.requests.append((req, timeout))
        if self.error is not None:
            raise self.error
        raw = self.output if isinstance(self.output, bytes) else json.dumps(self.output).encode()
        response = io.BytesIO(raw)
        response.status = self.status
        return response

    def witness(self, **kwargs):
        kwargs.setdefault("model", "gemini-3.1-pro-preview")
        return VertexWitness(retries=0, token_source=self.token, urlopen=self.urlopen, **kwargs)


def http_error(code, status, message=None):
    message = message or {429: "Too many requests", 503: "Service unavailable"}.get(code, "x")
    body = {"error": {"code": code, "status": status, "message": message}}
    return urllib.error.HTTPError(
        "https://x", code, "private-sentinel", {}, io.BytesIO(json.dumps(body).encode())
    )


def ask(tmp_path, io_, mode="auto", sleep=None):
    if sleep is None:

        def sleep(x):
            pass

    return Witness(tmp_path, io_.witness(), mode=mode, sleep=sleep).ask(
        "transcribe", [b"crop"], SPEC
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"project": "kyb-core-ops"},
        {"project": "other"},
        {"location": "us-central1"},
        {"host": "evil.example.com"},
        {"host": "us-aiplatform.googleapis.com"},
        {"model": "gpt-5"},
        {"model": "gemini-x/../../y"},
    ],
)
def test_config_outside_approved_route_refuses_before_token_or_network(kwargs):
    io_ = Io()
    with pytest.raises(WitnessRefused):
        io_.witness(**kwargs)
    assert not io_.token_calls and not io_.requests


def test_factory_builds_exact_vertex_model_and_rejects_other_shapes():
    assert adapter(BACKEND).transport == "vertex-gcp"
    assert adapter(BACKEND).model == "gemini-3.1-pro-preview"
    with pytest.raises(WitnessRefused):
        adapter("vertex-gcp:claude-opus-5-5")
    assert adapter("gemini-api:gemini-3.1-pro-preview").offline_transport is None


def test_request_url_headers_billing_project_and_redacted_records(tmp_path, monkeypatch):
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS"):
        monkeypatch.setenv(name, "ENV-KEY-SENTINEL")
    io_ = Io()
    reply = ask(tmp_path, io_)
    ((req, timeout),) = io_.requests
    assert req.full_url == (
        "https://aiplatform.googleapis.com/v1/projects/kyb-vertex-project/locations/global/"
        "publishers/google/models/gemini-3.1-pro-preview:generateContent"
    )
    headers = {k.lower(): v for k, v in req.header_items()}
    assert headers["x-goog-user-project"] == "kyb-vertex-project"
    assert headers["authorization"] == f"Bearer {TOKEN}"
    assert "x-goog-api-key" not in headers and "key=" not in req.full_url
    assert json.loads(req.data)["contents"][0]["role"] == "user" and timeout == gemini.TIMEOUT_S
    assert reply.model == "gemini-3.1-pro-preview"
    assert reply.provenance["transport"] == "vertex-gcp"
    assert reply.provenance["route"]["project"] == reply.provenance["route"]["quota_project"]
    assert reply.provenance["route"]["project"] == "kyb-vertex-project"
    stored = "".join(p.read_text() for p in (tmp_path / "silver/witness").rglob("*.json"))
    assert TOKEN not in stored and "ENV-KEY-SENTINEL" not in stored


def test_gcloud_token_subprocess_has_no_shell_no_api_keys_and_redacts(monkeypatch):
    seen = {}

    def run(argv, **kwargs):
        seen.update(argv=argv, **kwargs)
        return type("R", (), {"returncode": 0, "stdout": TOKEN + "\n", "stderr": ""})()

    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/x")
    monkeypatch.setattr(vertex.shutil, "which", lambda _: "/usr/bin/gcloud")
    monkeypatch.setattr(vertex.subprocess, "run", run)
    assert vertex.adc_token() == TOKEN
    assert seen["argv"][1:] == ["auth", "application-default", "print-access-token"]
    assert not seen.get("shell") and seen["timeout"] == vertex.TOKEN_TIMEOUT_S
    assert not {"GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS"} & set(
        seen["env"]
    )
    monkeypatch.setattr(
        vertex.subprocess, "run",
        lambda *a, **k: type("R", (), {"returncode": 1, "stdout": TOKEN, "stderr": TOKEN})(),
    )  # fmt: skip
    with pytest.raises(WitnessRefused) as err:
        vertex.adc_token()
    assert TOKEN not in str(err.value)


@pytest.mark.parametrize("bad", ["", "short", "tok en with space", None])
def test_auth_failure_is_terminal_refusal_without_network_or_fallback(tmp_path, bad):
    io_ = Io()
    witness = io_.witness()
    witness._token_source = (lambda: bad) if bad != "boom" else None
    with pytest.raises(WitnessRefused):
        Witness(tmp_path, witness).ask("transcribe", [b"crop"], SPEC)
    assert not io_.requests
    record = json.loads(next((tmp_path / "silver/witness/raw").glob("*/failed/0.json")).read_text())
    assert record["failure_kind"] == "refused"
    assert "vertex_auth_failed" in record["provenance"]["refusal_evidence"]


def test_token_source_exception_text_never_reaches_record(tmp_path):
    io_ = Io()
    witness = io_.witness()

    def boom():
        raise RuntimeError(TOKEN)

    witness._token_source = boom
    with pytest.raises(WitnessRefused):
        Witness(tmp_path, witness).ask("transcribe", [b"crop"], SPEC)
    stored = "".join(p.read_text() for p in (tmp_path / "silver/witness").rglob("*.json"))
    assert TOKEN not in stored


def test_transport_and_route_bind_cache_identity_and_cross_route_replay_is_refused(tmp_path):
    vertex_a = Io().witness()
    direct = GeminiWitness(model="gemini-3.1-pro-preview", retries=0,
                           offline_transport=lambda *a, **k: io.BytesIO(b"{}"))  # fmt: skip
    keys = {Witness(tmp_path, w)._request("transcribe", [b"crop"], SPEC)[0]
            for w in (vertex_a, direct)}  # fmt: skip
    assert len(keys) == 2
    ask(tmp_path, Io())
    path = next((tmp_path / "silver/witness/raw").glob("*/0.json"))
    for field, value in (("project", "other-project"), ("location", "us-central1"),
                         ("host", "other.googleapis.com")):  # fmt: skip
        record = json.loads(path.read_text())
        original = copy.deepcopy(record)
        record["provenance"]["route"][field] = value
        record.pop("content_hash")
        record["content_hash"] = sha256_json(record)
        path.write_text(json.dumps(record))
        with pytest.raises(WitnessRefused):
            ask(tmp_path, Io(), mode="replay")
        path.write_text(json.dumps(original))
    # A legacy gemini-api record has no route and cannot be replayed as Vertex.
    record = json.loads(path.read_text())
    record["provenance"].pop("route")
    record["provenance"]["transport"] = "gemini-api"
    record.pop("content_hash")
    record["content_hash"] = sha256_json(record)
    path.write_text(json.dumps(record))
    with pytest.raises(WitnessRefused):
        ask(tmp_path, Io(), mode="replay")


def test_replay_of_valid_vertex_record_makes_no_token_or_network_call(tmp_path):
    ask(tmp_path, Io())
    io_ = Io()
    reply = ask(tmp_path, io_, mode="replay")
    assert reply.replayed and not io_.token_calls and not io_.requests


@pytest.mark.parametrize(
    "code,status",
    [(400, "INVALID_ARGUMENT"), (401, "UNAUTHENTICATED"), (403, "PERMISSION_DENIED"),
     (404, "NOT_FOUND")],
)  # fmt: skip
def test_http_auth_config_and_unknown_errors_refuse_not_unavailable(tmp_path, code, status):
    with pytest.raises(WitnessRefused):
        ask(tmp_path, Io(error=http_error(code, status)))


@pytest.mark.parametrize("code,status", [(429, "RESOURCE_EXHAUSTED"), (503, "UNAVAILABLE")])
def test_recognized_transient_http_is_unavailable(tmp_path, code, status):
    with pytest.raises(WitnessUnavailable):
        ask(tmp_path, Io(error=http_error(code, status)))


def test_transient_status_with_refusal_text_refuses_and_hides_diagnostic(tmp_path):
    with pytest.raises(WitnessRefused):
        ask(tmp_path, Io(error=http_error(503, "UNAVAILABLE", "permission private-sentinel")))
    raw = next((tmp_path / "silver/witness/raw").glob("*/failed/0.json")).read_text()
    assert "private-sentinel" not in raw


@pytest.mark.parametrize("reason", ["SAFETY", "MAX_TOKENS", "FUTURE_REASON"])
def test_non_stop_refuses_over_availability(tmp_path, reason):
    with pytest.raises(WitnessRefused):
        ask(tmp_path, Io(envelope(reason)))


@pytest.mark.parametrize(
    "payload",
    [envelope(data={"crops": "x"}), envelope(data=[1]), envelope(data={"crops": [{"crop_id": 1}]}),
     b"{not json", b"[]", envelope(model="claude-opus-5-5"), {**envelope(), "modelVersion": None},
     {k: v for k, v in envelope().items() if k != "modelVersion"}],
)  # fmt: skip
def test_malformed_non_scalar_or_wrong_model_payload_refuses(tmp_path, payload):
    with pytest.raises(WitnessRefused):
        ask(tmp_path, Io(payload))


def test_actual_model_is_required_from_the_response_not_the_request(tmp_path):
    reply = ask(tmp_path, Io(envelope(model="gemini-3.1-pro-preview-001")))
    assert reply.model == "gemini-3.1-pro-preview-001"
    assert reply.provenance["reported_model"] == reply.model


def test_final_hash_is_shared_and_tampering_refuses_replay_after_rechecksum(tmp_path):
    ask(tmp_path, Io())
    path = next((tmp_path / "silver/witness/raw").glob("*/0.json"))
    record = json.loads(path.read_text())
    assert record["provenance"]["final_output_hash"] == sha256_json(record["data"])
    for edit in ("data", "final_output_hash"):
        tampered = copy.deepcopy(record)
        if edit == "data":
            tampered["data"]["crops"][0]["transcription"] = "Irene 99%"
        else:
            tampered["provenance"]["final_output_hash"] = sha256_json({"x": 1})
        tampered.pop("content_hash")
        tampered["content_hash"] = sha256_json(tampered)
        path.write_text(json.dumps(tampered))
        with pytest.raises(WitnessRefused):
            ask(tmp_path, Io(), mode="replay")


def test_redirects_and_proxies_are_disabled():
    request = vertex.urllib.request.Request("https://aiplatform.googleapis.com/")
    assert vertex._NoRedirect().redirect_request(request, None, 302, "", {}, "https://x") is None


def test_router_vertex_refusal_never_falls_back_but_transient_does(tmp_path):
    downstream = []

    def route(io_, root):
        attest(root, "gpt-6.1-sol")

        def factory(backend):
            if backend.startswith("vertex-gcp:"):
                return io_.witness()
            downstream.append(Transport(backend))
            return downstream[-1]

        return RoutedWitness(root, config=Config("A", (BACKEND, "claude-p:claude-opus-5-5")),
                             factory=factory).for_document(SUB)["A"]  # fmt: skip

    with pytest.raises(WitnessRefused):
        route(Io(error=http_error(403, "PERMISSION_DENIED")), tmp_path / "a").ask(
            "transcribe", [b"crop"], SPEC
        )
    assert not downstream
    reply = route(Io(error=http_error(503, "UNAVAILABLE")), tmp_path / "b").ask(
        "transcribe", [b"crop"], SPEC
    )
    assert reply.model == "claude-opus-5-5" and len(downstream) == 1


def test_router_default_routes_vertex_and_google_reader_b_maps_to_vertex(tmp_path):
    from kyb.witness.strata import DEFAULT_A

    assert DEFAULT_A[0] == BACKEND and not any(b.startswith("gemini-api") for b in DEFAULT_A)
    attest(tmp_path, "gemini-3.1-pro-preview")
    made = []

    def factory(backend):
        made.append(backend)
        return Transport(backend)

    sub = {**SUB, "produced_by": {"reader": {**SUB["produced_by"]["reader"]}}}
    route = RoutedWitness(tmp_path, config=Config("B"), factory=factory).for_document(sub)
    assert route["B"].backends == (BACKEND,)


def _pdf_setup(tmp_path):
    import shutil

    root = Path(verify.__file__).resolve().parents[3]
    assets = root / "examples"
    source = root / "tests/fixtures/witness_b05"
    for part in ("claims", "layout", "pages"):
        shutil.copytree(source / "silver" / part, tmp_path / "silver" / part)
    path = tmp_path / "silver/claims/es-iberia-software/r1/05.json"
    sub = json.loads(path.read_text())
    sub["claims"] = [c for c in sub["claims"] if c["claim_id"] == "c-B05-08"]
    sub["produced_by"] = SUB["produced_by"]
    path.write_text(json.dumps(sub))
    attest(tmp_path)
    data = copy.deepcopy(DATA)
    data["crops"][0]["transcription"] = "Lucas Navarro Solís 68%"
    data["crops"][0]["reread"]["values"][0]["subject"] = "Lucas Navarro Solís"
    data["crops"][0]["reread"]["values"][0]["value"] = "68%"
    data["crops"].append({"crop_id": "k2", "transcription": "pendiente", "readable": True,
                          "reread": {"values": []}})  # fmt: skip
    return data, next((assets / "packets/es-iberia-software").glob("05*.pdf"))


def test_actual_pdf_vertex_stop_positive_and_blocked_negative(tmp_path):
    data, pdf = _pdf_setup(tmp_path)

    def run(output):
        io_ = Io(output)
        router = RoutedWitness(tmp_path, config=Config("A", (BACKEND,)),
                               factory=lambda backend: io_.witness())  # fmt: skip
        return verify.verify_document(tmp_path, "es-iberia-software", "r1", "05", router, pdf), io_

    rec, io_ = run(envelope(data=data))
    assert (
        io_.requests and rec["results"][0]["verified"] and rec["model"] == "gemini-3.1-pro-preview"
    )
    assert rec["witness_family"] == "google"
    other = tmp_path / "blocked"
    other.mkdir()
    _pdf_setup(other)
    io_ = Io(envelope("SAFETY", data=data))
    router = RoutedWitness(other, config=Config("A", (BACKEND,)),
                           factory=lambda backend: io_.witness())  # fmt: skip
    blocked = verify.verify_document(other, "es-iberia-software", "r1", "05", router, pdf)
    assert not blocked["results"][0]["verified"] and blocked["model"] is None
    assert blocked["witness_family"] == "unknown"
    assert io_.requests


def test_transient_failure_bounded_retries_auto_mode(tmp_path):
    io_ = Io(error=http_error(429, "RESOURCE_EXHAUSTED"))
    with pytest.raises(WitnessUnavailable):
        ask(tmp_path, io_)
    raw_dir = tmp_path / "silver/witness/raw"
    folder = next(raw_dir.iterdir())
    assert (folder / "failed/0.json").exists()
    assert len(io_.requests) == 1

    io_.error = http_error(429, "RESOURCE_EXHAUSTED")
    with pytest.raises(WitnessUnavailable):
        ask(tmp_path, io_)
    assert (folder / "failed/1.json").exists()
    assert len(io_.requests) == 2

    io_.error = http_error(429, "RESOURCE_EXHAUSTED")
    with pytest.raises(WitnessUnavailable):
        ask(tmp_path, io_)
    assert (folder / "failed/2.json").exists()
    assert len(io_.requests) == 3

    io_.error = http_error(429, "RESOURCE_EXHAUSTED")
    with pytest.raises(WitnessUnavailable) as exc_info:
        ask(tmp_path, io_)
    assert "recorded attempt failed" in str(exc_info.value)
    assert len(io_.requests) == 3


def test_terminal_failure_not_retried_auto_mode(tmp_path):
    io_ = Io(error=http_error(403, "PERMISSION_DENIED"))
    with pytest.raises(WitnessRefused):
        ask(tmp_path, io_)
    raw_dir = tmp_path / "silver/witness/raw"
    folder = next(raw_dir.iterdir())
    assert (folder / "failed/0.json").exists()
    assert len(io_.requests) == 1

    io_.error = http_error(403, "PERMISSION_DENIED")
    with pytest.raises(WitnessRefused) as exc_info:
        ask(tmp_path, io_)
    assert "recorded attempt failed; explicit live retry required" in str(exc_info.value)
    assert len(io_.requests) == 1


def test_tampered_record_raises_integrity_error_and_no_live_call(tmp_path):
    # 1. Create a failed recording first
    io_ = Io(error=http_error(429, "RESOURCE_EXHAUSTED"))
    with pytest.raises(WitnessUnavailable):
        ask(tmp_path, io_)

    raw_dir = tmp_path / "silver/witness/raw"
    folder = next(raw_dir.iterdir())
    failed_file = folder / "failed/0.json"
    assert failed_file.exists()
    assert len(io_.requests) == 1

    # 2. Tamper with the failed recording content
    record = json.loads(failed_file.read_text())
    record["some_tampered_field"] = "evil_tampering"
    # Write back without updating content_hash, making content_hash invalid!
    failed_file.write_text(json.dumps(record, indent=2))

    # 3. Call ask again. Should raise integrity error and NOT make a second live call!
    io_.error = http_error(429, "RESOURCE_EXHAUSTED")
    with pytest.raises(WitnessRefused) as exc_info:
        ask(tmp_path, io_)
    assert "witness failure recording integrity mismatch" in str(exc_info.value)
    assert len(io_.requests) == 1  # No additional live call!


def test_request_key_mismatch_raises_integrity_error_and_no_live_call(tmp_path):
    io_ = Io(error=http_error(429, "RESOURCE_EXHAUSTED"))
    with pytest.raises(WitnessUnavailable):
        ask(tmp_path, io_)

    raw_dir = tmp_path / "silver/witness/raw"
    folder = next(raw_dir.iterdir())
    failed_file = folder / "failed/0.json"
    assert failed_file.exists()
    assert len(io_.requests) == 1

    # Tamper request key
    record = json.loads(failed_file.read_text())
    record["request"] = "different_request_key"
    # Recalculate content_hash so it's technically valid hash, but wrong request key!
    record.pop("content_hash", None)
    record["content_hash"] = sha256_json(record)
    failed_file.write_text(json.dumps(record, indent=2))

    io_.error = http_error(429, "RESOURCE_EXHAUSTED")
    with pytest.raises(WitnessRefused) as exc_info:
        ask(tmp_path, io_)
    assert "witness failure recording integrity mismatch" in str(exc_info.value)
    assert len(io_.requests) == 1


def test_bounded_backoff_and_retry_after(tmp_path):
    sleep_calls = []

    def spy_sleep(seconds):
        sleep_calls.append(seconds)

    # 1. Create a transient failure to start the chain
    io_ = Io(error=http_error(429, "RESOURCE_EXHAUSTED"))
    with pytest.raises(WitnessUnavailable):
        ask(tmp_path, io_, sleep=spy_sleep)

    raw_dir = tmp_path / "silver/witness/raw"
    folder = next(raw_dir.iterdir())
    assert (folder / "failed/0.json").exists()
    assert len(io_.requests) == 1
    assert sleep_calls == []  # No sleep on initial failure

    # 2. First retry (failed/0.json exists, failed_count is 1)
    # n = 0, delay = 15s * 2^0 = 15s
    io_.error = http_error(429, "RESOURCE_EXHAUSTED")
    with pytest.raises(WitnessUnavailable):
        ask(tmp_path, io_, sleep=spy_sleep)
    assert (folder / "failed/1.json").exists()
    assert len(io_.requests) == 2
    assert sleep_calls == [15.0]

    # 3. Second retry (failed/0.json and failed/1.json exist, failed_count is 2)
    # n = 1, delay = 15s * 2^1 = 30s
    io_.error = http_error(429, "RESOURCE_EXHAUSTED")
    with pytest.raises(WitnessUnavailable):
        ask(tmp_path, io_, sleep=spy_sleep)
    assert (folder / "failed/2.json").exists()
    assert len(io_.requests) == 3
    assert sleep_calls == [15.0, 30.0]

    # 4. Now let's test Retry-After. We can manually insert a retry_after into failed/0.json
    import shutil

    shutil.rmtree(tmp_path / "silver")

    sleep_calls.clear()
    io_ = Io(error=http_error(429, "RESOURCE_EXHAUSTED"))
    with pytest.raises(WitnessUnavailable):
        ask(tmp_path, io_, sleep=spy_sleep)

    # Let's write retry_after to failed/0.json's provenance
    raw_dir = tmp_path / "silver/witness/raw"
    folder = next(raw_dir.iterdir())
    failed_file = folder / "failed/0.json"
    record = json.loads(failed_file.read_text())
    record["provenance"]["retry_after"] = 3600.0
    record.pop("content_hash", None)
    record["content_hash"] = sha256_json(record)
    failed_file.write_text(json.dumps(record, indent=2))

    # the second run honours Retry-After, bounded at 60 s
    io_.error = http_error(429, "RESOURCE_EXHAUSTED")
    with pytest.raises(WitnessUnavailable):
        ask(tmp_path, io_, sleep=spy_sleep)
    assert sleep_calls == [60.0]  # a recorded Retry-After is capped at 60 s


@pytest.mark.parametrize(
    "message",
    [
        "Resource has been exhausted (e.g. check quota).",
        "Resource exhausted. Please try again later. Please refer to the error-code-429 page.",
        "Quota exceeded for aiplatform.googleapis.com/generate_content_requests_per_minute.",
    ],
)
def test_a_real_vertex_429_is_retried_whatever_its_message(tmp_path, message):
    """Live Vertex run: a 429 RESOURCE_EXHAUSTED whose message matched no known phrase was
    classified unknown, so terminal. The RPC status alone decides (issue R4)."""
    sleeps = []
    io_ = Io(error=http_error(429, "RESOURCE_EXHAUSTED", message))
    with pytest.raises(WitnessUnavailable):
        ask(tmp_path, io_, sleep=sleeps.append)
    io_.error = http_error(429, "RESOURCE_EXHAUSTED", message)
    with pytest.raises(WitnessUnavailable):
        ask(tmp_path, io_, sleep=sleeps.append)
    assert len(io_.requests) == 2 and sleeps == [15.0]


def test_a_permission_error_stays_terminal_even_with_a_retryable_message(tmp_path):
    io_ = Io(error=http_error(403, "PERMISSION_DENIED", "Service unavailable; quota exceeded"))
    with pytest.raises(WitnessRefused):
        ask(tmp_path, io_)
    io_.error = http_error(403, "PERMISSION_DENIED", "Service unavailable; quota exceeded")
    with pytest.raises(WitnessRefused):
        ask(tmp_path, io_)
    assert len(io_.requests) == 1


def test_vertex_reads_project_quota_project_and_location_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("KYB_VERTEX_PROJECT", "custom-vertex-proj")
    monkeypatch.setenv("KYB_VERTEX_QUOTA_PROJECT", "custom-billing-proj")
    monkeypatch.setenv("KYB_VERTEX_LOCATION", "us-central1")
    io_ = Io()
    reply = ask(tmp_path, io_)
    ((req, _),) = io_.requests
    assert req.full_url == (
        "https://aiplatform.googleapis.com/v1/projects/custom-vertex-proj/locations/us-central1/"
        "publishers/google/models/gemini-3.1-pro-preview:generateContent"
    )
    headers = {k.lower(): v for k, v in req.header_items()}
    assert headers["x-goog-user-project"] == "custom-billing-proj"
    assert reply.provenance["route"]["project"] == "custom-vertex-proj"
    assert reply.provenance["route"]["quota_project"] == "custom-billing-proj"
    assert reply.provenance["route"]["location"] == "us-central1"
