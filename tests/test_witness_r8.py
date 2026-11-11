"""R8 (issue): envelope anomalies. Unknown shapes are observed and recorded, never verify a
claim, never become silent availability; self-heal is bounded and replayable. Offline only."""

from __future__ import annotations

import copy
import io
import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_witness_outcomes import gemini_route
from test_witness_r7 import (
    both_modes_anomalous,
    both_modes_refuse_claude,
    both_modes_refuse_gemini,
)
from test_witness_strata import DATA, SPEC, SUB, no_answer_stream, parser_route
from test_witness_vertex import BACKEND, Io, _pdf_setup, envelope

from kyb.witness import adapters, gemini, verify
from kyb.witness.claude_stream import ClaudeStream
from kyb.witness.envelope import error_like_key
from kyb.witness.errors import WitnessAnomalous
from kyb.witness.port import Witness
from kyb.witness.strata import Config, RoutedWitness

ROOT = Path(__file__).parents[1]
PROBE = ROOT / "tests/fixtures/claude_cli/route-probe.jsonl"
FIX = Path(__file__).parent / "fixtures" / "claude_cli"
SCHEMA = json.loads((FIX / "schema.json").read_text())
SEEDS = range(24)
ZW = "​"


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def stream(events):
    return "\n".join(map(json.dumps, events))


def failed_anomalies(root):
    out = []
    for path in sorted((root / "silver/witness/raw").glob("*/failed/*.json")):
        out += json.loads(path.read_text())["provenance"].get("envelope_anomalies") or []
    return out


def real_positive(data=None):
    """A Gemini/Vertex positive carrying every field the recorded real responses carry."""
    out = envelope(data=data)
    out["candidates"][0] |= {"index": 0}
    out["candidates"][0]["content"]["role"] = "model"
    out["responseId"] = "V0HgapjrNc-4wbUPpvDlgAQ"
    out["usageMetadata"] = {
        "promptTokenCount": 1114, "candidatesTokenCount": 111, "totalTokenCount": 1225,
        "thoughtsTokenCount": 20,
        "promptTokensDetails": [{"modality": "IMAGE", "tokenCount": 972},
                                {"modality": "TEXT", "tokenCount": 142}],
        "serviceTier": "standard",
    }  # fmt: skip
    return out


def positive_claude():
    events = [
        {"type": "system", "subtype": "init", "tools": ["StructuredOutput"]},
        {"type": "assistant", "message": {"model": "claude-opus-5-5", "content": [
            {"type": "tool_use", "id": "t1", "name": "StructuredOutput", "input": DATA}]}},
        {"type": "user", "message": {"role": "user", "content": [
            {"tool_use_id": "t1", "type": "tool_result", "content": "ok"}]}},
        {"type": "result", "subtype": "success", "is_error": False,
         "structured_output": copy.deepcopy(DATA)},
    ]  # fmt: skip
    return events


def inject_points(node, path=()):
    """Every dict a provider could grow a new key in (not opaque/open-keyed maps)."""
    skip = {"input", "wire_tool_inputs", "structured_output", "modelUsage", "by_type",
            "unifiedWindows", "model_usage_since_last_result"}  # fmt: skip
    if isinstance(node, list):
        for i, item in enumerate(node):
            yield from inject_points(item, (*path, i))
    elif isinstance(node, dict):
        yield path, node
        for k, v in node.items():
            if k not in skip:
                yield from inject_points(v, (*path, k))


def random_key(rng):
    while True:
        key = "".join(rng.choice("abcdefghjklmnopqrstuvwxyz") for _ in range(rng.randint(4, 9)))
        if not error_like_key(key):
            return key + rng.choice(["", "X", "_v2"])


def random_value(rng):
    return rng.choice([1, "novel", True, [1, 2], {"a": {"b": 1}}, None, 0.5])


def tag(frame):
    frame["__marker"] = True
    return frame


# 1. R8 repros: Gemini/Vertex error shapes at the root and in usageMetadata -----------------

GEMINI_ERROR_SHAPES = [
    {"fault": "boom"}, {"failure": {"code": 3}}, {"exception": "x"}, {"err": "boom"},
    {"fatal_error": "x"}, {"status": "ERROR"}, {"status": "FAILED"},
    {ZW + "fault": "boom"}, {" failure ": "boom"}, {"Err​or": "boom"}, {"ＥＲＲＯＲ": "x"},
]  # fmt: skip


@pytest.mark.parametrize("where", ["root", "usageMetadata"])
@pytest.mark.parametrize("extra", GEMINI_ERROR_SHAPES)
def test_gemini_error_shapes_are_error_anomalies_and_never_complete(
    tmp_path, monkeypatch, where, extra
):
    out = real_positive()
    (out if where == "root" else out["usageMetadata"]).update(extra)
    both_modes_refuse_gemini(tmp_path, monkeypatch, out)
    found = failed_anomalies(tmp_path)
    assert found and all(a["severity"] == "error" for a in found)
    assert found[0]["path"].startswith("$") and found[0]["provider"] == "gemini-api"
    assert found[0]["requested_model"] == "gemini-3.1-pro-preview"
    assert found[0]["served_model"] == "gemini-3.1-pro-preview"
    assert "boom" not in json.dumps(found)  # diagnostic text is hashed, never stored


def test_gemini_non_stop_finish_reason_is_an_error_anomaly(tmp_path, monkeypatch):
    both_modes_refuse_gemini(tmp_path, monkeypatch, envelope("MAX_TOKENS"))
    assert any(a["key"] == "finishReason" and a["severity"] == "error"
               for a in failed_anomalies(tmp_path))  # fmt: skip


# 2. R8 repros: Claude init containers, assistant metadata, user frame --------------------------

CONTAINER_ERRORS = [
    {"fault": "x"}, {"failure": "x"}, {"exception": "x"}, {"status": "ERROR"},
    {"status": "FAILED"}, {"err" + ZW + "or": "x"}, {" error": "x"}, {"Fatal_Error": "x"},
]  # fmt: skip


@pytest.mark.parametrize(
    "holder", ["plugins", "agents", "capabilities", "skills", "slash_commands"]
)
@pytest.mark.parametrize("shape", CONTAINER_ERRORS)
@pytest.mark.parametrize("base", ["no_answer", "positive"])
def test_claude_init_container_error_shapes_refuse_and_never_fall_back(
    tmp_path, monkeypatch, holder, shape, base
):
    events = (
        [json.loads(x) for x in no_answer_stream().splitlines()]
        if base == "no_answer"
        else positive_claude()
    )
    events[0][holder] = [shape]
    both_modes_refuse_claude(tmp_path, monkeypatch, stream(events))
    found = failed_anomalies(tmp_path)
    assert any(a["severity"] == "error" and a["path"].startswith(f"init.{holder}[0]")
               for a in found)  # fmt: skip


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("diagnostics", [{"error": "x"}]), ("container", {"status": "FAILED"}),
        ("context_management", {"fault": 1}), ("input_transformations", [{"failure": "x"}]),
        ("stop_details", {"status": "ERROR"}),
    ],
)  # fmt: skip
def test_claude_assistant_metadata_error_shapes_refuse(tmp_path, monkeypatch, key, value):
    events = positive_claude()
    events[1]["message"][key] = value
    both_modes_refuse_claude(tmp_path, monkeypatch, stream(events), 0)
    assert any(a["path"] == f"assistant.message.{key}" and a["severity"] == "error"
               for a in failed_anomalies(tmp_path))  # fmt: skip


@pytest.mark.parametrize("where", ["user_message", "tool_result", "user_frame"])
def test_claude_user_frame_unknown_key_is_an_anomaly_and_error_key_refuses(
    tmp_path, monkeypatch, where
):
    for key, benign in (("novelThing", True), ("fault", False)):
        events = positive_claude()
        holder = {"user_message": events[2]["message"],
                  "tool_result": events[2]["message"]["content"][0],
                  "user_frame": events[2]}[where]  # fmt: skip
        holder[key] = "x"
        root = tmp_path / key
        if benign:
            router, calls, downstream = parser_route(root, monkeypatch, stream(events), 0)
            both_modes_anomalous(router, calls, downstream, claude_calls=2)
            assert failed_anomalies(root)[0]["key"] == "novelThing"
        else:
            both_modes_refuse_claude(root, monkeypatch, stream(events), 0)


# 3. Unknown benign key on a real positive: anomaly, no verify, self-heal, replay ---------------


def test_unknown_usage_key_is_unknown_severity_and_not_verified_on_that_attempt(tmp_path):
    out = real_positive()
    out["usageMetadata"]["brandNewMeter"] = 7
    with pytest.raises(WitnessAnomalous) as caught:
        Witness(tmp_path, Io(out).witness()).ask("transcribe", [b"crop"], SPEC)
    [anomaly] = caught.value.anomalies
    assert anomaly["path"] == "$.usageMetadata.brandNewMeter" and anomaly["severity"] == "unknown"
    assert anomaly["key"] == "brandNewMeter" and anomaly["value"] == "7"
    assert anomaly["frame"] == "response" and anomaly["provider"] == "vertex-gcp"
    assert anomaly["served_model"] == "gemini-3.1-pro-preview"
    assert anomaly["requested_model"] == "gemini-3.1-pro-preview"
    assert not list((tmp_path / "silver/witness/raw").glob("*/0.json"))  # nothing completed
    with pytest.raises(WitnessAnomalous):  # recorded anomalous outcome replays without a call
        Witness(tmp_path, Io(out).witness(), mode="replay").ask("transcribe", [b"crop"], SPEC)


def test_value_is_truncated_to_200_chars(tmp_path):
    out = real_positive()
    out["usageMetadata"]["brandNewMeter"] = "v" * 900
    with pytest.raises(WitnessAnomalous) as caught:
        Witness(tmp_path, Io(out).witness()).ask("transcribe", [b"crop"], SPEC)
    assert len(caught.value.anomalies[0]["value"]) == 200


class Sequence(Io):
    """Mock Vertex that serves scripted responses in order and counts calls."""

    def __init__(self, outputs):
        super().__init__(outputs[0])
        self.outputs = list(outputs)

    def urlopen(self, req, timeout):
        self.output = self.outputs[min(len(self.requests), len(self.outputs) - 1)]
        return super().urlopen(req, timeout)


def pdf_data():
    from test_witness_vertex import DATA as D

    data = copy.deepcopy(D)
    data["crops"][0]["transcription"] = "Lucas Navarro Solís 68%"
    data["crops"][0]["reread"]["values"][0]["subject"] = "Lucas Navarro Solís"
    data["crops"][0]["reread"]["values"][0]["value"] = "68%"
    data["crops"].append({"crop_id": "k2", "transcription": "pendiente", "readable": True,
                          "reread": {"values": []}})  # fmt: skip
    return data


def pdf_router(root, io_):
    return RoutedWitness(
        root, config=Config("A", (BACKEND,)), factory=lambda backend: io_.witness()
    )


def test_self_heal_verifies_only_the_clean_retry_and_replays_with_zero_calls(tmp_path):
    _, pdf = _pdf_setup(tmp_path)
    anomalous = real_positive(pdf_data())
    anomalous["usageMetadata"]["brandNewMeter"] = 7
    io_ = Sequence([anomalous, real_positive(pdf_data())])
    rec = verify.verify_document(
        tmp_path, "es-iberia-software", "r1", "05", pdf_router(tmp_path, io_), pdf
    )
    assert rec["results"][0]["verified"] and len(io_.requests) == 2
    assert [a["status"] for a in rec["attempts"]] == ["anomalous", "complete"]
    assert [a["attempt"] for a in rec["attempts"]] == [0, 1]
    assert rec["attempts"][0]["anomalies"][0]["key"] == "brandNewMeter"
    assert verify.anomaly_summary(rec) == {"count": 1, "severities": {"unknown": 1}}
    # the record is persisted exactly like verify_case does, then read by every consumer
    target = tmp_path / "silver/witness/es-iberia-software/r1"
    target.mkdir(parents=True)
    (target / "05.json").write_text(json.dumps(rec))
    assert verify.assurance_now(tmp_path, "es-iberia-software", "r1", "05")["anomalies"] == {
        "count": 1, "severities": {"unknown": 1},
    }  # fmt: skip
    replay_io = Io(error=AssertionError("replay must not call"))  # zero calls allowed
    again_router = pdf_router(tmp_path, replay_io)
    again_router.mode = "replay"
    again = verify.verify_document(tmp_path, "es-iberia-software", "r1", "05", again_router, pdf)
    assert not replay_io.requests and again["results"] == rec["results"]
    assert again["attempts"] == rec["attempts"]


def test_persistent_anomaly_keeps_every_claim_claimed_with_anomalies_attached(tmp_path):
    _, pdf = _pdf_setup(tmp_path)
    anomalous = real_positive(pdf_data())
    anomalous["brandNewRoot"] = {"a": 1}
    io_ = Sequence([anomalous])
    rec = verify.verify_document(
        tmp_path, "es-iberia-software", "r1", "05", pdf_router(tmp_path, io_), pdf
    )
    assert not any(r["verified"] for r in rec["results"]) and rec["model"] is None
    assert len(io_.requests) == 2  # one call plus exactly one retry, no loop
    assert [a["status"] for a in rec["attempts"]] == ["anomalous", "anomalous"]
    assert any(f["kind"] == "witness_envelope_anomalies" for f in rec["findings"])
    assert rec["envelope_anomalous"] and rec["assurance"] != "W2"
    assert verify.anomaly_summary(rec)["count"] == 2


def test_bounded_chain_is_two_attempts_then_one_on_the_other_backend(tmp_path, monkeypatch):
    from test_witness_strata import attest

    attest(tmp_path, "gpt-6.1-sol")
    out = real_positive()
    out["brandNewRoot"] = 1
    calls = []

    def factory(backend):
        return gemini.GeminiWitness(
            model=backend.split(":")[1], retries=0,
            offline_transport=lambda *a, **k: calls.append(backend) or io.BytesIO(
                json.dumps(out).encode()),
        )  # fmt: skip

    backends = ("gemini-api:gemini-3.1-pro-preview", "gemini-api:gemini-3.8-flash",
                "gemini-api:gemini-3.8-pro")  # fmt: skip
    router = RoutedWitness(tmp_path, config=Config("A", backends), factory=factory)
    route = router.for_document(SUB)["A"]
    with pytest.raises(WitnessAnomalous):
        route.ask("transcribe", [b"crop"], SPEC)
    assert len(calls) == 3  # same backend x2, then ONE attempt on the next; the 3rd is not tried
    assert [(a["backend"], a["attempt"]) for a in route.attempts] == [
        (backends[0], 0), (backends[0], 1), (backends[1], 0)]  # fmt: skip


# 4. Generative: a random unknown key at a random depth always yields an anomaly --------------


@pytest.mark.parametrize("seed", SEEDS)
def test_generated_unknown_key_in_gemini_envelope_is_recorded_in_auto_and_replay(
    tmp_path, monkeypatch, seed
):
    rng = random.Random(seed)
    out = real_positive()
    path, holder = rng.choice(list(inject_points(out)))
    key = random_key(rng)
    holder[key] = random_value(rng)
    router, calls, downstream = gemini_route(tmp_path, monkeypatch, out)
    both_modes_anomalous(router, calls, downstream, claude_calls=2)
    first = failed_anomalies(tmp_path)[0]
    assert first["key"] == key and first["severity"] == "unknown"
    assert first["path"].startswith("$") and first["path"].endswith(key)


@pytest.mark.parametrize("seed", SEEDS)
def test_generated_unknown_key_in_claude_stream_is_recorded_in_auto_and_replay(
    tmp_path, monkeypatch, seed
):
    rng = random.Random(seed)
    events = positive_claude()
    events.insert(1, {"type": "rate_limit_event", "uuid": "u", "session_id": "s",
                      "rate_limit_info": {"status": "allowed", "unifiedWindows": {
                          "five_hour": {"status": "allowed"}}}})  # fmt: skip
    events[0] |= {"plugins": [{"name": "n", "path": "builtin", "source": "s"}], "agents": ["a"]}
    events[-1]["usage"] = {"input_tokens": 1, "cache_creation": {"ephemeral_1h_input_tokens": 0}}
    path, holder = rng.choice(list(inject_points(events)))
    key = random_key(rng)
    holder[key] = random_value(rng)
    router, calls, downstream = parser_route(tmp_path, monkeypatch, stream(events), 0)
    both_modes_anomalous(router, calls, downstream, claude_calls=2)
    assert failed_anomalies(tmp_path)[0]["key"] == key


@pytest.mark.parametrize("seed", SEEDS)
def test_generated_unknown_key_in_probe_never_completes(monkeypatch, seed):
    rng = random.Random(seed)
    events = lines(PROBE)
    path, holder = rng.choice(list(inject_points(events)))
    key = random_key(rng)
    holder[key] = random_value(rng)
    monkeypatch.setattr(
        adapters.subprocess, "run",
        lambda cmd, **kw: SimpleNamespace(returncode=0, stdout=stream(events).encode(), stderr=b""),
    )  # fmt: skip
    w = adapters.ClaudeWitness("claude-haiku-4-5")
    with pytest.raises(WitnessAnomalous) as caught:
        w.run("p", [b"png"], SCHEMA)
    assert any(a["key"] == key for a in caught.value.anomalies)
    assert w.last_provenance["classification"] == "anomalous"
    assert w.last_provenance["normal_terminal"] is False


# 5. Controls: every real recording completes with ZERO anomalies; transients fall back --------


def test_probe_completes_with_zero_anomalies(monkeypatch):
    monkeypatch.setattr(
        adapters.subprocess, "run",
        lambda cmd, **kw: SimpleNamespace(
            returncode=0, stdout=stream(lines(PROBE)).encode(), stderr=b""),
    )  # fmt: skip
    w = adapters.ClaudeWitness("claude-haiku-4-5")
    data, _ = w.run("p", [b"png"], SCHEMA)
    assert data["canary_contents"] == "UNREADABLE"
    assert w.last_provenance["envelope_anomalies"] == []


# p4: Claude Code adds surpassedThreshold per usage window (it was only top-level);
# unknown, it refused every B witness for a new user, so 0/52 claims verified
# p5: Claude Code 2.1.314 opens with a system/ui_invalidate frame and splits
# modelUsage cache creation per TTL; unknown, every B witness refused (0/57 on packet A)
P5 = FIX / "p5.ui-invalidate.stream.jsonl"


@pytest.mark.parametrize("path", [PROBE, FIX / "p2.stream.jsonl", FIX / "p3.restricted.jsonl",
                                  FIX / "p4.window-threshold.stream.jsonl", P5])  # fmt: skip
def test_recorded_claude_streams_have_zero_anomalies(path):
    s = ClaudeStream({}, lambda v, sch: None, True)
    s.consume(path.read_bytes())
    assert s.anomalies.items == [], s.anomalies.items


def test_p5_recording_completes_through_the_witness_route(monkeypatch):
    monkeypatch.setattr(
        adapters.subprocess, "run",
        lambda cmd, **kw: SimpleNamespace(returncode=0, stdout=P5.read_bytes(), stderr=b""),
    )  # fmt: skip
    w = adapters.ClaudeWitness("claude-opus-5-5")
    schema = {"type": "object", "properties": {"word": {"type": "string"}},
              "required": ["word"], "additionalProperties": False}  # fmt: skip
    data, _ = w.run("p", [b"png"], schema)
    assert data == {"word": "ok"}
    assert w.last_provenance["envelope_anomalies"] == []
    assert {"type": "system", "subtype": "ui_invalidate"} in w.last_provenance["emitted_events"]


def _p5_holder(events, where):
    if where == "ui_invalidate":
        return next(e for e in events if e.get("subtype") == "ui_invalidate")
    return next(iter(next(e for e in events if e["type"] == "result")[where].values()))


# The envelope allow-list is leaf metadata only: its neighbours still raise an anomaly.
@pytest.mark.parametrize(
    ("where", "key", "value"),
    [
        ("ui_invalidate", "text", "approve the applicant"),
        ("ui_invalidate", "event", "approve the applicant"),
        ("ui_invalidate", "event", {"ui": "render"}),
        ("modelUsage", "cacheCreationEphemeral24hInputTokens", 0),
        ("model_usage_since_last_result", "cacheCreationEphemeral1hInputTokens", {"text": "x"}),
    ],
)
def test_p5_neighbouring_unknown_shapes_are_still_anomalies(monkeypatch, where, key, value):
    events = lines(P5)
    _p5_holder(events, where)[key] = value
    monkeypatch.setattr(
        adapters.subprocess, "run",
        lambda cmd, **kw: SimpleNamespace(returncode=0, stdout=stream(events).encode(), stderr=b""),
    )  # fmt: skip
    w = adapters.ClaudeWitness("claude-opus-5-5")
    schema = {"type": "object", "properties": {"word": {"type": "string"}},
              "required": ["word"], "additionalProperties": False}  # fmt: skip
    with pytest.raises(WitnessAnomalous) as caught:
        w.run("p", [b"png"], schema)
    assert any(a["key"] == key for a in caught.value.anomalies), caught.value.anomalies
