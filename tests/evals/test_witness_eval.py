"""The witness backend eval (L6): the labels, the eval's own fault test, record/replay, the
dev-only `agy` adapter's blindness, and replay of the recorded runs. No network: `agy` is a fake
script here and the API backend is never constructed."""

from __future__ import annotations

import json
import os
import stat
import sys

import pytest

from kyb.evals.witness import backends, comparator, faults, labelled
from kyb.evals.witness import evaluate as wev
from kyb.evals.witness.record import Recorder

# --- the labelled set and the eval's own check -----------------------------------------------


def test_crops_are_pinned_by_hash():
    data = labelled.labels()
    assert len(data["crops"]) >= 12
    for c in data["crops"]:
        labelled.crop_bytes(c)  # raises if the PNG drifted from its label
    cats = {c["category"] for c in data["crops"]}
    assert {"amendment", "register_row", "sanctions_identifier", "pep_identifier"} <= cats
    negatives = [p for c in data["crops"] for p in c["probes"] if p["polarity"] == "negative"]
    assert len(negatives) >= 15  # rows carrying another party's value


def test_the_labels_as_a_witness_pass_and_every_fault_fails():
    t = wev.self_test()
    assert t["oracle_passes"] and t["oracle"]["false_accepts"] == 0
    assert t["oracle"]["positives_verified"] == t["oracle"]["positives"]
    assert set(t["faults"]) == {"swap-digits", "invent-value", "ignore-strike"}
    assert all(f["fails_gate"] for f in t["faults"].values()), t["faults"]
    assert "b05-share-18current" in t["faults"]["ignore-strike"]["false_accept_ids"]


def test_faults_rewrite_like_the_failure_they_name():
    crop = {
        "transcription": "Company number: 3184920",
        "readable": True,
        "reread": {"subject": "X", "value": "46%", "struck": ["32%"], "annotations": []},
    }
    assert faults.swap_digits(crop)["transcription"] == "Company number: 1384920"
    assert faults.invent_value(crop)["reread"]["value"] == "37%"
    ig = faults.ignore_strike(crop)
    assert ig["reread"]["value"] == "32%" and ig["reread"]["struck"] == []
    assert crop["reread"]["struck"] == ["32%"]  # the original is untouched


def test_a_negative_probe_never_verifies_against_another_partys_row():
    data = labelled.labels()
    o = labelled.oracle(data)
    crop = next(c for c in data["crops"] if c["id"] == "b04-navarro")
    neg = next(p for p in crop["probes"] if p["id"] == "b04-navarro-ortega46")
    assert not comparator.verifies(neg["claim"], o["b04-navarro"])
    pos = next(p for p in crop["probes"] if p["polarity"] == "positive")
    assert comparator.verifies(pos["claim"], o["b04-navarro"])


# --- record / replay ---------------------------------------------------------------------------


class FakeAdapter:
    model, prompt_version = "fake-1", "witness-v3"

    def __init__(self):
        self.calls = 0
        self.last_provenance = {"reported_model": "fake-1-0001"}

    def call(self, kind, images, spec):
        self.calls += 1
        return {
            "crops": [
                {
                    "crop_id": k,
                    "transcription": "x",
                    "readable": True,
                    "reread": {"subject": "s", "value": "v"},
                }
                for k in spec["crop_ids"]
            ]
        }, {"promptTokenCount": 3}


def test_record_then_replay_with_zero_calls(tmp_path):
    b = {"id": "fake", "model": "fake-1", "transport": "api"}
    live = Recorder(tmp_path, b, FakeAdapter(), live=True)
    r1 = live.ask("transcribe", [b"img"], {"crop_ids": ["k1"]})
    assert not r1.replayed and r1.model == "fake-1-0001"
    rec = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert rec["reported_model"] == "fake-1-0001" and rec["latency_s"] is not None
    replay = Recorder(tmp_path, b)
    r2 = replay.ask("transcribe", [b"img"], {"crop_ids": ["k1"]})
    assert r2.replayed and r2.data == r1.data
    again = FakeAdapter()
    Recorder(tmp_path, b, again, live=True).ask("transcribe", [b"img"], {"crop_ids": ["k1"]})
    assert again.calls == 0  # a recording is never re-asked
    with pytest.raises(LookupError):
        replay.ask("transcribe", [b"other"], {"crop_ids": ["k1"]})


# --- the CLI adapters (dev/eval only) ---------------------------------------------------------

FAKE_AGY = r"""#!{py}
import json, os, sys
argv = sys.argv[1:]
log = {{"cwd_files": sorted(os.listdir(".")), "argv": argv}}
open(os.environ["FAKE_AGY_LOG"], "w").write(json.dumps(log))
prompt = argv[argv.index("-p") + 1]
ids = [f[:-4] for f in sorted(os.listdir(".")) if f.endswith(".png")]
resp = {{"crops": [{{"crop_id": i, "transcription": "t", "readable": True,
         "reread": {{"subject": "s", "value": "v"}}}} for i in ids]}}
print(json.dumps({{"conversation_id": "conv-1", "status": "SUCCESS",
                  "response": json.dumps(resp)}}))
"""


@pytest.fixture
def fake_agy(tmp_path, monkeypatch):
    path = tmp_path / "agy"
    path.write_text(FAKE_AGY.format(py=sys.executable))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "agy.log"
    monkeypatch.setenv("FAKE_AGY_LOG", str(log))
    monkeypatch.setattr(backends, "_BASE_ENV", (*backends._BASE_ENV, "FAKE_AGY_LOG"))
    return str(path), log


def test_agy_sees_only_the_crops_in_an_empty_sandboxed_dir(fake_agy):
    binary, log = fake_agy
    w = backends.AgyWitness("gemini-3.8-flash-high", binary=binary)
    claim = "Lucas Navarro Solís 46%"
    data, usage = w.call("transcribe", [b"\x89PNG a", b"\x89PNG b"], {"crop_ids": ["k1", "k2"]})
    seen = json.loads(log.read_text())
    assert seen["cwd_files"] == ["k1.png", "k2.png"]  # nothing but the crops
    argv = seen["argv"]
    assert "--sandbox" in argv and argv[argv.index("--model") + 1] == "gemini-3.8-flash-high"
    schema = json.loads(argv[argv.index("--json-schema") + 1])
    assert schema["additionalProperties"] is False and schema["type"] == "object"
    assert claim not in " ".join(argv)  # claim-blind: the prompt names crop ids only
    assert [c["crop_id"] for c in data["crops"]] == ["k1", "k2"] and usage == {}
    prov = w.last_provenance
    assert prov["conversation_id"] == "conv-1" and prov["reported_model"] is None
    assert prov["workdir"]["unchanged"] and "reported_model_gap" in prov


def test_the_product_never_imports_the_eval_adapters():
    import pathlib

    src = pathlib.Path(backends.__file__).resolve().parents[2]
    offenders = [
        p
        for p in src.rglob("*.py")
        if "evals" not in p.parts
        and "commands" not in p.parts
        and "kyb.evals" in p.read_text("utf-8")
    ]
    assert offenders == []


def test_json_schema_is_strict_json_schema():
    from kyb.witness import gemini

    s = backends.json_schema(gemini.TRANSCRIBE_SCHEMA)
    text = json.dumps(s)
    assert '"OBJECT"' not in text and '"STRING"' not in text
    item = s["properties"]["crops"]["items"]
    assert item["additionalProperties"] is False and "crop_id" in item["required"]


# --- recorded runs ------------------------------------------------------------------------------


# Explicit comparator gains over the historical bind-v5 verdicts, reviewed with l1-a4.
# Every other verdict, including every recorded positive, must remain identical.
_BIND_V6_GAINS = {
    "20261027T052502Z-claude-es-iberia-software": {
        "c-B01-14",
        "c-B01-15",
        "c-B01-16",
        "c-B02-02",
        "c-B02-03",
        "c-B02-04",
        "c-B02-05",
        "c-B02-06",
        "c-B02-07",
        "c-B02-08",
        "c-B05-05",
        "c-B05-10",
    },
    "20261027T054817Z-claude-es-iberia-software": {
        "c-B01-14",
        "c-B01-15",
        "c-B01-16",
        "c-B01-17",
        "c-B01-21",
        "c-B01-22",
        "c-B01-23",
        "c-B02-03",
        "c-B05-04",
        "c-B05-09",
    },
    "20261027T055400Z-claude-es-iberia-software": {
        "c-B02-06",
        "c-B02-07",
        "c-B02-08",
        "c-B05-06",
        "c-B05-12",
    },
}


def test_no_secret_reaches_a_recording():
    key = os.environ.get("GEMINI_API_KEY")
    root = labelled.crop_path("x").parents[1] / "recordings"
    for f in root.rglob("*.json") if root.exists() else []:
        text = f.read_text("utf-8")
        assert "x-goog-api-key" not in text and (not key or key not in text), f


FAKE_CLI = r"""#!{py}
import json, os, sys
argv = sys.argv[1:]
open(os.environ["FAKE_CLI_LOG"], "w").write(json.dumps({{"cwd_files": sorted(os.listdir(".")),
    "argv": argv, "stdin": sys.stdin.read(), "env": sorted(os.environ)}}))
ids = [f[:-4] for f in sorted(os.listdir(".")) if f.endswith(".png")]
resp = {{"crops": [{{"crop_id": i, "transcription": "t", "readable": True,
         "reread": {{"subject": "s", "value": "v"}}}} for i in ids]}}
if "exec" in argv:  # codex: the answer goes to -o, events to stdout
    open(argv[argv.index("-o") + 1], "w").write(json.dumps(resp))
    print(json.dumps({{"type": "thread.started", "thread_id": "t-1"}}))
    print(json.dumps({{"type": "turn.completed", "usage": {{"input_tokens": 10,
                      "output_tokens": 5, "reasoning_output_tokens": 2}}}}))
else:  # claude -p
    print(json.dumps({{"type": "result", "is_error": False, "structured_output": resp,
                      "total_cost_usd": 0.01, "session_id": "s-1", "num_turns": 2,
                      "modelUsage": {{"claude-opus-5-5": {{"inputTokens": 7,
                                                          "outputTokens": 9}}}}}}))
"""


@pytest.mark.parametrize("kind", ["codex", "claude"])
def test_codex_and_claude_see_only_the_crops(kind, tmp_path, monkeypatch):
    path = tmp_path / kind
    path.write_text(FAKE_CLI.format(py=sys.executable))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "cli.log"
    monkeypatch.setenv("FAKE_CLI_LOG", str(log))
    monkeypatch.setenv("GEMINI_API_KEY", "not-for-a-cli")
    monkeypatch.setattr(backends, "_BASE_ENV", (*backends._BASE_ENV, "FAKE_CLI_LOG"))
    b = next(x for x in backends.BACKENDS if x["transport"] == kind)
    w = backends.CLI[kind](b["model"], binary=str(path), **b.get("options", {}))
    data, usage = w.call("transcribe", [b"\x89PNG a", b"\x89PNG b"], {"crop_ids": ["k1", "k2"]})
    seen = json.loads(log.read_text())
    assert seen["cwd_files"] == ["k1.png", "k2.png"] and seen["stdin"] == ""
    assert "GEMINI_API_KEY" not in seen["env"]  # Codex R1: no parent secret reaches a CLI
    argv = seen["argv"]
    assert [c["crop_id"] for c in data["crops"]] == ["k1", "k2"]
    if kind == "codex":
        assert argv[argv.index("--sandbox") + 1] == "read-only" and "--ephemeral" in argv
        assert argv.count("-i") == 2 and usage["promptTokenCount"] == 10
        assert {"shell_tool", "unified_exec"} <= set(argv)  # no shell: the sandbox reads the disk
        assert w.last_provenance["reported_model"] is None
    else:
        assert "--restricted" in argv and argv[argv.index("--tools") + 1] == "Read"
        assert w.last_provenance["reported_model"] == "claude-opus-5-5"
        assert usage["cost_usd"] == 0.01
    assert w.last_provenance["workdir"]["unchanged"]


def test_codex_schema_requires_every_property():
    s = backends.CodexWitness("gpt-6.1-sol").schema()
    rr = s["properties"]["crops"]["items"]["properties"]["reread"]
    assert set(rr["required"]) == set(rr["properties"])
    assert rr["properties"]["struck"]["type"] == ["array", "null"]


def test_strata_follow_the_reader_family():
    st = {b["id"]: backends.stratum(b) for b in backends.BACKENDS}
    assert st["claude-opus-5-5"].startswith("B") and st["codex-gpt-6.1-sol"].startswith("A")
    assert {b["family"] for b in backends.BACKENDS} == {"google", "openai", "anthropic"}


def test_reading_is_scored_apart_from_binding():
    """A witness-v3 answer whose one slot holds another value of the row reads correctly but
    does not bind: a lost verification, never a misread and never a false accept."""
    data = labelled.labels()
    crop = next(c for c in data["crops"] if c["id"] == "a02-rows")
    resp = {**labelled.oracle(data)["a02-rows"]}
    resp["reread"] = {**resp["reread"], "value": "60%"}
    scored = labelled.score_crop(crop, resp)
    assert scored["correct"] and scored["unbound_positives"] == ["a02-rows+herrera-dob"]
    assert not any(p["verified"] for p in scored["probes"] if p["polarity"] == "negative")


def test_an_ungrounded_reread_is_a_misread():
    crop = {
        "transcription": "Company number: 3184920",
        "readable": True,
        "reread": {"subject": "N", "value": "Company number: 3184920 (reg. 9 Jan 2026)"},
    }
    assert labelled._ungrounded(crop) == ["Company number: 3184920 (reg. 9 Jan 2026)"]
    crop["reread"]["value"] = "Company number: 3184920"
    assert labelled._ungrounded(crop) == []
    crop["reread"]["value"] = "Lorem Ipsum Dolor"
    assert labelled._ungrounded(crop) == ["Lorem Ipsum Dolor"]


def test_handwriting_is_marked_by_value_or_by_word():
    marked = {
        "transcription": "* L. Navarro recibe 10% extra",
        "reread": {"annotations": ["Nota manuscrita en azul"]},
    }
    assert labelled._annotated(marked, "10%")
    assert labelled._annotated({"reread": {"annotations": ["* recibe 10% extra"]}}, "10%")
    assert not labelled._annotated({"reread": {"annotations": []}}, "10%")
    unseen = {"transcription": "L. Navarro", "reread": {"annotations": ["handwritten"]}}
    assert not labelled._annotated(unseen, "10%")  # a bare marker is not an observation


def test_a_fault_counts_as_caught_only_if_it_made_things_worse():
    base = {"correct": 10, "false_accepts": 0, "gate": False}
    assert not wev._caught(base, {"correct": 10, "false_accepts": 0, "gate": False})
    assert wev._caught(base, {"correct": 6, "false_accepts": 0, "gate": False})
    assert wev._caught(base, {"correct": 10, "false_accepts": 1, "gate": False})


def test_review_r1_a_handwritten_value_must_be_observed():
    """Codex R1: B05 with its re-read turned back to 46% and a bare "handwritten" note no longer
    shows 68% anywhere, so the crop is not read correctly."""
    data = labelled.labels()
    crop = next(c for c in data["crops"] if c["id"] == "b05-share")
    resp = labelled.oracle(data)["b05-share"]
    resp = {
        **resp,
        "reread": {**resp["reread"], "value": "Folio 28", "annotations": ["handwritten"]},
    }
    assert not labelled.score_crop(crop, resp)["correct"]


def test_review_r1_grounding_matches_whole_numbers():
    """Codex R1: `887` is a substring of 3184920 but not a number the witness read."""
    data = labelled.labels()
    crop = next(c for c in data["crops"] if c["id"] == "c01-regno")
    resp = labelled.oracle(data)["c01-regno"]
    resp = {**resp, "reread": {**resp["reread"], "value": "887"}}
    scored = labelled.score_crop(crop, resp)
    assert not scored["correct"] and any("887" in m for m in scored["misses"])
    assert labelled._ungrounded({"transcription": "64,00%", "reread": {"value": "64%"}}) == []


def test_review_r1_a_failed_probe_is_never_containment(tmp_path):
    """Codex R1: a CLI that exits 2 with no output looked contained."""
    path = tmp_path / "claude"
    path.write_text(f"#!{sys.executable}\nimport sys\nsys.exit(2)\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    b = backends.backend("claude-opus-5-5")
    res = backends.isolation_probe(b, b"\x89PNG", backends.ClaudeWitness(b["model"], str(path)))
    for probe in ("file", "env"):
        assert res[probe]["status"] == "failed" and res[probe]["leaked"] is None


def test_review_r1_seeded_replies_count_in_usage():
    """Codex R1: the pin's product-seeded replies were left out of its usage."""
    calls = [
        {
            "source": "product seed",
            "latency_s": None,
            "reported_model": "m",
            "gaps": [],
            "usage": {"promptTokenCount": 100, "candidatesTokenCount": 10},
        },
        {
            "source": "live",
            "latency_s": 2.0,
            "reported_model": "m",
            "gaps": [],
            "usage": {"promptTokenCount": 5, "candidatesTokenCount": 1},
        },
    ]
    u = wev._usage(calls)
    assert (u["tokens_in"], u["tokens_out"]) == (105, 11)
    assert u["by_source"]["product seed"]["tokens_in"] == 100 and u["timed_requests"] == 1


@pytest.mark.parametrize(
    ("crop_id", "value"),
    [("c02-member", "32%"), ("a03-herrera", "256"),
     ("c02-director", "Corporate director (beneficial owner)"), ("c04-founder", "ZZ-0944999")],
)  # fmt: skip
def test_review_r2_a_reread_must_be_on_the_crop(crop_id, value):
    """Codex R2: a thousands group is not a number (50,000 ≠ 50; 256.000 ≠ 220), an identifier
    is a whole token (ZZ-0944999 ≠ PY-0833921), and a value's words must all be on the crop."""
    data = labelled.labels()
    crop = next(c for c in data["crops"] if c["id"] == crop_id)
    resp = labelled.oracle(data)[crop_id]
    resp = {**resp, "reread": {**resp["reread"], "value": value}}
    assert not labelled.score_crop(crop, resp)["correct"]


def test_review_r2_five_corrupted_rereads_fail_the_gate():
    data = labelled.labels()
    o = labelled.oracle(data)
    for cid, v in (("c02-member", "32%"), ("a03-herrera", "256"), ("c04-founder", "ZZ-0944999"),
                   ("c02-director", "Corporate director (beneficial owner)"),
                   ("c03-ubo-row", "1971-09-16")):  # fmt: skip
        o[cid] = {**o[cid], "reread": {**o[cid]["reread"], "value": v}}
    assert not labelled.score(data, o)["gate"]


def test_review_r2_paired_latency_is_the_median_of_slower_calls():
    a = [{"request": k, "latency_s": t} for k, t in zip("xyz", (1, 1, 100), strict=True)]
    b = [{"request": k, "latency_s": t} for k, t in zip("xyz", (100, 1, 1), strict=True)]
    lat = wev.paired_latency(a, b)
    assert (lat["median"], lat["max"], lat["timed"]) == (100, 100, 3)
    seeded = [{"request": "x", "latency_s": None}]
    assert wev.paired_latency(seeded, b)["untimed"] == 1


def test_review_r3_short_tokens_and_signs_are_checked():
    """Codex R3 (the round-4 prediction): five short substitutions, and negated percentages,
    each scored 14/14 with the gate passing. Every one must now be a misread."""
    data = labelled.labels()
    o = labelled.oracle(data)
    subs = {"c01-regno": "N/A", "c02-director": "UBO", "c03-office": "PEP", "c04-founder": "XXX",
            "c03-ubo-row": "N/A"}  # fmt: skip
    short = {k: ({**v, "reread": {**v["reread"], "value": subs[k]}} if k in subs else v)
             for k, v in o.items()}  # fmt: skip
    scored = labelled.score(data, short)
    assert (scored["correct"], scored["gate"]) == (9, False)
    nominee = {**o["c02-director"], "reread": {**o["c02-director"]["reread"],
               "value": "Corporate director (not nominee)"}}  # fmt: skip
    crop = next(c for c in data["crops"] if c["id"] == "c02-director")
    assert not labelled.score_crop(crop, nominee)["correct"]
    negated = {k: ({**v, "reread": {**v["reread"], "value": "-" + v["reread"]["value"]}}
                   if v["reread"]["value"].endswith("%") else v) for k, v in o.items()}  # fmt: skip
    n_pct = sum(v["reread"]["value"].endswith("%") for v in o.values())
    scored = labelled.score(data, negated)
    assert n_pct >= 6 and scored["correct"] == 14 - n_pct and not scored["gate"]


# l1-a4: historical replay must ask and key the actual v3 request; product stays v4.


def test_historical_benchmark_pins_the_actual_v3_prompt_and_schema(monkeypatch):
    from kyb.evals.witness import prompts
    from kyb.util.hashing import sha256_bytes, sha256_json
    from kyb.witness import gemini

    assert sha256_bytes(prompts.TRANSCRIBE_PROMPT.encode()) == (
        "sha256:fa7b96bb6c6623d3a15f4de843be74fde8b18c0e378db93aeae316542aa7736e"
    )
    assert sha256_json(prompts.TRANSCRIBE_SCHEMA) == (
        "sha256:9fc7438705822b76f2bc4e4b8c6dfebba4f7e16d3c5298c952a7d4528ac3fde5"
    )
    assert gemini.PROMPT_VERSION == gemini.GeminiWitness.prompt_version == "witness-v4"
    assert (
        "values"
        in gemini.TRANSCRIBE_SCHEMA["properties"]["crops"]["items"]["properties"]["reread"][
            "properties"
        ]
    )
    spec = {"crop_ids": ["k1"]}
    historical = backends._prompt(spec, "witness-v3")
    assert prompts.GUARD in historical
    monkeypatch.setattr(gemini, "_GUARD", "future product guard")
    assert backends._prompt(spec, "witness-v3") == historical
    assert "future product guard" in backends._prompt(spec, "witness-v4")
    assert all(b["prompt_version"] == "witness-v3" for b in backends.BACKENDS)
    monkeypatch.setenv("GEMINI_API_KEY", "test-only")
    for b in backends.BACKENDS:
        if b["transport"] == "api":
            continue
        adapter = backends.adapter(b)
        assert adapter.prompt_version == "witness-v3"
        if isinstance(adapter, backends.CliWitness):
            expected = backends.json_schema(prompts.TRANSCRIBE_SCHEMA)
            if isinstance(adapter, backends.CodexWitness):
                expected = backends._all_required(expected)
            assert adapter.schema() == expected


def test_benchmark_api_sends_the_version_it_records(monkeypatch, tmp_path):
    from kyb.evals.witness import prompts

    bodies = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps(
                {
                    "modelVersion": "fake-served",
                    "candidates": [{"content": {"parts": [{"text": '{"crops": []}'}]}}],
                }
            ).encode()

    def open_request(req, **kwargs):
        bodies.append(json.loads(req.data))
        return Response()

    monkeypatch.setattr(backends, "adc_token", lambda: "ya29.fake-token-sentinel-9876543210")
    monkeypatch.setattr(backends.urllib.request, "urlopen", open_request)
    b = backends.backend("vertex-gcp:gemini-3.1-pro-preview")
    adapter = backends.adapter(b)
    recorder = Recorder(tmp_path, b, adapter, live=True)
    spec = {"crop_ids": ["k1"]}
    assert not recorder.ask("transcribe", [b"image"], spec).replayed
    body = bodies[0]
    assert body["contents"][0]["parts"][0]["text"] == backends._prompt(spec, "witness-v3")
    assert body["generationConfig"]["responseSchema"] == prompts.TRANSCRIBE_SCHEMA
    recorded = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert recorded["prompt_version"] == adapter.prompt_version == "witness-v3"
    assert backends._prompt(spec, "witness-v4") != backends._prompt(spec, "witness-v3")
    assert Recorder(tmp_path, {**b, "prompt_version": "witness-v4"}).key(
        "transcribe", [b"image"], spec
    ) != recorder.key("transcribe", [b"image"], spec)
    with pytest.raises(ValueError, match="unknown witness prompt"):
        backends._prompt(spec, "witness-unknown")


def test_benchmark_never_files_a_response_under_another_prompt_version(tmp_path):
    adapter = FakeAdapter()
    recorder = Recorder(
        tmp_path,
        {"id": "fake", "model": "fake-1", "prompt_version": "witness-v4"},
        adapter,
        live=True,
    )
    with pytest.raises(ValueError, match="adapter prompt version differs"):
        recorder.ask("transcribe", [b"image"], {"crop_ids": ["k1"]})
    assert adapter.calls == 0 and not list(tmp_path.glob("*.json"))


def test_live_benchmark_backend_vertex_route(monkeypatch):
    """issue: LIVE witness benchmark uses the approved Vertex route, never an API key."""
    import io

    # 1. With GEMINI_API_KEY set, the LIVE benchmark backend never reads or uses it
    monkeypatch.setenv("GEMINI_API_KEY", "prohibited-developer-api-key")

    # 2. The benchmark backend constructs a VertexWitness for a vertex-gcp: id
    constructed = []
    real_vertex_witness = backends.VertexWitness

    class StubVertexWitness(real_vertex_witness):
        def __init__(self, *args, **kwargs):
            constructed.append((args, kwargs))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(backends, "VertexWitness", StubVertexWitness)
    monkeypatch.setattr(backends, "adc_token", lambda: "ya29.stub-token-sentinel-9876543210")

    b = backends.backend("vertex-gcp:gemini-3.1-pro-preview")
    assert b["transport"] == "vertex-gcp"
    adapter = backends.adapter(b)
    assert len(constructed) == 1
    assert constructed[0][1].get("model") == "gemini-3.1-pro-preview"
    assert adapter.prompt_version == "witness-v3"
    assert adapter.model == "gemini-3.1-pro-preview"

    # Verify LIVE call passes through VertexWitness without reading GEMINI_API_KEY
    captured_requests = []

    def mock_urlopen(req, timeout=None):
        captured_requests.append(req)
        resp = io.BytesIO(
            json.dumps(
                {
                    "modelVersion": "gemini-3.1-pro-preview",
                    "candidates": [
                        {
                            "finishReason": "STOP",
                            "content": {
                                "parts": [
                                    {
                                        "text": json.dumps(
                                            {
                                                "crops": [
                                                    {
                                                        "crop_id": "k1",
                                                        "transcription": "t",
                                                        "readable": True,
                                                        "reread": {"subject": "s", "value": "v"},
                                                    }
                                                ]
                                            }
                                        )
                                    }
                                ]
                            },
                        }
                    ],
                    "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
                }
            ).encode()
        )
        resp.status = 200
        return resp

    monkeypatch.setattr(backends.urllib.request, "urlopen", mock_urlopen)
    data, usage = adapter.call("transcribe", [b"fake-image"], {"crop_ids": ["k1"]})
    assert len(captured_requests) == 1
    req = captured_requests[0]
    assert req.headers.get("Authorization") == "Bearer ya29.stub-token-sentinel-9876543210"
    assert req.headers.get("X-goog-user-project") == "kyb-vertex-project"
    assert "x-goog-api-key" not in [k.lower() for k in req.headers]
    assert "prohibited-developer-api-key" not in req.full_url
    assert "prohibited-developer-api-key" not in str(req.headers)
    assert data["crops"][0]["crop_id"] == "k1"

    # 3. An unsupported backend refuses
    with pytest.raises((ValueError, backends.WitnessRefused), match="prohibited|unsupported"):
        backends.adapter(
            {
                "id": "api-gemini-3.1-pro-preview",
                "transport": "api",
                "model": "gemini-3.1-pro-preview",
            }
        )

    with pytest.raises((ValueError, backends.WitnessRefused), match="unsupported"):
        backends.adapter({"id": "unknown-backend", "transport": "unknown", "model": "foo"})


def test_legacy_and_pending_vertex_backends():
    """issue: Verify legacy api-gemini-* recordings keep original IDs

    and Vertex-GCP is pending.
    """
    from kyb.evals.witness import backends
    from kyb.evals.witness import evaluate as wev

    # 1. Check legacy api-gemini-* backends exist and have legacy_route set
    b_api_31 = backends.backend("api-gemini-3.1-pro-preview")
    assert b_api_31["transport"] == "api"
    assert b_api_31["legacy_route"] == "api-key (prohibited since spec)"

    # 2. Check evaluating without live (replay) reports legacy backends
    # under their legacy ID, never as vertex-gcp
    report = wev.evaluate(live=False)
    rows_by_backend = {r["backend"]: r for r in report["rows"]}

    # Legacy api-gemini should have actual results and are marked as legacy_route
    assert "api-gemini-3.1-pro-preview" in rows_by_backend
    assert "api-gemini-3.8-flash" in rows_by_backend

    api_31_row = rows_by_backend["api-gemini-3.1-pro-preview"]
    assert api_31_row.get("legacy_route") == "api-key (prohibited since spec)"
    # legacy has complete recordings
    assert api_31_row["correct"] is None

    # vertex-gcp backends are pending and have no recordings
    assert "vertex-gcp:gemini-3.1-pro-preview" in rows_by_backend
    assert "vertex-gcp:gemini-3.8-flash" in rows_by_backend

    # vertex-gcp should have missing entries and be pending
    missing_31 = rows_by_backend["vertex-gcp:gemini-3.1-pro-preview"]["missing"]
    assert any("labelled: " in m for m in missing_31)
    assert rows_by_backend["vertex-gcp:gemini-3.1-pro-preview"]["correct"] is None


def test_a_legacy_backend_never_records_live():
    """A live run treats the prohibited API-route backends as replay-only (issue review)."""
    from kyb.evals.witness import backends, evaluate

    legacy = next(b for b in backends.BACKENDS if b.get("legacy_route"))
    approved = next(b for b in backends.BACKENDS if b["transport"] == "vertex-gcp")
    assert evaluate._recorder(legacy, True).live is False
    assert evaluate._recorder(approved, True).live is True
