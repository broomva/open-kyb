"""J-SUPPORT on Jev (spec): recorded port, veto-only judge, labelled replay.

The labelled replay reads datasets/support/recordings and must not call an adapter. Unit tests
use a fake port. The live TypeSafe call happens once, outside this file, when the recordings
are produced.
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request

import pytest

from kyb.judges import support as support_mod
from kyb.judges.support import (
    RECORDINGS,
    FixtureSupportPort,
    evaluate_support_set,
    j_support,
    load_support_cases,
    passes_false_pass_gate,
    support_body,
    support_case,
)
from kyb.util.hashing import sha256_json
from kyb.witness import jev as jev_mod
from kyb.witness.jev import (
    ENDPOINT,
    MODEL_ALIAS,
    PROMPT_VERSION,
    JevClient,
    JevPort,
    JevResponse,
    JevUnavailable,
)

ALLOWED = {("pass", "none"), ("veto", "manual_review"), ("veto", "unknown")}


class FakePort:
    def __init__(self, noul=0.9, *, family="typesafe-jev", model="jev-1.14.0", error=None):
        self.family = family
        self.noul = noul
        self.model = model
        self.error = error
        self.bodies: list[dict] = []

    def answerer_model(self, instance, answer):
        return "gpt-6.1-sol"

    def ask(self, body):
        self.bodies.append(body)
        if self.error is not None:
            raise self.error
        return JevResponse(
            model=self.model,
            answers={"supported": {"type": "noul", "noul": self.noul}},
            usage={"input_tokens": 1, "output_tokens": 1},
            request_hash="sha256:" + "ab" * 32,
            replayed=False,
        )


def _bundle(predicate="doc_is_proof_of_address", value=True, excerpts=("A fiscal address.",)):
    return support_case(
        {
            "id": "unit-1",
            "predicate": predicate,
            "value": value,
            "evidence": list(excerpts),
            "label": "supported",
            "why": "unit",
        }
    )


def test_supported_passes_with_effect_none(monkeypatch):
    monkeypatch.setattr(support_mod, "support_threshold", 0.7)
    port = FakePort(0.7)
    outcome = j_support(*_bundle(), port=port)
    assert (outcome.verdict, outcome.effect) == ("pass", "none")
    assert (outcome.verdict, outcome.effect) in ALLOWED


def test_unsupported_vetoes_manual_review(monkeypatch):
    monkeypatch.setattr(support_mod, "support_threshold", 0.7)
    outcome = j_support(*_bundle(), port=FakePort(0.6999))
    assert (outcome.verdict, outcome.effect) == ("veto", "manual_review")


def test_unavailable_vetoes_unknown():
    err = JevUnavailable("down", request_hash="sha256:" + "cd" * 32, model=None)
    outcome = j_support(*_bundle(), port=FakePort(error=err))
    assert (outcome.verdict, outcome.effect) == ("veto", "unknown")
    assert outcome.request_hash == err.request_hash
    assert "support_threshold=" in outcome.reason


def test_outcome_records_threshold_and_cross_family(monkeypatch):
    monkeypatch.setattr(support_mod, "support_threshold", 0.7)
    outcome = j_support(*_bundle(), port=FakePort(0.91, model="jev-1.14.0"))
    assert outcome.judge_id == "J-SUPPORT"
    assert outcome.target == "unit-1"
    assert outcome.model == "jev-1.14.0"
    assert outcome.request_hash.startswith("sha256:")
    assert "support_threshold=0.7000" in outcome.reason
    assert "families_differ=true" in outcome.reason
    assert "judge_family=typesafe-jev" in outcome.reason
    assert "answerer_family=openai" in outcome.reason
    assert "p_supported=0.9100" in outcome.reason


def test_same_family_cannot_pass():
    port = FakePort(1.0, family="openai")
    outcome = j_support(*_bundle(), port=port)
    assert (outcome.verdict, outcome.effect) == ("veto", "unknown")
    assert port.bodies == []
    assert "families_differ=false" in outcome.reason


def test_missing_family_vetoes_unknown():
    port = FakePort(1.0, family="")
    outcome = j_support(*_bundle(), port=port)
    assert (outcome.verdict, outcome.effect) == ("veto", "unknown")
    assert port.bodies == []
    assert "judge_family=missing" in outcome.reason


def test_missing_citation_vetoes_manual_review():
    instance, answer, evidence = _bundle(excerpts=("one", "two"))
    port = FakePort(1.0)
    outcome = j_support(instance, answer, evidence[:1], port=port)
    assert (outcome.verdict, outcome.effect) == ("veto", "manual_review")
    assert port.bodies == []


def test_empty_evidence_vetoes_without_a_call():
    port = FakePort(0.95)
    outcome = j_support(*_bundle(excerpts=()), port=port)
    assert (outcome.verdict, outcome.effect) == ("veto", "manual_review")
    assert port.bodies == []


def test_instance_mismatch_vetoes_unknown():
    instance, answer, evidence = _bundle()
    shifted = answer.model_copy(update={"instance_id": "other"})
    port = FakePort(1.0)
    outcome = j_support(instance, shifted, evidence, port=port)
    assert (outcome.verdict, outcome.effect) == ("veto", "unknown")
    assert outcome.target == instance.instance_id
    assert port.bodies == []


def test_unknown_value_vetoes_without_a_call():
    instance, answer, evidence = _bundle()
    unknown = answer.model_copy(update={"value": "UNKNOWN"})
    port = FakePort(1.0)
    outcome = j_support(instance, unknown, evidence, port=port)
    assert (outcome.verdict, outcome.effect) == ("veto", "unknown")
    assert port.bodies == []


def test_malformed_noul_vetoes_unknown_and_keeps_hash():
    port = FakePort(0.4)
    port.ask = lambda body: JevResponse(
        model="jev-1.14.0",
        answers={"supported": {"type": "noul", "noul": 1.4}},
        usage={},
        request_hash="sha256:" + "ef" * 32,
        replayed=False,
    )
    outcome = j_support(*_bundle(), port=port)
    assert (outcome.verdict, outcome.effect) == ("veto", "unknown")
    assert outcome.model == "jev-1.14.0"
    assert outcome.request_hash.endswith("ef" * 32)
    assert "p_supported=unavailable" in outcome.reason


def test_request_sends_exact_typed_target_args():
    instance, answer, evidence = support_case(load_support_cases()[0])
    body = support_body(instance, answer, evidence)
    encoded = json.dumps(body)
    assert body["model"] == MODEL_ALIAS
    assert body["questions"]["supported"]["type"] == "noul"
    assert body["state"]["predicate_args"] == instance.args.model_dump(mode="json")
    assert evidence[0].excerpt[:40] in encoded
    assert "untrusted" in body["questions"]["supported"]["instructions"]


def test_replay_makes_zero_calls(tmp_path):
    calls = {"n": 0}

    class Adapter:
        def ask(self, body):
            calls["n"] += 1
            return {
                "model": "jev-1.14.0",
                "answers": {"supported": {"type": "noul", "noul": 0.88}},
                "usage": {"input_tokens": 2, "output_tokens": 1},
            }

    body = support_body(*_bundle())
    live = JevPort(tmp_path, Adapter(), mode="auto")
    first = live.ask(body)
    assert first.replayed is False
    assert calls["n"] == 1

    class Boom:
        def ask(self, body):
            raise AssertionError("replay called the adapter")

    second = JevPort(tmp_path, Boom(), mode="replay").ask(body)
    assert second.replayed is True
    assert second.request_hash == first.request_hash
    assert calls["n"] == 1


def test_recording_stores_reported_model(tmp_path):
    class Adapter:
        def ask(self, body):
            return {
                "model": "jev-1.14.0",
                "answers": {"supported": {"type": "noul", "noul": 0.5}},
                "usage": {"input_tokens": 2, "output_tokens": 1},
            }

    body = {"model": MODEL_ALIAS, "state": {"n": 1}, "questions": {"supported": {"type": "noul"}}}
    response = JevPort(tmp_path, Adapter(), mode="live").ask(body)
    assert response.model == "jev-1.14.0"
    saved = json.loads(next(tmp_path.glob("*/0.json")).read_text(encoding="utf-8"))
    assert saved["model"] == "jev-1.14.0"
    assert saved["alias"] == MODEL_ALIAS
    assert saved["prompt_version"] == PROMPT_VERSION
    assert saved["request"] == response.request_hash


def test_request_is_content_addressed(tmp_path):
    class Adapter:
        def __init__(self):
            self.calls = 0

        def ask(self, body):
            self.calls += 1
            return {
                "model": "jev-1.14.0",
                "answers": {"supported": {"type": "noul", "noul": 0.2}},
                "usage": {},
            }

    adapter = Adapter()
    port = JevPort(tmp_path, adapter, mode="auto")
    body = support_body(*_bundle())
    first = port.ask(body)
    second = port.ask(dict(reversed(list(body.items()))))
    assert first.request_hash == second.request_hash
    assert adapter.calls == 1
    other = dict(body)
    other["state"] = {**body["state"], "proposed_value": False}
    assert port.ask(other).request_hash != first.request_hash
    salted = sha256_json({"v": PROMPT_VERSION, "body": body})
    assert first.request_hash == salted
    assert "unit-test-key" not in salted


def test_replay_miss_is_unavailable(tmp_path):
    body = support_body(*_bundle())
    with pytest.raises(JevUnavailable) as caught:
        JevPort(tmp_path, None, mode="replay").ask(body)
    assert caught.value.request_hash is not None

    class TrustedReplay:
        family = "typesafe-jev"

        def answerer_model(self, instance, answer):
            return "gpt-6.1-sol"

        def ask(self, body):
            return JevPort(tmp_path, None, mode="replay").ask(body)

    outcome = j_support(*_bundle(), port=TrustedReplay())
    assert (outcome.verdict, outcome.effect) == ("veto", "unknown")
    assert outcome.request_hash == caught.value.request_hash


def test_client_refuses_without_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY"):
        JevClient()


def test_request_targets_systemone_noul(monkeypatch):
    captured = {}

    class Dummy(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_open(request, timeout=0):
        captured["url"] = request.full_url
        captured["header"] = request.get_header("Authorization")
        captured["body"] = request.data.decode()
        return Dummy(
            b'{"model":"jev-1.14.0","answers":{"supported":{"type":"noul","noul":0.4}},'
            b'"usage":{"input_tokens":3,"output_tokens":1}}'
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_open)
    parsed = JevClient(api_key="unit-test-key").ask(support_body(*_bundle()))
    assert captured["url"] == ENDPOINT
    assert captured["header"] == "Bearer unit-test-key"
    assert "unit-test-key" not in captured["body"]
    sent = json.loads(captured["body"])
    assert sent["model"] == "jev-latest"
    assert sent["questions"]["supported"]["type"] == "noul"
    assert parsed["model"] == "jev-1.14.0"


def test_client_does_not_retry_422(monkeypatch):
    monkeypatch.setattr(jev_mod.time, "sleep", lambda _seconds: None)
    calls = {"n": 0}

    def fake_open(request, timeout=0):
        calls["n"] += 1
        raise urllib.error.HTTPError(
            ENDPOINT,
            422,
            "no",
            hdrs=None,
            fp=io.BytesIO(b'{"detail":"unit-test-key leaked"}'),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_open)
    with pytest.raises(JevUnavailable) as caught:
        JevClient(api_key="unit-test-key").ask(
            {"model": "jev-latest", "state": {}, "questions": {}}
        )
    assert calls["n"] == 1
    assert "unit-test-key" not in str(caught.value)


def test_client_retries_503_then_succeeds(monkeypatch):
    monkeypatch.setattr(jev_mod.time, "sleep", lambda _seconds: None)
    calls = {"n": 0}

    class Dummy(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_open(request, timeout=0):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(ENDPOINT, 503, "busy", hdrs=None, fp=io.BytesIO(b"busy"))
        return Dummy(
            b'{"model":"jev-1.14.0","answers":{"q":{"type":"noul","noul":0.1}},'
            b'"usage":{"input_tokens":1,"output_tokens":1}}'
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_open)
    parsed = JevClient(api_key="unit-test-key").ask({"model": "jev-latest"})
    assert calls["n"] == 2
    assert parsed["model"] == "jev-1.14.0"


def test_jev_family_differs_from_the_predicate_answerer():
    assert JevPort.family == "typesafe-jev"
    assert JevPort.family != "openai"


def test_always_supported_port_fails_the_suite():
    report = evaluate_support_set(
        FakePort(1.0, family="typesafe-jev", model="jev-fault-always-supported")
    )
    unsupported = sum(1 for case in load_support_cases() if case["label"] == "unsupported")
    assert unsupported == 8
    assert report["false_passes"] == unsupported
    assert passes_false_pass_gate(report) is False


def test_prompt_version_changes_the_hash(monkeypatch, tmp_path):
    class Adapter:
        def ask(self, body):
            return {
                "model": "jev-1.14.0",
                "answers": {"supported": {"type": "noul", "noul": 0.5}},
                "usage": {},
            }

    body = support_body(*_bundle())
    first = JevPort(tmp_path, Adapter(), mode="live").ask(body)
    monkeypatch.setattr(jev_mod, "PROMPT_VERSION", "j-support-v3")
    second = JevPort(tmp_path / "other", Adapter(), mode="live").ask(body)
    assert first.request_hash != second.request_hash


def test_only_cited_evidence_reaches_the_model():
    instance, answer, evidence = _bundle(excerpts=("silent on address", "address exists"))
    answer = answer.model_copy(update={"evidence": [evidence[0].evidence_id]})
    port = FakePort(0.1)
    j_support(instance, answer, evidence, port=port)
    assert [x["excerpt"] for x in port.bodies[0]["state"]["evidence"]] == ["silent on address"]


@pytest.mark.parametrize("fault", ["unverified", "duplicate", "outside-universe"])
def test_unsafe_citations_veto_before_call(fault):
    instance, answer, evidence = _bundle()
    if fault == "unverified":
        evidence = [evidence[0].model_copy(update={"verified": False})]
    elif fault == "duplicate":
        evidence = evidence + evidence
    else:
        instance = instance.model_copy(update={"evidence_universe": []})
    port = FakePort(1)
    outcome = j_support(instance, answer, evidence, port=port)
    assert (outcome.verdict, outcome.effect) == ("veto", "manual_review")
    assert port.bodies == []


@pytest.mark.parametrize(
    "field,value",
    [("request", "sha256:" + "00" * 32), ("prompt_version", "other"), ("model", "other")],
)
def test_replay_rejects_mismatched_provenance(tmp_path, field, value):
    body = support_body(*_bundle())

    class Adapter:
        def ask(self, body):
            return {"model": "jev-1.14.0", "answers": {"supported": {"type": "noul", "noul": 1}}}

    JevPort(tmp_path, Adapter(), mode="live").ask(body)
    path = next(tmp_path.glob("*/0.json"))
    record = json.loads(path.read_text())
    record[field] = value
    path.write_text(json.dumps(record))
    with pytest.raises(JevUnavailable):
        JevPort(tmp_path, None, mode="replay").ask(body)


@pytest.mark.parametrize(
    "model,request_hash", [("", "sha256:" + "ab" * 32), ("jev-1.14.0", "not-a-hash")]
)
def test_missing_response_provenance_cannot_pass(model, request_hash):
    port = FakePort()
    port.ask = lambda body: JevResponse(
        model, {"supported": {"type": "noul", "noul": 1}}, {}, request_hash, False
    )
    outcome = j_support(*_bundle(), port=port)
    assert (outcome.verdict, outcome.effect) == ("veto", "unknown")


@pytest.mark.parametrize("model", [None, "", "unknown", "kyb-predicates", "jev-1.14.0"])
def test_unknown_or_same_family_provenance_vetoes_without_call(model):
    port = FakePort()
    port.answerer_model = lambda instance, answer: model
    outcome = j_support(*_bundle(), port=port)
    assert (outcome.verdict, outcome.effect) == ("veto", "unknown")
    assert port.bodies == []
    assert "families_differ=false" in outcome.reason


def test_missing_provenance_capability_fails_safe(tmp_path):
    outcome = j_support(*_bundle(), port=JevPort(tmp_path, None, mode="replay"))
    assert (outcome.verdict, outcome.effect) == ("veto", "unknown")
    assert "no trusted model provenance" in outcome.reason


def test_target_changes_request_hash_and_cannot_use_original_replay():
    case = next(c for c in load_support_cases() if c["id"] == "sup-npd-c-true")
    instance, answer, evidence = support_case(case)
    changed = instance.model_copy(
        update={"args": instance.args.model_copy(update={"company": "unrelated-company"})}
    )
    original = support_body(instance, answer, evidence)
    other = support_body(changed, answer, evidence)
    assert original != other
    assert sha256_json(original) != sha256_json(other)

    class TrustedReplay:
        family = "typesafe-jev"

        def answerer_model(self, instance, answer):
            return "gpt-6.1-sol"

        def ask(self, body):
            return JevPort(RECORDINGS / "v2", None, mode="replay").ask(body)

    outcome = j_support(changed, answer, evidence, port=TrustedReplay())
    assert (outcome.verdict, outcome.effect) == ("veto", "unknown")


def test_fixture_provenance_is_bound_to_exact_input():
    case = load_support_cases()[0]
    instance, answer, evidence = support_case(case)
    port = FixtureSupportPort(FakePort(), [case])
    assert port.answerer_model(instance, answer) == "fixture:support-v1"
    changed = answer.model_copy(update={"value": not answer.value})
    assert port.answerer_model(instance, changed) is None
    assert j_support(instance, changed, evidence, port=port).effect == "unknown"


def test_unavailable_eval_cannot_masquerade_as_a_calibration_pass():
    report = evaluate_support_set(FakePort(error=JevUnavailable("offline")))
    assert report["false_passes"] == 0
    assert passes_false_pass_gate(report) is False


def test_replay_rejects_modified_request_body(tmp_path):
    body = support_body(*_bundle())

    class Adapter:
        def ask(self, body):
            return {"model": "jev-1.14.0", "answers": {}}

    JevPort(tmp_path, Adapter(), mode="live").ask(body)
    path = next(tmp_path.glob("*/0.json"))
    record = json.loads(path.read_text())
    record["request_body"]["state"]["predicate_args"]["document"]["doc"] = "99"
    path.write_text(json.dumps(record))
    with pytest.raises(JevUnavailable):
        JevPort(tmp_path, None, mode="replay").ask(body)


def test_always_supported_fails_target_specific_evaluation():
    cases = load_support_cases()
    port = FixtureSupportPort(FakePort(1.0, model="jev-fault-always-supported"), cases)
    report = evaluate_support_set(port, cases)
    assert report["false_passes"] == 8
    assert not passes_false_pass_gate(report)


def test_provenance_lookup_failure_vetoes_unknown():
    port = FakePort()

    def unavailable(instance, answer):
        raise OSError("no attestation")

    port.answerer_model = unavailable
    assert j_support(*_bundle(), port=port).effect == "unknown"
    assert port.bodies == []
