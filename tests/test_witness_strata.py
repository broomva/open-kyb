"""spec routing, actual-model evidence and propagation. Fakes prove contracts, not confinement."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kyb.audit.chain import AuditLog
from kyb.facts.derive import DocInput, _verification
from kyb.util.hashing import sha256_bytes, sha256_json
from kyb.witness import adapters, verify
from kyb.witness.errors import ROUTING_VERSION, WitnessRefused, WitnessUnavailable
from kyb.witness.port import Witness
from kyb.witness.strata import Config, RoutedWitness, RoutingRefused

SPEC = {"crop_ids": ["k1"]}
DATA = {"crops": [{"crop_id": "k1", "transcription": "Irene 25%", "readable": True,
                  "reread": {"values": [{"subject": "Irene", "attribute": "pct",
                                          "value": "25%"}]}}]}  # fmt: skip
SUB = {"produced_by": {"reader": {"context": "isolated", "session": "s", "agent_id": "a",
                                  "model": "gpt-false-session-metadata"}},
       "claims": [{"claim_id": "c", "subject": "Irene", "value": {"pct": 25}}]}  # fmt: skip


def append(log, action, detail):
    from kyb.util.hashing import sha256_json

    log.append(
        "harness",
        action,
        input_hash=sha256_json(detail),
        output_hash=sha256_json(detail),
        detail=detail,
    )


def attest(root, model="claude-opus-5-5", *, complete=True):
    log = AuditLog(root / "ledger/audit.jsonl")
    for action, detail in [
        ("harness.subagent_start", {}),
        ("harness.subagent_stop", {"models": [model], "complete": complete}),
    ]:
        append(log, action, {"session": "s", "agent_id": "a",
                                              "hook_source": "plugin", **detail})  # fmt: skip


class Transport:
    prompt_version = "strata-test"

    def __init__(self, backend, *, actual="default", fail=False, data=None):
        self.transport, self.model = backend.split(":", 1)
        self.family = {"gemini-api": "google", "vertex-gcp": "google", "claude-p": "anthropic",
                       "codex-exec": "openai"}[self.transport]  # fmt: skip
        self.actual = self.model if actual == "default" else actual
        self.fail, self.calls = fail, []
        self.data = copy.deepcopy(DATA if data is None else data)
        self.last_provenance = {}

    def call(self, kind, images, spec):
        self.calls.append((kind, images, spec))
        self.last_provenance = {
            "transport": self.transport,
            "response_hash": sha256_json({"fixture": self.actual}),
            "family": self.family,
            "validation_version": getattr(self, "validation_version", ROUTING_VERSION),
            "reported_model": None if self.fail else self.actual,
            "tools": [],
            "no_data_tools": True,
            "classification": "unavailable" if self.fail else "complete",
            "refusal_evidence": [],
            "normal_terminal": not self.fail,
        }
        if self.transport in ("gemini-api", "vertex-gcp"):
            self.last_provenance |= {
                "http_status": 429 if self.fail else 200,
                "schema_valid": not self.fail,
                "schema_text_hash": sha256_bytes(json.dumps(self.data).encode()),
                "schema_hash": sha256_json(self.data),
                "final_output_hash": sha256_json(self.data),
                "data_hash": sha256_json(self.data),
                "prompt_block_reason": None,
                "prompt_safety": [],
                "finish_diagnostics": [],
                "error_diagnostics": [],
                "terminal_evidence": []
                if self.fail
                else [{"field": "finishReason", "value": "STOP"}],
                "candidates": [] if self.fail else [{"finish_reason": "STOP", "safety": []}],
                "error_status": {"code": 429, "status": "RESOURCE_EXHAUSTED"}
                if self.fail
                else None,
            }
            if self.fail:
                for key in (
                    "schema_valid",
                    "schema_hash",
                    "schema_text_hash",
                    "data_hash",
                    "final_output_hash",
                ):
                    self.last_provenance.pop(key)
        elif self.transport == "claude-p":
            self.last_provenance |= {
                "result_evidence": [
                    {
                        "subtype": "success",
                        "is_error": bool(self.fail),
                        "api_error_status": 429 if self.fail else None,
                        "errors": [],
                        "result_diagnostics": [],
                        "stop_reason": None,
                        "terminal_reason": None,
                        "final_output_hash": None if self.fail else sha256_json(self.data),
                    }
                ],
                "schema_checks": []
                if self.fail
                else [{"valid": True, "value_hash": sha256_json(self.data), "source": "final"}],
                "reported_models": [] if self.fail else [self.actual],
                "provider_errors": [],
                "other_diagnostics": [],
                "api_error_status": 429 if self.fail else None,
                "schema_values_checked": 0 if self.fail else 1,
                "called_tools": [],
                "workdir": {"before": ["k1.png"], "after": ["k1.png"]},
                "assistant_diagnostics": [],
                "timed_out": False,
                "returncode": 1 if self.fail else 0,
                "nonblank_lines": 3,
                "process_exception": None,
            }
        if self.transport == "claude-p":
            from kyb.witness.channels import channel, stdout_diagnostics

            frames = [{"type": "system", "subtype": "init", "tools": []}]
            if not self.fail:
                frames.append(
                    {"type": "assistant", "message": {"model": self.actual, "content": []}}
                )
            frames.append({"type": "result", "subtype": "success", "is_error": bool(self.fail)})
            raw = "\n".join(map(json.dumps, frames)).encode()
            streams = {"stdout": channel(raw), "stderr": channel(b"", diagnostic_channel=True)}
            streams["stdout"]["diagnostics"] = stdout_diagnostics(self.last_provenance)
            streams["stdout"]["statuses"] = [429] if self.fail else []
            self.last_provenance |= {
                "channels": streams,
                "stdout_bytes": len(raw),
                "stderr_bytes": 0,
                "nonblank_lines": len(frames),
                "rate_limit_events": [],
                "response_hash": sha256_json({k: v["hash"] for k, v in streams.items()}),
                "final_output_hash": None if self.fail else sha256_json(self.data),
            }
        if self.fail:
            raise WitnessUnavailable("offline transport failure")
        return self.data, {"promptTokenCount": 5}


def test_default_b_uses_stopped_actual_model_and_fresh_blind_request(tmp_path):
    attest(tmp_path)
    made = []

    def factory(backend):
        made.append(Transport(backend))
        return made[-1]

    router = RoutedWitness(tmp_path, factory=factory)
    routes = router.for_document(SUB)
    reply = routes["B"].ask("transcribe", [b"pixels"], SPEC)
    assert list(routes) == ["B"]
    assert reply.model == "claude-opus-5-5"
    assert made[0].model != SUB["produced_by"]["reader"]["model"]
    assert made[0].calls == [("transcribe", [b"pixels"], SPEC)]
    rec = json.loads(next((tmp_path / "silver/witness/raw").glob("*/0.json")).read_text())
    assert rec["provenance"]["reported_model"] == reply.model
    assert rec["requested_model"] == reply.model and rec["content_hash"]


@pytest.mark.parametrize(
    "defect",
    [
        "missing_stop",
        "incomplete",
        "unknown_model",
        "mixed",
        "resumed",
        "second_registration",
        "unattested",
    ],
)
def test_no_attestation_gap_can_be_filled_by_session_metadata(tmp_path, defect):
    sub = copy.deepcopy(SUB)
    if defect != "missing_stop":
        attest(tmp_path, "mystery" if defect == "unknown_model" else "claude-opus-5-5",
               complete=defect != "incomplete")  # fmt: skip
    log = AuditLog(tmp_path / "ledger/audit.jsonl")
    if defect == "mixed":
        append(log, "harness.subagent_stop", {"session": "s", "agent_id": "a",
            "complete": True, "models": ["claude-sonnet-5-5"],
            "hook_source": "plugin"})  # fmt: skip
    if defect == "resumed":
        append(log, "harness.subagent_start", {"session": "s", "agent_id": "a",
                                               "hook_source": "plugin"})  # fmt: skip
    if defect == "second_registration":
        append(log, "tool.call", {"session": "s", "hook_source": "workspace"})
    if defect == "unattested":
        sub["produced_by"]["reader"]["context"] = "unattested"
    with pytest.raises(RoutingRefused, match="attestation"):
        RoutedWitness(tmp_path, factory=lambda _: pytest.fail("must not call")).for_document(sub)


def test_a_refuses_same_declared_family_without_spending(tmp_path):
    attest(tmp_path)
    fake = Transport("claude-p:claude-sonnet-5-5")
    route = RoutedWitness(tmp_path, config=Config("A", ("claude-p:claude-sonnet-5-5",)),
                          factory=lambda _: fake).for_document(SUB)["A"]  # fmt: skip
    with pytest.raises(RoutingRefused):
        route.ask("transcribe", [b"crop"], SPEC)
    assert not fake.calls
    assert route.attempts == [{"backend": "claude-p:claude-sonnet-5-5",
                               "status": "same_family_refused"}]  # fmt: skip


@pytest.mark.parametrize("actual", [None, "unknown-model", "claude-opus-5-5"])
def test_a_refuses_missing_unknown_or_wrong_actual_family_and_keeps_evidence(tmp_path, actual):
    attest(tmp_path)
    fake = Transport("gemini-api:gemini-3.1-pro-preview", actual=actual)
    route = RoutedWitness(tmp_path, config=Config("A"), factory=lambda _: fake).for_document(SUB)[
        "A"
    ]
    with pytest.raises(WitnessRefused):
        route.ask("transcribe", [b"crop"], SPEC)
    assert len(fake.calls) == 1  # no paid fallback after model assurance mismatch
    raw = json.loads(next((tmp_path / "silver/witness/raw").glob("*/failed/0.json")).read_text())
    assert raw["provenance"]["reported_model"] == actual

    assert raw["status"] == "failed" and raw["failure_kind"] == "refused"


def test_b_refuses_different_actual_model_even_in_same_family(tmp_path):
    attest(tmp_path)
    route = RoutedWitness(tmp_path, factory=lambda b: Transport(b, actual="claude-sonnet-5-5"))
    with pytest.raises(RoutingRefused, match="differs"):
        route.for_document(SUB)["B"].ask("transcribe", [b"crop"], SPEC)


def test_transport_failure_is_recorded_and_fallback_order_is_evaluated_order(tmp_path):
    attest(tmp_path)
    made = []

    def factory(backend):
        made.append(Transport(backend, fail="pro" in backend))
        return made[-1]

    route = RoutedWitness(tmp_path, config=Config("A"), factory=factory).for_document(SUB)["A"]
    reply = route.ask("transcribe", [b"crop"], SPEC)
    assert reply.model == "gemini-3.8-flash"
    assert [r["status"] for r in route.attempts] == ["unavailable", "complete"]
    failure = json.loads(
        next((tmp_path / "silver/witness/raw").glob("*/failed/0.json")).read_text()
    )
    assert failure["status"] == "failed" and failure["provenance"]
    route.mode = "replay"
    assert route.ask("transcribe", [b"crop"], SPEC).replayed
    assert sum(len(w.calls) for w in made) == 2


def test_replay_uses_actual_model_without_key_or_call_and_detects_tampering(tmp_path):
    fake = Transport("gemini-api:gemini-3.1-pro-preview")
    port = Witness(tmp_path, fake)
    first = port.ask("transcribe", [b"crop"], SPEC)
    fake.call = lambda *args: pytest.fail("replay must make zero calls")
    again = Witness(tmp_path, fake, mode="replay").ask("transcribe", [b"crop"], SPEC)
    assert again.replayed and again.model == first.model and again.provenance == first.provenance
    path = next((tmp_path / "silver/witness/raw").glob("*/0.json"))
    rec = json.loads(path.read_text())
    rec["model"] = "claude-opus-5-5"
    path.write_text(json.dumps(rec))
    with pytest.raises(WitnessRefused, match="hash"):
        port.ask("transcribe", [b"crop"], SPEC)


def test_failed_attempt_is_not_retried_automatically_and_can_explicitly_recover(tmp_path):
    fake = Transport("gemini-api:gemini-3.1-pro-preview", fail=True)
    with pytest.raises(RuntimeError):
        Witness(tmp_path, fake).ask("transcribe", [b"crop"], SPEC)
    fake.fail = False
    with pytest.raises(RuntimeError, match="explicit live"):
        Witness(tmp_path, fake).ask("transcribe", [b"crop"], SPEC)
    assert len(fake.calls) == 1
    assert not Witness(tmp_path, fake, mode="live").ask("transcribe", [b"crop"], SPEC).replayed
    assert Witness(tmp_path, fake).ask("transcribe", [b"crop"], SPEC).replayed


def test_transport_identity_prevents_cross_transport_replay(tmp_path):
    a = Transport("gemini-api:gemini-3.1-pro-preview")
    b = Transport("codex-exec:gemini-3.1-pro-preview")
    assert Witness(tmp_path, a)._request("transcribe", [b"crop"], SPEC)[0] != Witness(
        tmp_path, b)._request("transcribe", [b"crop"], SPEC)[0]  # fmt: skip


def document_setup(root, monkeypatch):
    path = root / "silver/claims/C/r1/01.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(SUB))
    monkeypatch.setattr(verify, "load_layout", lambda *args: SimpleNamespace(material_facts=[]))


def test_refused_route_persists_visible_unverified_record(tmp_path, monkeypatch):
    document_setup(tmp_path, monkeypatch)
    rec = verify.verify_document(tmp_path, "C", "r1", "01", RoutedWitness(tmp_path), Path("x"))
    assert not rec["results"][0]["verified"] and rec["routing_refused"]
    assert rec["findings"][0]["kind"] == "witness_routing_refused" and rec["assurance"] == "W1"


def fake_verification(root, case, run, doc, port, pdf):
    reply = port.ask("transcribe", [b"crop"], SPEC)
    ok = reply.data["crops"][0]["reread"]["values"][0]["value"] == "25%"
    return {"document_id": "C/01", "model": reply.model,
            "witness_family": port.reader["reader_family"]
            if port.stratum == "B" else "google", "request": reply.request_hash,
            "assurance": "W1" if port.stratum == "B" else "W2", "findings": [],
            "coverage": [],
            "crops": reply.data["crops"],
            "results": [{"claim_id": "c", "verified": ok}]}  # fmt: skip


def test_b_flags_records_results_summary_source_and_frozen_factset_findings(tmp_path, monkeypatch):
    document_setup(tmp_path, monkeypatch)
    attest(tmp_path)
    monkeypatch.setattr(verify, "_verify_document", fake_verification)
    rec = verify.verify_document(tmp_path, "C", "r1", "01",
                                RoutedWitness(tmp_path, factory=Transport), Path("x"))  # fmt: skip
    assert rec["self_referential"] and rec["results"][0]["self_referential"]
    assert rec["assurance"] == "W1" and rec["results"][0]["verified"]
    frozen = _verification(DocInput("01", "x", [], {}, assurance=rec["assurance"], isolated=True,
                                   findings=[f["kind"] for f in rec["findings"]]))  # fmt: skip
    assert frozen.assurance == "W1" and "self_referential_verification" in frozen.findings


@pytest.mark.parametrize("agree", [True, False])
def test_ab_keeps_both_records_and_requires_both_verdicts(tmp_path, monkeypatch, agree):
    document_setup(tmp_path, monkeypatch)
    attest(tmp_path)
    monkeypatch.setattr(verify, "_verify_document", fake_verification)

    def factory(backend):
        data = copy.deepcopy(DATA)
        if not agree and backend.startswith("vertex-gcp:"):
            data["crops"][0]["reread"]["values"][0]["value"] = "68%"
            data["crops"][0]["transcription"] = "Irene 68%"
        return Transport(backend, data=data)

    rec = verify.verify_document(tmp_path, "C", "r1", "01",
        RoutedWitness(tmp_path, config=Config("A+B"), factory=factory), Path("x"))  # fmt: skip
    assert set(rec["strata"]) == {"A", "B"}
    assert rec["strata"]["B"]["results"][0]["self_referential"]
    assert rec["results"][0]["verified"] == agree
    assert any(f["kind"] == "witness_strata_disagreement" for f in rec["findings"]) != agree


def test_blind_request_and_schema_reject_hidden_claims():
    with pytest.raises(ValueError, match="non-blind"):
        adapters.request("transcribe", {**SPEC, "claim": "Irene 46%"})
    with pytest.raises(ValueError, match="positional"):
        adapters.request("transcribe", {"crop_ids": ["Irene-50"]})
    schema = adapters.request("transcribe", SPEC)[1]
    adapters.validate(DATA, schema)
    with pytest.raises(ValueError, match="closed schema"):
        adapters.validate({**DATA, "status": "APPROVE"}, schema)


def test_unprovable_cli_routes_fail_without_spending():
    for cls, model in [(adapters.CodexWitness, "gpt-6.1-sol"),
                       (adapters.AgyWitness, "gemini-3.8-flash-high")]:  # fmt: skip
        w = cls(model)
        with pytest.raises(WitnessRefused, match="served"):
            w.call("transcribe", [b"crop"], SPEC)
        assert w.last_provenance["reported_model"] is None


def test_claude_launch_plumbing_has_no_data_tools_or_parent_secrets(monkeypatch):
    """Offline process fake checks arguments/shape only; it is not live confinement proof."""
    monkeypatch.setenv("GEMINI_API_KEY", "must-not-inherit")
    monkeypatch.setenv("KYB_SECRET_CANARY", "must-not-inherit")
    seen = []

    def run(cmd, **kwargs):
        seen.append((cmd, kwargs))
        assert sorted(p.name for p in Path(kwargs["cwd"]).iterdir()) == ["k1.png"]
        assert "GEMINI_API_KEY" not in kwargs["env"] and "KYB_SECRET_CANARY" not in kwargs["env"]
        message = json.loads(kwargs["input"])
        assert message["message"]["content"][1]["type"] == "image"
        events = [
            {"type": "system", "subtype": "init", "tools": []},
            {"type": "assistant", "message": {"model": "claude-opus-5-5", "content": []}},
            {"type": "result", "subtype": "success", "is_error": False, "structured_output": DATA},
        ]
        return SimpleNamespace(returncode=0, stdout="\n".join(map(json.dumps, events)), stderr="")

    monkeypatch.setattr(adapters.subprocess, "run", run)
    w = adapters.ClaudeWitness("claude-opus-5-5")
    assert w.call("transcribe", [b"crop"], SPEC)[0] == DATA
    cmd = seen[0][0]
    assert cmd[cmd.index("--tools") + 1] == "" and "--restricted" in cmd
    assert "--safe-mode" in cmd and "--no-session-persistence" in cmd
    assert w.last_provenance["no_data_tools"] and w.last_provenance["reported_model"]


def gate_document(stratum="B"):
    return {"claims": 1, "verified": 1, "context": "isolated", "context_now": "isolated",
            "reader_model": "claude-opus-5-5", "witness_model": "claude-opus-5-5",
            "reported_model": "claude-opus-5-5", "assurance": "W1", "stratum": stratum,
            "self_referential": True, "self_referential_claims": True,
            "findings": [], "routing_refused": False}  # fmt: skip


def dogfood_module():
    path = Path(__file__).resolve().parents[1] / "scripts/dogfood.py"
    spec = importlib.util.spec_from_file_location("dogfood_strata", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_b_dogfood_requires_actual_eligible_attestation_and_every_self_referential_flag():
    _gate = dogfood_module()._gate

    doc = gate_document()
    workspace = {"workspace": "ws", "next": "decide", "audit_ok": True,
                 "stratum": "B", "documents": {"01": doc}}  # fmt: skip
    agent = {"exit": 0, "subtype": "success"}
    assert not _gate("claude", 1, workspace, agent)
    for change in [
        {"context": "unattested"},
        {"context_now": "unattested"},
        {"reader_model": None},
        {"reader_model": "mystery"},
        {"witness_model": "claude-sonnet-5-5"},
        {"reported_model": None},
        {"assurance": "W2"},
        {"self_referential": False},
        {"self_referential_claims": False},
        {"routing_refused": True},
    ]:
        workspace["documents"]["01"] = {**doc, **change}
        assert _gate("claude", 1, workspace, agent), change
    workspace["documents"]["01"] = {**doc, "context": "unattested"}
    assert _gate("codex", 1, workspace, agent)  # no unattested Codex exception for spec


def test_a_and_ab_dogfood_require_actual_cross_family_and_valid_agreeing_b():
    _stratum_evidence = dogfood_module()._stratum_evidence

    a = {**gate_document("A"), "witness_model": "gemini-3.1-pro-preview",
         "reported_model": "gemini-3.1-pro-preview", "assurance": "W2",
         "self_referential": False}  # fmt: skip
    assert _stratum_evidence("A", a)
    assert not _stratum_evidence("A", {**a, "witness_model": "claude-opus-5-5",
                                       "reported_model": "claude-opus-5-5"})  # fmt: skip
    ab = {**a, "stratum": "A+B", "strata": {
        "A": {"model": a["witness_model"], "reported_model": a["reported_model"],
              "assurance": "W2"},
        "B": {"model": a["reader_model"], "reported_model": a["reader_model"],
              "assurance": "W1", "self_referential": True,
              "self_referential_claims": True}}}  # fmt: skip
    assert _stratum_evidence("A+B", ab)
    assert not _stratum_evidence("A+B", {**ab, "strata": {}})
    assert not _stratum_evidence("A+B", {**ab, "findings": ["witness_strata_disagreement"]})
    for key in (
        "model",
        "reported_model",
        "self_referential",
        "self_referential_claims",
        "assurance",
    ):
        broken = copy.deepcopy(ab)
        broken["strata"]["B"][key] = None
        assert not _stratum_evidence("A+B", broken), key


def test_actual_binding_pipeline_keeps_b_self_reference_without_reader_claim_leak(tmp_path):
    """Use a frozen real PDF/layout through render, transcription matching and bind-v7.

    The offline transcription is explicit fixture data, not proof of a live model's reading.
    """
    import shutil

    assets = Path(__file__).resolve().parents[1] / "examples"
    source = Path(__file__).resolve().parents[1] / ("tests/fixtures/witness_b05")
    for part in ("claims", "layout", "pages"):
        shutil.copytree(source / "silver" / part, tmp_path / "silver" / part)
    path = tmp_path / "silver/claims/es-iberia-software/r1/05.json"
    sub = json.loads(path.read_text())
    claim = next(c for c in sub["claims"] if c["claim_id"] == "c-B05-08")
    sub["claims"] = [claim]
    sub["produced_by"] = SUB["produced_by"]
    path.write_text(json.dumps(sub))
    attest(tmp_path)
    data = copy.deepcopy(DATA)
    data["crops"][0]["transcription"] = "Lucas Navarro Solís 68%"
    data["crops"][0]["reread"]["values"][0]["subject"] = "Lucas Navarro Solís"
    data["crops"][0]["reread"]["values"][0]["value"] = "68%"
    data["crops"].append({"crop_id": "k2", "transcription": "pendiente", "readable": True,
                          "reread": {"values": []}})  # fmt: skip
    made = []

    def factory(backend):
        made.append(Transport(backend, data=data))
        return made[-1]

    pdf = next((assets / "packets/es-iberia-software").glob("05*.pdf"))
    rec = verify.verify_document(tmp_path, "es-iberia-software", "r1", "05",
                                 RoutedWitness(tmp_path, factory=factory), pdf)  # fmt: skip
    assert rec["results"][0]["verified"] and rec["results"][0]["self_referential"]
    assert rec["assurance"] == "W1" and rec["model"] == "claude-opus-5-5"
    assert made[0].calls[0][2] == {"crop_ids": ["k1", "k2"]}


def test_v7_replay_oracle_refuses_future_versions_datasets_and_any_changed_baseline(monkeypatch):
    from kyb.evals.witness import fullrun

    for rd in fullrun.runs():
        old = fullrun.recorded_verdicts(rd)
        assert fullrun.expected_verdicts(rd, binding="bind-v7") == fullrun.expected_verdicts(
            rd, binding="bind-v6"
        )
        with pytest.raises(ValueError, match="no reviewed"):
            fullrun.expected_verdicts(rd, binding="bind-v9")
        for claim in old:  # every baseline position is pinned, beyond the named gains
            with monkeypatch.context() as mp:
                changed = {**old, claim: not old[claim]}
                mp.setattr(fullrun, "recorded_verdicts", lambda _, changed=changed: changed)
                with pytest.raises(ValueError, match="baseline changed"):
                    fullrun.expected_verdicts(rd, binding="bind-v7")
        with monkeypatch.context() as mp:
            mp.setattr(fullrun, "recorded_verdicts", lambda _, old=old: old)
            with pytest.raises(ValueError, match="no reviewed"):
                fullrun.expected_verdicts(Path("unreviewed-dataset"), binding="bind-v7")


@pytest.mark.parametrize("defect", ["Read", "called_Read", "missing_model", "schema"])
def test_r1_real_claude_parser_refusals_are_terminal_and_recorded(tmp_path, monkeypatch, defect):
    """Offline process events exercise actual parser+port+router; not live containment proof."""
    attest(tmp_path, "gpt-6.1-sol")
    calls, fallback = [], []

    def run(cmd, **kwargs):
        calls.append(cmd)
        msg = {"model": "claude-opus-5-5", "content": []}
        if defect == "missing_model":
            msg.pop("model")
        if defect == "called_Read":
            msg["content"] = [{"type": "tool_use", "name": "Read"}]
        events = [
            {"type": "system", "subtype": "init", "tools": ["Read"] if defect == "Read" else []},
            {"type": "assistant", "message": msg},
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "structured_output": {} if defect == "schema" else DATA,
            },
        ]
        return SimpleNamespace(returncode=0, stdout="\n".join(map(json.dumps, events)), stderr="")

    def factory(backend):
        if backend.startswith("claude-p:"):
            return adapters.ClaudeWitness("claude-opus-5-5")
        fallback.append(backend)
        return Transport(backend)

    monkeypatch.setattr(adapters.subprocess, "run", run)
    router = RoutedWitness(
        tmp_path,
        config=Config("A", ("claude-p:claude-opus-5-5", "gemini-api:gemini-3.1-pro-preview")),
        factory=factory,
    )
    for mode in ("auto", "replay"):
        router.mode = mode  # a refusal retains its terminal class during zero-call replay too
        with pytest.raises(WitnessRefused):
            router.for_document(SUB)["A"].ask("transcribe", [b"crop"], SPEC)
    assert len(calls) == 1 and not fallback
    raw = json.loads(next((tmp_path / "silver/witness/raw").glob("*/failed/0.json")).read_text())
    assert raw["failure_kind"] == "refused" and raw["status"] == "failed"


def test_r1_deleting_new_response_integrity_hash_is_refused(tmp_path):
    fake = Transport("gemini-api:gemini-3.1-pro-preview")
    port = Witness(tmp_path, fake)
    port.ask("transcribe", [b"crop"], SPEC)
    path = next((tmp_path / "silver/witness/raw").glob("*/0.json"))
    raw = json.loads(path.read_text())
    raw.pop("content_hash")
    raw["data"]["crops"][0]["transcription"] = "tampered"
    path.write_text(json.dumps(raw))
    with pytest.raises(WitnessRefused, match="missing its content hash"):
        Witness(tmp_path, fake, mode="replay").ask("transcribe", [b"crop"], SPEC)
    assert len(fake.calls) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("annotations", ["pending transfer"]),
        ("struck", ["25%"]),
        ("values", [{"subject": "Irene", "attribute": "pct", "value": "25%", "struck": True}]),
    ],
)
def test_r1_ab_structured_disagreements_survive_identical_text_and_verdicts(
    tmp_path, monkeypatch, field, value
):
    document_setup(tmp_path, monkeypatch)
    attest(tmp_path)
    monkeypatch.setattr(verify, "_verify_document", fake_verification)

    def factory(backend):
        data = copy.deepcopy(DATA)
        if backend.startswith("vertex-gcp:"):
            data["crops"][0]["reread"][field] = value
        return Transport(backend, data=data)

    rec = verify.verify_document(tmp_path, "C", "r1", "01",
        RoutedWitness(tmp_path, config=Config("A+B"), factory=factory), Path("x"))  # fmt: skip
    assert rec["results"][0]["verified"]  # both independently bound; finding still vetoes approval
    assert any(f["kind"] == "witness_strata_disagreement" for f in rec["findings"])
    frozen = _verification(DocInput("01", "x", [], {}, assurance="W2", isolated=True,
                                   findings=[f["kind"] for f in rec["findings"]]))  # fmt: skip
    assert "witness_strata_disagreement" in frozen.findings


@pytest.mark.parametrize(
    "previous",
    [
        "strata-v1",
        "strata-v2",
        "strata-v3",
        "strata-v4",
        "strata-v5",
        "strata-v5.1",
        "strata-future",
        None,
    ],
)
def test_r1_old_routed_protocol_cannot_claim_current_assurance(tmp_path, monkeypatch, previous):
    document_setup(tmp_path, monkeypatch)
    path = tmp_path / "silver/witness/C/r1/01.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"stratum": "B", "routing_config": {"version": previous},
                               "assurance": "W1", "witness_family": "anthropic"}))  # fmt: skip
    with pytest.raises(ValueError, match="new run"):
        verify.assurance_now(tmp_path, "C", "r1", "01")
    assert ROUTING_VERSION == "strata-v6"


@pytest.mark.parametrize("field", ["failure_kind", "failure_type", "status"])
def test_r1_removed_typed_failure_status_cannot_become_fallback(tmp_path, field):
    from kyb.util.hashing import sha256_json

    attest(tmp_path)
    made = []

    def factory(backend):
        made.append(Transport(backend, fail="pro" in backend))
        return made[-1]

    first = Transport("vertex-gcp:gemini-3.1-pro-preview", fail=True)
    with pytest.raises(WitnessUnavailable):
        Witness(tmp_path, first).ask("transcribe", [b"crop"], SPEC)
    path = next((tmp_path / "silver/witness/raw").glob("*/failed/0.json"))
    raw = json.loads(path.read_text())
    raw.pop(field)
    raw.pop("content_hash")
    raw["content_hash"] = sha256_json(raw)  # even a valid checksum cannot replace typed evidence
    path.write_text(json.dumps(raw))
    route = RoutedWitness(tmp_path, config=Config("A"), factory=factory).for_document(SUB)["A"]
    with pytest.raises(WitnessRefused, match="typed refusal status|required cache"):
        route.ask("transcribe", [b"crop"], SPEC)
    assert len(made) == 1 and not made[0].calls


def claude_stream_case(defect=None, ending="success"):
    events = [
        {"type": "system", "subtype": "init", "tools": ["StructuredOutput"]},
        {"type": "assistant", "message": {"model": "claude-opus-5-5", "content": []}},
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "structured_output": copy.deepcopy(DATA),
        },
    ]
    if defect == "missing_model":
        events[1]["message"].pop("model")
    elif defect == "mixed_model":
        events.insert(
            2, {"type": "assistant", "message": {"model": "claude-sonnet-5-5", "content": []}}
        )
    elif defect == "tools":
        events[0]["tools"] = ["Read"]
        events[1]["message"]["content"] = [{"type": "tool_use", "name": "Read"}]
    elif defect == "advertised_only":
        events[0]["tools"] = ["Read"]
    elif defect == "called_only":
        events[1]["message"]["content"] = [{"type": "tool_use", "name": "Read"}]
    elif defect == "unknown_model":
        events[1]["message"]["model"] = "unknown-model"
    elif defect == "wrong_family":
        events[1]["message"]["model"] = "gpt-6.1-sol"
    elif defect == "message_shape":
        events[1]["message"] = []
    elif defect == "content_shape":
        events[1]["message"]["content"] = "not a block list"
    elif defect == "block_shape":
        events[1]["message"]["content"] = [None]
    elif defect == "schema":
        events[-1]["structured_output"] = {}
    elif defect == "schema_tool":
        events[1]["message"]["content"] = [{"type": "tool_use", "name": "StructuredOutput",
                                              "input": {}}]  # fmt: skip
    elif defect == "non_object":
        events.insert(2, ["not", "an", "event"])
    elif defect == "duplicate_init":
        events.insert(1, copy.deepcopy(events[0]))
    elif defect == "duplicate_result":
        events.append(copy.deepcopy(events[-1]))
    elif defect == "permission_denial":
        events[-1]["permission_denials"] = [{"tool_name": "Read"}]
    elif defect == "result_stop_reason":
        events[-1]["stop_reason"] = "refusal"
    elif defect == "result_status_missing":
        events[-1].pop("is_error")
    if ending in {"max_turns", "provider_error"}:
        events[-1] |= {"is_error": True, "subtype": "error_max_turns"}
        if ending == "provider_error":
            events[-1] |= {"subtype": "error_during_execution", "api_error_status": 429}
    if ending == "no_result_exit":
        events = [e for e in events if not isinstance(e, dict) or e.get("type") != "result"]
    if defect == "result_status_missing":
        for event in events:
            if isinstance(event, dict) and event.get("type") == "result":
                event.pop("is_error", None)
    output = "\n".join(map(json.dumps, events))
    if ending in {"truncated_exit", "truncated"} or defect == "malformed":
        output += '\n{"type":'
    if defect == "duplicate_json_key":
        output += '\n{"type":"system","subtype":"init","tools":["Read"],"tools":[]}'
    if defect == "invalid_unicode_escape":
        output += "\n" + json.dumps({"type": "system", "subtype": chr(0xD800)})
    if defect == "deep_json":
        output += "\n" + "[" * 1500 + "0" + "]" * 1500
    if defect == "invalid_utf8":
        output = output.encode() + b"\n\xff"
    return output, (
        "timeout"
        if ending == "timeout"
        else 1
        if ending in {"exit", "truncated_exit", "no_result_exit"}
        else 0
    )


def parser_route(root, monkeypatch, output, returncode, *, diagnostic=""):
    """Actual Claude parser -> recording port -> A router -> eligible actual GPT reader."""
    attest(root, "gpt-6.1-sol")
    calls, downstream = [], []

    def process(cmd, **kwargs):
        calls.append(cmd)
        if returncode == "timeout":
            raise adapters.subprocess.TimeoutExpired(cmd, 1, output=output, stderr=diagnostic)
        if isinstance(returncode, Exception):
            raise returncode
        return SimpleNamespace(returncode=returncode, stdout=output, stderr=diagnostic)

    def factory(backend):
        if backend.startswith("claude-p:"):
            return adapters.ClaudeWitness("claude-opus-5-5")
        candidate = Transport(backend)
        downstream.append(candidate)
        return candidate

    monkeypatch.setattr(adapters.subprocess, "run", process)
    router = RoutedWitness(
        root,
        config=Config("A", ("claude-p:claude-opus-5-5", "gemini-api:gemini-3.1-pro-preview")),
        factory=factory,
    )
    return router, calls, downstream


@pytest.mark.parametrize(
    ("defect", "ending"),
    [
        ("missing_model", "exit"),
        ("mixed_model", "max_turns"),
        ("tools", "truncated_exit"),
    ],
)
def test_r2_exact_combined_refusal_wins_over_process_provider_and_trailer(
    tmp_path, monkeypatch, defect, ending
):
    router, calls, downstream = parser_route(
        tmp_path, monkeypatch, *claude_stream_case(defect, ending)
    )
    for mode in ("auto", "replay"):
        router.mode = mode
        with pytest.raises(WitnessRefused):
            router.for_document(SUB)["A"].ask("transcribe", [b"crop"], SPEC)
    assert len(calls) == 1 and not downstream
    raw = json.loads(next((tmp_path / "silver/witness/raw").glob("*/failed/0.json")).read_text())
    assert raw["failure_kind"] == "refused" and raw["failure_type"] == "WitnessRefused"


@pytest.mark.parametrize(
    "defect",
    [
        "missing_model",
        "mixed_model",
        "unknown_model",
        "wrong_family",
        "tools",
        "advertised_only",
        "called_only",
        "message_shape",
        "content_shape",
        "block_shape",
        "schema",
        "schema_tool",
        "non_object",
        "duplicate_init",
        "duplicate_result",
        "permission_denial",
        "result_stop_reason",
        "malformed",
        "duplicate_json_key",
        "invalid_utf8",
        "result_status_missing",
        "invalid_unicode_escape",
        "deep_json",
    ],
)
@pytest.mark.parametrize(
    "ending",
    [
        "success",
        "exit",
        "max_turns",
        "provider_error",
        "truncated_exit",
        "no_result_exit",
        "timeout",
    ],
)
def test_r2_refusal_evidence_dominates_error_crossproduct(tmp_path, monkeypatch, defect, ending):
    """Bounded adversarial wire cases through actual parser/port/router, not confinement proof."""
    router, calls, downstream = parser_route(
        tmp_path, monkeypatch, *claude_stream_case(defect, ending)
    )
    for mode in ("auto", "replay"):
        router.mode = mode
        with pytest.raises(WitnessRefused):
            router.for_document(SUB)["A"].ask("transcribe", [b"crop"], SPEC)
    assert len(calls) == 1 and not downstream
    raw = json.loads(next((tmp_path / "silver/witness/raw").glob("*/failed/0.json")).read_text())
    assert raw["failure_kind"] == "refused" and raw["provenance"]["refusal_evidence"]
    assert raw["record_version"] == raw["validation_version"] == "strata-v6"
    if defect == "tools":
        assert raw["provenance"]["tools"] == raw["provenance"]["called_tools"] == ["Read"]
        assert not raw["provenance"]["no_data_tools"]
    if defect == "mixed_model":
        assert raw["provenance"]["reported_models"] == ["claude-opus-5-5", "claude-sonnet-5-5"]
    if defect in {"schema", "schema_tool"} and not (
        defect == "schema" and ending == "no_result_exit"
    ):
        assert any(not c["valid"] for c in raw["provenance"]["schema_checks"])


def no_answer_stream(*, code="rate_limit", status=None):
    events = [{"type": "system", "subtype": "init", "tools": ["StructuredOutput"]}]
    if code is not None:
        events.append({"type": "assistant", "error": code,
                       "message": {"model": "<synthetic>", "content": []}})  # fmt: skip
    events.append(
        {
            "type": "result",
            "is_error": True,
            "subtype": "success" if status is not None else "error_during_execution",
        }
    )
    if status is not None:
        events[-1]["api_error_status"] = status
    return "\n".join(map(json.dumps, events))


@pytest.mark.parametrize(
    ("output", "returncode"),
    [
        (b"", 1),
        (b" \n", 2),
        (b"", "timeout"),
        (b"", FileNotFoundError("missing executable")),
        (no_answer_stream(), 0),
        (no_answer_stream(code="server_error"), 1),
        (no_answer_stream(code=None, status=429), 0),
    ],
)
def test_r2_only_genuine_no_answer_availability_falls_back_and_replays(
    tmp_path, monkeypatch, output, returncode
):
    router, calls, downstream = parser_route(tmp_path, monkeypatch, output, returncode)
    for mode in ("auto", "replay"):
        router.mode = mode
        route = router.for_document(SUB)["A"]
        reply = route.ask("transcribe", [b"crop"], SPEC)
        assert reply.model == "gemini-3.1-pro-preview"
        assert [a["status"] for a in route.attempts] == ["unavailable", "complete"]
        assert reply.replayed == (mode == "replay")
    assert len(calls) == 1 and sum(len(c.calls) for c in downstream) == 1
    raw = json.loads(next((tmp_path / "silver/witness/raw").glob("*/failed/0.json")).read_text())
    assert raw["failure_kind"] == "unavailable" and not raw["provenance"]["refusal_evidence"]
    assert raw["provenance"]["reported_model"] is None


@pytest.mark.parametrize(
    "diagnostic", ["error: unknown option --tools", "unknown failure", b"\xff"]
)
@pytest.mark.parametrize("returncode", [1, "timeout"])
def test_r2_diagnostic_without_response_is_not_an_empty_transport_failure(
    tmp_path, monkeypatch, diagnostic, returncode
):
    router, calls, downstream = parser_route(
        tmp_path, monkeypatch, "", returncode, diagnostic=diagnostic
    )
    for mode in ("auto", "replay"):
        router.mode = mode
        with pytest.raises(WitnessRefused):
            router.for_document(SUB)["A"].ask("transcribe", [b"crop"], SPEC)
    assert len(calls) == 1 and not downstream


@pytest.mark.parametrize("reason", ["refusal", "max_tokens", "tool_deferred", "unknown", False, {}])
@pytest.mark.parametrize("ending", ["status", "provider", "max_turns"])
def test_r2_final_stop_reason_cannot_be_hidden_by_no_answer_error(
    tmp_path, monkeypatch, reason, ending
):
    output = no_answer_stream(code=None, status=429) if ending == "status" else no_answer_stream()
    events = [json.loads(line) for line in output.splitlines()]
    events[-1]["stop_reason"] = reason
    if ending == "max_turns":
        events[-1]["subtype"] = "error_max_turns"
    router, calls, downstream = parser_route(
        tmp_path, monkeypatch, "\n".join(map(json.dumps, events)), 1
    )
    for mode in ("auto", "replay"):
        router.mode = mode
        with pytest.raises(WitnessRefused):
            router.for_document(SUB)["A"].ask("transcribe", [b"crop"], SPEC)
    assert len(calls) == 1 and not downstream
    raw = json.loads(next((tmp_path / "silver/witness/raw").glob("*/failed/0.json")).read_text())
    assert raw["failure_kind"] == "refused"
    assert "result_incomplete_or_refused" in raw["provenance"]["refusal_evidence"]


@pytest.mark.parametrize("reason", ["end_turn", "tool_use"])
def test_r2_completion_stop_without_assistant_is_not_no_answer_availability(
    tmp_path, monkeypatch, reason
):
    events = [json.loads(line) for line in no_answer_stream(code=None, status=429).splitlines()]
    events[-1]["stop_reason"] = reason
    router, calls, downstream = parser_route(
        tmp_path, monkeypatch, "\n".join(map(json.dumps, events)), 1
    )
    for mode in ("auto", "replay"):
        router.mode = mode
        with pytest.raises(WitnessRefused):
            router.for_document(SUB)["A"].ask("transcribe", [b"crop"], SPEC)
    assert len(calls) == 1 and not downstream


@pytest.mark.parametrize(
    ("output", "returncode"),
    [
        (b"", 0),
        (None, 1),
        ("arbitrary error text", 1),
        (claude_stream_case()[0], 1),
        (claude_stream_case()[0], "timeout"),
        (no_answer_stream(code="unknown"), 1),
        (no_answer_stream(code="max_output_tokens"), 1),
        ("", RuntimeError("unknown cause")),
        ("", OSError("unknown OS failure")),
        ('{"type":"system","subtype":"init","tools":[]}', 1),
    ],
)
def test_r2_ambiguous_partial_and_unknown_failures_never_fall_back(
    tmp_path, monkeypatch, output, returncode
):
    router, calls, downstream = parser_route(tmp_path, monkeypatch, output, returncode)
    for mode in ("auto", "replay"):
        router.mode = mode
        with pytest.raises(WitnessRefused):
            router.for_document(SUB)["A"].ask("transcribe", [b"crop"], SPEC)
    assert len(calls) == 1 and not downstream


@pytest.mark.parametrize("stop_reason", [None, "end_turn", "tool_use"])
def test_r2_valid_complete_schema_stream_is_recorded_and_replayed(
    tmp_path, monkeypatch, stop_reason
):
    events = [
        {"type": "system", "subtype": "init", "tools": ["StructuredOutput"]},
        {
            "type": "assistant",
            "message": {
                "model": "claude-opus-5-5",
                "content": [
                    {"type": "tool_use", "id": "schema1", "name": "StructuredOutput", "input": DATA}
                ],
            },
        },
        {
            "type": "user",
            "message": {
                "content": [
                    {"type": "tool_result", "tool_use_id": "schema1", "content": "accepted"}
                ]
            },
        },
        {
            "type": "assistant",
            "message": {"model": "claude-opus-5-5", "content": [{"type": "text", "text": "done"}]},
        },
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "stop_reason": stop_reason,
            "structured_output": DATA,
            "total_cost_usd": 0.003,
            "usage": {"input_tokens": 3, "cache_read_input_tokens": 7, "output_tokens": 10},
        },
    ]
    router, calls, downstream = parser_route(
        tmp_path, monkeypatch, "\n".join(map(json.dumps, events)), 0
    )
    for mode in ("auto", "replay"):
        router.mode = mode
        reply = router.for_document(SUB)["A"].ask("transcribe", [b"crop"], SPEC)
        assert reply.model == "claude-opus-5-5" and reply.data == DATA
        assert reply.usage == {
            "promptTokenCount": 10,
            "candidatesTokenCount": 10,
            "cost_usd": 0.003,
        }
        assert reply.provenance["no_data_tools"] and len(reply.provenance["schema_checks"]) == 2
        assert reply.provenance["validation_version"] == "strata-v6"
    assert len(calls) == 1 and not downstream


@pytest.mark.parametrize(
    "defect",
    [
        "missing_stderr",
        "stderr_response",
        "unknown_exception_prefix",
        "unmatched_tool_result",
        "late_tools",
        "nested_assistant",
    ],
)
def test_r2_other_emitted_evidence_cannot_be_hidden_by_error_path(tmp_path, monkeypatch, defect):
    output, returncode = claude_stream_case("tools", "exit")
    diagnostic = ""
    if defect == "missing_stderr":
        diagnostic = None
    elif defect == "stderr_response":
        diagnostic, output = output, ""
    elif defect == "unknown_exception_prefix":
        error = RuntimeError("unknown failure")
        error.output, error.stderr = output, ""
        returncode = error
    else:
        events = [json.loads(line) for line in no_answer_stream(code=None, status=429).splitlines()]
        if defect == "unmatched_tool_result":
            events.insert(
                1,
                {
                    "type": "user",
                    "message": {
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "unseen-read",
                                "content": "redacted",
                            }
                        ]
                    },
                },
            )
        elif defect == "late_tools":
            events.insert(1, {"type": "system", "subtype": "status", "tools": ["Read"]})
        else:
            events.insert(
                1,
                {
                    "type": "assistant",
                    "parent_tool_use_id": "foreign",
                    "message": {"model": "claude-opus-5-5", "content": []},
                },
            )
        output = "\n".join(map(json.dumps, events))
    router, calls, downstream = parser_route(
        tmp_path, monkeypatch, output, returncode, diagnostic=diagnostic
    )
    for mode in ("auto", "replay"):
        router.mode = mode
        with pytest.raises(WitnessRefused):
            router.for_document(SUB)["A"].ask("transcribe", [b"crop"], SPEC)
    assert len(calls) == 1 and not downstream
    raw = json.loads(next((tmp_path / "silver/witness/raw").glob("*/failed/0.json")).read_text())
    if defect in {"missing_stderr", "stderr_response", "unknown_exception_prefix"}:
        assert "Read" in raw["provenance"]["tools"]
        assert "Read" in raw["provenance"]["called_tools"]


@pytest.mark.parametrize(
    "previous", ["strata-v1", "strata-v2", "strata-v3", "strata-v4", "strata-v5", "strata-v5.1"]
)
def test_r2_prior_unsafe_raw_chain_never_becomes_current(tmp_path, monkeypatch, previous):
    from kyb.witness import port

    # Seed the old failure -> complete cache chain, never relabel or bless its records.
    with monkeypatch.context() as mp:
        mp.setattr(port, "ROUTING_VERSION", previous)
        for backend, fail in [
            ("claude-p:claude-opus-5-5", True),
            ("gemini-api:gemini-3.1-pro-preview", False),
        ]:
            old = Transport(backend, fail=fail)
            old.validation_version = previous
            if fail:
                with pytest.raises(WitnessUnavailable):
                    Witness(tmp_path, old).ask("transcribe", [b"crop"], SPEC)
            else:
                Witness(tmp_path, old).ask("transcribe", [b"crop"], SPEC)
    router, calls, downstream = parser_route(
        tmp_path, monkeypatch, *claude_stream_case("missing_model", "exit")
    )
    router.mode = "replay"
    with pytest.raises(WitnessRefused):
        router.for_document(SUB)["A"].ask("transcribe", [b"crop"], SPEC)
    assert not calls and not any(c.calls for c in downstream)
    router.mode = "auto"
    with pytest.raises(WitnessRefused):
        router.for_document(SUB)["A"].ask("transcribe", [b"crop"], SPEC)
    assert len(calls) == 1 and not any(c.calls for c in downstream)
    records = [json.loads(p.read_text()) for p in (tmp_path / "silver/witness/raw").rglob("*.json")]
    assert any(
        r["record_version"] == previous and r.get("failure_kind") == "unavailable" for r in records
    )
    assert any(
        r["record_version"] == "strata-v6" and r.get("failure_kind") == "refused" for r in records
    )


@pytest.mark.parametrize("field", ["record_version", "validation_version"])
def test_r2_raw_validation_identity_is_required_for_complete_and_failed_records(tmp_path, field):
    from kyb.util.hashing import sha256_json

    for fail in (False, True):
        root = tmp_path / str(fail)
        candidate = Transport("gemini-api:gemini-3.1-pro-preview", fail=fail)
        if fail:
            with pytest.raises(WitnessUnavailable):
                Witness(root, candidate).ask("transcribe", [b"crop"], SPEC)
        else:
            Witness(root, candidate).ask("transcribe", [b"crop"], SPEC)
        path = next((root / "silver/witness/raw").rglob("0.json"))
        record = json.loads(path.read_text())
        record[field] = "strata-v2"
        record.pop("content_hash")
        record["content_hash"] = sha256_json(record)
        path.write_text(json.dumps(record))
        with pytest.raises(WitnessRefused, match="validation identity"):
            Witness(root, candidate, mode="replay").ask("transcribe", [b"crop"], SPEC)
        assert len(candidate.calls) == 1


@pytest.mark.parametrize("promoted_status", ["failed", "complete"])
def test_r2_recorded_observed_refusal_cannot_be_erased_by_status_flags(
    tmp_path, monkeypatch, promoted_status
):
    from kyb.util.hashing import sha256_json

    router, calls, downstream = parser_route(
        tmp_path, monkeypatch, *claude_stream_case("tools", "truncated_exit")
    )
    with pytest.raises(WitnessRefused):
        router.for_document(SUB)["A"].ask("transcribe", [b"crop"], SPEC)
    path = next((tmp_path / "silver/witness/raw").glob("*/failed/0.json"))
    record = json.loads(path.read_text())
    record |= {"failure_kind": "unavailable", "failure_type": "WitnessUnavailable",
               "status": promoted_status, "data": DATA}  # fmt: skip
    record["provenance"]["classification"] = "unavailable"
    record.pop("content_hash")
    record["content_hash"] = sha256_json(record)
    if promoted_status == "complete":
        path = path.parent.parent / "0.json"
    path.write_text(json.dumps(record))
    for mode in ("auto", "replay"):
        router.mode = mode
        with pytest.raises(WitnessRefused, match="remains terminal"):
            router.for_document(SUB)["A"].ask("transcribe", [b"crop"], SPEC)
    assert len(calls) == 1 and not downstream
