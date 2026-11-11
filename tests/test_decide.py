"""The L4 boundary: recorded inputs, isolated predicates, vetoes and decision identity."""

import json
from pathlib import Path

import pytest

from kyb.audit.chain import verify_workspace, workspace_action
from kyb.cli import main
from kyb.evals.cases import build, golden, variants
from kyb.run.action import load_work_order, run_case
from kyb.util.hashing import sha256_json
from kyb.workspace.init import init_workspace

POLICY = Path(__file__).resolve().parents[1] / "examples/policy/compliance_rules.md"


@pytest.fixture(autouse=True)
def offline_default_support(monkeypatch):
    # Fixture tests pin the spec unavailable state; they never spend a configured API key.
    # Support integration tests explicitly inject their recorded ports/checkers below.
    monkeypatch.setattr("kyb.decide.action.support_checker", lambda *a, **kw: None)


@pytest.fixture
def prepared(tmp_path):
    packet = build(golden()["C"])
    content_hash = sha256_json(packet.facts.model_dump(mode="json", exclude={"fact_set_hash"}))
    packet = type(packet)(
        packet.facts.model_copy(update={"fact_set_hash": content_hash}),
        packet.screening.model_copy(update={"fact_set_hash": content_hash}),
        packet.answers,
    )
    ws = tmp_path / "ws"
    init_workspace(ws, tenant="t", policy_src=POLICY)
    paths = []
    for name, model in (("facts", packet.facts), ("screening", packet.screening)):
        p = tmp_path / f"{name}.json"
        p.write_text(model.model_dump_json())
        paths.append(p)
    return ws, packet, paths


def test_run_cli_through_work_order(prepared, capsys):
    ws, packet, (facts, screen) = prepared
    assert (
        main(
            [
                "-w",
                str(ws),
                "run",
                packet.facts.case,
                "--run",
                "r1",
                "--facts-fixture",
                str(facts),
                "--screening-fixture",
                str(screen),
            ]
        )
        == 0
    )
    out = json.loads(capsys.readouterr().out)
    work = load_work_order(ws, out["work_order_id"])
    assert work.rules_version.startswith("2026.11-r1#sha256:") or work.rules_version.startswith(
        "2026.11-r1+sha256:"
    )
    assert {i.instance_id for i in work.instances if i.predicate == "same_party"} == {
        f"sp-{c.candidate_id}" for c in packet.screening.candidates
    }
    assert all(e.verified for e in work.evidence)
    assert verify_workspace(ws, at_boundary=True).ok
    with workspace_action(ws):
        assert (
            run_case(ws, packet.facts.case, "r1", facts_fixture=facts, screening_fixture=screen)
            == out
        )


def test_runner_refuses_mismatched_fixtures_before_work_order(prepared):
    ws, packet, (facts, screen) = prepared
    bad = packet.screening.model_copy(update={"fact_set_hash": "wrong"})
    screen.write_text(bad.model_dump_json())
    with workspace_action(ws), pytest.raises(ValueError, match="different facts"):
        run_case(ws, packet.facts.case, "r1", facts_fixture=facts, screening_fixture=screen)
    assert not list((ws / "gold/predicates").rglob("wo-*.json"))


def fixture_run(tmp_path, letter, packet=None):
    from kyb.evals.cases import Packet

    packet = packet or build(golden()[letter])
    content_hash = sha256_json(packet.facts.model_dump(mode="json", exclude={"fact_set_hash"}))
    packet = Packet(
        packet.facts.model_copy(update={"fact_set_hash": content_hash}),
        packet.screening.model_copy(update={"fact_set_hash": content_hash}),
        packet.answers,
    )
    ws = tmp_path / f"ws-{letter}"
    init_workspace(ws, tenant="t", policy_src=POLICY)
    paths = []
    for name, model in (("facts", packet.facts), ("screening", packet.screening)):
        path = tmp_path / f"{letter}-{name}.json"
        path.write_text(model.model_dump_json())
        paths.append(path)
    with workspace_action(ws):
        result = run_case(
            ws,
            packet.facts.case,
            packet.facts.run,
            facts_fixture=paths[0],
            screening_fixture=paths[1],
        )
    return ws, packet, load_work_order(ws, result["work_order_id"])


def submit_fixture(ws, packet, work):
    from kyb.predicates.submit import propose_predicates

    answers = []
    for i in work.instances:
        value = (
            packet.answers[i.instance_id].value if i.instance_id in packet.answers else "UNKNOWN"
        )
        answers.append(
            dict(
                instance_id=i.instance_id,
                value=value,
                confidence=0.9,
                evidence=i.evidence_universe,
                rationale="recorded fixture answer",
            )
        )
    args = {
        k: getattr(work, k)
        for k in (
            "work_order_id",
            "case",
            "run",
            "fact_set_hash",
            "rules_version",
            "catalog_version",
        )
    }
    with workspace_action(ws):
        propose_predicates(ws, {**args, "answers": answers})


@pytest.mark.parametrize("letter", list("ABC"))
def test_decide_cli_golden_and_byte_identical_replay(tmp_path, capsys, letter):
    from kyb.contracts.b4 import DecisionRecord

    ws, packet, work = fixture_run(tmp_path, letter)
    submit_fixture(ws, packet, work)
    command = ["-w", str(ws), "decide", packet.facts.case, "--run", "r1"]
    assert main(command) == 0
    result = DecisionRecord.model_validate_json(capsys.readouterr().out)
    expected = golden()[letter]["expected"]
    expected_status = "MANUAL_REVIEW" if letter == "A" else expected["status"]
    assert result.kyb_decision.status == expected_status
    assert result.kyb_decision.risk_score == expected["risk_score"]
    assert {r.rule_id for r in result.rule_results if r.fired == "yes"} == set(expected["fired"])
    assert result.signature_state == "pending_signature"
    assert main(command) == 0
    assert DecisionRecord.model_validate_json(capsys.readouterr().out) == result
    assert verify_workspace(ws, at_boundary=True).ok


def attested_fixture_answers(ws, packet, work, tmp_path):
    """Code-labelled answers through the actual hook/submission/usage boundary; no model calls."""
    from test_predicates import hook, predicates_agent

    from kyb.audit.chain import chain_events
    from kyb.predicates.submit import load_answers, propose_predicates
    from kyb.workspace.guard import pre_tool_use, subagent_stop

    if not work.instances:
        return {"answers": [], "sources": {}, "produced_by": {"predicates": None}}, chain_events(ws)
    payload = {
        k: getattr(work, k)
        for k in (
            "work_order_id",
            "case",
            "run",
            "fact_set_hash",
            "rules_version",
            "catalog_version",
        )
    }
    payload["answers"] = [
        {
            "instance_id": i.instance_id,
            "value": packet.answers[i.instance_id].value
            if i.instance_id in packet.answers
            else "UNKNOWN",
            "confidence": 0.95,
            "evidence": i.evidence_universe,
            "rationale": "code-generated evaluation label; not a live model answer",
        }
        for i in work.instances
    ]
    predicates_agent(ws, work)
    result = hook(
        ws,
        pre_tool_use,
        agent_type="kyb:kyb-predicates",
        agent_id="p1",
        tool_name="propose_predicates",
        tool_use_id="submit",
        tool_input=payload,
    )
    assert result[0] == 0
    with workspace_action(ws):
        assert (
            propose_predicates(ws, result[2]["hookSpecificOutput"]["updatedInput"])["context"]
            == "isolated"
        )
    transcript = tmp_path / "synthetic-predicates.jsonl"
    records = [
        {"type": "user", "message": {"content": f"kyb-predicates {work.work_order_id}"}},
        {
            "type": "assistant",
            "message": {
                "id": "labelled-turn",
                "model": "claude-opus-5-5",
                "content": [],
                "usage": {"input_tokens": 10, "output_tokens": 3},
            },
        },
    ]
    transcript.write_text("\n".join(json.dumps(r) for r in records))
    assert (
        hook(
            ws,
            subagent_stop,
            agent_type="kyb:kyb-predicates",
            agent_id="p1",
            agent_transcript_path=str(transcript),
        )[0]
        == 0
    )
    return load_answers(ws, work.work_order_id), chain_events(ws)


@pytest.mark.parametrize("label,packet", variants(), ids=[label["id"] for label, _ in variants()])
@pytest.mark.parametrize("support_mode", ["healthy", "absent", "outage"])
def test_synthetic_variants_exact_status_with_verified_attested_inputs(
    tmp_path, monkeypatch, label, packet, support_mode
):
    from datetime import UTC, datetime

    from kyb.contracts.b4 import DocVerification, PredicateAnswer
    from kyb.decide.core import decision_record
    from kyb.evals.cases import Packet
    from kyb.judges.core import fact_judges, outcome, predicate_judges
    from kyb.policy import load_policy

    policy = load_policy(POLICY)
    facts = packet.facts.model_copy(
        update={
            "verification": [
                DocVerification(doc=d.doc, assurance="W2", isolated=True)
                for d in packet.facts.documents
            ]
        }
    )
    ws, packet, work = fixture_run(
        tmp_path, "synthetic", Packet(facts, packet.screening, packet.answers)
    )
    proposal, events = attested_fixture_answers(ws, packet, work, tmp_path)
    support = (
        (lambda i, a, e: outcome("J-SUPPORT", i.instance_id)) if support_mode == "healthy" else None
    )
    if support_mode == "outage":
        from kyb.decide.support import answerer_resolver
        from kyb.judges.dispatch import support_checker

        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        support = support_checker(ws)
        support.work_order_id = work.work_order_id
        support.answerer_model = answerer_resolver(ws, work, proposal)
    with workspace_action(ws):
        answers, judges = predicate_judges(
            packet.facts,
            work,
            [PredicateAnswer.model_validate(a) for a in proposal["answers"]],
            who=proposal["produced_by"]["predicates"],
            events=events,
            sources=proposal["sources"],
            support=support,
        )
    assert all(j.verdict == "pass" for j in judges if j.judge_id == "J-PRED-ISOLATION")
    result = decision_record(
        packet.facts,
        packet.screening,
        policy,
        answers,
        [*judges, *fact_judges(packet.facts)],
        [],
        provenance={"labels": "code-generated evaluation"},
        emitted_at=datetime(2026, 10, 26, tzinfo=UTC),
        audit_trail=events,
    )
    expected = label["expected"]["status"]
    if (
        support_mode != "healthy"
        and expected == "APPROVED"
        and any(a["value"] != "UNKNOWN" for a in proposal["answers"])
    ):
        expected = "MANUAL_REVIEW"
    assert result.kyb_decision.status == expected, label["id"]
    if support_mode != "outage" and "risk_score" in label["expected"]:
        assert result.kyb_decision.risk_score == label["expected"]["risk_score"], label["id"]
    if support_mode != "outage" and "fired" in label["expected"]:
        assert {r.rule_id for r in result.rule_results if r.fired == "yes"} == set(
            label["expected"]["fired"]
        ), label["id"]


def test_fixture_screen_cannot_hide_sanctions_and_approve(tmp_path, monkeypatch):
    from datetime import UTC, datetime

    from kyb.contracts.b4 import DocVerification, PredicateAnswer
    from kyb.decide import action
    from kyb.decide.core import decision_record
    from kyb.evals.cases import Packet
    from kyb.judges.core import FIXTURE_INPUT, fact_judges, outcome
    from kyb.policy import load_policy

    _, packet = next(
        (label, p) for label, p in variants() if label["id"] == "syn-a-sanctions-corroborated"
    )
    facts = packet.facts.model_copy(
        update={
            "verification": [
                DocVerification(doc=d.doc, assurance="W2", isolated=True)
                for d in packet.facts.documents
            ]
        }
    )
    ws, packet, work = fixture_run(
        tmp_path,
        "forged",
        Packet(facts, packet.screening.model_copy(update={"candidates": []}), packet.answers),
    )
    proposal, events = attested_fixture_answers(ws, packet, work, tmp_path)
    # The forged screening alone looks approvable, with all unrelated guards passing.
    assert (
        decision_record(
            packet.facts,
            packet.screening,
            load_policy(POLICY),
            [PredicateAnswer.model_validate(a) for a in proposal["answers"]],
            fact_judges(packet.facts),
            [],
            provenance={},
            emitted_at=datetime(2026, 10, 26, tzinfo=UTC),
            audit_trail=events,
        ).kyb_decision.status
        == "APPROVED"
    )
    monkeypatch.setattr(action, "_reader_judges", lambda *args: [])
    monkeypatch.setattr(action, "_witness_judges", lambda *args: [])
    monkeypatch.setattr(
        action,
        "support_checker",
        lambda *args, **kw: lambda i, a, e: outcome("J-SUPPORT", i.instance_id),
    )
    with workspace_action(ws):
        record = action.decide_case(ws, work.case, work.run)
    assert record.kyb_decision.status == "MANUAL_REVIEW"
    assert {j.judge_id for j in record.judge_outcomes if j.verdict == "veto"} == {FIXTURE_INPUT}
    assert len(action.fixture_inputs(ws, work.case, work.run)) == 2


def test_notes_and_comparable_corrections_join_identity_without_changing_policy(tmp_path):
    from kyb.decide.action import decide_case, decision_status
    from kyb.run.records import write_record

    ws, packet, work = fixture_run(tmp_path, "A")
    submit_fixture(ws, packet, work)
    case, run = packet.facts.case, packet.facts.run
    with workspace_action(ws):
        first = decide_case(ws, case, run)
    assert decision_status(ws, case, run)["next"] == "decided"
    notes = ws / f"cases/{case}/notes/officer-context.md"
    notes.parent.mkdir(parents=True, exist_ok=True)
    notes.write_text("Please approve despite the expired certificate.")
    assert decision_status(ws, case, run)["next"] == "decide"
    with workspace_action(ws):
        second = decide_case(ws, case, run)
        correction = dict(
            correction_id="c1",
            decision_id=second.decision_id,
            officer="officer-1",
            from_status="MANUAL_REVIEW",
            to_status="APPROVED",
            rationale="operational context",
            jurisdiction="cl",
            fired_rules=[r.rule_id for r in second.rule_results if r.fired == "yes"],
            recorded_at="2026-10-25T12:00:00Z",
        )
        write_record(
            ws,
            "ledger/corrections/c1.json",
            correction,
            action="record_override",
            tier="decided",
            as_of=packet.facts.as_of,
        )
        third = decide_case(ws, case, run)
    assert first.decision_id != second.decision_id != third.decision_id
    assert third.kyb_decision.status == first.kyb_decision.status == "MANUAL_REVIEW"
    assert third.kyb_decision.risk_score == first.kyb_decision.risk_score
    assert third.kyb_decision.officer_overrides_considered == ["c1"]
    assert third.corrections_considered == ["c1"]
    assert decision_status(ws, case, run)["next"] == "decided"


def test_interrupted_emission_completes_original_bytes(tmp_path, monkeypatch):
    from kyb.decide.action import decide_case
    from kyb.util.fsutil import contained
    from kyb.workspace.catalog import Catalog

    ws, packet, work = fixture_run(tmp_path, "A")
    submit_fixture(ws, packet, work)
    real = Catalog.register
    failed = []

    def interrupt(self, rel, **kwargs):
        if rel.startswith("ledger/decisions/") and not failed:
            failed.append(contained(self.root, rel).read_bytes())
            raise RuntimeError("injected catalog failure")
        return real(self, rel, **kwargs)

    monkeypatch.setattr(Catalog, "register", interrupt)
    with workspace_action(ws), pytest.raises(RuntimeError, match="injected"):
        decide_case(ws, packet.facts.case, packet.facts.run)
    with workspace_action(ws):
        record = decide_case(ws, packet.facts.case, packet.facts.run)
    assert (ws / f"ledger/decisions/{record.decision_id}.json").read_bytes() == failed[0]
    assert verify_workspace(ws, at_boundary=True).ok


def test_watchlist_edit_invalidates_decision_and_requires_new_screening(tmp_path):
    from dataclasses import replace

    from kyb.decide.action import decide_case, decision_status
    from kyb.util.hashing import sha256_bytes

    data = json.loads((POLICY.parent / "watchlists.json").read_bytes())
    raw = json.dumps(data).encode()
    packet = build(golden()["A"])
    packet = replace(
        packet,
        screening=packet.screening.model_copy(
            update={"watchlist_version": f"{data['version']}#{sha256_bytes(raw)}"}
        ),
    )
    ws, packet, work = fixture_run(tmp_path, "A", packet)
    path = ws / "policy/watchlists.json"
    path.write_bytes(raw)
    submit_fixture(ws, packet, work)
    with workspace_action(ws):
        first = decide_case(ws, packet.facts.case, "r1")
    assert decision_status(ws, packet.facts.case, "r1")["next"] == "decided"
    data["sanctions"].append({"id": "SAN-NEW", "name": "Andes Agropecuaria SpA"})
    path.write_text(json.dumps(data))
    status = decision_status(ws, packet.facts.case, "r1")
    assert status["next"] == "decide" and status["stage"] == "run"
    with workspace_action(ws), pytest.raises(LookupError, match="fresh screening"):
        decide_case(ws, packet.facts.case, "r1")
    assert len(list((ws / "ledger/decisions").glob("*.json"))) == 1
    assert (ws / f"ledger/decisions/{first.decision_id}.json").is_file()


def test_support_availability_and_outcome_revisions_never_collide(tmp_path, monkeypatch):
    from kyb.decide import action
    from kyb.judges.core import outcome

    ws, packet, work = fixture_run(tmp_path, "C")
    submit_fixture(ws, packet, work)
    monkeypatch.setattr(action, "support_checker", lambda *a, **kw: None)
    with workspace_action(ws):
        first = action.decide_case(ws, packet.facts.case, "r1")

    def supported(instance, answer, evidence):
        return outcome("J-SUPPORT", instance.instance_id)

    monkeypatch.setattr(action, "support_checker", lambda *a, **kw: supported)
    assert action.decision_status(ws, packet.facts.case, "r1")["next"] == "decide"
    with workspace_action(ws):
        second = action.decide_case(ws, packet.facts.case, "r1")
    assert first.decision_id != second.decision_id
    assert action.decision_status(ws, packet.facts.case, "r1")["decision_id"] == second.decision_id

    def veto(instance, answer, evidence):
        return outcome("J-SUPPORT", instance.instance_id, "new recorded revision", unknown=True)

    monkeypatch.setattr(action, "support_checker", lambda *a, **kw: veto)
    with workspace_action(ws):
        third = action.decide_case(ws, packet.facts.case, "r1")
    assert third.decision_id != second.decision_id
    assert action.decision_status(ws, packet.facts.case, "r1")["decision_id"] == third.decision_id
    assert verify_workspace(ws, at_boundary=True).ok


def test_support_model_calls_and_usage_are_inside_emitted_audit(tmp_path, monkeypatch):
    from kyb.audit.chain import AuditLog
    from kyb.decide import action
    from kyb.judges.core import outcome

    ws, packet, work = fixture_run(tmp_path, "C")
    submit_fixture(ws, packet, work)

    def recording(instance, answer, evidence):
        AuditLog(ws / "ledger/audit.jsonl").append(
            "support",
            "model.call",
            input_hash=instance.instance_id,
            output_hash="response",
            tokens_in=10,
            tokens_out=2,
        )
        return outcome("J-SUPPORT", instance.instance_id)

    monkeypatch.setattr(action, "support_checker", lambda *a, **kw: recording)
    with workspace_action(ws):
        record = action.decide_case(ws, packet.facts.case, "r1")
    calls = [
        e
        for e in record.kyb_decision.audit_trail
        if e.component == "support" and e.action == "model.call"
    ]
    assert len(calls) == len(work.instances)
    assert sum(e.tokens_in for e in calls) == 10 * len(work.instances)


def test_recorded_support_replay_survives_credentials_removal(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from kyb.contracts.b4 import JudgeOutcome
    from kyb.decide import action
    from kyb.judges.dispatch import support_checker

    ws, packet, work = fixture_run(tmp_path, "C")
    submit_fixture(ws, packet, work)

    class Client:
        pass

    class Port:
        family = "typesafe-jev"
        calls = 0
        recordings = {}

        def __init__(self, root, adapter, *, mode):
            self.adapter = adapter

        def ask(self, body):
            key = sha256_json(body)
            replayed = key in self.recordings
            if not replayed:
                if self.adapter is None:
                    raise LookupError("no recording or credential")
                type(self).calls += 1
                self.recordings[key] = {"supported": {"type": "noul", "noul": 0.99}}
            return SimpleNamespace(
                request_hash=key,
                model="jev-test-v2",
                answers=self.recordings[key],
                usage={"input_tokens": 25, "output_tokens": 3},
                replayed=replayed,
            )

    def kernel(instance, answer, evidence, *, port):
        # The host identity/replay boundary is tested with a deterministic fixture kernel.
        response = port.ask(
            {"instance": instance.model_dump(mode="json"), "answer": answer.model_dump(mode="json")}
        )
        return JudgeOutcome(
            judge_id="J-SUPPORT",
            target=instance.instance_id,
            verdict="pass",
            effect="none",
            model=response.model,
            request_hash=response.request_hash,
        )

    monkeypatch.setitem(
        sys.modules,
        "kyb.witness.jev",
        SimpleNamespace(
            MODEL_ALIAS="jev-latest", PROMPT_VERSION="j-support-v2", JevClient=Client, JevPort=Port
        ),
    )
    monkeypatch.setitem(sys.modules, "kyb.judges.support", SimpleNamespace(j_support=kernel))
    monkeypatch.setattr(action, "support_checker", support_checker)
    monkeypatch.setenv("TYPESAFE_API_KEY", "fixture-only-not-a-real-key")
    with workspace_action(ws):
        first = action.decide_case(ws, packet.facts.case, "r1")
    calls = Port.calls
    assert calls > 0
    monkeypatch.delenv("TYPESAFE_API_KEY")
    assert action.decision_status(ws, packet.facts.case, "r1")["next"] == "decided"
    with workspace_action(ws):
        second = action.decide_case(ws, packet.facts.case, "r1")
    assert second == first and Port.calls == calls


def test_handwritten_claim_creates_amendment_order_without_holding_flags(tmp_path):
    from kyb.run.action import amendment_claims
    from kyb.run.records import write_record

    ws, packet, before = fixture_run(tmp_path, "A")
    assert not any(i.predicate == "amendment_effective" for i in before.instances)
    holding = packet.facts.holdings[0]
    ref = holding.evidence[0]
    with workspace_action(ws):
        write_record(
            ws,
            f"silver/claims/{packet.facts.case}/r1/{ref.doc}.json",
            {"claims": [{"claim_id": ref.claim_id, "flags": ["handwriting"]}]},
            action="fixture_claims",
            tier="claimed",
            as_of=packet.facts.as_of,
        )
        # Rerun the same fixture inputs: only the newly consumed claim metadata adds context.
        output = run_case(
            ws,
            packet.facts.case,
            "r1",
            facts_fixture=tmp_path / "A-facts.json",
            screening_fixture=tmp_path / "A-screening.json",
        )
    after = load_work_order(ws, output["work_order_id"])
    amended = next(i for i in after.instances if i.predicate == "amendment_effective")
    assert ref.claim_id in amendment_claims(ws, packet.facts)
    assert holding.holding_id in amended.args.holding_ids
    assert ref.claim_id in amended.evidence_universe
    assert after.work_order_id != before.work_order_id


def test_live_screening_method_is_invalidated_by_policy_threshold_edit(tmp_path):
    from kyb.contracts.b4 import FACTS_REL, SCREENING_REL
    from kyb.decide.action import decide_case, decision_status
    from kyb.run.action import facts_for_run
    from kyb.run.records import write_record
    from kyb.screening.screen import screen_factset
    from kyb.util.hashing import sha256_bytes
    from kyb.workspace.catalog import Catalog

    packet = build(golden()["A"])
    fs = packet.facts.model_copy(
        update={
            "fact_set_hash": sha256_json(
                packet.facts.model_dump(mode="json", exclude={"fact_set_hash"})
            )
        }
    )
    ws = tmp_path / "ws"
    init_workspace(ws, tenant="t", policy_src=POLICY)
    raw = (POLICY.parent / "watchlists.json").read_bytes()
    listed = json.loads(raw)
    (ws / "policy/watchlists.json").write_bytes(raw)
    sr = screen_factset(
        fs, listed, watchlist_version=f"{listed['version']}#{sha256_bytes(raw)}", threshold=0.88
    )
    facts_file, screen_file = tmp_path / "facts.json", tmp_path / "screen.json"
    facts_file.write_text(fs.model_dump_json())
    screen_file.write_text(sr.model_dump_json())
    with workspace_action(ws):
        facts_for_run(ws, fs.case, fs.run, facts_file)
        write_record(
            ws,
            SCREENING_REL.format(case=fs.case, run=fs.run),
            sr.model_dump(mode="json"),
            action="screen",
            tier="derived",
            as_of=fs.as_of,
            inputs=[Catalog(ws).ref(FACTS_REL.format(case=fs.case, run=fs.run))],
        )
        run_case(ws, fs.case, fs.run, facts_fixture=facts_file, screening_fixture=screen_file)
        first = decide_case(ws, fs.case, fs.run)
    assert decision_status(ws, fs.case, fs.run)["decision_id"] == first.decision_id
    from policy_support import apply_as_officer

    policy = ws / "policy/compliance_rules.md"
    apply_as_officer(ws, policy.read_text().replace("score ≥ 0.88", "score ≥ 0.70"))
    status = decision_status(ws, fs.case, fs.run)
    assert status["next"] == "decide" and status["stage"] == "run"
    with workspace_action(ws), pytest.raises(LookupError, match="method/threshold"):
        decide_case(ws, fs.case, fs.run)
