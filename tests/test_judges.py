"""Favourable answers cannot bypass evidence, support, arithmetic or precedence guards."""

from datetime import UTC, datetime

import pytest
from test_decide import POLICY

from kyb.contracts.b4 import PredicateAnswer
from kyb.decide.core import decision_record
from kyb.evals.cases import build, golden
from kyb.judges.core import decision_judges, fact_judges, outcome, predicate_judges
from kyb.policy import evaluate, load_policy
from kyb.run.action import build_work_order


def test_favourable_answer_needs_evidence_and_support():
    packet = build(golden()["C"])
    policy = load_policy(POLICY)
    work = build_work_order(packet.facts, packet.screening, policy)
    i = next(i for i in work.instances if i.predicate == "same_party")
    a = PredicateAnswer(
        instance_id=i.instance_id,
        value=True,
        confidence=0.99,
        evidence=i.evidence_universe,
        rationale="fixture",
    )
    checked, judges = predicate_judges(packet.facts, work, [a], who=None, events=[])
    assert next(x for x in checked if x.instance_id == i.instance_id).value is True
    assert any(
        j.judge_id == "j-support-unavailable" and j.effect == "manual_review" for j in judges
    )
    for evidence in ([], ["foreign"]):
        checked, judges = predicate_judges(
            packet.facts,
            work,
            [a.model_copy(update={"evidence": evidence})],
            who=None,
            events=[],
            support=lambda i, a, e: outcome("J-SUPPORT", i.instance_id),
        )
        assert next(x for x in checked if x.instance_id == i.instance_id).value == "UNKNOWN"
    checked, judges = predicate_judges(
        packet.facts,
        work,
        [a],
        who=None,
        events=[],
        support=lambda i, a, e: outcome("J-SUPPORT", i.instance_id, "unsupported", unknown=True),
    )
    assert next(x for x in checked if x.instance_id == i.instance_id).value == "UNKNOWN"


def test_exact_judges_detect_bad_score_status_and_schema():
    p = build(golden()["A"])
    policy = load_policy(POLICY)
    record = decision_record(
        p.facts,
        p.screening,
        policy,
        [],
        [],
        [],
        provenance={},
        emitted_at=datetime(2026, 10, 26, tzinfo=UTC),
        audit_trail=[],
    )
    rules = evaluate(p.facts, p.screening, policy).rule_results
    forged = record.kyb_decision.model_copy(update={"status": "REJECTED", "risk_score": 90})
    judges = decision_judges(forged, rules, policy)
    assert {j.judge_id for j in judges if j.verdict == "veto"} == {"J-ARITH", "J-PRECEDENCE"}
    assert decision_judges({}, rules, policy)[0].judge_id == "J-SCHEMA"


def test_verification_gaps_and_forged_evidence_are_vetoes():
    p = build(golden()["A"])
    e = p.facts.evidence[0].model_copy(update={"verified": False})
    fs = p.facts.model_copy(update={"evidence": [e, *p.facts.evidence[1:]]})
    judges = fact_judges(fs)
    assert any(j.judge_id == "J-EVIDENCE" and j.verdict == "veto" for j in judges)
    assert any(j.judge_id == "J-VERIFICATION" and j.verdict == "veto" for j in judges)


def test_support_dispatch_retries_transport_but_never_rescues_a_content_veto(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from kyb.audit.chain import verify_workspace, workspace_action
    from kyb.judges.dispatch import support_checker
    from kyb.workspace.init import init_workspace

    packet = build(golden()["C"])
    policy = load_policy(POLICY)
    work = build_work_order(packet.facts, packet.screening, policy)
    instance = next(i for i in work.instances if i.predicate == "same_party")
    evidence = [e for e in work.evidence if e.evidence_id in instance.evidence_universe]
    answer = PredicateAnswer(
        instance_id=instance.instance_id,
        value=True,
        confidence=0.99,
        evidence=instance.evidence_universe,
        rationale="fixture",
    )
    ws = tmp_path / "ws"
    init_workspace(ws, tenant="t", policy_src=POLICY)

    class Port:
        family = "typesafe-jev"
        revision = "r1"
        calls = 0
        available = False
        supported = True

        def ask(self):
            self.calls += 1
            if not self.available:
                raise LookupError("unavailable")

    def kernel(instance, answer, evidence, *, port):
        port.ask()
        return outcome(
            "J-SUPPORT", instance.instance_id, "unsupported" if not port.supported else ""
        ).model_copy(update={"model": "jev-1.14.0", "request_hash": "sha256:" + "a" * 64})

    monkeypatch.setitem(sys.modules, "kyb.judges.support", SimpleNamespace(j_support=kernel))
    port = Port()
    first = support_checker(ws, port=port)
    first.work_order_id = work.work_order_id
    with workspace_action(ws):
        unavailable = first(instance, answer, evidence)
    assert unavailable.verdict == "veto" and port.calls == 1
    port.available = True
    with workspace_action(ws):
        assert first(instance, answer, evidence).verdict == "pass"
        assert first(instance, answer, evidence).verdict == "pass"
    assert port.calls == 2  # outage recovers without a revision; success replays without calls
    port.revision = "r2"
    port.supported = False
    second = support_checker(ws, port=port)
    second.work_order_id = work.work_order_id
    with workspace_action(ws):
        veto = second(instance, answer, evidence)
    assert veto.verdict == "veto" and veto.effect == "manual_review" and port.calls == 3
    port.revision, port.supported = "r3", True
    third = support_checker(ws, port=port)
    third.work_order_id = work.work_order_id
    with workspace_action(ws):
        paraphrase = answer.model_copy(update={"rationale": "try again", "confidence": 1.0})
        assert third(instance, paraphrase, list(reversed(evidence))) == veto
        assert first(instance, answer, evidence) == veto  # even an earlier cached pass loses
        checked, _ = predicate_judges(
            packet.facts,
            work.model_copy(update={"instances": [instance]}),
            [answer],
            who=None,
            events=[],
            support=third,
        )
        assert checked[0].value == "UNKNOWN"
    assert port.calls == 3
    assert verify_workspace(ws, at_boundary=True).ok


def test_self_reference_and_unknown_verification_findings_veto_approval():

    from kyb.contracts.b4 import DocVerification
    from kyb.evals.cases import variants

    _, packet = next((label, p) for label, p in variants() if label["id"] == "syn-a-clean")
    verification = [
        DocVerification(doc=d.doc, assurance="W2", isolated=True) for d in packet.facts.documents
    ]
    fs = packet.facts.model_copy(update={"verification": verification})
    policy = load_policy(POLICY)

    def result(facts):
        return decision_record(
            facts,
            packet.screening,
            policy,
            list(packet.answers.values()),
            fact_judges(facts),
            [],
            provenance={},
            emitted_at=datetime(2026, 10, 26, tzinfo=UTC),
            audit_trail=[],
        ).kyb_decision.status

    assert result(fs) == "APPROVED"  # positive control: the guard must cause the downgrade
    for assurance, finding in (
        ("W1", "self_referential_verification"),
        ("W2", "witness_strata_disagreement"),
        ("W2", "witness_routing_refused"),
        ("W2", "future-assurance-finding"),
    ):
        changed = [
            verification[0].model_copy(update={"assurance": assurance, "findings": [finding]}),
            *verification[1:],
        ]
        assert result(fs.model_copy(update={"verification": changed})) == "MANUAL_REVIEW"
    with pytest.raises(ValueError, match="Extra inputs"):
        DocVerification.model_validate(
            {"doc": "01", "assurance": "W2", "isolated": True, "self_referential": True}
        )
    with pytest.raises(ValueError):
        DocVerification(doc="01", assurance="W3", isolated=True)


def test_self_reference_is_approval_only_and_does_not_downgrade_reject_or_review():
    from kyb.contracts.b4 import DocVerification
    from kyb.evals.cases import variants
    from kyb.judges.core import APPROVAL_ASSURANCE

    _, clean = next((label, p) for label, p in variants() if label["id"] == "syn-a-clean")
    policy = load_policy(POLICY)
    for expected, packet, activities in (
        ("APPROVED", clean, None),
        ("REJECTED", clean, ["Manufacturing firearms"]),
        ("MANUAL_REVIEW", build(golden()["B"]), None),
    ):
        fs = packet.facts
        if activities:
            fs = fs.model_copy(update={"corporate_object": activities})
        fs = fs.model_copy(
            update={
                "verification": [
                    DocVerification(
                        doc=d.doc,
                        assurance="W1",
                        isolated=True,
                        findings=["self_referential_verification"],
                    )
                    for d in fs.documents
                ]
            }
        )
        record = decision_record(
            fs,
            packet.screening,
            policy,
            list(packet.answers.values()),
            fact_judges(fs),
            [],
            provenance={},
            emitted_at=datetime(2026, 10, 26, tzinfo=UTC),
            audit_trail=[],
        )
        assert record.kyb_decision.status == (
            "MANUAL_REVIEW" if expected == "APPROVED" else expected
        )
        guards = [j for j in record.judge_outcomes if j.judge_id == APPROVAL_ASSURANCE]
        assert guards and all(
            j.verdict == ("veto" if expected == "APPROVED" else "pass") for j in guards
        )
