"""Predicate scope and harness isolation regressions."""

import json

import pytest
from test_decide import prepared as prepared

from kyb.audit.chain import workspace_action
from kyb.run.action import load_work_order, run_case


def work_and_payload(prepared):
    ws, packet, (facts, screen) = prepared
    with workspace_action(ws):
        out = run_case(ws, packet.facts.case, "r1", facts_fixture=facts, screening_fixture=screen)
    work = load_work_order(ws, out["work_order_id"])
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
        dict(
            instance_id=i.instance_id,
            value="UNKNOWN",
            confidence=0.5,
            evidence=[],
            rationale="not established",
        )
        for i in work.instances
    ]
    return ws, work, payload


def hook(ws, fn, **kw):
    from kyb.workspace.guard import HOOK_SOURCE

    token = HOOK_SOURCE.set("plugin")
    try:
        return fn({"cwd": str(ws), "session_id": "s1", **kw})
    finally:
        HOOK_SOURCE.reset(token)


def predicates_agent(ws, work):
    from kyb.workspace.guard import pre_tool_use, subagent_start

    assert (
        hook(
            ws,
            pre_tool_use,
            tool_name="Agent",
            tool_use_id="launch",
            tool_input={
                "subagent_type": "kyb:kyb-predicates",
                "prompt": f"kyb-predicates {work.work_order_id}",
            },
        )[0]
        == 0
    )
    assert hook(ws, subagent_start, agent_type="kyb:kyb-predicates", agent_id="p1")[0] == 0
    assert (
        hook(
            ws,
            pre_tool_use,
            agent_type="kyb:kyb-predicates",
            agent_id="p1",
            tool_name="mcp__kyb__predicate_work_order",
            tool_use_id="read-work",
            tool_input={"work_order_id": work.work_order_id},
        )[0]
        == 0
    )


def test_predicate_scope_and_pinning(prepared):
    from kyb.predicates.submit import answers_rel, propose_predicates

    ws, work, payload = work_and_payload(prepared)
    with workspace_action(ws):
        for field, value in (
            ("fact_set_hash", "other"),
            ("rules_version", "other"),
            ("catalog_version", "other"),
        ):
            with pytest.raises(ValueError, match=field):
                propose_predicates(ws, {**payload, field: value})
        with pytest.raises(ValueError, match="exactly once"):
            propose_predicates(ws, {**payload, "answers": payload["answers"] * 2})
        assert not (ws / answers_rel(work.work_order_id)).exists()
        bad = [{**a, "evidence": ["foreign"]} for a in payload["answers"]]
        with pytest.raises(ValueError, match="outside"):
            propose_predicates(ws, {**payload, "answers": bad})
        out = propose_predicates(ws, payload)
        assert out["context"] == "unattested"
        assert propose_predicates(ws, payload) == out
        with pytest.raises(PermissionError, match="pinned"):
            propose_predicates(
                ws, {**payload, "answers": [{**a, "value": True} for a in payload["answers"]]}
            )


def test_predicates_attestation_and_late_lifecycle_failure(prepared, tmp_path):
    from kyb.audit.chain import chain_events
    from kyb.predicates.lifecycle import lifecycle_reasons
    from kyb.predicates.submit import load_answers, propose_predicates
    from kyb.workspace.guard import pre_tool_use, subagent_start, subagent_stop

    ws, work, payload = work_and_payload(prepared)
    predicates_agent(ws, work)
    assert (
        hook(
            ws,
            pre_tool_use,
            agent_type="kyb:kyb-predicates",
            agent_id="p1",
            tool_name="Read",
            tool_input={"file_path": "/tmp/x"},
        )[0]
        == 2
    )
    assert (
        hook(
            ws,
            pre_tool_use,
            agent_type="kyb:kyb-predicates",
            agent_id="p1",
            tool_name="predicate_work_order",
            tool_input={"work_order_id": "wo-other"},
        )[0]
        == 2
    )
    result = hook(
        ws,
        pre_tool_use,
        agent_type="kyb:kyb-predicates",
        agent_id="p1",
        tool_name="propose_predicates",
        tool_use_id="submit",
        tool_input=payload,
    )
    executed = result[2]["hookSpecificOutput"]["updatedInput"]
    with workspace_action(ws):
        assert propose_predicates(ws, executed)["context"] == "isolated"
    transcript = tmp_path / "agent-p1.jsonl"
    records = [
        {"type": "user", "message": {"content": f"kyb-predicates {work.work_order_id}"}},
        {
            "type": "assistant",
            "message": {
                "id": "m1",
                "model": "claude-opus",
                "content": [{"type": "tool_use", "name": "mcp__kyb__propose_predicates"}],
                "usage": {"input_tokens": 12, "output_tokens": 6},
            },
        },
    ]
    transcript.write_text("\n".join(json.dumps(r) for r in records))
    hook(
        ws,
        subagent_stop,
        agent_type="kyb:kyb-predicates",
        agent_id="p1",
        agent_transcript_path=str(transcript),
    )
    who = load_answers(ws, work.work_order_id)["produced_by"]["predicates"]

    def check():
        return lifecycle_reasons(
            chain_events(ws),
            who,
            f"kyb-predicates {work.work_order_id}",
            {"predicate_work_order", "propose_predicates"},
        )

    assert check() == []
    hook(ws, subagent_start, agent_type="kyb:kyb-predicates", agent_id="p1")
    assert any("resumed" in r for r in check())


def test_policy_edit_reuses_only_unchanged_predicate_revisions(prepared, tmp_path):
    from policy_support import apply_as_officer, reinstall

    from kyb.predicates.submit import load_answers, propose_predicates
    from kyb.run.action import run_case

    ws, work, payload = work_and_payload(prepared)
    with workspace_action(ws):
        propose_predicates(ws, payload)
    policy = ws / "policy/compliance_rules.md"
    apply_as_officer(ws, policy.read_text().replace("SCORE +20", "SCORE +21"))
    _, packet, (facts, screen) = prepared
    with workspace_action(ws):
        changed = run_case(
            ws, packet.facts.case, "r1", facts_fixture=facts, screening_fixture=screen
        )
    assert changed["work_order_id"] != work.work_order_id
    reused = load_answers(ws, changed["work_order_id"])
    assert reused["pending"] == []
    assert {s["work_order_id"] for s in reused["sources"].values()} == {work.work_order_id}
    # a wording change the engine must be taught: it arrives with code, by `kyb init --force`
    reworded = policy.read_text().replace(
        "evidence of the registered or operating address",
        "confirmation of the registered or operating address",
    )
    reinstall(ws, reworded, tmp_path / "reworded")
    with workspace_action(ws):
        changed2 = run_case(
            ws, packet.facts.case, "r1", facts_fixture=facts, screening_fixture=screen
        )
    partial = load_answers(ws, changed2["work_order_id"])
    assert "poa-05" in partial["pending"]
    assert "sp-SAN-0001-strait-capital" in partial["sources"]


def test_renamed_rule_dependencies_cannot_reuse_a_stale_answer_pin(prepared):
    from kyb.predicates.submit import pin_rel

    _, work, _ = work_and_payload(prepared)
    instance = next(i for i in work.instances if i.predicate == "doc_is_proof_of_address")
    renamed = {**work.policy_text, "R-docs": work.policy_text["DOC-02"]}
    del renamed["DOC-02"]
    first = work.model_copy(update={"policy_text": renamed})
    second = first.model_copy(
        update={
            "rules_version": work.rules_version + "-edited",
            "policy_text": {**renamed, "R-docs": "changed document requirement"},
        }
    )
    assert pin_rel(first, instance) != pin_rel(second, instance)


def test_support_answerer_resolver_uses_exact_attested_answer_and_stopped_model(prepared, tmp_path):
    from kyb.contracts.b4 import PredicateAnswer
    from kyb.decide.support import answerer_resolver
    from kyb.predicates.submit import load_answers, propose_predicates
    from kyb.workspace.guard import pre_tool_use, subagent_start, subagent_stop

    ws, work, payload = work_and_payload(prepared)
    for a in payload["answers"]:
        instance = next(i for i in work.instances if i.instance_id == a["instance_id"])
        a.update(value=True, confidence=0.9, evidence=instance.evidence_universe)
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
    with workspace_action(ws):
        propose_predicates(ws, result[2]["hookSpecificOutput"]["updatedInput"])
    proposal = load_answers(ws, work.work_order_id)
    instance = work.instances[0]
    answer = PredicateAnswer.model_validate(
        next(a for a in proposal["answers"] if a["instance_id"] == instance.instance_id)
    )
    resolve = answerer_resolver(ws, work, proposal)
    assert resolve(instance, answer) is None  # no stopped model/usage yet
    transcript = tmp_path / "predicate-transcript.jsonl"
    records = [
        {"type": "user", "message": {"content": f"kyb-predicates {work.work_order_id}"}},
        {
            "type": "assistant",
            "message": {
                "id": "p-turn",
                "model": "claude-opus-5-5",
                "content": [{"type": "tool_use", "name": "mcp__kyb__propose_predicates"}],
                "usage": {"input_tokens": 20, "output_tokens": 4},
            },
        },
    ]
    transcript.write_text("\n".join(json.dumps(r) for r in records))
    hook(
        ws,
        subagent_stop,
        agent_type="kyb:kyb-predicates",
        agent_id="p1",
        agent_transcript_path=str(transcript),
    )
    assert resolve(instance, answer) == "claude-opus-5-5"
    assert resolve(instance, answer.model_copy(update={"confidence": 0.4})) is None
    assert resolve(instance.model_copy(update={"evidence_universe": []}), answer) is None
    hook(ws, subagent_start, agent_type="kyb:kyb-predicates", agent_id="p1")
    assert resolve(instance, answer) is None


def test_interrupted_usage_receipts_cap_then_recover_exactly_once(prepared, tmp_path, monkeypatch):
    from kyb.audit.chain import AuditLog, chain_events, verify_workspace
    from kyb.contracts.b4 import PredicateAnswer
    from kyb.judges.core import predicate_judges
    from kyb.predicates.lifecycle import lifecycle_reasons
    from kyb.predicates.submit import load_answers, propose_predicates
    from kyb.workspace.guard import pre_tool_use, subagent_stop

    ws, work, payload = work_and_payload(prepared)
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
    with workspace_action(ws):
        propose_predicates(ws, result[2]["hookSpecificOutput"]["updatedInput"])
    transcript = tmp_path / "predicate-usage.jsonl"
    records = [
        {"type": "user", "message": {"content": f"kyb-predicates {work.work_order_id}"}},
        {
            "type": "assistant",
            "message": {
                "id": "m-usage",
                "model": "claude-opus-5-5",
                "content": [],
                "usage": {"input_tokens": 14, "output_tokens": 7},
            },
        },
    ]
    transcript.write_text("\n".join(json.dumps(r) for r in records))
    append = AuditLog.append_record

    def interrupt(self, component, action, **kwargs):
        if action == "model.call":
            raise RuntimeError("injected usage append failure")
        return append(self, component, action, **kwargs)

    monkeypatch.setattr(AuditLog, "append_record", interrupt)
    with pytest.raises(RuntimeError, match="injected usage"):
        hook(
            ws,
            subagent_stop,
            agent_type="kyb:kyb-predicates",
            agent_id="p1",
            agent_transcript_path=str(transcript),
        )
    assert verify_workspace(ws, at_boundary=True).ok
    proposal = load_answers(ws, work.work_order_id)
    who = proposal["produced_by"]["predicates"]
    reasons = lifecycle_reasons(
        chain_events(ws),
        who,
        f"kyb-predicates {work.work_order_id}",
        {"predicate_work_order", "propose_predicates"},
    )
    assert any("audit: incomplete" in reason for reason in reasons)
    _, packet, _ = prepared
    _, judges = predicate_judges(
        packet.facts,
        work,
        [PredicateAnswer.model_validate(a) for a in proposal["answers"]],
        who=who,
        sources=proposal["sources"],
        events=chain_events(ws),
    )
    assert any(
        j.judge_id == "J-PRED-ISOLATION" and j.verdict == "veto" and "audit: incomplete" in j.reason
        for j in judges
    )
    monkeypatch.setattr(AuditLog, "append_record", append)
    for _ in range(2):
        hook(
            ws,
            subagent_stop,
            agent_type="kyb:kyb-predicates",
            agent_id="p1",
            agent_transcript_path=str(transcript),
        )
    assert (
        lifecycle_reasons(
            chain_events(ws),
            who,
            f"kyb-predicates {work.work_order_id}",
            {"predicate_work_order", "propose_predicates"},
        )
        == []
    )
    usage = [e for e in chain_events(ws) if e["action"] == "model.call"]
    assert len(usage) == 1 and usage[0]["tokens_in"] == 14 and usage[0]["tokens_out"] == 7


def test_l4j_runtime_resolver_integration_and_credential_free_replay(
    prepared, tmp_path, monkeypatch
):
    import kyb.judges.support as support
    from kyb.contracts.b4 import PredicateAnswer
    from kyb.decide.action import decide_case, decision_status
    from kyb.decide.support import answerer_resolver, default_support_port
    from kyb.predicates.submit import load_answers, propose_predicates
    from kyb.witness import jev
    from kyb.workspace.guard import pre_tool_use, subagent_start, subagent_stop

    ws, work, payload = work_and_payload(prepared)
    for a in payload["answers"]:
        instance = next(i for i in work.instances if i.instance_id == a["instance_id"])
        a.update(value=True, confidence=0.95, evidence=instance.evidence_universe)
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
    with workspace_action(ws):
        propose_predicates(ws, result[2]["hookSpecificOutput"]["updatedInput"])
    transcript = tmp_path / "attested-predicates.jsonl"
    records = [
        {"type": "user", "message": {"content": f"kyb-predicates {work.work_order_id}"}},
        {
            "type": "assistant",
            "message": {
                "id": "v2-model-turn",
                "model": "claude-opus-5-5",
                "content": [],
                "usage": {"input_tokens": 40, "output_tokens": 8},
            },
        },
    ]
    transcript.write_text("\n".join(json.dumps(r) for r in records))
    hook(
        ws,
        subagent_stop,
        agent_type="kyb:kyb-predicates",
        agent_id="p1",
        agent_transcript_path=str(transcript),
    )
    calls = []

    class FixtureClient:
        unavailable = True

        def ask(self, body):
            calls.append(body)
            if self.unavailable:
                raise jev.JevUnavailable("injected temporary outage")
            return {
                "model": "jev-1.14.0",
                "answers": {"supported": {"type": "noul", "noul": 0.99}},
                "usage": {"input_tokens": 60, "output_tokens": 3},
            }

    monkeypatch.setattr(jev, "JevClient", FixtureClient)  # real v2 kernel/port, no network adapter
    monkeypatch.setenv("TYPESAFE_API_KEY", "fixture-only-unused")
    monkeypatch.setenv("KYB_SUPPORT_REVISION", "must-have-no-effect")
    with workspace_action(ws):
        unavailable = decide_case(ws, work.case, work.run)
    assert all(j.verdict == "veto" for j in unavailable.judge_outcomes if j.judge_id == "J-SUPPORT")
    FixtureClient.unavailable = False
    with workspace_action(ws):
        first = decide_case(ws, work.case, work.run)
    assert first.decision_id != unavailable.decision_id
    assert calls and all("predicate_args" in body["state"] for body in calls)
    assert any(j.judge_id == "J-SUPPORT" and j.verdict == "pass" for j in first.judge_outcomes)
    count = len(calls)
    monkeypatch.delenv("TYPESAFE_API_KEY")
    monkeypatch.setenv("KYB_SUPPORT_REVISION", "environment-cannot-reset-support")
    assert decision_status(ws, work.case, work.run)["next"] == "decided"
    with workspace_action(ws):
        assert decide_case(ws, work.case, work.run) == first
    assert len(calls) == count
    proposal = load_answers(ws, work.work_order_id)
    resolver = answerer_resolver(ws, work, proposal)
    port = default_support_port(ws, answerer_model=resolver)
    instance = next(i for i in work.instances if i.predicate == "doc_is_proof_of_address")
    answer = PredicateAnswer.model_validate(
        next(a for a in proposal["answers"] if a["instance_id"] == instance.instance_id)
    )
    evidence = [e for e in work.evidence if e.evidence_id in answer.evidence]
    changed = answer.model_copy(update={"rationale": "different answer"})
    assert support.j_support(instance, changed, evidence, port=port).verdict == "veto"
    assert (
        support.j_support(
            instance.model_copy(update={"evidence_universe": []}), answer, evidence, port=port
        ).verdict
        == "veto"
    )
    hook(ws, subagent_start, agent_type="kyb:kyb-predicates", agent_id="p1")
    assert support.j_support(instance, answer, evidence, port=port).verdict == "veto"
    with workspace_action(ws):
        resumed = decide_case(ws, work.case, work.run)
    assert resumed.decision_id != first.decision_id
    assert all(j.verdict == "veto" for j in resumed.judge_outcomes if j.judge_id == "J-SUPPORT")
    assert len(calls) == count  # every provenance failure vetoed before the API boundary
