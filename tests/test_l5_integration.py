"""L5 against L4's real writer (issue): a decision `kyb decide` emits is queued, explained,
signed, and an override comes back through L4's own comparable-profile retrieval."""

from __future__ import annotations

import json

import pytest

# built on L4's `kyb decide`: skipped until L4 (PR #33) is on this branch
pytest.importorskip("kyb.decide.action")
test_decide = pytest.importorskip("test_decide")
fixture_run, submit_fixture = test_decide.fixture_run, test_decide.submit_fixture

from kyb.audit.chain import verify_workspace, workspace_action  # noqa: E402
from kyb.copilot.answer import record_answer  # noqa: E402
from kyb.copilot.bundle import ask  # noqa: E402
from kyb.corrections import comparable  # noqa: E402
from kyb.decide.action import decide_case  # noqa: E402
from kyb.signing import officers  # noqa: E402
from kyb.signing.queue import review_queue  # noqa: E402
from kyb.signing.sign import sign_decision  # noqa: E402


@pytest.fixture(autouse=True)
def offline_support(monkeypatch):
    # fixture runs pin the spec unavailable support state; no API key is ever spent here
    monkeypatch.setattr("kyb.decide.action.support_checker", lambda *a, **kw: None)


@pytest.fixture
def decided(tmp_path):
    ws, packet, work = fixture_run(tmp_path, "A")
    submit_fixture(ws, packet, work)
    case, run = packet.facts.case, packet.facts.run
    with workspace_action(ws):
        record = decide_case(ws, case, run)
        token = officers.enroll(ws, "ana")["token"]
    return ws, case, run, record, token


def test_a_real_decision_is_queued_explained_and_signed(decided):
    ws, case, run, record, token = decided
    (entry,) = review_queue(ws, case)["queue"]
    assert (
        entry["decision_id"] == record.decision_id and entry["status"] == record.kyb_decision.status
    )
    assert entry["signature_state"] == "pending_signature"
    with workspace_action(ws):
        out = record_answer(
            ws,
            case,
            "why?",
            [{"type": "explain", "decision_id": record.decision_id, "rule_ids": []}],
            run,
        )
    assert out["accepted"] and record.kyb_decision.status in out["text"]
    with workspace_action(ws):
        signed = sign_decision(ws, decision_id=record.decision_id, verdict="assent",
                               officer="ana", officer_token=token)  # fmt: skip
    assert signed["signature_state"] == "signed" and review_queue(ws, case)["queue"] == []
    assert verify_workspace(ws, at_boundary=True).ok


def test_an_override_is_retrieved_by_l4_as_a_comparable_profile_and_never_edits_policy(decided):
    ws, case, run, record, token = decided
    fired = sorted({r.rule_id for r in record.rule_results if r.fired == "yes"})
    with workspace_action(ws):
        signed = sign_decision(ws, decision_id=record.decision_id, verdict="override",
                               officer="ana", officer_token=token, status="APPROVED",
                               rationale="operational context")  # fmt: skip
        (cid,) = signed["correction_ids"]
        facts = json.loads((ws / f"gold/facts/{case}/{run}.json").read_text())
        app = next(e for e in facts["entities"] if e["entity_id"] == facts["applicant"])
        found = [c for jur in app["jurisdiction"] for c in comparable(ws, jur, fired)]
        again = decide_case(ws, case, run)
    assert [c.correction_id for c in found] == [cid]
    # it informs the next decision, never the policy: same status, same score, new identity
    assert again.corrections_considered == [cid]
    assert again.kyb_decision.officer_overrides_considered == [cid]
    assert again.kyb_decision.status == record.kyb_decision.status
    assert again.kyb_decision.risk_score == record.kyb_decision.risk_score
    assert again.decision_id != record.decision_id
    # the new recommendation is queued; the signed one is not
    queue = review_queue(ws, case)["queue"]
    assert [e["decision_id"] for e in queue] == [again.decision_id]
    assert queue[0]["officer_overrides_considered"] == [cid]
    assert verify_workspace(ws, at_boundary=True).ok


def test_ask_answers_from_the_fixture_facts_the_decision_was_made_on(decided):
    ws, case, run, record, token = decided
    bundle = ask(ws, case, record.kyb_decision.entity_name, run)
    assert bundle["decisions"][0]["decision_id"] == record.decision_id
    assert bundle["facts"], "the recorded FactSet is the copilot's evidence"
    top = bundle["facts"][0]
    with workspace_action(ws):
        out = record_answer(ws, case, "q", [{"type": "fact", "subject": top["subject"],
                            "attribute": top["attribute"], "value": top["value"],
                            "cites": [top["fact_id"]]}], run)  # fmt: skip
    assert out["accepted"]
