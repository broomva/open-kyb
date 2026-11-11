"""spec: the opt-in `approve-measured` decision profile (issue).

Strict stays the default and byte-identical. Under the profile a failed claim or an uncovered
fact vetoes approval only when a live rule reads it; everything else keeps its veto."""

from __future__ import annotations

import pytest
from test_decide import fixture_run, submit_fixture

from kyb.audit.chain import chain_events, workspace_action
from kyb.cli import main
from kyb.contracts import b4
from kyb.decide.profile import ENV, resolve_profile
from kyb.judges.core import (
    APPROVE_MEASURED,
    PROFILE_JUDGE,
    STRICT,
    WARNING,
    fact_judges,
    live_rules,
)
from kyb.policy import evaluate
from kyb.sim.l4 import L4Decider, record_for, verified_inputs
from kyb.synth import build as B
from kyb.synth.golden import golden_specs
from kyb.synth.label import label

pytest_plugins = ["synth_support"]


@pytest.fixture
def pool(raw, tmp_path):
    cases, _, _, _ = B.build(40, 11, raw, cache=tmp_path / "cache")
    return [*cases, *golden_specs().values()]


def first(pool, status):
    return next(c for c in pool if label(c).status == status)


def uncovered(kind, doc=None, findings=("coverage_incomplete",)):
    def apply(fs):
        target = doc or sorted(d.doc for d in fs.documents)[0]
        gap = b4.UncoveredFact(
            block_id="b0", kind=kind, subject=fs.applicant, attribute="x", value="v"
        )
        ver = [
            v.model_copy(update={"uncovered": [gap], "findings": list(findings)})
            if v.doc == target
            else v
            for v in fs.verification
        ]
        return fs.model_copy(update={"verification": ver})

    return apply


def failed_claim(fs):
    ref = b4.EvidenceRef(
        claim_id="c-failed-0", doc=sorted(d.doc for d in fs.documents)[0], page=1, verified=False
    )
    return fs.model_copy(update={"unverified": [ref]})


def vetoes(record):
    return {(j.judge_id, j.target) for j in record.judge_outcomes if j.verdict == "veto"}


# --- selection ---


def test_default_is_strict_and_the_choice_is_explicit(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert resolve_profile() == STRICT
    monkeypatch.setenv(ENV, "approve-measured")
    assert resolve_profile() == APPROVE_MEASURED
    assert resolve_profile("strict") == STRICT  # the flag wins over the environment
    monkeypatch.setenv(ENV, "approve-measuerd")  # a misspelt opt-in is an error, not strict
    with pytest.raises(ValueError, match="unknown decision profile"):
        resolve_profile()


# --- the strict default is unchanged ---


def test_strict_records_no_profile_trace(pool):
    rec = record_for(first(pool, "APPROVED"))
    assert PROFILE_JUDGE not in {j.judge_id for j in rec.judge_outcomes}
    assert not any(j.reason.startswith(WARNING) for j in rec.judge_outcomes)


def test_strict_judges_are_the_unscoped_judges(pool):
    for case in pool[:12]:
        fs, _, _ = verified_inputs(case)
        assert fact_judges(fs) == fact_judges(fs, None)


def test_a_clean_case_decides_the_same_under_both_profiles(pool):
    for status in ("APPROVED", "MANUAL_REVIEW", "REJECTED"):
        case = first(pool, status)
        a, b = record_for(case), record_for(case, profile=APPROVE_MEASURED)
        assert (a.kyb_decision.status, a.kyb_decision.risk_score) == (
            b.kyb_decision.status,
            b.kyb_decision.risk_score,
        )
        assert a.decision_id != b.decision_id  # distinguishable by identity


# --- scoping ---


def test_a_gap_no_rule_reads_stops_blocking_a_clean_case(pool, monkeypatch):
    """With the dependency map saying no rule reads `stated_validity`, an uncovered fact of that
    kind and a failed claim are warnings under the profile and vetoes under strict."""
    from kyb.policy import rules

    monkeypatch.setitem(rules.GAP_RULES, "stated_validity", ())
    case = first(pool, "APPROVED")
    gap = uncovered("stated_validity")
    strict = record_for(case, gaps=gap)
    assert strict.kyb_decision.status == "MANUAL_REVIEW"
    assert any(j[0] == "J-VERIFICATION" for j in vetoes(strict))
    measured = record_for(case, profile=APPROVE_MEASURED, gaps=gap)
    assert measured.kyb_decision.status == "APPROVED"
    assert not vetoes(measured)
    warned = [j for j in measured.judge_outcomes if j.reason.startswith(WARNING)]
    assert warned and all(j.verdict == "pass" and j.effect == "none" for j in warned)
    marker = next(j for j in measured.judge_outcomes if j.judge_id == PROFILE_JUDGE)
    assert marker.verdict == "pass" and APPROVE_MEASURED in marker.reason


def test_an_uncovered_fact_a_rule_reads_still_blocks(pool):
    case = first(pool, "APPROVED")
    for kind in ("ownership", "date", "address", "corporate_activity"):
        rec = record_for(case, profile=APPROVE_MEASURED, gaps=uncovered(kind))
        assert rec.kyb_decision.status == "MANUAL_REVIEW", kind
        assert "cannot be established" in rec.kyb_decision.rationale  # unknown rules block


def test_a_failed_claim_still_blocks_when_it_cannot_be_placed(pool):
    # an unclassified claim feeds every rule: the rules read unknown and J-EVIDENCE vetoes
    rec = record_for(first(pool, "APPROVED"), profile=APPROVE_MEASURED, gaps=failed_claim)
    assert rec.kyb_decision.status == "MANUAL_REVIEW"
    assert ("J-EVIDENCE", "unverified") in vetoes(rec)


def test_only_the_gap_vetoes_are_scoped(pool, monkeypatch):
    from kyb.policy import rules

    monkeypatch.setitem(rules.GAP_RULES, "stated_validity", ())
    case = first(pool, "APPROVED")
    for findings in (
        ("coverage_incomplete", "coverage_conflict"),
        ("coverage_incomplete", "layer_vs_pixel"),
        ("coverage_incomplete", "integrity"),
    ):
        rec = record_for(
            case, profile=APPROVE_MEASURED, gaps=uncovered("stated_validity", findings=findings)
        )
        assert rec.kyb_decision.status == "MANUAL_REVIEW", findings
        assert any(j[0] == "J-VERIFICATION" for j in vetoes(rec))

    def not_isolated(fs):
        return fs.model_copy(
            update={
                "verification": [v.model_copy(update={"isolated": False}) for v in fs.verification]
            }
        )

    def w1(fs):
        return fs.model_copy(
            update={
                "verification": [v.model_copy(update={"assurance": "W1"}) for v in fs.verification]
            }
        )

    def tampered(fs):
        ev = [e.model_copy(update={"excerpt": " "}) for e in fs.evidence]
        return fs.model_copy(update={"evidence": ev})

    for gaps in (not_isolated, w1, tampered):
        rec = record_for(case, profile=APPROVE_MEASURED, gaps=gaps)
        assert rec.kyb_decision.status == "MANUAL_REVIEW", gaps.__name__


def test_live_rules_are_the_rules_that_can_move_the_status_or_score(pool):
    fs, sr, pol = verified_inputs(first(pool, "APPROVED"))
    rules = evaluate(fs, sr, pol, []).rule_results
    live = live_rules(rules, pol)
    assert {"JUR-01", "OWN-01", "DOC-02", "MOD-01"} <= live
    assert "POLICY-GUARD" not in live


def test_the_scoping_decision_is_made_per_gap_kind(pool):
    fs, sr, pol = verified_inputs(first(pool, "APPROVED"))
    fs = uncovered("date")(fs)
    assert any(
        j.verdict == "veto" for j in fact_judges(fs, frozenset({"pep_active"}))
    )  # PEP-01 reads dates
    outs = fact_judges(
        fs, frozenset({"industry_elevated"})
    )  # only ACT-02 is live; it does not read dates
    assert not any(j.verdict == "veto" and j.target == fs.verification[0].doc for j in outs)
    assert any(j.reason.startswith(WARNING) for j in outs)
    # no live rule at all: fail closed
    assert any(j.verdict == "veto" for j in fact_judges(fs, frozenset()))


# --- never more favourable to REJECTED / MANUAL_REVIEW ---


def test_rejected_and_manual_review_are_never_made_more_favourable(pool):
    d_strict = L4Decider()
    d_measured = L4Decider(profile=APPROVE_MEASURED)
    for case in pool:
        a, b = d_strict.decide(case), d_measured.decide(case)
        assert (a.status, a.score, a.fired_rules) == (b.status, b.score, b.fired_rules)
        if a.status in ("REJECTED", "MANUAL_REVIEW"):
            assert b.status == a.status


# --- the real decide path: recorded, distinguishable, separate state ---


def test_decide_case_records_the_profile_and_keeps_states_apart(tmp_path, monkeypatch, capsys):
    from kyb.contracts.b4 import DecisionRecord
    from kyb.decide.action import decide_case, decision_status

    monkeypatch.setattr("kyb.decide.action.support_checker", lambda *a, **kw: None)
    ws, packet, work = fixture_run(tmp_path, "C")
    submit_fixture(ws, packet, work)
    case, run = packet.facts.case, packet.facts.run
    monkeypatch.delenv(ENV, raising=False)
    with workspace_action(ws):
        strict = decide_case(ws, case, run)
    assert PROFILE_JUDGE not in {j.judge_id for j in strict.judge_outcomes}
    assert decision_status(ws, case, run)["next"] == "decided"

    monkeypatch.setenv(ENV, APPROVE_MEASURED)
    assert decision_status(ws, case, run)["next"] == "decide"  # a different state, not a cache hit
    with workspace_action(ws):
        measured = decide_case(ws, case, run)
    assert measured.decision_id != strict.decision_id
    assert measured.kyb_decision.status == strict.kyb_decision.status == "MANUAL_REVIEW"
    marker = [j for j in measured.judge_outcomes if j.judge_id == PROFILE_JUDGE]
    assert len(marker) == 1 and APPROVE_MEASURED in marker[0].reason
    events = [e for e in chain_events(ws) if e["detail"].get("action") == "decide.profile"]
    assert len(events) == 1 and events[0]["detail"]["profile"] == APPROVE_MEASURED
    assert any(
        e.detail.get("action") == "decide.profile" for e in measured.kyb_decision.audit_trail
    )
    with workspace_action(ws):
        decide_case(ws, case, run)
    assert len([e for e in chain_events(ws) if e["detail"].get("action") == "decide.profile"]) == 1

    monkeypatch.delenv(ENV)  # the flag, not the environment
    assert main(["-w", str(ws), "decide", case, "--run", run, "--profile", APPROVE_MEASURED]) == 0
    out = DecisionRecord.model_validate_json(capsys.readouterr().out)
    assert out.decision_id == measured.decision_id
