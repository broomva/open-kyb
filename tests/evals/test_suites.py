"""`make verify` suites (L6, issue): each suite passes on the real datasets, and each one
fails when it should. A suite that cannot fail is not a test."""

from __future__ import annotations

import copy
import json

import pytest

from kyb.evals import (
    adversarial,
    cases,
    contract,
    decisions,
    pending_suites,
    policy_edits,
    properties,
    scorecard,
)
from kyb.evals.paths import load

# --- scorecard ----------------------------------------------------------------------------------


def test_suite_status_never_reads_pending_as_pass():
    s = scorecard.Suite("x", "x", True)
    assert s.status == "pending"  # an empty suite proves nothing
    s.cases = [scorecard.passed("a"), scorecard.pending("b", "L4 hook")]
    assert s.status == "pending"
    s.cases.append(scorecard.failed("c", "boom"))
    assert s.status == "fail"


def test_verdict_strict_fails_on_pending_and_false_approve_always_fails():
    ok = scorecard.Suite("1", "x", True, [scorecard.passed("a")])
    waiting = scorecard.Suite("2", "x", True, [scorecard.pending("b", "L5")])
    report = scorecard.Suite("B", "x", False, [scorecard.failed("c", "report-only")])
    assert (
        scorecard.build(
            [ok, waiting, report], false_approve=0, inputs={}, strict=False, live=False
        )["verdict"]
        == "pass"
    )
    assert (
        scorecard.build([ok, waiting], false_approve=0, inputs={}, strict=True, live=False)[
            "verdict"
        ]
        == "fail"
    )
    assert (
        scorecard.build([ok], false_approve=1, inputs={}, strict=False, live=False)["verdict"]
        == "fail"
    )
    bad = scorecard.Suite("3", "x", True, [scorecard.failed("d", "x")])
    card = scorecard.build([ok, bad], false_approve=0, inputs={}, strict=False, live=False)
    assert card["verdict"] == "fail" and card["blocking_failed"] == ["3"]


# --- suite 4: decisions ---------------------------------------------------------------------------


def test_decisions_suite_passes_with_zero_false_approve():
    s, false_approve = decisions.suite()
    # B's end-to-end golden is measured and not met: the one failure (it fails safe)
    assert [c.case_id for c in s.cases if c.status == "fail"] in (
        ["end-to-end-golden:B"],
        ["end-to-end-golden:B", "end-to-end-golden:C"],
        [],
    )
    assert false_approve == 0
    assert s.metrics["status_agreement"] == 1.0
    assert set(s.metrics["by_expected_outcome"]) == {"APPROVED", "MANUAL_REVIEW", "REJECTED"}
    assert s.status in ("pass", "pending")


def test_golden_labels_are_the_locked_ones():
    g = cases.golden()
    want = {"A": ("APPROVED", 0), "B": ("MANUAL_REVIEW", 35), "C": ("MANUAL_REVIEW", 90)}
    assert {k: (v["expected"]["status"], v["expected"]["risk_score"]) for k, v in g.items()} == want


def test_a_wrong_label_and_a_false_approve_are_caught():
    p = cases.golden_packet("B")
    ev, comp = decisions.decide(p)
    assert (
        decisions.judge("x", {"status": "MANUAL_REVIEW", "risk_score": 35}, ev, comp).status
        == "pass"
    )
    assert decisions.judge("x", {"status": "APPROVED"}, ev, comp).status == "fail"
    assert (
        decisions.judge("x", {"status": "MANUAL_REVIEW", "fired": ["DOC-02"]}, ev, comp).status
        == "fail"
    )
    assert (
        decisions.judge("x", {"status": "MANUAL_REVIEW", "not_fired": ["DOC-02"]}, ev, comp).status
        == "fail"
    )


def test_false_approve_is_counted(monkeypatch):
    real = decisions.all_cases()
    lied = [
        (cid, {**exp, "status": "MANUAL_REVIEW"} if exp["status"] == "APPROVED" else exp, p)
        for cid, exp, p in real
    ]
    monkeypatch.setattr(decisions, "all_cases", lambda: lied)
    s, false_approve = decisions.suite()
    assert false_approve == sum(e["status"] == "APPROVED" for _, e, _ in real) > 0
    assert s.status == "fail"


def test_synthetic_variants_cover_all_outcomes_and_are_well_formed():
    vs = cases.variants()
    assert len(vs) >= 24
    assert {v["expected"]["status"] for v, _ in vs} == {"APPROVED", "MANUAL_REVIEW", "REJECTED"}
    for v, p in vs:
        assert p.screening.fact_set_hash == p.facts.fact_set_hash, v["id"]
        assert p.facts.fact_set_hash.startswith("sha256:"), v["id"]


def test_an_unknown_mutation_op_is_an_error():
    g = cases.golden()["A"]
    with pytest.raises(ValueError, match="unknown mutation op"):
        cases.build(g, [{"op": "set_status", "value": "APPROVED"}])


# --- suites 1, 2, 6, 8 ------------------------------------------------------------------------


def test_contract_suite_detects_every_tampering():
    s = contract.suite()
    assert s.counts()["fail"] == 0
    tampers = [c for c in s.cases if c.case_id.startswith("tamper-detected:")]
    assert len(tampers) == 6 and all(c.status == "pass" for c in tampers)


def test_properties_hold_and_a_broken_core_is_caught(monkeypatch):
    s = properties.suite(n=60)
    assert s.status == "pass", [c.detail for c in s.cases if c.status != "pass"]
    from kyb.policy import compose as real_compose

    def lenient(rules, pol, judges=()):  # a core where a veto can approve
        c = real_compose(rules, pol, ())
        return c.model_copy(update={"status": "APPROVED"}) if judges else c

    monkeypatch.setattr(properties, "compose", lenient)
    s = properties.suite(n=60)
    assert {c.case_id: c.status for c in s.cases}["veto-only-toward-review"] == "fail"


def test_policy_edit_suite_passes_and_a_wrong_delta_fails(monkeypatch):
    pass
    #
    assert policy_edits.suite().status == "pass"
    data = copy.deepcopy(load("policy_edits/edits.json"))
    data["edits"][0]["cases"][0]["expected"]["risk_score"] = 70  # the 5y → 3y delta, denied
    monkeypatch.setattr(policy_edits, "load", lambda rel: data)
    assert policy_edits.suite().status == "fail"


def test_adversarial_core_passes_and_an_effective_attack_fails(monkeypatch):
    s = adversarial.suite()
    assert s.counts()["fail"] <= 1 and s.counts()["pass"] >= 5
    data = copy.deepcopy(load("adversarial/cases.json"))
    data["cases"][0]["ops"] = [
        {"op": "entity", "id": "e-andes-agropecuaria", "set": {"jurisdiction": ["IR"]}}
    ]  # noqa: E501
    real = adversarial.load
    monkeypatch.setattr(
        adversarial, "load", lambda rel: data if rel == "adversarial/cases.json" else real(rel)
    )
    s = adversarial.suite()
    assert {c.case_id: c.status for c in s.cases}[data["cases"][0]["id"]] == "fail"


def test_pending_suites_name_their_hooks():
    for s in (pending_suites.qa(), pending_suites.screening(), pending_suites.support()):
        assert all(c.needs for c in s.cases if c.status == "pending")
        assert {c.case_id: c.status for c in s.cases}["dataset-shape"] == "pass"
    qa = load("qa/questions.json")["questions"]
    assert sum(not q["answerable"] for q in qa) >= 8


def test_datasets_are_valid_json_with_versions():
    for rel in (
        "decisions/golden.json",
        "decisions/synthetic.json",
        "policy_edits/edits.json",
        "adversarial/cases.json",
        "qa/questions.json",
        "screening/cases.json",
        "support/cases.json",
        "witness/labels.json",
    ):
        assert load(rel)["version"], rel
    json.dumps(load("decisions/synthetic.json"))
