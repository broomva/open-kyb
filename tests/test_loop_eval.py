"""Evaluator tests for the evidence-loop matrix (`kyb.evals.loop_eval`).

These test the evaluator itself: fixtures against the existing FactSet contract, pending-not-pass,
the bypass self-check, label sensitivity and accounting. They use explicit doubles and are not
product proof. The real planner is exercised only by the parent-owned integrated run.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path

import pytest

from kyb.contracts.b4 import FactSet
from kyb.evals import loop_eval as m
from kyb.facts.derive import fact_set_hash
from kyb.util.hashing import canonical_json

REPO = Path(__file__).resolve().parents[1]


def _fixture(case_id: str) -> m.Fixture:
    return next(f for f in m.FIXTURES if f.case_id == case_id)


def _plan(fx: m.Fixture, **over) -> dict:
    plan = {
        "state": "ready",
        "gaps": [],
        "candidates": [],
        "fact_set_hash": fx.facts["fact_set_hash"],
    }
    plan.update(over)
    return plan


def _run(fx: m.Fixture, plan: dict) -> dict:
    return {"plan": plan, "again": plan, "baseline": plan, "before": fx.facts,
            "current": fx.after or fx.facts}  # fmt: skip


def test_fixtures_validate_against_the_existing_factset_contract():
    for f in m.FIXTURES:
        for doc in (f.facts, f.after):
            if doc is None:
                continue
            model = FactSet.model_validate(doc)
            assert fact_set_hash(model) == doc["fact_set_hash"], f.case_id


def test_only_the_forged_fixture_has_unresolved_evidence_refs():
    unresolved = set()
    for f in m.FIXTURES:
        doc = f.after or f.facts
        ids = {e["evidence_id"] for e in doc["evidence"]}
        refs = [r for e in doc["entities"] for r in e["evidence"]]
        refs += [r for h in doc["holdings"] for r in h["evidence"]]
        refs += [r for d in doc["documents"] for r in d["evidence"]]
        if any(r["claim_id"] not in ids for r in refs):
            unresolved.add(f.case_id)
    assert unresolved == {"S05-forged-evidence-ref"}


def test_case_ids_are_unique_and_every_group_is_in_a_suite():
    ids = [f.case_id for f in m.FIXTURES]
    assert len(ids) == len(set(ids))
    assert {f.group for f in m.FIXTURES} <= {grp for _, _, grp in m.SUITES}


def test_missing_core_is_pending_and_exits_two(monkeypatch):
    monkeypatch.setitem(sys.modules, "kyb.loop.planner", None)
    with pytest.raises(m.CoreMissing):
        m.load_core()
    doc = m.run_matrix(core=None)
    assert doc["matrix_verdict"] == "pending"
    assert doc["product_pass"] is False
    assert m.exit_code(doc) == 2
    planner_cases = [c for s in doc["suites"] for c in s["cases"] if c["id"] in
                     {f.case_id for f in m.FIXTURES}]  # fmt: skip
    assert planner_cases and all(c["status"] == "pending" for c in planner_cases)


def test_bypass_planner_cannot_score_green():
    doc = m.run_matrix(core=m._bypass_core())
    assert doc["selfcheck"]["ran"] is True
    assert doc["selfcheck"]["missed"] == []
    assert doc["selfcheck"]["must_fail"] > 0
    assert doc["matrix_verdict"] == "fail"


def test_bypass_double_would_pass_only_fixtures_that_do_not_require_a_planner_output():
    failing = {}
    for f in m.FIXTURES:
        run = m.run_planner(m._bypass_core(), f)
        failing[f.case_id] = m.score(f, run).status
    for f in m.FIXTURES:
        if m._requires_non_ready(f):
            assert failing[f.case_id] == "fail", f.case_id


def test_mutating_an_expectation_turns_a_pass_into_a_fail():
    fx = _fixture("V01-missing-jurisdiction-repairable")
    gap = {
        "gap_id": "g1",
        "kind": "jurisdiction",
        "entity_ids": ["e-t1"],
        "doc_ids": ["02"],
        "field": "jurisdiction",
        "repairable": True,
        "reason": "",
        "evidence_ids": [],
    }
    plan = _plan(fx, state="repair", gaps=[gap])
    assert m.score(fx, _run(fx, plan)).status == "pass"
    mutated = dataclasses.replace(fx, expect={**fx.expect, "state": "ready"})
    assert m.score(mutated, _run(mutated, plan)).status == "fail"


def test_same_name_pair_with_no_candidate_is_a_fail_not_a_vacuous_pass():
    fx = _fixture("S01-same-name-diff-country")
    assert m.score(fx, _run(fx, _plan(fx))).status == "fail"


def test_supported_candidate_without_applied_false_is_a_false_promotion():
    fx = _fixture("C01-true-supported-identity")
    cand = {
        "left": "e-p1",
        "right": "e-p2",
        "state": "supported",
        "reason": "r",
        "evidence_ids": [],
    }
    unrecorded = m.score(fx, _run(fx, _plan(fx, candidates=[cand])))
    assert unrecorded.status == "fail"
    assert unrecorded.data["breach"] == "false_promotion"
    recorded = m.score(fx, _run(fx, _plan(fx, candidates=[{**cand, "applied_to_factset": False}])))
    assert recorded.status == "pass"


def test_forbidden_ready_is_counted_as_a_false_accept():
    fx = _fixture("S06-unverified-claim-input")
    got = m.score(fx, _run(fx, _plan(fx, state="ready")))
    assert got.status == "fail"
    assert got.data["breach"] == "false_accept"


def test_planner_exception_is_a_failure_not_a_pending_pass():
    def raises(*_a, **_k):
        raise ValueError("boom")

    core = m.Core(raises, lambda **kw: kw, None)
    doc = m.run_matrix(core=core, fixtures=[_fixture("C02-fully-covered-ready")], selfcheck=False)
    assert doc["matrix_verdict"] == "fail"
    assert "boom" in canonical_json(doc).decode()


def test_planner_stub_is_pending_not_failure():
    def stub(*_a, **_k):
        raise NotImplementedError("lane A")

    core = m.Core(stub, lambda **kw: kw, None)
    doc = m.run_matrix(core=core, fixtures=[_fixture("C02-fully-covered-ready")], selfcheck=False)
    assert doc["matrix_verdict"] == "pending"


def test_unknown_usage_is_never_recorded_as_zero():
    unmetered = m.cost_metric({"cost_usd": {"status": "measured", "value": 0.0,
                                            "provenance": "flat_quota"}})  # fmt: skip
    assert unmetered["status"] == "unknown"
    assert unmetered["value"] is None
    billed = m.cost_metric({"cost_usd": {"status": "measured", "value": 1.5,
                                         "provenance": "billed_amount"}})  # fmt: skip
    assert billed["status"] == "measured"
    missing = m.usage_metric(None, "tokens")
    assert missing == {"status": "unknown", "value": None, "provenance": "not_supplied"}


def test_cost_per_gain_is_undefined_when_gain_is_zero_or_cost_unknown():
    row = {"verified_gain": 0, "usage": {k: {"status": "unknown", "value": None, "provenance": ""}
                                         for k in m.METRICS}}  # fmt: skip
    assert m._cost([row])["cost_per_verified_gain"]["value"] is None
    row["verified_gain"] = 2
    per_gain = m._cost([row])["cost_per_verified_gain"]
    assert per_gain["value"] is None
    assert "unknown" in per_gain["reason"]


def test_verified_gain_counts_only_verified_typed_inputs():
    before = _fixture("V01-missing-jurisdiction-repairable").facts
    after = _fixture("V02-fresh-run-closes-gap").after
    assert len(m.verified_inputs(after) - m.verified_inputs(before)) == 1
    forged = _fixture("S05-forged-evidence-ref").facts
    assert ("holding", "e-h3", "e-app", "50", "None") not in m.verified_inputs(forged)


def test_receipt_is_byte_deterministic_for_identical_inputs():
    first = m.run_matrix(core=m._bypass_core(), code_version="test")
    second = m.run_matrix(core=m._bypass_core(), code_version="test")
    assert canonical_json(first) == canonical_json(second)


def test_script_exit_status_and_receipt_for_a_missing_core(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "kyb.loop.planner", None)
    spec = importlib.util.spec_from_file_location(
        "eval_evidence_loop", REPO / "scripts" / "eval_evidence_loop.py"
    )
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    assert script.main(["--out", str(tmp_path)]) == 2
    receipt = (tmp_path / "evidence-loop-eval.json").read_text("utf-8")
    assert '"matrix_verdict": "pending"' in receipt
    assert '"product_pass": false' in receipt
    assert "pending action-layer cases" in (tmp_path / "summary.txt").read_text("utf-8")
