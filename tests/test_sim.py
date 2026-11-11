"""L11 autonomy-progression simulator. Offline: no network, no model call."""

from __future__ import annotations

import json

import pytest

from kyb.sim.decide import Decision, FaultyDecider, FixtureDecider, LabelDecider
from kyb.sim.engine import SimConfig, arrivals, by_week, profile_key, simulate, totals
from kyb.sim.officer import (
    JevOfficer,
    ScriptedJev,
    Signature,
    SimOfficer,
    jev_body,
    record_fixture,
)
from kyb.sim.report import build_report, markdown, write
from kyb.synth import build as B
from kyb.synth.golden import golden_specs
from kyb.synth.label import label
from kyb.witness.jev import JevPort

pytest_plugins = ["synth_support"]


@pytest.fixture
def pool(raw, tmp_path):
    cases, _, _, _ = B.build(40, 11, raw, cache=tmp_path / "cache")
    return cases


class Fixed:
    """A decider that returns one decision for every case (a scripted decider for the dial)."""

    def __init__(self, status="APPROVED", rules=()):
        self.name = f"fixed-{status}"
        self.d = Decision(status=status, score=0, fired_rules=tuple(rules))

    def decide(self, case):
        return self.d


class Agree(SimOfficer):
    """An officer who signs whatever the decider proposed."""

    name = "always-agree"

    def review(self, case, proposal, gold, key):
        return Signature(
            status=proposal.status,
            signer="officer",
            approved_via="officer",
            latency_h=self.latency_h,
        )


def one_profile_pool():
    a = golden_specs()["A"]
    return [a.model_copy(update={"case_id": f"x-{i}"}) for i in range(3)]


# --- arrivals, determinism ------------------------------------------------------------------------


def test_arrivals_are_weekly_seeded_and_reproducible(pool):
    cfg = SimConfig(weeks=8, per_week=10, seed=5)
    a, b = arrivals(pool, cfg), arrivals(pool, cfg)
    assert [(w, i, c.case_id) for w, i, c in a] == [(w, i, c.case_id) for w, i, c in b]
    assert {w for w, _, _ in a} == set(range(1, 9))
    assert all(8 <= sum(1 for w, _, _ in a if w == k) <= 12 for k in range(1, 9))
    assert [c.case_id for _, _, c in arrivals(pool, SimConfig(seed=6))] != [
        c.case_id for _, _, c in a
    ]


def test_simulation_is_reproducible(pool):
    cfg = SimConfig(seed=3)
    r1 = simulate(pool, LabelDecider(), SimOfficer(), cfg)
    r2 = simulate(pool, LabelDecider(), SimOfficer(), cfg)
    assert build_report(r1, "x") == build_report(r2, "x")


# --- the dial ---


def test_rejected_is_never_auto_finalised_outside_full_auto_jev(pool):
    for decider in (LabelDecider(), FixtureDecider(), FaultyDecider(FixtureDecider(), 0.3, 1)):
        res = simulate(pool, decider, SimOfficer(), SimConfig(weeks=12, per_week=15, k=2, seed=2))
        rejected = [o for o in res.outcomes if o.proposed == "REJECTED"]
        assert rejected and all(o.blocking_review and not o.auto_finalised for o in rejected)
        assert totals(res)["rejected_auto_finalised"] == 0
        # nothing but APPROVED proposals is ever auto-finalised
        assert {o.proposed for o in res.outcomes if o.auto_finalised} <= {"APPROVED"}


def test_promotion_after_k_agreements_then_level_2_then_demotion():
    pool = one_profile_pool()
    cfg = SimConfig(weeks=6, per_week=10, k=2, audit_rate=1.0, seed=1)
    res = simulate(pool, Fixed("APPROVED"), Agree(), cfg)
    key = profile_key(pool[0], Fixed().d)
    first = [o for o in res.outcomes if o.profile == key]
    assert [o.level for o in first[:5]] == [0, 0, 1, 1, 2]  # K=2 agreements → L1, K=2 audits → L2
    assert [o.auto_finalised for o in first[:3]] == [False, False, True]
    assert all(o.audited for o in first[2:])  # audit_rate 1.0
    ttp = totals(res)["time_to_promote_weeks"][key]
    assert [t["to_level"] for t in ttp] == [1, 2]

    # an officer override demotes the profile and records a precedent (context, not an outcome)
    class Overrides(SimOfficer):
        def review(self, case, proposal, gold, key):
            n = getattr(self, "n", 0) + 1
            self.n = n
            status = "MANUAL_REVIEW" if n == 4 else proposal.status
            return Signature(
                status=status,
                signer="officer",
                approved_via="officer",
                rationale="needs a source-of-funds document" if n == 4 else None,
            )

    res = simulate(pool, Fixed("APPROVED"), Overrides(), cfg)
    out = [o for o in res.outcomes if o.profile == key]
    assert res.precedents and res.precedents[0].profile == key
    assert res.precedents[0].rationale == "needs a source-of-funds document"
    assert (res.precedents[0].from_status, res.precedents[0].to_status) == (
        "APPROVED",
        "MANUAL_REVIEW",
    )
    demoted_at = next(i for i, o in enumerate(out) if o.override)
    assert out[demoted_at + 1].level == 0 and not out[demoted_at + 1].auto_finalised
    assert out[demoted_at + 1].precedents_considered == 1  # the next case sees it, as context
    assert totals(res)["demotions"] == len(res.precedents) >= 1


def test_false_approval_accounting():
    # a decider that approves a case gold says needs review, and an officer who agrees with it
    pool = one_profile_pool()
    pool = [
        p.model_copy(
            update={
                "company": p.company.model_copy(
                    update={"mass_registration_address": True, "operating_address": None}
                )
            }
        )
        for p in pool
    ]  # gold: MANUAL_REVIEW (OPS-01 + DOC-02)
    assert label(pool[0]).status == "MANUAL_REVIEW"
    cfg = SimConfig(weeks=4, per_week=10, k=2, audit_rate=1.0, seed=1)
    res = simulate(pool, Fixed("APPROVED"), Agree(), cfg)
    t = totals(res)
    auto = [o for o in res.outcomes if o.auto_finalised]
    assert auto and t["false_approvals"] == len(
        auto
    )  # every auto-approval of a gold-MR case counts
    assert all(o.false_approval for o in auto)
    assert not any(o.false_approval for o in res.outcomes if not o.auto_finalised)
    assert sum(r["false_approvals"] for r in by_week(res)) == t["false_approvals"]
    # with a real officer auditing, the first false approval is caught and the profile demoted
    res2 = simulate(pool, Fixed("APPROVED"), SimOfficer(), cfg)
    assert totals(res2)["false_approvals"] == 0  # officer overrides at L0, so it never promotes
    assert totals(res2)["demotions"] == len(res2.outcomes)


def test_false_rejects_are_counted(pool):
    approved = [c for c in pool if label(c).status == "APPROVED"][:2]
    res = simulate(
        approved,
        Fixed("REJECTED"),
        SimOfficer(error_rate=0.0),
        SimConfig(weeks=2, per_week=5, seed=1),
    )
    assert totals(res)["false_rejects"] == 0  # the officer overrides the wrong REJECT
    res = simulate(approved, Fixed("REJECTED"), Agree(), SimConfig(weeks=2, per_week=5, seed=1))
    assert totals(res)["false_rejects"] == totals(res)["arrivals"]


def test_label_decider_smoke_has_zero_false_approvals(pool):
    res = simulate(pool, LabelDecider(), SimOfficer(), SimConfig(weeks=8, per_week=20, seed=4))
    t = totals(res)
    assert t["false_approvals"] == 0 and t["agreement_rate"] == 1.0 and t["auto_finalised"] > 0


# --- Jev ---


def test_jev_officer_replay_makes_zero_calls_and_marks_every_signature(pool, tmp_path):
    decisions = {c.case_id: (c, LabelDecider().decide(c)) for c in pool}
    gold = {k: label(c).status for k, (c, _) in decisions.items()}
    recorded = record_fixture(tmp_path / "rec", decisions, gold, error_rate=0.0, seed=1)
    assert recorded == sum(d.status == "MANUAL_REVIEW" for _, d in decisions.values()) > 0

    class Counting:
        calls = 0

        def ask(self, body):
            Counting.calls += 1
            raise AssertionError("replay must not call the adapter")

    jev = JevOfficer(JevPort(tmp_path / "rec", Counting(), mode="replay"))
    res = simulate(pool, LabelDecider(), jev, SimConfig(weeks=6, per_week=10, seed=1))
    assert Counting.calls == 0
    assert {o.approved_via for o in res.outcomes} == {"jev-auto"}
    assert all(o.signer == "jev" for o in res.outcomes)
    mr = [o for o in res.outcomes if o.proposed == "MANUAL_REVIEW"]
    assert mr and all(o.final in ("APPROVED", "REJECTED") for o in mr)  # every MR resolved
    assert totals(res)["false_approvals"] == 0  # a perfect scripted Jev
    # the full-auto profile is the only one that auto-finalises REJECTED
    assert any(o.auto_finalised and o.final == "REJECTED" for o in res.outcomes)


def test_jev_errors_surface_as_false_approvals(pool, tmp_path):
    decisions = {c.case_id: (c, LabelDecider().decide(c)) for c in pool}
    gold = {k: label(c).status for k, (c, _) in decisions.items()}
    record_fixture(tmp_path / "rec", decisions, gold, error_rate=1.0, seed=1)  # always wrong
    jev = JevOfficer(JevPort(tmp_path / "rec", None, mode="replay"))
    res = simulate(pool, LabelDecider(), jev, SimConfig(weeks=6, per_week=10, seed=1))
    assert totals(res)["false_approvals"] > 0  # Jev approved MR cases whose gold is not APPROVED


def test_missing_recording_is_unresolved_not_a_guess(pool, tmp_path):
    jev = JevOfficer(JevPort(tmp_path / "empty", None, mode="replay"))
    res = simulate(pool, LabelDecider(), jev, SimConfig(weeks=3, per_week=10, seed=1))
    mr = [o for o in res.outcomes if o.proposed == "MANUAL_REVIEW"]
    assert mr and all(o.final == "MANUAL_REVIEW" and not o.auto_finalised for o in mr)
    assert all(o.approved_via == "jev-auto" for o in res.outcomes)


def test_jev_request_carries_no_names(pool):
    c = pool[0]
    body = json.dumps(
        jev_body(c, Decision(status="MANUAL_REVIEW", score=5, fired_rules=("DOC-01",)))
    )
    assert c.company.name not in body and c.holders[0].name not in body and c.case_id in body
    assert ScriptedJev.MODEL != "jev-latest"


# --- reports ---


def test_report_has_headline_week_table_chart_and_case_ids_only(pool, tmp_path):
    res = simulate(pool, FixtureDecider(), SimOfficer(), SimConfig(weeks=8, per_week=10, seed=1))
    out = write(res, "t", tmp_path)
    md = (out / "report.md").read_text()
    js = (out / "report.json").read_text()
    assert md.index("false approvals") < md.index("auto rate")  # the headline comes first
    assert "```mermaid" in md and md.count("\n| ") >= 9  # one row per week plus the header
    names = {c.company.name for c in pool} | {h.name for c in pool for h in c.holders}
    assert not [n for n in names if n in md or n in js]
    assert json.loads(js)["headline"]["false_approvals"] == totals(res)["false_approvals"]
    assert markdown(build_report(res, "t")) == md
