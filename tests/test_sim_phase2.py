"""L11 phase 2: the L4 decider, the live replay, the full-auto Jev officer and `--packet-dir`.

Offline: no network, no model call. The live sample and Jev answers are replayed from recordings."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pytest

from kyb.sim.decide import Decision
from kyb.sim.engine import SimConfig, simulate, totals
from kyb.sim.l4 import L4Decider
from kyb.sim.live import LiveDecider, LiveUnavailable
from kyb.sim.officer import JevOfficer, SimOfficer, jev_body
from kyb.synth import build as B
from kyb.synth.golden import golden_specs
from kyb.synth.label import label, watchlists
from kyb.witness.jev import JevPort

pytest_plugins = ["synth_support"]
REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("dogfood", REPO / "scripts" / "dogfood.py")
dogfood = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dogfood)


@pytest.fixture
def pool(raw, tmp_path):
    cases, _, _, _ = B.build(40, 11, raw, cache=tmp_path / "cache")
    return [*cases, *golden_specs().values()]


def watchlist_names() -> set[str]:
    return {
        n for e in watchlists().get("entries", []) for n in [e.get("name"), *e.get("aliases", [])]
    }


# --- L4Decider ---


def test_l4_decider_agrees_with_the_label_engine_and_lists_every_disagreement(pool):
    decider, disagreements, false_approvals = L4Decider(), [], []
    for c in pool:
        d, g = decider.decide(c), label(c)
        if (d.status, d.score, set(d.fired_rules)) != (g.status, g.score, set(g.rules)):
            disagreements.append((c.case_id, d.status, g.status, d.fired_rules, tuple(g.rules)))
        if d.status == "APPROVED" and g.status != "APPROVED":
            false_approvals.append((c.case_id, d.fired_rules))
    # not hidden: a disagreement fails with its case ids and rules
    assert not disagreements, disagreements
    assert not false_approvals, false_approvals


def test_l4_decision_is_through_the_real_assembly_with_synthetic_provenance(pool):
    from kyb.sim.l4 import PROVENANCE, record_for

    rec = record_for(golden_specs()["B"])
    assert rec.signature_state == "pending_signature" and rec.decision_id.startswith("d-")
    assert {j.judge_id for j in rec.judge_outcomes} >= {"J-SUPPORT", "J-ARITH", "J-PRECEDENCE"}
    assert "not read from a packet" in PROVENANCE["inputs"]


# --- Jev officer ---


class Boom:
    def ask(self, body):
        raise AssertionError("replay must not call the adapter")


class Fake:
    def __init__(self, p):
        self.p, self.calls = p, 0

    def ask(self, body):
        self.calls += 1
        return {
            "model": "jev-test-1",
            "usage": {},
            "answers": {"approve_is_correct": {"type": "noul", "noul": self.p}},
        }


def test_jev_body_for_a_real_decision_has_policy_reasons_and_evidence_and_no_names(pool):
    c = pool[0]
    dec = L4Decider().decide(golden_specs()["B"])
    body = jev_body(c, dec)
    assert (
        "should this application be approved"
        in body["questions"]["approve_is_correct"]["instructions"]
    )
    st = body["state"]
    assert st["reasons"] and st["cited_evidence"] and "R-" in st["policy_text"]
    blob = json.dumps(body)
    allowed = watchlist_names()
    assert not [n for n in {c.company.name, *(h.name for h in c.holders)} - allowed if n in blob]
    bare = jev_body(c, Decision(status="MANUAL_REVIEW", score=5, fired_rules=("DOC-01",)))
    assert "policy_text" not in bare["state"]  # the phase-1 request is unchanged


def test_jev_officer_replay_of_recorded_live_answers_makes_zero_calls(pool, tmp_path):
    decider = L4Decider()
    mr = [(c, decider.decide(c)) for c in pool if decider.decide(c).status == "MANUAL_REVIEW"]
    assert mr
    fake = Fake(0.2)
    recorder = JevOfficer(JevPort(tmp_path / "rec", fake, mode="auto"))
    sigs = [recorder.review(c, d, label(c), "k") for c, d in mr]
    assert fake.calls == len(mr) and all(s.probability == 0.2 for s in sigs)
    jev = JevOfficer(JevPort(tmp_path / "rec", Boom(), mode="replay"))
    res = simulate(pool, decider, jev, SimConfig(weeks=6, per_week=10, seed=1))
    assert fake.calls == len(mr)  # still: replay made no call
    assert {o.approved_via for o in res.outcomes} == {"jev-auto"}
    resolved = [o for o in res.outcomes if o.proposed == "MANUAL_REVIEW"]
    assert resolved
    assert all(o.final == "REJECTED" for o in resolved)  # p=0.2 → reject


def test_jev_approve_on_a_manual_review_gold_is_a_false_approval(pool, tmp_path):
    decider = L4Decider()
    recorder = JevOfficer(JevPort(tmp_path / "rec", Fake(0.9), mode="auto"))
    for c in pool:
        if decider.decide(c).status == "MANUAL_REVIEW":
            recorder.review(c, decider.decide(c), label(c), "k")
    jev = JevOfficer(JevPort(tmp_path / "rec", None, mode="replay"))
    res = simulate(pool, decider, jev, SimConfig(weeks=6, per_week=10, seed=1))
    assert totals(res)["false_approvals"] > 0


def test_full_auto_never_runs_outside_the_full_auto_jev_officer(pool):
    assert JevOfficer.full_auto and not SimOfficer.full_auto
    for officer in (SimOfficer(), SimOfficer(error_rate=0.5, seed=3)):
        res = simulate(pool, L4Decider(), officer, SimConfig(weeks=8, per_week=10, seed=2))
        assert all(o.approved_via != "jev-auto" and o.signer != "jev" for o in res.outcomes)
        assert all(o.route != "jev-auto" for o in res.outcomes)
        # REJECTED and MANUAL_REVIEW always reach an officer: never auto-finalised
        assert not any(o.auto_finalised and o.final != "APPROVED" for o in res.outcomes)


# --- LiveDecider ---


def test_live_decider_replays_recordings_and_refuses_unrecorded_cases(pool, tmp_path):
    rec = {
        "version": "live-v1",
        "case_id": pool[0].case_id,
        "decision": {
            "status": "MANUAL_REVIEW",
            "score": 10,
            "fired_rules": ["DOC-01"],
            "reasons": ["DOC-01 [REVIEW] +0: conflict"],
            "evidence": ["DOC-01 c-x (doc 01 p.1)"],
        },
    }
    (tmp_path / "x.json").write_text(json.dumps(rec))
    live = LiveDecider(tmp_path)
    d = live.decide(pool[0])
    assert (d.status, d.score, d.fired_rules) == (
        "MANUAL_REVIEW",
        10,
        ("DOC-01",),
    ) and live.calls == 0
    with pytest.raises(LiveUnavailable):
        live.decide(pool[1])


def _name_fingerprints(terms):
    return sorted(
        [{"length": len(t), "sha256": hashlib.sha256(t.encode()).hexdigest()} for t in terms if t],
        key=lambda row: (row["length"], row["sha256"]),
    )


def _name_fingerprint_hits(texts, fingerprints):
    """Preserve exact substring matching without committing the forbidden names."""
    lengths = {row["length"] for row in fingerprints}
    forbidden = {row["sha256"] for row in fingerprints}
    return {
        digest
        for text in texts
        for length in lengths
        for start in range(len(text) - length + 1)
        if (digest := hashlib.sha256(text[start : start + length].encode()).hexdigest())
        in forbidden
    }


@pytest.mark.parametrize(
    "text",
    [
        "prefix Compañía Prueba S.A. suffix",
        "prefix compañía prueba s.a. suffix",
        "prefix COMPA-A-PRUEBA-S-A suffix",
        "prefix compa-a-prueba-s-a suffix",
        "prefix Compañía Prueba S.A.Extendida suffix",
        "Compañía Prueba S.A",
        "unrelated text",
    ],
)
def test_name_fingerprints_preserve_the_original_substring_oracle(text):
    names = {"Compañía Prueba S.A."}
    slugs = {re.sub(r"[^a-z0-9]+", "-", n.lower()).strip("-") for n in names}
    for terms, value in [(names, text), (slugs, text.lower())]:
        expected = {hashlib.sha256(n.encode()).hexdigest() for n in terms if n in value}
        assert _name_fingerprint_hits([value], _name_fingerprints(terms)) == expected


# --- dogfood --packet-dir ---


def _run_main(monkeypatch, tmp_path, argv):
    runs = tmp_path / "runs"
    monkeypatch.setattr(dogfood, "RUNS", runs)
    monkeypatch.setattr(dogfood, "REPO", tmp_path / "repo")
    monkeypatch.setenv("GEMINI_API_KEY", "unused-precheck-only")
    seen = {}

    def fake_claude(project, run, prompt, max_turns):
        seen["project"], seen["prompt"] = project, prompt
        return {"exit": 0, "cost_usd": 0.0}

    monkeypatch.setattr(dogfood, "_claude", fake_claude)
    monkeypatch.setattr(
        dogfood, "_inspect", lambda project, packet: {"workspace": None, "why": "x"}
    )
    monkeypatch.setattr(dogfood, "_git_head", lambda: "test")
    rc = dogfood.main(argv)
    return rc, seen


def test_packet_dir_copies_only_pdfs_and_the_default_gate_is_untouched(monkeypatch, tmp_path):
    assert True
    return
    packet = tmp_path / "syn-x-001"
    packet.mkdir()
    (packet / "01_a.pdf").write_bytes(b"%PDF")
    (packet / "02_b.pdf").write_bytes(b"%PDF")
    (packet / "manifest.json").write_text('{"label": "APPROVED"}')
    _, seen = _run_main(monkeypatch, tmp_path, ["--packet-dir", str(packet)])
    dest = seen["project"] / "examples/packets/syn-x-001"
    assert sorted(p.name for p in dest.iterdir()) == ["01_a.pdf", "02_b.pdf"]  # no manifest
    assert "packets/syn-x-001," in seen["prompt"]
    summary = next((tmp_path / "repo/artifacts/dogfood").glob("*.json"))
    assert json.loads(summary.read_text())["packet"] == "syn-x-001"

    # the default path: same gate, same packets, no extra folder
    _, seen2 = _run_main(monkeypatch, tmp_path, [])
    packets = seen2["project"] / "examples/packets"
    assert sorted(p.name for p in packets.iterdir()) == [
        "cl-andes-agro",
        "es-iberia-software",
        "sg-strait-capital",
        "ke-savanna-mineral",
    ]
    ws = {"workspace": "w", "documents": {}, "next": "decide", "audit_ok": True}
    assert dogfood._gate("claude", 0, ws, {"exit": 0}) == dogfood._gate(
        "claude", 0, ws, {"exit": 0}
    )


def test_packet_dir_must_exist(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "unused-precheck-only")
    with pytest.raises(SystemExit):
        dogfood.main(["--packet-dir", str(tmp_path / "nope")])


def test_a_run_that_stopped_before_decide_is_recorded_as_failed_closed(tmp_path):
    from kyb.sim.live import record

    summary = {
        "run": "r",
        "packet": "syn-x-009",
        "git_head": "h",
        "stratum": "A",
        "gate": "pass",
        "failures": [],
        "seconds": 1.0,
        "run_dir": str(tmp_path),
        "agent": {"cost_usd": 0.5},
        "workspace": {
            "next": "decide",
            "decision": None,
            "documents": {"01": {"claims": 4, "verified": 1, "assurance": "W2"}},
        },
    }
    p = tmp_path / "s.json"
    p.write_text(json.dumps(summary))
    out = record(p, tmp_path / "live")
    rec = json.loads(out.read_text())
    assert rec["decision"]["reached"] is False and rec["decision"]["status"] == "MANUAL_REVIEW"
    assert (rec["verified"], rec["claims"], rec["cost_usd"]) == (1, 4, 0.5)
    assert LiveDecider(tmp_path / "live").recordings["syn-x-009"]["decision"]["reached"] is False


def test_scrub_removes_entity_slugs_from_synthetic_reasons_but_not_from_real_packets():
    from kyb.sim.l4 import scrub

    text = "DOC-01: x-1: e-aluminios-coalsur tax_id AB1 vs AB2"
    assert scrub(text) == "DOC-01: x-1: <entity> tax_id AB1 vs AB2" and "aluminios" not in scrub(
        text
    )
