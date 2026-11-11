"""J-ANOMALY and L4's approval guards over L8's real witness records.

Records come from L8's own `verify_document` routing (strata, attempts, self_referential,
routing_config); only the per-claim verification and the transport are faked. A healed
self-heal is the shape `StratumWitness` writes: an `anomalous` attempt before a `complete` one.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_decide import POLICY, attested_fixture_answers, fixture_run
from test_witness_strata import SUB, Transport, attest, fake_verification

from kyb.audit.chain import workspace_action
from kyb.contracts.b4 import DocVerification
from kyb.decide import action
from kyb.decide.core import decision_record
from kyb.evals.cases import Packet, build, golden, variants
from kyb.judges.core import ANOMALY, APPROVAL_ASSURANCE, fact_judges, outcome, witness_judges
from kyb.policy import load_policy
from kyb.run.records import write_record
from kyb.witness import verify
from kyb.witness.strata import Config, RoutedWitness

ANOMALY_ENTRY = {
    "path": "candidates[0].groundingMetadataV2",
    "key": "groundingMetadataV2",
    "value": "{}",
    "severity": "unknown",
    "frame": None,
    "provider": "vertex-gcp",
    "requested_model": "gemini-3.1-pro-preview",
    "served_model": "gemini-3.1-pro-preview",
}


def healed(attempts: list[dict]) -> list[dict]:
    """One anomalous attempt that self-healed on the retry, as StratumWitness records it."""
    first = attempts[0]
    return [
        {"backend": first["backend"], "status": "anomalous", "attempt": 0,
         "anomalies": [dict(ANOMALY_ENTRY)]},
        {**first, "attempt": 1},
        *attempts[1:],
    ]  # fmt: skip


def variant(label_id: str):
    if label_id in "ABC":
        return build(golden()[label_id])
    return next(p for label, p in variants() if label["id"] == label_id)


def witnessed(tmp_path, monkeypatch, name, packet, *, stratum="A", anomalous=(), b_only=False):
    """A fixture run whose documents carry real L8 witness records. Unrelated guards (reader
    lifecycle, fixture producer) are stubbed passing so only witness judges can cap."""
    facts = packet.facts.model_copy(
        update={
            "verification": [
                DocVerification(doc=d.doc, assurance="W2", isolated=True)
                for d in packet.facts.documents
            ]
        }
    )
    ws, packet, work = fixture_run(tmp_path, name, Packet(facts, packet.screening, packet.answers))
    (tmp_path / name).mkdir()
    attested_fixture_answers(ws, packet, work, tmp_path / name)
    monkeypatch.setattr(verify, "load_layout", lambda *a: SimpleNamespace(material_facts=[]))
    monkeypatch.setattr(verify, "_verify_document", fake_verification)
    case = packet.facts.case
    with workspace_action(ws):
        attest(ws)
        for d in packet.facts.documents:
            write_record(ws, f"silver/claims/{case}/r1/{d.doc}.json", SUB,
                         action="fixture_claims", tier="claimed", as_of=facts.as_of)  # fmt: skip
            router = RoutedWitness(ws, config=Config(stratum), factory=Transport)
            rec = verify.verify_document(ws, case, "r1", d.doc, router, Path("x"))
            if d.doc in anomalous:
                if rec.get("strata"):
                    for s in ["B"] if b_only else ["A", "B"]:
                        rec["strata"][s]["attempts"] = healed(rec["strata"][s]["attempts"])
                else:
                    rec["attempts"] = healed(rec["attempts"])
            write_record(ws, f"silver/witness/{case}/r1/{d.doc}.json", rec,
                         action="verify_quotes", tier="verified", as_of=facts.as_of)  # fmt: skip
    monkeypatch.setattr(action, "_reader_judges", lambda *args: [])
    monkeypatch.setattr(action, "fixture_inputs", lambda *args: [])
    monkeypatch.setattr(
        action,
        "support_checker",
        lambda *args, **kw: lambda i, a, e: outcome("J-SUPPORT", i.instance_id),
    )
    return ws, packet, work


def decide(ws, work):
    with workspace_action(ws):
        return action.decide_case(ws, work.case, work.run)


def vetoes(record):
    return {j.judge_id for j in record.judge_outcomes if j.verdict == "veto"}


def test_approved_composition_with_one_healed_anomalous_document_is_capped(tmp_path, monkeypatch):
    clean = variant("syn-a-clean")
    ws, _, work = witnessed(tmp_path, monkeypatch, "control", clean)
    control = decide(ws, work)
    assert control.kyb_decision.status == "APPROVED"  # positive control: W2, no anomalies
    assert vetoes(control) == set()
    assert all(j.verdict == "pass" for j in control.judge_outcomes if j.judge_id == ANOMALY)

    ws, packet, work = witnessed(tmp_path, monkeypatch, "anomalous", clean, anomalous={"03"})
    case = packet.facts.case
    rec = json.loads((ws / f"silver/witness/{case}/r1/03.json").read_text())
    # The healed anomaly leaves no finding: the existing verification guard cannot see it.
    assert "witness_envelope_anomalies" not in {f["kind"] for f in rec["findings"]}
    assert verify.assurance_now(ws, case, "r1", "03")["anomalies"]["count"] == 1
    assert verify.assurance_now(ws, case, "r1", "03")["assurance"] == "W2"
    record = decide(ws, work)
    assert record.kyb_decision.status == "MANUAL_REVIEW"
    assert record.kyb_decision.risk_score == control.kyb_decision.risk_score
    assert vetoes(record) == {ANOMALY}
    (veto,) = [j for j in record.judge_outcomes if j.verdict == "veto"]
    assert veto.target == "03" and veto.effect == "manual_review"
    assert "03" in veto.reason and "1 witness envelope anomalies" in veto.reason
    assert "unknown=1" in veto.reason


@pytest.mark.parametrize(
    "label_id,expected,score",
    [("syn-a-sanctions-corroborated", "REJECTED", None), ("B", "MANUAL_REVIEW", 35)],
)
def test_anomaly_never_changes_reject_or_review(tmp_path, monkeypatch, label_id, expected, score):
    ws, _, work = witnessed(tmp_path, monkeypatch, "x", variant(label_id), anomalous={"01", "02"})
    record = decide(ws, work)
    assert record.kyb_decision.status == expected
    if score is not None:
        assert record.kyb_decision.risk_score == score
    assert ANOMALY not in vetoes(record)
    recorded = [j for j in record.judge_outcomes if j.judge_id == ANOMALY and j.target in "0102"]
    assert {j.target for j in recorded} == {"01", "02"}
    for j in recorded:
        assert j.verdict == "pass" and "1 witness envelope anomalies" in j.reason
        assert f"approval-only guard not required for {expected}" in j.reason


def test_anomaly_decision_replays_deterministically(tmp_path, monkeypatch):
    ws, packet, work = witnessed(
        tmp_path, monkeypatch, "replay", variant("syn-a-clean"), anomalous={"02"}
    )
    first = decide(ws, work)
    status = action.decision_status(ws, work.case, work.run)
    assert status["next"] == "decided" and status["decision_id"] == first.decision_id
    second = decide(ws, work)
    assert second.model_dump_json() == first.model_dump_json()
    # The pure judge and assembly are byte-identical from identical inputs.
    now = action.witness_now(ws, work.case, work.run, [d.doc for d in packet.facts.documents])
    assert witness_judges(now) == witness_judges(json.loads(json.dumps(now)))
    built = [
        decision_record(
            packet.facts,
            packet.screening,
            load_policy(POLICY),
            list(packet.answers.values()),
            [*fact_judges(packet.facts), *witness_judges(now)],
            [],
            provenance={},
            emitted_at=datetime(2026, 10, 26, tzinfo=UTC),
            audit_trail=[],
        ).model_dump_json()
        for _ in range(2)
    ]
    assert built[0] == built[1]


def test_no_anomaly_means_no_veto_and_missing_record_is_unknown():
    clean = {"assurance": "W2", "self_referential": False,
             "anomalies": {"count": 0, "severities": {}}}  # fmt: skip
    assert all(j.verdict == "pass" for j in witness_judges({"01": clean, "02": clean}))
    (missing,) = [j for j in witness_judges({"01": None}) if j.judge_id == ANOMALY]
    assert missing.verdict == "veto" and "no witness record" in missing.reason


def test_stratum_b_record_caps_approval_even_when_factset_claims_w2(tmp_path, monkeypatch):
    """L8 stratum B is W1/self-referential through assurance_now, whatever the facts recorded."""
    ws, packet, work = witnessed(tmp_path, monkeypatch, "b", variant("syn-a-clean"), stratum="B")
    case = packet.facts.case
    now = verify.assurance_now(ws, case, "r1", "01")
    assert now["assurance"] == "W1" and now["self_referential"] is True
    assert all(v.assurance == "W2" for v in packet.facts.verification)  # stale/forged fixture
    record = decide(ws, work)
    assert record.kyb_decision.status == "MANUAL_REVIEW"
    assert vetoes(record) == {APPROVAL_ASSURANCE}
    targets = {j.target for j in record.judge_outcomes if j.verdict == "veto"}
    assert targets == {f"witness/{d.doc}" for d in packet.facts.documents}


def test_ab_record_shape_counts_anomalies_in_either_stratum(tmp_path, monkeypatch):
    """A+B keeps per-stratum attempts and drops B's self-reference at the top level."""
    clean = variant("syn-a-clean")
    ws, packet, work = witnessed(tmp_path, monkeypatch, "ab", clean, stratum="A+B")
    rec = json.loads((ws / f"silver/witness/{packet.facts.case}/r1/01.json").read_text())
    assert rec["self_referential"] is False and set(rec["strata"]) == {"A", "B"}
    assert rec["strata"]["B"]["self_referential"] is True
    assert "self_referential_verification" not in {f["kind"] for f in rec["findings"]}
    assert decide(ws, work).kyb_decision.status == "APPROVED"  # agreeing A+B, no anomalies

    ws, _, work = witnessed(
        tmp_path, monkeypatch, "ab-anomalous", clean, stratum="A+B", anomalous={"04"}, b_only=True
    )
    record = decide(ws, work)
    assert record.kyb_decision.status == "MANUAL_REVIEW" and vetoes(record) == {ANOMALY}
    assert [j.target for j in record.judge_outcomes if j.verdict == "veto"] == ["04"]


def test_tampered_witness_record_is_refused(tmp_path, monkeypatch):
    ws, packet, work = witnessed(
        tmp_path, monkeypatch, "tamper", variant("syn-a-clean"), anomalous={"01"}
    )
    path = ws / f"silver/witness/{packet.facts.case}/r1/01.json"
    rec = json.loads(path.read_text())
    rec["attempts"] = [a for a in rec["attempts"] if a["status"] != "anomalous"]
    path.write_text(json.dumps(rec))
    with pytest.raises(PermissionError, match="catalog receipt"):
        decide(ws, work)


def test_fixture_decision_without_witness_records_cannot_approve(tmp_path, monkeypatch):
    """No witness record is unknown anomaly state: approval-only, unrelated guards passing."""
    clean = variant("syn-a-clean")
    facts = clean.facts.model_copy(
        update={
            "verification": [
                DocVerification(doc=d.doc, assurance="W2", isolated=True)
                for d in clean.facts.documents
            ]
        }
    )
    ws, packet, work = fixture_run(tmp_path, "bare", Packet(facts, clean.screening, clean.answers))
    attested_fixture_answers(ws, packet, work, tmp_path)
    monkeypatch.setattr(action, "_reader_judges", lambda *args: [])
    monkeypatch.setattr(action, "fixture_inputs", lambda *args: [])
    monkeypatch.setattr(
        action,
        "support_checker",
        lambda *args, **kw: lambda i, a, e: outcome("J-SUPPORT", i.instance_id),
    )
    record = decide(ws, work)
    assert record.kyb_decision.status == "MANUAL_REVIEW" and vetoes(record) == {ANOMALY}
