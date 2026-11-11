"""L3 policy engine: parse, evaluate, compose.

Golden A/B/C come from fixture FactSets built to the b4 contract from the packet walkthrough and
the locked labels (docs/analysis/golden-labels.md). Policy edits are EVALS suite 8. The property
tests are seeded (≥ 500 cases each), so a failure reproduces exactly.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pytest

from kyb.contracts import b4
from kyb.policy import compose, evaluate, load_policy, model_instances, parse_policy, version_id
from kyb.policy.parse import KNOWN_SKELETONS, skeleton_hashes
from kyb.policy.rules import GUARD_ID

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "examples/policy/compliance_rules.md"
RAW = POLICY.read_bytes()
TEXT = RAW.decode("utf-8")
AS_OF = "2026-10-25"
PRESENT, ABSENT = {"state": "present"}, {"state": "absent"}


# --- fixtures built to the contract ------------------------------------------------------------


def ref(claim: str, page: int = 1) -> dict:
    """ "A01-03" → a verified EvidenceRef to claim c-A01-03 on document 01."""
    return {"claim_id": f"c-{claim}", "doc": claim[1:3], "page": page, "verified": True}


def _refs(node: Any) -> list[dict]:
    if isinstance(node, dict):
        own = [node] if "claim_id" in node else []
        return own + [r for v in node.values() for r in _refs(v)]
    if isinstance(node, list):
        return [r for v in node for r in _refs(v)]
    return []


def facts(case: str, **kw: Any) -> b4.FactSet:
    """A FactSet whose evidence table resolves every EvidenceRef used in it."""
    body = {
        "case": case,
        "run": "r1",
        "as_of": AS_OF,
        "fact_set_hash": f"sha256:{case}",
        "conflicts": [],
        "corporate_object": [],
        "verification": [],
        **kw,
    }
    seen = {r["claim_id"]: r for r in _refs(body)}
    body["evidence"] = [
        {
            "evidence_id": cid,
            "doc": r["doc"],
            "document_name": f"{r['doc']}.pdf",
            "page": r["page"],
            "bbox": (0, 0, 10, 10),
            "excerpt": cid,
            "verified": r["verified"],
        }
        for cid, r in sorted(seen.items())
    ]
    return b4.FactSet.model_validate(body)


def entity(eid: str, kind: str, name: str, *claims: str, **kw: Any) -> dict:
    return {
        "entity_id": eid,
        "kind": kind,
        "names": [name],
        "evidence": [ref(c) for c in claims],
        **kw,
    }


def holding(
    hid: str, owner: str, owned: str, pct: int | None, claim: str, status: str = "current"
) -> dict:
    return {
        "holding_id": hid,
        "owner": owner,
        "owned": owned,
        "pct": pct,
        "status": status,
        "evidence": [ref(claim)],
    }


def role(rid: str, holder: str, of: str, kind: str, claim: str) -> dict:
    return {"role_id": rid, "holder": holder, "of": of, "role": kind, "evidence": [ref(claim)]}


def doc(key: str, case: str, doc_type: str, **kw: Any) -> dict:
    return {
        "doc": key,
        "document_name": f"{key}.pdf",
        "doc_type": doc_type,
        "evidence": [ref(f"{case}{key}-00")],
        **kw,
    }


def ops(ii: dict, iv: dict, v: dict) -> dict:
    return {
        "distinct_operating_address": ii,
        "bank_reference_or_financials": iv,
        "employees_or_website": v,
    }


def screening(fs: b4.FactSet, *candidates: dict) -> b4.ScreeningResult:
    return b4.ScreeningResult(
        case=fs.case,
        run=fs.run,
        fact_set_hash=fs.fact_set_hash,
        watchlist_version="wl-test",
        method="candidates-v1+jw-0.85",
        candidates=list(candidates),
    )


def candidate(
    cid: str,
    eid: str,
    entry: dict,
    jw: float,
    *,
    corr: list[str] = (),
    conf: list[str] = (),
    alias_exact: bool = False,
) -> dict:
    return {
        "candidate_id": cid,
        "entity_id": eid,
        "entry": entry,
        "matched_name": entry["name"],
        "jw_score": jw,
        "alias_exact": alias_exact,
        "triggers_fuzzy": jw >= 0.85 or alias_exact,
        "identifiers": {"corroborating": list(corr), "conflicting": list(conf)},
    }


def answer(iid: str, value: bool | str) -> b4.PredicateAnswer:
    return b4.PredicateAnswer(
        instance_id=iid, value=value, confidence=0.9, evidence=[], rationale="fixture"
    )


def packet_a(**over: Any) -> tuple[b4.FactSet, b4.ScreeningResult, dict]:
    kw = dict(
        applicant="e-cl-andes-agro",
        entities=[
            entity(
                "e-cl-andes-agro",
                "company",
                "Andes Agropecuaria SpA",
                "A01-01",
                jurisdiction=["CO"],
                incorporated=["2019-03-15"],
                addresses=[
                    {
                        "kind": "registered",
                        "evidence": [ref("A01-02")],
                        "text": "Avenida El Dorado No. 68C-61, Of. 404, Bogotá",
                        "mass_registration": ABSENT,
                    }
                ],
            ),
            entity(
                "e-herrera",
                "person",
                "Herrera",
                "A02-01",
                nationality=["CO"],
                dob=["1982-11-03"],
                jurisdiction=["CO"],
            ),
            entity(
                "e-gaviria",
                "person",
                "Gaviria",
                "A02-02",
                nationality=["CO"],
                dob=["1981-02-25"],
                jurisdiction=["CO"],
            ),
        ],
        holdings=[
            holding("h-1", "e-herrera", "e-cl-andes-agro", 55, "A02-03"),
            holding("h-2", "e-gaviria", "e-cl-andes-agro", 45, "A02-04"),
        ],
        roles=[role("r-1", "e-herrera", "e-cl-andes-agro", "legal_rep", "A01-03")],
        documents=[
            doc("01", "A", "incorporation", issued="2026-09-07", stated_validity_days=30),
            doc("02", "A", "ubo_declaration", issued="2026-08-25", signed=True),
            doc("03", "A", "share_register", issued="2024-06-15"),
            doc("04", "A", "proof_of_address", issued="2026-08-11"),
            doc("05", "A", "bank_reference", issued="2026-08-25"),
        ],
        operations=ops({"state": "absent", "evidence": [ref("A01-02")]}, PRESENT, PRESENT),
        corporate_object=["Actividades de las agencias de carga (freight forwarding, CIIU 5229)"],
    )
    kw.update(over)
    fs = facts("A", **kw)
    return fs, screening(fs), {}


def packet_b(**over: Any) -> tuple[b4.FactSet, b4.ScreeningResult, dict]:
    kw = dict(
        applicant="e-mch",
        entities=[
            entity(
                "e-mch",
                "company",
                "Iberia Software S.L.",
                "B01-01",
                jurisdiction=["MX"],
                incorporated=["2020-04-12"],
                addresses=[
                    {
                        "kind": "fiscal",
                        "text": "Av. Insurgentes Sur 1602 P14, CDMX",
                        "mass_registration": {"state": "unknown"},
                        "evidence": [ref("B02-02")],
                    }
                ],
            ),
            entity(
                "e-navarro",
                "person",
                "L. Navarro",
                "B03-01",
                nationality=["MX"],
                dob=["1969-10-15"],
                jurisdiction=["MX"],
            ),
            entity(
                "e-ortega",
                "person",
                "Valeria Ortega Méndez",
                "B03-02",
                nationality=["MX"],
                dob=["1976-10-15"],
                jurisdiction=["MX"],
            ),
            entity(
                "e-altiplano",
                "company",
                "Inversiones Altiplano S.A.",
                "B03-03",
                jurisdiction=["PA"],
            ),
        ],
        holdings=[
            holding("h-1", "e-navarro", "e-mch", 50, "B04-01"),
            holding("h-2", "e-ortega", "e-mch", 25, "B04-02"),
            holding("h-3", "e-altiplano", "e-mch", 25, "B04-03"),
            holding("h-4", "e-navarro", "e-mch", 60, "B01-02", "historical"),
            holding("h-5", "e-ortega", "e-mch", 40, "B01-03", "historical"),
            holding("h-6", "e-navarro", "e-mch", 60, "B05-01", "pending"),
        ],
        roles=[role("r-1", "e-navarro", "e-mch", "president", "B05-02")],
        documents=[
            doc("01", "B", "incorporation", issued="2020-04-12"),
            doc("02", "B", "tax_certificate", issued="2026-10-25"),
            doc("03", "B", "ubo_declaration", issued="2026-10-25", signed=True),
            doc("04", "B", "share_register", issued="2026-10-25"),
            doc("05", "B", "board_minutes", issued="2026-08-19"),
        ],
        conflicts=[
            {
                "conflict_id": "x-1",
                "subject": "e-navarro",
                "attribute": "ownership_pct",
                "values": [
                    {"value": "50", "evidence": [ref("B04-01")]},
                    {"value": "60", "evidence": [ref("B01-02"), ref("B05-01")]},
                ],
            },
            {
                "conflict_id": "x-2",
                "subject": "e-ortega",
                "attribute": "ownership_pct",
                "values": [
                    {"value": "40", "evidence": [ref("B01-03")]},
                    {"value": "25", "evidence": [ref("B04-02")]},
                ],
            },
            {
                "conflict_id": "x-3",
                "subject": "e-altiplano",
                "attribute": "owner_presence",
                "values": [
                    {"value": "absent", "evidence": [ref("B01-04")]},
                    {"value": "25", "evidence": [ref("B04-03")]},
                ],
            },
        ],
        operations=ops(
            {"state": "absent", "evidence": [ref("B02-02")]},
            {"state": "absent", "evidence": [ref("B03-04")]},
            {"state": "absent", "evidence": [ref("B03-05")]},
        ),
        corporate_object=[
            "la adquisición y tenencia de acciones de otras sociedades",
            "la inversión en bienes inmuebles",
        ],
    )
    kw.update(over)
    fs = facts("B", **kw)
    pep = {
        "entry_id": "PEP-0103",
        "list_kind": "pep",
        "name": "Valeria Ortega",
        "nationality": "ES",
        "dob": "1958-10-26",
        "in_office": False,
        "left_office": "2015-06-30",
    }
    sr = screening(fs, candidate("c-1", "e-ortega", pep, 0.91, conf=["nationality", "dob"]))
    return fs, sr, {"poa-02": answer("poa-02", True), "sp-c-1": answer("sp-c-1", False)}


def packet_c(**over: Any) -> tuple[b4.FactSet, b4.ScreeningResult, dict]:
    kw = dict(
        applicant="e-vmfl",
        entities=[
            entity(
                "e-vmfl",
                "company",
                "Strait Capital Pte. Ltd.",
                "C01-01",
                jurisdiction=["VG"],
                incorporated=["2026-01-09"],
                identifiers={"reg_no": ["3184920"]},
                addresses=[
                    {
                        "kind": "registered",
                        "text": "Apex Chambers, Wickhams Cay, Sovereign Town, Tortola",
                        "mass_registration": {"state": "present", "evidence": [ref("C01-02")]},
                        "evidence": [ref("C01-02")],
                    },
                    {
                        "kind": "operating",
                        "text": "Asunción",
                        "evidence": [ref("C05-01")],
                        "mass_registration": ABSENT,
                    },
                ],
            ),
            entity("e-apex", "agent", "Apex Corporate Services Ltd", "C02-01", jurisdiction=["VG"]),
            entity("e-helios", "foundation", "Sovereign Trust", "C04-01", jurisdiction=["PA"]),
            entity(
                "e-paredes",
                "person",
                "Sebastián Paredes Vega",
                "C03-01",
                nationality=["PY"],
                dob=["1971-10-15"],
                jurisdiction=["PY"],
            ),
        ],
        holdings=[holding("h-1", "e-helios", "e-vmfl", 100, "C02-02")],
        roles=[
            role("r-1", "e-apex", "e-vmfl", "nominee_director", "C02-03"),
            role("r-2", "e-apex", "e-vmfl", "registered_agent", "C01-03"),
            role("r-3", "e-paredes", "e-helios", "beneficiary", "C04-02"),
            role("r-4", "e-paredes", "e-helios", "council_member", "C04-03"),
            role("r-5", "e-paredes", "e-vmfl", "signatory", "C05-02"),
        ],
        documents=[
            doc("01", "C", "incorporation", issued="2026-01-09"),
            doc("02", "C", "share_register", issued="2026-01-09"),
            doc("03", "C", "ubo_declaration", issued="2026-09-06", signed=True),
            doc("04", "C", "other", issued="2025-11-25"),
            doc("05", "C", "lease", signed_on="2026-01-25", signed=True),
            doc("06", "C", "bank_reference", issued="2026-09-07"),
        ],
        operations=ops(PRESENT, PRESENT, PRESENT),
        corporate_object=["agricultural trade finance"],
    )
    kw.update(over)
    fs = facts("C", **kw)
    san = {
        "entry_id": "SAN-0001",
        "list_kind": "sanctions",
        "name": "Strait Capital FZE",
        "aliases": ["Strait Maritime"],
        "jurisdiction": "AE",
        "registration_number": "DMCC-417733",
    }
    pep = {
        "entry_id": "PEP-0101",
        "list_kind": "pep",
        "name": "Sebastián Paredes Vega",
        "nationality": "PY",
        "dob": "1971-10-15",
        "in_office": False,
        "left_office": "2024-08-14",
    }
    sr = screening(
        fs,
        candidate("c-1", "e-vmfl", san, 0.904, conf=["reg_no", "jurisdiction"]),
        candidate("c-2", "e-paredes", pep, 1.0, corr=["nationality+dob"]),
    )
    answers = {
        "npd-e-vmfl": answer("npd-e-vmfl", True),
        "poa-06": answer("poa-06", False),
        "sp-c-1": answer("sp-c-1", False),
    }
    return fs, sr, answers


def run(packet: tuple, policy: b4.PolicyModel | None = None, answers: dict | None = None):
    fs, sr, ans = packet
    pol = policy or parse_policy(RAW)
    ev = evaluate(fs, sr, pol, ans if answers is None else answers)
    return ev, compose(ev.rule_results, pol)


def fired(ev) -> set[str]:
    return {r.rule_id for r in ev.rule_results if r.fired == "yes"}


def edited(*pairs: tuple[str, str]) -> b4.PolicyModel:
    text = TEXT
    for old, new in pairs:
        assert text.count(old) >= 1, old
        text = text.replace(old, new)
    return parse_policy(text.encode("utf-8"))


# --- parsing ----------------------------------------------------------------------------------


def test_policy_parses_completely_with_version_and_hash():
    pol = load_policy(POLICY)
    assert pol.rules_version == "2026.11-r1"
    assert pol.content_hash.startswith("sha256:") and len(pol.content_hash) == 71
    assert pol.unrepresentable == []
    assert skeleton_hashes(RAW) == KNOWN_SKELETONS
    p = pol.params
    assert p["prohibited_jurisdictions"] == ["AF", "BY", "IR", "KP", "SY"]
    assert set(p["prohibited_regions"]) == {"UA"}
    assert p["high_risk_jurisdictions"] == ["BZ", "MM", "SC", "VG", "VU", "WS"]
    assert set(p["high_risk_regions"]) == {"AE"}
    assert (p["ownership_threshold_pct"], p["high_risk_unresolved_pct"]) == (20, 40)
    assert p["name_similarity_threshold"] == 0.88
    assert (p["pep_lookback_years"], p["pep_former_points"]) == (3, 15)
    assert (p["shell_review_min"], p["shell_reject_min"], p["young_company_months"]) == (2, 3, 12)
    assert (p["ubo_declaration_max_age_months"], p["proof_of_address_max_age_months"]) == (6, 3)
    assert (p["max_holding_layers"], p["conflict_extra_points"]) == (2, 10)
    assert (p["auto_approve_max_score"], p["score_cap"], p["conflict_ownership_points"]) == (
        25,
        90,
        2,
    )
    assert (
        "trade finance" in p["high_risk_industries"] and "shell bank" in p["prohibited_industries"]
    )
    specs = {r.rule_id: (r.severity, r.score_delta) for r in pol.rules}
    assert specs == {
        "SCR-01": ("REJECT", 0),
        "SCR-02": ("REVIEW", 0),
        "JUR-01": ("REJECT", 0),
        "JUR-02": ("SCORE", 20),
        "OWN-01": ("REVIEW", 25),
        "OWN-02": ("REJECT", 0),
        "GOV-01": ("SCORE", 15),
        "GOV-02": ("REVIEW", 0),
        "GOV-03": ("REVIEW", 15),
        "PEP-01": ("REVIEW", 30),
        "PEP-02": ("SCORE", 15),
        "DOC-01": ("REVIEW", 20),
        "DOC-02": ("REVIEW", 15),
        "OPS-01": ("REVIEW", 20),
        "OPS-02": ("REJECT", 0),
        "ACT-01": ("REJECT", 0),
        "ACT-02": ("SCORE", 10),
        "MOD-01": ("SCORE", 10),
    }
    assert all(r.text.startswith("## ") for r in pol.rules)  # verbatim section text for the model


def test_parsing_is_deterministic():
    assert parse_policy(RAW).model_dump_json() == parse_policy(RAW).model_dump_json()


# --- golden A/B/C --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("packet", "status", "score", "rules"),
    [
        (packet_a, "MANUAL_REVIEW", 15, {"DOC-02"}),
        (
            packet_b,
            "MANUAL_REVIEW",
            85,
            {"OWN-01", "DOC-01", "OPS-01", "MOD-01"},
        ),
        (
            packet_c,
            "MANUAL_REVIEW",
            90,
            {"SCR-02", "JUR-02", "GOV-01", "PEP-01", "DOC-02", "OPS-01", "ACT-02"},
        ),
    ],
)
def test_golden(packet, status, score, rules):
    ev, comp = run(packet())
    assert fired(ev) == rules
    assert [r.rule_id for r in ev.rule_results if r.fired == "unknown"] == []
    assert (comp.status, comp.risk_score) == (status, score)
    for r in ev.rule_results:
        if r.fired == "yes":
            assert r.evidence, f"{r.rule_id} fired without evidence"


def test_golden_details():
    ev_b, _ = run(packet_b())
    by = {r.rule_id: r for r in ev_b.rule_results}
    assert by["MOD-01"].score_delta == 20
    assert "(ii)" not in by["OPS-01"].description and "'ii', 'iv', 'v'" in by["OPS-01"].description
    assert by["OPS-02"].fired == "no"
    assert "screening note" in by["PEP-01"].description
    assert by["DOC-02"].predicates == ["poa-02"]
    ev_c, _ = run(packet_c())
    by = {r.rule_id: r for r in ev_c.rule_results}
    assert by["SCR-01"].fired == "no" and by["OWN-01"].fired == "no"
    assert by["GOV-02"].fired == "no" and by["GOV-02"].predicates == ["npd-e-vmfl"]
    assert by["PEP-01"].score_delta == 30 and by["PEP-02"].fired == "no"
    ev_a, _ = run(packet_a())
    assert (
        "expired: certificate of incorporation"
        in {r.rule_id: r for r in ev_a.rule_results}["DOC-02"].description
    )


def test_code_predicates_are_recorded():
    ev, _ = run(packet_c())
    ids = {i.instance_id for i in ev.code_instances}
    assert ids == {a.instance_id for a in ev.code_answers}
    assert "cp-pep_within_window-c-2" in ids and "cp-doc_fresh-05" in ids
    assert {a.instance_id: a.value for a in ev.code_answers}["cp-doc_fresh-05"] is False


def test_model_instances_name_what_the_rules_read():
    fs, sr, _ = packet_c()
    got = {i.instance_id: i for i in model_instances(fs, sr, parse_policy(RAW))}
    assert set(got) == {"sp-c-1", "sp-c-2", "npd-e-vmfl", "poa-06"}
    assert all(i.kind == "model" for i in got.values())
    fs, sr, _ = packet_b()
    assert {i.instance_id for i in model_instances(fs, sr, parse_policy(RAW))} == {
        "sp-c-1",
        "poa-02",
    }
    fs, sr, _ = packet_a()
    assert model_instances(fs, sr, parse_policy(RAW)) == []


# --- unknowns: missing answers, failed verification ----------------------------------------------


def test_missing_or_unknown_answers_read_as_unknown():
    ev, comp = run(packet_c(), answers={})
    by = {r.rule_id: r.fired for r in ev.rule_results}
    assert by["GOV-02"] == "unknown"
    assert comp.status == "MANUAL_REVIEW" and "GOV-02" in comp.unknown
    ev, _ = run(packet_b(), answers={"poa-02": answer("poa-02", "UNKNOWN")})
    assert {r.rule_id: r.fired for r in ev.rule_results}["DOC-02"] == "unknown"


def test_same_party_escalates_but_never_clears():
    fs, sr, ans = packet_c()
    ev, _ = run((fs, sr, {**ans, "sp-c-1": answer("sp-c-1", False)}))
    assert "SCR-02" in fired(ev)
    low = sr.model_copy(
        update={"candidates": [sr.candidates[0].model_copy(update={"jw_score": 0.6})]}
    )
    ev, _ = run((fs, low, ans))
    assert "SCR-02" not in fired(ev)
    ev, _ = run((fs, low, {**ans, "sp-c-1": answer("sp-c-1", True)}))
    assert "SCR-02" in fired(ev) and "SCR-01" not in fired(ev)


def test_unverified_or_uncovered_facts_turn_no_into_unknown():
    fs, sr, ans = packet_a()
    gap = [
        {
            "doc": "05",
            "assurance": "W2",
            "isolated": True,
            "uncovered": [
                {
                    "block_id": "b1",
                    "kind": "operations",
                    "subject": "e-cl-andes-agro",
                    "attribute": "employees",
                    "value": "23",
                }
            ],
        }
    ]
    ev, _ = run(
        (
            facts(
                "A",
                **{
                    **fs.model_dump(
                        exclude={
                            "case",
                            "run",
                            "as_of",
                            "fact_set_hash",
                            "evidence",
                            "contracts_version",
                        }
                    ),
                    "verification": gap,
                },
            ),
            sr,
            ans,
        )
    )
    by = {r.rule_id: r for r in ev.rule_results}
    assert by["OPS-01"].fired == "unknown" and "uncovered operations" in by["OPS-01"].description
    assert by["DOC-02"].fired == "yes"
    assert by["JUR-01"].fired == "no"


def test_clean_packet_is_approved():
    ev, comp = run(packet_a_clean())
    assert fired(ev) == set() and (comp.status, comp.risk_score) == ("APPROVED", 0)


def packet_a_clean():
    fs, sr, ans = packet_a()
    docs = [
        d.model_copy(update={"stated_validity_days": None})
        if d.doc == "01"
        else d.model_copy(update={"issued": "2026-06-10"})
        if d.doc == "03"
        else d
        for d in fs.documents
    ]
    return fs.model_copy(update={"documents": docs}), sr, ans


# --- EVALS suite 8: policy edits ------------------------------------------------------------------


def test_edit_pep_window_3y_to_1y():
    pol = edited(
        ("within the last **3 years**", "within the last **1 years**"),
        ("more than 3 years ago", "more than 1 years ago"),
    )
    assert pol.unrepresentable == [] and pol.params["pep_lookback_years"] == 1
    assert pol.content_hash != parse_policy(RAW).content_hash
    ev, comp = run(packet_c(), pol)
    assert "PEP-01" not in fired(ev) and "PEP-02" in fired(ev)
    assert (comp.status, comp.risk_score) == ("MANUAL_REVIEW", 90)


def test_edit_pep_window_inconsistently_is_unrepresentable():
    pol = edited(("within the last **3 years**", "within the last **5 years**"))
    assert any(u.startswith("pep: the PEP look-back") for u in pol.unrepresentable)
    ev, comp = run(packet_a_clean(), pol)
    assert {r.rule_id: r.fired for r in ev.rule_results}["PEP-01"] == "unknown"
    assert comp.status == "MANUAL_REVIEW"


def test_edit_add_a_country():
    pol = edited(("`BY`, or the", "`BY`, `CO`, or the"))
    assert pol.unrepresentable == [] and "CO" in pol.params["prohibited_jurisdictions"]
    ev, comp = run(packet_a_clean(), pol)
    assert "JUR-01" in fired(ev) and comp.status == "REJECTED"
    pol = edited(("`SC`, or free zones", "`SC`, `CO`, or free zones"))
    assert pol.unrepresentable == [] and "CO" in pol.params["high_risk_jurisdictions"]
    ev, comp = run(packet_a_clean(), pol)
    assert fired(ev) == {"JUR-02"} and (comp.status, comp.risk_score) == ("APPROVED", 20)


def test_edit_thresholds():
    pol = edited(("score ≥ 0.88", "score ≥ 0.95"))
    assert pol.params["name_similarity_threshold"] == 0.95 and pol.unrepresentable == []
    ev, _ = run(packet_c(), pol)
    assert "SCR-02" not in fired(ev)
    pol = edited(
        ("`risk_score ≤ 25`", "`risk_score ≤ 15`"), ("`risk_score ≥ 26`", "`risk_score ≥ 16`")
    )
    ev, comp = run(packet_a_clean(), pol)
    assert comp.status == "APPROVED"
    pol3 = edited(
        ("`SC`, or free zones", "`SC`, `CO`, or free zones"),
        ("`risk_score ≤ 25`", "`risk_score ≤ 15`"),
        ("`risk_score ≥ 26`", "`risk_score ≥ 16`"),
    )
    ev, comp = run(packet_a_clean(), pol3)
    assert (
        comp.status == "MANUAL_REVIEW" and "score above the auto-approval band" in comp.reasons[0]
    )
    pol = edited(("`risk_score ≤ 25`", "`risk_score ≤ 15`"))
    assert any(u.startswith("scoring:") for u in pol.unrepresentable)
    ev, comp = run(packet_a_clean(), pol)
    assert comp.status == "MANUAL_REVIEW" and any(r.rule_id == GUARD_ID for r in ev.rule_results)


def test_edit_severity():
    pol = edited(
        (
            "## 7. Documentary Integrity & Discrepancies — **REVIEW +20**",
            "## 7. Documentary Integrity & Discrepancies — **REJECT**",
        )
    )
    assert pol.unrepresentable == []
    assert {r.rule_id: r.severity for r in pol.rules}["DOC-01"] == "REJECT"
    _, comp = run(packet_b(), pol)
    assert comp.status == "REJECTED" and comp.risk_score == 65
    _, comp = run(packet_a(), pol)
    assert comp.status == "MANUAL_REVIEW"


def test_edit_reworded_cosmetically_changes_nothing_but_the_hash():
    pol = edited(
        (
            "**2 or more indicators → REVIEW.**",
            "Two or more indicators -> **REVIEW**.",
        ),
        ("## 4. Beneficial Ownership", "##   4.  Beneficial Ownership"),
    )
    assert pol.unrepresentable == [] and pol.content_hash != parse_policy(RAW).content_hash
    for packet in (packet_a, packet_b, packet_c):
        base, new = run(packet()), run(packet(), pol)
        assert base[0].rule_results == new[0].rule_results and base[1] == new[1]


def test_edit_reworded_rule_reaches_the_model_text():
    pol = edited(
        (
            "the packet names no individual who actually runs the company",
            "the packet names no individual who actually runs or controls the company",
        )
    )
    spec = {r.rule_id: r for r in pol.rules}["GOV-02"]
    assert "runs or controls" in spec.text
    assert any(u.startswith("governance: wording differs") for u in pol.unrepresentable)
    ev, comp = run(packet_c(), pol)
    by = {r.rule_id: r.fired for r in ev.rule_results}
    assert by["GOV-01"] == by["GOV-02"] == "unknown" and comp.status == "MANUAL_REVIEW"


def test_edit_unrepresentable_is_a_visible_guard():
    pol = edited(("or a UBO resident in:", "or a UBO or director resident in:"))
    assert [u.split(":")[0] for u in pol.unrepresentable] == ["prohibited_jurisdictions"]
    ev, comp = run(packet_a_clean(), pol)
    assert {r.rule_id: r.fired for r in ev.rule_results}["JUR-01"] == "unknown"
    assert comp.status == "MANUAL_REVIEW"
    pol = parse_policy(RAW + "\n## 14. Adverse media — **REVIEW**\nAny hit.\n".encode())
    assert any("unrecognised section" in u for u in pol.unrepresentable)
    ev, comp = run(packet_a_clean(), pol)
    assert ev.rule_results[-1].rule_id == GUARD_ID and comp.status == "MANUAL_REVIEW"
    pol = edited(("administration.", "administration, crowdfunding platforms."))
    assert any("crowdfunding platforms" in u for u in pol.unrepresentable)
    _, comp = run(packet_a_clean(), pol)
    assert comp.status == "MANUAL_REVIEW"


def test_edit_removing_a_section_is_unrepresentable():
    start = TEXT.index("## 10. Sector")
    pol = parse_policy((TEXT[:start] + TEXT[TEXT.index("## 11.") :]).encode())
    assert "industry: section missing" in pol.unrepresentable
    _, comp = run(packet_a_clean(), pol)
    assert comp.status == "MANUAL_REVIEW"


# --- composition ------------------------------------------------------------------------------


def rr(rid: str, sev: str, f: str = "yes", delta: int = 0) -> b4.RuleResult:
    return b4.RuleResult(rule_id=rid, severity=sev, fired=f, score_delta=delta, description="")


def test_compose_precedence_and_cap():
    pol = parse_policy(RAW)
    assert compose([rr("a", "SCORE", delta=25)], pol).status == "APPROVED"
    c = compose([rr("a", "SCORE", delta=26)], pol)
    assert c.status == "MANUAL_REVIEW" and c.reasons[0].startswith("score above the auto-approval")
    c = compose([rr("a", "SCORE", delta=80), rr("b", "SCORE", delta=40)], pol)
    assert (c.risk_score, c.uncapped_score) == (90, 120)
    assert compose([rr("a", "REJECT", "unknown")], pol).status == "MANUAL_REVIEW"
    veto = b4.JudgeOutcome(
        judge_id="J-EVIDENCE",
        target="SCR-01",
        verdict="veto",
        effect="manual_review",
        reason="excerpt not verified",
    )
    c = compose([rr("SCR-01", "REJECT")], pol, [veto])
    assert c.status == "MANUAL_REVIEW"  # judges veto toward MANUAL_REVIEW only
    with pytest.raises(ValueError):
        compose([rr("a", "SCORE"), rr("a", "SCORE")], pol)


def test_property_reject_outranks_any_score():
    pol, rng = parse_policy(RAW), random.Random(3512)
    for i in range(600):
        others = [
            rr(
                f"r{j}",
                rng.choice(["REVIEW", "SCORE", "REJECT"]),
                rng.choice(["yes", "no", "unknown"]),
                rng.randint(0, 60),
            )
            for j in range(rng.randint(0, 12))
        ]
        rules = [*others, rr("reject", "REJECT", delta=rng.randint(0, 100))]
        rng.shuffle(rules)
        c = compose(rules, pol)
        assert c.status == "REJECTED", (i, rules)
        assert 0 <= c.risk_score <= 90


def test_property_adding_a_trigger_never_improves_the_outcome():
    pol, rng = parse_policy(RAW), random.Random(1512)
    rank = {"APPROVED": 0, "MANUAL_REVIEW": 1, "REJECTED": 2}
    for _ in range(600):
        rules = [
            rr(
                f"r{j}",
                rng.choice(["REVIEW", "SCORE", "REJECT"]),
                rng.choice(["yes", "no", "unknown"]),
                rng.randint(0, 40),
            )
            for j in range(rng.randint(0, 8))
        ]
        extra = rr(
            "x",
            rng.choice(["REVIEW", "SCORE", "REJECT"]),
            rng.choice(["yes", "unknown"]),
            rng.randint(0, 40),
        )
        assert rank[compose([*rules, extra], pol).status] >= rank[compose(rules, pol).status]


def _shuffled(fs: b4.FactSet, sr: b4.ScreeningResult, rng: random.Random):
    def sh(xs):
        xs = list(xs)
        rng.shuffle(xs)
        return xs

    ents = [
        e.model_copy(update={"evidence": sh(e.evidence), "addresses": sh(e.addresses)})
        for e in fs.entities
    ]
    lists = (
        "holdings",
        "roles",
        "documents",
        "conflicts",
        "evidence",
        "corporate_object",
        "verification",
    )
    fs2 = fs.model_copy(update={"entities": sh(ents), **{k: sh(getattr(fs, k)) for k in lists}})
    return fs2, sr.model_copy(update={"candidates": sh(sr.candidates)})


def test_property_permuting_documents_changes_nothing():
    rng = random.Random(29)
    pol = parse_policy(RAW)
    for packet in (packet_a, packet_b, packet_c, packet_a_clean):
        fs, sr, ans = packet()
        base = evaluate(fs, sr, pol, ans)
        base_c = compose(base.rule_results, pol)
        for _ in range(130):
            fs2, sr2 = _shuffled(fs, sr, rng)
            ev = evaluate(fs2, sr2, pol, dict(sorted(ans.items(), key=lambda _: rng.random())))
            assert ev == base
            assert compose(ev.rule_results, pol) == base_c


def test_screening_must_match_the_facts():
    fs, sr, ans = packet_a()
    with pytest.raises(ValueError):
        evaluate(fs, sr.model_copy(update={"fact_set_hash": "sha256:other"}), parse_policy(RAW))


# --- the load_policy action -------------------------------------------------------------------


def test_record_policy_writes_once_and_audits(tmp_path):
    from kyb.audit.chain import verify_workspace, workspace_action
    from kyb.policy.store import record_policy
    from kyb.workspace.init import init_workspace

    ws = tmp_path / "ws"
    init_workspace(ws, tenant="t", policy_src=POLICY)
    with workspace_action(ws):
        rel, model = record_policy(ws, as_of=AS_OF)
    assert rel == f"gold/policy/{model.content_hash.removeprefix('sha256:')}.json"
    assert b4.PolicyModel.model_validate_json((ws / rel).read_bytes()) == model
    with workspace_action(ws):
        assert record_policy(ws, as_of=AS_OF) == (rel, model)  # idempotent: nothing new
    assert verify_workspace(ws).ok


# --- regressions from the cross-review (Codex round 1) -------------------------------------------


def clean_with(entities=None, **over):
    """Clean A (APPROVED 0) with some facts replaced."""
    fs, sr, ans = packet_a_clean()
    body = fs.model_dump(
        exclude={"case", "run", "as_of", "fact_set_hash", "evidence", "contracts_version"}
    )
    if entities is not None:
        body["entities"] = entities(body["entities"])
    body.update(over)
    fs = facts("A", **body)
    return fs, screening(fs, *[c.model_dump() for c in sr.candidates]), ans


def set_jurisdiction(code):
    def f(ents):
        return [
            {**e, "jurisdiction": [code]} if e["entity_id"] == "e-cl-andes-agro" else e
            for e in ents
        ]  # noqa: E501

    return f


def test_r1_corporate_owners_must_reach_natural_persons():
    corps = [entity(f"e-c{i}", "company", f"Corp {i}", f"A02-1{i}") for i in range(5)]
    hold = [holding("h-top", "e-hold", "e-cl-andes-agro", 100, "A02-20")] + [
        holding(f"h-c{i}", f"e-c{i}", "e-hold", 19, f"A02-2{i}") for i in range(5)
    ]
    pk = clean_with(
        lambda ents: [ents[0], entity("e-hold", "company", "Hold", "A02-30"), *corps], holdings=hold
    )
    ev, comp = run(pk)
    assert {r.rule_id: r.fired for r in ev.rule_results}["OWN-01"] == "yes"
    assert comp.status == "MANUAL_REVIEW"


def test_r1_unanswered_same_party_is_unknown():
    fs, _, ans = packet_a_clean()
    entry = {"entry_id": "SAN-9", "list_kind": "sanctions", "name": "Andes Logistic"}
    sr = screening(fs, candidate("low", "e-cl-andes-agro", entry, 0.60))
    for answers in ({}, {"sp-low": answer("sp-low", "UNKNOWN")}):
        ev, comp = run((fs, sr, answers))
        assert {r.rule_id: r.fired for r in ev.rule_results}["SCR-02"] == "unknown"
        assert comp.status == "MANUAL_REVIEW"
    _, comp = run((fs, sr, {"sp-low": answer("sp-low", False)}))
    assert comp.status == "APPROVED"


def test_r1_heading_words_beyond_the_grammar_trip_the_guard():
    pol = edited(
        (
            "Embargoed Jurisdictions — **REJECT**",
            "Embargoed Jurisdictions — **REJECT unless incorporated before 2020**",
        )
    )
    assert any(
        u.startswith("prohibited_jurisdictions: heading severity") for u in pol.unrepresentable
    )
    _, comp = run(clean_with(set_jurisdiction("IR")), pol)
    assert comp.status == "MANUAL_REVIEW"


def test_r1_a_score_severity_applies_its_points():
    pol = edited(
        ("Embargoed Jurisdictions — **REJECT**", "Embargoed Jurisdictions — **SCORE +40**")
    )
    assert pol.unrepresentable == []
    _, comp = run(clean_with(set_jurisdiction("IR")), pol)
    assert (comp.status, comp.risk_score) == ("MANUAL_REVIEW", 40)


def test_r1_a_global_guard_never_lets_a_reject_through():
    pol = edited(("→ `REJECTED`.", "→ `REJECTED`, unless incorporated before 2020."))
    assert any(u.startswith("protocol:") for u in pol.unrepresentable)
    ev, comp = run(clean_with(set_jurisdiction("IR")), pol)
    assert "JUR-01" in fired(ev) and comp.status == "MANUAL_REVIEW"
    _, comp = run(clean_with(set_jurisdiction("IR")))
    assert comp.status == "REJECTED"


def test_r1_a_verification_gap_cannot_manufacture_a_reject():
    def ents(es):
        es = set_jurisdiction("VG")(es)
        return [{**e, "dob": []} if e["entity_id"] == "e-herrera" else e for e in es]

    gap = [
        {
            "doc": "02",
            "assurance": "W2",
            "isolated": True,
            "uncovered": [
                {
                    "block_id": "b",
                    "kind": "identifier",
                    "subject": "e-herrera",
                    "attribute": "dob",
                    "value": "1982-11-03",
                }
            ],
        }
    ]
    hold = [holding("h-1", "e-herrera", "e-cl-andes-agro", 100, "A02-03")]
    ev, comp = run(clean_with(ents, holdings=hold, verification=gap))
    by = {r.rule_id: r.fired for r in ev.rule_results}
    assert by["OWN-02"] == "unknown" and by["OWN-01"] == "unknown"
    assert comp.status == "MANUAL_REVIEW"
    ev, comp = run(clean_with(ents, holdings=hold))
    assert {r.rule_id: r.fired for r in ev.rule_results}["OWN-02"] == "yes"
    assert comp.status == "REJECTED"


def test_r1_industry_synonyms_and_negation():
    _, comp = run(clean_with(corporate_object=["Manufacturing firearms"]))
    assert comp.status == "REJECTED"
    ev, comp = run(
        clean_with(corporate_object=["Software development; the company does not trade in weapons"])
    )
    assert {r.rule_id: r.fired for r in ev.rule_results}["ACT-01"] == "unknown"
    assert comp.status == "MANUAL_REVIEW"
    ev, _ = run(clean_with(corporate_object=["asociación sin fines de lucro"]))
    assert "ACT-02" in fired(ev)


def test_r1_future_or_unevaluable_dates_are_not_fresh():
    fs, sr, ans = packet_a_clean()
    docs = [
        d.model_copy(update={"issued": "2027-01-01"}) if d.doc in ("02", "04") else d
        for d in fs.documents
    ]
    ev, comp = run((fs.model_copy(update={"documents": docs}), sr, ans))
    assert {r.rule_id: r.fired for r in ev.rule_results}["DOC-02"] == "unknown"
    assert comp.status == "MANUAL_REVIEW"
    docs = [
        d.model_copy(update={"issued": None, "stated_validity_days": 30}) if d.doc == "01" else d
        for d in fs.documents
    ]
    ev, comp = run((fs.model_copy(update={"documents": docs}), sr, ans))
    assert {r.rule_id: r.fired for r in ev.rule_results}["DOC-02"] == "unknown"


def test_r1_version_id_carries_the_hash():
    a, b = parse_policy(RAW), edited(("score ≥ 0.88", "score ≥ 0.95"))
    assert a.rules_version == b.rules_version == "2026.11-r1"
    assert version_id(a) != version_id(b) and version_id(a).endswith(a.content_hash)


# --- regressions from the cross-review (Codex round 2) -------------------------------------------


def test_r2_composite_heading_points_and_pointless_score_are_guarded():
    pol = edited(
        (
            "Beneficial Ownership Resolution — **REVIEW / REJECT**",
            "Beneficial Ownership Resolution — **SCORE +40 / REJECT**",
        ),
        ("**OWN-01 (REVIEW +25)", "**OWN-01 (SCORE)"),
    )
    assert any(u.startswith("ownership:") for u in pol.unrepresentable)

    def no_dob(es):
        return [{**e, "dob": []} if e["entity_id"] == "e-herrera" else e for e in es]

    _, comp = run(clean_with(no_dob), pol)
    assert comp.status == "MANUAL_REVIEW"
    pol = edited(
        (
            "Documentary Integrity & Discrepancies — **REVIEW +20**",
            "Documentary Integrity & Discrepancies — **SCORE**",
        )
    )
    assert any("conflicts" in u for u in pol.unrepresentable)


def test_r2_future_certificate_is_not_established():
    fs, sr, ans = packet_a_clean()
    for validity in (None, 30):
        docs = [
            d.model_copy(update={"issued": "2027-01-01", "stated_validity_days": validity})
            if d.doc == "01"
            else d
            for d in fs.documents
        ]
        ev, comp = run((fs.model_copy(update={"documents": docs}), sr, ans))
        assert {r.rule_id: r.fired for r in ev.rule_results}["DOC-02"] == "unknown"
        assert comp.status == "MANUAL_REVIEW"


@pytest.mark.parametrize(
    "activity",
    [
        "Software development without weapons trading",
        "Desarrollo de software sin comercio de armas",
        "Manufacturing water guns and toy pistols for children",
    ],
)
def test_r2_negated_or_imitation_industry_is_unknown_not_reject(activity):
    ev, comp = run(clean_with(corporate_object=[activity]))
    assert {r.rule_id: r.fired for r in ev.rule_results}["ACT-01"] == "unknown"
    assert comp.status == "MANUAL_REVIEW"


def test_r2_own02_reads_every_holders_jurisdiction():
    def ents(es):
        es = [{**e, "jurisdiction": ["VG"]} if e["entity_id"] == "e-herrera" else e for e in es]
        return [*es, entity("e-opaque", "company", "Opaca S.A.S.", "A02-40", jurisdiction=["CO"])]

    hold = [
        holding("h-1", "e-herrera", "e-cl-andes-agro", 25, "A02-03"),
        holding("h-2", "e-opaque", "e-cl-andes-agro", 75, "A02-04"),
    ]
    ev, comp = run(clean_with(ents, holdings=hold))
    assert {r.rule_id: r.fired for r in ev.rule_results}["OWN-02"] == "yes"
    assert comp.status == "REJECTED"


def test_r2_parsing_is_identical_across_hash_seeds():
    import os
    import subprocess
    import sys

    edit = TEXT.replace(
        "Sanctions & Watchlist Screening — **REJECT / REVIEW**",
        "Sanctions & Watchlist Screening — **REJECT / SCORE**",
    )
    code = (
        "import sys; from kyb.policy import parse_policy as p; "
        "sys.stdout.write(p(sys.stdin.buffer.read()).model_dump_json())"
    )
    outs = {
        subprocess.run(
            [sys.executable, "-c", code],
            input=edit.encode(),
            capture_output=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        ).stdout
        for seed in ("1", "2", "3", "4", "5")
    }
    assert len(outs) == 1 and b"sanctions" in next(iter(outs))


# --- regressions from the cross-review (Codex round 3) -------------------------------------------


def test_r3_repeated_share_observations_are_one_holding():
    def ents(es):
        es = [{**e, "jurisdiction": ["IR"]} if e["entity_id"] == "e-herrera" else e for e in es]
        return [
            *es,
            entity(
                "e-small",
                "person",
                "Small",
                "A02-50",
                nationality=["CO"],
                dob=["1990-01-01"],
                jurisdiction=["CO"],
            ),
        ]

    def h(hid, owner, shares, claim):
        return {**holding(hid, owner, "e-cl-andes-agro", None, claim), "shares": shares}

    base = [
        h("h-1", "e-herrera", 25, "A02-03"),
        h("h-2", "e-gaviria", 70, "A02-04"),
        h("h-3", "e-small", 5, "A02-51"),
    ]
    _, comp = run(clean_with(ents, holdings=base))
    assert comp.status == "REJECTED"
    _, comp = run(clean_with(ents, holdings=[*base, h("h-4", "e-small", 5, "A03-01")]))
    assert comp.status == "REJECTED"
    ev, comp = run(clean_with(ents, holdings=[*base, h("h-4", "e-small", 7, "A03-01")]))
    assert comp.status == "MANUAL_REVIEW" and "JUR-01" not in fired(ev)


@pytest.mark.parametrize(
    "activity",
    [
        "Repair and sale of water pistols for children",
        "Manufacturing hot glue guns for arts and crafts",
        "Manufacturing nail guns for construction",
    ],
)
def test_r3_weak_weapon_terms_are_unknown_not_reject(activity):
    ev, comp = run(clean_with(corporate_object=[activity]))
    assert {r.rule_id: r.fired for r in ev.rule_results}["ACT-01"] == "unknown"
    assert comp.status == "MANUAL_REVIEW"
    _, comp = run(clean_with(corporate_object=["Comercio de armas de fuego y municiones"]))
    assert comp.status == "REJECTED"


def by_rule(ev) -> dict[str, str]:
    return {r.rule_id: r.fired for r in ev.rule_results}


SAN = {
    "entry_id": "SAN-9",
    "list_kind": "sanctions",
    "name": "Andes Agropecuaria SAS",
    "jurisdiction": "CO",
    "registration_number": "900-1",
}


def test_l3b_r1a_corroboration_wins_over_a_conflicting_identifier():
    fs, _, _ = packet_a_clean()
    mixed = screening(
        fs, candidate("s", "e-cl-andes-agro", SAN, 0.95, corr=["jurisdiction+dob"], conf=["reg_no"])
    )
    ev, comp = run((fs, mixed, {}))
    assert by_rule(ev)["SCR-01"] == "yes" and comp.status == "REJECTED"
    assert "conflicts" in {r.rule_id: r for r in ev.rule_results}["SCR-01"].description
    assert by_rule(ev)["SCR-02"] == "yes"
    ev, comp = run(
        (fs, screening(fs, candidate("s", "e-cl-andes-agro", SAN, 0.95, conf=["reg_no"])), {})
    )  # noqa: E501
    assert by_rule(ev)["SCR-01"] == "no" and comp.status == "MANUAL_REVIEW"
    low = screening(fs, candidate("s", "e-cl-andes-agro", SAN, 0.60, corr=["reg_no"], conf=["dob"]))
    ev, comp = run((fs, low, {"sp-s": answer("sp-s", False)}))
    assert by_rule(ev)["SCR-01"] == "no" and comp.status == "APPROVED"


def test_l3b_r6_corroboration_wins_under_d012():
    fs, _, _ = packet_a_clean()
    pep = {
        "entry_id": "PEP-9",
        "list_kind": "pep",
        "name": "Herrera",
        "nationality": "PE",
        "dob": "1982-11-03",
        "in_office": True,
    }
    mixed = screening(fs, candidate("p", "e-herrera", pep, 1.0, corr=["dob"], conf=["nationality"]))
    ev, comp = run((fs, mixed, {}))
    assert by_rule(ev)["PEP-01"] == "yes" and (comp.status, comp.risk_score) == (
        "MANUAL_REVIEW",
        30,
    )
    only = screening(fs, candidate("p", "e-herrera", pep, 1.0, conf=["nationality", "dob"]))
    ev, comp = run((fs, only, {"sp-p": answer("sp-p", False)}))
    assert by_rule(ev)["PEP-01"] == "no" and comp.status == "APPROVED"


def in_region(eid: str, code: str, *stated: str):
    def f(ents):
        return [
            {**e, "jurisdiction": [code], "regions": list(stated)} if e["entity_id"] == eid else e
            for e in ents
        ]

    return f


@pytest.mark.parametrize(
    ("stated", "r2", "status"),
    [
        (("Autonomous Republic of Crimea",), "yes", "REJECTED"),
        (("Luhans'k Oblast",), "yes", "REJECTED"),
        (("Kyiv", "Donetsk Oblast"), "yes", "REJECTED"),
        (("Donets'k Oblast",), "yes", "REJECTED"),
        (("Kramatorsk",), "yes", "REJECTED"),
        (("Kyiv",), "no", "APPROVED"),
        (("Odesa Oblast", "Lviv"), "no", "APPROVED"),
        ((), "unknown", "MANUAL_REVIEW"),
        (("Донецька область",), "unknown", "MANUAL_REVIEW"),
        (("Donetzk Oblasst",), "unknown", "MANUAL_REVIEW"),
        (("Eastern Ukraine",), "unknown", "MANUAL_REVIEW"),
        (("not stated",), "unknown", "MANUAL_REVIEW"),
        (("N/A",), "unknown", "MANUAL_REVIEW"),
        (("Possibly Donetsk Oblast",), "unknown", "MANUAL_REVIEW"),
        (("Kyiv (not Donetsk)",), "unknown", "MANUAL_REVIEW"),
        (("Kyiv", "Lugansk"), "yes", "REJECTED"),
        (("Kyiv", "N/A"), "unknown", "MANUAL_REVIEW"),
    ],
)
def test_l3b_r2_reads_ua_regions(stated, r2, status):
    ev, comp = run(clean_with(in_region("e-cl-andes-agro", "UA", *stated)))
    assert by_rule(ev)["JUR-01"] == r2 and comp.status == status


def test_l3b_r2_reads_a_ubos_region():
    ev, comp = run(clean_with(in_region("e-herrera", "UA", "Lugansk")))
    assert by_rule(ev)["JUR-01"] == "yes" and comp.status == "REJECTED"
    ev, comp = run(clean_with(in_region("e-herrera", "UA", "Lviv")))
    assert by_rule(ev)["JUR-01"] == "no" and comp.status == "APPROVED"


@pytest.mark.parametrize(
    ("eid", "stated", "r3", "outcome"),
    [
        ("e-cl-andes-agro", ("DMCC Free Zone",), "yes", ("APPROVED", 20)),
        ("e-cl-andes-agro", ("Jumeirah Lakes Towers, Dubai",), "yes", ("APPROVED", 20)),
        ("e-gaviria", ("JAFZA",), "yes", ("APPROVED", 20)),
        ("e-cl-andes-agro", ("Dubai (mainland)",), "no", ("APPROVED", 0)),
        ("e-cl-andes-agro", ("Dubai",), "unknown", ("MANUAL_REVIEW", 0)),
        ("e-cl-andes-agro", (), "unknown", ("MANUAL_REVIEW", 0)),
        ("e-cl-andes-agro", ("Dubai mainland (not a free zone)",), "unknown", ("MANUAL_REVIEW", 0)),
        ("e-cl-andes-agro", ("Dubai mainland near JAFZA",), "unknown", ("MANUAL_REVIEW", 0)),
    ],
)
def test_l3b_r3_reads_ae_free_zones(eid, stated, r3, outcome):
    ev, comp = run(clean_with(in_region(eid, "AE", *stated)))
    assert by_rule(ev)["JUR-02"] == r3 and (comp.status, comp.risk_score) == outcome


@pytest.mark.parametrize(
    "stated",
    [
        "Dubai mainland (not a free zone)",
        "Dubai without free-zone status",
        "Dubai (DMCC excluded)",
        "formerly JAFZA, now Dubai",
        "Dubai cannot be a free zone",
        "Dubai lacks free-zone status",
        "Dubai may be a free zone",
    ],
)
def test_l3b_a_negated_free_zone_cannot_manufacture_r8_reject(stated):
    shell = ops(ABSENT, ABSENT, ABSENT)
    ev, comp = run(clean_with(in_region("e-cl-andes-agro", "AE", stated), operations=shell))
    assert by_rule(ev)["JUR-02"] == by_rule(ev)["OPS-02"] == "unknown"
    assert comp.status == "MANUAL_REVIEW"
    ev, comp = run(
        clean_with(in_region("e-cl-andes-agro", "AE", "DMCC Free Zone"), operations=shell)
    )  # noqa: E501
    assert by_rule(ev)["OPS-02"] == "yes" and comp.status == "REJECTED"


@pytest.mark.parametrize("apostrophe", ["'", "‘", "’", "‛", "ʼ", "`", "´", "′", "＇"])
@pytest.mark.parametrize("case", [str.lower, str.upper, str.title], ids=["lower", "upper", "title"])
@pytest.mark.parametrize(
    "phrase",
    [
        "Dubai isn't a free zone",
        "Dubai doesn't have free-zone status",
        "Dubai can't be a free zone",
        "Dubai won't have free-zone status",
        "Dubai hasn't been a free zone",
        "Dubai wouldn't be a free zone",
    ],
)
def test_l3b_contracted_negation_cannot_manufacture_reject(apostrophe, case, phrase):
    stated = case(phrase.replace("'", apostrophe))
    ev, comp = run(
        clean_with(
            in_region("e-cl-andes-agro", "AE", stated), operations=ops(ABSENT, ABSENT, ABSENT)
        )  # noqa: E501
    )
    assert by_rule(ev)["JUR-02"] == by_rule(ev)["OPS-02"] == "unknown"
    assert comp.status == "MANUAL_REVIEW"
    ev, comp = run(
        clean_with(
            in_region("e-cl-andes-agro", "UA", stated.replace(case("Dubai"), case("Donetsk")))
        )  # noqa: E501
    )
    assert by_rule(ev)["JUR-01"] == "unknown" and comp.status == "MANUAL_REVIEW"


@pytest.mark.parametrize(
    "stated",
    [
        "Donets'k Oblast",
        "Donets’k Oblast",
        "LUHANS’K OBLAST",
        "Donets‘k Oblast",
        "Donets＇k Oblast",
    ],
)
def test_l3b_transliteration_apostrophes_still_match_positive_regions(stated):
    ev, comp = run(clean_with(in_region("e-cl-andes-agro", "UA", stated)))
    assert by_rule(ev)["JUR-01"] == "yes" and comp.status == "REJECTED"


@pytest.mark.parametrize(
    "phrase",
    [
        "cannot be",
        "lacks",
        "may be",
        "might be",
        "could be",
        "possibly in",
        "perhaps in",
        "allegedly in",
        "reportedly in",
        "potentially in",
        "not necessarily in",
    ],
)
@pytest.mark.parametrize("code,place", [("AE", "Dubai free zone"), ("UA", "Donetsk Oblast")])
def test_l3b_denied_or_uncertain_region_is_unknown(phrase, code, place):
    ev, comp = run(
        clean_with(
            in_region("e-cl-andes-agro", code, f"{phrase} {place}"),
            operations=ops(ABSENT, ABSENT, ABSENT),
        )
    )
    assert by_rule(ev)["JUR-02" if code == "AE" else "JUR-01"] == "unknown"
    assert comp.status == "MANUAL_REVIEW"


@pytest.mark.parametrize(
    "code,stated",
    [
        ("AE", "Dubai free-zone status denied"),
        ("AE", "Dubai free-zone status unverified"),
        ("AE", "Pending confirmation of DMCC registration"),
        ("AE", "Dubai is unlikely to be a free zone"),
        ("UA", "Donetsk residence denied"),
        ("UA", "Donetsk Oblast (unverified)"),
        ("UA", "Kyiv (unverified)"),
        ("AE", "Is Dubai a free zone?"),
        ("UA", "Is Donetsk the registered address?"),
    ],
)
def test_l3b_r5_unestablished_region_membership_cannot_reject_or_clear(code, stated):
    ev, comp = run(
        clean_with(
            in_region("e-cl-andes-agro", code, stated), operations=ops(ABSENT, ABSENT, ABSENT)
        )  # noqa: E501
    )
    assert by_rule(ev)["JUR-02" if code == "AE" else "JUR-01"] == "unknown"
    if code == "AE":
        assert by_rule(ev)["OPS-02"] == "unknown"
    assert comp.status == "MANUAL_REVIEW"


@pytest.mark.parametrize(
    "context",
    [
        "denied",
        "unverified",
        "pending confirmation",
        "unlikely",
        "hypothetically",
        "purported",
        "unexplained-marker-xyz",
    ],
)
@pytest.mark.parametrize("code,place", [("AE", "DMCC free zone"), ("UA", "Donetsk Oblast")])
def test_l3b_unexplained_context_is_unknown_without_a_denyword_list(context, code, place):
    for stated in (f"{context} {place}", f"{place} ({context})"):
        ev, comp = run(
            clean_with(
                in_region("e-cl-andes-agro", code, stated), operations=ops(ABSENT, ABSENT, ABSENT)
            )  # noqa: E501
        )
        assert by_rule(ev)["JUR-02" if code == "AE" else "JUR-01"] == "unknown"
        assert comp.status == "MANUAL_REVIEW"


@pytest.mark.parametrize(
    "code,stated,rule,status",
    [
        ("AE", "Dubai is a free zone", "OPS-02", "REJECTED"),
        ("AE", "The company is registered in DMCC Free Zone, Dubai, UAE", "OPS-02", "REJECTED"),
        ("AE", "registered office in JAFZA", "OPS-02", "REJECTED"),
        ("UA", "The applicant is resident in Donetsk Oblast", "JUR-01", "REJECTED"),
        ("UA", "registered address in Donetsk Oblast, Ukraine", "JUR-01", "REJECTED"),
        ("UA", "Donets‛k Oblast", "JUR-01", "REJECTED"),
        ("UA", "Donets′k Oblast", "JUR-01", "REJECTED"),
        ("UA", "registered address in Kyiv", "JUR-01", "MANUAL_REVIEW"),
    ],
)
def test_l3b_known_positive_context_and_transliteration_still_fire(code, stated, rule, status):
    ev, comp = run(
        clean_with(
            in_region("e-cl-andes-agro", code, stated), operations=ops(ABSENT, ABSENT, ABSENT)
        )  # noqa: E501
    )
    assert by_rule(ev)[rule] == ("no" if stated.endswith("Kyiv") else "yes")
    assert comp.status == status


@pytest.mark.parametrize(
    "code,stated",
    [
        ("AE", "Is Dubai a free zone"),
        ("AE", "Are Dubai free zones"),
        ("AE", "Has Dubai free-zone status"),
        ("AE", "Does Dubai have free-zone status"),
        ("AE", "Can Dubai be a free zone"),
        ("AE", "Dubai mainland free zone"),
        ("AE", "DMCC mainland"),
        ("AE", "DMCC Free Zone status"),
        ("AE", "DMCC registration status: ✗"),
        ("AE", "DMCC registration status: ☐"),
        ("UA", "Is the applicant resident in Donetsk"),
        ("UA", "Are the owners resident in Donetsk"),
        ("UA", "Is the registered address in Kyiv"),
        ("UA", "Has the applicant residence in Kyiv"),
        ("UA", "Does the applicant have residence in Kyiv"),
        ("UA", "Kyiv residence status"),
        ("UA", "DonetskUnverified"),
        ("UA", "KyivUnverified"),
    ],
)
def test_l3b_r6_questions_inversions_and_unsupported_forms_stay_unknown(code, stated):
    kwargs = {"operations": ops(ABSENT, ABSENT, ABSENT)} if code == "AE" else {}
    ev, comp = run(clean_with(in_region("e-cl-andes-agro", code, stated), **kwargs))
    assert by_rule(ev)["JUR-02" if code == "AE" else "JUR-01"] == "unknown"
    if code == "AE":
        assert by_rule(ev)["OPS-02"] == "unknown"
    assert comp.status == "MANUAL_REVIEW"


@pytest.mark.parametrize(
    "mark",
    [
        "✗",
        "☐",
        "☒",
        "✓",
        "☑",
        "✔",
        "✅",
        "❌",
        "×",
        "✕",
        "✘",
        "□",
        "■",
        "√",
        "?",
        "[ ]",
        "[x]",
        "()",
        "0",
        "-",
        "–",
        "—",
        "ⓧ",
        "⁰",
        "\u200b",
        "\x00",
        "\x1c",
        "\n",
    ],
)
@pytest.mark.parametrize("code,place", [("AE", "DMCC"), ("UA", "Donetsk"), ("UA", "Kyiv")])
def test_l3b_r6_raw_status_marks_cannot_be_folded_into_a_hit_or_clearance(mark, code, place):
    kwargs = {"operations": ops(ABSENT, ABSENT, ABSENT)} if code == "AE" else {}
    for stated in (f"{place} residence status: {mark}", f"{place} {mark}"):
        ev, comp = run(clean_with(in_region("e-cl-andes-agro", code, stated), **kwargs))
        assert by_rule(ev)["JUR-02" if code == "AE" else "JUR-01"] == "unknown"
        assert comp.status == "MANUAL_REVIEW"


@pytest.mark.parametrize(
    "code,stated,state",
    [
        ("AE", "DMCC Free Zone, Dubai, UAE.", "yes"),
        ("AE", "The company is registered in DMCC Free-Zone, Dubai, UAE.", "yes"),
        ("AE", "DMCC (Free Zone)", "yes"),
        ("AE", "Dubai (mainland).", "no"),
        ("AE", "Dubai is mainland.", "no"),
        ("UA", "Autonomous Republic of Crimea.", "yes"),
        ("UA", "Donetsk (Oblast).", "yes"),
        ("UA", "The applicant is resident in Donetsk Oblast, Ukraine.", "yes"),
        ("UA", "Registered address: Kyiv.", "no"),
        ("UA", "The registered address is in Kyiv.", "no"),
    ],
)
def test_l3b_anchored_positive_forms_consume_benign_punctuation(code, stated, state):
    ev, comp = run(clean_with(in_region("e-cl-andes-agro", code, stated)))
    assert by_rule(ev)["JUR-02" if code == "AE" else "JUR-01"] == state
    assert comp.status == ("REJECTED" if code == "UA" and state == "yes" else "APPROVED")


def test_l3b_region_names_come_from_the_policy_text():
    pol = edited(("Crimea/Donetsk/Luhansk/Zaporizhzhia/Kherson regions", "Crimea/Luhansk regions"))
    assert pol.unrepresentable == [] and pol.params["prohibited_regions"]["UA"]
    ev, comp = run(clean_with(in_region("e-cl-andes-agro", "UA", "Donetsk")), pol)
    assert by_rule(ev)["JUR-01"] == "unknown"
    ev, comp = run(clean_with(in_region("e-cl-andes-agro", "UA", "Crimea")), pol)
    assert by_rule(ev)["JUR-01"] == "yes" and comp.status == "REJECTED"
    ev, comp = run(clean_with(in_region("e-cl-andes-agro", "UA", "Kyiv")), pol)
    assert by_rule(ev)["JUR-01"] == "no" and comp.status == "APPROVED"
    pol = edited(
        (
            "Crimea/Donetsk/Luhansk/Zaporizhzhia/Kherson regions",
            "Crimea/Donetsk/Luhansk/Mykolaiv regions",
        )
    )
    assert any(
        u.startswith("prohibited_jurisdictions:") and "mykolaiv" in u for u in pol.unrepresentable
    )
    ev, comp = run(clean_with(in_region("e-cl-andes-agro", "UA", "Mykolaiv")), pol)
    assert by_rule(ev)["JUR-01"] == "unknown" and comp.status == "MANUAL_REVIEW"


def test_l3b_an_unverified_region_claim_is_a_gap():
    fs, sr, ans = clean_with(in_region("e-cl-andes-agro", "UA", "Kyiv"))
    ev_ok, _ = run((fs, sr, ans))
    assert by_rule(ev_ok)["JUR-01"] == "no"
    bad = {"claim_id": "c-A01-09", "doc": "01", "page": 1, "verified": False}
    body = fs.model_dump(
        exclude={"case", "run", "as_of", "fact_set_hash", "evidence", "contracts_version"}
    )
    fs2 = facts("A", **{**body, "unverified": [bad]})
    fs2 = fs2.model_copy(
        update={
            "evidence": [
                e.model_copy(update={"field": "[e-cl-andes-agro].region"})
                if e.evidence_id == "c-A01-09"
                else e
                for e in fs2.evidence
            ]
        }
    )
    ev, comp = run((fs2, screening(fs2), ans))
    assert by_rule(ev)["JUR-01"] == "unknown" and comp.status == "MANUAL_REVIEW"
    assert by_rule(ev)["ACT-02"] == by_rule(ev)["DOC-02"] == "no"


def unresolved_corp(eid: str, claim: str) -> dict:
    return entity(eid, "company", eid, claim, jurisdiction=["CO"])


@pytest.mark.parametrize(
    ("stakes", "own01", "status"),
    [
        ((70, 15, 15), "yes", "MANUAL_REVIEW"),
        ((82, 9, 9), "no", "APPROVED"),
        ((75, 25, 0), "yes", "MANUAL_REVIEW"),
        ((75, 0, 0), "yes", "MANUAL_REVIEW"),
    ],
)
def test_l3b_own01_fires_on_the_aggregate(stakes, own01, status):
    herrera, x, y = stakes
    hold = [
        holding("h-1", "e-herrera", "e-cl-andes-agro", herrera, "A02-03"),
        *([holding("h-2", "e-x", "e-cl-andes-agro", x, "A02-61")] if x else []),
        *([holding("h-3", "e-y", "e-cl-andes-agro", y, "A02-62")] if y else []),
    ]
    pk = clean_with(
        lambda es: [*es, unresolved_corp("e-x", "A02-60"), unresolved_corp("e-y", "A02-63")],
        holdings=hold,
    )
    ev, comp = run(pk)
    assert by_rule(ev)["OWN-01"] == own01 and comp.status == status
    if own01 == "yes":
        cited = {e.claim_id for e in {r.rule_id: r for r in ev.rule_results}["OWN-01"].evidence}
        used = {h.evidence[0].claim_id for h in pk[0].holdings}
        assert used <= cited and "c-A02-03" in cited


def test_l3b_own01_aggregate_with_a_gap_is_unknown_not_yes():
    hold = [
        holding("h-1", "e-herrera", "e-cl-andes-agro", 70, "A02-03"),
        holding("h-2", "e-x", "e-cl-andes-agro", 15, "A02-61"),
        holding("h-3", "e-y", "e-cl-andes-agro", 15, "A02-62"),
    ]
    gap = [
        {
            "doc": "03",
            "assurance": "W2",
            "isolated": True,
            "uncovered": [
                {
                    "block_id": "b",
                    "kind": "ownership",
                    "subject": "e-cl-andes-agro",
                    "attribute": "holder",
                    "value": "?",
                }
            ],
        }
    ]
    pk = clean_with(
        lambda es: [*es, unresolved_corp("e-x", "A02-60"), unresolved_corp("e-y", "A02-63")],
        holdings=hold,
        verification=gap,
    )
    ev, comp = run(pk)
    assert by_rule(ev)["OWN-01"] == "unknown" and comp.status == "MANUAL_REVIEW"


FACTS = ROOT / "tests/fixtures/facts"


def packet_c_from_l2() -> tuple[b4.FactSet, b4.ScreeningResult, dict]:
    fs = b4.FactSet.model_validate_json((FACTS / "sg-strait-capital.factset.json").read_bytes())
    san = {
        "entry_id": "SAN-0001",
        "list_kind": "sanctions",
        "name": "Strait Capital FZE",
        "aliases": ["Strait Maritime"],
        "jurisdiction": "AE",
        "registration_number": "DMCC-417733",
    }
    pep = {
        "entry_id": "PEP-0101",
        "list_kind": "pep",
        "name": "Sebastián Paredes Vega",
        "nationality": "PY",
        "dob": "1971-10-15",
        "in_office": False,
        "left_office": "2024-08-14",
    }
    sr = screening(
        fs,
        candidate("c-1", fs.applicant, san, 0.903627, conf=["jurisdiction", "reg_no"]),
        {
            **candidate(
                "c-2",
                "e-sebastian-paredes-vega",
                pep,
                1.0,
                corr=["dob", "nationality+dob"],
                alias_exact=True,
            ),
            "triggers_fuzzy": False,
        },
    )
    answers = {i: answer(i, False) for i in ("sp-c-1", "poa-04", "poa-06")}
    return fs, sr, answers


def test_l3b_golden_c_reproduces_from_l2s_real_facts():
    fs, sr, ans = packet_c_from_l2()
    assert {i.instance_id for i in model_instances(fs, sr, parse_policy(RAW))} == {
        "poa-05",
        "poa-06",
        "sp-c-1",
    }
    ev, comp = run((fs, sr, ans))
    assert fired(ev) == {"ACT-02", "SCR-02", "DOC-01", "MOD-01", "OWN-01", "JUR-02"}
    assert [r.rule_id for r in ev.rule_results if r.fired == "unknown"] == ["DOC-02", "OPS-01"]
    assert (comp.status, comp.risk_score) == ("MANUAL_REVIEW", 85)
    act = {r.rule_id: r for r in ev.rule_results}["ACT-02"]
    assert "maritime vessel" in act.description and act.score_delta == 10


@pytest.mark.parametrize(
    ("activity", "r10"),
    [
        ("Financiamiento de exportaciones agrícolas", "yes"),
        ("Pre-export financing for coffee growers", "yes"),
        ("issuing letters of credit for importers", "yes"),
        ("payment to sellers against bills of lading", "yes"),
        ("la importación y exportación de mercancías, así como obtener financiamiento", "no"),
        ("freight forwarding; issuing house bills of lading for export shipments", "no"),
        ("financial advisory on important corporate matters", "no"),
        ("Development and export of financial software", "no"),
        ("financial reporting of shipments for importers", "no"),
        ("Develop and export financial software", "no"),
        ("Import financial software for accounting departments", "no"),
        ("commodity trade financing", "yes"),
        ("import financing for distributors", "yes"),
    ],
)
def test_l3b_trade_finance_is_read_from_what_the_object_describes(activity, r10):
    ev, _ = run(clean_with(corporate_object=[activity]))
    assert by_rule(ev)["ACT-02"] == r10


def test_l3b_every_lexicon_term_is_well_formed():
    from kyb.policy import industry, regions
    from kyb.policy.text import matches

    terms = [t.lstrip("~") for ts in industry.LEXICON.values() for t in ts]
    terms += [
        t
        for lex in (regions.PLACES, regions.CATEGORIES, regions.OUTSIDE)
        for ts in lex.values()
        for t in ts
    ]
    for t in terms:
        matches(t, "")
    with pytest.raises(ValueError):
        matches("a & b & c", "a b c")


def test_l3b_own01_cites_the_ownership_chain_it_resolves():
    def ents(dob):
        return lambda es: [
            *es,
            unresolved_corp("e-x", "A02-60"),
            unresolved_corp("e-y", "A02-63"),
            entity(
                "e-px", "person", "PX", "A02-65", nationality=["CO"], dob=dob, jurisdiction=["CO"]
            ),
        ]

    hold = [
        holding("h-1", "e-herrera", "e-cl-andes-agro", 70, "A02-03"),
        holding("h-2", "e-x", "e-cl-andes-agro", 15, "A02-61"),
        holding("h-3", "e-y", "e-cl-andes-agro", 15, "A02-62"),
        holding("h-4", "e-px", "e-x", 100, "A02-64"),
    ]
    ev, comp = run(clean_with(ents([]), holdings=hold))
    own01 = {r.rule_id: r for r in ev.rule_results}["OWN-01"]
    assert own01.fired == "yes" and comp.status == "MANUAL_REVIEW"
    cited = {e.claim_id for e in own01.evidence}
    assert {"c-A02-03", "c-A02-61", "c-A02-62", "c-A02-64", "c-A02-65"} <= cited
    ev, _ = run(clean_with(ents(["1970-01-01"]), holdings=hold))
    assert by_rule(ev)["OWN-01"] == "no"


def test_single_holding_layer_does_not_fire_gov03():
    """A single resolved corporate holding layer does not trigger GOV-03 (dormant_or_layered)."""
    hold = [
        holding("h-1", "e-hold", "e-cl-andes-agro", 100, "A02-03"),
        holding("h-2", "e-herrera", "e-hold", 100, "A02-04"),
    ]
    pk = clean_with(
        lambda es: [*es, entity("e-hold", "company", "HoldCo", "A02-30", jurisdiction=["CO"])],
        holdings=hold,
    )
    ev, comp = run(pk)
    assert "GOV-03" not in fired(ev) and comp.status == "APPROVED"


def test_layered_or_dormant_company_fires_gov03():
    """Two or more corporate holding layers or a declared dormant status fires GOV-03."""
    hold = [
        holding("h-1", "e-hold1", "e-cl-andes-agro", 100, "A02-03"),
        holding("h-2", "e-hold2", "e-hold1", 100, "A02-04"),
        holding("h-3", "e-herrera", "e-hold2", 100, "A02-05"),
    ]
    pk = clean_with(
        lambda es: [
            *es,
            entity("e-hold1", "company", "HoldCo 1", "A02-30", jurisdiction=["CO"]),
            entity("e-hold2", "company", "HoldCo 2", "A02-31", jurisdiction=["CO"]),
        ],
        holdings=hold,
    )
    ev, comp = run(pk)
    assert "GOV-03" in fired(ev) and comp.status == "MANUAL_REVIEW"
    ev2, comp2 = run(clean_with(corporate_object=["Sociedad en inactividad temporal (dormant)"]))
    assert "GOV-03" in fired(ev2) and comp2.status == "MANUAL_REVIEW"
