"""Reader vocabulary reaches the work order without changing fact or trust semantics."""

from __future__ import annotations

import json
import shutil
from decimal import Decimal
from pathlib import Path

import pytest

from kyb.facts.derive import DocInput, _claims, derive_factset
from kyb.read.field_guide import reader_contract

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "examples"


def claim(cid, typ, attr, value):
    return {
        "claim_id": cid,
        "type": typ,
        "attribute": attr,
        "subject": "Example Holdings Ltd",
        "value": value,
        "excerpt": f"Example Holdings Ltd {value}",
        "page": 1,
        "bbox": [10, 10, 300, 50],
        "block_id": "b1",
        "source": "pixels",
        "confidence": 0.9,
    }


def derive(extra, *, rejected=(), document_name="declaration.pdf"):
    claims = [claim("c-X01-01", "identity", "legal_name", "Example Holdings Ltd"), *extra]
    doc = DocInput(
        "01",
        document_name,
        claims,
        {c["claim_id"]: c["claim_id"] not in rejected for c in claims},
    )
    return derive_factset("example", "r1", "2026-10-25", [doc], materiality=Decimal("5"))


@pytest.mark.parametrize(
    "field", reader_contract()["preferred_fields"], ids=lambda f: f["attribute"]
)
def test_preferred_fields_match_the_production_classifier(field):
    c = claim("c-X01-01", field["type"], field["attribute"], "literal source value")
    (classified,) = _claims([DocInput("01", "declaration.pdf", [c], {c["claim_id"]: True})])
    assert classified.kind == field["fact_class"]


def test_document_date_does_not_become_an_incorporation_date():
    fs = derive([claim("c-X01-02", "document", "document_date", "2020-02-03")])
    assert fs.documents[0].issued == "2020-02-03"
    assert next(e for e in fs.entities if e.entity_id == fs.applicant).incorporated == []
    fs = derive(
        [
            claim("c-X01-02", "document", "document_date", "2020-02-03"),
            claim("c-X01-03", "identity", "incorporation_date", "2010-01-02"),
        ]
    )
    assert fs.documents[0].issued == "2020-02-03"
    assert next(e for e in fs.entities if e.entity_id == fs.applicant).incorporated == [
        "2010-01-02"
    ]


def test_unverified_activity_and_date_stay_unknown():
    fs = derive(
        [
            claim("c-X01-02", "activity", "corporate_object", "logistics"),
            claim("c-X01-03", "identity", "incorporation_date", "2010-01-02"),
        ],
        rejected={"c-X01-02", "c-X01-03"},
    )
    assert fs.corporate_object == []
    assert next(e for e in fs.entities if e.entity_id == fs.applicant).incorporated == []
    assert {r.claim_id for r in fs.unverified} == {"c-X01-02", "c-X01-03"}


def test_disclosure_wording_is_evidence_not_a_controller_or_holding():
    wording = "controller information will be supplied separately"
    fs = derive([claim("c-X01-02", "statement", "controller_disclosure", wording)])
    assert fs.roles == [] and fs.holdings == []
    assert len(fs.entities) == 1 and fs.entities[0].kind == "company"
    evidence = next(e for e in fs.evidence if e.evidence_id == "c-X01-02")
    assert evidence.verified and wording in evidence.excerpt and evidence.field is None
    absent = derive([])
    assert all(wording not in e.excerpt for e in absent.evidence)


def test_pending_acquisition_uses_transfer_without_changing_registered_holding():
    holding = claim("c-X01-02", "owns", "ownership", {"pct": 30})
    pending = claim(
        "c-X01-03",
        "transfer",
        "transfer",
        "acquires 15% of Example Holdings Ltd, pending registration",
    )
    for c in (holding, pending):
        c["subject"] = "Cedar Ventures Ltd"
        c["object"] = "Example Holdings Ltd"
    holding["excerpt"] = "Cedar Ventures Ltd holds 30% of Example Holdings Ltd."
    pending["excerpt"] = (
        "Cedar Ventures Ltd acquires 15% of Example Holdings Ltd, pending registration."
    )
    fs = derive([holding, pending])
    assert {(h.status, h.pct) for h in fs.holdings} == {
        ("current", Decimal("30")),
        ("pending", Decimal("15")),
    }
    assert sum(h.pct for h in fs.holdings if h.status == "current") == Decimal("30")


def test_other_claim_can_introduce_a_subject_without_a_controller_role():
    disclosure = claim("c-X01-02", "statement", "controller_disclosure", "details deferred")
    disclosure["subject"] = "Example Investor Ltd"
    disclosure["excerpt"] = "Example Investor Ltd details deferred"
    fs = derive([disclosure])
    assert any("Example Investor Ltd" in e.names for e in fs.entities)
    assert fs.roles == [] and fs.holdings == []


def test_other_is_not_policy_neutral_in_a_bank_document():
    c = claim("c-X01-02", "statement", "controller_disclosure", "details deferred")
    ordinary = derive([c])
    bank = derive([c], document_name="bank_reference.pdf")
    assert ordinary.operations.bank_reference_or_financials.state == "unknown"
    assert bank.operations.bank_reference_or_financials.state == "present"
    assert bank.roles == [] and bank.holdings == []


def test_contract_is_fresh_generic_json_without_case_answers():
    first = reader_contract()
    expected = json.dumps(first, sort_keys=True)
    first["preferred_fields"][0]["meaning"] = "mutated"
    first["review_before_submission"].clear()
    assert json.dumps(reader_contract(), sort_keys=True) == expected
    assert not any(
        s in expected for s in ("Iberia", "Altiplano", "Belice", "MANUAL_REVIEW", "70", "45")
    )


@pytest.mark.skipif(shutil.which("pdftotext") is None, reason="poppler missing")
def test_real_mcp_work_order_contains_guide_and_keeps_document_scope(tmp_path, monkeypatch):
    from kyb import mcp_server as mcp
    from kyb.witness.port import Witness

    class LayoutOnly:
        model, prompt_version = "offline-layout-double", "test"

        def call(self, kind, images, spec):
            assert kind == "layout"
            return {
                "pages": [
                    {
                        "page": 1,
                        "blocks": [
                            {
                                "block_id": "b1",
                                "kind": "paragraph",
                                "box_2d": [0, 0, 1000, 1000],
                                "refers_to": [],
                            }
                        ],
                    }
                ],
                "material_facts": [],
            }, {}

    packet = tmp_path / "packet"
    packet.mkdir()
    for source in sorted((ASSETS / "packets/es-iberia-software").glob("*.pdf"))[:2]:
        shutil.copyfile(source, packet / source.name)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("KYB_WORKSPACE", raising=False)
    monkeypatch.setattr(
        "kyb.witness.port.make_witness", lambda root, mode: Witness(root, LayoutOnly())
    )
    mcp.init_workspace(policy=str(ASSETS / "policy/compliance_rules.md"))
    mcp.ingest_packet(str(packet), "2026-10-25", case="example")
    mcp.pages("example")
    mcp.layout_pages("example")
    entries = mcp.read_plan("example", "r1")
    assert len(entries) == 2
    doc = entries[0]["doc"]
    (entry,) = mcp.read_plan("example", "r1", doc)
    assert entry["doc"] == doc and entry["reader_contract"] == reader_contract()
    assert entry["launch_prompt"] == f"kyb-reader example r1 {doc}"
    assert entry["skill"].startswith("kyb-read-packet@0.2.2 ")
    assert all(Path(p["png"]).exists() for p in entry["pages"])
    entry["reader_contract"]["preferred_fields"].clear()
    assert mcp.read_plan("example", "r1", doc)[0]["reader_contract"] == reader_contract()
    with pytest.raises(KeyError):
        mcp.read_plan("example", "r1", "not-this-document")
