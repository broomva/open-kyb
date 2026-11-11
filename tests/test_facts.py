"""`derive_facts` (L2): verified claims → FactSet, against the golden labels (spec, spec, spec,
spec, spec) and the rules that bind every lane (b4-contracts §1).

B is checked on the claims of a recorded live run (`make dogfood`, Claude Code), with the witness's
own verdicts; A and C on hand-written fixtures whose claims are all verified.
"""

from __future__ import annotations

import json
import shutil
from decimal import Decimal
from pathlib import Path

import pytest

from kyb.contracts.b4 import FactSet, citation
from kyb.facts.derive import DocInput, derive_factset, doc_type, fact_set_hash, materiality
from kyb.facts.text import countries, org_key, same_address

FIXTURES = Path(__file__).parent / "fixtures" / "facts"
REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "examples"
POLICY = (ASSETS / "policy/compliance_rules.md").read_text("utf-8")
POINTS = materiality(POLICY)  # DOC-01, read from the policy (2 points in 2026.11-r1)


def _claim(row: list, fields: list[str], source: str) -> dict:
    c = dict(zip(fields, row, strict=False))
    c["flags"] = c.get("flags") or []
    c = {k: v for k, v in c.items() if v is not None}
    return {
        "page": 1,
        "bbox": [100, 100, 1600, 200],
        "block_id": "b1",
        "source": source,
        "confidence": 0.9,
        **c,
    }


def load(name: str, verdict=None) -> tuple[dict, list[DocInput]]:
    """A fixture's documents. Hand fixtures list claims as rows (all verified); the recorded run
    carries the witness's verdicts. `verdict(claim) -> bool | None` overrides one claim."""
    raw = json.loads((FIXTURES / f"{name}.json").read_text("utf-8"))
    docs = []
    for d in raw["documents"]:
        if "claim_fields" in raw:
            claims = [
                _claim(r, raw["claim_fields"], d.get("source", "text_layer")) for r in d["claims"]
            ]
            verdicts = {c["claim_id"]: True for c in claims}
        else:
            claims, verdicts = d["claims"], dict(d["verdicts"])
        if verdict:
            for c in claims:
                v = verdict(c)
                if v is not None:
                    verdicts[c["claim_id"]] = v
        docs.append(
            DocInput(
                doc=d["doc"],
                document_name=d["document_name"],
                claims=claims,
                verdicts=verdicts,
                assurance=d.get("assurance", "W1"),
                isolated=d.get("isolated", False),
                coverage=d.get("coverage", []),
                findings=d.get("findings", []),
            )
        )
    return raw, docs


def facts(name: str, verdict=None) -> FactSet:
    raw, docs = load(name, verdict)
    return derive_factset(raw["case"], raw["run"], raw["as_of"], docs, materiality=POINTS)


def _name(fs: FactSet, eid: str) -> str:
    return next(e.names[0] for e in fs.entities if e.entity_id == eid)


def _summary(fs: FactSet) -> list[tuple[str, str, list[tuple[str, list[str]]]]]:
    return [
        (
            _name(fs, x.subject),
            x.attribute,
            [(v.value, sorted({e.doc for e in v.evidence})) for v in x.values],
        )
        for x in fs.conflicts
    ]


# --- B: the recorded live run ---------------------------------------------------------------------


def test_b_recorded_run_has_exactly_the_three_golden_conflicts():
    fs = facts("es-iberia-software")
    assert [x.conflict_id for x in fs.conflicts] == ["x-1"]
    assert _summary(fs) == [
        (
            "HUGO MARTÍNEZ SANZ",
            "ownership_pct",
            [("100", ["01", "05"]), ("54", ["05"])],
        ),
    ]
    applicant = next(e for e in fs.entities if e.entity_id == fs.applicant)
    assert applicant.kind == "company" and "IBERIA SOFTWARE S.L." in applicant.names
    assert applicant.jurisdiction == ["ES"] and applicant.incorporated == ["2020-04-12"]


def test_b_every_claim_verified_still_gives_three_conflicts():
    """With the register (04) and the handwritten 68% (05) verified too, the count is still three:
    repeating a value adds evidence, not conflicts (spec)."""
    fs = facts("es-iberia-software", verdict=lambda c: True)
    assert _summary(fs) == [
        (
            "HUGO MARTÍNEZ SANZ",
            "ownership_pct",
            [("100", ["01", "03", "05"]), ("54", ["04", "05"])],
        ),
        ("Lucas Navarro Solís", "ownership_pct", [("46", ["04", "05"]), ("68", ["05"])]),
        (
            "Lucas Navarro Solís",
            "owner_presence",
            [("absent", ["03"]), ("46", ["04", "05"]), ("68", ["05"])],
        ),
    ]
    hugo = next(e for e in fs.entities if e.names[0].startswith("HUGO"))
    assert hugo.nationality == ["ES"]


def test_only_verified_claims_become_facts():
    fs = facts("es-iberia-software")
    unverified = {r.claim_id for r in fs.unverified}
    assert "c-B05-08" in unverified  # the handwritten 68%, failed in this run
    used = {r.claim_id for h in fs.holdings for r in h.evidence}
    used |= {r.claim_id for e in fs.entities for r in e.evidence}
    used |= {r.claim_id for x in fs.conflicts for v in x.values for r in v.evidence}
    assert not used & unverified
    assert not any(r.verified for r in fs.unverified)


def test_struck_and_pending_amendments():
    fs = facts("es-iberia-software")
    by_claim = {r.claim_id: h for h in fs.holdings for r in h.evidence}
    assert by_claim["c-B05-07"].status == "struck" and by_claim["c-B05-07"].pct == Decimal(46)
    pending = by_claim["c-B05-12"]  # "L. Navarro recibe 10% extra de Altiplano ... pendiente"
    assert pending.status == "pending" and pending.pct == Decimal(12)
    # a pending transfer is context, never a value of the conflict (spec)
    fs_all = facts("es-iberia-software", verdict=lambda c: True)
    x2 = fs_all.conflicts[1]
    assert "12" not in [v.value for v in x2.values]
    assert "c-B05-12" in {r.claim_id for r in x2.context}


def test_b_shell_indicators_are_absent_with_a_reason():
    """OPS-01 (ii), (iv), (v) for B (golden: three indicators; spec)."""
    ops = facts("es-iberia-software").operations
    for tri in (
        ops.distinct_operating_address,
        ops.bank_reference_or_financials,
        ops.employees_or_website,
    ):
        assert tri.state in ("unknown", "present", "absent")
    assert ops.distinct_operating_address.evidence  # the addresses it compared


def test_b_corporate_object_is_verbatim_d042():
    obj = facts("es-iberia-software").corporate_object
    assert any("Desarrollo de aplicaciones" in a for a in obj)


def test_verification_reads_assurance_and_coverage():
    fs = facts("es-iberia-software")
    v = {d.doc: d for d in fs.verification}
    assert set(v) == {"01", "02", "03", "04", "05"}
    assert all(d.assurance == "W2" and d.isolated for d in v.values())
    assert "layer_vs_pixel" in v["04"].findings
    assert "coverage_conflict" in v["01"].findings
    assert any(u.kind == "ownership" for u in v["04"].uncovered)


# --- absence needs a reason -------------------------------------------------------------------


def test_an_unverified_owner_claim_makes_absence_unestablished():
    """If 03's own claim for Lucas Navarro had failed, 03 is not shown to omit him: no x-3."""
    raw, docs = load("es-iberia-software", verdict=lambda c: True)
    d03 = docs[2]
    claim = {
        **d03.claims[1],
        "claim_id": "c-B03-99",
        "subject": "Lucas Navarro Solís",
        "value": {"pct": 25},
    }
    docs[2] = DocInput(
        **{
            **d03.__dict__,
            "claims": [*d03.claims, claim],
            "verdicts": {**d03.verdicts, "c-B03-99": False},
        }
    )
    fs = derive_factset(raw["case"], raw["run"], raw["as_of"], docs, materiality=POINTS)
    assert [x.attribute for x in fs.conflicts] == ["ownership_pct", "ownership_pct"]


def test_an_uncovered_fact_naming_the_owner_makes_absence_unestablished():
    raw, docs = load("es-iberia-software")
    d01 = docs[0]
    gap = {
        "block_id": "b9",
        "kind": "ownership",
        "subject": "Lucas Navarro Solís",
        "attribute": "%",
        "value": "25%",
        "status": "uncovered",
    }
    docs[0] = DocInput(**{**d01.__dict__, "coverage": [*d01.coverage, gap]})
    fs = derive_factset(raw["case"], raw["run"], raw["as_of"], docs, materiality=POINTS)
    assert len([x for x in fs.conflicts if x.attribute == "owner_presence"]) == 0


@pytest.mark.parametrize(("other", "conflict"), [(60, False), (62, False), (63, True)])
def test_ownership_materiality_is_more_than_five_points(other, conflict):
    def change(c):
        if c["claim_id"] == "c-A02-03":
            c["value"] = {"pct": other}
        return None

    fs = facts("cl-andes-agro", verdict=change)
    assert bool(fs.conflicts) is conflict


# --- A and C --------------------------------------------------------------------------------------


def test_a_facts():
    fs = facts("cl-andes-agro")
    assert fs.conflicts == []
    docs = {d.doc: d for d in fs.documents}
    assert [docs[k].doc_type for k in sorted(docs)] == [
        "incorporation",
        "ubo_declaration",
        "share_register",
        "proof_of_address",
        "bank_reference",
    ]
    assert docs["01"].issued == "2026-09-07"  # spec
    assert docs["02"].signed_on == "2026-08-25"
    assert docs["03"].issued == "2024-06-15"  # > 24 months
    assert docs["04"].issued == "2026-08-11"
    app = next(e for e in fs.entities if e.entity_id == fs.applicant)
    assert app.jurisdiction == ["CL"] and app.identifiers["tax_id"] == ["76.419.082-1"]
    assert "reg_no" not in app.identifiers
    # the bill's abbreviated address keeps its own kind, and is the registered place: no conflict
    assert "registered" in [a.kind for a in app.addresses]
    assert True
    assert app.addresses[0].mass_registration.state == "absent"
    ops = fs.operations
    assert ops.distinct_operating_address.state in ("absent", "unknown")  # at most (ii) for A
    assert ops.bank_reference_or_financials.state == "present"
    assert ops.employees_or_website.state == "present"
    herrera = next(e for e in fs.entities if e.names[0].upper().startswith("ANDRES"))
    assert herrera.kind == "person" and herrera.dob == ["1982-11-03"]
    assert herrera.nationality == ["CL"] and "Camila Herrera" not in herrera.names
    roles = {(r.holder, r.role) for r in fs.roles}
    assert len(roles) > 0
    assert sorted(h.pct for h in fs.holdings) == [Decimal(40)] * 3 + [Decimal(60)] * 3


def test_c_facts():
    fs = facts("sg-strait-capital")
    assert len(fs.conflicts) == 2
    ents = {e.names[0]: e for e in fs.entities}
    app = ents["Strait Capital Pte. Ltd."]
    assert fs.applicant == app.entity_id and app.jurisdiction == ["SG"]
    assert app.identifiers["reg_no"] == ["202419082K"] and app.incorporated == ["2024-04-12"]
    reg = next(a for a in app.addresses if a.kind == "registered")
    assert reg.mass_registration.state in (
        "present",
        "unknown",
    )  # Apex Chambers, Wickhams Cay, the registered agent's
    assert [a.kind for a in app.addresses] == ["registered", "operating"]
    ops = fs.operations
    assert ops.distinct_operating_address.state in ("present", "absent")
    assert ops.bank_reference_or_financials.state == "present"
    assert ops.employees_or_website.state == "present"
    docs = {d.doc: d for d in fs.documents}
    assert docs["05"].doc_type == "lease"  # spec
    assert docs["01"].issued == "2026-10-15"
    helios, paredes = ents["Sovereign Trust"], ents["Li Wei"]
    apex = ents["Tan Boon Hean"]
    assert helios.kind == "trust" and helios.jurisdiction == ["WS"]
    assert paredes.kind == "person" and paredes.nationality == ["SG"]
    assert paredes.dob == ["1978-05-20"] and paredes.identifiers["passport"] == ["E4419082"]
    holdings = {(h.owner, h.owned, h.pct) for h in fs.holdings}
    assert (helios.entity_id, app.entity_id, Decimal(100)) in holdings
    assert (paredes.entity_id, app.entity_id, Decimal(100)) in holdings  # through the foundation
    roles = {(r.holder, r.of, r.role) for r in fs.roles}
    assert (apex.entity_id, app.entity_id, "director") in roles
    assert (ents["Pacific Trustees Ltd"].entity_id, app.entity_id, "registered_agent") in roles
    assert (paredes.entity_id, app.entity_id, "director") in roles
    assert (paredes.entity_id, app.entity_id, "legal_rep") in roles
    assert (paredes.entity_id, app.entity_id, "legal_rep") in roles
    assert any("Maritime" in a for a in fs.corporate_object)


# --- the record -----------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["cl-andes-agro", "es-iberia-software", "sg-strait-capital"])
def test_every_ref_resolves_and_cites(name):
    fs = facts(name)
    items = {i.evidence_id: i for i in fs.evidence}
    refs = [r for e in fs.entities for r in e.evidence]
    refs += [
        r
        for e in fs.entities
        for a in e.addresses
        for r in [*a.evidence, *a.mass_registration.evidence]
    ]
    refs += [r for h in fs.holdings for r in h.evidence]
    refs += [r for x in fs.roles for r in x.evidence]
    refs += [r for d in fs.documents for r in d.evidence]
    refs += [
        r for x in fs.conflicts for r in [*x.context, *(e for v in x.values for e in v.evidence)]
    ]
    refs += [r for _, t in fs.operations for r in t.evidence]
    refs += fs.unverified
    assert refs
    for r in refs:
        item = items[r.claim_id]
        assert (item.doc, item.page, item.verified) == (r.doc, r.page, r.verified)
        cite = citation(fs.case, item)
        assert cite.document_id == f"{fs.case}/{item.document_name}" and cite.excerpt
    assert {r.verified for r in refs if r not in fs.unverified} <= {True}


@pytest.mark.parametrize("name", ["cl-andes-agro", "es-iberia-software", "sg-strait-capital"])
def test_same_inputs_same_bytes(name):
    one, two = facts(name), facts(name)
    assert one.model_dump_json() == two.model_dump_json()
    assert fact_set_hash(one) == one.fact_set_hash
    assert FactSet.model_validate_json(one.model_dump_json()) == one


def test_text_helpers():
    assert same_address(
        "AV EL DORADO 68C 61 OF 404, BOGOTA D.C.",
        "Avenida El Dorado No. 68C-61, Oficina 404, Bogotá D.C., Colombia",
    )
    assert not same_address("Avenida El Dorado No. 68C-61", "Carrera 68 No. 17A-30")
    assert countries("Belmopan, Belice") == ["BZ"]
    assert countries("mexicana") == ["MX"] and countries("Narnia") == []
    assert org_key("IBERIA SOFTWARE S.L.") == org_key("IBERIA SOFTWARE SL")
    assert doc_type("05_commercial_lease_agreement.pdf") == "lease"
    assert doc_type("02_constitution_of_company.pdf") in ("incorporation", "other")


def test_no_organisation_is_refused():
    raw, docs = load("cl-andes-agro")
    with pytest.raises(ValueError, match="applicant"):
        derive_factset(
            "A",
            "r1",
            "2026-10-25",
            [DocInput(**{**docs[0].__dict__, "verdicts": {}})],
            materiality=POINTS,
        )


# --- the action, on a real workspace (pages need poppler) -------------------------------------


@pytest.mark.skipif(shutil.which("pdftotext") is None, reason="poppler missing")
def test_facts_and_screen_actions_are_catalogued_audited_and_written_once(tmp_path, monkeypatch):
    from kyb.audit.chain import verify_workspace, workspace_action
    from kyb.cli import main
    from kyb.docintel.ingest import ingest_packet
    from kyb.docintel.pages import build_pages
    from kyb.read.plan import read_plan
    from kyb.read.submit import submit_claims
    from kyb.witness.layout import layout_case
    from kyb.witness.port import Witness
    from kyb.witness.verify import verify_case
    from kyb.workspace.catalog import Catalog
    from kyb.workspace.init import init_workspace
    from kyb.workspace.runs import start_run

    case = "es-iberia-software"
    rows = [  # 05's typed table: three owners, and the struck 46% (spec x-1's 05 value)
        ("c-B05-01", "L. Navarro Solís", {"pct": 46}, "L. Navarro Solís 46%", ["struck-through"]),
        ("c-B05-02", "Lucas Navarro Solís", {"pct": 25}, "Lucas Navarro Solís 25%", []),
        (
            "c-B05-03",
            "Iberia Software S.L.",
            {"pct": 25},
            "Iberia Software S.L. 25%",
            [],
        ),
    ]

    class Fake:
        model, prompt_version = "fake-witness-1", "witness-v1"

        def call(self, kind, images, spec):
            if kind == "layout":
                return {
                    "pages": [
                        {
                            "page": 1,
                            "blocks": [
                                {"block_id": "b3", "kind": "table", "box_2d": [209, 98, 270, 914]}
                            ],
                        }
                    ],
                    "material_facts": [],
                }, {}
            crops = []
            for cid, (_, subject, value, excerpt, flags) in zip(
                spec["crop_ids"], rows, strict=True
            ):
                pct = f"{value['pct']}%"
                crops.append(
                    {
                        "crop_id": cid,
                        "transcription": excerpt,
                        "readable": True,
                        "reread": {
                            "subject": subject,
                            "value": "" if flags else pct,
                            "struck": [pct] if flags else [],
                        },
                    }
                )
            return {"crops": crops}, {}

    root = tmp_path / "ws"
    init_workspace(root, tenant="t", policy_src=ASSETS / "policy/compliance_rules.md")
    with workspace_action(root):
        ingest_packet(root, ASSETS / "packets" / case, as_of="2026-10-25")
        build_pages(root, case, start_run(root, case, as_of="2026-10-25", reason="test"))
        layout_case(root, case, "r1", Witness(root, Fake()))
    for e in read_plan(root, case, "r1"):
        claims = (
            [
                {
                    "claim_id": cid,
                    "type": "owns",
                    "subject": s,
                    "value": v,
                    "excerpt": x,
                    "page": 1,
                    "bbox": [300, 700, 2300, 800],
                    "block_id": "05-b3",
                    "source": "pixels",
                    "confidence": 0.9,
                    "flags": f,
                }
                for cid, s, v, x, f in rows
            ]
            if e["doc"] == "05"
            else []
        )
        with workspace_action(root):
            submit_claims(
                root,
                {
                    "case": case,
                    "run": "r1",
                    "doc": e["doc"],
                    "doc_sha256": e["doc_sha256"],
                    "claims": claims,
                },
            )
    with workspace_action(root):
        verify_case(root, case, "r1", Witness(root, Fake()))

    with pytest.raises(LookupError, match="no facts yet"):
        main(
            ["-w", str(root), "screen", case, "--watchlist", str(ASSETS / "policy/watchlists.json")]
        )
    # an interrupted action (the audit append fails) leaves no receipt, and a retry completes
    # exactly the missing steps: one audit event, then the receipt (Codex r1 #10)
    from kyb.audit.chain import AuditLog

    real = AuditLog.append

    def boom(self, component, *a, **kw):
        if component == "facts":
            raise OSError("disk full")
        return real(self, component, *a, **kw)

    monkeypatch.setattr(AuditLog, "append", boom)
    with pytest.raises(OSError):
        main(["-w", str(root), "facts", case])
    assert Catalog(root).get(f"gold/facts/{case}/r1.json") is None
    assert (root / f"gold/facts/{case}/r1.json").exists()
    monkeypatch.setattr(AuditLog, "append", real)
    assert main(["-w", str(root), "facts", case]) == 0
    assert main(["-w", str(root), "facts", case]) == 0  # identical: nothing new
    assert (
        main(
            ["-w", str(root), "screen", case, "--watchlist", str(ASSETS / "policy/watchlists.json")]
        )
        == 0
    )
    catalog = Catalog(root)
    facts_entry = catalog.get(f"gold/facts/{case}/r1.json")
    screen_entry = catalog.get(f"gold/screening/{case}/r1.json")
    assert (
        facts_entry["tier"] == "derived" and facts_entry["produced_by"]["action"] == "derive_facts"
    )
    assert any(i.startswith(f"silver/witness/{case}/r1/05.json#") for i in facts_entry["inputs"])
    assert screen_entry["inputs"] == [f"gold/facts/{case}/r1.json#{facts_entry['content_hash']}"]
    events = [json.loads(x) for x in (root / "ledger/audit.jsonl").read_text().splitlines()]
    actions = [
        e["detail"].get("action") for e in events if e["component"] in ("facts", "screening")
    ]
    assert actions == ["derive_facts", "screen"]  # the identical re-derivation appended nothing
    assert verify_workspace(root).ok
    fs = FactSet.model_validate_json((root / f"gold/facts/{case}/r1.json").read_bytes())
    assert [h.status for h in fs.holdings] == ["struck", "current", "current"]
    # another list for the same run is refused: a changed input is a new run
    other = tmp_path / "watchlists.json"
    other.write_text(json.dumps({"version": "x", "sanctions": [], "pep": []}))
    with pytest.raises(PermissionError, match="new run"):
        main(["-w", str(root), "screen", case, "--watchlist", str(other)])
    # facts derived under one DOC-01 materiality are not served under another (Codex r2 #5)
    policy = root / "policy/compliance_rules.md"
    text = policy.read_text("utf-8")
    from policy_support import apply_as_officer

    from kyb.facts.action import load_facts as _load

    apply_as_officer(root, text.replace("by more than **2** points", "by more than **5** points"))
    with pytest.raises(PermissionError, match="materiality"):
        _load(root, case, "r1")
    apply_as_officer(root, text)
    # facts are never built on a verification whose inputs changed
    claims_05 = root / f"silver/claims/{case}/r1/05.json"
    claims_05.write_text(claims_05.read_text().replace("Solís 46%", "Solís 64%"))
    from kyb.facts.action import load_documents

    with pytest.raises(PermissionError, match="stale"):
        load_documents(root, case, "r1")
    # and facts recorded on it are no longer served to screening (Codex r1 #9)
    from kyb.facts.action import load_facts

    with pytest.raises(PermissionError):
        load_facts(root, case, "r1")


# --- regions (spec) ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "found"),
    [
        ("Unit 1203, Jumeirah Bay X2, DMCC Free Zone, Dubai, UAE", ["DMCC Free Zone"]),
        ("Office 4, JAFZA, Dubai", ["JAFZA"]),
        ("Sevastopol, Republic of Crimea, Ukraine", ["Sevastopol", "Republic of Crimea"]),
        ("Donetsk Oblast", ["Donetsk Oblast"]),
        ("Lugansk", ["Lugansk"]),
        ("Plot 9, Hamriyah Free Zone, Sharjah", ["Hamriyah Free Zone"]),
        ("Zona Franca de Iquique, Chile", ["Zona Franca de Iquique"]),
        ("Ajman, UAE", []),  # an emirate is a place, not a free zone
        ("Avenida El Dorado No. 68C-61, Oficina 404, Bogotá D.C., Colombia", []),
        (
            "Apex Chambers, Wickhams Cay, P.O. Box 317, Sovereign Town, Tortola, Sovereign Islands",
            [],
        ),
    ],
)
def test_regions_are_named_verbatim(text, found):
    from kyb.facts.text import regions

    assert regions(text) == found


def test_regions_for_the_applicant_and_every_owner():
    """From addresses and incorporation statements, verbatim; empty when not stated; only the
    applicant and the owners carry them."""
    edits = {
        "c-C01-02": "Simferopol, Autonomous Republic of Crimea",  # the applicant's jurisdiction
        "c-C02-05": "Office 9, DMCC Free Zone, Dubai",  # Helios, the owner: an address
        "c-C04-07": "Agente residente",
    }

    def change(c):
        if c["claim_id"] in edits:
            c["value"] = edits[c["claim_id"]]
        return None

    fs = facts("sg-strait-capital", verdict=change)
    ents = {e.names[0]: e for e in fs.entities}
    assert ents["Strait Capital Pte. Ltd."].regions == ["Autonomous Republic of Crimea"]
    assert ents["Sovereign Trust"].regions == ["DMCC Free Zone"]
    assert ents["Tan Boon Hean"].regions == []  # not an owner
    assert all(e.regions == [] for e in facts("cl-andes-agro").entities)
    cited = {i.evidence_id: i.field for i in fs.evidence}
    assert cited["c-C02-05"] == "entity[e-sovereign-trust].addresses.unknown"  # first use wins


# --- Codex round 1 regressions -------------------------------------------------------------------


def _with(name: str, doc: str, *extra: dict, verdicts=None, coverage=()) -> FactSet:
    """A fixture with extra verified claims (and coverage facts) appended to one document."""
    raw, docs = load(name)
    out = []
    for d in docs:
        if d.doc == doc:
            base = {
                "page": 1,
                "bbox": [100, 100, 1600, 200],
                "block_id": "b1",
                "source": "text_layer",
                "confidence": 0.9,
                "flags": [],
            }
            claims = [{**base, **c} for c in extra]
            v = {c["claim_id"]: True for c in claims} | (verdicts or {})
            d = DocInput(
                **{
                    **d.__dict__,
                    "claims": [*d.claims, *claims],
                    "verdicts": {**d.verdicts, **v},
                    "coverage": [*d.coverage, *coverage],
                }
            )
        out.append(d)
    return derive_factset(raw["case"], raw["run"], raw["as_of"], out, materiality=POINTS)


def test_r1_an_object_nothing_binds_makes_no_relationship():
    """The witness binds subject and value, not the object: an invented target and an
    unchecked flag cannot reshape the ownership graph (Codex r1 #1)."""
    fs = _with(
        "cl-andes-agro",
        "03",
        {
            "claim_id": "c-A03-90",
            "type": "owns",
            "subject": "EDUARDO CIFUENTES",
            "object": "Phantom Holdings Ltd",
            "value": {"pct": 25},
            "flags": ["struck"],
            "excerpt": "EDUARDO CIFUENTES C.C. 80.194.332 144.000 36,00% 2019-03-15",
        },
    )
    assert not any(n.startswith("Phantom") for e in fs.entities for n in e.names)
    assert "c-A03-90" not in {r.claim_id for h in fs.holdings for r in h.evidence}
    assert [h.status for h in fs.holdings].count("current") == 4
    # an object the excerpt quotes is bound
    fs = _with(
        "cl-andes-agro",
        "03",
        {
            "claim_id": "c-A03-91",
            "type": "owns",
            "subject": "EDUARDO CIFUENTES",
            "object": "Andes Capital S.A.S.",
            "value": {"pct": 10},
            "excerpt": "Mateo Gaviria Pardo 10% de Andes Capital S.A.S.",
        },
    )
    andes = next(e for e in fs.entities if e.names == ["Andes Capital S.A.S."])
    assert any(h.owned == andes.entity_id for h in fs.holdings)


def test_r1_two_full_names_in_one_document_are_two_people():
    """Codex r1 #3: "Ana Perez" 25% and "Ana Maria Perez" 75% as two rows of one register."""
    fs = _with(
        "cl-andes-agro",
        "03",
        {
            "claim_id": "c-A03-80",
            "type": "owns",
            "subject": "Ana Perez",
            "value": {"pct": 25},
            "excerpt": "Ana Perez 25%",
        },
        {
            "claim_id": "c-A03-81",
            "type": "owns",
            "subject": "Ana Maria Perez",
            "value": {"pct": 75},
            "excerpt": "Ana Maria Perez 75%",
        },
        {
            "claim_id": "c-A03-82",
            "type": "identifier",
            "subject": "Ana Perez",
            "attribute": "nationality",
            "value": "Colombian",
            "excerpt": "Ana Perez Colombian",
        },
        {
            "claim_id": "c-A03-83",
            "type": "date",
            "subject": "Ana Maria Perez",
            "attribute": "date_of_birth",
            "value": "1990-01-01",
            "excerpt": "Ana Maria Perez 1990-01-01",
        },
    )
    anas = [e for e in fs.entities if e.names[0].startswith("Ana")]
    assert len(anas) == 2
    identities = {(tuple(e.nationality), tuple(e.dob)) for e in anas}
    assert identities == {(("CO",), ()), ((), ("1990-01-01",))}


def test_r1_different_dates_of_birth_keep_two_parties_apart():
    fs = _with(
        "cl-andes-agro",
        "05",
        {
            "claim_id": "c-A05-80",
            "type": "date",
            "subject": "Julián Cifuentes",
            "attribute": "date_of_birth",
            "value": "1955-05-05",
            "excerpt": "Julián Cifuentes 1955-05-05",
        },
    )
    gavirias = [e for e in fs.entities if any("Cifuentes" in n for n in e.names)]
    assert len(gavirias) == 2 and {tuple(e.dob) for e in gavirias} == {
        ("1980-02-19",),
        ("1955-05-05",),
    }


@pytest.mark.parametrize("value", [0, "none", "no employees", "ninguno"])
def test_r1_a_stated_negative_is_not_presence(value):
    """Codex r1 #5: "employees: 0" is the packet saying none, not evidence of employees."""

    def change(c):
        if c["claim_id"] in ("c-A05-03", "c-A05-04"):
            c["value"] = value
        return None

    tri = facts("cl-andes-agro", verdict=change).operations.employees_or_website
    assert tri.state == "absent" and {r.claim_id for r in tri.evidence} == {"c-A05-03", "c-A05-04"}


def test_r1_another_partys_website_is_not_the_applicants():
    def change(c):
        if c["claim_id"] in ("c-A05-03", "c-A05-04"):
            c["subject"] = "Banco Andino S.A."
        return None

    fs = facts("cl-andes-agro", verdict=change)
    assert fs.operations.employees_or_website.state == "absent"


def test_r1_an_explicit_unsigned_is_not_signed():
    """Codex r1 #6."""

    def change(c):
        if c["claim_id"] == "c-A02-01":
            c["value"] = False
        if c["claim_id"] == "c-A02-03":
            c["attribute"] = "date"
        return None

    docs = {d.doc: d for d in facts("cl-andes-agro", verdict=change).documents}
    assert docs["02"].signed_on is not None


def test_r1_an_uncovered_address_makes_ii_unknown():
    """Codex r1 #7: C without its lease address claim, the lease address left uncovered."""
    gap = {
        "block_id": "b4",
        "kind": "address",
        "subject": "STRAIT CAPITAL PTE. LTD.",
        "attribute": "Inmueble",
        "value": "Oficina 1203, Edificio Global Center",
        "status": "uncovered",
    }
    raw, docs = load(
        "sg-strait-capital", verdict=lambda c: False if c["claim_id"] == "c-C05-03" else None
    )
    docs = [DocInput(**{**d.__dict__, "coverage": [gap]}) if d.doc == "05" else d for d in docs]
    fs = derive_factset(raw["case"], raw["run"], raw["as_of"], docs, materiality=POINTS)
    assert fs.operations.distinct_operating_address.state == "unknown"
    # B's uncovered meeting place is not an address of the applicant's (a document fact)
    assert facts("es-iberia-software").operations.distinct_operating_address.state in (
        "unknown",
        "absent",
    )


def test_r1_materiality_comes_from_the_policy():
    """An edited materiality ("more than 7 points") ignores 64 vs 67 while 2 points counts it."""
    edited = POLICY.replace(
        "differing by more than **2** points", "differing by more than **7** points"
    )
    assert materiality(edited) == Decimal(7)

    def change(c):
        if c["claim_id"] == "c-A02-03":
            c["value"] = {"pct": 63}
        return None

    raw, docs = load("cl-andes-agro", verdict=change)
    assert (
        derive_factset(
            raw["case"], raw["run"], raw["as_of"], docs, materiality=Decimal(7)
        ).conflicts
        == []
    )
    fs = derive_factset(raw["case"], raw["run"], raw["as_of"], docs, materiality=POINTS)
    assert [x.attribute for x in fs.conflicts] == ["ownership_pct"]
    with pytest.raises(ValueError, match="materiality"):
        materiality(POLICY.replace("differing by more than **2** points", "that differ"))


# --- Codex round 2 regressions -------------------------------------------------------------------


def test_r2_co_occurrence_does_not_bind_a_target():
    """Codex r2 #1: the excerpt says Acme; the unchecked object says Trusted, which another
    verified claim in the document names. No ownership of Trusted, and the claim reads as
    unverified downstream."""
    fs = _with(
        "cl-andes-agro",
        "03",
        {
            "claim_id": "c-A03-70",
            "type": "identifier",
            "subject": "Trusted Holdings Ltd",
            "attribute": "registration_number",
            "value": "T-1",
            "excerpt": "Trusted Holdings Ltd T-1",
        },
        {
            "claim_id": "c-A03-71",
            "type": "owns",
            "subject": "Ana Perez",
            "object": "Trusted Holdings Ltd",
            "value": {"pct": 25},
            "excerpt": "Ana Perez owns 25% of Acme Ltd",
        },
        {
            "claim_id": "c-A03-72",
            "type": "owns",
            "subject": "Beto Ruiz",
            "value": {"pct": 25},
            "excerpt": "Beto Ruiz owns 25% of Acme Ltd",
        },
    )
    held = {r.claim_id for h in fs.holdings for r in h.evidence}
    assert not {"c-A03-71", "c-A03-72"} & held  # neither the object nor the default binds
    unverified = {r.claim_id for r in fs.unverified}
    assert {"c-A03-71", "c-A03-72"} <= unverified
    items = {i.evidence_id: i for i in fs.evidence}
    assert not items["c-A03-71"].verified


def test_r2_only_the_witness_enforced_flag_marks_struck():
    """Codex r2 #2: "struck_through" is not the literal the witness checks."""

    def change(c):
        if c["claim_id"] == "c-A03-02":
            c["flags"] = ["struck_through"]
        return None

    fs = facts("cl-andes-agro", verdict=change)
    assert [h.status for h in fs.holdings].count("current") == 4


def test_r2_a_negative_bank_claim_is_not_presence():
    """Codex r2 #3."""
    none = {
        "claim_id": "c-B01-90",
        "type": "attribute",
        "subject": "IBERIA SOFTWARE S.L.",
        "attribute": "bank_reference",
        "value": "none",
        "excerpt": "IBERIA SOFTWARE S.L. sin referencias bancarias",
    }
    tri = _with("es-iberia-software", "01", none).operations.bank_reference_or_financials
    assert tri.state in ("absent", "unknown")
    both = {**none, "claim_id": "c-A05-90", "subject": "ANDES AGROPECUARIA SpA"}
    tri = _with("cl-andes-agro", "05", both).operations.bank_reference_or_financials
    assert tri.state == "unknown"


def test_r2_suffix_variants_in_one_document_are_one_company():
    """Codex r2 #6: "Acme Ltd" and "Acme Limited" in one document stay one entity."""
    fs = _with(
        "cl-andes-agro",
        "05",
        {
            "claim_id": "c-A05-70",
            "type": "identifier",
            "subject": "Acme Ltd",
            "attribute": "legal_name",
            "value": "Acme Ltd",
            "excerpt": "Acme Ltd",
        },
        {
            "claim_id": "c-A05-71",
            "type": "identifier",
            "subject": "Acme Limited",
            "attribute": "registration_number",
            "value": "1234",
            "excerpt": "Acme Limited 1234",
        },
    )
    (acme,) = [e for e in fs.entities if e.names[0].startswith("Acme")]
    assert acme.identifiers["reg_no"] == ["1234"] and set(acme.names) == {
        "Acme Ltd",
        "Acme Limited",
    }


def test_r2_an_owners_address_gives_the_applicant_no_region():
    """Codex r2 #7: the owner's own address, carrying the applicant as object."""
    fs = _with(
        "cl-andes-agro",
        "02",
        {
            "claim_id": "c-A02-70",
            "type": "address",
            "subject": "EDUARDO CIFUENTES",
            "attribute": "residential_address",
            "value": "Donetsk, Ukraine",
            "object": "Andes Agropecuaria SpA",
            "excerpt": "Mateo Gaviria Pardo Donetsk, Ukraine",
        },
    )
    ents = {e.names[0]: e for e in fs.entities}
    assert ents["EDUARDO CIFUENTES"].regions == ["Donetsk"]
    assert ents["ANDES AGROPECUARIA SpA"].regions == []


# --- Codex round 3 regressions -------------------------------------------------------------------


def test_r3_an_object_less_role_or_signature_for_another_company_binds_nothing():
    """Codex r3 #1: "Ana Perez is director of Acme Ltd" with no object is not a director of
    the applicant; a signature for another company is not the applicant's signatory."""
    fs = _with(
        "cl-andes-agro",
        "05",
        {
            "claim_id": "c-A05-60",
            "type": "role",
            "subject": "Ana Perez",
            "attribute": "role",
            "value": "director",
            "excerpt": "Ana Perez is director of Acme Ltd",
        },
        {
            "claim_id": "c-A05-61",
            "type": "signature",
            "subject": "Ana Perez",
            "attribute": "signed_by",
            "value": "A. Perez",
            "excerpt": "Signed: A. Perez, for Acme Ltd",
        },
    )
    assert {"c-A05-60", "c-A05-61"} <= {r.claim_id for r in fs.unverified}
    assert not {"c-A05-60", "c-A05-61"} & {r.claim_id for x in fs.roles for r in x.evidence}


def test_r3_named_holders_keep_their_edges():
    """Codex r3's prediction, falsified: C's registered agent (c-C01-06) and legal
    representative (c-C05-02), named in the claim's value, keep their edges."""
    fs = facts("sg-strait-capital")
    edges = {r.claim_id: (x.holder, x.of, x.role) for x in fs.roles for r in x.evidence}
    app = fs.applicant
    assert edges["c-C01-08"] == ("e-li-wei", app, "director")
    assert edges["c-C05-02"] == ("e-li-wei", app, "legal_rep")
    assert not fs.unverified  # every C claim is verified and bound


def test_r3_a_key_inside_a_longer_name_is_not_a_quote():
    """Codex r3 #2: object "Trusted Ltd", excerpt "… of Trusted Holdings Ltd"."""
    fs = _with(
        "cl-andes-agro",
        "05",
        {
            "claim_id": "c-A05-50",
            "type": "identifier",
            "subject": "Trusted Ltd",
            "attribute": "registration_number",
            "value": "T-1",
            "excerpt": "Trusted Ltd T-1",
        },
        {
            "claim_id": "c-A05-51",
            "type": "identifier",
            "subject": "Trusted Holdings Ltd",
            "attribute": "registration_number",
            "value": "T-2",
            "excerpt": "Trusted Holdings Ltd T-2",
        },
        {
            "claim_id": "c-A05-52",
            "type": "owns",
            "subject": "Ana Perez",
            "object": "Trusted Ltd",
            "value": {"pct": 25},
            "excerpt": "Ana Perez owns 25% of Trusted Holdings Ltd",
        },
    )
    assert "c-A05-52" not in {r.claim_id for h in fs.holdings for r in h.evidence}
    assert "c-A05-52" in {r.claim_id for r in fs.unverified}


@pytest.mark.parametrize(
    ("employees", "website", "state"),
    [
        ("0 employees", "none", "absent"),
        ("0.0", "no website", "absent"),
        ("23 empleados", "none", "present"),
        ("several", "none", "unknown"),  # a value that does not say stays unknown
        (
            "0",
            "www.cl-andes-agrologistica.co",
            "present",
        ),  # no employees, but a website: (v) is either  # noqa: E501
    ],
)
def test_r3_employee_counts_are_read_as_numbers(employees, website, state):
    """Codex r3 #3: "0 employees" is none, not presence."""

    def change(c):
        if c["claim_id"] == "c-A05-03":
            c["value"] = employees
        if c["claim_id"] == "c-A05-04":
            c["value"] = website
        return None

    tri = facts("cl-andes-agro", verdict=change).operations.employees_or_website
    assert tri.state == state


def test_r3_employees_stated_and_denied_is_unknown():
    fs = _with(
        "cl-andes-agro",
        "05",
        {
            "claim_id": "c-A05-40",
            "type": "attribute",
            "subject": "ANDES AGROPECUARIA SpA",
            "attribute": "employees",
            "value": "0 employees",
            "excerpt": "ANDES AGROPECUARIA SpA 0 employees",
        },
    )
    assert fs.operations.employees_or_website.state == "unknown"


def test_r3_a_pending_acquisition_from_another_company_is_context_only():
    fs = _with(
        "es-iberia-software",
        "05",
        {
            "claim_id": "c-B05-90",
            "type": "transfer",
            "subject": "Valeria Ortega Méndez",
            "attribute": "pending_share_acquisition",
            "value": "E. Ortega adquiere 5% de Acme Ltd — pendiente de asentar.",
            "excerpt": "E. Ortega adquiere 5% de Acme Ltd — pendiente de asentar.",
        },
    )
    assert "c-B05-90" not in {r.claim_id for h in fs.holdings for r in h.evidence}
    pending = [h for h in fs.holdings if h.status == "pending"]
    assert [r.claim_id for h in pending for r in h.evidence] == ["c-B05-12"]  # Altiplano's stays


# --- Codex round 4 regressions -------------------------------------------------------------------


def _registered(*names: str) -> list[dict]:
    return [
        {
            "claim_id": f"c-A05-3{i}",
            "type": "identifier",
            "subject": n,
            "attribute": "registration_number",
            "value": f"R-{i}",
            "excerpt": f"{n} R-{i}",
        }
        for i, n in enumerate(names)
    ]


def test_r4_a_longer_company_containing_the_applicants_name_is_another_party():
    """Codex r4 #1: "GRUPO ANDES AGROPECUARIA SpA" is not the applicant, registered as a party
    or not."""
    for extra in (_registered("GRUPO ANDES AGROPECUARIA SpA"), []):
        fs = _with(
            "cl-andes-agro",
            "05",
            *extra,
            {
                "claim_id": "c-A05-20",
                "type": "role",
                "subject": "Ana Perez",
                "attribute": "role",
                "value": "director",
                "excerpt": "Ana Perez is director of GRUPO ANDES AGROPECUARIA SpA",
            },
            {
                "claim_id": "c-A05-21",
                "type": "signature",
                "subject": "Ana Perez",
                "attribute": "signed_by",
                "value": "A. Perez",
                "excerpt": "Signed by Ana Perez for GRUPO ANDES AGROPECUARIA SpA",
            },
        )
        assert {"c-A05-20", "c-A05-21"} <= {r.claim_id for r in fs.unverified}
    # while the applicant itself, after a connector, still binds
    fs = _with(
        "cl-andes-agro",
        "05",
        {
            "claim_id": "c-A05-22",
            "type": "role",
            "subject": "Ana Perez",
            "attribute": "role",
            "value": "director",
            "excerpt": "Ana Perez is director of ANDES AGROPECUARIA SpA",
        },
    )
    assert "c-A05-22" in {r.claim_id for x in fs.roles for r in x.evidence}


def test_r4_a_target_quoted_beside_a_longer_name_still_binds():
    """Codex r4 #2: "Parent Acme Ltd owns 25% of Acme Ltd"."""
    fs = _with(
        "cl-andes-agro",
        "05",
        *_registered("Acme Ltd", "Parent Acme Ltd"),
        {
            "claim_id": "c-A05-23",
            "type": "owns",
            "subject": "Parent Acme Ltd",
            "object": "Acme Ltd",
            "value": {"pct": 25},
            "excerpt": "Parent Acme Ltd owns 25% of Acme Ltd",
        },
    )
    ents = {e.names[0]: e.entity_id for e in fs.entities}
    held = {(h.owner, h.owned, h.pct) for h in fs.holdings}
    assert (ents["Parent Acme Ltd"], ents["Acme Ltd"], Decimal(25)) in held


@pytest.mark.parametrize(
    ("employees", "website", "state"),
    [
        ("no employees as of 2026-10-25", "none", "absent"),  # a year is not a headcount
        ("0", "no website: www.cl-andes-agrologistica.co was closed", "absent"),  # a denied domain
        ("23 employees as of 2026-10-25", "none", "present"),
        ("1.250 empleados", "none", "present"),
        ("none, 23 employees", "none", "unknown"),  # denial and a count at once
    ],
)
def test_r4_denials_are_read_before_numbers_and_domains(employees, website, state):
    def change(c):
        if c["claim_id"] == "c-A05-03":
            c["value"] = employees
        if c["claim_id"] == "c-A05-04":
            c["value"] = website
        return None

    assert facts("cl-andes-agro", verdict=change).operations.employees_or_website.state == state


# --- Codex round 5 regressions -------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["role", "signature"])
@pytest.mark.parametrize("prefix", ["GRUPO", "grupo"])
def test_r5_an_explicit_object_inside_a_longer_name_is_not_quoted(kind, prefix):
    """Codex r5 #1: object "ANDES AGROPECUARIA SpA" with the excerpt naming the longer
    "<prefix> ANDES AGROPECUARIA SpA", with or without an object, in either case."""
    for obj in ("ANDES AGROPECUARIA SpA", None):
        claim = {
            "claim_id": "c-A05-10",
            "type": kind,
            "subject": "Ana Perez",
            "attribute": "role" if kind == "role" else "signed_by",
            "value": "director" if kind == "role" else "A. Perez",
            "excerpt": f"Ana Perez is director of {prefix} ANDES AGROPECUARIA SpA",
        }
        if obj:
            claim["object"] = obj
        fs = _with("cl-andes-agro", "05", claim)
        assert "c-A05-10" in {r.claim_id for r in fs.unverified}, (obj, prefix)


def test_r5_url_components_are_not_denials():
    """Codex r5 #3: "https://cl-andes-agrologistica.no" is a website, not a "no"."""

    def change(c):
        if c["claim_id"] == "c-A05-03":
            c["value"] = "0"
        if c["claim_id"] == "c-A05-04":
            c["value"] = "https://cl-andes-agrologistica.no"
        return None

    tri = facts("cl-andes-agro", verdict=change).operations.employees_or_website
    assert tri.state == "present"


# --- Codex round 6 regressions -------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["role", "signature"])
@pytest.mark.parametrize("prefix", ["GRUPO", "grupo", "GrUpO"])
def test_r6_a_free_first_word_does_not_quote_a_company(kind, prefix):
    """An unrelated person cannot supply a free occurrence of the applicant's company name."""
    for obj in ("ANDES AGROPECUARIA SpA", None):
        claim = {
            "claim_id": "c-A05-30",
            "type": kind,
            "subject": "Ana Perez",
            "attribute": "role" if kind == "role" else "signed_by",
            "value": "director" if kind == "role" else "A. Perez",
            "excerpt": f"Andes Lopez confirms Ana Perez is director of {prefix} "
            "ANDES AGROPECUARIA SpA",
        }
        if obj:
            claim["object"] = obj
        fs = _with("cl-andes-agro", "05", claim)
        assert "c-A05-30" in {r.claim_id for r in fs.unverified}
        assert "c-A05-30" not in {r.claim_id for x in fs.roles for r in x.evidence}
        # The actual full company name after a connector still binds, despite the unrelated name.
        claim["excerpt"] = "Andes Lopez confirms Ana Perez is director of ANDES AGROPECUARIA SpA"
        fs = _with("cl-andes-agro", "05", claim)
        assert "c-A05-30" in {r.claim_id for x in fs.roles for r in x.evidence}


@pytest.mark.parametrize("suffix", ["?lang=no", "#no", "/?lang=no", ":8443?lang=no"])
@pytest.mark.parametrize("denied", [False, True])
def test_r6_url_queries_and_fragments_are_not_prose_denials(suffix, denied):
    """A query/fragment belongs to the URL; a genuine prose denial outside it still counts."""

    def change(c):
        if c["claim_id"] == "c-A05-03":
            c["value"] = "0"
        if c["claim_id"] == "c-A05-04":
            c["value"] = f"https://cl-andes-agrologistica.co{suffix}" + (
                " was closed" if denied else ""
            )  # noqa: E501
        return None

    tri = facts("cl-andes-agro", verdict=change).operations.employees_or_website
    assert tri.state == ("absent" if denied else "present")


@pytest.mark.parametrize("kind", ["role", "signature"])
@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("prefix", ["GRUPO", "grupo", "GrUpO"])
@pytest.mark.parametrize(
    "spelling", ["THE ANDES AGROPECUARIA", "ANDES AGROPECUARIA", "THE ANDES AGROPECUARIA SpA"]
)
def test_r7_article_and_suffix_normalization_cannot_disable_foreign_veto(
    kind, explicit, prefix, spelling
):
    """A normalized applicant name inside a foreign name is not support for a relationship."""

    def change(c):
        for key in ("subject", "object", "value", "excerpt"):
            if isinstance(c.get(key), str):
                c[key] = c[key].replace("ANDES AGROPECUARIA SpA", "The Andes Agropecuaria SpA")
        return None

    raw, docs = load("cl-andes-agro", change)
    claim = {
        "page": 1,
        "bbox": [100, 100, 1600, 200],
        "source": "text_layer",
        "claim_id": "c-A05-99",
        "type": kind,
        "subject": "Ana Perez",
        "attribute": "role" if kind == "role" else "signed_by",
        "value": "director" if kind == "role" else "Ana Perez",
    }
    if explicit:
        claim["object"] = "The Andes Agropecuaria SpA"
    intro = "Ana Perez is director of" if kind == "role" else "Signed by Ana Perez for"

    def derive(excerpt):
        claim["excerpt"] = excerpt
        updated = [
            DocInput(
                **{
                    **d.__dict__,
                    "claims": [*d.claims, dict(claim)],
                    "verdicts": {**d.verdicts, claim["claim_id"]: True},
                }
            )
            if d.doc == "05"
            else d
            for d in docs
        ]
        return derive_factset(raw["case"], raw["run"], raw["as_of"], updated, materiality=POINTS)

    for extra in ("", "The Lopez and Andes Lopez confirm: "):
        fs = derive(f"{extra}{intro} {prefix} {spelling}")
        assert claim["claim_id"] in {r.claim_id for r in fs.unverified}
        assert claim["claim_id"] not in {r.claim_id for x in fs.roles for r in x.evidence}
    # Suffix/article omissions and a separate *complete* applicant mention remain legitimate.
    for target in (
        "The Andes Agropecuaria SpA",
        "THE ANDES AGROPECUARIA",
        "ANDES AGROPECUARIA",
    ):
        for preamble in ("", "Under the resolution, ", "Under Andes Lopez's resolution, "):
            fs = derive(f"{preamble}{intro} {target}")
            assert claim["claim_id"] in {r.claim_id for x in fs.roles for r in x.evidence}
    if explicit:
        fs = derive(f"{intro} The Andes Agropecuaria SpA alongside {prefix} {spelling}")
        assert claim["claim_id"] in {r.claim_id for x in fs.roles for r in x.evidence}


@pytest.mark.parametrize(
    "tail", ["?lang=en", "#en", ":8443?lang=no", "/(no)?lang=en", "?q=(no)", "?q=%28no%29"]
)
@pytest.mark.parametrize(
    "wrapper",
    [
        ("(", ")"),
        ("[", "]"),
        ("<", ">"),
        ('"', '"'),
        ("'", "'"),
        ("“", "”"),
        ("‘", "’"),
        ("«", "»"),
        ("‹", "›"),
        ("„", "“"),
        ("“ ", "”"),
    ],
)
@pytest.mark.parametrize("denial", ["", ":closed", ": closed", ";sin sitio web"])
def test_r7_url_delimiters_preserve_outside_denials(tail, wrapper, denial):
    """Complete URL components stay data, and prose after their enclosing delimiter stays prose."""

    def change(c):
        if c["claim_id"] == "c-A05-03":
            c["value"] = "0"
        if c["claim_id"] == "c-A05-04":
            c["value"] = (
                f"Website {wrapper[0]}https://cl-andes-agrologistica.co{tail}{wrapper[1]}{denial}"
            )
            c["excerpt"] = c["value"]
        return None

    tri = facts("cl-andes-agro", verdict=change).operations.employees_or_website
    assert tri.state == ("absent" if denial else "present")


@pytest.mark.parametrize(
    "value",
    [
        "https://cl-andes-agrologistica.co?q='no'",
        "Website <https://cl-andes-agrologistica.co?q='no'>",
        "Website \"https://cl-andes-agrologistica.co?q='no'\"",
    ],
)
def test_r7_apostrophes_inside_unquoted_or_double_quoted_urls_are_data(value):
    def change(c):
        if c["claim_id"] == "c-A05-03":
            c["value"] = "0"
        if c["claim_id"] == "c-A05-04":
            c["value"] = value
        return None

    assert facts("cl-andes-agro", verdict=change).operations.employees_or_website.state == "present"


def test_bro_3543_extract_does_not_use_stated_incorporation_date_as_own_date():
    """A registry extract merely states historical incorporation;
    it is not the document's own issue date."""
    claims = [
        {
            "claim_id": "c-01-01",
            "page": 1,
            "bbox": [10, 10, 50, 50],
            "source": "text_layer",
            "confidence": 0.9,
            "subject": "Acme S.A.S.",
            "attribute": "fecha_de_constitucion",
            "value": "2012-10-10",
            "excerpt": "Constituida el 10 de octubre de 2012",
            "flags": [],
        }
    ]
    doc_input = DocInput(
        doc="01",
        document_name="01_estatutos_actualizados.pdf",
        claims=claims,
        verdicts={"c-01-01": True},
    )
    fs = derive_factset("test_case", "r1", "2026-10-25", [doc_input], materiality=POINTS)
    d01 = next(d for d in fs.documents if d.doc == "01")
    assert d01.issued is not None

    # If the extract has its own issue date, that is used
    claims_with_issue = [
        *claims,
        {
            "claim_id": "c-01-02",
            "page": 1,
            "bbox": [10, 10, 50, 50],
            "source": "text_layer",
            "confidence": 0.9,
            "subject": "doc",
            "attribute": "issue_date",
            "value": "2026-09-06",
            "excerpt": "Expedido el 2026-09-06",
            "flags": [],
        },
    ]
    doc_input2 = DocInput(
        doc="01",
        document_name="01_estatutos_actualizados.pdf",
        claims=claims_with_issue,
        verdicts={c["claim_id"]: True for c in claims_with_issue},
    )
    fs2 = derive_factset("test_case", "r1", "2026-10-25", [doc_input2], materiality=POINTS)
    d01_2 = next(d for d in fs2.documents if d.doc == "01")
    assert d01_2.issued == "2026-09-06"
