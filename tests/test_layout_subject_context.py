"""Tests for claim-blind layout instructions, cache-identity, and legacy adapter=None collision."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

import pytest

from kyb.audit.chain import verify_workspace, workspace_action
from kyb.docintel.ingest import ingest_packet
from kyb.docintel.pages import build_pages
from kyb.facts.derive import DocInput, derive_factset, materiality
from kyb.read.plan import record_layout
from kyb.read.schema import LayoutRecord
from kyb.read.submit import SubmissionRejected, submit_claims
from kyb.util.hashing import sha256_json
from kyb.witness import gemini
from kyb.witness.bind import Scope, binding
from kyb.witness.port import Response, Witness
from kyb.witness.verify import _clip, _verify_document

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "examples"
OLD_LAYOUT_PROMPT = "Legacy layout prompt v0 (pre-context-repair)"
CASE = "context_fixture"
COMPANY = "IBERIA SOFTWARE S.L."
DEED = "REGISTRO DE SOCIEDADES DE MADRID"


class MockTransportAdapter:
    """Mock witness adapter that implements the transport and route identity protocols."""

    transport = "mock-transport-gcp"
    family = "google"
    model = "gemini-3.8-flash"
    prompt_version = "witness-strata-v1"
    validation_version = "v1"
    route_identity = {"project": "test-project", "location": "us-central1"}

    def call(self, kind: str, images: list[bytes], spec: dict[str, Any]) -> tuple[dict, dict]:
        return {"mock": "data"}, {"usage": 0}


def test_layout_prompt_identity_separation_and_transcription_preservation(tmp_path):
    """Test modern transport cache hashes LAYOUT_PROMPT such that prompt changes change layout

    cache identity, while transcription cache identity remains preserved when LAYOUT_PROMPT changes.
    Record/replay behavior is exercised separately below, not inferred from key inequality.
    """
    adapter = MockTransportAdapter()
    images = [b"mock_image_bytes"]
    spec = {"kinds": ["date", "capital"], "doc_sha256": "sha256:abc"}

    # Calculate key with new layout prompt
    witness_new = Witness(tmp_path, adapter, prompt_version="witness-strata-v1")
    key_new, _ = witness_new._request("layout", images, spec)

    # Temporary monkeypatch LAYOUT_PROMPT to verify key changes
    original_prompt = gemini.LAYOUT_PROMPT
    try:
        gemini.LAYOUT_PROMPT = OLD_LAYOUT_PROMPT
        key_old, _ = witness_new._request("layout", images, spec)
    finally:
        gemini.LAYOUT_PROMPT = original_prompt

    # Ensure changed layout prompt separates evidence (layout request identity separation)
    assert key_new != key_old, "Layout request keys should be distinct when LAYOUT_PROMPT changes"

    # Now verify transcription identity is preserved (does not depend on LAYOUT_PROMPT)
    spec_transcribe = {"crop_ids": ["k1"]}
    transcribe_key_new, _ = witness_new._request("transcribe", images, spec_transcribe)

    try:
        gemini.LAYOUT_PROMPT = OLD_LAYOUT_PROMPT
        transcribe_key_old, _ = witness_new._request("transcribe", images, spec_transcribe)
    finally:
        gemini.LAYOUT_PROMPT = original_prompt

    assert transcribe_key_new == transcribe_key_old, (
        "Transcription keys should remain identical regardless of LAYOUT_PROMPT changes"
    )


def test_legacy_adapter_none_collision_characterization(tmp_path):
    """Test and characterize legacy adapter=None cache key calculation collision blocker.

    When adapter=None (or does not have the transport attribute), the prompt hash is not included,
    so prompt changes are silently ignored by the cache.
    """
    images = [b"mock_image_bytes"]
    spec = {"kinds": ["date", "capital"], "doc_sha256": "sha256:abc"}

    # Calculate layout key under adapter=None
    witness_none = Witness(tmp_path, adapter=None)
    key_new, _ = witness_none._request("layout", images, spec)

    original_prompt = gemini.LAYOUT_PROMPT
    try:
        gemini.LAYOUT_PROMPT = OLD_LAYOUT_PROMPT
        key_old, _ = witness_none._request("layout", images, spec)
    finally:
        gemini.LAYOUT_PROMPT = original_prompt

    # For adapter=None, the keys collide!
    assert key_new == key_old, (
        "LEGACY COLLISION: adapter=None request key does not include prompt hash, "
        "causing prompt changes to collide!"
    )


def test_actual_cache_replay_separates_layout_and_preserves_transcription(tmp_path, monkeypatch):
    """Use the real record/replay port with an explicit zero-network adapter double."""
    from kyb.util.hashing import sha256_bytes
    from kyb.witness.errors import ROUTING_VERSION, WitnessRefused, WitnessUnavailable

    class RecordedAdapter(MockTransportAdapter):
        transport = "vertex-gcp"
        validation_version = ROUTING_VERSION

        def __init__(self):
            self.calls = 0

        def call(self, kind, images, spec):
            self.calls += 1
            data = {"pages": [], "material_facts": []} if kind == "layout" else {"crops": []}
            self.last_provenance = {
                "transport": self.transport,
                "response_hash": sha256_json({"offline_fixture": data}),
                "route": self.route_identity,
                "family": self.family,
                "validation_version": self.validation_version,
                "reported_model": self.model,
                "tools": [],
                "no_data_tools": True,
                "classification": "complete",
                "normal_terminal": True,
                "refusal_evidence": [],
                "http_status": 200,
                "schema_valid": True,
                "schema_text_hash": sha256_bytes(json.dumps(data).encode()),
                "schema_hash": sha256_json(data),
                "data_hash": sha256_json(data),
                "final_output_hash": sha256_json(data),
                "prompt_block_reason": None,
                "prompt_safety": [],
                "finish_diagnostics": [],
                "error_diagnostics": [],
                "terminal_evidence": [{"field": "finishReason", "value": "STOP"}],
                "candidates": [{"finish_reason": "STOP", "safety": []}],
                "error_status": None,
            }
            return data, {}

    adapter = RecordedAdapter()
    images, layout_spec, transcribe_spec = (
        [b"offline-pixels"],
        {"kinds": [], "doc_sha256": "sha256:fixture"},
        {"crop_ids": []},
    )
    with monkeypatch.context() as patch:
        patch.setattr(gemini, "LAYOUT_PROMPT", OLD_LAYOUT_PROMPT)
        old = Witness(tmp_path, adapter)
        first_layout = old.ask("layout", images, layout_spec)
        first_transcription = old.ask("transcribe", images, transcribe_spec)
        _, old_folder = old._request("layout", images, layout_spec)
        old_bytes = (old_folder / "0.json").read_bytes()
    replay = Witness(tmp_path, adapter, mode="replay")
    with pytest.raises(WitnessUnavailable, match="no recorded witness response"):
        replay.ask("layout", images, layout_spec)
    assert adapter.calls == 2
    transcription = replay.ask("transcribe", images, transcribe_spec)
    assert transcription.replayed and transcription.request_hash == first_transcription.request_hash
    assert adapter.calls == 2
    # Moving an old record to the new key cannot manufacture current-prompt evidence.
    _, new_folder = replay._request("layout", images, layout_spec)
    new_folder.mkdir(parents=True)
    (new_folder / "0.json").write_bytes(old_bytes)
    with pytest.raises(WitnessRefused, match="different request"):
        replay.ask("layout", images, layout_spec)
    with monkeypatch.context() as patch:
        patch.setattr(gemini, "LAYOUT_PROMPT", OLD_LAYOUT_PROMPT)
        again = Witness(tmp_path, adapter, mode="replay").ask("layout", images, layout_spec)
        assert again.replayed and again.request_hash == first_layout.request_hash
    assert adapter.calls == 2 and (old_folder / "0.json").read_bytes() == old_bytes


@pytest.mark.skipif(shutil.which("pdftotext") is None, reason="poppler missing")
def test_layout_subject_context_fixture_and_adverse_matrix(tmp_path, monkeypatch):
    """Reuse fresh accurate fixtures for real submit/internalverify/derive.

    Covers selected wrong-subject, date-owner, money-role, source-identity, off-block,
    ellipsis and >600 controls; not the full rival-party/footnote matrix. All model
    observations and layout regions below are explicit fixture doubles.
    """
    from kyb.docintel import poppler
    from kyb.workspace.init import init_workspace
    from kyb.workspace.runs import start_run

    # 1. Setup workspace and ingest/pages
    packet_dir = tmp_path / "packet"
    packet_dir.mkdir()
    pdf_name = "01_extracto_registro.pdf"
    shutil.copy2(ASSETS / "packets/es-iberia-software" / pdf_name, packet_dir / pdf_name)

    ws = tmp_path / "workspace"
    init_workspace(ws, tenant="fixture", policy_src=ASSETS / "policy/compliance_rules.md")

    # Double: Explicitly labeled semantic layout regions as a LayoutRecord fixture double
    # Double: Fixed image-hash witness oracle as a transcription double
    # Double: Reader input claims hand-authored (unattested) as reader double
    # Double: Attestation as a mock offline attestation double

    with workspace_action(ws):
        manifest = ingest_packet(ws, packet_dir, as_of="2026-10-25", case=CASE)
        run = start_run(ws, CASE, as_of="2026-10-25", reason="offline semantic-region test")
        build_pages(ws, CASE, run)

    page_path = ws / f"silver/pages/{CASE}/{run}/01/p1.json"
    page = json.loads(page_path.read_text())
    text_layer = page["text_layer"]
    pdf_path = ws / f"bronze/packets/{CASE}/{pdf_name}"
    sha = manifest["files"][0]["sha256"]
    page_size = [2550, 3300]

    def get_span(val):
        m = re.search(r"\s+".join(re.escape(t) for t in val.split()), text_layer)
        assert m, f"Value '{val}' not found in page text_layer"
        return m.span()

    def get_quote(subj, val):
        a, b = get_span(subj), get_span(val)
        return text_layer[min(a[0], b[0]) : max(a[1], b[1])]

    # 4 semantic regions as a layout fixture double
    regions = {"header": [235, 452, 2162, 604], "entity": [235, 637, 2295, 1307]}
    subjects = [
        (COMPANY, "legal_name", COMPANY, "entity", "legal_name"),
        (DEED, "signing_date", "2020-04-12", "header", "date"),
        (
            COMPANY,
            "share_capital",
            {"amount": 50000, "currency": "EUR"},
            "entity",
            "capital",
        ),
        (
            COMPANY,
            "corporate_object",
            "Desarrollo de aplicaciones",
            "entity",
            "attribute",
        ),
    ]

    claims = []
    for i, (subj, attr, val, region, typ) in enumerate(subjects, 1):
        raw_val = "EUR 50.000" if isinstance(val, dict) else val
        claims.append(
            {
                "claim_id": f"c-CTX-{i:02d}",
                "type": typ,
                "subject": subj,
                "attribute": attr,
                "value": val,
                "excerpt": get_quote(subj, raw_val),
                "page": 1,
                "bbox": regions[region],
                "block_id": region,
                "source": "text_layer",
                "confidence": 1.0,
            }
        )

    facts = [
        {
            "block_id": c["block_id"],
            "kind": kind,
            "subject": c["subject"],
            "attribute": c["attribute"],
            "value": val,
        }
        for c, kind, val in zip(
            claims,
            ["identifier", "date", "capital", "corporate_activity"],
            [COMPANY, "2020-04-12", "EUR 50.000", "Desarrollo de aplicaciones"],
            strict=True,
        )
    ]

    layout = LayoutRecord.model_validate(
        {
            "document_id": f"{CASE}/{pdf_name}",
            "doc_sha256": sha,
            "model": "offline-fixture-layout-double",
            "pages": [
                {
                    "page": 1,
                    "page_hash": page["render"]["sha256"],
                    "size_px": page_size,
                    "blocks": [
                        {"block_id": k, "kind": "semantic-region", "bbox": v}
                        for k, v in regions.items()
                    ],
                }
            ],
            "material_facts": facts,
        }
    )

    with workspace_action(ws):
        record_layout(
            ws,
            CASE,
            run,
            "01",
            layout,
            produced_by={
                "action": "offline-fixture-layout-double",
                "fixture": True,
            },
        )

    # Prepare oracle transcription double responses
    header_transcription = get_quote(DEED, "2020-04-12")
    entity_transcription = get_quote(COMPANY, "EUR 50.000")

    header_response = {
        "transcription": header_transcription,
        "readable": True,
        "reread": {
            "values": [{"subject": DEED, "attribute": "signing_date", "value": "2020-04-12"}]
        },
    }
    entity_response = {
        "transcription": entity_transcription,
        "readable": True,
        "reread": {
            "values": [
                {"subject": COMPANY, "attribute": "legal_name", "value": COMPANY},
                {"subject": COMPANY, "attribute": "capital", "value": "EUR 50.000"},
                {
                    "subject": COMPANY,
                    "attribute": "corporate_object",
                    "value": "Desarrollo de aplicaciones",
                },
            ]
        },
    }

    oracle_cache = {}
    for r_id, r_data in [("header", header_response), ("entity", entity_response)]:
        cropped_img = poppler.render_region(
            pdf_path, 1, _clip(regions[r_id], page_size), tmp_path / f"{r_id}.png"
        ).read_bytes()
        oracle_cache[hashlib.sha256(cropped_img).hexdigest()] = r_data

    class TestFixtureWitnessDouble:
        """Explicitly labeled mock offline Witness double."""

        def __init__(self):
            self.calls = 0

        def ask(self, kind: str, images: list[bytes], spec: dict[str, Any]) -> Response:
            assert kind == "transcribe"
            self.calls += 1
            rows = []
            for cid, img in zip(spec["crop_ids"], images, strict=True):
                img_hash = hashlib.sha256(img).hexdigest()
                rows.append({"crop_id": cid, **copy.deepcopy(oracle_cache[img_hash])})
            return Response(
                data={"crops": rows},
                usage={},
                model="offline-fixture-witness-double",
                request_hash=sha256_json({"fixture": "context-regions-test"}),
                replayed=False,
                provenance={"transport": "offline-fixture-double", "cost_usd": None},
            )

    payload = {
        "case": CASE,
        "run": run,
        "doc": "01",
        "doc_sha256": sha,
        "claims": claims,
    }

    # --- Adverse matrix & negative controls ---

    # 1. Wrong-subject structured rejection (structured negative admission)
    # The submission validator should reject wrong-subject claims without making any writes.
    bad_payload = copy.deepcopy(payload)
    bad_payload["claims"][0]["subject"] = "Different Company Ltd"

    # Store directory snapshot before bad submission
    def get_files_hash(dir_path: Path):
        return {
            str(p.relative_to(dir_path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in dir_path.rglob("*")
            if p.is_file() and not p.name.endswith(".lock")
        }

    before_files = get_files_hash(ws)
    with pytest.raises(SubmissionRejected) as exc_info, workspace_action(ws):
        submit_claims(ws, bad_payload)

    assert any(
        "subject" in str(err) or "excerpt_missing" in str(err) for err in exc_info.value.errors
    )
    # Proves no nonlock bytes are modified on failed structured admission
    assert before_files == get_files_hash(ws), "Failed claims submission modified workspace files!"

    # Wrong work-order identities must refuse without changing any non-lock bytes.
    for field, value, failure in [
        ("case", "missing-case", FileNotFoundError),
        ("run", "missing-run", SubmissionRejected),
        ("doc", "missing-doc", SubmissionRejected),
        ("doc_sha256", "sha256:" + "0" * 64, SubmissionRejected),
    ]:
        bad_identity = copy.deepcopy(payload)
        bad_identity[field] = value
        before_identity = get_files_hash(ws)
        with pytest.raises(failure), workspace_action(ws):
            submit_claims(ws, bad_identity)
        assert before_identity == get_files_hash(ws), field

    # 2. Rival-company / wrong date-owner (swapped date role)
    # Verify that date attributes with notary as subject but bound to deed context fail binding.
    scope = Scope(
        [],
        layout.material_facts,
        {"header": header_response, "entity": entity_response},
    )
    swapped_date_claim = copy.deepcopy(claims[1])
    swapped_date_claim["subject"] = "Notaria Publica No. 62 de Guadalajara"
    assert not binding(
        swapped_date_claim,
        header_response["reread"],
        header_response["transcription"],
        scope,
    ), "Swapped date-owner/role must refuse binding"

    # 3. Capital as par-value binding control (wrong attribute)
    capital_wrong_attr = copy.deepcopy(claims[2])
    capital_wrong_attr["attribute"] = "par_value"
    assert not binding(
        capital_wrong_attr,
        entity_response["reread"],
        entity_response["transcription"],
        scope,
    ), "Wrong attribute for capital must refuse binding"

    # 4. Off-block crop
    # Crop is outside its layout block bbox.
    off_block_payload = copy.deepcopy(payload)
    off_block_payload["claims"][0]["bbox"] = [10, 10, 100, 100]
    with pytest.raises(SubmissionRejected) as exc_info, workspace_action(ws):
        submit_claims(ws, off_block_payload)
    assert any(
        "crop is not inside its single layout block" in str(err) for err in exc_info.value.errors
    )

    # 5. Ellipsis rejection
    # Claim with ellipsis in excerpt should raise ValidationError or SubmissionRejected.
    with pytest.raises(SubmissionRejected) as exc_info:
        ellipsis_claims = copy.deepcopy(claims)
        ellipsis_claims[0]["excerpt"] = "GRUPO IBERIA... S.A."
        with workspace_action(ws):
            submit_claims(
                ws,
                {
                    "case": CASE,
                    "run": run,
                    "doc": "01",
                    "doc_sha256": sha,
                    "claims": ellipsis_claims,
                },
            )
    assert any(
        "excerpts must be contiguous: no ellipses" in str(err) for err in exc_info.value.errors
    )

    # 6. Over 600 character rejection
    # Claim excerpt exceeding 600 characters should raise ValidationError or SubmissionRejected.
    with pytest.raises(SubmissionRejected) as exc_info:
        long_excerpt_claims = copy.deepcopy(claims)
        long_excerpt_claims[0]["excerpt"] = "A" * 601
        with workspace_action(ws):
            submit_claims(
                ws,
                {
                    "case": CASE,
                    "run": run,
                    "doc": "01",
                    "doc_sha256": sha,
                    "claims": long_excerpt_claims,
                },
            )
    assert any("excerpt" in str(err) or "600" in str(err) for err in exc_info.value.errors)

    # --- Positive path verification ---

    # Submit clean claims and run verification under a single workspace_action context
    with workspace_action(ws):
        admitted = submit_claims(ws, payload)
        verified = _verify_document(ws, CASE, run, "01", TestFixtureWitnessDouble(), pdf_path)

    assert admitted["context"] == "unattested"
    assert admitted["claims"] == 4

    assert all(x["verified"] for x in verified["results"]), "Not all positive claims were verified"
    assert verified["assurance"] != "W2", "Fixture assurance should not be W2"

    # Pure typed FactSet derivation
    docs = [
        DocInput(
            doc="01",
            document_name=pdf_name,
            claims=claims,
            verdicts={x["claim_id"]: x["verified"] for x in verified["results"]},
            assurance=verified["assurance"],
            isolated=False,
            coverage=verified["coverage"],
            findings=[x["kind"] for x in verified["findings"]],
        )
    ]
    derived = derive_factset(
        CASE,
        run,
        "2026-10-25",
        docs,
        materiality=materiality((ws / "policy/compliance_rules.md").read_text()),
    )

    # Assertions on positive outcomes
    assert derived.documents[0].signed_on == "2020-04-12"
    assert "Desarrollo de aplicaciones" in derived.corporate_object
    # D069: retain the verified fixed-capital evidence without creating holdings or a
    # current-total amount. The ontology has no current-total-capital field.
    capital = claims[2]
    assert capital["type"] == "capital" and capital["value"] == {
        "amount": 50000,
        "currency": "EUR",
    }
    capital_evidence = next(e for e in derived.evidence if e.evidence_id == capital["claim_id"])
    assert capital_evidence.verified and "capital" in capital_evidence.excerpt.casefold()
    assert not derived.holdings
    from kyb.copilot.answer import Rejected, _check_fact, _check_missing
    from kyb.copilot.statements import FactStatement, NotInDocuments
    from kyb.copilot.universe import Cite, Fact, Universe

    cite = Cite(capital["claim_id"], pdf_name, 1, tuple(capital["bbox"]), capital["excerpt"])
    fact = Fact(
        capital["claim_id"],
        (COMPANY,),
        "share_capital",
        "50000 EUR",
        note="fixed capital; instrument signed 2020-04-12",
        cites=(cite,),
    )
    universe = Universe(CASE, run, facts={fact.fact_id: fact})
    statement = dict(
        type="fact",
        subject=COMPANY,
        attribute="share_capital",
        value=fact.value,
        cites=[fact.fact_id],
    )
    historical = _check_fact(universe, FactStatement(**statement))
    assert "2020-04-12" in historical and "50000 EUR" in historical
    with pytest.raises(Rejected, match="attribute"):
        _check_fact(universe, FactStatement(**{**statement, "attribute": "current_total_capital"}))
    missing = _check_missing(
        ws, universe, NotInDocuments(type="NOT_IN_DOCUMENTS", topic="capital social actual")
    )
    assert "Not in the documents" in missing

    # Workspace audit chain integrity check
    chain = verify_workspace(ws, at_boundary=True)
    assert chain.ok
    assert chain.events == 7


def test_grouped_rival_headers_do_not_lend_equal_numbers_to_another_party():
    """Production binding/Scope, explicit multi-party observation doubles; no inference."""
    from kyb.read.schema import MaterialFact

    alpha, beta = "Alpha Holdings SA", "Beta Holdings SA"
    facts = [
        MaterialFact(block_id="group", kind="ownership", subject=name, attribute="pct", value="25%")
        for name in (alpha, beta)
    ]
    rows = [{"subject": name, "attribute": "legal_name", "value": name} for name in (alpha, beta)]
    anonymous = {"values": [*rows, {"subject": "%", "attribute": "pct", "value": "25%"}]}
    text = f"{alpha} 25%; {beta} 25%"
    scope = Scope([alpha, beta], facts, {"group": {"reread": anonymous}})
    claim = {
        "subject": alpha,
        "attribute": "pct",
        "type": "owns",
        "value": {"pct": 25},
        "block_id": "group",
    }
    assert not binding(claim, anonymous, text, scope)
    only_beta = {"values": [*rows, {"subject": beta, "attribute": "pct", "value": "25%"}]}
    assert not binding(claim, only_beta, text, scope)
    named = {
        "values": [
            *rows,
            *[{"subject": name, "attribute": "pct", "value": "25%"} for name in (alpha, beta)],
        ]
    }
    assert binding(claim, named, text, scope)
    assert binding({**claim, "subject": beta}, named, text, scope)
    assert not binding({**claim, "subject": "Gamma Holdings SA"}, named, text, scope)


def _context_pdf(path):
    """Fresh ASCII fixture: every row and contradicting pending note is physically printed."""
    lines = [
        "FIXTURE ONLY",
        "Alpha Holdings SA 25%",
        "Beta Holdings SA 25%",
        "Alpha Holdings SA proposed 10% *",
        "* 10% transfer pending; current holding remains 25%.",
    ]
    content = (
        "BT /F1 12 Tf\n"
        + "\n".join(f"1 0 0 1 40 {770 - i * 30} Tm ({line}) Tj" for i, line in enumerate(lines))
        + "\nET"
    )
    stream = content.encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream",
    ]
    data, offsets = bytearray(b"%PDF-1.4\n"), [0]
    for i, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(f"{i} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = len(data)
    data.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        data.extend(f"{offset:010} 00000 n \n".encode())
    data.extend(f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    path.write_bytes(data)


@pytest.mark.skipif(shutil.which("pdftotext") is None, reason="poppler missing")
@pytest.mark.parametrize("context", ["missing", "unreadable", "readable"])
def test_grouped_row_requires_its_contradictory_pending_footnote(tmp_path, context):
    from kyb.workspace.init import init_workspace
    from kyb.workspace.runs import start_run

    packet = tmp_path / "packet"
    packet.mkdir()
    _context_pdf(packet / "01.pdf")
    ws = tmp_path / "workspace"
    init_workspace(ws, tenant="fixture", policy_src=ASSETS / "policy/compliance_rules.md")
    with workspace_action(ws):
        manifest = ingest_packet(ws, packet, as_of="2026-10-25", case=CASE)
        run = start_run(ws, CASE, as_of="2026-10-25", reason="offline grouped-footnote control")
        build_pages(ws, CASE, run)
    page = json.loads((ws / f"silver/pages/{CASE}/{run}/01/p1.json").read_text())
    assert "transfer pending" in page["text_layer"]
    sha = manifest["files"][0]["sha256"]
    group, note = [150, 155, 2400, 485], [150, 545, 2400, 620]
    layout = LayoutRecord.model_validate(
        {
            "document_id": f"{CASE}/01.pdf",
            "doc_sha256": sha,
            "model": "offline-layout-double",
            "pages": [
                {
                    "page": 1,
                    "page_hash": page["render"]["sha256"],
                    "size_px": [2550, 3300],
                    "blocks": [
                        {
                            "block_id": "group",
                            "kind": "paragraph",
                            "bbox": group,
                            "refers_to": ["note"],
                        },
                        {
                            "block_id": "note",
                            "kind": "footnote",
                            "bbox": note,
                            "qualifiers": ["pending"],
                        },
                    ],
                }
            ],
            "material_facts": [
                {
                    "block_id": "group",
                    "kind": "ownership",
                    "subject": "Alpha Holdings SA",
                    "attribute": "pct",
                    "value": "10%",
                }
            ],
        }
    )
    claim = {
        "claim_id": "c-CTX-footnote",
        "type": "owns",
        "subject": "Alpha Holdings SA",
        "attribute": "pct",
        "value": {"pct": 10},
        "page": 1,
        "bbox": group,
        "block_id": "group",
        "source": "pixels",
        "confidence": 1.0,
        "excerpt": "Alpha Holdings SA proposed 10% *",
        "linked_blocks": [] if context == "missing" else ["note"],
    }

    class FootnoteDouble:
        def ask(self, kind, images, spec):
            assert kind == "transcribe" and len(images) == (1 if context == "missing" else 2)
            rows = [
                {
                    "crop_id": "k1",
                    "transcription": "Alpha Holdings SA 25%; Beta Holdings SA 25%; "
                    + claim["excerpt"],
                    "readable": True,
                    "reread": {
                        "values": [
                            {"subject": claim["subject"], "attribute": "pct", "value": "10%"}
                        ]
                    },
                }
            ]
            if context != "missing":
                rows.append(
                    {
                        "crop_id": "k2",
                        "transcription": "* 10% transfer pending; current holding remains 25%.",
                        "readable": context == "readable",
                        "reread": {"values": []},
                    }
                )
            return Response(
                {"crops": rows},
                {},
                "offline-witness-double",
                sha256_json({"footnote": context}),
                False,
            )

    with workspace_action(ws):
        record_layout(
            ws,
            CASE,
            run,
            "01",
            layout,
            produced_by={"action": "offline-layout-double", "fixture": True},
        )
        submit_claims(
            ws, {"case": CASE, "run": run, "doc": "01", "doc_sha256": sha, "claims": [claim]}
        )
        result = _verify_document(
            ws, CASE, run, "01", FootnoteDouble(), ws / f"bronze/packets/{CASE}/01.pdf"
        )
    row = result["results"][0]
    if context == "readable":
        assert row["verified"] and row["context_qualifiers"] == ["pending"]
        # Verified proposed evidence remains qualified; this is not a current-holding assertion.
    else:
        assert not row["verified"] and row["binding"] == "incomplete"
        assert row["missing_context"] == (["note"] if context == "missing" else [])
        assert row["unread_context"] == (["note"] if context == "unreadable" else [])
    assert verify_workspace(ws, at_boundary=True).ok
