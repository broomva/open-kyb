"""Fresh synthetic UI sample. Model answers, hook events and decision records are test doubles.

This is not a live agent run or a production accuracy/approval demonstration.
"""

from pathlib import Path

import test_witness as witness_test
from decision_fixtures import decision, record

from kyb.audit.chain import verify_workspace, workspace_action
from kyb.docintel.ingest import ingest_packet
from kyb.docintel.pages import build_pages
from kyb.facts.action import derive_facts
from kyb.read.plan import read_plan
from kyb.signing.officers import enroll
from kyb.signing.sign import sign_decision
from kyb.synth.render import pdf_bytes
from kyb.witness.layout import layout_case
from kyb.witness.port import Witness
from kyb.witness.verify import verify_case
from kyb.workspace.init import init_workspace
from kyb.workspace.runs import start_run

POLICY = Path(__file__).resolve().parents[1] / "examples/policy/compliance_rules.md"
CASE = "DEMO_Nimbus"


def build_sample(base: Path) -> Path:
    root = base / "officer-workspace"
    packet = base / "packet"
    packet.mkdir()
    text = "Nimbus Holdings S.A. 25%"
    (packet / "05.pdf").write_bytes(pdf_bytes([(text, False)], "Synthetic case workspace sample"))
    init_workspace(root, tenant="synthetic-ui-sample", policy_src=POLICY)
    layout = {
        "pages": [
            {
                "page": 1,
                "blocks": [{"block_id": "b3", "kind": "paragraph", "box_2d": [35, 78, 80, 800]}],
            }
        ],
        "material_facts": [
            {
                "block_id": "b3",
                "kind": "ownership",
                "subject": "Nimbus Holdings S.A.",
                "attribute": "pct",
                "value": "25%",
            }
        ],
    }
    with workspace_action(root):
        ingest_packet(root, packet, as_of="2026-10-25", case=CASE)
        start_run(
            root, CASE, as_of="2026-10-25", reason="synthetic UI sample; inference/hook doubles"
        )
        build_pages(root, CASE, "r1")
        layout_case(root, CASE, "r1", Witness(root, witness_test.FakeWitness(layout=layout)))
    plans = {e["doc"]: e for e in read_plan(root, CASE, "r1")}
    payload = {
        "case": CASE,
        "run": "r1",
        "doc": "05",
        "doc_sha256": plans["05"]["doc_sha256"],
        "claims": [
            witness_test._claim(
                "c-demo-1", "Nimbus Holdings S.A.", {"pct": 25}, text, [200, 140, 1400, 230]
            )
        ],
    }
    old_case = witness_test.CASE
    try:
        witness_test.CASE = CASE
        witness_test._attested_reader(root, plans, payload)
    finally:
        witness_test.CASE = old_case
    with workspace_action(root):
        verify_case(
            root,
            CASE,
            "r1",
            Witness(
                root,
                witness_test.FakeWitness(
                    [witness_test.crop(text, "Nimbus Holdings S.A.", "25%")],
                    model="gemini-3.1-pro-preview",
                ),
            ),
        )
        derive_facts(root, CASE, "r1")
        for case in ("DEMO_Cedar", "DEMO_Solana", "DEMO_Harbor"):
            ingest_packet(root, packet, as_of="2026-10-25", case=case)
            if case != "DEMO_Cedar":
                start_run(root, case, as_of="2026-10-25", reason="synthetic decision fixture")
    record(root, decision("DEMO_Solana", status="REJECTED", name="Solana Parts · sample"))
    signed = record(
        root, decision("DEMO_Harbor", status="MANUAL_REVIEW", name="Harbor Supply · sample")
    )
    with workspace_action(root):
        officer = enroll(root, "demo-officer")
        sign_decision(
            root,
            decision_id=signed,
            verdict="assent",
            officer="demo-officer",
            officer_token=officer["token"],
        )
    assert verify_workspace(root).ok
    return root
