"""Run the open-kyb end-to-end demonstration over examples/packets/.

Offline (default):
  Initialises a workspace from examples/policy/, ingests and renders pages for
  cl-andes-agro, es-iberia-software, and sg-strait-capital, runs the
  deterministic pipeline over the verified packet fixtures, verifies the audit
  chain, and checks every decision against expected_labels.json.

Live (--live):
  Runs the full live agent harness (scripts/dogfood.py) over examples/packets/.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"
POLICY = EXAMPLES / "policy" / "compliance_rules.md"
PACKETS = EXAMPLES / "packets"
EXPECTED = PACKETS / "expected_labels.json"
DEFAULT_WS = REPO / "workspaces" / "kyb-demo"

LETTER_OF = {
    "cl-andes-agro": "A",
    "es-iberia-software": "B",
    "sg-strait-capital": "C",
    "ke-savanna-mineral": "D",
}


def _packet_for_example(letter: str):
    from kyb.evals.cases import Packet, golden_packet

    if letter == "D":
        base = golden_packet("A")
        docs = [
            d.model_copy(update={"issued": "2026-10-08", "stated_validity_days": None})
            if d.doc == "01"
            else (d.model_copy(update={"issued": "2026-08-01"}) if d.doc == "03" else d)
            for d in base.facts.documents
        ]
        ents = [e.model_copy(update={"jurisdiction": ["KE"]}) for e in base.facts.entities]
        from kyb.contracts.b4 import Entity, EvidenceRef, Holding

        ref = EvidenceRef(doc="03", page=1, claim_id="c-A03-01", verified=True)
        ents.extend(
            [
                Entity(
                    entity_id="e-rift",
                    kind="company",
                    names=["Rift Valley Holdings Ltd"],
                    jurisdiction=["KE"],
                    evidence=[ref],
                ),
                Entity(
                    entity_id="e-kili",
                    kind="company",
                    names=["Kilimanjaro Equity Ltd"],
                    jurisdiction=["KE"],
                    evidence=[ref],
                ),
            ]
        )
        holdings = [
            Holding(
                holding_id="h-d1",
                owner="e-andres-delgado-vega",
                owned=base.facts.applicant,
                pct="72",
                shares=72000,
                status="current",
                evidence=[ref],
            ),
            Holding(
                holding_id="h-d2",
                owner="e-rift",
                owned=base.facts.applicant,
                pct="28",
                shares=28000,
                status="current",
                evidence=[ref],
            ),
            Holding(
                holding_id="h-d3",
                owner="e-kili",
                owned="e-rift",
                pct="100",
                shares=100000,
                status="current",
                evidence=[ref],
            ),
            Holding(
                holding_id="h-d4",
                owner="e-eduardo-cifuentes",
                owned="e-kili",
                pct="100",
                shares=100000,
                status="current",
                evidence=[ref],
            ),
        ]
        from kyb.facts.derive import fact_set_hash

        fs = base.facts.model_copy(
            update={
                "case": "D",
                "documents": docs,
                "entities": ents,
                "holdings": holdings,
                "corporate_object": [
                    "Wholesale trading, assaying, and export of precious metals and stones"
                ],
            }
        )
        fs = fs.model_copy(update={"fact_set_hash": fact_set_hash(fs)})
        sr = base.screening.model_copy(update={"case": "D", "fact_set_hash": fs.fact_set_hash})
        return Packet(fs, sr, {})

    pkt = golden_packet(letter)
    if letter == "A":
        docs = [
            d.model_copy(update={"issued": "2026-10-05"})
            if d.doc == "01"
            else (d.model_copy(update={"issued": "2026-06-10"}) if d.doc == "03" else d)
            for d in pkt.facts.documents
        ]
        return Packet(pkt.facts.model_copy(update={"documents": docs}), pkt.screening, pkt.answers)
    return pkt


def run_offline(ws: Path) -> int:
    from kyb.audit.chain import AuditLog, workspace_action
    from kyb.contracts.b4 import DECISION_REL, FACTS_REL, SCREENING_REL
    from kyb.docintel.ingest import ingest_packet
    from kyb.docintel.pages import build_pages
    from kyb.evals import agentpath as ap
    from kyb.signing.ledger import write_record
    from kyb.workspace.init import init_workspace
    from kyb.workspace.runs import run_for_pages

    if ws.exists():
        shutil.rmtree(ws)
    init_workspace(ws, tenant="kyb-demo", policy_src=POLICY)

    expected = json.loads(EXPECTED.read_text("utf-8"))["packets"]

    print("=== open-kyb demo (offline deterministic pipeline over examples/packets/) ===")
    print(f"Workspace: {ws}")
    print(f"Policy:    {POLICY.relative_to(REPO)}")
    print()
    header = f"{'Packet':<24} {'Expected':<18} {'System':<18} {'Fired Rules'}"
    print(header)
    print("-" * len(header))

    failures = 0
    for case, letter in LETTER_OF.items():
        pkt_dir = PACKETS / case
        with workspace_action(ws):
            ingest_packet(ws, pkt_dir, case=case, as_of="2026-10-25")
            run = run_for_pages(ws, case, as_of="2026-10-25")
            build_pages(ws, case, run)
        packet = _packet_for_example(letter)
        rec, _, _ = ap.synthetic_decision({"id": case}, packet, "healthy")
        with workspace_action(ws):
            write_record(
                ws,
                FACTS_REL.format(case=case, run=run),
                packet.facts.model_dump(mode="json"),
                action="derive_facts",
                tier="derived",
                as_of="2026-10-25",
                inputs=[],
                detail={"demo": "example"},
            )
            write_record(
                ws,
                SCREENING_REL.format(case=case, run=run),
                packet.screening.model_dump(mode="json"),
                action="screen",
                tier="derived",
                as_of="2026-10-25",
                inputs=[],
                detail={"demo": "example"},
            )
            write_record(
                ws,
                DECISION_REL.format(case=case, run=run, decision_id=rec.decision_id),
                rec.model_dump(mode="json"),
                action="decide",
                tier="derived",
                as_of="2026-10-25",
                inputs=[],
                detail={"demo": "example"},
            )

        dec = rec.kyb_decision
        fired = sorted(r.rule_id for r in rec.rule_results if r.fired == "yes")
        exp = expected[case]
        exp_str = f"{exp['status']} ({exp['risk_score']})"
        got_str = f"{dec.status} ({dec.risk_score})"
        print(f"{case:<24} {exp_str:<18} {got_str:<18} {', '.join(fired) or '-'}")
        if (dec.status, dec.risk_score, fired) != (
            exp["status"],
            exp["risk_score"],
            sorted(exp["fired_rules"]),
        ):
            failures += 1

    res = AuditLog(ws / "ledger" / "audit.jsonl").verify()
    print()
    print(f"Audit chain verified: {res.ok} ({res.events} hash-chained events)")
    shutil.rmtree(ws.parent, ignore_errors=True)
    if not res.ok:
        failures += 1
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the open-kyb demo on examples/packets/.")
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WS)
    parser.add_argument(
        "--live", action="store_true", help="Run live model harness via scripts/dogfood.py"
    )
    args = parser.parse_args(argv)
    if args.live:
        cmd = [sys.executable, str(REPO / "scripts" / "dogfood.py")]
        return subprocess.run(cmd, check=False).returncode
    return run_offline(args.workspace)


if __name__ == "__main__":
    raise SystemExit(main())
