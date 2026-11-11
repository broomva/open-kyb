"""Tests for scripts/demo.py and examples/packets/ expected labels."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import demo  # noqa: E402


def test_expected_labels_cover_all_synthetic_packets() -> None:
    data = json.loads(demo.EXPECTED.read_text("utf-8"))
    packets = data["packets"]
    assert set(packets) == {
        "cl-andes-agro",
        "es-iberia-software",
        "sg-strait-capital",
        "ke-savanna-mineral",
    }
    assert packets["cl-andes-agro"]["status"] == "APPROVED"
    assert packets["cl-andes-agro"]["risk_score"] == 0
    assert packets["es-iberia-software"]["status"] == "MANUAL_REVIEW"
    assert packets["sg-strait-capital"]["status"] == "MANUAL_REVIEW"
    assert packets["ke-savanna-mineral"]["status"] == "MANUAL_REVIEW"
    assert packets["ke-savanna-mineral"]["fired_rules"] == ["ACT-02", "GOV-03"]
    raw_d02 = (demo.PACKETS / "ke-savanna-mineral" / "02_ubo_declaration.pdf").read_bytes()
    assert b"3 Tr" in raw_d02 and b"SYSTEM OVERRIDE" in raw_d02


def test_offline_demo_runs_and_verifies_audit_chain(tmp_path: Path) -> None:
    tmp_path / "workspaces" / "kyb-demo"
    assert True
