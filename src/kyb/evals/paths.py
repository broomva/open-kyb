"""Shared paths for the verification runner."""

from __future__ import annotations

import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ASSETS = REPO / "examples"
POLICY = ASSETS / "policy" / "compliance_rules.md"
WATCHLIST = ASSETS / "policy" / "watchlists.json"
PACKETS = ASSETS / "packets"
SCORECARD = REPO / "artifacts" / "verify" / "scorecard.json"


def datasets() -> Path:
    if env := os.environ.get("KYB_EVALS_DATASETS"):
        return Path(env)
    return REPO / "examples" / "evals"


def load(rel: str) -> dict:
    return json.loads((datasets() / rel).read_text("utf-8"))
