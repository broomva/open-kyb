"""Read comparable officer corrections. L5 owns attributed writes and signatures."""

from __future__ import annotations

from pathlib import Path

from kyb.contracts.b4 import CorrectionRecord, comparable_key
from kyb.run.records import read_record
from kyb.workspace.catalog import Catalog


def comparable(root: Path, jurisdiction: str, fired_rules: list[str]) -> list[CorrectionRecord]:
    key = comparable_key(jurisdiction, fired_rules)
    out = []
    for rel in sorted(Catalog(root).entries()):
        if not rel.startswith("ledger/corrections/") or not rel.endswith(".json"):
            continue
        record = CorrectionRecord.model_validate(read_record(root, rel))
        if comparable_key(record.jurisdiction, record.fired_rules) == key:
            out.append(record)
    ids = [r.correction_id for r in out]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate correction ids")
    return sorted(out, key=lambda r: r.correction_id)
