"""Read the recorded decisions and the signatures that finalise them (L4 writes the former)."""

from __future__ import annotations

from pathlib import Path

from kyb.contracts.b4 import DECISION_REL, DecisionRecord
from kyb.signing.ledger import read_record
from kyb.util.fsutil import safe_id
from kyb.workspace.catalog import Catalog

SIGNATURE_REL = "ledger/signatures/{decision_id}.json"  # keyed by decision: one signature each
_DECISIONS = "ledger/decisions/"
_SIGNATURES = "ledger/signatures/"


def decision_rel(decision_id: str) -> str:
    return DECISION_REL.format(decision_id=safe_id(decision_id, "decision id"))


def signature_rel(decision_id: str) -> str:
    return SIGNATURE_REL.format(decision_id=safe_id(decision_id, "decision id"))


def decision_paths(root: Path) -> list[str]:
    return sorted(r for r in Catalog(root).entries() if r.startswith(_DECISIONS)
                  and r.endswith(".json"))  # fmt: skip


def load_decision(root: Path, decision_id: str) -> DecisionRecord:
    """A decision as L4 receipted it. Refuses an id that is not recorded, and a record whose
    bytes or content id disagree with the path it sits at."""
    rel = decision_rel(decision_id)
    if Catalog(root).get(rel) is None:
        raise LookupError(f"no decision {decision_id!r} is recorded")
    record = DecisionRecord.model_validate(read_record(root, rel))
    if record.decision_id != decision_id:
        raise PermissionError(f"{rel}: holds decision {record.decision_id!r}")
    return record


def signature(root: Path, decision_id: str) -> dict | None:
    rel = signature_rel(decision_id)
    return read_record(root, rel) if Catalog(root).get(rel) is not None else None


def signed_ids(root: Path) -> set[str]:
    return {Path(r).stem for r in Catalog(root).entries() if r.startswith(_SIGNATURES)
            and r.endswith(".json")}  # fmt: skip
