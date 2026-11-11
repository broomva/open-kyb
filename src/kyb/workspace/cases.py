"""Caseload membership only: append-only archive/restore events never alter KYB outcomes."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from kyb.audit.chain import AuditLog, _scan, workspace_action
from kyb.run.records import read_record
from kyb.util.fsutil import safe_id
from kyb.util.hashing import sha256_json


class CaseLifecycle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    state: Literal["active", "archived", "unavailable"] = "active"
    changed_at: str | None = None
    event_hash: str | None = None


def case_states(root: Path) -> dict[str, CaseLifecycle]:
    raw = AuditLog(root / "ledger/audit.jsonl").snapshot()
    status, _ = _scan(raw)
    if not status.ok:
        raise PermissionError("Case archive state is unavailable: the audit chain did not verify.")
    states = {}
    for line in raw.decode().splitlines():
        event = json.loads(line)
        if event["action"] not in {"case.archived", "case.restored"}:
            continue
        try:
            case = safe_id(event["detail"]["case"], "case")
        except (KeyError, ValueError, TypeError) as exc:
            raise PermissionError("A case archive event has invalid case context.") from exc
        states[case] = CaseLifecycle(
            state="archived" if event["action"] == "case.archived" else "active",
            changed_at=event["timestamp"],
            event_hash=event["hash"],
        )
    return states


def set_archived(
    root: Path, case: str, *, archived: bool, expected_event_hash: str | None
) -> CaseLifecycle:
    """The caller owns native assent. Recheck the shown membership under the core lock."""
    case = safe_id(case, "case")
    with workspace_action(root):
        read_record(root, f"bronze/packets/{case}/manifest.json")
        current = case_states(root).get(case, CaseLifecycle())
        target = "archived" if archived else "active"
        if current.state == target:
            return current  # An uncertain delivery can safely read/repeat the same action.
        if current.event_hash != expected_event_hash:
            raise PermissionError("Case archive state changed. Refresh and confirm again in chat.")
        event = AuditLog(root / "ledger/audit.jsonl").append_record(
            "caseload",
            "case.archived" if archived else "case.restored",
            input_hash=sha256_json({"case": case, "previous": expected_event_hash}),
            output_hash=sha256_json({"case": case, "state": target}),
            detail={"case": case, "state": target},
        )
        return CaseLifecycle(state=target, changed_at=event["timestamp"], event_hash=event["hash"])
