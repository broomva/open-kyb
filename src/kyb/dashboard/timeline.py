"""A bounded projection of the existing chain; never a job scheduler or status authority."""

import json
from pathlib import Path

from kyb.audit.chain import AuditLog, _scan
from kyb.dashboard.contracts import Timeline
from kyb.run.records import read_record
from kyb.signing.decisions import load_decision

LABELS = {
    "case.archived": "Case archived from active queue",
    "case.restored": "Case restored to active queue",
    "ingest_packet": "Packet added",
    "run.started": "Review attempt created",
    "pages": "Document pages prepared",
    "layout_pages": "Page layout recorded",
    "read_document": "Reader submitted claims",
    "verify_quotes": "Claims checked against sources",
    "derive_facts": "Verified facts derived",
    "plan_loop": "Evidence retry request saved",
    "start_loop_repair": "Evidence retry allocated",
    "emit_decision": "Recommendation recorded",
    "sign_decision": "Officer signature recorded",
    "record_answer": "Cited answer recorded",
}


def verified_events(root: Path) -> list[dict]:
    raw = AuditLog(root / "ledger/audit.jsonl").snapshot()
    status, _ = _scan(raw)
    if not status.ok:
        raise PermissionError("The activity chain could not be verified.")
    return [json.loads(line) for line in raw.decode().splitlines()]


def timeline(root: Path, cases: list[str], limit: int = 200) -> Timeline:
    # Verify exactly the bytes we project using the core scanner, not a separately read head.
    try:
        events = verified_events(root)
    except PermissionError:
        return Timeline(
            chain_verified=False,
            events=[],
            total=0,
            truncated=False,
            error="Activity is unavailable because the audit chain did not verify.",
        )
    rows = []
    for event in events:
        detail = event["detail"]
        action = detail.get("action") if event["action"] == "tool.call" else event["action"]
        case, run, reference = detail.get("case"), detail.get("run"), detail.get("record")
        if not case and isinstance(reference, str) and reference.startswith("ledger/loops/"):
            case = reference.split("/")[2]
            try:
                record = read_record(root, reference)
                if record.get("case") != case:
                    continue
                run = record.get("run") or record.get("source_run")
            except LookupError, ValueError, PermissionError, OSError:
                continue
        if (
            not case
            and isinstance(reference, str)
            and reference.startswith(("ledger/decisions/", "ledger/signatures/", "ledger/qa/"))
        ):
            try:
                record = read_record(root, reference)
                if "case" in record:
                    case, run = record["case"], record.get("run")
                elif "decision_id" in record:
                    decision = load_decision(root, record["decision_id"])
                    case, run = decision.case, decision.run
            except LookupError, ValueError, PermissionError, OSError:
                continue
        if case not in cases or action not in LABELS:
            continue
        rows.append(
            {
                "seq": event["seq"],
                "at": event["timestamp"],
                "action": action,
                "label": LABELS[action],
                "case": case,
                "run": run,
                "reference": reference,
            }
        )
    return Timeline(
        chain_verified=True,
        events=list(reversed(rows[-limit:])),
        total=len(rows),
        truncated=len(rows) > limit,
    )
