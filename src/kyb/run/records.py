"""Immutable, receipted B4 records. Call writes under `workspace_action`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kyb.audit.chain import AuditLog, chain_events
from kyb.util.fsutil import contained, publish_exclusive
from kyb.util.hashing import canonical_json, sha256_bytes, sha256_json
from kyb.workspace.catalog import Catalog


def read_record(root: Path, rel: str) -> dict[str, Any]:
    entry = Catalog(root).get(rel)
    if entry is None:
        raise LookupError(f"{rel}: no catalog receipt")
    raw = contained(root, rel).read_bytes()
    if sha256_bytes(raw) != entry["content_hash"]:
        raise PermissionError(f"{rel}: bytes differ from catalog receipt")
    for ref in entry["inputs"]:
        path, _, want = ref.partition("#")
        if sha256_bytes(contained(root, path).read_bytes()) != want:
            raise PermissionError(f"{rel}: input {path} differs from receipt")
    return json.loads(raw)


def write_record(
    root: Path,
    rel: str,
    record: dict,
    *,
    action: str,
    tier: str,
    as_of: str,
    inputs: list[str] = (),
) -> str:
    body = canonical_json(record) + b"\n"
    out_hash = sha256_bytes(body)
    catalog = Catalog(root)
    if catalog.get(rel):
        read_record(root, rel)
        if sha256_bytes(contained(root, rel).read_bytes()) != out_hash:
            raise PermissionError(f"{rel}: different record already exists")
        return rel
    publish_exclusive(contained(root, rel), body)
    if not any(
        e["action"] == "tool.call"
        and e["detail"].get("action") == action
        and e["output_hash"] == out_hash
        for e in chain_events(root)
    ):
        AuditLog(root / "ledger/audit.jsonl").append(
            "decision-core",
            "tool.call",
            input_hash=sha256_json(sorted(inputs)),
            output_hash=out_hash,
            detail={"action": action, "record": rel},
        )
    catalog.register(
        rel,
        tier=tier,
        as_of=as_of,
        inputs=list(inputs),
        produced_by={"action": action, "contracts": "b4-v1"},
    )
    return rel
