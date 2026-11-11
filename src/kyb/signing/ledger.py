"""Receipted, write-once records for the sign-off paths. Call writes under `workspace_action`.

Same shape as every other B4 record: published exclusively, audited once (found by its output
hash, so a retry never appends twice), then given its catalog receipt last. A retry after any
failure completes exactly the missing steps, and nothing counts until the receipt exists.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kyb.audit.chain import AuditLog, chain_events
from kyb.util.fsutil import contained, publish_exclusive
from kyb.util.hashing import canonical_json, sha256_bytes, sha256_json
from kyb.workspace.catalog import Catalog

COMPONENT = "signing"


def read_record(root: Path, rel: str) -> dict[str, Any]:
    """A receipted record whose bytes and lineage still match its receipt."""
    entry = Catalog(root).get(rel)
    if entry is None:
        raise LookupError(f"{rel}: no catalog receipt")
    raw = contained(root, rel).read_bytes()
    if sha256_bytes(raw) != entry["content_hash"]:
        raise PermissionError(f"{rel}: bytes differ from its catalog receipt")
    for ref in entry["inputs"]:
        path, _, want = ref.partition("#")
        if sha256_bytes(contained(root, path).read_bytes()) != want:
            raise PermissionError(f"{rel}: input {path} differs from its receipt")
    return json.loads(raw)


def unreceipted(root: Path, rel: str) -> dict[str, Any] | None:
    """The record an interrupted write left behind (published, not yet receipted), if any."""
    path = contained(root, rel)
    if Catalog(root).get(rel) is not None or not path.exists():
        return None
    return json.loads(path.read_bytes())


def write_record(root: Path, rel: str, record: dict[str, Any], *, action: str, tier: str,
                 as_of: str, inputs: list[str], detail: dict[str, Any]) -> str:  # fmt: skip
    body = canonical_json(record) + b"\n"
    out_hash = sha256_bytes(body)
    catalog = Catalog(root)
    if catalog.get(rel) is not None:
        read_record(root, rel)
        if sha256_bytes(contained(root, rel).read_bytes()) != out_hash:
            raise PermissionError(f"{rel}: a different record already exists")
        return rel
    publish_exclusive(contained(root, rel), body)
    if not any(e["action"] == "tool.call" and e["detail"].get("action") == action
               and e["output_hash"] == out_hash for e in chain_events(root)):  # fmt: skip
        AuditLog(root / "ledger" / "audit.jsonl").append(
            COMPONENT, "tool.call", input_hash=sha256_json(sorted(inputs)), output_hash=out_hash,
            detail={"action": action, "record": rel, **detail},
        )  # fmt: skip
    catalog.register(rel, tier=tier, as_of=as_of, inputs=inputs,
                     produced_by={"action": action, "contracts": "b4-v1"})  # fmt: skip
    return rel
