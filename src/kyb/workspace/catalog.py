"""The workspace catalog: one entry per dataset, with its lineage (spec, spec).

Locally the catalog file is the registry. Hosted, Postgres's `artifacts` table is authoritative
and this file mirrors the API's receipts (spec). The entry shape is the same in both places.

Entries are append-only: registering the same path again is accepted only if the whole entry is
identical. A changed tier, lineage or producer is a new record, never an edit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kyb.util.fsutil import atomic_replace, contained, locked, safe_rel
from kyb.util.hashing import sha256_file

LAYERS = ("bronze", "silver", "gold", "ledger", "kg", "cases")
TIERS = ("untrusted", "claimed", "verified", "corroborated", "derived", "judged", "decided", "none")


class Catalog:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.path = root / "catalog.json"
        self._lock = root / ".catalog.lock"

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        raw = self.path.read_text("utf-8")
        return json.loads(raw) if raw.strip() else {}

    def register(
        self,
        rel_path: str,
        *,
        tier: str,
        produced_by: dict[str, Any],
        as_of: str,
        inputs: list[str] | None = None,
        schema_hash: str | None = None,
    ) -> dict[str, Any]:
        rel = safe_rel(rel_path).as_posix()
        entry = self.expected_entry(
            rel, content_hash=sha256_file(contained(self.root, rel)), tier=tier,
            produced_by=produced_by, as_of=as_of, inputs=inputs, schema_hash=schema_hash,
        )  # fmt: skip
        with locked(self._lock):
            data = self._read()
            existing = data.get(rel)
            if existing is not None:
                if existing != entry:
                    raise PermissionError(
                        f"{rel}: catalog entries are append-only (spec); "
                        "a different tier, lineage, producer or content is a new record"
                    )
                return existing
            data[rel] = entry
            body = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
            atomic_replace(self.path, body.encode("utf-8"))
        return entry

    def expected_entry(
        self,
        rel_path: str,
        *,
        content_hash: str,
        tier: str,
        produced_by: dict[str, Any],
        as_of: str,
        inputs: list[str] | None = None,
        schema_hash: str | None = None,
    ) -> dict[str, Any]:
        """The entry `register` would write, computed without touching the file (for preflight)."""
        rel = safe_rel(rel_path).as_posix()
        layer = rel.split("/", 1)[0]
        if layer not in LAYERS:
            raise ValueError(f"{rel}: not under a known layer {LAYERS}")
        if tier not in TIERS:
            raise ValueError(f"unknown tier {tier!r}")
        return {
            "layer": layer,
            "tier": tier,
            "content_hash": content_hash,
            "schema_hash": schema_hash,
            "inputs": sorted(inputs or []),
            "produced_by": produced_by,
            "as_of": as_of,
        }

    def conflicts(self, rel_path: str, entry: dict[str, Any]) -> bool:
        """True if a different entry is already registered at `rel_path`."""
        existing = self.get(rel_path)
        return existing is not None and existing != entry

    def get(self, rel_path: str) -> dict[str, Any] | None:
        with locked(self._lock, shared=True):
            return self._read().get(rel_path)

    def entries(self) -> dict[str, Any]:
        with locked(self._lock, shared=True):
            return self._read()

    def ref(self, rel_path: str) -> str:
        """A lineage reference: path#content_hash."""
        entry = self.get(rel_path)
        if entry is None:
            raise KeyError(rel_path)
        return f"{rel_path}#{entry['content_hash']}"
