"""`ingest_packet`: copy a packet into bronze, hash every file, freeze `as_of` (spec).

Nothing is parsed or believed here: bronze is untrusted. Everything is validated before anything
is written, and a rejected ingest leaves existing bytes exactly as they were.
"""

from __future__ import annotations

import json
import re
import tempfile
from datetime import date
from pathlib import Path

from kyb import __version__
from kyb.audit.chain import AuditLog, write_checkpoint
from kyb.docintel import poppler
from kyb.util.fsutil import contained, publish_exclusive, safe_id
from kyb.util.hashing import sha256_bytes, sha256_json
from kyb.workspace.catalog import Catalog

MAX_FILES = 50
MAX_PAGES_PER_DOC = 200
MAX_PAGES_PER_PACKET = 500
# The stem obeys the same rule as document keys (`safe_id`), so anything ingested can be paged.
_PDF_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,80}\.pdf", re.IGNORECASE)


def _source_pdfs(packet_dir: Path) -> list[Path]:
    pdfs = sorted(p for p in packet_dir.iterdir() if p.suffix.lower() == ".pdf")
    if not pdfs:
        raise FileNotFoundError(f"no PDFs in {packet_dir}")
    if len(pdfs) > MAX_FILES:
        raise ValueError(f"{len(pdfs)} files exceed the {MAX_FILES}-file limit")
    for p in pdfs:
        if not _PDF_NAME.fullmatch(p.name) or ".." in p.name or p.is_symlink() or not p.is_file():
            raise ValueError(f"refusing unsafe or non-regular file name {p.name!r}")
    return pdfs


def ingest_packet(root: Path, packet_dir: Path, *, as_of: str, case: str | None = None) -> dict:
    """Call under the workspace action lock."""
    date.fromisoformat(as_of)
    case = safe_id(case or packet_dir.name, "case id")
    dest_rel = f"bronze/packets/{case}"
    dest = contained(root, dest_rel)
    produced_by = {
        "action": "ingest_packet",
        "pin": f"kyb@{__version__}",
        "tool": poppler.version(),
    }

    # 1. Read every source once, inspect that snapshot, and build the complete result in memory.
    staged: list[tuple[str, bytes]] = []
    files = []
    total_pages = 0
    with tempfile.TemporaryDirectory(prefix="kyb-ingest-") as tmp:
        for pdf in _source_pdfs(packet_dir):
            data = pdf.read_bytes()
            snap = Path(tmp) / pdf.name
            snap.write_bytes(data)
            meta = poppler.info(snap)
            pages = int(meta["Pages"])
            if not 0 < pages <= MAX_PAGES_PER_DOC:
                raise ValueError(f"{pdf.name}: {pages} pages outside 1..{MAX_PAGES_PER_DOC}")
            total_pages += pages
            if total_pages > MAX_PAGES_PER_PACKET:
                raise ValueError(f"packet exceeds {MAX_PAGES_PER_PACKET} pages")
            words = sum(
                poppler.word_count(poppler.text_layer(snap, p)) for p in range(1, pages + 1)
            )
            staged.append((pdf.name, data))
            files.append({
                "name": pdf.name,
                "sha256": sha256_bytes(data),
                "pages": pages,
                "text_layer": words > 0,
                "words": words,
                "producer": meta.get("Producer", ""),
                "javascript": meta.get("JavaScript", "no") != "no",
                "encrypted": meta.get("Encrypted", "no") != "no",
            })  # fmt: skip
    manifest = {"packet_id": case, "as_of": as_of, "files": files,
                "trust": "untrusted: nothing here is parsed or believed yet"}  # fmt: skip
    manifest_bytes = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")

    # 2. Refuse before writing anything if any existing record would change.
    manifest_path = contained(root, f"{dest_rel}/manifest.json")
    if manifest_path.exists() and manifest_path.read_bytes() != manifest_bytes:
        raise PermissionError(f"{case}: bronze is immutable and as_of is frozen (spec)")
    for name, data in staged:
        target = contained(root, f"{dest_rel}/{name}")
        if target.exists() and target.read_bytes() != data:
            raise PermissionError(f"{target}: bronze is immutable; different bytes already exist")
    catalog = Catalog(root)
    planned = {
        f"{dest_rel}/{name}": catalog.expected_entry(
            f"{dest_rel}/{name}", content_hash=sha256_bytes(data), tier="untrusted",
            produced_by=produced_by, as_of=as_of,
        )
        for name, data in staged
    }  # fmt: skip
    planned[f"{dest_rel}/manifest.json"] = catalog.expected_entry(
        f"{dest_rel}/manifest.json", content_hash=sha256_bytes(manifest_bytes), tier="untrusted",
        produced_by=produced_by, as_of=as_of,
        inputs=[f"{rel}#{e['content_hash']}" for rel, e in planned.items()],
    )  # fmt: skip
    for rel, entry in planned.items():
        if catalog.conflicts(rel, entry):
            raise PermissionError(f"{rel}: catalog already records different provenance")

    # 3. Publish exclusively, then register and audit.
    dest.mkdir(parents=True, exist_ok=True)
    for name, data in staged:
        publish_exclusive(contained(root, f"{dest_rel}/{name}"), data)
        catalog.register(f"{dest_rel}/{name}", tier="untrusted", produced_by=produced_by,
                         as_of=as_of)  # fmt: skip
    publish_exclusive(manifest_path, manifest_bytes)
    rel_manifest = f"{dest_rel}/manifest.json"
    catalog.register(rel_manifest, tier="untrusted", produced_by=produced_by, as_of=as_of,
                     inputs=[catalog.ref(f"{dest_rel}/{f['name']}") for f in files])  # fmt: skip
    AuditLog(root / "ledger" / "audit.jsonl").append(
        "ingest", "tool.call",
        input_hash=sha256_json([f["sha256"] for f in files]),
        output_hash=sha256_bytes(manifest_bytes),
        detail={"action": "ingest_packet", "case": case, "files": len(files), "as_of": as_of},
    )  # fmt: skip
    write_checkpoint(root)
    return manifest
