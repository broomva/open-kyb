"""`pages` (pipeline step 2): render every page at 300 dpi and capture its text layer, per run
(spec).

A page record says where its text came from: a text layer, or nothing (`null`), in which case the
model reads the pixels in step 3. There is no OCR engine. Pages are rendered from a verified
snapshot of the bronze bytes, never from whatever is currently on disk.
"""

from __future__ import annotations

import json
import re
import tempfile
import time
from pathlib import Path

from kyb import __version__
from kyb.audit.chain import AuditLog, write_checkpoint
from kyb.docintel import poppler
from kyb.util.fsutil import contained, publish_exclusive, safe_id
from kyb.util.hashing import sha256_bytes, sha256_json
from kyb.workspace.catalog import Catalog
from kyb.workspace.runs import run_meta, safe_run_id

DEADLINE_SECONDS = 900  # whole-packet budget; each poppler call also has its own timeout


def doc_keys(names: list[str]) -> dict[str, str]:
    """Stable, unique, safe document keys: the numeric prefix when unique (`05`), else the stem."""
    stems = {n: Path(n).stem for n in names}
    prefixes = {n: s.split("_", 1)[0] for n, s in stems.items()}
    keys = {}
    for n in names:
        p = prefixes[n]
        unique_prefix = p.isdigit() and sum(1 for q in prefixes.values() if q == p) == 1
        keys[n] = safe_id(p if unique_prefix else re.sub(r"[^A-Za-z0-9_.-]", "_", stems[n]), "doc")
    if len(set(keys.values())) != len(keys):
        raise ValueError(f"document keys collide: {sorted(keys.values())}")
    return keys


def verified_manifest(root: Path, case: str) -> dict:
    """The packet manifest, only if its bytes are the ones ingest catalogued."""
    rel = f"bronze/packets/{safe_id(case, 'case id')}/manifest.json"
    data = contained(root, rel).read_bytes()
    entry = Catalog(root).get(rel)
    if entry is None or entry["content_hash"] != sha256_bytes(data):
        raise PermissionError(f"{rel}: manifest does not match its catalog entry")
    return json.loads(data.decode("utf-8"))


def build_pages(root: Path, case: str, run_id: str) -> list[dict]:
    """Call under the workspace action lock."""
    case, run_id = safe_id(case, "case id"), safe_run_id(run_id)
    started = time.monotonic()
    manifest = verified_manifest(root, case)
    as_of = manifest["as_of"]
    meta = run_meta(root, case, run_id)
    if meta.get("case") != case or meta.get("as_of") != as_of:
        raise PermissionError(f"{case} {run_id}: run as_of {meta.get('as_of')} != packet {as_of}")
    catalog = Catalog(root)
    audit = AuditLog(root / "ledger" / "audit.jsonl")
    keys = doc_keys([f["name"] for f in manifest["files"]])
    produced_by = {"action": "pages", "pin": f"kyb@{__version__}", "tool": poppler.version(),
                   "dpi": poppler.DPI, "run": run_id}  # fmt: skip
    records: list[dict] = []
    for f in manifest["files"]:
        src_rel = f"bronze/packets/{case}/{f['name']}"
        data = contained(root, src_rel).read_bytes()
        entry = catalog.get(src_rel)
        actual = sha256_bytes(data)
        if actual != f["sha256"] or entry is None or entry["content_hash"] != actual:
            raise PermissionError(f"{src_rel}: bronze bytes do not match the manifest and catalog")
        out_rel = f"silver/pages/{case}/{run_id}/{keys[f['name']]}"
        written: list[str] = []
        with tempfile.TemporaryDirectory(prefix="kyb-pages-") as tmp:
            snap = Path(tmp) / "src.pdf"
            snap.write_bytes(data)
            for page in range(1, f["pages"] + 1):
                if time.monotonic() - started > DEADLINE_SECONDS:
                    raise TimeoutError(f"{case}: pages exceeded the {DEADLINE_SECONDS}s budget")
                png_bytes = poppler.render(snap, page, Path(tmp) / f"p{page}.png").read_bytes()
                text = poppler.text_layer(snap, page)
                has_layer = poppler.word_count(text) > 0
                record = {
                    "document_id": f"{case}/{f['name']}",
                    "page": page,
                    "text_layer": text if has_layer else None,
                    "words": poppler.word_count(text),
                    "render": {"png": f"p{page}.png", "dpi": poppler.DPI,
                               "sha256": sha256_bytes(png_bytes)},
                    "source_sha256": actual,
                    "reader": "text layer present: the model reads it and the pixels"
                    if has_layer
                    else "no text layer: the model reads the pixels (spec)",
                }  # fmt: skip
                json_bytes = (json.dumps(record, indent=2, ensure_ascii=False) + "\n").encode()
                for name, blob in ((f"p{page}.png", png_bytes), (f"p{page}.json", json_bytes)):
                    rel = f"{out_rel}/{name}"
                    publish_exclusive(contained(root, rel), blob)
                    catalog.register(rel, tier="untrusted", produced_by=produced_by, as_of=as_of,
                                     inputs=[catalog.ref(src_rel)])  # fmt: skip
                    written.append(rel)
                records.append(record)
        audit.append(
            "docintel", "tool.call",
            input_hash=actual,
            output_hash=sha256_json([catalog.get(r)["content_hash"] for r in written]),
            detail={"action": "pages", "case": case, "run": run_id, "doc": f["name"],
                    "pages": f["pages"], "text_layer": f["text_layer"]},
        )  # fmt: skip
    write_checkpoint(root)
    return records
