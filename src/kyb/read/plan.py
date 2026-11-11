"""Layout records and the read plan (spec steps 4–5).

`record_layout` publishes the claim-blind layout of one document (the witness writes it through
`layout_pages`, before any reading). `read_plan` is the reader's work order: one entry per
document, built only from recorded pages and layouts.
"""

from __future__ import annotations

import json
from pathlib import Path

from kyb import __version__
from kyb.audit.chain import AuditLog
from kyb.docintel.pages import doc_keys, verified_manifest
from kyb.read.field_guide import reader_contract
from kyb.read.schema import MAX_CROPS_PER_DOC, MAX_PAGES_PER_DOC, LayoutRecord
from kyb.util.fsutil import contained, publish_exclusive, safe_id
from kyb.util.hashing import sha256_bytes
from kyb.workspace.catalog import Catalog
from kyb.workspace.runs import run_meta, safe_run_id

SKILL = "kyb-read-packet@0.2.2"


def layout_rel(case: str, run: str, doc: str) -> str:
    return f"silver/layout/{safe_id(case, 'case id')}/{safe_run_id(run)}/{safe_id(doc, 'doc')}.json"


def pages_rel(case: str, run: str, doc: str) -> str:
    return f"silver/pages/{safe_id(case, 'case id')}/{safe_run_id(run)}/{safe_id(doc, 'doc')}"


def _documents(root: Path, case: str) -> list[tuple[str, dict]]:
    files = verified_manifest(root, case)["files"]
    keys = doc_keys([f["name"] for f in files])
    return [(keys[f["name"]], f) for f in files]


def record_layout(root: Path, case: str, run: str, doc: str, layout: LayoutRecord, *,
                  produced_by: dict) -> str:  # fmt: skip
    """Publish a document's layout once. Call under the workspace action lock."""
    docs = dict(_documents(root, case))
    if doc not in docs:
        raise KeyError(f"{case}: no document {doc!r} in the manifest")
    if layout.doc_sha256 != docs[doc]["sha256"]:
        raise PermissionError(f"{case}/{doc}: layout is for different document bytes")
    as_of = run_meta(root, case, run)["as_of"]
    rel = layout_rel(case, run, doc)
    body = (json.dumps(layout.model_dump(), indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    publish_exclusive(contained(root, rel), body)
    catalog = Catalog(root)
    pages = [f"{pages_rel(case, run, doc)}/p{p.page}.png" for p in layout.pages]
    catalog.register(rel, tier="untrusted", produced_by=produced_by, as_of=as_of,
                     inputs=[catalog.ref(p) for p in pages])  # fmt: skip
    AuditLog(root / "ledger" / "audit.jsonl").append(
        "witness", "tool.call", input_hash=layout.doc_sha256, output_hash=sha256_bytes(body),
        detail={"action": "layout_pages", "case": case, "run": run, "doc": doc,
                "material_facts": len(layout.material_facts)},
    )  # fmt: skip
    return rel


def load_layout(root: Path, case: str, run: str, doc: str) -> LayoutRecord | None:
    path = contained(root, layout_rel(case, run, doc))
    if not path.exists():
        return None
    return LayoutRecord.model_validate_json(path.read_bytes())


def read_plan(root: Path, case: str, run: str, doc: str | None = None) -> list[dict]:
    """The reader's work order. Refuses until every document has its pages and its layout."""
    case, run = safe_id(case, "case id"), safe_run_id(run)
    entries = []
    for key, f in _documents(root, case):
        if doc is not None and key != doc:
            continue
        layout = load_layout(root, case, run, key)
        if layout is None:
            raise LookupError(f"{case}/{key}: no layout recorded for {run}; run layout_pages first")
        if len(layout.pages) > MAX_PAGES_PER_DOC:
            raise ValueError(f"{case}/{key}: {len(layout.pages)} pages exceed {MAX_PAGES_PER_DOC}")
        pages = []
        for p in layout.pages:
            meta = json.loads(
                contained(root, f"{pages_rel(case, run, key)}/p{p.page}.json").read_text()
            )
            pages.append({"page": p.page, "png": f"{pages_rel(case, run, key)}/p{p.page}.png",
                          "text_layer": meta["text_layer"] is not None, "size_px": list(p.size_px),
                          "blocks": [{"block_id": b.block_id, "kind": b.kind, "bbox": list(b.bbox),
                                      "heading": b.heading, "qualifiers": b.qualifiers,
                                      "markers": b.markers, "refers_to": b.refers_to}
                                     for b in p.blocks]})  # fmt: skip
        entries.append({
            "case": case, "run": run, "doc": key, "document": f["name"], "doc_sha256": f["sha256"],
            "pages": pages, "layout": layout_rel(case, run, key),
            "skill": f"{SKILL} (kyb@{__version__})",
            "reader_contract": reader_contract(),
            "launch_prompt": f"kyb-reader {case} {run} {key}",
            "limits": {"max_crops": MAX_CROPS_PER_DOC, "max_pages": MAX_PAGES_PER_DOC},
        })  # fmt: skip
    if doc is not None and not entries:
        raise KeyError(f"{case}: no document {doc!r}")
    return entries
