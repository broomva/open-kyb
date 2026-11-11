"""`kyb layout` (`layout_pages`): the claim-blind layout pass, before any reading (spec step 4).

The witness sees each document's page images only (no text layer, no claims) and returns blocks
and material facts. Boxes are converted from Gemini's 0–1000 `box_2d` to pixels of the 300-dpi
render. Every request is recorded, so a rerun replays.
"""

from __future__ import annotations

from pathlib import Path

from kyb.audit.chain import AuditLog
from kyb.docintel.poppler import png_size
from kyb.read.plan import _documents, load_layout, pages_rel, record_layout
from kyb.read.schema import LayoutRecord
from kyb.util.fsutil import contained
from kyb.util.hashing import sha256_bytes
from kyb.witness.kinds import kinds
from kyb.witness.port import Witness


class LayoutRejected(ValueError):
    """The witness's layout answer is unusable; nothing is recorded and the step can be retried."""


def to_px(box: list[int], size: tuple[int, int]) -> tuple[int, int, int, int]:
    """[ymin, xmin, ymax, xmax] in 0–1000 → [x0, y0, x1, y1] pixels: on the page, never empty."""
    w, h = size
    ymin, xmin, ymax, xmax = (max(0, min(1000, int(v))) for v in box)
    x0, x1 = sorted((round(xmin * w / 1000), round(xmax * w / 1000)))
    y0, y1 = sorted((round(ymin * h / 1000), round(ymax * h / 1000)))
    x0, y0 = min(x0, w - 1), min(y0, h - 1)
    return x0, y0, min(w, max(x1, x0 + 1)), min(h, max(y1, y0 + 1))


def _page_images(root: Path, case: str, run: str, doc: str, pages: int) -> list[bytes]:
    """Every page of the document, rendered and catalogued for this run, or LookupError."""
    from kyb.workspace.catalog import Catalog

    catalog, out = Catalog(root), []
    for n in range(1, pages + 1):
        rel = f"{pages_rel(case, run, doc)}/p{n}.png"
        if catalog.get(rel) is None:
            raise LookupError(f"{case}/{doc}: page {n} of {run} is not rendered; run pages first")
        out.append(contained(root, rel).read_bytes())
    return out


def _validated(data: dict, n_pages: int) -> tuple[dict[int, list[dict]], list[dict]]:
    """Check the raw answer before it is transformed: every page once, valid boxes, real refs."""
    if not isinstance(data, dict) or not isinstance(data.get("pages"), list):
        raise LayoutRejected("the layout answer has no pages")
    by_page: dict[int, list[dict]] = {}
    for p in data["pages"]:
        n = p.get("page") if isinstance(p, dict) else None
        if not isinstance(n, int) or not 1 <= n <= n_pages or n in by_page:
            raise LayoutRejected(f"page entry {n!r} is missing, out of range or repeated")
        blocks = p.get("blocks")
        if not isinstance(blocks, list):
            raise LayoutRejected(f"page {n} has no block list")
        if not blocks and p.get("blank") is not True:
            raise LayoutRejected(f"page {n} has no blocks and is not marked blank")
        for b in blocks:
            box = b.get("box_2d") if isinstance(b, dict) else None
            if not (
                isinstance(box, list) and len(box) == 4 and all(isinstance(v, int) for v in box)
            ):
                raise LayoutRejected(f"page {n}: a block has an invalid box_2d {box!r}")
            if not b.get("block_id"):
                raise LayoutRejected(f"page {n}: a block has no block_id")
        by_page[n] = blocks
    if len(by_page) != n_pages:
        raise LayoutRejected(f"the answer covers {len(by_page)} of {n_pages} pages")
    ids = [b["block_id"] for bs in by_page.values() for b in bs]
    if len(ids) != len(set(ids)):
        raise LayoutRejected("block ids repeat across pages")
    facts = data.get("material_facts")
    if not isinstance(facts, list):
        raise LayoutRejected("the layout answer has no material_facts list")
    for f in facts:
        if not isinstance(f, dict) or f.get("block_id") not in ids:
            raise LayoutRejected(f"a material fact refers to an unknown block: {f!r}"[:200])
    for bs in by_page.values():
        for b in bs:
            if any(r not in ids for r in b.get("refers_to") or []):
                raise LayoutRejected(f"block {b['block_id']} refers to an unknown block")
    return by_page, facts


def layout_document(root: Path, case: str, run: str, doc: str, f: dict, witness: Witness) -> str:
    images = _page_images(root, case, run, doc, int(f["pages"]))
    resp = witness.ask("layout", images, {"kinds": kinds(), "doc_sha256": f["sha256"]})
    by_page, raw_facts = _validated(resp.data, len(images))
    pages = []
    for n, img in enumerate(images, start=1):
        size = png_size(img)
        blocks = [{
            "block_id": f"{doc}-{b['block_id']}", "kind": b.get("kind") or "other",
            "bbox": to_px(b["box_2d"], size), "heading": b.get("heading"),
            "qualifiers": b.get("qualifiers") or [], "markers": b.get("markers") or [],
            "refers_to": [f"{doc}-{r}" for r in b.get("refers_to") or []],
        } for b in by_page[n]]  # fmt: skip
        pages.append({"page": n, "page_hash": sha256_bytes(img), "size_px": size, "blocks": blocks})
    facts = [{**x, "block_id": f"{doc}-{x['block_id']}"} for x in raw_facts]
    record = LayoutRecord.model_validate({
        "document_id": f"{case}/{doc}", "doc_sha256": f["sha256"], "model": resp.model,
        "pages": pages, "material_facts": facts})  # fmt: skip
    rel = record_layout(root, case, run, doc, record, produced_by={
        "action": "layout_pages", "model": resp.model, "request": resp.request_hash,
        "claim_blind": True})  # fmt: skip
    record_usage(root, "layout", case, run, doc, resp)
    return rel


def record_usage(root: Path, step: str, case: str, run: str, doc: str, resp) -> None:
    """A live call records its tokens; a replay records zero new usage, citing the original."""
    u = resp.usage or {}
    t_in = int(u.get("promptTokenCount", 0))
    t_out = int(u.get("candidatesTokenCount", 0)) + int(u.get("thoughtsTokenCount", 0))
    provenance = getattr(resp, "provenance", {})
    AuditLog(root / "ledger" / "audit.jsonl").append(
        "witness", "model.replay" if resp.replayed else "model.call",
        input_hash=resp.request_hash, output_hash=resp.request_hash,
        tokens_in=0 if resp.replayed else t_in, tokens_out=0 if resp.replayed else t_out,
        detail={"step": step, "case": case, "run": run, "doc": doc, "model": resp.model,
                "replayed": resp.replayed, "original_tokens": [t_in, t_out],
                "transport": provenance.get("transport"),
                "reported_model": provenance.get("reported_model"),
                "reported_cost_usd": provenance.get("cost_usd"),
                "cost_basis": "reported cost or unknown; no inferred flat-rate price"},
    )  # fmt: skip


def layout_case(root: Path, case: str, run: str, witness: Witness) -> list[str]:
    """Lay out every document that has no layout yet. Call under the workspace action lock."""
    done = []
    for doc, f in _documents(root, case):
        if load_layout(root, case, run, doc) is None:
            done.append(layout_document(root, case, run, doc, f, witness))
    return done
