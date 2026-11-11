"""Witness the text-layer spans a question points at (agent-driven-review §5, snippets).

Document content the ontology does not model ("Duración: 99 años") is in no claim, so no crop of
it was ever read. For the lines of the text layer that best match the question, this sends each
line's crop to the blind witness (it sees the crop only, never the line) and records what the
pixels say. A line counts as a snippet only if its exact text occurs in that pixel reading under
norm-v2. A line the pixels read differently ("9 años" under a layer saying "99 años") is not a
snippet and is recorded as an integrity finding: lying in a witnessed block is not enough.

Every witness answer is recorded and replayable. With no witness available the line is simply not
offered: unwitnessed text is never used (spec). Call under `workspace_action`.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from kyb.audit.chain import AuditLog, chain_events
from kyb.copilot import lexicon
from kyb.copilot.universe import SPANS_PREFIX, entity_stems, load_universe
from kyb.docintel import poppler
from kyb.read.plan import _documents, load_layout
from kyb.read.submit import claims_rel
from kyb.util.fsutil import contained, publish_exclusive, safe_id
from kyb.util.hashing import sha256_bytes, sha256_json
from kyb.witness.errors import WitnessRefused
from kyb.witness.layout import record_usage
from kyb.witness.norm import contains, norm
from kyb.witness.verify import _clip
from kyb.workspace.catalog import Catalog
from kyb.workspace.runs import run_meta

TOP = 4  # lines witnessed per question


def spans_rel(case: str, run: str, doc: str, key: str) -> str:
    return f"{SPANS_PREFIX.format(case=case, run=run)}{doc}-{key}.json"


def _attempted(root: Path, case: str, run: str) -> set[tuple[str, int, str]]:
    prefix, out = SPANS_PREFIX.format(case=case, run=run), set()
    for rel in Catalog(root).entries():
        if rel.startswith(prefix):
            rec = json.loads(contained(root, rel).read_text("utf-8"))
            out |= {(rec["doc"], s["page"], norm(s["text"])) for s in rec["spans"]}
    return out


def _candidates(root: Path, case: str, run: str, query: list[str], pdfs: dict[str, Path],
                skip: set[tuple[str, int, str]],
                entity: frozenset[str]) -> list[dict[str, Any]]:  # fmt: skip
    lines = []
    for doc, pdf in pdfs.items():
        layout = load_layout(root, case, run, doc)
        for pl in layout.pages if layout else []:
            lines += [{"doc": doc, "page": pl.page, "text": t, "bbox": b, "size": pl.size_px,
                       "page_hash": pl.page_hash}
                      for t, b in poppler.line_boxes(pdf, pl.page)]  # fmt: skip
    docs = [lexicon.stems(x["text"]) for x in lines]
    scores = lexicon.concept_bm25(query, docs)
    hits = [(s, n) for n, (s, d) in enumerate(zip(scores, docs, strict=True))
            if s > 0 and lexicon.concept_enough(
                *lexicon.concept_matches(query, d, entity))]  # fmt: skip
    hits.sort(key=lambda h: (-h[0], lines[h[1]]["doc"], lines[h[1]]["page"], lines[h[1]]["text"]))
    picked: dict[int, None] = {}
    for _, n in hits[:TOP]:
        picked[n] = None
        nxt = n + 1  # a sentence wraps: what a line asks about is often answered on the next one,
        # even when the matched line was already read (by a claim's crop) and is not read again
        if (nxt < len(lines) and lines[nxt]["doc"] == lines[n]["doc"]
                and lines[nxt]["page"] == lines[n]["page"]):  # fmt: skip
            picked.setdefault(nxt)
    return [
        lines[n]
        for n in picked
        if (lines[n]["doc"], lines[n]["page"], norm(lines[n]["text"])) not in skip
    ]


def _record(root: Path, case: str, run: str, doc: str, resp, spans: list[dict]) -> str:
    key = sha256_json([[s["page"], norm(s["text"])] for s in spans])[7:17]
    rel = spans_rel(case, run, doc, key)
    rec = {"document_id": f"{case}/{doc}", "doc": doc, "model": resp.model,
           "request": resp.request_hash, "comparator": "norm-v2", "saw_text_layer": False,
           "spans": spans,
           "findings": [{"kind": "layer_vs_pixel", "page": s["page"], "layer": s["text"],
                         "pixels": s["pixels"], "detail": "the layer says it; the pixels do not"}
                        for s in spans if not s["matched"] and s["readable"]]}  # fmt: skip
    body = (json.dumps(rec, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    out_hash = sha256_bytes(body)
    publish_exclusive(contained(root, rel), body)
    # published, then audited (once: found by its output hash), then receipted last, so a retry
    # after any failure completes exactly the missing steps
    if not any(e["action"] == "tool.call" and e["detail"].get("action") == "witness_spans"
               and e["output_hash"] == out_hash for e in chain_events(root)):  # fmt: skip
        AuditLog(root / "ledger" / "audit.jsonl").append(
            "copilot", "tool.call", input_hash=sha256_json(sorted(s["text"] for s in spans)),
            output_hash=out_hash,
            detail={"action": "witness_spans", "case": case, "run": run, "doc": doc,
                    "matched": sum(s["matched"] for s in spans),
                    "findings": len(rec["findings"])},
        )  # fmt: skip
    catalog = Catalog(root)
    catalog.register(rel, tier="verified", as_of=run_meta(root, case, run)["as_of"],
                     produced_by={"action": "verify_quotes", "model": resp.model,
                                  "comparator": "norm-v2", "purpose": "snippet"},
                     inputs=[catalog.ref(f"silver/layout/{case}/{run}/{doc}.json")])  # fmt: skip
    return rel


def _doc_witness(root: Path, case: str, run: str, doc: str, witness):
    """The witness for one document's lines. A routed witness reads a document only through the
    route its reader's attestation allows, as `verify_document` does: the cross-family stratum
    when there is one. A plain witness (no routing) is used as it is."""
    if not hasattr(witness, "for_document"):
        return witness
    sub = json.loads(contained(root, claims_rel(case, run, doc)).read_text("utf-8"))
    routes = witness.for_document(sub)
    return routes.get("A") or next(iter(routes.values()))


def witness_spans(root: Path, case: str, run: str, question: str, witness) -> dict[str, Any]:
    """Witness the lines that best match `question` and are not already witnessed. Returns what
    was recorded and what could not be witnessed."""
    case = safe_id(case, "case id")
    query = lexicon.stems(question, query=True)
    if Catalog(root).get(f"bronze/packets/{case}/manifest.json") is None:
        return {"recorded": [], "candidates": 0}  # no packet in this workspace: nothing to read
    known = _attempted(root, case, run)
    u = load_universe(root, case, run)
    for s in u.snippets.values():  # already read from pixels, by a claim's crop
        known.add((s.doc, s.page, norm(s.text)))
    done: list[str] = []
    refused: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="kyb-spans-") as tmp:
        pdfs: dict[str, Path] = {}
        for doc, f in _documents(root, case):
            data = contained(root, f"bronze/packets/{case}/{f['name']}").read_bytes()
            if sha256_bytes(data) != f["sha256"]:
                raise PermissionError(f"{f['name']}: bronze bytes do not match the manifest")
            pdfs[doc] = Path(tmp) / f"{doc}.pdf"
            pdfs[doc].write_bytes(data)
        picked = _candidates(root, case, run, query, pdfs, known, entity_stems(u)) if query else []
        for doc in sorted({x["doc"] for x in picked}):
            mine = [x for x in picked if x["doc"] == doc]
            try:
                port = _doc_witness(root, case, run, doc, witness)
            except (LookupError, WitnessRefused) as exc:  # this document has no attested route
                refused[doc] = str(exc)
                continue
            crops = []
            for i, x in enumerate(mine, 1):
                out = poppler.render_region(pdfs[doc], x["page"], _clip(x["bbox"], x["size"]),
                                            Path(tmp) / f"{doc}-{i}.png")  # fmt: skip
                crops.append(out.read_bytes())
            resp = port.ask(
                "transcribe", crops, {"crop_ids": [f"k{i}" for i in range(1, len(mine) + 1)]}
            )
            record_usage(root, "spans", case, run, doc, resp)  # tokens, or a zero-cost replay
            got = {r.get("crop_id"): r for r in resp.data.get("crops", []) if isinstance(r, dict)}
            spans = []
            for i, x in enumerate(mine, 1):
                r = got.get(f"k{i}", {})
                pixels = (r.get("transcription") or "").strip() if r.get("readable") else ""
                matched = bool(pixels) and contains(pixels, x["text"])
                spans.append({"page": x["page"], "bbox": list(x["bbox"]), "text": x["text"],
                              "page_hash": x["page_hash"], "readable": bool(pixels),
                              "pixels": pixels, "matched": matched})  # fmt: skip
            done.append(_record(root, case, run, doc, resp, spans))
    return {"recorded": done, "candidates": len(picked) if query else 0, "refused": refused}
