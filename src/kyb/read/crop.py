"""`render_crop`: a reader looks closer at a region without a shell (spec §3).

Code renders the region from the verified PDF bytes at a higher resolution and writes it to
`silver/crops/<case>/<run>/<doc>/`, the only place besides its own pages a reader may read.
The box is given in pixels of the 300-dpi render, like every claim's bbox.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from kyb.audit.chain import AuditLog
from kyb.docintel import poppler
from kyb.read.plan import _documents, read_plan
from kyb.read.schema import BBox, inside
from kyb.util.fsutil import contained, publish_exclusive
from kyb.util.hashing import sha256_bytes, sha256_json
from kyb.workspace.catalog import Catalog
from kyb.workspace.runs import run_meta

DPI_MAX = 600


def render_crop(root: Path, case: str, run: str, doc: str, page: int, bbox: BBox,
                dpi: int = DPI_MAX) -> dict:  # fmt: skip
    """Call under the workspace action lock. Returns the crop's workspace path."""
    (entry,) = read_plan(root, case, run, doc)
    pages = {p["page"]: p for p in entry["pages"]}
    if page not in pages:
        raise ValueError(f"{case}/{doc}: no page {page}")
    w, h = pages[page]["size_px"]
    if not inside(tuple(bbox), (0, 0, w, h)) or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        raise ValueError(f"bbox {list(bbox)} is not inside the {w}x{h} page")
    if not 72 <= dpi <= DPI_MAX:
        raise ValueError(f"dpi must be between 72 and {DPI_MAX}")
    name = dict(_documents(root, case))[doc]["name"]
    data = contained(root, f"bronze/packets/{case}/{name}").read_bytes()
    if sha256_bytes(data) != entry["doc_sha256"]:
        raise PermissionError(f"{case}/{name}: bronze bytes do not match the manifest")
    k = dpi / 300
    box = tuple(round(v * k) for v in bbox)
    with tempfile.TemporaryDirectory(prefix="kyb-crop-") as tmp:
        pdf = Path(tmp) / "doc.pdf"
        pdf.write_bytes(data)
        png = poppler.render_region(pdf, page, box, Path(tmp) / "crop.png", dpi=dpi).read_bytes()
    key = sha256_json({"doc": entry["doc_sha256"], "page": page, "bbox": list(bbox), "dpi": dpi})
    rel = f"silver/crops/{case}/{run}/{doc}/{key.split(':', 1)[1][:16]}.png"
    created = publish_exclusive(contained(root, rel), png)
    catalog = Catalog(root)
    if catalog.get(rel) is None:
        src = f"bronze/packets/{case}/{name}"
        catalog.register(rel, tier="untrusted", as_of=run_meta(root, case, run)["as_of"],
                         produced_by={"action": "render_crop", "dpi": dpi, "page": page,
                                      "bbox": list(bbox)},
                         inputs=[catalog.ref(src)])  # fmt: skip
    if created:
        AuditLog(root / "ledger" / "audit.jsonl").append(
            "reader", "tool.call", input_hash=key, output_hash=sha256_bytes(png),
            detail={"action": "render_crop", "case": case, "run": run, "doc": doc,
                    "page": page, "dpi": dpi},
        )  # fmt: skip
    return {"crop": rel, "dpi": dpi, "page": page, "bbox": list(bbox),
            "note": json.dumps({"read_with": "Read", "path": rel})}  # fmt: skip
