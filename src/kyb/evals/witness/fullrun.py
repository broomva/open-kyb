"""Full-run re-verification: a recorded run's claims and layouts, re-witnessed by one backend.

A vendored run (`datasets/witness/runs/<run>/silver/...`: the reader's submitted claims, the
claim-blind layout, the page text layers and the product's own witness records) is rebuilt in a
scratch workspace and every document goes through the product's `verify_document` with the
backend's recorder as the witness. The crops are cut by the product from the packet PDF, so the
inputs are held fixed and only the witness differs.

For the pinned backend the run's own recorded answers are the seed. Historical verdicts are
compared with the current comparator, retaining exact expectations for its named improvements;
unchanged comparator versions must reproduce their verdicts exactly.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from kyb.evals.paths import PACKETS, datasets


def runs() -> list[Path]:
    root = datasets() / "witness" / "runs"
    return sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []


def _case_run(run_dir: Path) -> tuple[str, str]:
    claims = run_dir / "silver" / "claims"
    case = next(p.name for p in claims.iterdir() if p.is_dir())
    run = next(p.name for p in (claims / case).iterdir() if p.is_dir())
    return case, run


def recorded_verdicts(run_dir: Path) -> dict[str, bool]:
    """The product's own per-claim verdicts, as it recorded them."""
    case, run = _case_run(run_dir)
    out = {}
    for f in sorted((run_dir / "silver" / "witness" / case / run).glob("*.json")):
        for r in json.loads(f.read_text("utf-8"))["results"]:
            out[r["claim_id"]] = bool(r["verified"])
    return out


# Reviewed L1 comparator transition: explicit claim IDs, never inferred from replay output.
_BIND_V8_EXTRA_GAINS = {
    "20261027T052502Z-claude-es-iberia-software": {"c-B01-01", "c-B01-03"},
    "20261027T054817Z-claude-es-iberia-software": {"c-B01-01", "c-B01-04", "c-B02-02"},
    "20261027T055400Z-claude-es-iberia-software": {"c-B01-01", "c-B02-02", "c-B05-01"},
}

_BIND_V6_GAINS = {
    "20261027T052502Z-claude-es-iberia-software": {
        "c-B01-14",
        "c-B01-15",
        "c-B01-16",
        "c-B02-02",
        "c-B02-03",
        "c-B02-04",
        "c-B02-05",
        "c-B02-06",
        "c-B02-07",
        "c-B02-08",
        "c-B05-05",
        "c-B05-10",
    },
    "20261027T054817Z-claude-es-iberia-software": {
        "c-B01-14",
        "c-B01-15",
        "c-B01-16",
        "c-B01-17",
        "c-B01-21",
        "c-B01-22",
        "c-B01-23",
        "c-B02-03",
        "c-B05-04",
        "c-B05-09",
    },
    "20261027T055400Z-claude-es-iberia-software": {
        "c-B02-06",
        "c-B02-07",
        "c-B02-08",
        "c-B05-06",
        "c-B05-12",
    },
}

# Hashes of the immutable original full verdict maps, not current comparator output.
# Pin non-gain claims too: a corrupted baseline must not become its own expected answer.
_BASELINES = {
    "20261027T052502Z-claude-es-iberia-software": (
        "sha256:9f469e87e385d0dda12a5883611bdd9067e4f51be0d1553f0ea5678b3570597d"
    ),
    "20261027T054817Z-claude-es-iberia-software": (
        "sha256:fd3f128e817e7554142b35a89e9098cff1e0ee433b2a1a53899183b779192f6e"
    ),
    "20261027T055400Z-claude-es-iberia-software": (
        "sha256:622b7beec96c02247834c40ee9e63d58d62d69bc94efb78b0dd90fb0598c8da3"
    ),
}


def expected_verdicts(run_dir: Path, *, binding: str | None = None) -> dict[str, bool]:
    """Version-bound oracle for historical inputs; unknown transitions fail closed.

    Original witness records remain immutable. bind-v7 preserves the exact reviewed
    bind-v6 gains on these historical inputs; its range rejection has dedicated negatives.
    A comparator drift cannot update its own expected answers.
    """
    from kyb.evals.witness import comparator
    from kyb.util.hashing import sha256_json

    version = binding or comparator.versions()["binding"]
    old = recorded_verdicts(run_dir)
    if version == "bind-v5":
        return old
    if version not in {"bind-v6", "bind-v7", "bind-v8"} or run_dir.name not in _BIND_V6_GAINS:
        raise ValueError(f"no reviewed replay expectation: {run_dir.name}/{version}")
    if sha256_json(old) != _BASELINES[run_dir.name]:
        raise ValueError(f"historical baseline changed: {run_dir.name}")
    gains = _BIND_V6_GAINS[run_dir.name]
    if version == "bind-v8":
        gains = gains | _BIND_V8_EXTRA_GAINS.get(run_dir.name, set())
    if not gains <= old.keys() or any(old[k] for k in gains):
        raise ValueError(f"historical baseline changed: {run_dir.name}")
    return {k: value or k in gains for k, value in old.items()}


def witness_model(run_dir: Path) -> str:
    """The model that witnessed the run in the product (every document's record names it)."""
    case, run = _case_run(run_dir)
    models = {
        json.loads(f.read_text("utf-8"))["model"]
        for f in (run_dir / "silver" / "witness" / case / run).glob("*.json")
    }
    if len(models) != 1:
        raise ValueError(f"{run_dir.name}: witnessed by {sorted(models)}, expected one model")
    return models.pop()


def recording_backend(run_dir: Path) -> dict[str, Any]:
    """The eval backend that is the run's own witness: its recorded answers are that backend's."""
    from kyb.evals.witness.backends import BACKENDS

    model = witness_model(run_dir)
    return next(b for b in BACKENDS if b["transport"] == "api" and b["model"] == model)


def seeds(run_dir: Path) -> list[Path]:
    raw = run_dir / "silver" / "witness" / "raw"
    return [raw] if raw.exists() else []


@contextlib.contextmanager
def render_cache() -> Iterator[None]:
    """Within the block, a crop already cut from the same PDF bytes, page, box and dpi is copied,
    not re-rendered. Only the benchmark uses it (every backend cuts the same crops); suite 5
    re-renders every time, because pinning the renderer is part of what it checks."""
    from kyb.docintel import poppler
    from kyb.util.hashing import sha256_bytes

    real, cache = poppler.render_region, {}

    def cached(pdf: Path, page: int, box, out_png: Path, dpi: int = poppler.DPI) -> Path:
        key = (sha256_bytes(Path(pdf).read_bytes()), page, tuple(box), dpi)
        if key not in cache:
            cache[key] = real(pdf, page, box, out_png, dpi).read_bytes()
        else:
            out_png.parent.mkdir(parents=True, exist_ok=True)
            out_png.write_bytes(cache[key])
        return out_png

    poppler.render_region = cached
    try:
        yield
    finally:
        poppler.render_region = real


def reverify(run_dir: Path, witness) -> dict[str, Any]:
    """Every document of the run through `verify_document` with `witness`."""
    from kyb.evals.renderer import require_legacy_binding

    require_legacy_binding(run_dir, witness)
    from kyb.witness.verify import verify_document

    case, run = _case_run(run_dir)
    docs = sorted(p.stem for p in (run_dir / "silver" / "claims" / case / run).glob("*.json"))
    names = {p.name[:2]: p for p in (PACKETS / case).glob("*.pdf")}
    verdicts: dict[str, bool] = {}
    per_doc: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="kyb-eval-run-") as tmp:
        root = Path(tmp)
        for part in ("claims", "layout", "pages"):
            shutil.copytree(run_dir / "silver" / part, root / "silver" / part)
        (root / "ledger").mkdir()
        for doc in docs:
            rec = verify_document(root, case, run, doc, witness, names[doc])
            for r in rec["results"]:
                verdicts[r["claim_id"]] = bool(r["verified"])
            per_doc[doc] = {
                "claims": len(rec["results"]),
                "verified": sum(r["verified"] for r in rec["results"]),
                "findings": sorted({f["kind"] for f in rec["findings"]}),
            }
    return {
        "run": run_dir.name,
        "case": case,
        "verdicts": verdicts,
        "documents": per_doc,
        "claims": len(verdicts),
        "verified": sum(verdicts.values()),
    }


def agreement(a: dict[str, bool], b: dict[str, bool]) -> float | None:
    common = sorted(set(a) & set(b))
    return round(sum(a[k] == b[k] for k in common) / len(common), 4) if common else None
