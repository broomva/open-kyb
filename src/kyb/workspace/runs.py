"""Runs: every pipeline pass over a case is a run (`r1`, `r2`, …).

Run-keyed records are written once (spec, spec). A re-run never overwrites; it gets the next id.
Run records live in the ledger, so they are catalogued and audited like any other record.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from kyb.audit.chain import AuditLog
from kyb.util.fsutil import contained, publish_exclusive, safe_id
from kyb.util.hashing import sha256_json
from kyb.workspace.catalog import Catalog

_RUN = re.compile(r"r[1-9]\d{0,5}")


# What to do when a run cannot be re-run under changed inputs (a policy threshold, the watchlist):
# every run step reads the latest run, so the steps after `pages` follow the new one.
FRESH_RUN = (
    "Start a new run of the case: pages(case, new_run=True) (`kyb pages <case> --new-run`), "
    "then layout_pages, the readers, verify_quotes, run_review and decide on that run."
)


def safe_run_id(run_id: str) -> str:
    if not _RUN.fullmatch(run_id):
        raise ValueError(f"invalid run id {run_id!r} (expected r1, r2, …)")
    return run_id


def _runs_rel(case: str) -> str:
    return f"ledger/runs/{safe_id(case, 'case id')}"


def list_runs(root: Path, case: str) -> list[str]:
    d = contained(root, _runs_rel(case))
    ids = [p.stem for p in d.glob("r*.json") if _RUN.fullmatch(p.stem)] if d.exists() else []
    return sorted(ids, key=lambda r: int(r[1:]))


def latest_run(root: Path, case: str) -> str | None:
    runs = list_runs(root, case)
    return runs[-1] if runs else None


def start_run(root: Path, case: str, *, as_of: str, reason: str) -> str:
    """Allocate the next run id. Call under the workspace action lock."""
    runs = list_runs(root, case)
    run_id = f"r{int(runs[-1][1:]) + 1 if runs else 1}"
    rel = f"{_runs_rel(case)}/{run_id}.json"
    meta = {"run_id": run_id, "case": case, "as_of": as_of, "reason": reason}
    publish_exclusive(contained(root, rel), (json.dumps(meta, indent=2) + "\n").encode("utf-8"))
    produced_by = {"action": "start_run", "reason": reason}
    Catalog(root).register(rel, tier="none", produced_by=produced_by, as_of=as_of)
    AuditLog(root / "ledger" / "audit.jsonl").append(
        "runs", "run.started",
        input_hash=sha256_json({"case": case, "as_of": as_of}), output_hash=sha256_json(meta),
        detail={"case": case, "run": run_id, "reason": reason},
    )  # fmt: skip
    return run_id


def run_for_pages(root: Path, case: str, *, as_of: str, new_run: bool = False) -> str:
    """The run `pages` renders into: the latest, or a new one when there is none, or when
    `new_run` asks for one and the latest has already consumed its inputs (facts or screening
    recorded). A retried `new_run` therefore reuses the run it allocated. Call under the lock."""
    latest = latest_run(root, case)
    if latest is not None and not (new_run and _consumed(root, case, latest)):
        return latest
    return start_run(root, case, as_of=as_of, reason="new run" if new_run else "pages")


def _consumed(root: Path, case: str, run: str) -> bool:
    from kyb.contracts.b4 import FACTS_REL, SCREENING_REL

    catalog = Catalog(root)
    return any(catalog.get(rel.format(case=case, run=run)) for rel in (FACTS_REL, SCREENING_REL))


def run_meta(root: Path, case: str, run_id: str) -> dict:
    path = contained(root, f"{_runs_rel(case)}/{safe_run_id(run_id)}.json")
    return json.loads(path.read_text("utf-8"))
