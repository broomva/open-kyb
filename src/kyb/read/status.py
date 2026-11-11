"""`kyb review <case> --status`: which step of kyb-review comes next (spec §2).

The agent asks the system instead of tracking state in its conversation.
"""

from __future__ import annotations

import json
from pathlib import Path

from kyb.read.plan import _documents, layout_rel, pages_rel
from kyb.read.submit import submitted
from kyb.util.fsutil import contained, safe_id
from kyb.workspace.catalog import Catalog
from kyb.workspace.runs import latest_run


def review_status(root: Path, case: str) -> dict:
    case = safe_id(case, "case id")
    if not contained(root, f"bronze/packets/{case}/manifest.json").exists():
        return {
            "case": case,
            "next": "ingest",
            "command": f"kyb ingest <packet_dir> --case {case} --as-of <date>",
        }
    run = latest_run(root, case)
    files = _documents(root, case)
    docs = [k for k, _ in files]
    catalog = Catalog(root)
    pages_done = run is not None and all(
        catalog.get(f"{pages_rel(case, run, k)}/p{n}.json") is not None
        for k, f in files for n in range(1, f["pages"] + 1)
    )  # fmt: skip
    if not pages_done:
        return {"case": case, "run": run, "next": "pages", "command": f"kyb pages {case}"}
    no_layout = [d for d in docs if catalog.get(layout_rel(case, run, d)) is None]
    if no_layout:
        return {"case": case, "run": run, "next": "layout", "pending": no_layout,
                "command": f"kyb layout {case}"}  # fmt: skip
    unread = [d for d in docs if not submitted(root, case, run, d)]
    if unread:
        return {"case": case, "run": run, "next": "read", "pending": unread,
                "launch": [f"kyb-reader {case} {run} {d}" for d in unread]}  # fmt: skip
    verified = f"silver/verified_claims/{case}/{run}.json"
    if catalog.get(verified) is None:
        return {"case": case, "run": run, "next": "verify",
                "command": f"kyb claims verify {case}"}  # fmt: skip
    stale = _verification_stale(root, catalog, case, run, docs, verified)
    if stale:
        return {"case": case, "run": run, "next": "stale", "detail": stale,
                "note": "the recorded verification no longer matches its inputs; "
                        "a fresh verification is a new run"}  # fmt: skip
    from kyb.witness.verify import assurance_now

    # assurance as of now: the summary file is a receipt of what verification established then
    current = {d: assurance_now(root, case, run, d)["assurance"] for d in docs}
    from kyb.decide.action import decision_status

    return {**decision_status(root, case, run), "verified": verified, "assurance": current}


def _verification_stale(root: Path, catalog: Catalog, case: str, run: str, docs: list[str],
                        verified: str) -> str | None:  # fmt: skip
    """Why the recorded verification cannot be built on, or None: the summary and every witness
    record must exist, hash to their receipts, and be witnessed against today's claims and
    layout."""
    from kyb.read.submit import claims_rel
    from kyb.util.hashing import sha256_bytes
    from kyb.witness.verify import _reuse

    path = contained(root, verified)
    if (
        not path.exists()
        or sha256_bytes(path.read_bytes()) != catalog.get(verified)["content_hash"]
    ):
        return f"{verified}: missing or differs from its catalog receipt"
    summary = json.loads(path.read_text("utf-8"))
    if any("assurance" in d for d in (summary.get("documents") or {}).values()):
        # a receipt written before assurance became `assurance_at_verify` would present a W2
        # that later evidence may have revoked as if it were current: never build on it
        return f"{verified}: a pre-receipt summary schema (bare `assurance`); verify in a new run"
    witnessed = []
    for d in docs:
        rel = f"silver/witness/{case}/{run}/{d}.json"
        if catalog.get(rel) is None or not contained(root, rel).exists():
            return f"{rel}: not witnessed"
        try:
            _reuse(root, catalog, rel, claims_rel(case, run, d), layout_rel(case, run, d))
        except (PermissionError, FileNotFoundError, TypeError, ValueError) as exc:
            return str(exc)
        witnessed.append(catalog.ref(rel))
    if sorted(catalog.get(verified).get("inputs") or []) != sorted(witnessed):
        return f"{verified}: summarises different witness records"
    return None
