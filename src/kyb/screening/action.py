"""The `screen` action: the recorded `FactSet` + a watchlist → `gold/screening/<case>/<run>.json`.

Call under the workspace action lock. The watchlist is pinned by its declared version and content
hash (`watchlist_version`); the similarity threshold is read from the workspace's policy file.
is written once: screening the same facts against the same list and policy gives the same bytes,
and anything else for the same run is refused. Completion is resumable: publish, audit once,
catalog receipt last (see `kyb.facts.action`).
"""

from __future__ import annotations

import json
from pathlib import Path

from kyb.audit.chain import AuditLog
from kyb.contracts.b4 import SCREENING_REL, ScreeningResult
from kyb.facts.action import audited, facts_rel, load_facts
from kyb.policy.provenance import approved_policy
from kyb.screening.screen import policy_threshold, screen_factset
from kyb.util.fsutil import contained, publish_exclusive, safe_id
from kyb.util.hashing import sha256_bytes, sha256_json
from kyb.workspace.catalog import Catalog
from kyb.workspace.runs import FRESH_RUN, run_meta, safe_run_id

WATCHLIST_REL = "policy/watchlists.json"  # where a workspace keeps its list, when it has one


def screening_rel(case: str, run: str) -> str:
    return SCREENING_REL.format(case=safe_id(case, "case id"), run=safe_run_id(run))


def default_watchlist(root: Path) -> Path | None:
    path = contained(root, WATCHLIST_REL)
    return path if path.is_file() else None


def screen_case(root: Path, case: str, run: str, watchlist: Path) -> tuple[ScreeningResult, str,
                                                                           bool]:  # fmt: skip
    """Screen and record. Returns (result, path, written); `written` is False when the identical
    record already existed."""
    case = safe_id(case, "case id")
    run = safe_run_id(run)
    as_of = run_meta(root, case, run)["as_of"]
    fs = load_facts(root, case, run)
    raw = Path(watchlist).read_bytes()
    listed = json.loads(raw)
    version = f"{listed.get('version', 'unversioned')}#{sha256_bytes(raw)}"
    policy = approved_policy(root)
    threshold = policy_threshold(policy.decode("utf-8"))
    result = screen_factset(fs, listed, watchlist_version=version, threshold=threshold)
    rel = screening_rel(case, run)
    body = (
        json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n"
    ).encode()
    out_hash = sha256_bytes(body)
    catalog = Catalog(root)
    if catalog.get(rel) is not None:
        if sha256_bytes(contained(root, rel).read_bytes()) != out_hash:
            raise PermissionError(
                f"{rel}: already screened differently (another watchlist, policy threshold or "
                f"facts); a run is screened once. {FRESH_RUN}"
            )
        return result, rel, False
    publish_exclusive(contained(root, rel), body)  # identical bytes from an interrupted try: ok
    inputs = [catalog.ref(facts_rel(case, run))]
    if not audited(root, "screen", out_hash):
        AuditLog(root / "ledger" / "audit.jsonl").append(
            "screening", "tool.call",
            input_hash=sha256_json({"facts": inputs, "watchlist": version,
                                    "policy": sha256_bytes(policy)}),
            output_hash=out_hash,
            detail={"action": "screen", "case": case, "run": run, "method": result.method,
                    "candidates": len(result.candidates),
                    "triggers_fuzzy": sum(c.triggers_fuzzy for c in result.candidates)},
        )  # fmt: skip
    catalog.register(rel, tier="derived", as_of=as_of, inputs=inputs,
                     produced_by={"action": "screen", "method": result.method,
                                  "watchlist_version": version,
                                  "policy_hash": sha256_bytes(policy)})  # fmt: skip
    return result, rel, True
