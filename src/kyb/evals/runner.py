"""`make verify` / `kyb verify`: run every suite, write the scorecard, return the exit status.

Exit 0 when no blocking suite fails and false-APPROVE is 0; `strict` (the release gate) also
fails on a pending blocking suite. `live` lets the witness backend eval record what is missing
(opt-in: it spends API calls and `agy` quota); everything else always replays.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from kyb import __version__
from kyb.contracts.b4 import CONTRACTS_VERSION
from kyb.evals import (
    adversarial,
    contract,
    decisions,
    determinism,
    extraction,
    pending_suites,
    policy_edits,
    properties,
    scorecard,
)
from kyb.evals.paths import REPO, datasets
from kyb.evals.scorecard import Suite, check, incomparable, pending
from kyb.evals.witness import comparator
from kyb.evals.witness import evaluate as wev
from kyb.util.hashing import sha256_bytes

SCORECARD = REPO / "artifacts" / "verify" / "scorecard.json"


def _dataset_files(root: Path) -> list[Path]:
    if root == (REPO / "datasets").resolve():
        # The released corpus is the index's membership, with working-tree contents.
        # Runtime locks and other untracked artifacts must not change its identity.
        tracked = subprocess.check_output(
            ["git", "-C", str(REPO), "ls-files", "--cached", "-z", "--", "datasets/"]
        )
        return sorted({REPO / os.fsdecode(p) for p in tracked.split(b"\0") if p})
    # KYB_DATASETS also supports an explicitly supplied corpus outside the checkout.
    return sorted(p for p in root.rglob("*") if p.is_file())


def _inputs() -> dict[str, Any]:
    from kyb.evals.decisions import base_policy

    root = datasets().resolve()
    tree = {str(p.relative_to(root)): sha256_bytes(p.read_bytes()) for p in _dataset_files(root)}
    pol = base_policy()
    return {
        "kyb": __version__,
        "contracts": CONTRACTS_VERSION,
        "rules_version": pol.rules_version,
        "policy_hash": pol.content_hash,
        "comparator": comparator.versions(),
        "datasets_files": len(tree),
        "datasets_hash": sha256_bytes(json.dumps(tree, sort_keys=True).encode()),
    }


def witness_suite(live: bool, out_dir: Path | None) -> Suite:
    s = Suite(
        "B-witness",
        "Benchmark: witness backends (report-only, spec input)",
        False,
        threshold="report-only; the eval's own fault test is blocking in suite 3",
    )
    report = wev.evaluate(live=live)
    for row in report["rows"]:
        if row.get("incomparable"):
            s.cases.append(
                incomparable(f"recorded:{row['backend']}", "; ".join(row["incomparable"])[:500])
            )
        elif row["missing"]:
            s.cases.append(
                pending(
                    f"recorded:{row['backend']}",
                    "make verify LIVE=1",
                    "; ".join(row["missing"])[:500],
                )
            )
        else:
            s.cases.append(check(f"recorded:{row['backend']}", True))
    s.metrics = {
        "comparator": report["comparator"],
        "rows": report["rows"],
        "isolation": report["isolation"],
        "self_test": report["self_test"],
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "witness-backends.json").write_text(
            json.dumps(report, indent=1, ensure_ascii=False) + "\n", "utf-8"
        )
        (out_dir / "witness-backends.md").write_text(wev.table(report) + "\n", "utf-8")
    return s


def run(
    *,
    live: bool = False,
    strict: bool = False,
    out: Path = SCORECARD,
    only: list[str] | None = None,
) -> tuple[int, dict[str, Any]]:
    builders = {
        "1": contract.suite,
        "2": properties.suite,
        "3": extraction.suite,
        "4": lambda: decisions.suite()[0],
        "5": determinism.suite,
        "6": adversarial.suite,
        "7": pending_suites.qa,
        "8": policy_edits.suite,
        "G": pending_suites.agent_gates,
        "B-witness": lambda: witness_suite(live, out.parent),
        "B-screening": pending_suites.screening,
        "B-support": pending_suites.support,
        "B-adapters": pending_suites.benchmarks,
    }
    if only is not None:
        unknown = sorted(set(only) - set(builders))
        if unknown or not only:
            raise ValueError(f"unknown suite ids {unknown}; known: {list(builders)}")
    suites = [fn() for sid, fn in builders.items() if not only or sid in only]
    false_approve = next(
        (s.metrics.get("false_approve", 0) for s in suites if s.suite_id == "4"), 0
    )
    card = scorecard.build(
        suites, false_approve=false_approve, inputs=_inputs(), strict=strict, live=live
    )
    scorecard.write(card, out)
    return (0 if card["verdict"] == "pass" else 1), card
