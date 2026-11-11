"""Run the offline evidence-loop matrix and write its receipt.

    uv run python scripts/eval_evidence_loop.py [--out DIR]

Exit status: 0 matrix pass, 1 fail, 2 pending (no core, or action-layer cases the parent owns).
Pending never passes. A green matrix is not a product PASS: `product_pass` is always false.
Writes `evidence-loop-eval.json` and `summary.txt` under DIR (default
`artifacts/gates/evidence-loop-evals`). No model, provider or network call is made.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from kyb.evals.loop_eval import exit_code, render_summary, run_matrix
from kyb.evals.scorecard import write

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "artifacts" / "gates" / "evidence-loop-evals"


def code_version() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
        )
    except OSError, subprocess.CalledProcessError:
        return "unrecorded"
    return out.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="receipt directory")
    args = parser.parse_args(argv)
    doc = run_matrix(code_version=code_version())
    write(doc, args.out / "evidence-loop-eval.json")
    (args.out / "summary.txt").write_text(render_summary(doc), "utf-8")
    print(render_summary(doc), end="")
    return exit_code(doc)


if __name__ == "__main__":
    raise SystemExit(main())
