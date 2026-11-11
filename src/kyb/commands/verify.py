"""`kyb verify` (L6, `make verify`): the eval suites and the JSON scorecard (EVALS.md)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

lazy from kyb.evals import runner
lazy from kyb.evals.witness import labelled

app = typer.Typer()


@app.command(help="run the eval suites, write the scorecard; non-zero on a blocking failure")
def verify(
    live: Annotated[
        bool, typer.Option("--live", help="record missing witness-eval calls (API, agy); opt-in")
    ] = False,
    strict: Annotated[
        bool, typer.Option("--strict", help="release gate: a pending blocking suite fails too")
    ] = False,
    suite: Annotated[
        list[str] | None,
        typer.Option("--suite", help="run only these suite ids (1..8, G, B-witness, ...)"),
    ] = None,
    out: Annotated[str | None, typer.Option("--out", help="scorecard path")] = None,
    rebuild_crops: Annotated[
        bool,
        typer.Option(
            "--rebuild-crops",
            help="re-render the labelled witness crops from the packets, then exit",
        ),
    ] = False,
) -> int:
    if rebuild_crops:
        print(json.dumps({"rebuilt": labelled.build_crops()}))
        return 0
    path = Path(out) if out else runner.SCORECARD
    try:
        code, card = runner.run(live=live, strict=strict, out=path, only=suite or None)
    except ValueError as exc:  # a typo must never become a vacuous pass
        raise typer.BadParameter(str(exc), param_hint="--suite") from exc
    summary = {
        "verdict": card["verdict"],
        "headline": card["headline"],
        "scorecard": str(path),
        "suites": {s["id"]: s["status"] for s in card["suites"]},
    }
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return code
