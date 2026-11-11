"""The runner's Typer commands (spec)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from kyb.audit.chain import workspace_action
from kyb.commands._common import root
from kyb.commands._common import run as resolve_run
from kyb.run.action import run_case

app = typer.Typer()


@app.command("decide")
def decide_command(
    ctx: typer.Context,
    case: str,
    run: Annotated[str | None, typer.Option()] = None,
    profile: Annotated[
        str | None,
        typer.Option(help="decision profile: strict (default) or approve-measured (spec)"),
    ] = None,
) -> int:
    """Compose the recorded predicates and emit a pending-signature KYBDecision."""
    from kyb.decide.action import decide_case

    ws = root(ctx)
    with workspace_action(ws):
        record = decide_case(ws, case, resolve_run(ws, case, run), profile=profile)
    print(record.model_dump_json())
    return 0


@app.command("run")
def run_command(
    ctx: typer.Context,
    case: str,
    run: Annotated[str | None, typer.Option()] = None,
    facts_fixture: Annotated[Path | None, typer.Option()] = None,
    screening_fixture: Annotated[Path | None, typer.Option()] = None,
) -> int:
    """Build code predicates and the isolated model-predicate work order."""
    ws = root(ctx)
    with workspace_action(ws):
        out = run_case(
            ws,
            case,
            resolve_run(ws, case, run),
            facts_fixture=facts_fixture,
            screening_fixture=screening_fixture,
        )
    print(json.dumps(out, ensure_ascii=False))
    return 0
