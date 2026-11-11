"""Evidence loop CLI commands (issue)."""

from __future__ import annotations

import json
from typing import Annotated

import typer

from kyb.commands._common import root, run
lazy from kyb.audit.chain import workspace_action
lazy from kyb.loop import action
lazy from kyb.loop.models import LoopBudget
lazy from kyb.util.fsutil import safe_id

app = typer.Typer(name="loop", help="evidence loop (issue)")


@app.command(help="inspect_loop: read-only evidence loop inspection")
def inspect(
    ctx: typer.Context,
    case: str,
    run_: Annotated[str | None, typer.Option("--run")] = None,
) -> None:
    ws = root(ctx)
    case_id = safe_id(case, "case id")
    run_id = run(ws, case_id, run_)
    res = action.inspect_loop(ws, case_id, run_id)
    print(json.dumps(res, indent=2, ensure_ascii=False))


@app.command(help="plan_loop: run the planner and record an immutable plan")
def plan(
    ctx: typer.Context,
    case: str,
    run_: Annotated[str | None, typer.Option("--run")] = None,
    previous_plan: Annotated[str | None, typer.Option("--previous-plan")] = None,
    max_attempts: Annotated[int | None, typer.Option("--max-attempts")] = None,
    max_elapsed_seconds: Annotated[int | None, typer.Option("--max-elapsed-seconds")] = None,
) -> None:
    ws = root(ctx)
    case_id = safe_id(case, "case id")
    run_id = run(ws, case_id, run_)

    budget_kwargs = {}
    if max_attempts is not None:
        budget_kwargs["max_attempts"] = max_attempts
    if max_elapsed_seconds is not None:
        budget_kwargs["max_elapsed_seconds"] = max_elapsed_seconds
    budget = LoopBudget(**budget_kwargs) if budget_kwargs else None

    with workspace_action(ws):
        res = action.record_loop_plan(
            ws, case_id, run_id, budget=budget, previous_plan=previous_plan
        )
    print(json.dumps(res, indent=2, ensure_ascii=False))


@app.command(help="start_loop_repair: idempotently allocate or start a repair run")
def repair(
    ctx: typer.Context,
    case: str,
    plan_id: Annotated[str, typer.Option("--plan", help="64-character lowerhex plan id")],
) -> None:
    ws = root(ctx)
    case_id = safe_id(case, "case id")
    with workspace_action(ws):
        res = action.start_loop_repair(ws, case_id, plan_id)
    print(json.dumps(res, indent=2, ensure_ascii=False))
