"""Sign-off (L5): `kyb queue`, `kyb sign`, `kyb officer enroll` (spec sub-app).

Run by the officer in their own terminal, or by the agent only after the officer approved it in
the harness (`gate.py`: AskUserQuestion answer plus the harness permission prompt). Overrides and
enrolment are the officer's alone. The token comes from `KYB_OFFICER_TOKEN` (the workspace
environment: the shell or harness settings `env` that launched the session) or a hidden prompt,
never from an argument, so it stays out of the shell history and the process list and the agent
never handles it.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Annotated, Literal

import typer

from kyb.commands._common import root
lazy from kyb.audit.chain import workspace_action
lazy from kyb.signing import officers
lazy from kyb.signing.queue import review_queue
lazy from kyb.signing.sign import sign_decision

app = typer.Typer()
officer_app = typer.Typer(name="officer", help="officers who may sign (spec)")
app.add_typer(officer_app)

Officer = Annotated[str, typer.Option("--officer", help="your officer id")]
TOKEN_ENV = "KYB_OFFICER_TOKEN"


def _token(who: str) -> str:
    return os.environ.get(TOKEN_ENV) or typer.prompt(f"officer token for {who}", hide_input=True)


def _approved_via() -> str:
    """Claude Code sets CLAUDECODE in the shells it runs: a sign made there was approved in the
    harness (gate.py), any other is the officer's own terminal."""
    return "harness-ask" if os.environ.get("CLAUDECODE") else "terminal"


def _refuse(exc: Exception) -> None:
    sys.exit(f"kyb: {type(exc).__name__}: {exc}")


@app.command(help="review_queue: unsigned recommendations, MANUAL_REVIEW first, oldest first")
def queue(ctx: typer.Context, case: str | None = None, include_archived: bool = False) -> None:
    ws = root(ctx)
    print(
        json.dumps(
            review_queue(ws, case, include_archived=include_archived), indent=2, ensure_ascii=False
        )
    )


@app.command(help="sign_decision: assent to a recommendation, or override it (officer only)")
def sign(
    ctx: typer.Context,
    decision_id: str,
    verdict: Literal["assent", "override"],
    officer: Officer,
    status: Annotated[str | None, typer.Option("--status", help="override: the new status")] = None,
    rationale: Annotated[str, typer.Option("--rationale", help="override: why")] = "",
) -> None:
    ws = root(ctx)
    try:
        with workspace_action(ws):
            result = sign_decision(ws, decision_id=decision_id, verdict=verdict, officer=officer,
                                   officer_token=_token(officer), status=status,
                                   rationale=rationale,
                                   approved_via=_approved_via())  # fmt: skip
    except (PermissionError, LookupError, ValueError) as exc:
        _refuse(exc)
    print(json.dumps(result, indent=2, ensure_ascii=False))


@app.command(help="override: sign a different status than the recommendation, with a rationale")
def override(
    ctx: typer.Context,
    decision_id: str,
    officer: Officer,
    status: Annotated[str, typer.Option("--status", help="the status you decide")],
    rationale: Annotated[str, typer.Option("--rationale", help="why (recorded, attributed)")],
) -> None:
    sign(ctx, decision_id, "override", officer, status, rationale)


@officer_app.command("enroll", help="enrol an officer; the token is shown once")
def enroll(
    ctx: typer.Context,
    officer: str,
    by: Annotated[
        str | None, typer.Option("--by", help="an enrolled officer (after the first)")
    ] = None,
) -> None:
    ws = root(ctx)
    try:
        with workspace_action(ws):
            by_token = _token(by) if by and officers.enrolled(ws) else None
            result = officers.enroll(ws, officer, by=by, by_token=by_token)
    except (PermissionError, FileExistsError, ValueError) as exc:
        _refuse(exc)
    note = (
        f"shown once. Keep it out of the shell that runs the harness; sign with the hidden "
        f"prompt, or {TOKEN_ENV} in your own terminal"
    )
    print(
        json.dumps({"officer": result["officer"], "token": result["token"], "note": note}, indent=2)
    )
