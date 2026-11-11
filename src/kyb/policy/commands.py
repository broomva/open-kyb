"""`kyb policy propose|apply`: the CLI of `kyb.policy.change`."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer

from kyb.commands._common import root, terminal_token
lazy from kyb.audit.chain import workspace_action
lazy from kyb.policy import change

app = typer.Typer(name="policy", help="propose and apply policy changes")
REFUSALS = (PermissionError, LookupError, ValueError, FileNotFoundError)


def _terminal_token(officer: str) -> str:
    return terminal_token(officer, "apply")


@app.command(help="propose_policy: validate a complete new policy text and record it")
def propose(
    ctx: typer.Context,
    file: Annotated[Path, typer.Option("--file", help="the complete proposed compliance_rules.md")],
) -> int:
    ws = root(ctx)
    try:
        with workspace_action(ws):
            out = change.propose_policy(ws, file.read_text("utf-8"))
    except REFUSALS as exc:
        sys.exit(f"kyb: {type(exc).__name__}: {exc}")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


@app.command(
    help="apply_policy as an officer, in their own terminal (an agent uses MCP apply_policy)"
)
def apply(
    ctx: typer.Context,
    proposal_id: str,
    expected_current_hash: Annotated[
        str, typer.Option("--expected-current-hash", help="the live policy's sha256 hash")
    ],
    officer: Annotated[str, typer.Option("--officer", help="the approving officer's id")],
) -> int:
    ws = root(ctx)
    try:
        # the officer's own terminal: show what is approved, then take the token typed at the
        # controlling terminal, never from the environment an agent's shell could read
        shown = change.summary(change.load_proposal(ws, proposal_id))
        rules = ", ".join(shown["changed_rules"]) or "none"
        print("\n".join(["Parameter diff:", *(f"  {d}" for d in shown["parameter_diff"]),
                         "Severity and points:", *(f"  {d}" for d in shown["rule_diff"]),
                         f"Rule text changed: {rules}"]), file=sys.stderr)  # fmt: skip
        token = _terminal_token(officer)
        with workspace_action(ws):
            out = change.apply_policy(ws, proposal_id, expected_current_hash, officer=officer,
                                      officer_token=token)  # fmt: skip
    except REFUSALS as exc:
        sys.exit(f"kyb: {type(exc).__name__}: {exc}")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0
