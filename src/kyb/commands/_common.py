"""What every command group shares: the workspace the root `-w` option names, and run lookup."""

from __future__ import annotations

import sys
from pathlib import Path

import typer

lazy from kyb.workspace import guard
lazy from kyb.workspace.runs import latest_run


def root(ctx: typer.Context) -> Path:
    workspace = ctx.obj.get("workspace", ".")
    found = guard.find_root(Path(workspace).resolve())
    if found is None:
        sys.exit(f"kyb: no workspace at or above {workspace} (run `kyb init` first)")
    return found


def run(root: Path, case: str, run: str | None) -> str:
    run = run or latest_run(root, case)
    if run is None:
        sys.exit(f"kyb: {case} has no run yet (run `kyb pages {case}` first)")
    return run


def terminal_token(officer: str, purpose: str) -> str:
    """The officer's token, typed at the controlling terminal: never stdin or the environment,
    which an agent's shell can fill."""
    import getpass

    try:
        open("/dev/tty").close()  # noqa: SIM115 - only probing for a controlling terminal
    except OSError:
        sys.exit("kyb: the officer's token is typed at a terminal; there is none here")
    return getpass.getpass(f"officer token for {officer} to {purpose}: ")
