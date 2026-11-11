"""`kyb audit verify` and `kyb audit append-hook` (an alias of `kyb-hook post`)."""

from __future__ import annotations

import json
from typing import Annotated, Literal

import typer

from kyb.commands._common import root
lazy from kyb.audit.chain import verify_workspace
lazy from kyb.util.fsutil import action_lock
lazy from kyb.workspace import guard

app = typer.Typer(name="audit", help="audit chain")

# "" is a hook written before `--source` existed; attestation counts it as untagged
Source = Annotated[
    Literal["", "plugin", "workspace"],
    typer.Option("--source", help="the hook registration: plugin | workspace"),
]


@app.command("verify", help="")
def verify(ctx: typer.Context) -> int:
    ws = root(ctx)
    with action_lock(ws):  # at a boundary: no action is in progress, so the tail must be pinned
        r = verify_workspace(ws, at_boundary=True)
    print(json.dumps({"ok": r.ok, "events": r.events, "head": r.head, "error": r.error}))
    return 0 if r.ok else 1


@app.command("append-hook", help="PostToolUse hook: append the tool call to the chain")
def append_hook(source: Source = "") -> int:
    return guard.run_hook(guard.append_hook, source)
