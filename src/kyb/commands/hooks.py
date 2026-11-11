"""`kyb guard <event>`: thin aliases of `kyb-hook <event>` for hooks written before spec."""

from __future__ import annotations

import typer

from kyb.commands.audit import Source
lazy from kyb.workspace import guard

app = typer.Typer(name="guard", help="harness hooks")


def _alias(event: str, handler: str) -> None:
    def command(source: Source = "") -> int:
        return guard.run_hook(getattr(guard, handler), source)

    app.command(event, help="")(command)


for _event, _handler in (("pre", "pre_tool_use"), ("stop", "stop"),
                         ("session-start", "session_start"),
                         ("subagent-start", "subagent_start"),
                         ("subagent-stop", "subagent_stop")):  # fmt: skip
    _alias(_event, _handler)
