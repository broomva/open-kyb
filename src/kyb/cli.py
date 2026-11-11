"""`kyb`: the thin CLI (spec). Every command is an action in the core; nothing decides here.

Each lane registers its commands as a Typer sub-app in `kyb.commands.<lane>` and adds one
`app.add_typer(...)` line below. The harness hooks have their own fast entry point, `kyb-hook`;
`kyb guard …` and `kyb audit append-hook` remain as aliases.
"""

from __future__ import annotations

import contextvars
from typing import Annotated

import typer

from kyb.commands import audit, hooks, intake, reading, verify
from kyb.copilot import commands as copilot
from kyb.dashboard import commands as dashboard
from kyb.facts import commands as facts
from kyb.loop import commands as loop
from kyb.policy import commands as policy
from kyb.run import commands as runner
from kyb.signing import commands as signing

# The exit status of the command that ran in this call of `main` (unset: none ran to completion)
_STATUS: contextvars.ContextVar[int] = contextvars.ContextVar("kyb_cli_status")


def _exit_status(code: int | None, **_params: object) -> None:
    # Every command returns its exit status; Typer would otherwise exit 0 after any command.
    _STATUS.set(code or 0)
    if code:
        raise typer.Exit(code)


app = typer.Typer(
    name="kyb",
    help="open-kyb KYB reviewer",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    result_callback=_exit_status,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.callback()
def _workspace(
    ctx: typer.Context,
    workspace: Annotated[
        str, typer.Option("-w", "--workspace", help="workspace path (default: search upward)")
    ] = ".",
) -> None:
    ctx.obj = {"workspace": workspace}


app.add_typer(intake.app)  # init, ingest, pages
app.add_typer(reading.app)  # read plan, claims submit|verify, layout, review
app.add_typer(audit.app)  # audit verify|append-hook
app.add_typer(hooks.app)  # guard <event> (aliases of kyb-hook)
app.add_typer(facts.app)  # facts, screen (L2)
app.add_typer(verify.app)  # verify (L6: make verify, the scorecard)
app.add_typer(signing.app)  # queue, sign, officer enroll (L5)
app.add_typer(copilot.app)  # ask, answer (L5)
app.add_typer(runner.app)  # run (L4: facts → screening → work order)
app.add_typer(loop.app)
app.add_typer(dashboard.app)
app.add_typer(policy.app)  # policy propose|apply


def main(argv: list[str] | None = None) -> int:
    """Run the CLI. A command's exit status is returned; a usage error raises SystemExit(2), and a
    refusal message raises SystemExit(message), exactly as the argparse CLI did."""
    token = _STATUS.set(-1)
    try:
        try:
            app(args=argv, prog_name="kyb")
        except SystemExit:
            if _STATUS.get() < 0:  # no command ran to completion: Click's or a refusal's exit
                raise
        return max(_STATUS.get(), 0)
    finally:
        _STATUS.reset(token)


if __name__ == "__main__":
    raise SystemExit(main())
