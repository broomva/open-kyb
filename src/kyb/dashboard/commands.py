"""Explicit browser lifetime for standalone use; the MCP server has its own lifetime."""

import json
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlencode

import typer

from kyb.commands._common import root
from kyb.dashboard import data
from kyb.dashboard.browser import DashboardServer

app = typer.Typer(
    name="dashboard",
    invoke_without_command=True,
    no_args_is_help=False,
    help="Open the case dashboard or inspect/prepare a case-bound chat request.",
)


@app.callback()
def dashboard(
    ctx: typer.Context,
    case: str | None = None,
    run: str | None = None,
    open_browser: Annotated[bool, typer.Option("--open-browser")] = False,
) -> None:
    if ctx.invoked_subcommand:
        return
    ws: Path = root(ctx)
    data.snapshot(ws, case, run)  # Refuse bad context before exposing a URL.
    server = DashboardServer(ws)
    url = server.url
    query = {k: v for k, v in {"case": case, "run": run}.items() if v}
    if query:
        url += "?" + urlencode(query)
    print(url, flush=True)
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _print(call) -> None:
    try:
        print(json.dumps(call(), ensure_ascii=False, indent=2))
    except data.EXPECTED as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command("snapshot", help="Read the dashboard's typed case/context/activity snapshot.")
def snapshot(ctx: typer.Context, case: str | None = None, run: str | None = None) -> None:
    _print(lambda: data.snapshot(root(ctx), case, run))


@app.command("request-retry", help="Save a retry request only; never start a run or sign.")
def request_retry(
    ctx: typer.Context,
    case: str,
    run: Annotated[str, typer.Option("--run")],
    snapshot_id: Annotated[str, typer.Option("--snapshot")],
    max_attempts: int = 2,
    max_elapsed_seconds: int = 900,
    workspace_id: str | None = None,
) -> None:
    _print(
        lambda: data.prepare(
            root(ctx), case, run, snapshot_id, max_attempts, max_elapsed_seconds, workspace_id
        )
    )


@app.command("handoff", help="Revalidate a recorded retry and read its allocation receipt.")
def handoff(ctx: typer.Context, case: str, plan_id: str, workspace_id: str | None = None) -> None:
    _print(lambda: data.handoff(root(ctx), case, plan_id, workspace_id))


@app.command("intent", help="Compose a case-bound question/work request for the officer's chat.")
def intent(
    ctx: typer.Context,
    case: str,
    workspace_id: Annotated[str, typer.Option("--workspace-id")],
    kind: Annotated[
        Literal[
            "question", "continue_review", "review_recommendation", "archive_case", "restore_case"
        ],
        typer.Option("--kind"),
    ],
    run: str | None = None,
    question: str = "",
    snapshot_id: str | None = None,
    case_state_hash: str | None = None,
) -> None:
    _print(
        lambda: data.intent(
            root(ctx), case, run, workspace_id, kind, question, snapshot_id, case_state_hash
        )
    )


@app.command(
    "archive", help="After chat confirmation, archive from active queues; keep all history."
)
def archive(
    ctx: typer.Context,
    case: str,
    workspace_id: Annotated[str, typer.Option("--workspace-id")],
    expected_event_hash: str | None = None,
) -> None:
    _print(
        lambda: data.change_case_membership(
            root(ctx), case, workspace_id, expected_event_hash, archived=True
        ).model_dump(mode="json")
    )


@app.command("restore", help="After chat confirmation, restore to active queues; starts no work.")
def restore(
    ctx: typer.Context,
    case: str,
    workspace_id: Annotated[str, typer.Option("--workspace-id")],
    expected_event_hash: Annotated[str, typer.Option("--expected-event-hash")],
) -> None:
    _print(
        lambda: data.change_case_membership(
            root(ctx), case, workspace_id, expected_event_hash, archived=False
        ).model_dump(mode="json")
    )
