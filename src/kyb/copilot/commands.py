"""Copilot (L5): `kyb ask`, `kyb answer`, `kyb harness ingest-session` (spec sub-app)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Literal

import typer

from kyb.commands._common import root
lazy from kyb.audit.chain import workspace_action
lazy from kyb.copilot.answer import record_answer
lazy from kyb.copilot.bundle import ask_witnessed
lazy from kyb.util.fsutil import safe_id
lazy from kyb.witness import port

app = typer.Typer()
harness = typer.Typer(name="harness", help="harness session capture (agent-driven-review §5)")
app.add_typer(harness)
Run = Annotated[str | None, typer.Option("--run")]
WitnessMode = Annotated[Literal["auto", "replay", "live"], typer.Option("--witness")]


@app.command(help="ask: the verified evidence bundle for a question (facts, then snippets)")
def ask(ctx: typer.Context, case: str, question: str, run_: Run = None,
        witness: WitnessMode = "auto") -> None:  # fmt: skip
    ws = root(ctx)
    try:
        with workspace_action(ws):
            out = ask_witnessed(ws, safe_id(case, "case id"), question, run_,
                                port.make_witness(ws, witness))  # fmt: skip
    except (LookupError, PermissionError, ValueError) as exc:
        sys.exit(f"kyb: {type(exc).__name__}: {exc}")
    print(json.dumps(out, indent=2, ensure_ascii=False))


@app.command(help="record_answer: check structured statements (JSON file), render the answer")
def answer(
    ctx: typer.Context,
    case: str,
    question: str,
    file: Annotated[str, typer.Option("--file", help='JSON list of statements, or "-" for stdin')],
    run_: Run = None,
) -> int:
    ws = root(ctx)
    raw = sys.stdin.read() if file == "-" else Path(file).read_text("utf-8")
    try:
        statements = json.loads(raw)
        if not isinstance(statements, list):
            raise ValueError("the statements file is a JSON list")
        with workspace_action(ws):
            out = record_answer(ws, safe_id(case, "case id"), question, statements, run_)
    except (LookupError, PermissionError, ValueError) as exc:
        sys.exit(f"kyb: {type(exc).__name__}: {exc}")
    print(out["text"] if out["accepted"] else json.dumps(out, indent=2, ensure_ascii=False))
    return 0 if out["accepted"] else 1


@harness.command(
    "ingest-session", help="display check: flag what a session showed beyond rendered answers"
)
def ingest_session(
    ctx: typer.Context,
    transcript: Annotated[str | None, typer.Option("--transcript")] = None,
    session: Annotated[str | None, typer.Option("--session")] = None,
) -> int:
    """With no options it reads the hook event on stdin (a Stop or SessionEnd hook's input)."""
    from kyb.copilot import display

    ws = root(ctx)
    if transcript is None:
        event = json.loads(sys.stdin.read() or "{}")
        transcript, session = event.get("transcript_path"), event.get("session_id")
    if not transcript or not session:
        sys.exit("kyb: --transcript and --session (or a hook event on stdin) are required")
    with workspace_action(ws):
        out = display.ingest_session(ws, session, transcript)
    print(json.dumps(out))
    return 1 if out["unverified_answer"] else 0
