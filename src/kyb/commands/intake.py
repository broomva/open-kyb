"""Intake: `kyb init`, `kyb ingest`, `kyb pages` (top-level commands)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer

from kyb.commands._common import root, terminal_token
lazy from kyb.audit.chain import workspace_action
lazy from kyb.docintel.ingest import ingest_packet
lazy from kyb.docintel.pages import build_pages, verified_manifest
lazy from kyb.util.fsutil import safe_id
lazy from kyb.workspace.init import init_workspace
lazy from kyb.workspace.runs import run_for_pages, run_meta

app = typer.Typer()


@app.command(help="materialise a tenant workspace from the template")
def init(
    path: str,
    tenant: Annotated[str, typer.Option("--tenant")],
    policy: Annotated[str, typer.Option("--policy", help="path to compliance_rules.md")],
    force: Annotated[bool, typer.Option("--force")] = False,
    officer: Annotated[
        str | None,
        typer.Option("--officer", help="re-init that changes the policy: the approving officer"),
    ] = None,
) -> None:
    token = terminal_token(officer, "re-initialise") if officer else None
    lock = init_workspace(Path(path), tenant=tenant, policy_src=Path(policy), force=force,
                          officer=officer, officer_token=token)  # fmt: skip
    print(json.dumps(lock, indent=2))


@app.command(help="copy a packet into bronze and freeze as_of")
def ingest(
    ctx: typer.Context,
    packet: str,
    as_of: Annotated[str, typer.Option("--as-of", help="YYYY-MM-DD, frozen at intake (spec)")],
    case: Annotated[str | None, typer.Option("--case")] = None,
) -> None:
    ws = root(ctx)
    with workspace_action(ws):
        m = ingest_packet(ws, Path(packet), as_of=as_of, case=case)
    print(f"ingested {m['packet_id']}: {len(m['files'])} files, as_of {m['as_of']}")
    for f in m["files"]:
        layer = "text layer" if f["text_layer"] else "scan (no text layer)"
        print(f"  {f['name']:<40} {f['pages']}p  {layer}")


@app.command(help="step 2: render pages and capture text layers for a run")
def pages(
    ctx: typer.Context,
    case: str,
    run: Annotated[str | None, typer.Option("--run")] = None,
    new_run: Annotated[
        bool, typer.Option("--new-run", help="start a fresh run (inputs changed since the last)")
    ] = False,
) -> None:
    ws = root(ctx)
    case_id = safe_id(case, "case id")
    if run and new_run:
        sys.exit("kyb: --run and --new-run are exclusive")
    with workspace_action(ws):
        manifest = verified_manifest(ws, case_id)
        run_id = run or run_for_pages(ws, case_id, as_of=manifest["as_of"], new_run=new_run)
        run_meta(ws, case_id, run_id)  # validates the id and fails loudly for an unknown run
        records = build_pages(ws, case_id, run_id)
    scans = sum(1 for r in records if r["text_layer"] is None)
    print(f"{case} {run_id}: {len(records)} pages ({scans} without a text layer)")
