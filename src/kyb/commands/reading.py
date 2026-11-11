"""Reading (spec): `kyb read plan`, `kyb claims submit|verify`, `kyb layout`, `kyb review`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

import typer

from kyb.commands._common import root, run
lazy from kyb.audit.chain import workspace_action
lazy from kyb.read.plan import read_plan
lazy from kyb.read.status import review_status
lazy from kyb.read.submit import SubmissionRejected, submit_claims
lazy from kyb.util.fsutil import safe_id
lazy from kyb.witness import port

app = typer.Typer()
read = typer.Typer(name="read", help="reading (spec)")
claims = typer.Typer(name="claims", help="claims (spec)")
app.add_typer(read)
app.add_typer(claims)

Run = Annotated[str | None, typer.Option("--run")]
WitnessMode = Annotated[Literal["auto", "replay", "live"], typer.Option("--witness")]


@read.command("plan", help="the reader work order (needs pages and layout)")
def read_plan_cmd(
    ctx: typer.Context,
    case: str,
    run_: Annotated[str | None, typer.Option("--run")] = None,
    doc: Annotated[str | None, typer.Option("--doc")] = None,
) -> None:
    ws = root(ctx)
    case_id = safe_id(case, "case id")
    run_id = run(ws, case_id, run_)
    entries = read_plan(ws, case_id, run_id, doc)
    for e in entries:
        from kyb.loop.action import get_repair_targets

        targets = get_repair_targets(ws, case_id, run_id, e["doc"])
        if targets:
            e["repair_targets"] = targets
    print(json.dumps(entries, indent=2))


@claims.command("submit", help="read_document: submit one document's claims (JSON file)")
def claims_submit(
    ctx: typer.Context,
    case: str,
    doc: Annotated[str, typer.Option("--doc")],
    file: Annotated[str, typer.Option("--file")],
    run_: Run = None,
) -> int:
    ws = root(ctx)
    case_id = safe_id(case, "case id")
    body = json.loads(Path(file).read_text("utf-8"))
    payload = {"case": case_id, "run": run(ws, case_id, run_), "doc": doc,
               "doc_sha256": body.get("doc_sha256"), "claims": body.get("claims")}  # fmt: skip
    try:
        with workspace_action(ws):
            result = submit_claims(ws, payload)
    except SubmissionRejected as exc:
        print(json.dumps({"rejected": exc.errors}, indent=2))
        return 1
    print(json.dumps(result, indent=2))
    return 0


@claims.command("verify", help="verify_quotes: witness every claim from pixels (blind)")
def claims_verify(ctx: typer.Context, case: str, run_: Run = None,
                  witness: WitnessMode = "auto") -> None:  # fmt: skip
    from kyb.witness.verify import verify_case

    ws = root(ctx)
    case_id = safe_id(case, "case id")
    with workspace_action(ws):
        summary = verify_case(ws, case_id, run(ws, case_id, run_), port.make_witness(ws, witness))
    print(json.dumps(summary, indent=2))


@app.command(help="layout_pages: the claim-blind layout pass (the witness)")
def layout(ctx: typer.Context, case: str, run_: Run = None,
           witness: WitnessMode = "auto") -> None:  # fmt: skip
    from kyb.witness.layout import layout_case

    ws = root(ctx)
    case_id = safe_id(case, "case id")
    with workspace_action(ws):
        done = layout_case(ws, case_id, run(ws, case_id, run_), port.make_witness(ws, witness))
    print(json.dumps({"laid_out": done}, indent=2))


@app.command(help="kyb-review workflow state")
def review(ctx: typer.Context, case: str,
           status: Annotated[bool, typer.Option("--status")]) -> None:  # fmt: skip
    print(json.dumps(review_status(root(ctx), case), indent=2))
