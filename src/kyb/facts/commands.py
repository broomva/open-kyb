"""Facts and screening (L2): `kyb facts <case>` and `kyb screen <case>` (spec sub-app)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer

from kyb.commands._common import root, run
lazy from kyb.audit.chain import workspace_action
lazy from kyb.facts.action import derive_facts
lazy from kyb.screening.action import default_watchlist, screen_case
lazy from kyb.util.fsutil import safe_id

app = typer.Typer()

Run = Annotated[str | None, typer.Option("--run")]


@app.command(help="derive_facts: verified claims → FactSet (gold/facts/<case>/<run>.json)")
def facts(ctx: typer.Context, case: str, run_: Run = None) -> None:
    ws = root(ctx)
    case_id = safe_id(case, "case id")
    with workspace_action(ws):
        fs, rel, written = derive_facts(ws, case_id, run(ws, case_id, run_))
    print(json.dumps({"facts": rel, "written": written, "fact_set_hash": fs.fact_set_hash,
                      "applicant": fs.applicant, "entities": len(fs.entities),
                      "holdings": len(fs.holdings), "conflicts": [
                          {"id": x.conflict_id, "subject": x.subject, "attribute": x.attribute,
                           "values": [v.value for v in x.values]} for x in fs.conflicts],
                      "operations": {k: v.state for k, v in fs.operations},
                      "unverified": len(fs.unverified)}, indent=2, ensure_ascii=False))  # fmt: skip


@app.command(
    help="screen: FactSet + watchlist → ScreeningResult (gold/screening/<case>/<run>.json)"
)
def screen(
    ctx: typer.Context,
    case: str,
    run_: Run = None,
    watchlist: Annotated[
        str | None,
        typer.Option("--watchlist", help="watchlists.json (default: policy/ in the workspace)"),
    ] = None,
) -> None:
    ws = root(ctx)
    case_id = safe_id(case, "case id")
    path = Path(watchlist) if watchlist else default_watchlist(ws)
    if path is None:
        sys.exit("kyb: no watchlist in the workspace; pass --watchlist <watchlists.json>")
    with workspace_action(ws):
        result, rel, written = screen_case(ws, case_id, run(ws, case_id, run_), path)
    print(json.dumps({"screening": rel, "written": written, "method": result.method,
                      "watchlist_version": result.watchlist_version, "candidates": [
                          {"id": c.candidate_id, "entity": c.entity_id, "entry": c.entry.entry_id,
                           "matched_name": c.matched_name, "jw_score": c.jw_score,
                           "alias_exact": c.alias_exact, "triggers_fuzzy": c.triggers_fuzzy,
                           "corroborating": c.identifiers.corroborating,
                           "conflicting": c.identifiers.conflicting}
                          for c in result.candidates]}, indent=2, ensure_ascii=False))  # fmt: skip
