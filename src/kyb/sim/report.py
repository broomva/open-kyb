"""report.json / report.md for a simulation run. Case and arrival ids only: no names."""

from __future__ import annotations

import json
from pathlib import Path

from kyb.sim.engine import SimResult, by_week, totals


def build_report(res: SimResult, run: str) -> dict:
    return {
        "run": run,
        "decider": res.decider,
        "officer": res.officer,
        "config": res.config.model_dump(),
        "headline": {
            "false_approvals": totals(res)["false_approvals"],
            "definition": "auto-finalised or Jev APPROVED where the gold label is not APPROVED",
        },
        "totals": totals(res),
        "weeks": by_week(res),
        "profiles": res.profiles,
        "precedents": [p.model_dump() for p in res.precedents],
        "outcomes": [o.model_dump() for o in res.outcomes],
    }


def chart(weeks: list[dict]) -> str:
    xs = ", ".join(f"w{w['week']}" for w in weeks)
    rate = ", ".join(f"{100 * w['auto_rate']:.0f}" for w in weeks)
    work = ", ".join(str(w["officer_blocking"] + w["officer_audits"]) for w in weeks)
    top = max([100, *(w["officer_blocking"] + w["officer_audits"] for w in weeks)])
    return (
        '```mermaid\nxychart-beta\n    title "Auto-finalised % (line) and officer reviews '
        '(bars) by week"\n'
        f'    x-axis [{xs}]\n    y-axis "% / reviews" 0 --> {top}\n'
        f"    bar [{work}]\n    line [{rate}]\n```"
    )


def markdown(rep: dict) -> str:
    t, w = rep["totals"], rep["weeks"]
    cfg = rep["config"]
    pct = lambda x: "—" if x is None else f"{100 * x:.0f}%"  # noqa: E731
    lines = [
        f"# Autonomy simulation: `{rep['run']}`",
        "",
        f"**Headline — false approvals: {t['false_approvals']}** "
        f"(auto-finalised or Jev APPROVED where gold is not APPROVED).",
        "",
        f"- decider `{rep['decider']}`, officer `{rep['officer']}`; {cfg['weeks']} weeks × about "
        f"{cfg['per_week']} arrivals, K={cfg['k']}, audit rate {cfg['audit_rate']}, "
        f"seed {cfg['seed']}",
        f"- arrivals {t['arrivals']}; auto-finalised {t['auto_finalised']} "
        f"({pct(t['auto_rate'])}); "
        f"REJECTED auto-finalised {t['rejected_auto_finalised']}",
        f"- officer: {t['officer_blocking']} blocking reviews, {t['officer_audits']} audits, "
        f"{t['officer_hours']} h; agreement {pct(t['agreement_rate'])}; overrides {t['overrides']}",
        f"- false rejects {t['false_rejects']}; demotions {t['demotions']}; "
        f"accuracy vs gold {pct(t['accuracy_vs_gold'])}",
        "",
        "## By week",
        "",
        "| week | arrivals | **false approvals** | auto-finalised | auto rate | blocking | audits "
        "| hours | false rejects | agreement | demotions | promotions |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in w:
        lines.append(
            f"| {r['week']} | {r['arrivals']} | **{r['false_approvals']}** | "
            f"{r['auto_finalised']} | {pct(r['auto_rate'])} | {r['officer_blocking']} | "
            f"{r['officer_audits']} | {r['officer_hours']} | {r['false_rejects']} | "
            f"{pct(r['agreement_rate'])} | {r['demotions']} | {r['promotions']} |"
        )
    lines += ["", chart(w), "", "## Time to promote", ""]
    ttp = t["time_to_promote_weeks"]
    if ttp:
        lines += ["| profile | to level | weeks from first seen | week |", "|---|---:|---:|---:|"]
        for k, rows in sorted(ttp.items()):
            lines += [f"| `{k}` | {r['to_level']} | {r['weeks']} | {r['week']} |" for r in rows]
    else:
        lines.append("No profile reached the promotion threshold in this run.")
    lines += ["", "## Precedents (officer overrides, context only)", ""]
    if rep["precedents"]:
        lines += ["| id | week | profile | case | from → to |", "|---|---:|---|---|---|"]
        lines += [
            f"| {p['id']} | {p['week']} | `{p['profile']}` | {p['case_id']} | "
            f"{p['from_status']} → {p['to_status']} |"
            for p in rep["precedents"]
        ]
    else:
        lines.append("None recorded.")
    if t["false_approval_cases"]:
        lines += ["", "## False approvals", "", "Arrivals: " + ", ".join(t["false_approval_cases"])]
    return "\n".join(lines) + "\n"


def write(res: SimResult, run: str, root: Path) -> Path:
    out = root / run
    out.mkdir(parents=True, exist_ok=True)
    rep = build_report(res, run)
    (out / "report.json").write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    (out / "report.md").write_text(markdown(rep))
    return out
