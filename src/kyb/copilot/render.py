"""The words of an answer. Only code writes them, from recorded fields and verbatim excerpts.

A rendered answer ends with its marker line, so the display check can find it in what the
harness showed, byte for byte.
"""

from __future__ import annotations

from kyb.contracts.b4 import DECISION_REL, DecisionRecord
from kyb.copilot.universe import Cite, Fact, Snippet

MARK = "[kyb-answer {answer_id}]"
SHOWN = 2  # excerpts shown per cited fact


def _quote(cites: list[Cite] | tuple[Cite, ...]) -> list[str]:
    seen, out = set(), []
    for c in cites:
        key = (c.document_name, c.page, c.excerpt)
        if key not in seen and len(seen) < SHOWN:
            seen.add(key)
            out.append(f'    ↳ {c.document_name} p.{c.page}: "{c.excerpt}"')
    return out


def render_fact(facts: list[Fact]) -> str:
    f = facts[0]
    head = f"• {f.subjects[0]} — {f.attribute}: {f.value}" + (f" ({f.note})" if f.note else "")
    return "\n".join([head, *_quote([c for x in facts for c in x.cites])])


def render_snippet(attribute: str, value: str, snippets: list[Snippet]) -> str:
    lines = [f"• {attribute}: {value}"]
    for s in snippets:
        lines.append(f'    ↳ {s.document_name} p.{s.page} (read from the page image): "{s.text}"')
    return "\n".join(lines)


def render_missing(topic: str) -> str:
    return f"• Not in the documents: {topic}."


def _evidence(d: DecisionRecord, rule_id: str) -> list[str]:
    cites = [c for v in d.kyb_decision.violations if v.rule_id == rule_id for c in v.evidence]
    return [f'      ↳ {c.document_id} {c.locator}: "{c.excerpt}"' for c in cites[:SHOWN]]


def render_explain(d: DecisionRecord, rule_ids: list[str], sig: dict | None) -> str:
    k = d.kyb_decision
    state = f"signed by {sig['officer']} ({sig['verdict']})" if sig else "pending signature"
    lines = [
        f"• Decision {d.decision_id} for {k.entity_name}: {k.status}, "
        f"risk score {k.risk_score}, {state}"
    ]

    if sig and sig["verdict"] == "override":
        lines.append(
            f"  - the officer overrode the recommendation ({sig['from_status']}) to "
            f"{sig['to_status']}: {' '.join(sig['rationale'].split())}"
        )

    if k.officer_overrides_considered:
        lines.append(f"  - officer corrections considered: {len(k.officer_overrides_considered)}")

    for j in d.judge_outcomes:
        if j.verdict == "veto":
            reason = (j.reason[:77] + "...") if len(j.reason) > 80 else j.reason
            lines.append(f"  - judge {j.judge_id} vetoed {j.target}: {reason}")

    if rule_ids:
        for r in (r for r in d.rule_results if r.rule_id in rule_ids):
            if r.fired == "yes":
                desc = (r.description[:157] + "...") if len(r.description) > 160 else r.description
                lines.append(f"  - {r.rule_id} [{r.severity}] fired: {desc}")
            elif r.fired == "no":
                lines.append(f"  - {r.rule_id} did not fire")
            elif r.fired == "unknown":
                lines.append(f"  - {r.rule_id} could not be established")
    else:
        fired = [r for r in d.rule_results if r.fired == "yes"]
        unknown = [r for r in d.rule_results if r.fired == "unknown"]

        for r in fired:
            desc = (r.description[:157] + "...") if len(r.description) > 160 else r.description
            lines.append(f"  - {r.rule_id} [{r.severity}] fired: {desc}")

    cites = [c for v in k.violations for c in v.evidence]
    for c in cites[:5]:
        excerpt = (c.excerpt[:77] + "...") if len(c.excerpt) > 80 else c.excerpt
        lines.append(f'  - cite: {c.document_id}, {c.locator}: "{excerpt}"')

    if not rule_ids:
        unknown = [r for r in d.rule_results if r.fired == "unknown"]
        if unknown:
            first_few = ", ".join(r.rule_id for r in unknown[:3])
            lines.append(f"  - unknown rules: {len(unknown)} (e.g. {first_few})")

    ledger_path = DECISION_REL.format(decision_id=d.decision_id)
    lines.append(f"full record: {ledger_path}")

    return "\n".join(lines)


def render_answer(blocks: list[str], answer_id: str) -> str:
    return "\n".join([*blocks, MARK.format(answer_id=answer_id)])
