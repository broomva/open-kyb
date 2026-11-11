"""`ask`: the verified evidence bundle for a question (agent-driven-review §5).

Facts and verified claims come first, then BM25-ranked witnessed snippets. Nothing matches, the
bundle is empty, and the only honest answer is NOT_IN_DOCUMENTS. `ask` is a read: it ranks what
the record already holds and makes no model call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kyb.copilot import lexicon
from kyb.copilot.universe import Fact, Snippet, decisions, entity_stems, load_universe
from kyb.util.fsutil import safe_id
from kyb.workspace.runs import latest_run

LIMIT = 12
HOW = ("Answer only through record_answer, with fact (cite fact_id), snippet (cite snippet_id), "
       "explain (decision_id) or NOT_IN_DOCUMENTS statements. Code renders the text; show it "
       "verbatim and add no factual prose.")  # fmt: skip


def resolve_run(root: Path, case: str, run: str | None) -> str:
    run = run or latest_run(root, safe_id(case, "case id"))
    if run is None:
        raise LookupError(f"{case} has no run yet; run `kyb pages {case}` first")
    return run


def _ranked[T](query: list[str], items: list[T], text, where, limit: int,
               entity: frozenset[str]) -> list[tuple[T, float]]:  # fmt: skip
    """`query` is the question's stems; `text` is what an item says and `where` the document it
    comes from. A document's name can make an item a candidate, but ranks below content."""
    body = [lexicon.stems(text(i)) for i in items]
    named = [lexicon.stems(where(i)) for i in items]
    scores = lexicon.concept_bm25(query, body)
    place = lexicon.concept_bm25(query, named)
    hits = []
    for i, b, n, s, w in zip(items, body, named, scores, place, strict=True):
        if lexicon.concept_enough(*lexicon.concept_matches(query, [*b, *n], entity)):
            hits.append((i, round(s + lexicon.WHERE_WEIGHT * w, 6)))
    return sorted((h for h in hits if h[1] > 0), key=lambda h: -h[1])[:limit]


def fact_view(f: Fact, score: float | None = None) -> dict[str, Any]:
    view = {"fact_id": f.fact_id, "subject": f.subjects[0], "attribute": f.attribute,
            "value": f.value, "note": f.note, "cites": [c.view() for c in f.cites]}  # fmt: skip
    return view if score is None else {**view, "score": score}


def ask(root: Path, case: str, question: str, run: str | None = None,
        limit: int = LIMIT) -> dict[str, Any]:  # fmt: skip
    case, run = safe_id(case, "case id"), resolve_run(root, case, run)
    u = load_universe(root, case, run)
    query, names = lexicon.stems(question, query=True), entity_stems(u)
    facts = _ranked(query, sorted(u.facts.values(), key=lambda f: f.fact_id),
                    Fact.index_text, Fact.index_where, limit, names)  # fmt: skip
    snippets = _ranked(query, sorted(u.snippets.values(), key=lambda s: s.snippet_id),
                       Snippet.index_text, Snippet.index_where, limit, names)  # fmt: skip
    return {
        "case": case, "run": run, "question": question,
        "terms": sorted(set(lexicon.terms(question))),
        "facts": [fact_view(f, s) for f, s in facts],
        "snippets": [{**x.view(), "score": s} for x, s in snippets],
        "decisions": decisions(root, case)[:3],
        "empty": not facts and not snippets,
        "integrity": u.integrity,
        "notes": u.notes,
        "how_to_answer": HOW,
    }  # fmt: skip


def snippet_view(s: Snippet) -> dict[str, Any]:
    return s.view()


def ask_witnessed(root: Path, case: str, question: str, run: str | None, witness) -> dict[str, Any]:
    """`ask`, first having the witness read the text-layer lines the question points at (the
    recorded, replayable `verify_quotes` kind of call). With no witness available those lines
    are not offered, and the bundle says so. Call under `workspace_action`."""
    from kyb.copilot.spans import witness_spans
    from kyb.witness.errors import WitnessRefused

    case, run = safe_id(case, "case id"), resolve_run(root, case, run)
    note = None
    try:
        read = witness_spans(root, case, run, question, witness)
        if read.get("refused"):
            note = "text-layer lines of some documents were not witnessed: " + "; ".join(
                f"{doc}: {why}" for doc, why in sorted(read["refused"].items())
            )
    except (LookupError, WitnessRefused) as exc:  # no recorded answer, no witness, or no routing
        note = f"text-layer lines were not witnessed, so they are not offered: {exc}"
    out = ask(root, case, question, run)
    return {**out, "notes": [*out["notes"], note] if note else out["notes"]}
