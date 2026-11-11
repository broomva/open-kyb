"""`record_answer`: check every statement against the record, render the text, record the outcome.

The agent supplies structure; code supplies the words. A fact must be a recorded fact (or verified
claim) with the same subject and attribute and an equal value under norm-v2. A snippet's value
must occur verbatim in the witnessed snippet it cites. An explanation is rendered from the
recorded decision. NOT_IN_DOCUMENTS is accepted only when nothing in the record covers the topic.
One failing statement rejects the whole answer, and the reason is recorded with it: the officer
sees nothing the record does not support (invariant 7).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from kyb.copilot import lexicon
from kyb.copilot.bundle import resolve_run
from kyb.copilot.render import (
    render_answer,
    render_explain,
    render_fact,
    render_missing,
    render_snippet,
)
from kyb.copilot.statements import (
    MAX_STATEMENTS,
    ExplainStatement,
    FactStatement,
    NotInDocuments,
    SnippetStatement,
    Statement,
)
from kyb.copilot.universe import Universe, decisions, load_universe
from kyb.signing.decisions import load_decision, signature
from kyb.signing.ledger import write_record
from kyb.util.fsutil import safe_id
from kyb.util.hashing import sha256_bytes, sha256_json
from kyb.witness.norm import contains, norm
from kyb.workspace.runs import run_meta

QA_REL = "ledger/qa/{answer_id}.json"
DECISION_WORDS = frozenset({"decision", "status", "rule", "rules", "score", "risk", "approved",
                            "rejected", "manual", "review", "signed", "signature", "override",
                            "why"})  # fmt: skip
_STATEMENTS = TypeAdapter(list[Statement])


class Rejected(ValueError):
    """One statement fails its check; the message is the recorded reason."""


def _check_fact(u: Universe, s: FactStatement) -> str:
    facts = []
    for cid in dict.fromkeys(s.cites):
        f = u.facts.get(cid)
        if f is None:
            raise Rejected(f"{cid!r} is not a recorded fact or verified claim of {u.case}")
        if norm(s.subject) not in {norm(x) for x in f.subjects}:
            raise Rejected(f"{cid}: subject {s.subject!r} is not {f.subjects[0]!r}")
        if norm(s.attribute) != norm(f.attribute):
            raise Rejected(f"{cid}: attribute {s.attribute!r} is not {f.attribute!r}")
        if norm(s.value) != norm(f.value):
            raise Rejected(f"{cid}: value {s.value!r} is not the recorded {f.value!r}")
        facts.append(f)
    return render_fact(facts)


def _check_snippet(u: Universe, s: SnippetStatement) -> str:
    snippets = []
    for sid in dict.fromkeys(s.cites):
        sn = u.snippets.get(sid)
        if sn is None:
            raise Rejected(f"{sid!r} is not a witnessed snippet of {u.case}")
        if not contains(sn.text, s.value):
            raise Rejected(f"{sid}: {s.value!r} does not occur in the witnessed text "
                           f"{sn.text!r}")  # fmt: skip
        snippets.append(sn)
    return render_snippet(s.question_attribute, s.value, snippets)


def _check_explain(root: Path, u: Universe, s: ExplainStatement) -> str:
    try:
        d = load_decision(root, safe_id(s.decision_id, "decision id"))
    except (LookupError, PermissionError, ValueError) as exc:
        raise Rejected(f"decision {s.decision_id!r} is not recorded ({exc})") from exc
    if d.case != u.case:
        raise Rejected(f"decision {s.decision_id} is for {d.case}, not {u.case}")
    unknown = sorted(set(s.rule_ids) - {r.rule_id for r in d.rule_results})
    if unknown:
        raise Rejected(f"decision {s.decision_id} has no rule {', '.join(unknown)}")
    return render_explain(d, s.rule_ids, signature(root, d.decision_id))


def _check_missing(root: Path, u: Universe, s: NotInDocuments) -> str:
    topic = lexicon.terms(s.topic)
    if not topic:
        raise Rejected("name the topic that is missing, in the documents' own words")
    wanted = len(set(topic))
    covered = [f.fact_id for f in u.facts.values()
               if lexicon.covers(topic, lexicon.terms(f.text())) == wanted]  # fmt: skip
    covered += [x.snippet_id for x in u.snippets.values()
                if lexicon.covers(topic, lexicon.terms(x.text)) == wanted]  # fmt: skip
    if covered:
        raise Rejected(f"the record covers {s.topic!r}: {', '.join(sorted(covered)[:5])}")
    if DECISION_WORDS & set(topic) and decisions(root, u.case):
        raise Rejected(f"{s.topic!r} is about a recorded decision: answer with an explain "
                       "statement")  # fmt: skip
    return render_missing(s.topic)


def _check(root: Path, u: Universe, s: Statement) -> str:
    if isinstance(s, FactStatement):
        return _check_fact(u, s)
    if isinstance(s, SnippetStatement):
        return _check_snippet(u, s)
    if isinstance(s, ExplainStatement):
        return _check_explain(root, u, s)
    return _check_missing(root, u, s)


def record_answer(root: Path, case: str, question: str, statements: list[dict[str, Any]],
                  run: str | None = None) -> dict[str, Any]:  # fmt: skip
    """Check, render and record an answer. Call under `workspace_action`. A rejected answer is
    recorded too (with its reasons) and returned with `accepted: false` and no text."""
    case, run = safe_id(case, "case id"), resolve_run(root, case, run)
    u = load_universe(root, case, run)
    results, parsed = [], []
    try:
        parsed = _STATEMENTS.validate_python(statements)
        problem = None if 0 < len(parsed) <= MAX_STATEMENTS else (
            f"an answer has 1 to {MAX_STATEMENTS} statements (got {len(parsed)})")  # fmt: skip
    except ValidationError as exc:
        problem = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
    if problem:
        results = [{"index": None, "type": None, "ok": False, "reason": problem}]
    else:
        for i, s in enumerate(parsed):
            try:
                results.append({"index": i, "type": s.type, "ok": True, "text": _check(root, u, s)})
            except Rejected as exc:
                results.append({"index": i, "type": s.type, "ok": False, "reason": str(exc)})
    accepted = bool(results) and all(r["ok"] for r in results)
    answer_id = "a-" + sha256_json({"case": case, "run": run, "question": question,
                                    "statements": statements})[7:19]  # fmt: skip
    text = render_answer([r["text"] for r in results], answer_id) if accepted else None
    record = {"answer_id": answer_id, "case": case, "run": run, "question": question,
              "statements": statements, "accepted": accepted, "results": results, "text": text,
              "text_sha256": sha256_bytes(text.encode()) if text else None}  # fmt: skip
    write_record(root, QA_REL.format(answer_id=answer_id), record, action="record_answer",
                 tier="derived", as_of=u.as_of or run_meta(root, case, run)["as_of"], inputs=[],
                 detail={"case": case, "run": run, "answer_id": answer_id,
                         "accepted": accepted, "text_sha256": record["text_sha256"]})  # fmt: skip
    return {"answer_id": answer_id, "accepted": accepted, "text": text,
            "rejected": [{k: r[k] for k in ("index", "type", "reason")}
                         for r in results if not r["ok"]]}  # fmt: skip
