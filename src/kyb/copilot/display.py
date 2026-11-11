"""The display check (agent-driven-review §5): what the officer was shown, captured and flagged.

At every Stop and at SessionEnd, `ingest_session` reads the assistant messages the harness wrote
since the last cursor. A message that states a fact about a processed entity must carry a rendered
answer from its turn, byte for byte; any other factual sentence about an entity is flagged
`unverified_answer` in the audit chain, and the release gate wants zero of them.

The limit is stated, not hidden: inside a harness the officer's screen belongs to the harness.
This captures, checks and flags what was shown. It cannot stop one display; the hosted chat API,
which renders only `record_answer` output, is the hard boundary.

Imports are light on purpose: the Stop hook runs this on every turn (spec).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

lazy from kyb.audit.chain import AuditLog, chain_events, workspace_action
lazy from kyb.copilot.lexicon import fold
lazy from kyb.util.hashing import sha256_json
lazy from kyb.workspace.catalog import Catalog

CURSOR = "harness.session_ingest"
FLAG = "harness.unverified_answer"
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
MIN_TERM = 4  # an alias or identifier shorter than this (a bare "SA") is too common to mean one
MIN_NAME = 3  # an entity's own name may be short ("Jev", "UBS")


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(b.get("text", "") for b in content or [] if isinstance(b, dict)
                     and b.get("type") == "text")  # fmt: skip


def _is_prompt(rec: dict) -> bool:
    """A real user turn: not the harness feeding a tool result back."""
    content = (rec.get("message") or {}).get("content")
    if isinstance(content, str):
        return True
    return not any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content or [])


def _result_text(block: dict) -> str:
    content = block.get("content")
    return content if isinstance(content, str) else _text(content)


def watched_terms(root: Path) -> set[str]:
    """Names and identifiers of every processed entity, folded: what makes a sentence a statement
    about one. A primary name counts from three letters; aliases and identifiers from four."""
    primary: set[str] = set()
    secondary: set[str] = set()
    for rel in Catalog(root).entries():
        path = root / rel
        if rel.startswith("gold/facts/") and rel.endswith(".json"):
            for e in json.loads(path.read_text("utf-8")).get("entities", []):
                names = e.get("names", [])
                primary.update(names[:1])
                secondary.update(names[1:])
                for values in (e.get("identifiers") or {}).values():
                    secondary.update(values)
        elif rel.startswith("ledger/decisions/") and rel.endswith(".json"):
            primary.add(json.loads(path.read_text("utf-8"))["kyb_decision"]["entity_name"])
    folded = {fold(t).strip() for t in primary}
    folded |= {
        re.sub(r"\b(pte|ltd|inc|llc|spa|sl|sa|sas|bv|ag)\b\.?", "", t).strip(" .,")
        for t in list(folded)
    }
    return {t for t in folded if len(t) >= MIN_NAME} | {
        t for t in (fold(x).strip() for x in secondary) if len(t) >= MIN_TERM
    }


def _answer_text(root: Path, answer_id: str) -> str | None:
    rel = f"ledger/qa/{answer_id}.json"
    if not re.fullmatch(r"a-[0-9a-f]{12}", answer_id) or Catalog(root).get(rel) is None:
        return None
    rec = json.loads((root / rel).read_text("utf-8"))
    return rec["text"] if rec.get("accepted") else None


def _cursor(root: Path, session: str, path: str) -> dict:
    last = {}
    for e in chain_events(root):
        d = e["detail"]
        if e["action"] == CURSOR and d.get("session") == session and d.get("transcript") == path:
            last = d
    return last


def check_message(text: str, watch: set[str], rendered: list[str]) -> dict | None:
    """None if the message is clean; else why it is flagged. Rendered answers are removed byte for
    byte first, so only what is left over can be an unverified statement."""
    rest = text
    for block in rendered:
        rest = rest.replace(block, "")
    sentences = [s.strip() for s in _SENTENCE.split(rest) if s.strip()]
    hits = [s for s in sentences if any(t in fold(s) for t in watch)]
    if not hits:
        return None
    reason = (
        "a statement about a processed entity, with no rendered answer in this turn"
        if not rendered
        else "factual text about an entity beyond the rendered answer"
    )
    return {"reason": reason, "sentences": len(hits),
            "sha256": [sha256_json(s) for s in hits[:5]]}  # fmt: skip


def ingest_session(root: Path, session: str, transcript: str) -> dict[str, int]:
    """Read the transcript from the stored cursor; flag unverified answers; move the cursor.
    Call under `workspace_action`."""
    path = Path(transcript)
    if path.suffix != ".jsonl" or not path.is_file():
        return {"messages": 0, "unverified_answer": 0}
    cur = _cursor(root, session, str(path))
    offset, turn = int(cur.get("offset", 0)), list(cur.get("turn_answers", []))
    raw = path.read_bytes()[offset:]
    end = raw.rfind(b"\n") + 1  # only whole lines: a half-written one waits for the next hook
    watch, log = watched_terms(root), AuditLog(root / "ledger" / "audit.jsonl")
    # tool_use id -> a record_answer call awaiting its result; kept in the cursor, because the
    # call and its result can fall on either side of a hook
    pending: dict[str, str] = dict(cur.get("pending", {}))
    messages = flagged = 0
    for n, line in enumerate(raw[:end].splitlines(), 1):
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if not isinstance(rec, dict):
            continue
        kind, content = rec.get("type"), (rec.get("message") or {}).get("content")
        if kind == "user" and _is_prompt(rec):
            turn, pending = [], {}  # a new officer prompt starts a new turn
        for b in content if isinstance(content, list) else []:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "tool_use" and str(b.get("name", "")).endswith("record_answer"):
                pending[b.get("id", "")] = "record_answer"
            elif b.get("type") == "tool_result" and b.get("tool_use_id") in pending:
                pending.pop(b["tool_use_id"], None)
                m = re.search(r'"answer_id"\s*:\s*"(a-[0-9a-f]{12})"', _result_text(b))
                if m:
                    turn.append(m[1])
        if kind == "assistant" and (text := _text(content).strip()):
            messages += 1
            rendered = [t for a in turn if (t := _answer_text(root, a))]
            if (why := check_message(text, watch, rendered)) is not None:
                flagged += 1
                log.append("harness", FLAG, input_hash=sha256_json(text),
                           output_hash=sha256_json(why),
                           detail={"session": session, "transcript": str(path),
                                   "line": cur.get("lines", 0) + n, "answers": turn,
                                   **why})  # fmt: skip
    lines = cur.get("lines", 0) + len(raw[:end].splitlines())
    log.append("harness", CURSOR, input_hash=sha256_json([session, str(path)]),
               output_hash=sha256_json([offset + end, flagged]),
               detail={"session": session, "transcript": str(path), "offset": offset + end,
                       "lines": lines, "turn_answers": turn, "pending": pending,
                       "messages": messages, "flagged": flagged})  # fmt: skip
    return {"messages": messages, "unverified_answer": flagged}


def run(event: dict[str, Any]) -> dict[str, int] | None:
    """The hook form: the event names the session and its transcript. Never blocks the harness."""
    from kyb.workspace import guard

    root = guard.find_root(guard.session_start_dir(event))
    session, transcript = event.get("session_id") or "", event.get("transcript_path") or ""
    if root is None or not session or not transcript:
        return None
    with workspace_action(root):
        return ingest_session(root, session, transcript)
