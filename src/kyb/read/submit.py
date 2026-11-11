"""`read_document`: a reader submits one document's claims (spec step 7).

Everything here is enforced before anything is written: the schema, the work-order scope, page
and bbox bounds, one layout block per crop, text-layer sources, and limits. Identity comes from
the harness's hook attestation (kyb.read.harness); a submission that is not attested is still
recorded, as `unattested`, and the judges cap any decision that consumes it.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from kyb.audit.chain import AuditLog
from kyb.read.harness import READER_TYPES, attest, split_token
from kyb.read.plan import load_layout, read_plan
from kyb.read.schema import Claim, ClaimsSubmission, inside
from kyb.util.fsutil import contained, publish_exclusive
from kyb.util.hashing import sha256_bytes, sha256_json
from kyb.witness.bind import _CUR, _part_value, components, value_text
from kyb.witness.norm import VERSION as NORMALIZATION
from kyb.witness.norm import contains, norm
from kyb.workspace.catalog import Catalog
from kyb.workspace.runs import run_meta

TOOL = "read_document"


SubmissionError = str | dict[str, Any]


class SubmissionRejected(ValueError):
    def __init__(self, errors: list[SubmissionError]) -> None:
        super().__init__("; ".join(e if isinstance(e, str) else e["message"] for e in errors))
        self.errors = errors


def claims_rel(case: str, run: str, doc: str) -> str:
    return f"silver/claims/{case}/{run}/{doc}.json"


# --- quantity literals ---------------------------------------------------------------------------
#
# A typed numeric component is compared only with a COMPLETE literal of the normalised excerpt,
# never with a piece of one. The grammar, with `_` for optional whitespace:
#
#     literal := (sign _)* ( "(" _ literal _ ")" | core ) (_ unit)*
#     core    := currency _ number | number (_ unit)?
#     sign    := "+" | a dash (`_dash`; after norm-v2 only `-` is a minus)
#     unit    := "%" | acciones | shares | acc. | currency
#
# `_quantities` consumes all of it before anything is compared (every sign chain outside or
# inside the wrappers, every wrapper, the number and its unit), together with every further unit
# marker printed in or after it. The parser owns each span it starts: scanning resumes after the
# literal's whole extent, valid or not, so a rescan never exposes an inner scalar (`- +(25%)` is
# one literal with two signs) and an operator always meets a complete endpoint (`10% a (25%%)`).
# Four rules then decide whether the literal may supply a component:
#
#   1. whole token: no letter, digit, dash, joiner (`_ . , / +`) or decimal continuation touches
#      it, and no sign governs it through a `(` that never closes (`-(25%`); such a sign owns
#      everything up to that parenthetical's end. A dash joins tokens on both sides as `-` does,
#      whatever its neighbour parses as (`N<U+2010>123`);
#   2. complete unit: no unit marker is left over (`MXN25%`, `(25%%)`, `25 % MXN`, `MXN -5`);
#   3. one meaning: one sign at most, `+` or `-`; money takes a sign or one level of
#      accounting parentheses, never both and never nested ones;
#   4. not a range endpoint: no range operator joins it to a compatible quantity. Each endpoint
#      is measured with the parentheticals around it, whether they were read as its own
#      wrapper or as prose (`10% a (25% de las acciones)`); standing alone it keeps its value.
#
# bind-v6 then compares a canonical spelling of the literal's own sign, unit and digits, so the
# comparator never re-reads context the lexer has already resolved. Ambiguous compounds fail
# closed: the reader gets a structured error and re-claims from the page.

_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_CURRENCY = r"(?:mxn|usd|eur|cop|us\$|\$)"
_PREFIX = re.compile(_CURRENCY)
# A unit after the number matches greedily, as before: glued to more letters (`25 accionesxyz`)
# it fails rule 1 rather than leaving a bare number behind.
_SUFFIX = re.compile(r"%|acciones|shares|acc\.|" + _CURRENCY)
# A further marker is a whole unit token: `25% coparticipación` holds no `cop`.
_MARKER = re.compile(r"%|(?:acciones|shares|mxn|usd|eur|cop)(?!\w)|acc\.|us\$|\$")
# A currency the literal after it could not consume: `MXN -5.000`, `$ (5.000)`.
_STRANDED = re.compile(r"(?:(?<!\w)(?:mxn|usd|eur|cop)|us\$|\$)\s*\Z")
# The explicit range operators of the existing typed comparator; every dash is one too.
_WORD_OPERATORS = frozenset({"a", "al", "to", "y"})
# Printed dimensions allowed for each numeric component. Bare amounts remain supported;
# equality alone never turns a percentage/count into capital or a nominal share value.
_NUMERIC_UNITS = {
    "pct": frozenset({"pct"}),
    "shares": frozenset({"", "shares"}),
    "shares_acquired": frozenset({"", "shares"}),
    "amount": frozenset({"", "money"}),
    "capital": frozenset({"", "money"}),
    "par_value": frozenset({"", "money"}),
}


def _dash(ch: str) -> bool:
    """Unicode dash punctuation (Pd), plus the other Dash/Hyphen-property code points.

    norm-v2 folds en and em dashes and U+2212 into `-`, but keeps U+2010 (also NFKC's image of
    U+2011), U+2012, U+2015, the soft hyphen and the rest; each still joins a token or a range.
    """
    return unicodedata.category(ch) == "Pd" or ch in (
        "\N{SOFT HYPHEN}\N{SWUNG DASH}\N{SUPERSCRIPT MINUS}\N{SUBSCRIPT MINUS}\N{MINUS SIGN}"
    )


def _sign(ch: str) -> bool:
    return ch == "+" or _dash(ch)


def _skip(text: str, i: int) -> int:
    while i < len(text) and text[i].isspace():
        i += 1
    return i


@dataclass(frozen=True)
class _Quantity:
    start: int  # the literal's extent in the normalised excerpt: every character it owns
    end: int
    text: str  # that raw span
    signs: str  # every explicit sign, outermost first
    wrappers: int  # balanced parentheses around the number
    number: str  # the digits as printed
    unit: str  # as printed: "", "%", "acciones", "mxn", "$", ...
    leftover: tuple[str, ...] = ()  # unit markers printed in its extent that it does not consume
    unclosed: int = 0  # wrappers a sign opened that never close (`-(25%`)
    at: int = 0  # where the number starts: the parentheticals around it are those around this

    @property
    def dimension(self) -> str:
        """`""` for an unmarked number; pct, shares or money:<canonical currency>."""
        if self.unit in _CUR:
            return "money:" + _CUR[self.unit]
        return {"": "", "%": "pct"}.get(self.unit, "shares")

    @property
    def malformed(self) -> bool:
        """No single unit or closed extent: a leftover marker, or a signed wrapper left open."""
        return bool(self.leftover or self.unclosed)


def _opens(text: str, k: int) -> bool:
    """Does the marker at `k` begin the next literal (`25% MXN 1.000`) instead of trailing one?"""
    prefix = _PREFIX.match(text, k)
    return bool(prefix and _NUMBER.match(text, _skip(text, prefix.end())))


def _trailing(text: str, end: int) -> tuple[int, list[str]]:
    """Every further whole unit marker after `end`, kept in the extent rather than dropped."""
    markers = []
    while (k := _skip(text, end)) < len(text) and (marker := _MARKER.match(text, k)):
        if _opens(text, k):
            break
        markers.append(marker.group())
        end = marker.end()
    return end, markers


def _group_end(text: str, opening: int) -> int:
    """Just past the `)` that closes the `(` at `opening`, or the excerpt's end."""
    depth = 0
    for k in range(opening, len(text)):
        depth += {"(": 1, ")": -1}.get(text[k], 0)
        if not depth:
            return k + 1
    return len(text)


def _parenthesized(text: str) -> list[tuple[int, int]]:
    """Every parenthetical as a half-open `[start, end)`: it holds position `p` when
    `start <= p < end`. An unmatched `)` closes one opened at the start of the excerpt, which may
    be a number itself (`10% de las acciones) a 25%`), and an unmatched `(` runs to its end: the
    excerpt is a quote, cut anywhere."""
    spans, opened = [], []
    for k, ch in enumerate(text):
        if ch == "(":
            opened.append(k)
        elif ch == ")":
            spans.append((opened.pop() if opened else 0, k + 1))
    return spans + [(k, len(text)) for k in opened]


def _parse(text: str, i: int) -> _Quantity | None:
    """The literal that starts at `i`, owning its whole span even when it is invalid, or None.

    Every sign chain (`- +`, spaced or not) belongs to the one literal, at every wrapper level,
    and unit markers left over inside or after a wrapper stay in its extent (`(25%%)`). A
    wrapper that never closes is owned, and so rejected, when a sign governs it: the literal
    then runs to that parenthetical's end, prose included (`-(25%% de las acciones)`). An
    unsigned unclosed `(` is prose, and the literal starts after it (`(25% de las acciones)`).
    """
    j, signs, chain, chains = i, "", "", []  # chains[k]: the signs printed before wrapper k
    openings = []  # where each of those wrappers opens
    while True:
        while j < len(text) and _sign(text[j]):
            chain, j = chain + text[j], _skip(text, j + 1)
        if not text.startswith("(", j):
            break
        chains.append(chain)
        openings.append(j)
        signs, chain, j = signs + chain, "", _skip(text, j + 1)
    signs += chain
    number = None
    if prefix := _PREFIX.match(text, j):
        unit, number = prefix.group(), _NUMBER.match(text, _skip(text, prefix.end()))
        end = number.end() if number else j
    elif number := _NUMBER.match(text, j):
        suffix = _SUFFIX.match(text, _skip(text, number.end()))
        unit, end = (suffix.group(), suffix.end()) if suffix else ("", number.end())
    if number is None:
        return None
    end, leftover = _trailing(text, end)
    closed = 0
    for _ in chains:  # the innermost wrapper closes first
        k = _skip(text, end)
        if not text.startswith(")", k):
            break
        end, more = _trailing(text, k + 1)
        leftover, closed = leftover + more, closed + 1
    unclosed = len(chains) - closed  # the outermost wrappers stay open
    if unclosed and not any(chains[:unclosed]):
        return None
    if unclosed:
        end = max(end, _group_end(text, openings[0]))
    return _Quantity(
        i, end, text[i:end], signs, closed, number.group(), unit, tuple(leftover), unclosed,
        number.start(),
    )  # fmt: skip


def _quantities(text: str) -> list[_Quantity]:
    """Every literal of `text`, left to right. Scanning resumes after a literal's whole extent,
    valid or not, so nothing inside an owned span is read again as a quantity of its own."""
    out: list[_Quantity] = []
    i = 0
    while i < len(text):
        q = _parse(text, i)
        if q is None:
            i += 1
            continue
        if stranded := _STRANDED.search(text, out[-1].end if out else 0, q.start):  # `MXN -5`
            start, marker = stranded.start(), stranded.group().strip()
            q = replace(q, start=start, text=text[start : q.end], leftover=(marker, *q.leftover))
        out.append(q)
        i = q.end
    return out


def _edges(text: str, q: _Quantity) -> tuple[bool, bool]:
    """Rule 1, lexically: is the literal free of a larger token on its left and its right?

    On the left that includes the end of another quantity or group: the dash of `10%-25 acciones`
    joins two quantities and is not the second one's sign.
    """
    before = text[q.start - 1] if q.start else " "
    after = text[q.end : q.end + 2] or " "
    left = not (before.isalnum() or before in "_.,/+%)" or _dash(before))
    right = not (after[0].isalnum() or after[0] in "_/+" or _dash(after[0]))
    return left, right and not re.match(r"[.,]\w", after)


def _admissible(text: str, q: _Quantity) -> bool:
    """Rules 1-3: may this literal supply a component at all?"""
    if not all(_edges(text, q)) or q.malformed:
        return False  # glued to a larger token, a leftover unit, or a signed wrapper left open
    if len(q.signs) > 1 or q.signs not in ("", "+", "-"):
        return False  # `- +(25%)`, `-(-25%)`, or a hyphen written where a minus would be
    money = q.dimension.startswith("money:")
    return not (money and q.wrappers and (q.signs or q.wrappers > 1))  # `-(MXN 5)`, `((MXN 5))`


def _range_members(text: str, quantities: list[_Quantity]) -> set[int]:
    """Rule 4: an operator between compatible quantities makes both endpoints inadmissible.

    A bare endpoint takes its neighbour's unit (`10 a 25%`); explicitly different units are
    independent components (`25% y 1.250 acciones`); a malformed literal has no single unit, so
    it pairs with any, and its extent is complete, so the operator beside it is the one seen
    (`10% a (25%%)`). A dash the right literal took as its sign is the binary operator when only
    whitespace precedes it (`1.000 -5.000 MXN`). Membership only rejects, so it ignores the
    outer edges: an endpoint glued to an identifier (`N-10 a 25%`) is still one.

    Each endpoint is measured with its structural extent: its own span, widened to the outermost
    parenthetical around its number that does not also hold the other's. A parenthetical read
    as prose is still part of the endpoint it holds (`10% a (25% de las acciones)`), so prose,
    a malformed unit or a group left open inside it never hides the operator beside it.
    """
    groups = _parenthesized(text)
    ranged: set[int] = set()
    for i, (left, right) in enumerate(pairwise(quantities)):
        units = {q.dimension for q in (left, right) if q.dimension and not q.malformed}
        if len(units) > 1:
            continue
        holds = [(a, b, a <= left.at < b, a <= right.at < b) for a, b in groups]
        end = max([left.end] + [b for a, b, mine, theirs in holds if mine and not theirs])
        start = min([right.start] + [a for a, b, mine, theirs in holds if theirs and not mine])
        operator = text[end:start].strip() if end < start else ""
        if not operator and end <= start < len(text) and _dash(text[start]):
            operator = text[start]
        if operator in _WORD_OPERATORS or (len(operator) == 1 and _dash(operator)):
            ranged.update((i, i + 1))
    return ranged


def _canonical(q: _Quantity) -> str:
    """bind-v6's own spelling of the literal: holdings ignore wrappers (`(25%)` is 25), and
    money in one level of parentheses is an accounting negative (`(MXN 5.000)` is -5000)."""
    money = q.dimension.startswith("money:")
    minus = "-" if q.signs == "-" or (money and q.wrappers) else ""
    if money:
        return f"{minus}{q.unit} {q.number}"
    return f"{minus}{q.number}{q.unit}" if q.unit == "%" else f"{minus}{q.number} {q.unit}".rstrip()


def _unit_compatible(part: str, unit: str) -> bool:
    dimension = "money" if unit.startswith("money:") else unit
    if part.startswith("money:") or part == "currency":
        return dimension == "money"
    # Other typed numeric attributes (e.g. duration) require an unmarked number. Their
    # equality comparator has no authority to infer a percentage, holding or money unit.
    return dimension in _NUMERIC_UNITS.get(part, frozenset({""}))


def _quantity_occurs(text: str, part: str, value: Any) -> bool:
    quantities = _quantities(text)
    ranged = _range_members(text, quantities)
    for i, q in enumerate(quantities):
        if i in ranged or not _admissible(text, q) or not _unit_compatible(part, q.dimension):
            continue
        written = {"value": _canonical(q)}
        if q.dimension.startswith("money:") and part in {"amount", "capital", "par_value"}:
            # A scalar amount still keeps the money literal's actual accounting sign. Its
            # currency is not invented or returned as an additional claimed component.
            currency = q.dimension.removeprefix("money:")
            if _part_value("money:" + part, (value, currency), written):
                return True
        elif _part_value(part, value, written):
            return True
    return False


def _value_occurs(excerpt: str, part: str, value: Any) -> bool:
    """Occurrence, not binding: text stays verbatim; typed components use bind-v6's rules.

    Numeric candidates are complete literals (the grammar above), compared with their own sign,
    wrappers and unit; dates are whole tokens. Neither is ever a substring of an ID or number.
    The blind witness still checks whose value it is and the component's semantic role.
    """
    raw = value_text(value)
    if isinstance(value, dict) and len(value) == 1:
        value = next(iter(value.values()))
    if not isinstance(value, (str, int, float, tuple)) or isinstance(value, bool):
        return False
    text = norm(excerpt)
    if (
        part.startswith("money:")
        or part == "currency"
        or part in _NUMERIC_UNITS
        or isinstance(value, (int, float))
    ):
        return _quantity_occurs(text, part, value)
    if part == "series":
        stated_label = bool(re.fullmatch(r"series? [\w]+", norm(raw)))
        return (
            (stated_label and contains(text, raw))
            or contains(text, f"Serie {raw}")
            or contains(text, f"Series {raw}")
        )
    if part in {"date", "dob", "valid_until"} or part.endswith("_date"):
        # These are the complete date forms understood by norm.parse_date. The comparator
        # validates the month and all components; a year/day alone can never satisfy a date.
        forms = (
            r"\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{4}|"
            r"\d{1,2} (?:de )?[^\W\d_]+\.? (?:de |del )?\d{4}|"
            r"[^\W\d_]+\.? \d{1,2},? \d{4}"
        )
        us = "MM/DD/YYYY " if "mm/dd/yyyy" in text else ""
        return any(
            _part_value(part, value, {"value": us + m.group()})
            for m in re.finditer(r"(?<![\w/-])(?:" + forms + r")(?![\w/-])", text)
            # every dash joins a date to an identifier, as `-` does (`N<U+2010>2026-10-31`)
            if not _dash(text[m.start() - 1 : m.start()] or " ")
            and not _dash(text[m.end() : m.end() + 1] or " ")
        )
    if part in {"tax_id", "registry_id"}:
        # Identifier punctuation may separate its letters/digits, but no token is truncated.
        tokens = [t for t in re.split(r"[\s:;,()\[\]{}]+", text) if t]
        return any(
            _part_value(part, value, {"value": " ".join(tokens[i:j])})
            for i in range(len(tokens))
            for j in range(i + 1, len(tokens) + 1)
        )
    return contains(excerpt, raw)


def _excerpt_errors(c: Claim) -> list[SubmissionError]:
    """Name every missing part so a reader can re-read the page before retrying."""
    errors: list[SubmissionError] = []

    def missing(part: str, component: str | None = None) -> None:
        detail = f"value.{component}" if component else part
        errors.append(
            {
                "code": "excerpt_missing",
                "claim_id": c.claim_id,
                "part": part,
                "component": component,
                "normalization": NORMALIZATION,
                "message": f"{c.claim_id}: {detail} does not occur in excerpt; re-read the page, "
                "use its exact words and a contiguous subject/value quote in one block",
            }
        )

    if not contains(c.excerpt, c.subject):
        missing("subject")
    if isinstance(c.value, dict):
        unsupported = [
            key
            for key, value in c.value.items()
            if not isinstance(value, (str, int, float))
            or isinstance(value, bool)
            or value == ""
            or (key == "currency" and not isinstance(value, str))
        ]
        if unsupported:
            for key in unsupported:
                missing("value", key)
            return errors  # components() must never hide falsey currency or nested input
    parts = components(c.model_dump())
    if not parts:
        missing("value", "incompatible_components")
    else:
        for part, value in parts:
            if not _value_occurs(c.excerpt, part, value):
                missing("value", part)
    return errors


def _check(sub: ClaimsSubmission, root: Path) -> list[SubmissionError]:
    try:
        (entry,) = read_plan(root, sub.case, sub.run, sub.doc)
    except (LookupError, KeyError, ValueError, PermissionError) as exc:
        return [f"not in the read plan: {exc}"]
    errors: list[SubmissionError] = []
    if sub.doc_sha256 != entry["doc_sha256"]:
        errors.append("doc_sha256 does not match the work order")
    layout = load_layout(root, sub.case, sub.run, sub.doc)
    pages = {p["page"]: p for p in entry["pages"]}  # blocks come from the layout record itself
    blocks = {b.block_id: (pl.page, b) for pl in layout.pages for b in pl.blocks}
    for c in sub.claims:
        errors.extend(_excerpt_errors(c))
        where = f"{c.claim_id}"
        page = pages.get(c.page)
        if page is None:
            errors.append(f"{where}: page {c.page} does not exist")
            continue
        w, h = page["size_px"]
        if c.bbox[2] > w or c.bbox[3] > h:
            errors.append(f"{where}: bbox {list(c.bbox)} exceeds the page ({w}x{h})")
        if c.source == "text_layer" and not page["text_layer"]:
            errors.append(f"{where}: source text_layer, but page {c.page} has no text layer")
        found = blocks.get(c.block_id)
        if found is None or found[0] != c.page:
            errors.append(f"{where}: block {c.block_id!r} is not a layout block on page {c.page}")
        elif not inside(c.bbox, found[1].bbox):
            errors.append(f"{where}: crop is not inside its single layout block {c.block_id}")
        unknown = [b for b in c.linked_blocks if b not in blocks]
        if unknown:
            errors.append(f"{where}: linked blocks {unknown} are not in the layout")
    return errors


def _finish(root: Path, rel: str, record: dict, body: bytes) -> dict[str, Any]:
    """Complete a published claims record: audit, consume, then the catalog receipt, last.

    Each step is skipped if it already happened, so a retry after any failure completes exactly
    the missing steps with the record's original provenance. The receipt is written last, so a
    submission never counts until everything it needs is recorded.
    """
    from kyb.audit.chain import chain_events

    sub, who = record, record["produced_by"]["reader"]
    target = {"case": sub["case"], "run": sub["run"], "doc": sub["doc"]}
    input_hash, out_hash = record["produced_by"]["input_hash"], sha256_bytes(body)
    events = chain_events(root)
    log = AuditLog(root / "ledger" / "audit.jsonl")
    if not any(e["action"] == "tool.call" and e["detail"].get("action") == TOOL
               and e["output_hash"] == out_hash for e in events):  # fmt: skip
        log.append("reader", "tool.call", input_hash=input_hash, output_hash=out_hash,
                   detail={"action": TOOL, **target, "claims": len(sub["claims"]),
                           "context": who["context"]})  # fmt: skip
    if who["context"] == "isolated" and not any(
        e["action"] == "harness.attestation_consumed"
        and e["detail"].get("token_hash") == who["token_hash"] for e in events
    ):  # fmt: skip
        log.append("harness", "harness.attestation_consumed", input_hash=input_hash,
                   output_hash=out_hash,
                   detail={"token_hash": who["token_hash"], "tool_use_id": who["tool_use_id"],
                           "agent_id": who["agent_id"], "session": who["session"],
                           "target": target})  # fmt: skip
    catalog = Catalog(root)
    entry = read_plan(root, sub["case"], sub["run"], sub["doc"])[0]
    inputs = [catalog.ref(entry["layout"])] + [catalog.ref(p["png"]) for p in entry["pages"]]
    as_of = run_meta(root, sub["case"], sub["run"])["as_of"]
    catalog.register(rel, tier="claimed", produced_by={"action": TOOL, "context": who["context"]},
                     as_of=as_of, inputs=inputs)  # fmt: skip
    return {"recorded": rel, "claims": len(sub["claims"]), "context": who["context"],
            "unattested_reasons": who["unattested_reasons"]}  # fmt: skip


def submitted(root: Path, case: str, run: str, doc: str) -> bool:
    """A submission counts once its catalog receipt exists, not when its file appears."""
    return Catalog(root).get(claims_rel(case, run, doc)) is not None


def submit_claims(root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and record one document's claims. Call under the workspace action lock.

    `payload` is the tool input as executed: the reader's arguments plus the hook's one-time
    attestation token. Raises SubmissionRejected and records nothing on any failed check.
    A record that was published but never receipted (an interrupted submission) is completed with
    its original provenance when the same claims are submitted again; different claims are refused.
    """
    args, token = split_token(payload)
    try:
        sub = ClaimsSubmission.model_validate(args)
    except ValidationError as exc:
        raise SubmissionRejected([f"schema: {e['loc']}: {e['msg']}" for e in exc.errors()]) from exc
    errors = _check(sub, root)
    if errors:
        raise SubmissionRejected(errors)
    input_hash = sha256_json(args)
    rel = claims_rel(sub.case, sub.run, sub.doc)
    path = contained(root, rel)
    if path.exists():
        existing = json.loads(path.read_text("utf-8"))
        if existing["produced_by"]["input_hash"] != input_hash:
            raise PermissionError(f"{rel}: claims are written once; different claims exist")
        if submitted(root, sub.case, sub.run, sub.doc):
            return {"recorded": rel, "claims": len(sub.claims), "already": True,
                    "context": existing["produced_by"]["reader"]["context"]}  # fmt: skip
        return _finish(root, rel, existing, path.read_bytes())  # complete an interrupted one
    target = {"case": sub.case, "run": sub.run, "doc": sub.doc}
    who = attest(root, tool=TOOL, input_hash=input_hash, token=token, allowed_types=READER_TYPES,
                 target=target)  # fmt: skip
    record = {
        **sub.model_dump(),
        "tier": "claimed",
        "produced_by": {"action": TOOL, "reader": who.identity(), "input_hash": input_hash},
    }
    body = (json.dumps(record, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    publish_exclusive(path, body)  # one submission per document per run
    return _finish(root, rel, record, body)
