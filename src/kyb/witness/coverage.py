"""Match layout facts to components of already verified claims, never to raw predictions."""

from __future__ import annotations

import re

from kyb.witness.bind import (
    canon_attr,
    claim_attr,
    components,
    subject_match,
    value_match,
    value_text,
)
from kyb.witness.norm import norm

VERSION = "coverage-v3"


def check_record(record: dict) -> None:
    """An immutable result from the unreleased unsafe matcher cannot be promoted by reuse.
    Legacy unversioned records retain their historical interpretation, never a v3 claim."""
    if record.get("coverage_version") not in {None, VERSION}:
        raise ValueError("coverage matcher changed; verification needs a new run")
    for inner in (record.get("strata") or {}).values():
        check_record(inner)


def _plain_numeric(value, attr: str) -> bool:
    """Project only an unqualified number and its unit. Unknown trailing dimensions must
    not vanish merely because the numeric comparator can find an equal number in them."""
    unit = r"%" if attr == "pct" else r"(?:acciones|shares|acc\.)"
    return bool(re.fullmatch(r"[+-]?\s*\d+(?:[.,]\d+)*(?:\s*" + unit + r")?", norm(str(value))))


def attribute(label: str) -> str:
    # A printed percent column is a typed field, not an empty/unknown attribute. Do not
    # generalize this to arbitrary symbols or series-qualified share labels.
    return "pct" if label.strip() == "%" else canon_attr(label)


def _values(claim: dict, attr: str) -> list:
    own = claim_attr(claim)
    parts = components(claim) if isinstance(claim["value"], dict) else None
    if parts is not None:
        # bind.components preserves a single-field typed value as a dict. Unwrap only
        # that exact field, never an arbitrary number from a compound value.
        parts = [
            (
                k,
                next(iter(v.values()))
                if isinstance(v, dict) and len(v) == 1 and canon_attr(next(iter(v))) == k
                else v,
            )
            for k, v in parts
        ]
    # We do not yet establish equivalence of qualified holdings and layout totals.
    # In particular, dropping series B here could incorrectly cover series A.
    if (
        attr in {"pct", "shares"}
        and parts is not None
        and any(k not in {"pct", "shares"} for k, _ in parts)
    ):
        return []
    if own == attr:
        # A multi-component holding must not compare all its numbers with one column.
        selected = [v for k, v in parts or [] if k == attr]
        values = selected or [claim["value"]]
    elif attr in {"pct", "shares"} and own in {"shareholding", "ownership", "pct", "shares"}:
        values = [v for k, v in parts or [] if k == attr]
    else:
        values = []
    return [v for v in values if _plain_numeric(v, attr)] if attr in {"pct", "shares"} else values


def coverage(facts: list[dict], verified: list[dict]) -> list[dict]:
    """Same block, same subject, same typed component. A corroborating claim cannot hide
    another verified claim's disagreement with that component."""
    out = []
    for fact in facts:
        attr = attribute(fact["attribute"])
        same = [
            (c, values)
            for c in verified
            if c["block_id"] == fact["block_id"]
            and subject_match(fact["subject"], c["subject"])
            and (attr not in {"pct", "shares"} or _plain_numeric(fact["value"], attr))
            and (values := _values(c, attr))
        ]
        if not same:
            status = {"status": "uncovered"}
        elif any(value_match(v, fact["value"], attr) for _, values in same for v in values):
            status = {"status": "matched"}
        else:
            status = {"status": "conflict", "claims": [c["claim_id"] for c, _ in same]}
        out.append({**fact, "canonical": attr, **status})
    for claim in verified:
        groups: dict[str, list[dict]] = {}
        for fact in facts:
            attr = attribute(fact["attribute"])
            if (
                claim["block_id"] == fact["block_id"]
                and subject_match(fact["subject"], claim["subject"])
                and (attr not in {"pct", "shares"} or _plain_numeric(fact["value"], attr))
                and _values(claim, attr)
            ):
                groups.setdefault(attr, []).append(fact)
        for attr, group in groups.items():
            values = _values(claim, attr)
            if not any(value_match(v, f["value"], attr) for f in group for v in values):
                out.append(
                    {
                        "block_id": claim["block_id"],
                        "kind": "claim",
                        "subject": claim["subject"],
                        "attribute": attr,
                        "canonical": attr,
                        "value": value_text(values[0]),
                        "status": "conflict",
                        "claims": [claim["claim_id"]],
                        "layout_values": [f["value"] for f in group],
                    }
                )
    return out
