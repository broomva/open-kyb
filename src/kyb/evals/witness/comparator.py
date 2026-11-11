"""The production comparator, whichever version is on this checkout.

L1 (issue) moves binding from `kyb.witness.verify._binding` (bind-v5) to
`kyb.witness.bind.binding` (bind-v6). The eval resolves whichever exists, records its version
with every result, and never carries a copy: a probe verifies exactly when the pipeline would
verify that claim against that crop's response.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from kyb.witness import norm


def _bind() -> tuple[Callable[..., bool], str]:
    try:
        mod = importlib.import_module("kyb.witness.bind")
        return mod.binding, mod.BIND
    except ImportError:
        from kyb.witness import verify

        return verify._binding, verify.BIND


def versions() -> dict[str, str]:
    from kyb.witness import gemini

    return {"binding": _bind()[1], "norm": norm.VERSION, "prompt": gemini.PROMPT_VERSION}


def readable(crop: dict) -> bool:
    return bool(crop.get("readable")) and bool((crop.get("transcription") or "").strip())


def verifies(claim: dict[str, Any], crop: dict) -> bool:
    """The pipeline's per-claim verdict for a claim with no linked context (verify_document):
    readable, the excerpt witnessed in the transcription, and the re-read binds."""
    bind, _ = _bind()
    c = {"claim_id": "probe", "object": None, "role": None, "linked_blocks": [], **claim}
    if not readable(crop) or not norm.contains(crop["transcription"], c["excerpt"]):
        return False
    return bool(bind(c, crop.get("reread") or {}, crop.get("transcription") or ""))


def struck_values(crop: dict) -> list[str]:
    rr = crop.get("reread") or {}
    out = [str(x) for x in rr.get("struck") or []]
    out += [
        str(t.get("value", ""))
        for t in rr.get("values") or []
        if isinstance(t, dict) and t.get("struck")
    ]
    return out


def stated(crop: dict) -> str:
    """Everything the response states, flattened, for the "must not appear" checks."""
    rr = crop.get("reread") or {}
    parts = [
        crop.get("transcription") or "",
        str(rr.get("subject") or ""),
        str(rr.get("value") or ""),
        *struck_values(crop),
        *(str(a) for a in rr.get("annotations") or []),
        *(
            f"{t.get('subject', '')} {t.get('value', '')}"
            for t in rr.get("values") or []
            if isinstance(t, dict)
        ),
    ]
    return "\n".join(parts)
