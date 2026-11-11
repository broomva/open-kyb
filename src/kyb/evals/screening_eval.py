"""L2 screening on the eval sets: the homoglyph attack (suite 6) and the fuzzy method on the
spec screening set (benchmark, report-only). Offline: `kyb.screening` is pure."""

from __future__ import annotations

import json
from typing import Any

from kyb.contracts.b4 import Entity, EvidenceRef
from kyb.evals.agentpath import WATCHLIST, derived_packet
from kyb.evals.paths import POLICY, load
from kyb.screening.screen import _candidate, entries, policy_threshold, screen_factset

HOMOGLYPH = "Strаit Cаpitаl FZE"  # Cyrillic а inside 'Strait Capital FZE'


def _threshold() -> float:
    return policy_threshold(POLICY.read_text("utf-8"))


def homoglyph() -> tuple[bool, str]:
    """The sanctioned name in homoglyphs, as a party of packet C, still raises its candidate."""
    packet = derived_packet("C")
    party = packet.facts.entities[0]
    renamed = [
        e.model_copy(update={"names": [HOMOGLYPH]}) if e is party else e
        for e in packet.facts.entities
    ]
    facts = packet.facts.model_copy(update={"entities": renamed})
    result = screen_factset(
        facts, json.loads(WATCHLIST.read_bytes()), watchlist_version="eval", threshold=_threshold()
    )
    hits = [
        c
        for c in result.candidates
        if c.entity_id == party.entity_id and c.entry.entry_id == "SAN-0001"
    ]
    plain = HOMOGLYPH.encode("ascii", "ignore").decode() == HOMOGLYPH  # guard: it is not ASCII
    ok = bool(hits) and hits[0].triggers_fuzzy and not plain
    return (
        ok,
        f"homoglyph party candidates: {[(c.entry.entry_id, c.jw_score) for c in hits]}",
    )


def fuzzy() -> dict[str, Any]:
    """recall / precision / false clears of `candidates-v1 + JW` on the screening pairs."""
    listed = {e.entry_id: e for e in entries(json.loads(WATCHLIST.read_bytes()))}
    thr = _threshold()
    tp = fn = fp = tn = corroborated_different = 0
    rows = []
    for p in load("screening/cases.json")["pairs"]:
        a = p["party"]
        entity = Entity(
            entity_id="e-eval",
            kind=a["kind"],
            names=[a["name"]],
            identifiers={"reg_no": [a["reg_no"]]} if a.get("reg_no") else {},
            nationality=[a["nationality"]] if a.get("nationality") else [],
            dob=[a["dob"]] if a.get("dob") else [],
            jurisdiction=[a["jurisdiction"]] if a.get("jurisdiction") else [],
            evidence=[EvidenceRef(claim_id="c-eval", doc="01", page=1, verified=True)],
        )
        c = _candidate(entity, listed[p["entry"]], thr)
        raised = c is not None
        if p["label"] == "same":
            tp, fn = tp + raised, fn + (not raised)
        else:
            fp, tn = fp + raised, tn + (not raised)
            corroborated_different += bool(c and c.identifiers.corroborating)
        rows.append(
            {
                "id": p["id"],
                "label": p["label"],
                "raised": raised,
                "jw": round(c.jw_score, 4) if c else None,
            }
        )
    return {
        "pairs": len(rows),
        "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "false_clears": [r["id"] for r in rows if r["label"] == "same" and not r["raised"]],
        "different_raised": [r["id"] for r in rows if r["label"] == "different" and r["raised"]],
        "different_corroborated": corroborated_different,
        "threshold": thr,
        "rows": rows,
    }
