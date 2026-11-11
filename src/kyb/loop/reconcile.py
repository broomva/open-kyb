"""Conservative, non-applied identity candidates from typed verified evidence."""

import re
import unicodedata
from itertools import combinations

from kyb.contracts.b4 import Entity, FactSet
from kyb.facts.text import countries
from kyb.loop.models import ReconciliationCandidate


def _norm(value: str) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "",
        unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().lower(),
    )


def _refs(facts: FactSet, entity: Entity, field: str, value: str) -> list[str]:
    """Only evidence bound to this exact typed field and containing the value can support it."""
    refs = {(r.claim_id, r.doc, r.page) for r in entity.evidence if r.verified}
    target = f"entity[{entity.entity_id}].{field}"
    return sorted(
        {
            e.evidence_id
            for e in facts.evidence
            if e.verified
            and e.field == target
            and (e.evidence_id, e.doc, e.page) in refs
            and _norm(value)
            and (
                value.upper() in countries(e.excerpt)
                if field == "jurisdiction"
                else bool(
                    re.search(
                        r"(?<![A-Za-z0-9])" + re.escape(value) + r"(?![A-Za-z0-9])",
                        e.excerpt,
                        re.IGNORECASE,
                    )
                )
            )
        }
    )


def reconcile(facts: FactSet) -> list[ReconciliationCandidate]:
    out = []
    entities = sorted(facts.entities, key=lambda e: e.entity_id)
    for a, b in combinations(entities, 2):
        if a.kind != b.kind or a.kind != "company":
            continue
        if not ({_norm(n) for n in a.names if _norm(n)} & {_norm(n) for n in b.names if _norm(n)}):
            continue
        refs: set[str] = set()
        ids_a = {
            k: {_norm(v) for v in vs if _refs(facts, a, f"identifiers.{k}", v)}
            for k, vs in a.identifiers.items()
            if k in {"reg_no", "tax_id", "rfc"}
        }
        ids_b = {
            k: {_norm(v) for v in vs if _refs(facts, b, f"identifiers.{k}", v)}
            for k, vs in b.identifiers.items()
            if k in {"reg_no", "tax_id", "rfc"}
        }
        ja = {j.upper() for j in a.jurisdiction if _refs(facts, a, "jurisdiction", j)}
        jb = {j.upper() for j in b.jurisdiction if _refs(facts, b, "jurisdiction", j)}
        conflicts = any(
            ids_a[k] and ids_b[k] and ids_a[k] != ids_b[k] for k in ids_a.keys() & ids_b.keys()
        )
        conflicts |= any(len(values) > 1 for values in (*ids_a.values(), *ids_b.values()))
        conflicts |= len(ja) > 1 or len(jb) > 1 or bool(ja and jb and ja != jb)
        shared = any(ids_a[k] & ids_b[k] for k in ids_a.keys() & ids_b.keys())
        if conflicts:
            state, reason = "contradictory", "Verified identifiers or jurisdictions conflict."
        elif shared and len(ja) == len(jb) == 1 and ja == jb:
            state, reason = (
                "supported",
                "Equal verified strong identifier and explicit jurisdiction.",
            )
        else:
            state, reason = (
                "unresolved",
                "Name similarity is insufficient; verified identifiers and "
                "jurisdiction are required.",
            )
        for entity in (a, b):
            for k, values in entity.identifiers.items():
                if k in {"reg_no", "tax_id", "rfc"}:
                    for value in values:
                        refs.update(_refs(facts, entity, f"identifiers.{k}", value))
            for value in entity.jurisdiction:
                refs.update(_refs(facts, entity, "jurisdiction", value))
        out.append(
            ReconciliationCandidate(
                left=a.entity_id,
                right=b.entity_id,
                state=state,
                reason=reason,
                evidence_ids=sorted(refs),
            )
        )
    # A supported edge cannot act as a bridge into a contradictory company.
    conflicting = {x for c in out if c.state == "contradictory" for x in (c.left, c.right)}
    return [
        c.model_copy(
            update={"state": "unresolved", "reason": "A conflicting candidate prevents linking."}
        )
        if c.state == "supported" and (c.left in conflicting or c.right in conflicting)
        else c
        for c in out
    ]
