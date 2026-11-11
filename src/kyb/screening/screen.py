"""`screen` (L2; spec, spec, spec): `FactSet` + watchlist → `ScreeningResult`.

Every entity in the facts (the applicant, its owners, directors, agents and other parties) is
compared with every sanctions and PEP entry of a compatible type (organisations against
`entity` entries; persons against `individual` entries and PEPs; an entity of unknown kind
against all of them).

- **Candidates (recall, spec):** a pair of compatible types is a candidate when any recall
  normaliser scores it at or above `RECALL_CUTOFF`, or one name's tokens are contained in the
  other's. A pair the policy's own comparison hits is always a candidate, whatever its types,
  so the candidates contain every trigger. Candidates are what L4 turns into `same_party`
  instances (`sp-<candidate_id>`).
- **Trigger (spec):** `jw_score` is the policy's normalised Jaro-Winkler, the max over the
  entity's spellings and the entry's name and aliases; `alias_exact` is an exact normalised
  alias. `triggers_fuzzy` = `jw_score >= threshold or alias_exact`, for **sanctions** entries only:
  Sanctions screening checks all in-scope parties. A PEP name hit is recorded with the same score
  and identifiers, and the PEP evaluator reads it (a match needs a corroborating identifier;
  otherwise it is a note).
- **Identifiers:** `corroborating` lists only what the policy accepts as corroboration for that
  list (sanctions: `reg_no`, `jurisdiction+dob`; PEP: `dob`, `nationality+dob`);
  `conflicting` lists known values that differ (`reg_no`, `jurisdiction`, `nationality`,
  `dob`). Nothing is inferred from a missing value.

The threshold is read from the policy text, never hard-coded (invariant 3).
"""

from __future__ import annotations

import re
from typing import Any

from kyb.contracts.b4 import (
    Entity,
    FactSet,
    IdentifierCheck,
    ScreeningCandidate,
    ScreeningResult,
    WatchlistEntry,
)
from kyb.screening.normalize import NORMALISATION, d044, policy_score, recall_score

RECALL_CUTOFF = 0.80
_ORG_KINDS = {"company", "foundation", "trust", "nominee", "agent"}
_THRESHOLD = re.compile(
    r"(?:similarity\s+)?score\s*(?:≥|>=)\s*\**([01](?:\.\d+)?)\**"
    r"(?:\s*on a normali[sz]ed comparison|\s*where\b)",
    re.IGNORECASE,
)


def policy_threshold(policy_text: str) -> float:
    """Fuzzy-sanctions similarity threshold as the policy states it. Refuses a policy where it
    cannot be found: a silent default would screen against a rule nobody wrote."""
    found = {float(m) for m in _THRESHOLD.findall(policy_text)}
    if len(found) != 1:
        raise ValueError("the policy does not state exactly one similarity threshold")
    return found.pop()


def method(threshold: float) -> str:
    return f"candidates-v1+jw-{threshold:g}+{NORMALISATION}"


def entries(watchlist: dict[str, Any]) -> list[WatchlistEntry]:
    out = []
    for kind, key in (("sanctions", "sanctions"), ("pep", "pep")):
        for e in watchlist.get(key) or []:
            out.append(WatchlistEntry(
                entry_id=e["id"], list_kind=kind, name=e["name"], aliases=e.get("aliases") or [],
                entity_type=e.get("type"), nationality=e.get("nationality"),
                jurisdiction=e.get("jurisdiction"), dob=e.get("dob"),
                registration_number=e.get("registration_number"), program=e.get("program"),
                position=e.get("position"), in_office=e.get("in_office"),
                left_office=e.get("left_office"),
            ))  # fmt: skip
    return out


def _compatible(entity: Entity, entry: WatchlistEntry) -> bool:
    if entity.kind == "unknown":
        return True
    is_org = entity.kind in _ORG_KINDS
    if entry.list_kind == "pep":
        return not is_org
    if entry.entity_type == "entity":
        return is_org
    if entry.entity_type == "individual":
        return not is_org
    return True


def _alnum(s: str) -> str:
    return re.sub(r"[^0-9a-z]", "", d044(s).replace(" ", ""))


def identifiers(entity: Entity, entry: WatchlistEntry) -> IdentifierCheck:
    """Corroboration and conflict, each computed on its own: a registration number is compared
    only with the entity's registration numbers (never a passport or a tax id), and a match on
    one value never hides a different value beside it."""
    corroborating: list[str] = []
    conflicting: list[str] = []
    regs = {_alnum(v) for v in entity.identifiers.get("reg_no", [])} - {""}
    if entry.registration_number and regs:
        mine = _alnum(entry.registration_number)
        if mine in regs:
            corroborating.append("reg_no")
        if regs - {mine}:
            conflicting.append("reg_no")
    # each field against its own: an entry's jurisdiction against the entity's jurisdiction, its
    # nationality against the entity's nationality. Sanctions corroboration ("jurisdiction and date
    # of birth" reads, for an individual entry that states no jurisdiction, its nationality: the
    # one explicit fallback, and only within that entry's own field.
    jur_same = bool(entry.jurisdiction) and entry.jurisdiction in entity.jurisdiction
    nat_same = bool(entry.nationality) and entry.nationality in entity.nationality
    place_same = jur_same if entry.jurisdiction else nat_same
    dob_same = bool(entry.dob) and entry.dob in entity.dob
    if entry.list_kind == "sanctions":
        if dob_same and place_same:
            corroborating.append("jurisdiction+dob")
    else:
        if dob_same:
            corroborating.append("dob")
        if dob_same and nat_same:
            corroborating.append("nationality+dob")
    if entry.jurisdiction and set(entity.jurisdiction) - {entry.jurisdiction}:
        conflicting.append("jurisdiction")
    if entry.nationality and set(entity.nationality) - {entry.nationality}:
        conflicting.append("nationality")
    if entry.dob and set(entity.dob) - {entry.dob}:
        conflicting.append("dob")
    return IdentifierCheck(corroborating=corroborating, conflicting=conflicting)


def _candidate(
    entity: Entity, entry: WatchlistEntry, threshold: float
) -> ScreeningCandidate | None:
    listed = [entry.name, *entry.aliases]
    best, matched = -1.0, entry.name
    for x in listed:  # the list's own order breaks ties: name first, then aliases
        score = max(policy_score(n, x) for n in entity.names)
        if score > best:
            best, matched = score, x
    aliases = {d044(a) for a in entry.aliases}
    alias_exact = any(d044(n) in aliases for n in entity.names)
    name_hit = best >= threshold or alias_exact
    # the policy's own hit is always a candidate, whatever the types or the recall cut-off
    if not name_hit and not (
        _compatible(entity, entry)
        and max(recall_score(n, x) for n in entity.names for x in listed) >= RECALL_CUTOFF
    ):
        return None
    return ScreeningCandidate(
        candidate_id=f"{entry.entry_id}-{entity.entity_id.removeprefix('e-')}",
        entity_id=entity.entity_id, entry=entry, matched_name=matched, jw_score=best,
        alias_exact=alias_exact, triggers_fuzzy=entry.list_kind == "sanctions" and name_hit,
        identifiers=identifiers(entity, entry),
    )  # fmt: skip


def screen_factset(fs: FactSet, watchlist: dict[str, Any], *, watchlist_version: str,
                   threshold: float) -> ScreeningResult:  # fmt: skip
    """Deterministic: entities in fact order, entries in list order."""
    listed = entries(watchlist)
    candidates = []
    for entity in fs.entities:
        for entry in listed:
            if c := _candidate(entity, entry, threshold):
                candidates.append(c)
    return ScreeningResult(case=fs.case, run=fs.run, fact_set_hash=fs.fact_set_hash,
                           watchlist_version=watchlist_version, method=method(threshold),
                           candidates=candidates)  # fmt: skip
