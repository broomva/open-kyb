"""Who owns and controls the applicant, read from the typed facts.

Only `current` holdings count. A pending transfer, a struck value or a historical deed is reported
by the conflict and scoring rules, never applied. When documents give one owner several current
percentages (an unresolved conflict) they are alternatives, not additions: thresholds read the
highest (the cautious reading for a trigger) and coverage reads the lowest.
"""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal

from kyb.contracts.b4 import Entity, EvidenceRef, FactSet, Holding, Role

Tri = str  # "yes" | "no" | "unknown"
HUNDRED = Decimal(100)
DIRECTOR_ROLES = frozenset({"director", "president", "nominee_director"})
CONTROL_ROLES = DIRECTOR_ROLES | {"legal_rep"}
NOMINEE_ROLES = frozenset({"nominee_director", "nominee_shareholder"})
FOUNDATION_CONTROL = frozenset({"beneficiary", "council_member"})


class Pct:
    """An owner's stake: the lowest and highest current value, or unknown."""

    __slots__ = ("lo", "hi", "holdings")

    def __init__(self, lo: Decimal | None, hi: Decimal | None, holdings: list[Holding]) -> None:
        self.lo, self.hi, self.holdings = lo, hi, holdings

    @property
    def known(self) -> bool:
        return self.lo is not None and self.hi is not None


class Graph:
    def __init__(
        self,
        facts: FactSet,
        threshold: Decimal,
        *,
        doubted: frozenset[str] = frozenset(),
        doubt_all: bool = False,
    ) -> None:
        """`doubted`: entities whose identity or ownership facts have a verification gap;
        `doubt_all`: a gap that cannot be tied to one entity. A gap makes an absence `unknown`."""
        self.facts = facts
        self.t = threshold
        self._doubted, self._doubt_all = doubted, doubt_all
        self.ents: dict[str, Entity] = {e.entity_id: e for e in facts.entities}
        if facts.applicant not in self.ents:
            raise ValueError(f"applicant {facts.applicant!r} is not among the entities")
        self.app = self.ents[facts.applicant]
        self._into: dict[str, list[Holding]] = defaultdict(list)
        for h in facts.holdings:
            if h.status == "current":
                self._into[h.owned].append(h)
        self.roles_of: dict[str, list[Role]] = defaultdict(list)
        for r in facts.roles:
            self.roles_of[r.of].append(r)

    def kind(self, eid: str) -> str:
        e = self.ents.get(eid)
        return e.kind if e else "unknown"

    def owners(self, owned: str) -> dict[str, Pct]:
        """Each current owner's stake in `owned`. Several documents may observe the same holding:
        repeated observations are one holding, not more shares. Percentages come from `pct`, or
        from share counts when every owner's count is a single value (the denominator is then the
        sum over owners); if counts disagree, the denominator is unknown and so are the stakes."""
        grouped: dict[str, list[Holding]] = defaultdict(list)
        for h in self._into.get(owned, []):
            grouped[h.owner].append(h)
        counts = {o: {h.shares for h in g} for o, g in grouped.items()}
        by_shares = bool(grouped) and all(
            len(c) == 1 and None not in c and next(iter(c)) for c in counts.values()
        )
        total = sum(next(iter(c)) for c in counts.values()) if by_shares else 0

        def pct(h: Holding) -> Decimal | None:
            if h.pct is not None:
                return Decimal(h.pct)
            if by_shares and h.shares:
                return Decimal(h.shares) * HUNDRED / Decimal(total)
            return None

        out = {}
        for owner, group in sorted(grouped.items()):
            vals = {pct(h) for h in group}
            if None in vals:
                out[owner] = Pct(None, None, group)
            else:
                out[owner] = Pct(min(vals), max(vals), group)  # type: ignore[type-var]
        return out

    def chain(self) -> dict[str, Decimal | None]:
        """Every entity above the applicant with its effective stake (highest values, summed over
        paths). None means a percentage on the path is unknown."""
        out: dict[str, Decimal | None] = {}

        def walk(owned: str, factor: Decimal | None, seen: frozenset[str]) -> None:
            for owner, p in sorted(self.owners(owned).items()):
                if owner in seen:
                    continue
                share = None if factor is None or p.hi is None else factor * p.hi / HUNDRED
                prev = out.get(owner, Decimal(0))
                out[owner] = None if share is None or prev is None else prev + share
                walk(owner, share, seen | {owner})

        walk(self.app.entity_id, HUNDRED, frozenset({self.app.entity_id}))
        return out

    def intermediate_layers(self) -> tuple[int, list[EvidenceRef]]:
        """Maximum number of non-person holding layers along any ownership path above the
        applicant, and the evidence along paths of at least 2 layers."""
        best_depth = 0
        refs: list[EvidenceRef] = []

        def walk(
            owned: str,
            depth: int,
            path_refs: list[EvidenceRef],
            seen: frozenset[str],
        ) -> None:
            nonlocal best_depth
            for owner, p in sorted(self.owners(owned).items()):
                if owner in seen or self.kind(owner) == "person":
                    continue
                step_refs = [
                    *path_refs,
                    *[r for h in p.holdings for r in h.evidence],
                    *self.evidence_of(owner),
                ]
                cur = depth + 1
                if cur > best_depth:
                    best_depth = cur
                if cur >= 2:
                    refs.extend(step_refs)
                walk(owner, cur, step_refs, seen | {owner})

        walk(self.app.entity_id, 0, [], frozenset({self.app.entity_id}))
        return best_depth, refs

    def at_or_above(self, stake: Decimal | None) -> Tri:
        if stake is None:
            return "unknown"
        return "yes" if stake >= self.t else "no"

    def controllers_of(self, eid: str) -> list[Role]:
        """The beneficiaries and council members who resolve a foundation or trust."""
        return [r for r in self.roles_of.get(eid, []) if r.role in FOUNDATION_CONTROL]

    def ubos(self) -> dict[str, Tri]:
        """Natural persons owning ≥ threshold through the chain, and the named controllers of a
        foundation or trust holding ≥ threshold."""
        out: dict[str, Tri] = {}
        for eid, stake in self.chain().items():
            reach = self.at_or_above(stake)
            if reach == "no":
                continue
            holders = [eid] if self.kind(eid) == "person" else []
            if self.kind(eid) in ("foundation", "trust"):
                holders += [r.holder for r in self.controllers_of(eid)]
            for h in holders:
                if self.kind(h) == "person" and out.get(h) != "yes":
                    out[h] = reach
        return out

    def directors(self) -> list[Role]:
        return [r for r in self.roles_of.get(self.app.entity_id, []) if r.role in DIRECTOR_ROLES]

    def doubted(self, eid: str) -> bool:
        return self._doubt_all or eid in self._doubted

    def person_state(self, eid: str) -> Tri:
        e = self.ents.get(eid)
        if e and e.kind == "person" and e.nationality and e.dob:
            return "yes"
        return "unknown" if self.doubted(eid) else "no"

    def resolution(self, eid: str, seen: frozenset[str] = frozenset()) -> Tri:
        """Is this owner resolved to named natural persons with nationality and DOB?

        A person: by their own nationality and DOB. A foundation or trust: by a beneficiary or
        council member, named and resolved. Anything else is unwrapped: it is resolved only if the
        share of it that does *not* reach resolved persons stays below the threshold, counting
        every owner, so unresolved holders never pass as resolved."""
        kind = self.kind(eid)
        if kind == "person":
            return self.person_state(eid)
        if kind in ("foundation", "trust"):
            states = {self.person_state(r.holder) for r in self.controllers_of(eid)}
            if "yes" in states:
                return "yes"
            return "unknown" if "unknown" in states or self.doubted(eid) else "no"
        if eid in seen:
            return "no"
        if not self.owners(eid):
            return "unknown" if self.doubted(eid) else "no"
        lo, hi = self.unresolved_share(eid, seen | {eid})
        if lo >= self.t:
            return "no"
        return "unknown" if hi >= self.t else "yes"

    def unresolved_share(
        self, owned: str, seen: frozenset[str] = frozenset()
    ) -> tuple[Decimal, Decimal]:
        """(lowest, highest) share of `owned` not held by resolved owners. Unknown percentages and
        unknown resolutions widen the interval; they never count as resolved."""
        sure, maybe = Decimal(0), Decimal(0)
        for owner, p in sorted(self.owners(owned).items()):
            state = self.resolution(owner, seen | {owned})
            if state == "yes":
                sure += p.lo if p.lo is not None else Decimal(0)
                maybe += (p.hi if p.hi is not None else HUNDRED) - (p.lo or Decimal(0))
            elif state == "unknown":
                maybe += p.hi if p.hi is not None else HUNDRED
        hi = max(Decimal(0), HUNDRED - sure)
        lo = Decimal(0) if self.doubted(owned) else max(Decimal(0), hi - maybe)
        return lo, hi

    def resolution_evidence(
        self, eid: str, seen: frozenset[str] = frozenset()
    ) -> list[EvidenceRef]:
        """Every fact `resolution(eid)` reads: the entity, its controllers (foundation, trust),
        and each owner's holding and resolution evidence, down the chain. A rule that fires on a
        resolution cites all of it, since changing any of it can change the outcome."""
        if eid in seen:
            return []
        seen = seen | {eid}
        out = list(self.evidence_of(eid))
        for r in self.controllers_of(eid):
            out += [*r.evidence, *self.evidence_of(r.holder)]
        for owner, p in sorted(self.owners(eid).items()):
            out += [ref for h in p.holdings for ref in h.evidence]
            out += self.resolution_evidence(owner, seen)
        return out

    def evidence_of(self, *eids: str) -> list[EvidenceRef]:
        return [ref for eid in eids if eid in self.ents for ref in self.ents[eid].evidence]

    def nominees(self) -> tuple[list[Role], list[Holding]]:
        """Nominee roles, registered agents acting as directors, and nominee shareholders."""
        app = self.app.entity_id
        roles = self.roles_of.get(app, [])
        agents = {r.holder for r in roles if r.role == "registered_agent"}
        acting = {r.holder for r in roles if self.kind(r.holder) in ("agent", "nominee")} | agents
        nom_roles = [
            r
            for r in roles
            if r.role in NOMINEE_ROLES or (r.role in DIRECTOR_ROLES and r.holder in acting)
        ]
        nom_holdings = [
            h
            for owner, p in self.owners(app).items()
            if self.kind(owner) == "nominee"
            for h in p.holdings
        ]
        return nom_roles, nom_holdings
