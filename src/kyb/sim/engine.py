"""The autonomy-progression simulation: weekly arrivals, a decider, an officer, a per-profile dial.

The profile key is the policy's precedent key (compliance_rules.md §12, `b4.comparable_key`):
incorporation jurisdiction plus the set of fired rules. Per profile:

- Level 0: every case goes to an officer, who signs it.
- Level 1: after K consecutive officer agreements with zero overrides, the profile's APPROVED
  proposals are auto-finalised. Each is still audited by an officer afterwards (the audit is what
  keeps overrides observable).
- Level 2: after K further clean audits, the officer no longer reviews every APPROVED case: only a
  seeded `audit_rate` sample is audited. An override on any audit demotes the profile to Level 0.
- REJECTED and MANUAL_REVIEW proposals always go to an officer; REJECTED is auto-finalised only
  under `full-auto-jev` (an officer with `full_auto = True`, which bypasses the dial).
An override is recorded as a precedent: context for comparable decisions (§12), never an outcome.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict

from kyb.contracts.b4 import comparable_key
from kyb.sim.decide import Decider, Decision
from kyb.sim.officer import Officer, Signature
from kyb.synth.case import CaseSpec
from kyb.synth.label import Label, label

Route = Literal["officer", "auto", "auto+audit", "jev-auto", "unresolved"]


class SimConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    weeks: int = 8
    per_week: int = 10
    k: int = 5
    audit_rate: float = 0.25  # share of Level-2 auto-approvals an officer still audits
    seed: int = 11


class Outcome(BaseModel):
    model_config = ConfigDict(frozen=True)
    arrival_id: str
    week: int
    case_id: str
    profile: str
    level: int  # the profile's level when the case arrived
    proposed: str
    gold: str
    final: str
    route: Route
    signer: str
    approved_via: str
    auto_finalised: bool
    blocking_review: bool  # an officer had to sign before the case could be finalised
    audited: bool  # an officer reviewed it after auto-finalising
    override: bool
    false_approval: bool  # auto or Jev APPROVED while gold is not APPROVED
    false_reject: bool  # REJECTED while gold is APPROVED
    precedents_considered: int
    latency_h: float


class Precedent(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    week: int
    profile: str
    case_id: str
    from_status: str
    to_status: str
    rationale: str


class Profile:
    __slots__ = ("key", "level", "streak", "first_week", "promoted", "demotions", "approved_seen")

    def __init__(self, key: str, week: int) -> None:
        self.key, self.level, self.streak, self.first_week = key, 0, 0, week
        self.promoted: list[tuple[int, int, int]] = []  # (week, from level, to level)
        self.demotions = 0
        self.approved_seen = False


class SimResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    config: SimConfig
    decider: str
    officer: str
    outcomes: list[Outcome]
    precedents: list[Precedent]
    profiles: dict[str, dict]


def profile_key(case: CaseSpec, d: Decision) -> str:
    jur, rules = comparable_key(case.company.jurisdiction, list(d.fired_rules))
    return f"{jur}:{','.join(rules) or '-'}"


def arrivals(pool: Sequence[CaseSpec], cfg: SimConfig) -> list[tuple[int, str, CaseSpec]]:
    """Weekly batches, seeded: about `per_week` cases a week, drawn with replacement from the pool
    (a new applicant of a comparable profile arrives; the pool is a population, not a queue)."""
    rng = random.Random(f"arrivals/{cfg.seed}")
    out = []
    for week in range(1, cfg.weeks + 1):
        n = max(1, cfg.per_week + rng.randint(-2, 2))
        out += [(week, f"w{week:02d}-{j:02d}", rng.choice(pool)) for j in range(1, n + 1)]
    return out


def simulate(
    pool: Sequence[CaseSpec],
    decider: Decider,
    officer: Officer,
    cfg: SimConfig,
    gold: dict[str, Label] | None = None,
) -> SimResult:
    rng = random.Random(f"audit/{cfg.seed}")
    gold = gold or {}
    decisions: dict[str, Decision] = {}
    profiles: dict[str, Profile] = {}
    precedents: list[Precedent] = []
    outcomes: list[Outcome] = []

    for week, arrival_id, case in arrivals(pool, cfg):
        lab = gold.get(case.case_id) or label(case)
        gold.setdefault(case.case_id, lab)
        dec = decisions.setdefault(case.case_id, decider.decide(case))
        key = profile_key(case, dec)
        prof = profiles.setdefault(key, Profile(key, week))
        prof.approved_seen |= dec.status == "APPROVED"
        prior = sum(1 for p in precedents if p.profile == key)
        level_at = prof.level
        auto = blocking = audited = override = False
        route: Route
        sig: Signature
        latency = 0.0

        if officer.full_auto:
            sig = officer.review(case, dec, lab, arrival_id)
            final, route = sig.status, "jev-auto" if sig.status != "MANUAL_REVIEW" else "unresolved"
            auto = sig.status != "MANUAL_REVIEW"
        elif dec.status == "APPROVED" and prof.level >= 1:
            final, route, auto = "APPROVED", "auto", True  # type: ignore[assignment]
            sig = Signature(status="APPROVED", signer="officer", approved_via="auto-profile")
            if prof.level == 1 or rng.random() < cfg.audit_rate:
                audited, route = True, "auto+audit"
                aud = officer.review(case, dec, lab, arrival_id)
                latency = aud.latency_h
                if aud.status != dec.status:
                    override = True
                    _demote(prof, week, precedents, case, dec, aud, arrival_id)
                else:
                    prof.streak += 1
                    if prof.level == 1 and prof.streak >= cfg.k:
                        _promote(prof, week, 2)
        else:
            sig = officer.review(case, dec, lab, arrival_id)
            final, route, blocking, latency = sig.status, "officer", True, sig.latency_h
            if sig.status != dec.status:
                override = True
                _demote(prof, week, precedents, case, dec, sig, arrival_id)
            else:
                if dec.status == "APPROVED":
                    prof.streak += 1
                    if prof.level == 0 and prof.streak >= cfg.k:
                        _promote(prof, week, 1)

        outcomes.append(
            Outcome(
                arrival_id=arrival_id,
                week=week,
                case_id=case.case_id,
                profile=key,
                level=level_at,
                proposed=dec.status,
                gold=lab.status,
                final=final,
                route=route,
                signer=sig.signer,
                approved_via=sig.approved_via,
                auto_finalised=auto,
                blocking_review=blocking,
                audited=audited,
                override=override,
                false_approval=auto and final == "APPROVED" and lab.status != "APPROVED",
                false_reject=final == "REJECTED" and lab.status == "APPROVED",
                precedents_considered=prior,
                latency_h=latency,
            )
        )
    return SimResult(
        config=cfg,
        decider=decider.name,
        officer=officer.name,
        outcomes=outcomes,
        precedents=precedents,
        profiles={
            k: {
                "first_week": p.first_week,
                "level": p.level,
                "demotions": p.demotions,
                "promotions": [{"week": w, "from": a, "to": b} for w, a, b in p.promoted],
                "approved": p.approved_seen,
            }
            for k, p in sorted(profiles.items())
        },
    )


def _promote(prof: Profile, week: int, to: int) -> None:
    prof.promoted.append((week, prof.level, to))
    prof.level, prof.streak = to, 0


def _demote(
    prof: Profile,
    week: int,
    precedents: list[Precedent],
    case: CaseSpec,
    dec: Decision,
    sig: Signature,
    arrival_id: str,
) -> None:
    precedents.append(
        Precedent(
            id=f"prec-{len(precedents) + 1:03d}",
            week=week,
            profile=prof.key,
            case_id=case.case_id,
            from_status=dec.status,
            to_status=sig.status,
            rationale=sig.rationale or "officer override",
        )
    )
    prof.level, prof.streak = 0, 0
    prof.demotions += 1


def by_week(res: SimResult) -> list[dict]:
    rows = []
    for week in range(1, res.config.weeks + 1):
        o = [x for x in res.outcomes if x.week == week]
        reviewed = [x for x in o if x.blocking_review or x.audited]
        agree = sum(1 for x in reviewed if not x.override)
        rows.append(
            {
                "week": week,
                "arrivals": len(o),
                "false_approvals": sum(x.false_approval for x in o),
                "auto_finalised": sum(x.auto_finalised for x in o),
                "auto_rate": round(sum(x.auto_finalised for x in o) / len(o), 4) if o else 0.0,
                "officer_blocking": sum(x.blocking_review for x in o),
                "officer_audits": sum(x.audited for x in o),
                "officer_hours": round(sum(x.latency_h for x in o), 2),
                "false_rejects": sum(x.false_reject for x in o),
                "agreement_rate": round(agree / len(reviewed), 4) if reviewed else None,
                "overrides": sum(x.override for x in o),
                "demotions": sum(1 for p in res.precedents if p.week == week),
                "promotions": sum(
                    1 for p in res.profiles.values() for pr in p["promotions"] if pr["week"] == week
                ),
            }
        )
    return rows


def totals(res: SimResult) -> dict:
    o = res.outcomes
    reviewed = [x for x in o if x.blocking_review or x.audited]
    time_to_promote = {}
    for k, p in res.profiles.items():
        for pr in p["promotions"]:
            time_to_promote.setdefault(k, []).append(
                {"to_level": pr["to"], "weeks": pr["week"] - p["first_week"], "week": pr["week"]}
            )
    by_route: dict[str, int] = defaultdict(int)
    for x in o:
        by_route[x.route] += 1
    return {
        "false_approvals": sum(x.false_approval for x in o),
        "false_approval_cases": sorted({x.arrival_id for x in o if x.false_approval}),
        "arrivals": len(o),
        "auto_finalised": sum(x.auto_finalised for x in o),
        "auto_rate": round(sum(x.auto_finalised for x in o) / len(o), 4) if o else 0.0,
        "officer_blocking": sum(x.blocking_review for x in o),
        "officer_audits": sum(x.audited for x in o),
        "officer_hours": round(sum(x.latency_h for x in o), 2),
        "false_rejects": sum(x.false_reject for x in o),
        "agreement_rate": round(sum(1 for x in reviewed if not x.override) / len(reviewed), 4)
        if reviewed
        else None,
        "accuracy_vs_gold": round(sum(x.final == x.gold for x in o) / len(o), 4) if o else None,
        "overrides": sum(x.override for x in o),
        "demotions": len(res.precedents),
        "time_to_promote_weeks": time_to_promote,
        "routes": dict(sorted(by_route.items())),
        "rejected_auto_finalised": sum(1 for x in o if x.auto_finalised and x.final == "REJECTED"),
    }
