"""Rule evaluation (E-p): facts + screening + policy + predicate answers → `RuleResult[]`.

Pure: no I/O, no clock, no model client. Every input that can vary is an argument, and every
output list is sorted, so permuting the facts cannot change the result.

`fired` is three-valued:
- `yes`: the rule's trigger is evidenced by verified facts (or by an escalating model answer);
- `no`: the facts evidence that it does not apply, or nothing in the packet evidences it;
- `unknown`: a fact it needed failed verification or was not covered, a model predicate it needed
  was not answered (or was vetoed to UNKNOWN), or the policy text it rests on is unrepresentable.
  Composition sends any `unknown` to MANUAL_REVIEW.

Model predicates only ever move a rule toward caution, except the two in `b4.FAVOURABLE`, whose
`true` can clear a trigger (sole-controller nominee check, proof-of-address eligibility); that
residual risk is held by J-SUPPORT and the evidence judges, not here.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from kyb.contracts.b4 import (
    DirectsBusinessArgs,
    Document,
    Entity,
    EvidenceRef,
    FactSet,
    PolicyModel,
    PredicateAnswer,
    PredicateInstance,
    ProofOfAddressArgs,
    RuleResult,
    RuleSpec,
    SamePartyArgs,
    ScreeningCandidate,
    ScreeningResult,
    TriFact,
)
from kyb.policy import dates, industry, regions
from kyb.policy.compose import GUARD_ID
from kyb.policy.graph import CONTROL_ROLES, Graph, Pct
from kyb.policy.parse import SECTION_KINDS
from kyb.policy.text import fold, matches

Fired = Literal["yes", "no", "unknown"]
_ORDER = {"no": 0, "unknown": 1, "yes": 2}

# What each rule kind reads from `params`; a missing key makes the rule `unknown`.
NEEDS: dict[str, tuple[str, ...]] = {
    "sanctions_corroborated": ("name_similarity_threshold",),
    "sanctions_fuzzy": ("name_similarity_threshold",),
    "jurisdiction_prohibited": (
        "prohibited_jurisdictions",
        "prohibited_regions",
        "ownership_threshold_pct",
    ),
    "jurisdiction_monitored": (
        "high_risk_jurisdictions",
        "high_risk_regions",
        "ownership_threshold_pct",
    ),
    "ubo_unresolved": ("ownership_threshold_pct",),
    "ubo_high_risk_unresolved": (
        "ownership_threshold_pct",
        "high_risk_unresolved_pct",
        "high_risk_jurisdictions",
        "high_risk_regions",
    ),
    "conflicting_filings": ("conflict_ownership_points",),
    "pep_active": ("name_similarity_threshold", "pep_lookback_years", "ownership_threshold_pct"),
    "pep_former": (
        "name_similarity_threshold",
        "pep_lookback_years",
        "ownership_threshold_pct",
        "pep_former_points",
    ),
    "nominee_control": ("ownership_threshold_pct",),
    "nominee_sole_controller": ("ownership_threshold_pct",),
    "dormant_or_layered": ("max_holding_layers",),
    "shell_indicators": ("shell_review_min", "young_company_months"),
    "shell_severe": (
        "shell_reject_min",
        "young_company_months",
        "high_risk_jurisdictions",
        "high_risk_regions",
    ),
    "required_documents": ("ubo_declaration_max_age_months", "proof_of_address_max_age_months"),
    "industry_elevated": ("high_risk_industries",),
    "industry_prohibited": ("prohibited_industries",),
    "conflict_multiplier": ("conflict_extra_points", "conflict_ownership_points"),
    "document_age_penalty": ("doc_age_penalty_months", "doc_age_points"),
}
# Sections a rule kind also rests on, besides its own (a guard there makes it unknown too).
SECTION_DEPS: dict[str, tuple[str, ...]] = {
    "ubo_high_risk_unresolved": ("monitored_jurisdictions",),
    "shell_severe": ("monitored_jurisdictions",),
    "pep_active": ("sanctions",),
    "pep_former": ("sanctions",),
    "conflict_multiplier": ("conflicts",),
    "jurisdiction_prohibited": ("ownership",),
    "jurisdiction_monitored": ("ownership",),
}
RULE_SECTION = {kind: sec for sec, kinds in SECTION_KINDS.items() for kind in kinds}
RULE_SECTION |= {"conflict_multiplier": "scoring", "document_age_penalty": "scoring"}

# Fact kinds (`witness/kinds.py`) → the rule kinds that read them.
GAP_RULES: dict[str, tuple[str, ...]] = {
    "ownership": (
        "jurisdiction_prohibited",
        "jurisdiction_monitored",
        "ubo_unresolved",
        "ubo_high_risk_unresolved",
        "conflicting_filings",
        "pep_active",
        "pep_former",
        "nominee_control",
        "nominee_sole_controller",
        "dormant_or_layered",
        "conflict_multiplier",
    ),
    "identifier": (
        "sanctions_corroborated",
        "sanctions_fuzzy",
        "ubo_unresolved",
        "ubo_high_risk_unresolved",
        "conflicting_filings",
        "pep_active",
        "pep_former",
    ),
    "date": (
        "pep_active",
        "pep_former",
        "shell_indicators",
        "shell_severe",
        "required_documents",
        "document_age_penalty",
    ),
    "address": (
        "sanctions_corroborated",
        "conflicting_filings",
        "shell_indicators",
        "shell_severe",
        "required_documents",
    ),
    "role": (
        "sanctions_corroborated",
        "sanctions_fuzzy",
        "pep_active",
        "pep_former",
        "nominee_control",
        "nominee_sole_controller",
    ),
    "amendment": ("conflicting_filings", "conflict_multiplier"),
    "corporate_activity": ("industry_elevated", "industry_prohibited", "dormant_or_layered"),
    "jurisdiction": (
        "sanctions_corroborated",
        "sanctions_fuzzy",
        "jurisdiction_prohibited",
        "jurisdiction_monitored",
        "ubo_high_risk_unresolved",
        "shell_severe",
    ),
    "financial_reference": ("shell_indicators", "shell_severe"),
    "operations": ("shell_indicators", "shell_severe", "dormant_or_layered"),
    "stated_validity": ("required_documents",),
}
# An unverified claim's evidence `field` → its kind (first match wins).
_FIELD_KINDS = (
    ("holding", "ownership"),
    ("pct", "ownership"),
    ("share", "ownership"),
    ("amend", "amendment"),
    ("conflict", "amendment"),
    ("jurisdiction", "jurisdiction"),
    ("region", "jurisdiction"),
    ("free_zone", "jurisdiction"),
    ("nationality", "identifier"),
    ("dob", "identifier"),
    ("identifier", "identifier"),
    ("incorporated", "date"),
    ("address", "address"),
    ("role", "role"),
    ("validity", "stated_validity"),
    ("issued", "date"),
    ("signed", "date"),
    ("corporate_object", "corporate_activity"),
    ("activity", "corporate_activity"),
    ("operations", "operations"),
    ("employee", "operations"),
    ("website", "operations"),
    ("bank", "financial_reference"),
    ("financ", "financial_reference"),
)

# REJECT triggers that rest on something being absent; a verification gap that could explain the
# absence makes them `unknown` (MANUAL_REVIEW), so a gap cannot manufacture a rejection.
ABSENCE_REJECTS = frozenset({"ubo_high_risk_unresolved", "shell_severe"})
POA_BY_PREDICATE = frozenset({"lease", "tax_certificate", "bank_reference", "other"})
_DORMANT_TERMS = (
    "dormant",
    "inactiv*",
    "sin actividad*",
    "sociedad inactiva",
    "en inactividad",
)


class Evaluation(BaseModel):
    """What `evaluate` returns: the rule results, and the code predicates it decided, recorded
    in the same shape as model ones so the runner can write them to `gold/predicates/`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rule_results: list[RuleResult]
    code_instances: list[PredicateInstance]
    code_answers: list[PredicateAnswer]


class _Res:
    __slots__ = ("fired", "delta", "evidence", "predicates", "notes")

    def __init__(self, fired: Fired = "no", delta: int = 0) -> None:
        self.fired: Fired = fired
        self.delta = delta
        self.evidence: list[EvidenceRef] = []
        self.predicates: list[str] = []
        self.notes: list[str] = []

    def raise_to(self, fired: Fired, note: str, evidence: Iterable[EvidenceRef] = ()) -> None:
        """yes > unknown > no: a result only ever moves toward caution."""
        if _ORDER[fired] > _ORDER[self.fired]:
            self.fired = fired
        if fired != "no":
            self.notes.append(note)
            self.evidence += list(evidence)


class _Ctx:
    def __init__(
        self,
        facts: FactSet,
        screening: ScreeningResult,
        policy: PolicyModel,
        answers: Mapping[str, PredicateAnswer],
    ) -> None:
        self.facts, self.screening, self.policy, self.answers = facts, screening, policy, answers
        self.p: dict[str, Any] = policy.params
        self.as_of = _iso(facts.as_of, "FactSet.as_of")
        doubted, doubt_all = _doubted(facts)
        self.g = Graph(
            facts,
            Decimal(str(self.p.get("ownership_threshold_pct", 100))),
            doubted=doubted,
            doubt_all=doubt_all,
        )
        self.instances: dict[str, PredicateInstance] = {}
        self.code_answers: dict[str, PredicateAnswer] = {}

    def code(
        self,
        name: str,
        subject: str,
        value: bool | None,
        refs: Iterable[EvidenceRef],
        rationale: str,
        **args: Any,
    ) -> str:
        """Record a code predicate (date window, threshold, list membership) and return its id."""
        iid = f"cp-{name}-{subject}"
        ids = sorted({r.claim_id for r in refs if r.verified})
        self.instances[iid] = PredicateInstance(
            instance_id=iid,
            predicate=name,
            kind="code",
            args={"subject": subject, **{k: _jsonable(v) for k, v in sorted(args.items())}},
            evidence_universe=ids,
        )
        self.code_answers[iid] = PredicateAnswer(
            instance_id=iid,
            value="UNKNOWN" if value is None else value,
            confidence=1.0,
            evidence=ids,
            rationale=rationale[:600],
        )
        return iid

    def model(self, iid: str) -> bool | None:
        a = self.answers.get(iid)
        return None if a is None or a.value == "UNKNOWN" else bool(a.value)

    def jur(self, e: Entity | None, plain_key: str, regions_key: str) -> Fired:
        """Is `e` in a listed jurisdiction? A qualified entry (UA's named regions, AE's free zones)
        reads the regions the documents state for it; none stated reads `unknown`."""
        if e is None:
            return "no"
        plain, qualified = set(self.p[plain_key]), self.p[regions_key]
        codes = {c.upper() for c in e.jurisdiction}
        if codes & plain:
            return "yes"
        states = [
            regions.classify(k, qualified[k], e.regions) for k in sorted(codes & set(qualified))
        ]
        return max(states, key=_ORDER.__getitem__, default="no")


def _iso(value: str, what: str) -> date:
    d = dates.parse(value)
    if d is None:
        raise ValueError(f"{what} is not an ISO date: {value!r}")
    return d


def _jsonable(v: Any) -> Any:
    if isinstance(v, Decimal):
        return str(v)
    if isinstance(v, (list, tuple, set, frozenset)):
        return sorted(_jsonable(x) for x in v)
    return v


# --- sanctions & PEP screening ---------------------------------------------------------------


def _scope_sanctions(c: _Ctx) -> set[str]:
    """The entity, its UBOs, directors and every shareholder above it (corporate or not)."""
    g = c.g
    out = {g.app.entity_id, *g.chain(), *(r.holder for r in g.directors())}
    for eid in list(out):
        out |= {r.holder for r in g.controllers_of(eid)}
    return out


def _name_match(c: _Ctx, cand: ScreeningCandidate) -> bool:
    return cand.alias_exact or cand.jw_score >= float(c.p["name_similarity_threshold"])


def eval_sanctions(c: _Ctx) -> tuple[_Res, _Res]:
    a, b = _Res(), _Res()
    scope = _scope_sanctions(c)
    for cand in sorted(c.screening.candidates, key=lambda x: x.candidate_id):
        if cand.entry.list_kind != "sanctions":
            continue
        refs = c.g.evidence_of(cand.entity_id)
        if cand.entity_id not in scope:
            b.notes.append(f"{cand.candidate_id}: {cand.entity_id} is outside sanctions scope")
            continue
        name = _name_match(c, cand)
        corr, conf = cand.identifiers.corroborating, cand.identifiers.conflicting
        sp = f"sp-{cand.candidate_id}"
        same = c.model(sp)
        cp = c.code(
            "name_match",
            cand.candidate_id,
            name,
            refs,
            f"JW {cand.jw_score:.3f} vs {c.p['name_similarity_threshold']}, "
            f"alias_exact={cand.alias_exact}",
            jw_score=cand.jw_score,
            threshold=c.p["name_similarity_threshold"],
        )
        tag = (
            f"{cand.candidate_id} {cand.entry.entry_id} ({cand.matched_name}, "
            f"JW {cand.jw_score:.3f}); corroborating {corr or 'none'}, "
            f"conflicting {conf or 'none'}"
        )
        if name and corr:
            why = "; another identifier conflicts" if conf else ""
            a.raise_to("yes", f"corroborated match {tag}{why}", refs)
            a.predicates.append(cp)
        if name and (not corr or conf):
            b.raise_to("yes", f"name-only similarity {tag}", refs)
            b.predicates.append(cp)
        if not name:
            b.predicates.append(sp)
            if same is True:
                b.raise_to("yes", f"same_party: the same party (escalation): {tag}", refs)
            elif same is None:
                b.raise_to("unknown", f"{sp} unanswered or UNKNOWN: {tag}", refs)
            else:
                b.notes.append(f"below threshold, same_party says a different party: {tag}")
    return a, b


def _scope_pep(c: _Ctx) -> dict[str, Fired]:
    g = c.g
    out: dict[str, Fired] = dict(g.ubos())
    for r in g.directors():
        out[r.holder] = "yes"
    for owner, p in g.owners(g.app.entity_id).items():
        reach = g.at_or_above(p.hi)
        if reach != "no" and out.get(owner) != "yes":
            out[owner] = reach
    return out


def eval_pep(c: _Ctx) -> tuple[_Res, _Res]:
    cur, former = _Res(), _Res()
    scope = _scope_pep(c)
    window = dates.years_before(c.as_of, int(c.p["pep_lookback_years"]))
    for cand in sorted(c.screening.candidates, key=lambda x: x.candidate_id):
        if cand.entry.list_kind != "pep":
            continue
        refs = c.g.evidence_of(cand.entity_id)
        corr, conf = cand.identifiers.corroborating, cand.identifiers.conflicting
        tag = (
            f"{cand.candidate_id} {cand.entry.entry_id} ({cand.matched_name}, "
            f"JW {cand.jw_score:.3f}); corroborating {corr or 'none'}, "
            f"conflicting {conf or 'none'}"
        )
        name = _name_match(c, cand)
        match: Fired = "yes" if name and corr else "no"
        sp = f"sp-{cand.candidate_id}"
        used = []
        if match != "yes":
            same = c.model(sp)
            used.append(sp)
            if same is True:
                match = "yes"
            elif same is None:
                match = "unknown"
        if match == "no":
            cur.notes.append(f"screening note, not a PEP match: {tag}")
            continue
        reach = scope.get(cand.entity_id, "no")
        if reach == "no":
            cur.notes.append(
                f"{cand.entity_id} is not a UBO, director or ≥ threshold holder: {tag}"
            )
            continue
        e = cand.entry
        left = dates.parse(e.left_office)
        if e.in_office is True:
            within: bool | None = True
        elif e.in_office is False and left is not None:
            within = left >= window
        else:
            within = None
        cp = c.code(
            "pep_within_window",
            cand.candidate_id,
            within,
            refs,
            f"in_office={e.in_office}, left_office={e.left_office}, window from {window}",
            left_office=e.left_office,
            in_office=e.in_office,
            window_start=window.isoformat(),
        )
        state: Fired = "unknown" if "unknown" in (match, reach) else "yes"
        if within is None:
            cur.raise_to("unknown", f"office dates unknown: {tag}", refs)
            cur.predicates += [cp, *used]
        elif within:
            cur.raise_to(state, f"PEP (left {e.left_office or 'in office'}): {tag}", refs)
            cur.predicates += [cp, *used]
        else:
            former.raise_to(state, f"former PEP, left {e.left_office}: {tag}", refs)
            former.predicates += [cp, *used]
    cur.delta = _spec_delta(c, "pep_active") if cur.fired == "yes" else 0
    former.delta = _spec_delta(c, "pep_former") if former.fired == "yes" else 0
    return cur, former


# --- jurisdictions & ownership ---------------------------------------------------------------


def eval_prohibited_jurisdictions(c: _Ctx) -> _Res:
    res, g = _Res(), c.g
    keys = ("prohibited_jurisdictions", "prohibited_regions")
    state = c.jur(g.app, *keys)
    res.raise_to(state, f"applicant incorporated in {_where(g.app)}", g.app.evidence)
    for eid, reach in sorted(g.ubos().items()):
        e = g.ents[eid]
        hit = c.jur(e, *keys)
        if hit != "no":
            res.raise_to(
                "unknown" if "unknown" in (hit, reach) else "yes",
                f"UBO {eid} resident in {_where(e)}",
                e.evidence,
            )
    return res


def eval_monitored_jurisdictions(c: _Ctx) -> _Res:
    res, g = _Res(), c.g
    keys = ("high_risk_jurisdictions", "high_risk_regions")
    res.raise_to(c.jur(g.app, *keys), f"applicant incorporated in {_where(g.app)}", g.app.evidence)
    for eid, stake in sorted(g.chain().items()):
        e = g.ents.get(eid)
        if e is None:
            continue
        hit, reach = c.jur(e, *keys), g.at_or_above(stake)
        if hit != "no" and reach != "no":
            direct = g.owners(g.app.entity_id).get(eid, Pct(None, None, []))
            refs = [*e.evidence, *(r for h in direct.holdings for r in h.evidence)]
            res.raise_to(
                "unknown" if "unknown" in (hit, reach) else "yes",
                f"owner {eid} ({_where(e)}) holds {_fmt(stake)}%",
                refs,
            )
    if res.fired == "yes":
        res.delta = _spec_delta(c, "jurisdiction_monitored")
    return res


def eval_ownership(c: _Ctx) -> tuple[_Res, _Res]:
    a, b = _Res(), _Res()
    g = c.g
    app = g.app.entity_id
    hr_keys = ("high_risk_jurisdictions", "high_risk_regions")
    hr: list[Fired] = [c.jur(g.app, *hr_keys)]
    for owner, p in sorted(g.owners(app).items()):
        e = g.ents.get(owner)
        hr.append(c.jur(e, *hr_keys))
        state = g.resolution(owner)
        if state == "yes":
            continue
        reach = g.at_or_above(p.hi)
        if reach != "no":
            refs = [*(r for h in p.holdings for r in h.evidence), *g.resolution_evidence(owner)]
            fired: Fired = "yes" if state == "no" and reach == "yes" else "unknown"
            a.raise_to(
                fired,
                f"{owner} holds {_fmt(p.hi)}% and has no identified natural person "
                f"(nationality and date of birth) behind it",
                refs,
            )
    lo, hi = g.unresolved_share(app)
    if hi >= g.t:
        refs = [*g.app.evidence, *g.resolution_evidence(app)]
        a.raise_to(
            "yes" if lo >= g.t else "unknown",
            f"{_fmt(lo)}–{_fmt(hi)}% of the ownership in total does not reach resolved "
            f"natural persons (aggregate)",
            refs,
        )
    hr_min = Decimal(str(c.p["high_risk_unresolved_pct"]))
    worst = max(hr, key=_ORDER.get)
    if worst != "no" and hi >= hr_min:
        sure = lo >= hr_min and worst == "yes"
        b.raise_to(
            "yes" if sure else "unknown",
            f"{_fmt(lo)}–{_fmt(hi)}% unresolved with a monitored jurisdiction",
            g.app.evidence,
        )
    return a, b


# --- documentary conflicts & multiplier ------------------------------------------------------


def _material(c: _Ctx) -> list:
    points = Decimal(str(c.p["conflict_ownership_points"]))
    out = []
    for x in sorted(c.facts.conflicts, key=lambda x: x.conflict_id):
        if x.attribute in ("ownership_pct", "ownership"):
            try:
                nums = [Decimal(v.value.strip().rstrip("%").strip()) for v in x.values]
            except InvalidOperation:
                nums = []
            if nums and max(nums) - min(nums) <= points:
                continue
        out.append(x)
    return out


def eval_conflicts(c: _Ctx) -> tuple[_Res, _Res]:
    r, s = _Res(), _Res()
    found = _material(c)
    for x in found:
        refs = [ref for v in x.values for ref in v.evidence]
        vals = " vs ".join(v.value for v in x.values)
        r.raise_to("yes", f"{x.conflict_id}: {x.subject} {x.attribute} {vals}", refs)
    if len(found) >= 2:
        s.raise_to(
            "yes",
            f"{len(found)} unresolved conflicts: {len(found) - 1} beyond the first",
            [ref for x in found for v in x.values for ref in v.evidence],
        )
        s.delta = int(c.p["conflict_extra_points"]) * (len(found) - 1)
    return r, s


# --- governance, nominees & dormancy/layering ------------------------------------------------


def eval_governance(c: _Ctx) -> tuple[_Res, _Res, _Res]:
    score, sole, dormant = _Res(), _Res(), _Res()
    g = c.g
    if "max_holding_layers" in c.p:
        max_layers = int(c.p["max_holding_layers"])
        depth, layer_refs = g.intermediate_layers()
        if depth >= max_layers:
            dormant.raise_to(
                "yes",
                f"{depth} intermediate corporate holding layers (threshold {max_layers})",
                layer_refs or g.app.evidence,
            )
        for act in sorted(c.facts.corporate_object):
            text = fold(act)
            if any(matches(t, text) for t in _DORMANT_TERMS):
                dormant.raise_to(
                    "yes", f"declared dormant or inactive status: {act}", g.app.evidence
                )
        if dormant.fired == "yes":
            dormant.delta = _spec_delta(c, "dormant_or_layered")

    roles, holdings = g.nominees()
    if not roles and not holdings:
        return score, sole, dormant
    refs = [
        *(r for role in roles for r in role.evidence),
        *(r for h in holdings for r in h.evidence),
    ]
    who = sorted({*(r.holder for r in roles), *(h.owner for h in holdings)})
    score.raise_to("yes", f"nominee or agent acting as director/shareholder: {who}", refs)
    score.delta = _spec_delta(c, "nominee_control")
    nominees = set(who)
    natural = sorted(
        {
            r.holder
            for r in g.roles_of.get(g.app.entity_id, [])
            if r.role in CONTROL_ROLES and r.holder not in nominees and g.kind(r.holder) == "person"
        }
    )
    if natural:
        sole.notes.append(f"natural persons named as controllers: {natural}")
        return score, sole, dormant
    iid = f"npd-{g.app.entity_id}"
    sole.predicates.append(iid)
    directs = c.model(iid)
    if directs is True:
        sole.notes.append("a named individual is shown to run the business")
    elif directs is False:
        sole.raise_to("yes", "the nominee is the sole controller on record", refs)
    else:
        sole.raise_to("unknown", f"{iid} unanswered or UNKNOWN", refs)
    return score, sole, dormant


# --- shell indicators ------------------------------------------------------------------------


def _inverse(t: TriFact) -> Fired:
    """An indicator is the *absence* of the fact (no operating address, no bank reference...)."""
    return {"present": "no", "absent": "yes", "unknown": "unknown"}[t.state]  # type: ignore[return-value]  # noqa: E501


def eval_operations(c: _Ctx) -> tuple[_Res, _Res]:
    review, reject = _Res(), _Res()
    g, ops = c.g, c.facts.operations
    regs = [a for a in g.app.addresses if a.kind == "registered"]
    states = [a.mass_registration.state for a in regs]
    i: Fired = "yes" if "present" in states else "unknown" if "unknown" in states else "no"
    young = dates.months_before(c.as_of, int(c.p["young_company_months"]))
    inc = sorted({d for d in (dates.parse(x) for x in g.app.incorporated) if d is not None})
    verdicts = {d > young for d in inc}
    iii: Fired = "unknown" if len(verdicts) > 1 else "yes" if verdicts == {True} else "no"
    c.code(
        "young_company",
        g.app.entity_id,
        None if iii == "unknown" else iii == "yes",
        g.app.evidence,
        f"incorporated {[d.isoformat() for d in inc]}, cutoff {young}",
        incorporated=[d.isoformat() for d in inc],
        cutoff=young.isoformat(),
    )
    ind = {
        "i": (i, [r for a in regs for r in [*a.evidence, *a.mass_registration.evidence]]),
        "ii": (
            _inverse(ops.distinct_operating_address),
            ops.distinct_operating_address.evidence,
        ),
        "iii": (iii, g.app.evidence),
        "iv": (
            _inverse(ops.bank_reference_or_financials),
            ops.bank_reference_or_financials.evidence,
        ),
        "v": (_inverse(ops.employees_or_website), ops.employees_or_website.evidence),
    }
    present = [k for k, (s, _) in ind.items() if s == "yes"]
    maybe = [k for k, (s, _) in ind.items() if s == "unknown"]
    refs = [r for k in present for r in ind[k][1]]
    desc = f"indicators present {present}, unknown {maybe}"
    rmin, jmin = int(c.p["shell_review_min"]), int(c.p["shell_reject_min"])
    if len(present) >= rmin:
        review.raise_to("yes", desc, refs)
    elif len(present) + len(maybe) >= rmin:
        review.raise_to("unknown", desc, refs)
    else:
        review.notes.append(desc)
    hr = c.jur(g.app, "high_risk_jurisdictions", "high_risk_regions")
    if hr != "no":
        if len(present) >= jmin and hr == "yes":
            reject.raise_to("yes", f"{desc}; applicant in {g.app.jurisdiction}", refs)
        elif len(present) + len(maybe) >= jmin:
            reject.raise_to("unknown", f"{desc}; applicant in {g.app.jurisdiction}", refs)
    return review, reject


# --- required documents ----------------------------------------------------------------------


def eval_documents(c: _Ctx) -> _Res:
    res = _Res()
    docs = sorted(c.facts.documents, key=lambda d: (d.doc, d.document_name))
    as_of = c.as_of
    app = next((e for e in c.facts.entities if e.entity_id == c.facts.applicant), None)
    fallback = list(app.evidence) if app else []

    def _with_fallback(ev_iterable):
        ev_list = list(ev_iterable)
        return ev_list if ev_list else fallback

    def fresh(d: Document, months: int) -> bool | None:
        when, gone = dates.doc_date(d), dates.lapsed(d, as_of)
        if when is None or when > as_of or gone is None:
            if when is not None:
                c.code(
                    "doc_fresh",
                    d.doc,
                    None,
                    d.evidence,
                    f"{d.doc_type} dated {when}; after as_of or validity not evaluable",
                    dated=when.isoformat(),
                    months=months,
                    stated_validity_days=d.stated_validity_days,
                )
            return None
        ok = when >= dates.months_before(as_of, months) and not gone
        c.code(
            "doc_fresh",
            d.doc,
            ok,
            d.evidence,
            f"{d.doc_type} dated {when}, window {months} months, "
            f"stated validity {d.stated_validity_days}",
            dated=when.isoformat(),
            months=months,
            stated_validity_days=d.stated_validity_days,
        )
        return ok

    incs = [d for d in docs if d.doc_type == "incorporation"]
    for d in incs:
        c.code(
            "doc_lapsed",
            d.doc,
            dates.lapsed(d, as_of),
            d.evidence,
            f"issued {d.issued}, stated validity {d.stated_validity_days} days",
            issued=d.issued,
            stated_validity_days=d.stated_validity_days,
        )

    def cert_state(d: Document) -> bool | None:
        issued = dates.parse(d.issued)
        if issued is not None and issued > as_of:
            return None
        return dates.lapsed(d, as_of)

    gone = [cert_state(d) for d in incs]
    if not incs:
        res.raise_to("yes", "missing: certificate of incorporation", fallback)
    elif False in gone:
        pass
    elif None in gone:
        res.raise_to(
            "unknown",
            "certificate of incorporation: issued after as_of, or a stated "
            "validity without an issue date",
            _with_fallback(r for d in incs for r in d.evidence),
        )
    else:
        which = ", ".join(
            f"{d.doc} issued {d.issued}, valid {d.stated_validity_days} days" for d in incs
        )
        res.raise_to(
            "yes",
            f"expired: certificate of incorporation ({which})",
            _with_fallback(r for d in incs for r in d.evidence),
        )

    ubo_m = int(c.p["ubo_declaration_max_age_months"])
    ubos = [d for d in docs if d.doc_type == "ubo_declaration"]
    verdicts = [(d, fresh(d, ubo_m), d.signed) for d in ubos]
    if any(f is True and s is True for _, f, s in verdicts):
        pass
    elif any(f is not False and s is not False for _, f, s in verdicts):
        res.raise_to(
            "unknown",
            "UBO declaration: date or signature not established",
            _with_fallback(r for d in ubos for r in d.evidence),
        )
    else:
        what = "missing" if not ubos else "unsigned or older than the window"
        res.raise_to(
            "yes",
            f"{what}: executed UBO declaration dated within {ubo_m} months",
            _with_fallback(r for d in ubos for r in d.evidence),
        )

    poa_m = int(c.p["proof_of_address_max_age_months"])
    direct = [d for d in docs if d.doc_type == "proof_of_address"]
    via = [d for d in docs if d.doc_type in POA_BY_PREDICATE]
    direct_fresh = [fresh(d, poa_m) for d in direct]
    if True in direct_fresh:
        return res
    undetermined = None in direct_fresh
    for d in via:
        f = fresh(d, poa_m)
        if f is False:
            continue
        iid = f"poa-{d.doc}"
        res.predicates.append(iid)
        is_poa = c.model(iid)
        if f is True and is_poa is True:
            return res
        if is_poa is not False:
            undetermined = True
    candidates = direct + via
    if undetermined:
        res.raise_to(
            "unknown",
            "proof of address: date or eligibility not established",
            _with_fallback(r for d in candidates for r in d.evidence),
        )
    else:
        what = "missing or older than the window" if candidates else "missing"
        res.raise_to(
            "yes",
            f"{what}: proof of address dated within {poa_m} months",
            _with_fallback(r for d in candidates for r in d.evidence),
        )
    return res


# --- sector & industry -----------------------------------------------------------------------


def eval_industry(c: _Ctx) -> tuple[_Res, _Res]:
    score, prohibited = _Res(), _Res()
    activities = sorted(c.facts.corporate_object)
    refs = c.g.app.evidence
    for res, key, strict in (
        (prohibited, "prohibited_industries", True),
        (score, "high_risk_industries", False),
    ):
        for label in c.p[key]:
            if industry.key(label) not in industry.LEXICON:
                res.raise_to("unknown", f"no lexicon for {label!r}", refs)
                continue
            named, hedged = industry.classify(label, activities)
            value = True if named else None if hedged else False
            cp = c.code(
                "industry",
                industry.key(label).replace(" ", "_"),
                value,
                refs,
                f"{label}: named {named}, under negation {hedged}",
                label=label,
                named=named,
                hedged=hedged,
            )
            if named:
                licence = strict and industry.needs_licence(label)
                res.raise_to("unknown" if licence else "yes", f"{label}: {named}", refs)
                res.predicates.append(cp)
            elif hedged:
                res.raise_to(
                    "unknown",
                    f"{label} named under a negation or exclusion: {hedged}",
                    refs,
                )
                res.predicates.append(cp)
    if score.fired == "yes":
        score.delta = _spec_delta(c, "industry_elevated")
    return prohibited, score


# --- optional document age penalty -----------------------------------------------------------


def eval_doc_age(c: _Ctx) -> _Res:
    res = _Res()
    cutoff = dates.months_before(c.as_of, int(c.p["doc_age_penalty_months"]))
    for d in sorted(c.facts.documents, key=lambda d: d.doc):
        when = dates.doc_date(d)
        if when is not None and when < cutoff:
            res.raise_to("yes", f"{d.doc} ({d.doc_type}) dated {when}, before {cutoff}", d.evidence)
    if res.fired == "yes":
        res.delta = int(c.p["doc_age_points"])
    return res


# --- assembly --------------------------------------------------------------------------------


def _where(e: Entity) -> str:
    return f"{e.jurisdiction} {e.regions}" if e.regions else str(e.jurisdiction)


def _fmt(v: Decimal | None) -> str:
    return "?" if v is None else format(v.normalize(), "f")


def _spec_delta(c: _Ctx, kind: str) -> int:
    spec = next((s for s in c.policy.rules if (s.kind or s.rule_id) == kind), None)
    return spec.score_delta if spec else 0


def fact_rules(kind: str) -> tuple[str, ...]:
    """The rule kinds that read an uncovered material fact of this kind ("*": every rule)."""
    return GAP_RULES.get(kind, ("*",))


def claim_rules(field: str) -> tuple[str, ...]:
    """The rule kinds that read a claim whose evidence has this `field` ("*": every rule, because
    the field names no kind the policy reads)."""
    kind = next((k for frag, k in _FIELD_KINDS if frag in field.lower()), None)
    return GAP_RULES.get(kind, ("*",)) if kind else ("*",)


def _gaps(facts: FactSet) -> dict[str, list[str]]:
    """rule kind (or "*" for every rule) → why a fact it may read is unverified or uncovered."""
    out: dict[str, list[str]] = {}
    fields = {e.evidence_id: e.field for e in facts.evidence}
    for v in facts.verification:
        for u in v.uncovered:
            for k in fact_rules(u.kind):
                out.setdefault(k, []).append(
                    f"doc {v.doc}: uncovered {u.kind} {u.subject}.{u.attribute}"
                )
    for ref in facts.unverified:
        for k in claim_rules(fields.get(ref.claim_id) or ""):
            out.setdefault(k, []).append(
                f"doc {ref.doc} p.{ref.page}: claim {ref.claim_id} failed verification"
            )
    return out


_IDENTITY_KINDS = frozenset({"identifier", "ownership", "jurisdiction", "role"})
_SUBJECT = re.compile(r"\[([^\]]+)\]")


def _doubted(facts: FactSet) -> tuple[frozenset[str], bool]:
    """Entities whose identity or ownership facts have a gap, and whether some gap cannot be tied
    to one entity. Ownership resolution reads their absences as `unknown`, not as unresolved, so
    a gap can never manufacture an unresolved-ownership trigger."""
    ents = {e.entity_id for e in facts.entities}
    fields = {e.evidence_id: e.field for e in facts.evidence}
    out, everyone = set(), False
    for v in facts.verification:
        for u in v.uncovered:
            if u.kind in _IDENTITY_KINDS or u.kind not in GAP_RULES:
                if u.subject in ents:
                    out.add(u.subject)
                else:
                    everyone = True
    for ref in facts.unverified:
        field = (fields.get(ref.claim_id) or "").lower()
        kind = next((k for frag, k in _FIELD_KINDS if frag in field), None)
        if kind is not None and kind not in _IDENTITY_KINDS:
            continue
        subject = _SUBJECT.search(fields.get(ref.claim_id) or "")
        if subject and subject.group(1) in ents:
            out.add(subject.group(1))
        else:
            everyone = True
    return frozenset(out), everyone


def _guards(policy: PolicyModel) -> tuple[dict[str, list[str]], list[str]]:
    """unrepresentable → (rule_kind → reasons, global reasons)."""
    present = {s.kind or s.rule_id for s in policy.rules}
    per_kind: dict[str, list[str]] = {}
    glob: list[str] = []
    for item in policy.unrepresentable:
        sec = item.split(":", 1)[0]
        if sec in SECTION_KINDS:
            own = [k for k in SECTION_KINDS[sec] if k in present]
            deps = [k for k, ds in SECTION_DEPS.items() if sec in ds and k in present]
            hit = [*own, *deps]
            if not own:
                glob.append(item)
        else:
            glob.append(item)
            hit = (
                [k for k in ("conflict_multiplier", "document_age_penalty") if k in present]
                if sec == "scoring"
                else []
            )
        for k in hit:
            per_kind.setdefault(k, []).append(item)
    return per_kind, glob


def model_instances(
    facts: FactSet, screening: ScreeningResult, policy: PolicyModel
) -> list[PredicateInstance]:
    """The model predicates the rules will read, with the ids `evaluate` looks up.

    `same_party` per screening candidate (`sp-<candidate_id>`, escalate-only);
    `natural_person_directs_business` when a nominee may be the only controller
    (`npd-<applicant>`); `doc_is_proof_of_address` for each in-window document that can be
    accepted as proof of address by judgement (`poa-<doc>`).
    """
    c = _Ctx(facts, screening, policy, {})
    verified = sorted({e.evidence_id for e in facts.evidence if e.verified})

    def ids(refs: Iterable[EvidenceRef]) -> list[str]:
        return sorted({r.claim_id for r in refs if r.verified})

    out: list[PredicateInstance] = []
    for cand in sorted(screening.candidates, key=lambda x: x.candidate_id):
        e = c.g.ents.get(cand.entity_id)
        if e is None:
            continue
        out.append(
            PredicateInstance(
                instance_id=f"sp-{cand.candidate_id}",
                predicate="same_party",
                kind="model",
                args=SamePartyArgs(candidate_id=cand.candidate_id, entity=e, entry=cand.entry),
                evidence_universe=ids(e.evidence),
            )
        )
    if "ownership_threshold_pct" in c.p:
        _, sole, _ = eval_governance(c)
        if f"npd-{c.g.app.entity_id}" in sole.predicates:
            persons = [
                e for e in sorted(facts.entities, key=lambda e: e.entity_id) if e.kind == "person"
            ]
            out.append(
                PredicateInstance(
                    instance_id=f"npd-{c.g.app.entity_id}",
                    predicate="natural_person_directs_business",
                    kind="model",
                    args=DirectsBusinessArgs(
                        company=c.g.app.entity_id,
                        persons=persons,
                        roles=sorted(facts.roles, key=lambda r: r.role_id),
                    ),
                    evidence_universe=verified,
                )
            )
    if "proof_of_address_max_age_months" in c.p and "ubo_declaration_max_age_months" in c.p:
        for iid in eval_documents(c).predicates:
            doc = next(d for d in facts.documents if f"poa-{d.doc}" == iid)
            addr = [r for a in c.g.app.addresses for r in a.evidence]
            out.append(
                PredicateInstance(
                    instance_id=iid,
                    predicate="doc_is_proof_of_address",
                    kind="model",
                    args=ProofOfAddressArgs(document=doc),
                    evidence_universe=ids([*doc.evidence, *addr]),
                )
            )
    return out


def evaluate(
    facts: FactSet,
    screening: ScreeningResult,
    policy: PolicyModel,
    answers: Mapping[str, PredicateAnswer] | Iterable[PredicateAnswer] = (),
) -> Evaluation:
    """Evaluate every rule the policy defines. `answers` are the recorded model-predicate answers
    after the judges (a vetoed answer arrives as UNKNOWN); a missing answer reads as UNKNOWN."""
    if screening.fact_set_hash != facts.fact_set_hash or screening.case != facts.case:
        raise ValueError("screening was run on a different FactSet")
    if not isinstance(answers, Mapping):
        answers = {a.instance_id: a for a in answers}
    c = _Ctx(facts, screening, policy, answers)
    per_kind, glob = _guards(policy)
    gaps = _gaps(facts)
    have = set(c.p)
    present_kinds = {s.kind or s.rule_id for s in policy.rules}

    evaluators: list[tuple[tuple[str, ...], Any]] = [
        (("sanctions_corroborated", "sanctions_fuzzy"), eval_sanctions),
        (("jurisdiction_prohibited",), eval_prohibited_jurisdictions),
        (("jurisdiction_monitored",), eval_monitored_jurisdictions),
        (("ubo_unresolved", "ubo_high_risk_unresolved"), eval_ownership),
        (("conflicting_filings", "conflict_multiplier"), eval_conflicts),
        (("pep_active", "pep_former"), eval_pep),
        (("nominee_control", "nominee_sole_controller", "dormant_or_layered"), eval_governance),
        (("shell_indicators", "shell_severe"), eval_operations),
        (("required_documents",), eval_documents),
        (("industry_prohibited", "industry_elevated"), eval_industry),
        (("document_age_penalty",), eval_doc_age),
    ]
    computed: dict[str, _Res] = {}
    for kinds, fn in evaluators:
        if not any(k in present_kinds for k in kinds):
            continue
        if any(set(NEEDS.get(k, ())) - have for k in kinds if k in present_kinds):
            for k in kinds:
                computed[k] = _Res("unknown")
                computed[k].notes.append(
                    f"policy parameters missing: {sorted(set(NEEDS.get(k, ())) - have)}"
                )
            continue
        out = fn(c)
        for k, res in zip(kinds, out if isinstance(out, tuple) else (out,), strict=True):
            computed[k] = res

    results = []
    for spec in policy.rules:
        rid = spec.rule_id
        kind = spec.kind or spec.rule_id
        res = computed.get(kind, _Res("unknown"))
        if kind not in computed:
            res.notes.append(f"unrecognised rule kind {kind!r}")
        if kind in per_kind:
            res.fired, res.delta = "unknown", 0
            res.notes.append("policy text unrepresentable: " + "; ".join(per_kind[kind]))
        gap = kind in gaps or rid in gaps or "*" in gaps
        if gap and (res.fired == "no" or (res.fired == "yes" and kind in ABSENCE_REJECTS)):
            res.fired = "unknown"
            why = gaps.get(kind, []) + gaps.get(rid, []) + gaps.get("*", [])
            res.notes.append("needed facts not verified: " + "; ".join(sorted(set(why))[:8]))
        results.append(_result(rid, spec, res))
    if glob:
        why = "policy text unrepresentable: " + "; ".join(glob)
        results.append(
            RuleResult(rule_id=GUARD_ID, severity="REVIEW", fired="unknown", description=why)
        )
    return Evaluation(
        rule_results=results,
        code_instances=[c.instances[k] for k in sorted(c.instances)],
        code_answers=[c.code_answers[k] for k in sorted(c.code_answers)],
    )


def _result(rid: str, spec: RuleSpec | None, res: _Res) -> RuleResult:
    refs = {(r.doc, r.page, r.claim_id, r.verified): r for r in res.evidence}
    desc = "; ".join(dict.fromkeys(res.notes)) or "not triggered"
    return RuleResult(
        rule_id=rid,
        severity=spec.severity if spec else "REVIEW",
        fired=res.fired,
        score_delta=(res.delta or (spec.score_delta if spec else 0)) if res.fired == "yes" else 0,
        predicates=sorted(set(res.predicates)),
        evidence=[refs[k] for k in sorted(refs)],
        description=desc[:2000],
    )
