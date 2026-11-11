"""The deterministic label engine: policy rules and score thresholds applied to a CaseSpec.

It reads `compliance_rules.md` as written and nothing else: no model, no network, no clock (the
case carries its own `as_of`). It is independent of `kyb.policy`, which evaluates typed FactSets,
so the two can be compared (`FixtureDecider`) instead of one vouching for the other. The only code
shared is the name-similarity metric (`kyb.screening.normalize.policy_score`).
"""

from __future__ import annotations

import json
import re
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from kyb.policy.parse import load_policy
from kyb.screening.normalize import d044, policy_score
from kyb.synth.case import CaseSpec, Holder, Person

ROOT = Path(__file__).resolve().parents[3]
WATCHLISTS = ROOT / "examples/policy/watchlists.json"
POLICY_FILE = ROOT / "examples/policy/compliance_rules.md"

PROHIBITED_INDUSTRY = {
    "unlicensed banking or money transmission": (
        "unlicensed bank",
        "banca no autorizada",
        "captacion ilegal",
    ),
    "shell bank": ("banco pantalla", "shell bank", "banco fantasma"),
    "adult entertainment": (
        "contenido para adultos",
        "entretenimiento para adultos",
        "adult content",
        "adult entertainment",
        "pornograf",
    ),
    "online gambling": ("casino", "apuestas", "gambling", "betting", "juegos de azar"),
    "weapons or defence brokerage": (
        "armas de fuego",
        "armamento",
        "municiones",
        "weapons",
        "firearms",
        "ammunition",
        "defense contract",
        "defence broker",
    ),
}
UNLICENSED_INDUSTRY = {
    "unlicensed banking or money transmission": (
        "unlicensed bank",
        "banca no autorizada",
        "captacion ilegal",
    ),
}
HIGH_RISK_INDUSTRY = {
    "money service business": (
        "remesas",
        "casa de cambio",
        "money transfer",
        "money service",
        "transferencia de dinero",
        "servicios de pago",
    ),
    "virtual asset or crypto exchange": (
        "activos virtuales",
        "criptomoneda",
        "virtual asset",
        "crypto exchange",
        "intercambio de activos virtuales",
    ),
    "precious metals or stones trading": (
        "metales preciosos",
        "piedras preciosas",
        "joyeria",
        "precious metal",
        "diamantes",
    ),
    "maritime vessel chartering": (
        "fletamento maritimo",
        "fletamento de buques",
        "vessel charter",
        "maritime charter",
        "arrendamiento de naves",
    ),
    "trade finance": (
        "trade finance",
        "financiamiento comercial",
        "cartas de credito",
        "financiamiento de comercio exterior",
    ),
    "charitable foundation administration": (
        "sin fines de lucro",
        "beneficencia",
        "charit",
        "administracion de fundaciones",
    ),
}

Status = Literal["APPROVED", "REJECTED", "MANUAL_REVIEW"]


@lru_cache(maxsize=1)
def _policy():
    return load_policy(POLICY_FILE)


def _by_kind() -> dict[str, tuple[str, str, int]]:
    pol = _policy()
    return {r.kind: (r.rule_id, r.severity, r.score_delta) for r in pol.rules}


class Fired(BaseModel):
    model_config = ConfigDict(frozen=True)
    rule: str
    severity: str
    points: int = 0
    reason: str


class Label(BaseModel):
    model_config = ConfigDict(frozen=True)
    status: Status
    score: int
    fired: list[Fired]
    conflicts: int = 0
    unknown: list[str] = []
    notes: list[str] = []

    @property
    def rules(self) -> list[str]:
        return sorted({f.rule for f in self.fired})


def fold(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", d044(text))


def add_months(d: date, months: int) -> date:
    y, m = divmod(d.year * 12 + d.month - 1 + months, 12)
    m += 1
    last = [
        31,
        29 if y % 4 == 0 and (y % 100 or y % 400 == 0) else 28,
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ][m - 1]
    return date(y, m, min(d.day, last))


def code_of(domicile: str) -> str:
    return domicile.split(":", 1)[0].upper()


def region_of(domicile: str) -> str:
    return domicile.split(":", 1)[1].lower() if ":" in domicile else ""


def prohibited_place(code: str, region: str) -> bool:
    p = _policy().params
    ua_regs = {"crimea", "donetsk", "luhansk", "zaporizhzhia", "kherson"}
    return code in set(p["prohibited_jurisdictions"]) or (code == "UA" and region in ua_regs)


def high_risk_place(code: str, region: str) -> bool:
    p = _policy().params
    return (
        code in set(p["high_risk_jurisdictions"])
        or (code == "AE" and "free zone" in region.lower())
        or (
            code == "AE"
            and region.lower() in {"dmcc", "jafza", "difc", "adgm", "rakez", "ifza", "hamriyah"}
        )
    )


@lru_cache(maxsize=1)
def watchlists() -> dict:
    return json.loads(WATCHLISTS.read_text(encoding="utf-8"))


def _alnum(s: str | None) -> str:
    return re.sub(r"[^0-9a-z]", "", (s or "").lower())


def _name_hit(name: str, entry: dict) -> tuple[bool, float]:
    threshold = float(_policy().params["name_similarity_threshold"])
    listed = [entry["name"], *entry.get("aliases", [])]
    best = max(policy_score(name, n) for n in listed)
    alias = d044(name) in {d044(a) for a in entry.get("aliases", [])}
    return best >= threshold or alias, best


def ubo_persons(case: CaseSpec) -> list[Person]:
    """Natural persons behind a holding at or above the policy's UBO threshold."""
    threshold = int(_policy().params["ownership_threshold_pct"])
    seen: dict[str, Person] = {}
    for h in case.holders:
        if h.pct >= threshold and h.person is not None:
            seen.setdefault(h.person.pid, h.person)
    return list(seen.values())


def _persons_in_scope(case: CaseSpec, *, sanctions: bool = False) -> list[Person]:
    out = {p.pid: p for p in ubo_persons(case)}
    if sanctions:
        out.update({h.person.pid: h.person for h in case.holders if h.person is not None})
    for o in case.officers:
        if o.person is not None and o.role == "director":
            out.setdefault(o.person.pid, o.person)
    return list(out.values())


def _sanctions(case: CaseSpec, notes: list[str]) -> list[Fired]:
    c = case.company
    specs = _by_kind()
    rid_a, sev_a, pts_a = specs["sanctions_corroborated"]
    rid_b, sev_b, pts_b = specs["sanctions_fuzzy"]
    fired: list[Fired] = []
    for entry in watchlists()["sanctions"]:
        subjects: list[tuple[str, dict]] = []
        if entry["type"] == "entity":
            subjects.append((c.name, {"reg": c.tax_id, "place": c.jurisdiction, "dob": None}))
            for h in case.holders:
                if h.kind != "natural":
                    subjects.append(
                        (h.name, {"reg": h.reg_no, "place": code_of(h.domicile), "dob": None})
                    )
                    subjects += [
                        (n, {"reg": None, "place": code_of(h.domicile), "dob": None})
                        for n in h.chain
                    ]
            subjects += [
                (o.name, {"reg": None, "place": None, "dob": None})
                for o in case.officers
                if o.person is None
            ]
        else:
            for p in _persons_in_scope(case, sanctions=True):
                subjects.append((p.name, {"reg": None, "place": p.nationality, "dob": p.dob}))
        for name, ids in subjects:
            hit, score = _name_hit(name, entry)
            if not hit:
                continue
            reg_match = bool(entry.get("registration_number")) and (
                _alnum(entry["registration_number"]) == _alnum(ids["reg"])
            )
            place = entry.get("jurisdiction") or entry.get("nationality")
            dob_match = bool(ids["dob"]) and entry.get("dob") == ids["dob"].isoformat()
            if reg_match or (place == ids["place"] and dob_match):
                how = "registration number" if reg_match else "jurisdiction with date of birth"
                fired.append(
                    Fired(
                        rule=rid_a,
                        severity=sev_a,
                        points=pts_a,
                        reason=f"{entry['id']}: name match ({score:.2f}) corroborated by {how}",
                    )
                )
            else:
                fired.append(
                    Fired(
                        rule=rid_b,
                        severity=sev_b,
                        points=pts_b,
                        reason=f"{entry['id']}: name-only similarity {score:.2f}, no "
                        "corroborating identifier",
                    )
                )
    return fired


def _pep(case: CaseSpec, notes: list[str]) -> list[Fired]:
    specs = _by_kind()
    rid_cur, sev_cur, pts_cur = specs["pep_active"]
    rid_fmr, sev_fmr, pts_fmr = specs["pep_former"]
    lookback_years = int(_policy().params["pep_lookback_years"])
    fired: list[Fired] = []
    for entry in watchlists()["pep"]:
        for p in _persons_in_scope(case):
            hit, score = _name_hit(p.name, entry)
            if not hit:
                continue
            if not (
                entry.get("dob") == p.dob.isoformat() and entry.get("nationality") == p.nationality
            ):
                notes.append(
                    f"{entry['id']}: name-only similarity {score:.2f} with a conflicting "
                    "date of birth or nationality: screening note, not fired"
                )
                continue
            left = entry.get("left_office")
            if entry.get("in_office") or (
                left and date.fromisoformat(left) >= add_months(case.as_of, -12 * lookback_years)
            ):
                fired.append(
                    Fired(
                        rule=rid_cur,
                        severity=sev_cur,
                        points=pts_cur,
                        reason=f"{entry['id']}: PEP in office or <= {lookback_years}y ago",
                    )
                )
            else:
                fired.append(
                    Fired(
                        rule=rid_fmr,
                        severity=sev_fmr,
                        points=pts_fmr,
                        reason=f"{entry['id']}: former PEP, left > {lookback_years}y ago",
                    )
                )
    return fired


def conflict_items(case: CaseSpec) -> list[tuple[str, str, list[tuple[str, str]]]]:
    """One conflict per (subject, attribute) across the packet's documents:
    (subject, attribute, [(document kind, stated value)])."""
    mat = int(_policy().params["conflict_ownership_points"])
    docs = [d for d in case.documents if d.present]
    out: list[tuple[str, str, list[tuple[str, str]]]] = []
    listing = [d for d in docs if d.claims.holders is not None]
    for n in sorted({n for d in listing for n in d.claims.holders or {}}):
        vals = [
            (d.kind, str(d.claims.holders[n])) for d in listing if n in (d.claims.holders or {})
        ]
        if len(vals) < len(listing):
            stated = {d.kind for d in listing if n in (d.claims.holders or {})}
            vals += [(d.kind, "absent") for d in listing if d.kind not in stated]
            out.append((n, "owner_presence", vals))
        elif max(int(v) for _, v in vals) - min(int(v) for _, v in vals) > mat:
            out.append((n, "ownership_pct", vals))
    addrs = [(d.kind, d.claims.address) for d in docs if d.claims.address]
    if len({fold(a) for _, a in addrs}) > 1:
        out.append(("applicant", "address", addrs))
    regs = [(d.kind, d.claims.reg_no) for d in docs if d.claims.reg_no]
    if len({_alnum(r) for _, r in regs}) > 1:
        out.append(("applicant", "reg_no", regs))
    return out


def _conflicts(case: CaseSpec) -> list[str]:
    return [
        f"{subject}: {attr} differs ({', '.join(f'{k}={v}' for k, v in vals)})"
        for subject, attr, vals in conflict_items(case)
    ]


def _required_docs(case: CaseSpec) -> list[str]:
    p = _policy().params
    ubo_m = int(p["ubo_declaration_max_age_months"])
    poa_m = int(p["proof_of_address_max_age_months"])
    missing: list[str] = []
    inc, ubo, poa = (
        case.doc("incorporation"),
        case.doc("ubo_declaration"),
        case.doc("proof_of_address"),
    )
    if inc is None:
        missing.append("certificate of incorporation missing")
    elif (
        inc.stated_validity_days is not None
        and inc.issued is not None
        and ((case.as_of - inc.issued).days > inc.stated_validity_days)
    ):
        missing.append(
            f"certificate of incorporation expired (valid {inc.stated_validity_days} days)"
        )
    if ubo is None or not ubo.signed:
        missing.append("signed UBO declaration missing")
    elif ubo.issued is None or ubo.issued < add_months(case.as_of, -ubo_m):
        missing.append(f"UBO declaration older than {ubo_m} months")
    if poa is None:
        missing.append("proof of address missing")
    elif poa.issued is None or poa.issued < add_months(case.as_of, -poa_m):
        missing.append(f"proof of address older than {poa_m} months")
    return missing


def _industry(case: CaseSpec) -> tuple[list[str], list[str], list[str]]:
    text = " | ".join(fold(a) for a in case.company.activities)
    unlic = [k for k, ws in UNLICENSED_INDUSTRY.items() if any(w in text for w in ws)]
    proh = [
        k for k, ws in PROHIBITED_INDUSTRY.items() if k not in unlic and any(w in text for w in ws)
    ]
    risky = [k for k, ws in HIGH_RISK_INDUSTRY.items() if any(w in text for w in ws)]
    return proh, unlic, risky


def shell_indicators(case: CaseSpec) -> list[str]:
    c = case.company
    young_m = int(_policy().params["young_company_months"])
    ind = []
    if c.mass_registration_address:
        ind.append("i")
    if c.operating_address is None or fold(c.operating_address) == fold(c.registered_address):
        ind.append("ii")
    if c.incorporated > add_months(case.as_of, -young_m):
        ind.append("iii")
    if case.doc("bank_reference") is None:
        ind.append("iv")
    if not c.employees_or_website:
        ind.append("v")
    return ind


def label(case: CaseSpec) -> Label:
    c = case.company
    pol = _policy()
    p = pol.params
    specs = _by_kind()
    fired: list[Fired] = []
    notes: list[str] = []
    unknown: list[str] = []

    def add_kind(kind: str, reason: str, points: int | None = None) -> None:
        rid, sev, pts = specs[kind]
        fired.append(
            Fired(
                rule=rid,
                severity=sev,
                points=pts if points is None else points,
                reason=reason,
            )
        )

    fired += _sanctions(case, notes)
    fired += _pep(case, notes)

    ubo_places = [
        (person.name, code_of(person.domicile), region_of(person.domicile))
        for person in ubo_persons(case)
    ]
    if prohibited_place(c.jurisdiction, (c.region or "").lower()):
        add_kind(
            "jurisdiction_prohibited",
            f"incorporated in a prohibited jurisdiction ({c.jurisdiction})",
        )
    elif any(prohibited_place(code, reg) for _, code, reg in ubo_places):
        add_kind("jurisdiction_prohibited", "a UBO resides in a prohibited jurisdiction")

    own_t = int(p["ownership_threshold_pct"])
    big = [h for h in case.holders if h.pct >= own_t]
    risky_holder = [h for h in big if high_risk_place(code_of(h.domicile), region_of(h.domicile))]
    in_high_risk = high_risk_place(c.jurisdiction, (c.region or "").lower())
    if in_high_risk or risky_holder:
        add_kind(
            "jurisdiction_monitored",
            "applicant incorporated in a monitored jurisdiction"
            if in_high_risk
            else f"{risky_holder[0].name} (>= {own_t}%) domiciled in a monitored jurisdiction",
        )

    unresolved = [h for h in case.holders if h.person is None]
    if any(h.pct >= own_t for h in unresolved) or sum(h.pct for h in unresolved) >= own_t:
        add_kind("ubo_unresolved", f"a >= {own_t}% holding has no natural person behind it")
    hr_unres = int(p["high_risk_unresolved_pct"])
    if sum(h.pct for h in unresolved) >= hr_unres and (
        in_high_risk
        or any(high_risk_place(code_of(h.domicile), region_of(h.domicile)) for h in case.holders)
    ):
        add_kind(
            "ubo_high_risk_unresolved",
            f">= {hr_unres}% of ownership unresolved in a monitored jurisdiction",
        )

    conflicts = _conflicts(case)
    if conflicts:
        add_kind("conflicting_filings", "; ".join(conflicts))
    if len(conflicts) > 1:
        extra_pts = int(p["conflict_extra_points"])
        add_kind(
            "conflict_multiplier",
            f"{len(conflicts) - 1} conflict(s) beyond the first",
            extra_pts * (len(conflicts) - 1),
        )

    nominee_roles = [o for o in case.officers if o.role in ("nominee_director", "registered_agent")]
    nominee_holders = [h for h in case.holders if h.nominee]
    if nominee_roles or nominee_holders:
        add_kind(
            "nominee_control",
            "nominee director or shareholder, or corporate service provider acting as officer",
        )
        natural_control = any(o.person is not None for o in case.officers) or any(
            h.person is not None and not h.nominee for h in case.holders
        )
        if not natural_control:
            add_kind("nominee_sole_controller", "the nominee is the sole controller on record")

    max_layers = int(p["max_holding_layers"])
    layered = any((1 + len(h.chain)) >= max_layers for h in case.holders if h.kind != "natural")
    act_text = " | ".join(fold(a) for a in c.activities)
    dormant = c.status.upper() in ("DORMANT", "INACTIVE", "INACTIVA") or any(
        w in act_text for w in ("dormant", "inactiv", "sin actividad")
    )
    if layered or dormant:
        add_kind("dormant_or_layered", "multi-layer holding structure or dormant status")

    ind = shell_indicators(case)
    if len(ind) >= int(p["shell_review_min"]):
        add_kind("shell_indicators", f"{len(ind)} shell indicators ({', '.join(ind)})")
    if len(ind) >= int(p["shell_reject_min"]) and in_high_risk:
        add_kind("shell_severe", f"{len(ind)} shell indicators in a monitored jurisdiction")

    missing = _required_docs(case)
    if missing:
        add_kind("required_documents", "; ".join(missing))

    prohibited, unlicensed, risky = _industry(case)
    if prohibited:
        add_kind("industry_prohibited", f"prohibited industry: {', '.join(prohibited)}")
    else:
        if unlicensed:
            unknown.append(specs["industry_prohibited"][0])
            notes.append(f"{', '.join(unlicensed)}: licence unestablished, an officer decides")
        if risky:
            add_kind("industry_elevated", f"elevated-risk industry: {', '.join(risky)}")

    score = min(sum(f.points for f in fired), int(p["score_cap"]))
    if any(f.severity == "REJECT" for f in fired):
        status: Status = "REJECTED"
    elif any(f.severity == "REVIEW" for f in fired) or unknown:
        status = "MANUAL_REVIEW"
    elif score <= int(p["auto_approve_max_score"]):
        status = "APPROVED"
    else:
        status = "MANUAL_REVIEW"
    return Label(
        status=status,
        score=score,
        fired=fired,
        conflicts=len(conflicts),
        unknown=unknown,
        notes=notes,
    )


def profile_key(case: CaseSpec, lab: Label) -> str:
    """The policy's precedent key: incorporation jurisdiction plus the set of fired rules."""
    return f"{case.company.jurisdiction}|{','.join(lab.rules)}"


__all__ = ["Fired", "Holder", "Label", "label", "profile_key", "shell_indicators"]
