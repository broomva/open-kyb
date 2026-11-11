"""Injections: named, parameterised edits that provoke one rule in a draft case.

Every injection edits the `Draft` (facts), never the label: the label engine reads the result.
`INJECTIONS` maps each name in `case.INJECTION_RULES` to its function. An injection returns its
parameters (recorded in the manifest). Conflict injections run last, after `Draft.sync()` has
rebuilt document claims from the final holders.
"""

from __future__ import annotations

import copy
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta

from kyb.synth.case import (
    CaseSpec,
    Company,
    Doc,
    DocClaims,
    Holder,
    Injection,
    Officer,
    Person,
)
from kyb.synth.label import watchlists
from kyb.synth.pseudo import Pseudonymiser
from kyb.synth.sources import RawData

AS_OF = date(2026, 11, 1)
HIGH_RISK_ISO = ("MM", "VU", "VG", "BZ", "WS", "SC")
CONFLICT_LAST = ("conflict_pct", "conflict_presence", "conflict_address", "conflict_reg_no")

STREETS = (
    "Carrera 15 No. 93-40",
    "Calle 72 No. 10-34",
    "Avenida El Dorado 68C-61",
    "Calle 10 No. 4-21",
    "Carrera 43A No. 1 Sur-50",
    "Av. Insurgentes Sur 1602",
    "Calle Durango 215",
    "Paseo de la Reforma 222",
    "Av. Vasconcelos 345",
    "Calzada del Valle 400",
    "Av. Chapultepec 480",
    "Boulevard Kukulcán km 9",
)
CITIES = {
    "CO": ("Bogotá", "Medellín", "Cali", "Barranquilla", "Bucaramanga", "Cartagena"),
    "MX": ("CDMX", "Monterrey", "Guadalajara", "Puebla", "Querétaro", "Mérida"),
    "CL": ("Santiago", "Valparaíso", "Concepción"),
    "VG": ("Sovereign Town, Tortola",),
    "BZ": ("Belize City",),
    "VU": ("Port Vila",),
    "WS": ("Apia",),
    "SC": ("Victoria, Mahé",),
    "MM": ("Yangon",),
    "US": ("Miami, FL",),
    "ES": ("Madrid", "Barcelona", "Bilbao"),
}
ACTIVITIES_ES = (
    "Comercio al por mayor de productos alimenticios",
    "Servicios de consultoría empresarial",
    "Transporte de carga por carretera",
    "Desarrollo de sistemas informáticos a la medida",
    "Servicios de ingeniería y asesoría técnica",
    "Elaboración de productos de panadería",
    "Comercio al por menor de artículos de ferretería",
    "Actividades de publicidad y mercadeo",
    "Confección de prendas de vestir",
    "Servicios de mantenimiento de maquinaria comercial",
    "Comercio al por mayor de materiales de construcción",
    "Servicios de capacitación empresarial",
)
RISKY_ACTIVITY = (
    "Servicios de remesas y transferencia de dinero",
    "Comercio de metales preciosos y joyería",
    "Financiamiento comercial de exportaciones (trade finance)",
    "Fundación sin fines de lucro para programas sociales",
    "Casa de cambio de divisas",
    "Intercambio de activos virtuales y criptomonedas",
    "Fletamento marítimo de buques de carga",
)
PROHIBITED_ACTIVITY = (
    "Fabricación y comercio de armas de fuego y municiones",
    "Producción de contenido para adultos",
    "Operación de un banco pantalla (shell bank)",
    "Operación de casino y apuestas en línea",
)
UNLICENSED_ACTIVITY = (
    "Operación de banca no autorizada y captación ilegal de fondos",
    "Servicios de banca sin licencia (unlicensed bank)",
)
HOLD_WORDS = ("Inversiones", "Grupo", "Holding", "Capital", "Comercializadora", "Desarrollos")
SUFFIX = {
    "CO": "S.A.S.",
    "MX": "S.L.",
    "CL": "SpA",
    "VG": "Ltd",
    "BZ": "Ltd",
    "VU": "Ltd",
    "WS": "Ltd",
    "SC": "Ltd",
    "MM": "Ltd",
    "US": "LLC",
    "ES": "S.L.",
}


@dataclass
class Ctx:
    rng: random.Random
    pseud: Pseudonymiser
    raw: RawData
    persons: list = field(default_factory=list)
    n_person: int = 0
    used: set = field(default_factory=set)

    def person(self, nationality: str, domicile: str | None = None) -> Person:
        """A pseudonymised natural person, seeded from a real registry row when one is left. Never
        the same person twice in one case (a repeated draw falls back to a synthetic key)."""
        for _ in range(50):
            if self.persons:
                raw = self.persons.pop()
                key, raw_name = f"{raw.source}:{raw.key}", raw.name
            else:
                self.n_person += 1
                key, raw_name = f"syn:{self.rng.random():.12f}:{self.n_person}", ""
            p = self.pseud.person(key, raw_name)
            if p["id"] not in self.used:
                break
        self.used.add(p["id"])
        return Person(
            pid=f"p{p['id']}",
            name=p["name"],
            nationality=nationality,
            dob=date.fromisoformat(p["dob"]),
            domicile=domicile or nationality,
        )

    def address(self, country: str) -> str:
        return f"{self.rng.choice(STREETS)}, {self.rng.choice(CITIES.get(country, ('Ciudad',)))}"

    def holding_name(self, country: str) -> str:
        word = self.rng.choice(HOLD_WORDS)
        stem = self.rng.choice(
            (
                "Andes",
                "Cordillera",
                "Altiplano",
                "Bahía",
                "Sierra",
                "Ceiba",
                "Altamar",
                "Pacífico",
                "Lumbre",
                "Tairona",
                "Aurora",
                "Delmar",
            )
        )
        return f"{word} {stem} {SUFFIX.get(country, 'S.A.')}"

    def offshore_name(self, iso: str) -> str:
        pool = [o.name for o in self.raw.offshore if o.jurisdiction == iso]
        if pool:
            return self.rng.choice(pool)
        return (
            f"{self.rng.choice(HOLD_WORDS)} {self.rng.choice(('Azure', 'Harbour', 'Pinnacle'))}"
            f" {SUFFIX.get(iso, 'Ltd')}"
        )


@dataclass
class Draft:
    case_id: str
    company: Company
    holders: list[Holder]
    officers: list[Officer]
    docs: dict[str, Doc]
    injections: list[Injection] = field(default_factory=list)
    flags: dict[str, str] = field(default_factory=dict)

    def spec(self) -> CaseSpec:
        return CaseSpec(
            case_id=self.case_id,
            as_of=AS_OF,
            company=self.company,
            holders=list(self.holders),
            officers=list(self.officers),
            documents=list(self.docs.values()),
            injections=list(self.injections),
            source_flags=dict(self.flags),
        )

    def copy(self) -> Draft:
        return copy.deepcopy(self)

    def sync(self) -> None:
        """Rebuild document claims from the final holders, address and registration number."""
        stated = {h.name: h.pct for h in self.holders}
        for kind, doc in list(self.docs.items()):
            claims = DocClaims()
            if kind in ("ubo_declaration", "share_register"):
                claims = DocClaims(
                    holders=dict(stated),
                    reg_no=self.company.tax_id if kind == "ubo_declaration" else None,
                )
            elif kind == "incorporation":
                claims = DocClaims(
                    address=self.company.registered_address, reg_no=self.company.tax_id
                )
            elif kind == "proof_of_address":
                claims = DocClaims(address=self.company.registered_address)
            self.docs[kind] = doc.model_copy(update={"claims": claims})

    def set_doc(self, kind: str, **kw) -> None:
        self.docs[kind] = self.docs[kind].model_copy(update=kw)


def mark(d: Draft, name: str, **params) -> dict:
    d.injections.append(Injection(name=name, params=params))
    return params


def _natural(d: Draft) -> list[int]:
    return [i for i, h in enumerate(d.holders) if h.kind == "natural" and h.person is not None]


def _replace_holder_person(d: Draft, ctx: Ctx, person: Person, idx: int | None = None) -> int:
    """Make `person` the natural holder at `idx` (default: the largest natural holder)."""
    nat = _natural(d)
    i = idx if idx is not None else max(nat, key=lambda k: d.holders[k].pct)
    h = d.holders[i]
    d.holders[i] = h.model_copy(
        update={"name": person.name, "person": person, "domicile": person.domicile}
    )
    d.officers = [
        o.model_copy(update={"name": person.name, "person": person})
        if o.person is not None and o.person.pid == h.person.pid
        else o
        for o in d.officers
    ]
    return i


def _split_for_legal(d: Draft, pct: int) -> int:
    """Take `pct` points from the biggest holder to make room for a new holder; returns pct used."""
    i = max(range(len(d.holders)), key=lambda k: d.holders[k].pct)
    h = d.holders[i]
    take = min(pct, h.pct - 1)
    d.holders[i] = h.model_copy(update={"pct": h.pct - take})
    return take


def _new_legal(
    d: Draft,
    ctx: Ctx,
    name: str,
    pct: int,
    domicile: str,
    *,
    resolved: bool,
    kind: str = "legal",
    reg_no: str | None = None,
    chain: int = 0,
) -> Holder:
    take = _split_for_legal(d, pct)
    person = ctx.person(ctx.rng.choice(("CO", "MX", "ES", "US"))) if resolved else None
    h = Holder(
        hid=f"h{len(d.holders) + 1}",
        kind=kind,
        name=name,
        pct=take,
        domicile=domicile,
        reg_no=reg_no,
        person=person,
        chain=[ctx.holding_name("CL") for _ in range(chain)],
    )
    d.holders.append(h)
    return h


# --- score-only and approval-safe injections -----------------------------------------------------


def high_risk_jurisdiction(d: Draft, ctx: Ctx) -> dict:
    iso = ctx.rng.choice(HIGH_RISK_ISO)
    if ctx.rng.random() < 0.5:
        d.company = d.company.model_copy(
            update={
                "jurisdiction": iso,
                "source": "icij-or-synthetic",
                "name": ctx.offshore_name(iso),
            }
        )
        return mark(d, "high_risk_jurisdiction", where="company", jurisdiction=iso)
    h = _new_legal(
        d,
        ctx,
        ctx.offshore_name(iso),
        ctx.rng.choice((22, 28, 35)),
        iso,
        resolved=True,
        chain=0,
    )
    return mark(d, "high_risk_jurisdiction", where="holder", jurisdiction=iso, holder=h.hid)


def nominee(d: Draft, ctx: Ctx) -> dict:
    agent = f"{ctx.rng.choice(('Apex', 'Kestrel', 'Harbour'))} Corporate Services Ltd"
    d.officers.append(Officer(name=agent, role="nominee_director"))
    if ctx.rng.random() < 0.5:
        d.officers.append(Officer(name=agent, role="registered_agent"))
    return mark(d, "nominee", natural_controller=True)


def nominee_sole_controller(d: Draft, ctx: Ctx) -> dict:
    """Only a nominee runs the company and every holder is a legal entity with no UBO named."""
    agent = f"{ctx.rng.choice(('Apex', 'Kestrel'))} Corporate Services Ltd"
    d.officers = [Officer(name=agent, role="nominee_director")]
    pct = 62
    d.holders = [
        Holder(hid="h1", kind="legal", name=ctx.holding_name("CL"), pct=pct, domicile="CO"),
        Holder(hid="h2", kind="legal", name=ctx.holding_name("CO"), pct=100 - pct, domicile="CO"),
    ]
    return mark(d, "nominee_sole_controller", holders=2)


def dormant_or_layered(d: Draft, ctx: Ctx) -> dict:
    """Either a 2-layer holding chain or a declared dormant/inactive corporate status."""
    if ctx.rng.random() < 0.5:
        h = _new_legal(
            d,
            ctx,
            ctx.holding_name("CL"),
            ctx.rng.choice((24, 30, 35)),
            "CL",
            resolved=True,
            chain=1,
        )
        return mark(d, "dormant_or_layered", how="layered", holder=h.hid)
    act = "Sociedad en inactividad temporal (dormant)"
    d.company = d.company.model_copy(
        update={"status": "DORMANT", "activities": [*d.company.activities[:1], act]}
    )
    return mark(d, "dormant_or_layered", how="dormant")


def industry_high_risk(d: Draft, ctx: Ctx) -> dict:
    act = ctx.rng.choice(RISKY_ACTIVITY)
    d.company = d.company.model_copy(update={"activities": [*d.company.activities[:1], act]})
    return mark(d, "industry_high_risk", activity=act)


def industry_prohibited(d: Draft, ctx: Ctx) -> dict:
    act = ctx.rng.choice(PROHIBITED_ACTIVITY)
    d.company = d.company.model_copy(update={"activities": [act]})
    return mark(d, "industry_prohibited", activity=act)


def industry_unlicensed(d: Draft, ctx: Ctx) -> dict:
    act = ctx.rng.choice(UNLICENSED_ACTIVITY)
    d.company = d.company.model_copy(update={"activities": [act]})
    return mark(d, "industry_unlicensed", activity=act)


def stale_document(d: Draft, ctx: Ctx) -> dict:
    d.set_doc("share_register", issued=AS_OF - timedelta(days=ctx.rng.randint(760, 1100)))
    return mark(d, "stale_document", document="share_register")


def _pep_person(entry: dict) -> Person:
    return Person(
        pid=entry["id"].lower().replace("-", ""),
        name=entry["name"],
        nationality=entry["nationality"],
        dob=date.fromisoformat(entry["dob"]),
        domicile=entry["nationality"],
    )


def pep(d: Draft, ctx: Ctx, *, former: bool = False) -> dict:
    """A listed PEP as a UBO (active REVIEW) or, with `former`, left office before the window."""
    entries = watchlists()["pep"]
    pool = [
        e
        for e in entries
        if (e["left_office"] is not None and e["left_office"] < "2023-11-01") == former
    ]
    entry = ctx.rng.choice(pool)
    _replace_holder_person(d, ctx, _pep_person(entry))
    return mark(d, "pep_former" if former else "pep", entry=entry["id"])


def pep_former(d: Draft, ctx: Ctx) -> dict:
    return pep(d, ctx, former=True)


def pep_decoy(d: Draft, ctx: Ctx) -> dict:
    """A name-only PEP similarity with a conflicting date of birth and nationality: a
    screening note, not a trigger."""
    entry = ctx.rng.choice(watchlists()["pep"])
    person = Person(
        pid="decoy" + entry["id"][-3:],
        name=entry["aliases"][0] if entry["aliases"] else entry["name"].replace("é", "e"),
        nationality="BO",
        dob=date(1990, 7, 7),
        domicile="BO",
    )
    _replace_holder_person(d, ctx, person)
    return mark(d, "pep_decoy", entry=entry["id"])


# --- review injections ---------------------------------------------------------------------------


def unresolved_ownership(d: Draft, ctx: Ctx) -> dict:
    kind = ctx.rng.choice(("legal", "foundation", "trust"))
    country = ctx.rng.choice(("CO", "MX", "US", "ES"))
    h = _new_legal(
        d,
        ctx,
        ctx.holding_name(country),
        ctx.rng.choice((24, 30, 38)),
        country,
        resolved=False,
        kind=kind,
        chain=0,
    )
    return mark(d, "unresolved_ownership", holder=h.hid, kind=kind)


def unresolved_ownership_high_risk(d: Draft, ctx: Ctx) -> dict:
    iso = ctx.rng.choice(HIGH_RISK_ISO)
    d.holders = [
        Holder(
            hid="h1",
            kind="legal",
            name=ctx.offshore_name(iso),
            pct=ctx.rng.choice((44, 52, 68)),
            domicile=iso,
            chain=[],
        )
    ]
    rest = 100 - d.holders[0].pct
    d.holders.append(
        Holder(
            hid="h2",
            kind="natural",
            name=(p := ctx.person("CO")).name,
            pct=rest,
            domicile="CO",
            person=p,
        )
    )
    return mark(d, "unresolved_ownership_high_risk", jurisdiction=iso)


def shell_indicators(d: Draft, ctx: Ctx, *, high_risk: bool = False) -> dict:
    d.company = d.company.model_copy(
        update={
            "mass_registration_address": True,
            "operating_address": None,
            "employees_or_website": ctx.rng.random() < 0.5,
            **(
                {
                    "jurisdiction": (iso := ctx.rng.choice(HIGH_RISK_ISO)),
                    "name": ctx.offshore_name(iso),
                }
                if high_risk
                else {}
            ),
        }
    )
    if high_risk:
        d.company = d.company.model_copy(update={"employees_or_website": False})
        return mark(d, "shell_indicators_high_risk", jurisdiction=d.company.jurisdiction)
    return mark(d, "shell_indicators", indicators=["i", "ii"])


def shell_indicators_high_risk(d: Draft, ctx: Ctx) -> dict:
    return shell_indicators(d, ctx, high_risk=True)


def missing_document(d: Draft, ctx: Ctx) -> dict:
    kind = ctx.rng.choice(("proof_of_address", "ubo_declaration", "incorporation"))
    if ctx.rng.random() < 0.5:
        d.docs[kind] = d.docs[kind].model_copy(update={"present": False, "issued": None})
        return mark(d, "missing_document", document=kind, how="absent")
    if kind == "ubo_declaration":
        d.set_doc(kind, issued=AS_OF - timedelta(days=ctx.rng.randint(240, 500)))
    elif kind == "proof_of_address":
        d.set_doc(kind, issued=AS_OF - timedelta(days=ctx.rng.randint(120, 300)))
    else:
        d.docs[kind] = d.docs[kind].model_copy(update={"present": False, "issued": None})
    return mark(d, "missing_document", document=kind, how="stale")


def expired_document(d: Draft, ctx: Ctx) -> dict:
    d.set_doc(
        "incorporation",
        issued=AS_OF - timedelta(days=ctx.rng.randint(45, 90)),
        stated_validity_days=30,
    )
    return mark(d, "expired_document", document="incorporation", valid_days=30)


def sanctions_near_miss(d: Draft, ctx: Ctx) -> dict:
    """Name-only similarity (transliteration and spacing variants) with a conflicting identifier."""
    wl = watchlists()["sanctions"]
    entry = ctx.rng.choice(wl)
    sid = entry["id"]
    name = entry["aliases"][0] if entry.get("aliases") else entry["name"]
    if entry.get("type") == "individual":
        p = Person(
            pid="nm" + sid[-4:], name=name, nationality="LB", dob=date(1981, 4, 9), domicile="LB"
        )
        _replace_holder_person(d, ctx, p)
    else:
        _new_legal(
            d,
            ctx,
            name,
            30,
            ctx.rng.choice(("CY", "TR", "BS")),
            resolved=True,
            reg_no="X-" + str(ctx.rng.randint(100000, 999999)),
        )
    return mark(d, "sanctions_near_miss", entry=sid, variant=name)


def sanctions_corroborated(d: Draft, ctx: Ctx) -> dict:
    wl = {e["id"]: e for e in watchlists()["sanctions"]}
    choice = ctx.rng.choice(("applicant", "holder", "person", "person"))
    if choice == "applicant":
        e = wl["SAN-0003"]
        d.company = d.company.model_copy(
            update={
                "name": e["name"],
                "tax_id": e["registration_number"],
                "jurisdiction": e["jurisdiction"],
            }
        )
        return mark(d, "sanctions_corroborated", entry=e["id"], where="applicant")
    if choice == "holder":
        e = wl["SAN-0001"]
        _new_legal(
            d,
            ctx,
            ctx.rng.choice([e["name"], *e["aliases"][:1]]),
            30,
            "AE:DMCC free zone",
            resolved=False,
            reg_no=e["registration_number"],
        )
        return mark(d, "sanctions_corroborated", entry=e["id"], where="holder")
    e = wl[ctx.rng.choice(("SAN-0002", "SAN-0004"))]
    p = Person(
        pid=e["id"].lower().replace("-", ""),
        name=ctx.rng.choice([e["name"], *e["aliases"]]),
        nationality=e["nationality"],
        dob=date.fromisoformat(e["dob"]),
        domicile="TR",
    )
    _replace_holder_person(d, ctx, p)
    return mark(d, "sanctions_corroborated", entry=e["id"], where="ubo")


def prohibited_jurisdiction(d: Draft, ctx: Ctx) -> dict:
    if ctx.rng.random() < 0.5:
        iso = ctx.rng.choice(("KP", "IR", "SY", "AF", "BY"))
        d.company = d.company.model_copy(update={"jurisdiction": iso})
        return mark(d, "prohibited_jurisdiction", where="company", jurisdiction=iso)
    where = ctx.rng.choice(("IR", "SY", "BY", "UA:Crimea"))
    i = max(_natural(d), key=lambda k: d.holders[k].pct)
    p = d.holders[i].person.model_copy(update={"domicile": where})
    d.holders[i] = d.holders[i].model_copy(update={"person": p, "domicile": where})
    return mark(d, "prohibited_jurisdiction", where="ubo", jurisdiction=where)


# --- conflicts (run last) ------------------------------------------------------------------------


def conflict_pct(d: Draft, ctx: Ctx) -> dict:
    h = max(d.holders, key=lambda x: x.pct)
    reg = dict(d.docs["share_register"].claims.holders or {})
    delta = ctx.rng.choice((8, 12, 18))
    reg[h.name] = h.pct + delta if h.pct + delta <= 100 else h.pct - delta
    d.set_doc(
        "share_register", claims=d.docs["share_register"].claims.model_copy(update={"holders": reg})
    )
    return mark(d, "conflict_pct", holder=h.hid, stated=reg[h.name], actual=h.pct)


def conflict_presence(d: Draft, ctx: Ctx) -> dict:
    h = d.holders[-1]
    founders = {
        k: v for k, v in (d.docs["ubo_declaration"].claims.holders or {}).items() if k != h.name
    }
    d.set_doc(
        "incorporation",
        claims=d.docs["incorporation"].claims.model_copy(update={"holders": founders}),
    )
    return mark(d, "conflict_presence", holder=h.hid)


def conflict_address(d: Draft, ctx: Ctx) -> dict:
    other = ctx.address(d.company.jurisdiction if d.company.jurisdiction in CITIES else "CO")
    d.set_doc("proof_of_address", claims=DocClaims(address=other + " (otra sede)"))
    return mark(d, "conflict_address", document="proof_of_address")


def conflict_reg_no(d: Draft, ctx: Ctx) -> dict:
    t = list(d.company.tax_id)
    i = max(k for k, ch in enumerate(t) if ch.isalnum())
    t[i] = str((int(t[i]) + 1) % 10) if t[i].isdigit() else ("A" if t[i] != "A" else "B")
    d.set_doc(
        "ubo_declaration",
        claims=d.docs["ubo_declaration"].claims.model_copy(update={"reg_no": "".join(t)}),
    )
    return mark(d, "conflict_reg_no", document="ubo_declaration")


INJECTIONS: dict[str, Callable[[Draft, Ctx], dict]] = {
    "high_risk_jurisdiction": high_risk_jurisdiction,
    "nominee": nominee,
    "nominee_sole_controller": nominee_sole_controller,
    "dormant_or_layered": dormant_or_layered,
    "industry_high_risk": industry_high_risk,
    "industry_prohibited": industry_prohibited,
    "industry_unlicensed": industry_unlicensed,
    "pep": pep,
    "pep_former": pep_former,
    "pep_decoy": pep_decoy,
    "unresolved_ownership": unresolved_ownership,
    "unresolved_ownership_high_risk": unresolved_ownership_high_risk,
    "shell_indicators": shell_indicators,
    "shell_indicators_high_risk": shell_indicators_high_risk,
    "missing_document": missing_document,
    "expired_document": expired_document,
    "sanctions_near_miss": sanctions_near_miss,
    "sanctions_corroborated": sanctions_corroborated,
    "prohibited_jurisdiction": prohibited_jurisdiction,
    "conflict_pct": conflict_pct,
    "conflict_presence": conflict_presence,
    "conflict_address": conflict_address,
    "conflict_reg_no": conflict_reg_no,
}
