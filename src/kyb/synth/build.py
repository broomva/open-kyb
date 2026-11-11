"""Build N synthetic KYB cases, deterministically, from the raw open-data snapshots.

    uv run python -m kyb.synth.build --n 80 --seed 11        # make synth-build N=80 SEED=11

Company shells come from RUES and SAT rows (and ICIJ names for offshore applicants), ownership
chains from GLEIF RR shapes plus pseudonymised natural persons. A plan assigns each case a target
class (about 35% APPROVED, 35% MANUAL_REVIEW, 30% REJECTED) and a list of injections; the label
engine, not the plan, decides the final label, and a case whose label misses its target or whose
injected rules did not fire is rebuilt from the next attempt's random stream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import string
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from kyb.synth.case import INJECTION_RULES, CaseSpec, Company, Doc, Holder, Officer
from kyb.synth.inject import (
    ACTIVITIES_ES,
    AS_OF,
    CONFLICT_LAST,
    INJECTIONS,
    Ctx,
    Draft,
)
from kyb.synth.label import Label, label
from kyb.synth.pseudo import Pseudonymiser
from kyb.synth.sources import RawData, Shell, cache_root, load_raw

ROOT = Path(__file__).resolve().parents[3]
INDEX_PATH = ROOT / "datasets" / "synth" / "index.json"
MIX = {"APPROVED": 0.35, "MANUAL_REVIEW": 0.35, "REJECTED": 0.30}
MAX_ATTEMPTS = 40

CIIU_ES = {
    "6810": "Actividades inmobiliarias realizadas con bienes propios o arrendados",
    "6820": "Actividades inmobiliarias realizadas a cambio de una retribución o por contrato",
    "7020": "Actividades de consultoría de gestión",
    "7110": "Actividades de arquitectura e ingeniería",
    "4752": "Comercio al por menor de artículos de ferretería",
    "8299": "Otras actividades de servicio de apoyo a las empresas",
    "4111": "Construcción de edificios residenciales",
    "4659": "Comercio al por mayor de otros tipos de maquinaria",
    "7310": "Publicidad",
    "4290": "Construcción de otras obras de ingeniería civil",
    "1410": "Confección de prendas de vestir",
    "4923": "Transporte de carga por carretera",
    "6202": "Actividades de consultoría informática",
    "4631": "Comercio al por mayor de productos alimenticios",
    "6910": "Actividades jurídicas",
    "5611": "Expendio a la mesa de comidas preparadas",
}

# Plans: (injections). Cycled in order so a build of N >= ~60 exercises every rule at least once.
PLANS: dict[str, list[list[str]]] = {
    "APPROVED": [
        [],
        [],
        [],
        [],
        [],
        [],
        ["high_risk_jurisdiction"],
        ["nominee"],
        ["industry_high_risk"],
        ["pep_former"],
        ["industry_high_risk", "nominee"],
        ["industry_high_risk", "pep_former"],
        ["pep_decoy"],
    ],
    "MANUAL_REVIEW": [
        ["sanctions_near_miss"],
        ["unresolved_ownership"],
        ["conflict_pct"],
        ["conflict_presence"],
        ["conflict_address"],
        ["conflict_reg_no"],
        ["conflict_pct", "conflict_address"],
        ["pep"],
        ["nominee_sole_controller"],
        ["dormant_or_layered"],
        ["shell_indicators"],
        ["missing_document"],
        ["expired_document"],
        ["industry_high_risk", "high_risk_jurisdiction"],
        ["nominee", "pep_former"],
        ["industry_unlicensed"],
        ["pep", "sanctions_near_miss"],
        ["unresolved_ownership", "conflict_pct"],
    ],
    "REJECTED": [
        ["sanctions_corroborated"],
        ["prohibited_jurisdiction"],
        ["unresolved_ownership_high_risk"],
        ["shell_indicators_high_risk"],
        ["industry_prohibited"],
        ["sanctions_corroborated", "pep"],
        ["prohibited_jurisdiction", "nominee"],
        ["sanctions_corroborated"],
    ],
}


class BuildReport(BaseModel):
    model_config = ConfigDict(frozen=True)
    n: int
    seed: int
    labels: dict[str, int]
    rules: dict[str, int]  # rule id → number of cases it fired in
    injections: dict[str, int]
    sources_used: dict[str, int]
    retries: int
    source_status: dict[str, str] = {}


# --- shell → company -----------------------------------------------------------------------------


def fake_nit(rng: random.Random) -> str:
    body = f"{rng.choice((900, 901, 830, 860, 800))}{rng.randint(0, 999999):06d}"
    weights = (3, 7, 13, 17, 19, 23, 29, 37, 41, 43, 47, 53, 59, 67, 71)
    total = sum(int(c) * w for c, w in zip(reversed(body), weights, strict=False))
    r = total % 11
    return f"{body}-{r if r < 2 else 11 - r}"


def fake_rfc(rng: random.Random, name: str, inc: date) -> str:
    letters = "".join(c for c in name.upper() if c in string.ascii_uppercase)[:3].ljust(3, "X")
    home = "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(3))
    return f"{letters}{inc:%y%m%d}{home}"


def synthetic_shell(rng: random.Random, i: int) -> Shell:
    """Used only when a source is missing: a clearly synthetic registry row."""
    country = rng.choice(("CO", "MX"))
    return Shell(
        source="synthetic",
        name=f"Sociedad Sintética {i:03d} {'S.A.S.' if country == 'CO' else 'S.L.'}",
        jurisdiction=country,
        incorporated=date(2005, 1, 1) + timedelta(days=rng.randint(0, 6000)),
        status="ACTIVA",
    )


def company_from(shell: Shell, ctx: Ctx) -> Company:
    rng = ctx.rng
    cc = shell.jurisdiction
    ciiu = shell.ciiu if shell.ciiu and shell.ciiu != "9999" else None
    activity = CIIU_ES.get(ciiu or "") or rng.choice(ACTIVITIES_ES)
    inc = min(shell.incorporated, AS_OF - timedelta(days=400))
    tax = fake_nit(rng) if cc == "CO" else fake_rfc(rng, shell.name, inc)
    city = (shell.city or "").title() or None
    way = rng.choice(("Carrera", "Calle", "Avenida"))
    reg = f"{way} {rng.randint(1, 99)} No. {rng.randint(1, 99)}-{rng.randint(1, 99)}"
    registered = f"{reg}, {city}" if city and cc == "CO" else ctx.address(cc)
    return Company(
        name=shell.name,
        tax_id=tax,
        jurisdiction=cc,
        incorporated=inc,
        activities=[activity],
        ciiu=ciiu,
        status=shell.status,
        registered_address=registered,
        operating_address=ctx.address(cc),
        employees_or_website=True,
        source=shell.source,
    )


def base_holders(ctx: Ctx, nat: str) -> tuple[list[Holder], list[Officer]]:
    rng = ctx.rng
    pcts = rng.choice(
        [
            (100,),
            (70, 30),
            (65, 35),
            (75, 25),
            (45, 35, 20),
            (40, 35, 25),
            (62, 23, 15),
            (34, 33, 33),
        ]
    )
    holders: list[Holder] = []
    for i, pct in enumerate(pcts):
        person = ctx.person(nat)
        wrap = i > 0 and rng.random() < 0.35
        if wrap:
            holders.append(
                Holder(
                    hid=f"h{i + 1}",
                    kind="legal",
                    name=ctx.holding_name(nat),
                    pct=pct,
                    domicile=nat,
                    person=person,
                    chain=[],
                )
            )
        else:
            holders.append(
                Holder(
                    hid=f"h{i + 1}",
                    kind="natural",
                    name=person.name,
                    pct=pct,
                    domicile=nat,
                    person=person,
                )
            )
    first = next(h for h in holders if h.kind == "natural")
    officers = [Officer(name=first.person.name, role="legal_rep", person=first.person)]
    if len(holders) > 1 and rng.random() < 0.5:
        other = next((h for h in holders if h.kind == "natural" and h is not first), None)
        if other:
            officers.append(Officer(name=other.person.name, role="director", person=other.person))
    return holders, officers


def base_docs(ctx: Ctx, cc: str) -> dict[str, Doc]:
    rng = ctx.rng
    ago = lambda lo, hi: AS_OF - timedelta(days=rng.randint(lo, hi))  # noqa: E731
    docs = {
        "incorporation": Doc(
            kind="incorporation", issued=ago(3, 25), stated_validity_days=30 if cc == "CO" else None
        ),
        "ubo_declaration": Doc(kind="ubo_declaration", issued=ago(15, 140), signed=True),
        "share_register": Doc(kind="share_register", issued=ago(30, 300)),
        "proof_of_address": Doc(kind="proof_of_address", issued=ago(5, 70)),
        "bank_reference": Doc(kind="bank_reference", issued=ago(5, 70)),
    }
    return {k: v.model_copy(update={"scan": rng.random() < 0.2}) for k, v in docs.items()}


def draft_case(case_id: str, shell: Shell, ctx: Ctx) -> Draft:
    company = company_from(shell, ctx)
    holders, officers = base_holders(ctx, company.jurisdiction)
    flags = {"sat_69b": shell.flag} if shell.flag else {}
    if shell.source == "rues" and shell.status != "ACTIVA":
        flags["rues_status"] = shell.status
    d = Draft(
        case_id=case_id,
        company=company,
        holders=holders,
        officers=officers,
        docs=base_docs(ctx, company.jurisdiction),
        flags=flags,
    )
    d.sync()
    return d


# --- planning ------------------------------------------------------------------------------------


def targets(n: int) -> list[str]:
    counts = {k: round(n * v) for k, v in MIX.items()}
    counts["MANUAL_REVIEW"] += n - sum(counts.values())
    out: list[str] = []
    for k, c in counts.items():
        out += [k] * c
    random.Random("mix").shuffle(out)
    return out


def _expected_rules(injections: list[str]) -> list[str]:
    return [r for r in (INJECTION_RULES[i] for i in injections) if r]


def build_one(
    i: int,
    target: str,
    plan: list[str],
    seed: int,
    raw: RawData,
    pseud: Pseudonymiser,
    shells: list[Shell],
) -> tuple[CaseSpec, Label, int]:
    case_id = f"syn-{seed:02d}-{i:03d}"
    last: tuple[CaseSpec, Label] | None = None
    for attempt in range(MAX_ATTEMPTS):
        rng = random.Random(f"{seed}/{i}/{attempt}")
        ctx = Ctx(rng=rng, pseud=pseud, raw=raw, persons=_queue(raw, seed, i, attempt))
        if shells:
            offshore_ok = {"high_risk_jurisdiction", "shell_indicators_high_risk"} & set(plan)
            pool = shells if target != "APPROVED" else [s for s in shells if s.status == "ACTIVA"]
            pool = pool or shells
            shell = rng.choice(pool if not offshore_ok else shells)
        else:
            shell = synthetic_shell(rng, i)
        d = draft_case(case_id, shell, ctx)
        for name in [n for n in plan if n not in CONFLICT_LAST]:
            INJECTIONS[name](d, ctx)
        d.sync()
        for name in [n for n in plan if n in CONFLICT_LAST]:
            INJECTIONS[name](d, ctx)
        spec = d.spec()
        lab = label(spec)
        last = (spec, lab)
        want = _expected_rules(plan)
        fired = set(lab.rules) | set(lab.unknown)
        ok_rules = all(r in fired for r in want)
        clean_ok = plan or not lab.fired
        if lab.status == target and ok_rules and clean_ok:
            return spec, lab, attempt
    assert last is not None
    raise RuntimeError(
        f"case {case_id}: could not reach {target} with {plan} in {MAX_ATTEMPTS} "
        f"attempts (last: {last[1].status} {last[1].rules})"
    )


def _queue(raw: RawData, seed: int, i: int, attempt: int) -> list:
    """A per-case stack of raw persons: disjoint across cases (so one registry row seeds at most one
    pseudonym per case), deterministic in (seed, i, attempt)."""
    if not raw.persons:
        return []
    r = random.Random(f"{seed}/persons/{i}/{attempt}")
    return [raw.persons[r.randrange(len(raw.persons))] for _ in range(12)]


def build(
    n: int, seed: int, raw: RawData | None = None, *, cache: Path | None = None
) -> tuple[list[CaseSpec], list[Label], BuildReport, Pseudonymiser]:
    raw = raw if raw is not None else load_raw()
    pseud = Pseudonymiser(seed, cache)
    shells = sorted(raw.shells, key=lambda s: (s.source, s.name, s.incorporated))
    tgt = targets(n)
    cursor = {k: 0 for k in PLANS}
    cases, labels, retries = [], [], 0
    for i, target in enumerate(tgt):
        plan = PLANS[target][cursor[target] % len(PLANS[target])]
        cursor[target] += 1
        spec, lab, attempts = build_one(i, target, plan, seed, raw, pseud, shells)
        cases.append(spec)
        labels.append(lab)
        retries += attempts
    pseud.save()
    return cases, labels, summarise(n, seed, cases, labels, raw, retries), pseud


def summarise(
    n: int, seed: int, cases: list[CaseSpec], labels: list[Label], raw: RawData, retries: int
) -> BuildReport:
    rules: Counter[str] = Counter()
    for lab in labels:
        rules.update(lab.rules)
        rules.update(lab.unknown)
    inj = Counter(i.name for c in cases for i in c.injections)
    return BuildReport(
        n=n,
        seed=seed,
        labels=dict(Counter(lab.status for lab in labels)),
        rules=dict(sorted(rules.items())),
        injections=dict(sorted(inj.items())),
        sources_used=raw.used,
        retries=retries,
    )


def index_rows(cases: list[CaseSpec], labels: list[Label]) -> list[dict]:
    """Case ids, labels and rule sets only: no names, safe to commit."""
    return [
        {
            "case_id": c.case_id,
            "jurisdiction": c.company.jurisdiction,
            "source": c.company.source,
            "status": lab.status,
            "score": lab.score,
            "rules": lab.rules,
            "injections": [i.name for i in c.injections],
        }
        for c, lab in zip(cases, labels, strict=True)
    ]


def write_outputs(
    cases: list[CaseSpec],
    labels: list[Label],
    report: BuildReport,
    seed: int,
    out: Path | None = None,
) -> Path:
    out = out or cache_root() / "cases" / f"seed-{seed}-n{len(cases)}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "cases.jsonl").write_text(
        "".join(c.model_dump_json() + "\n" for c in cases), encoding="utf-8"
    )
    (out / "labels.jsonl").write_text(
        "".join(lab.model_dump_json() + "\n" for lab in labels), encoding="utf-8"
    )
    (out / "report.json").write_text(report.model_dump_json(indent=1), encoding="utf-8")
    return out


def load_cases(path: Path) -> list[CaseSpec]:
    return [
        CaseSpec.model_validate_json(ln)
        for ln in (path / "cases.jsonl").read_text().splitlines()
        if ln.strip()
    ]


def digest(cases: list[CaseSpec]) -> str:
    return hashlib.sha256("".join(c.model_dump_json() for c in cases).encode()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build synthetic KYB cases from the raw snapshots.")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--raw", type=Path, default=None, help="raw snapshot dir (default: the cache)")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument(
        "--no-index", action="store_true", help="do not write datasets/synth/index.json"
    )
    args = ap.parse_args(argv)
    raw = load_raw(args.raw)
    cases, labels, report, _ = build(args.n, args.seed, raw)
    out = write_outputs(cases, labels, report, args.seed)
    if not args.no_index:
        INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        INDEX_PATH.write_text(
            json.dumps(
                {
                    "seed": args.seed,
                    "n": args.n,
                    "policy": "2026.11-r1",
                    "as_of": AS_OF.isoformat(),
                    "cases_sha256": digest(cases),
                    "cases": index_rows(cases, labels),
                },
                indent=1,
                ensure_ascii=False,
            )
            + "\n"
        )
    if not args.no_render:
        from kyb.synth.render import render_all

        render_all(cases, labels, cache_root() / "packets", seed=args.seed)
    print(report.model_dump_json(indent=1))
    print(f"cases: {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
