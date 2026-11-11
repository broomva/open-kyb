"""Three hand-encoded synthetic case specs (A, B, C) with locked labels.

The label engine must reproduce `GOLDEN` exactly (A, B, C: MANUAL_REVIEW 15, 85, 90).
The applicants are fictional; nothing here is sourced.
"""

from __future__ import annotations

from datetime import date

from kyb.synth.case import CaseSpec, Company, Doc, DocClaims, Holder, Officer, Person

AS_OF = date(2026, 10, 25)
GOLDEN: dict[str, tuple[str, int, tuple[str, ...]]] = {
    "A": ("MANUAL_REVIEW", 15, ("DOC-02",)),
    "B": ("MANUAL_REVIEW", 85, ("DOC-01", "MOD-01", "OPS-01", "OWN-01")),
    "C": (
        "MANUAL_REVIEW",
        90,
        ("ACT-02", "DOC-02", "GOV-01", "JUR-02", "OPS-01", "PEP-01", "SCR-02"),
    ),
}


def _person(pid: str, name: str, nat: str, dob: date, dom: str | None = None) -> Person:
    return Person(pid=pid, name=name, nationality=nat, dob=dob, domicile=dom or nat)


def packet_a() -> CaseSpec:
    herrera = _person("herrera", "Herrera", "CO", date(1984, 4, 27))
    gaviria = _person("gaviria", "Gaviria", "CO", date(1981, 8, 6))
    held = {"Herrera": 55, "Gaviria": 45}
    addr = "Avenida El Dorado No. 68C-61, Of. 404, Bogotá"
    return CaseSpec(
        case_id="cl-andes-agro",
        as_of=AS_OF,
        company=Company(
            name="Andes Agropecuaria SpA",
            tax_id="900123456-7",
            jurisdiction="CO",
            incorporated=date(2019, 5, 20),
            ciiu="5229",
            activities=["Actividades de las agencias de carga (freight forwarding)"],
            registered_address=addr,
        ),
        holders=[
            Holder(hid="h1", kind="natural", name="Herrera", pct=55, domicile="CO", person=herrera),
            Holder(hid="h2", kind="natural", name="Gaviria", pct=45, domicile="CO", person=gaviria),
        ],
        officers=[Officer(name="Herrera", role="legal_rep", person=herrera)],
        documents=[
            Doc(
                kind="incorporation",
                issued=date(2026, 9, 8),
                stated_validity_days=30,
                claims=DocClaims(address=addr),
            ),
            Doc(kind="ubo_declaration", issued=date(2026, 8, 21), claims=DocClaims(holders=held)),
            Doc(kind="share_register", issued=date(2026, 6, 10), claims=DocClaims(holders=held)),
            Doc(kind="proof_of_address", issued=date(2026, 10, 2), scan=True),
            Doc(kind="bank_reference", issued=date(2026, 8, 25)),
        ],
    )


def packet_b() -> CaseSpec:
    navarro = _person("navarro", "L. Navarro", "MX", date(1969, 1, 25))
    ortega = _person("ortega", "Valeria Ortega Méndez", "MX", date(1976, 7, 12))
    register = {"L. Navarro": 50, "Valeria Ortega Méndez": 25, "Inversiones Altiplano S.A.": 25}
    return CaseSpec(
        case_id="es-iberia-software",
        as_of=AS_OF,
        company=Company(
            name="Iberia Software S.L.",
            tax_id="MCH180416AB1",
            jurisdiction="MX",
            incorporated=date(2018, 4, 16),
            activities=[
                "la adquisición y tenencia de acciones de otras sociedades",
                "la inversión en bienes inmuebles",
            ],
            registered_address="Avenida Chapultepec 480 P9, CDMX",
            employees_or_website=False,
        ),
        holders=[
            Holder(
                hid="h1", kind="natural", name="L. Navarro", pct=50, domicile="MX", person=navarro
            ),
            Holder(
                hid="h2",
                kind="natural",
                name="Valeria Ortega Méndez",
                pct=25,
                domicile="MX",
                person=ortega,
            ),
            Holder(
                hid="h3", kind="legal", name="Inversiones Altiplano S.A.", pct=25, domicile="PA"
            ),
        ],
        officers=[Officer(name="L. Navarro", role="director", person=navarro)],
        documents=[
            Doc(
                kind="incorporation",
                issued=date(2018, 4, 16),
                claims=DocClaims(holders={"L. Navarro": 60, "Valeria Ortega Méndez": 40}),
            ),
            Doc(
                kind="ubo_declaration",
                issued=date(2026, 10, 10),
                claims=DocClaims(holders=register),
            ),
            Doc(
                kind="share_register", issued=date(2026, 7, 23), claims=DocClaims(holders=register)
            ),
            Doc(kind="proof_of_address", issued=date(2026, 5, 10)),
            Doc(kind="bank_reference", present=False),
        ],
    )


def packet_c() -> CaseSpec:
    paredes = _person("paredes", "Sebastián Paredes Vega", "PY", date(1971, 12, 4))
    return CaseSpec(
        case_id="sg-strait-capital",
        as_of=AS_OF,
        company=Company(
            name="Strait Capital Pte. Ltd.",
            tax_id="202419082K",
            jurisdiction="SG",
            incorporated=date(2026, 1, 14),
            activities=["agricultural trade finance"],
            registered_address="Apex Chambers, Wickhams Cay, Sovereign Town, Tortola",
            mass_registration_address=True,
            operating_address="Asunción",
        ),
        holders=[
            Holder(
                hid="h1",
                kind="foundation",
                name="Helios Foundation",
                pct=100,
                domicile="PA",
                person=paredes,
            )
        ],
        officers=[
            Officer(name="Apex Corporate Services Ltd", role="nominee_director"),
            Officer(name="Apex Corporate Services Ltd", role="registered_agent"),
        ],
        documents=[
            Doc(kind="incorporation", issued=date(2026, 1, 14)),
            Doc(kind="ubo_declaration", issued=date(2026, 9, 11)),
            Doc(kind="share_register", issued=date(2026, 1, 14)),
            Doc(kind="proof_of_address", issued=date(2026, 2, 3), scan=False),
            Doc(kind="bank_reference", issued=date(2026, 9, 9), scan=True),
        ],
    )


def golden_specs() -> dict[str, CaseSpec]:
    return {"A": packet_a(), "B": packet_b(), "C": packet_c()}
