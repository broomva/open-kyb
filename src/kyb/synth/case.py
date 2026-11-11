"""The typed case spec: everything the label engine, the renderer and the fixture decider read.

A `CaseSpec` is the *facts* of a synthetic applicant packet: a company, who owns it, who runs it,
five documents with their dates and the claims each makes, and the named `injections` the builder
applied to provoke rules. The label engine derives status and score from the facts alone; the
injections are provenance (the manifest records them, the tests check each one fires its rule).
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

DocKind = Literal[
    "incorporation", "ubo_declaration", "share_register", "proof_of_address", "bank_reference"
]
DOC_KINDS: tuple[str, ...] = (
    "incorporation",
    "ubo_declaration",
    "share_register",
    "proof_of_address",
    "bank_reference",
)
HolderKind = Literal["natural", "legal", "foundation", "trust"]
OfficerRole = Literal["director", "legal_rep", "nominee_director", "registered_agent"]
INJECTION_RULES = {
    "sanctions_corroborated": "SCR-01",
    "sanctions_near_miss": "SCR-02",
    "prohibited_jurisdiction": "JUR-01",
    "high_risk_jurisdiction": "JUR-02",
    "unresolved_ownership": "OWN-01",
    "unresolved_ownership_high_risk": "OWN-02",
    "conflict_pct": "DOC-01",
    "conflict_presence": "DOC-01",
    "conflict_address": "DOC-01",
    "conflict_reg_no": "DOC-01",
    "pep": "PEP-01",
    "pep_former": "PEP-02",
    "nominee": "GOV-01",
    "nominee_sole_controller": "GOV-02",
    "dormant_or_layered": "GOV-03",
    "shell_indicators": "OPS-01",
    "shell_indicators_high_risk": "OPS-02",
    "missing_document": "DOC-02",
    "expired_document": "DOC-02",
    "industry_high_risk": "ACT-02",
    "industry_prohibited": "ACT-01",
    "industry_unlicensed": "ACT-01",
    "pep_decoy": "",
}  # injection name → the rule it must fire (checked by tests)


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Person(_Frozen):
    pid: str
    name: str
    nationality: str  # ISO-3166 alpha-2
    dob: date
    domicile: str  # ISO alpha-2, optionally "UA:Crimea" (code:region)


class Holder(_Frozen):
    """A direct holder of the applicant. A legal holder is `resolved` when its chain ends in
    `person`; a natural holder is its own person."""

    hid: str
    kind: HolderKind
    name: str
    pct: int = Field(ge=0, le=100)
    domicile: str
    reg_no: str | None = None
    nominee: bool = False
    chain: list[str] = []  # intermediate entities between the holder and `person`, outermost first
    person: Person | None = None  # the natural person the chain resolves to, if it does


class Officer(_Frozen):
    name: str
    role: OfficerRole
    person: Person | None = None  # None for a nominee or an agent (a legal person)


class DocClaims(_Frozen):
    """What one document says about the packet's other facts. `None` means it does not say."""

    holders: dict[str, int] | None = None  # holder name → percentage the document states
    address: str | None = None
    reg_no: str | None = None


class Doc(_Frozen):
    kind: DocKind
    present: bool = True
    issued: date | None = None
    signed: bool = True
    stated_validity_days: int | None = None
    scan: bool = False  # the renderer degrades it to an image-only scan
    claims: DocClaims = DocClaims()


class Company(_Frozen):
    name: str
    tax_id: str  # RFC / NIT / registration number
    jurisdiction: str  # ISO alpha-2
    region: str | None = None  # "Crimea", "DMCC free zone" ...
    incorporated: date
    activities: list[str]  # the declared corporate object, verbatim
    ciiu: str | None = None
    status: str = "ACTIVA"
    registered_address: str
    mass_registration_address: bool = False
    operating_address: str | None = None  # None: only the registered address is given
    employees_or_website: bool = True
    source: str = "synthetic"


class Injection(_Frozen):
    name: str
    params: dict[str, Any] = {}


class CaseSpec(_Frozen):
    case_id: str
    as_of: date
    company: Company
    holders: list[Holder]
    officers: list[Officer] = []
    documents: list[Doc]
    injections: list[Injection] = []
    source_flags: dict[str, str] = {}  # e.g. {"sat_69b": "Definitivo"}: recorded, never a rule

    def doc(self, kind: str) -> Doc | None:
        return next((d for d in self.documents if d.kind == kind and d.present), None)

    def injection(self, name: str) -> Injection | None:
        return next((i for i in self.injections if i.name == name), None)
