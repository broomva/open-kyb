"""`derive_facts` (L2, spec): verified claims → `FactSet` (b4-v1). Pure: no I/O, no model.

Only claims the witness verified become facts. A failed claim is listed in `FactSet.unverified`
and can only make a fact *unknown* (never absent, never present). Readers write claims in an open
vocabulary (`type`, `attribute`, `value`), so each claim is classified by its attribute first and
its type second; a claim no class covers stays in the evidence table and feeds no fact.

- **Entities:** every party a claim names, clustered across spellings. Organisations match on
  their name without legal-form suffixes (`org_key`); persons on ordered name tokens, initials
  allowed (`verify.subject_match`, the binding's own matcher). An ambiguous match makes a new
  entity rather than merging two parties. The applicant is the organisation most documents name.
- **Holdings:** one per (owner, owned, document, status). `struck` from the reader's
  `struck-through` flag; `pending` from a transfer or amendment the document itself calls
  pending; `historical` for an incorporation deed superseded by a later dated document; else
  `current`. Other transfers are kept as conflict *context*.
- **Conflicts (spec):** one per (subject, attribute) with two or more material values, never
  resolved: ownership beyond the policy's point tolerance, an owner present in one shareholder
  listing and absent from another, different registered addresses, different registration or
  tax numbers. A pending transfer is context, not a value. Absence counts only where the listing
  is complete as far as verification can tell.
- **Operations (shell indicators ii/iv/v):** `absent` only with a reason the packet shows; otherwise
  `unknown`.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any

from kyb.contracts.b4 import (
    Address,
    Conflict,
    ConflictValue,
    Document,
    DocVerification,
    Entity,
    EvidenceItem,
    EvidenceRef,
    FactSet,
    Holding,
    Operations,
    Role,
    TriFact,
    UncoveredFact,
)
from kyb.facts.text import (
    FOUNDATION_MARKERS,
    TRUST_MARKERS,
    akey,
    base_norm,
    countries,
    fold,
    is_org,
    merged_tokens,
    org_key,
    regions,
    same_address,
    significant_tokens,
    strip_suffixes,
)
from kyb.util.hashing import sha256_json
from kyb.witness.norm import contains, parse_date
from kyb.witness.verify import canon_attr, subject_match

METHOD = "facts-v2"
_CONFLICT_POINTS = re.compile(
    r"differing by more than\s+\**(\d+(?:[.,]\d+)?)\**\s+(?:percentage\s+)?points",
    re.IGNORECASE,
)


def materiality(policy_text: str) -> Decimal:
    """Ownership-conflict materiality as the policy states it.
    Refuses a policy where it cannot be found: a default would count conflicts by a rule nobody
    wrote (invariant 3)."""
    found = {_dec(m) for m in _CONFLICT_POINTS.findall(policy_text)}
    if len(found) != 1 or None in found:
        raise ValueError("the policy does not state exactly one ownership materiality "
                         "('differing by more than <n> points')")  # fmt: skip
    return found.pop()


@dataclass(frozen=True)
class DocInput:
    """One document as verification left it: its claims, the witness's verdict per claim, and
    the assurance as of now (`assurance_now`)."""

    doc: str
    document_name: str
    claims: list[dict]
    verdicts: dict[str, bool]
    assurance: str = "W1"
    isolated: bool = False
    coverage: list[dict] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)


# --- the claim vocabulary ---------------------------------------------------------------------

_DOC_DATE = (
    "issue_date",
    "issued_on",
    "date_issued",
    "issued",
    "fecha_de_emision",
    "fecha_de_expedicion",
    "expedition_date",
    "document_date",
    "date",
    "fecha",
    "letter_date",
    "statement_date",
    "certificate_date",
    "declaration_date",
    "last_updated",
    "last_update",
    "ultima_actualizacion",
    "updated_on",
    "as_at",
    "as_of",
    "register_entry_date",
    "entry_date",
    "meeting_date",
    "fecha_de_sesion",
    "session_date",
)  # in priority order
_SIGN_DATE = {
    "signing_date",
    "signed_on",
    "date_signed",
    "fecha_de_firma",
    "signature_date",
    "signed_date",
    "date_of_signature",
}
_VALIDITY = {"stated_validity", "validity_period", "vigencia", "valid_for", "validity", "validez"}
_DOC_OTHER = {
    "place_of_signing",
    "place_of_meeting",
    "meeting_location",
    "deed_number",
    "notary",
    "notaria",
    "register_entry",
    "verification_code",
    "codigo_de_verificacion",
    "term",
    "lease_term",
    "plazo",
    "rent",
    "canon",
    "canon_mensual",
    "page",
    "document_title",
    "issuer",
    "issuing_authority",
}
_SIGNATURE = {"signed_by", "signature", "signatory_name", "firma", "firmante", "signed"}
_DOB = {"dob", "date_of_birth", "birth_date", "fecha_de_nacimiento", "born", "birthdate"}
_NATIONALITY = {"nationality", "nacionalidad", "citizenship", "ciudadania"}
_JURISDICTION = {
    "jurisdiction",
    "country",
    "country_of_incorporation",
    "place_of_incorporation",
    "incorporated_in",
    "jurisdiction_of_incorporation",
    "constituted_under",
    "governing_law",
    "pais",
    "pais_de_constitucion",
    "domicile_country",
    "residence",
    "country_of_residence",
    "residencia",
    "pais_de_residencia",
    "incorporation_jurisdiction",
}
_INCORPORATED = {
    "incorporation_date",
    "incorporated_on",
    "fecha_de_constitucion",
    "date_of_incorporation",
    "constitution_date",
    "incorporated",
}
_NAME = {
    "legal_name",
    "name",
    "razon_social",
    "company_name",
    "denominacion",
    "denominacion_social",
    "full_name",
    "nombre",
    "entity_name",
}
_IDS = {
    "tax_id": {"tax_id", "rfc", "nit", "rut", "cuit", "tin", "ruc", "vat", "tax_number", "ein"},
    "reg_no": {
        "registry_id",
        "folio_mercantil",
        "folio_mercantil_electronico",
        "registration_number",
        "company_number",
        "matricula",
        "matricula_mercantil",
        "reg_no",
        "registry_number",
        "registration_no",
        "company_no",
        "numero_de_registro",
    },
    "passport": {"passport", "pasaporte", "passport_number", "passport_no"},
    "national_id": {
        "cedula",
        "c_c",
        "cc",
        "dni",
        "curp",
        "national_id",
        "id_number",
        "identification",
        "identificacion",
        "cedula_de_ciudadania",
    },
}
_ADDR = {
    "registered": {
        "registered_address",
        "domicilio_social",
        "registered_domicile",
        "registered_office",
        "domicilio_principal",
        "principal_domicile",
        "domicilio_legal",
        "legal_address",
    },
    "operating": {
        "operating_address",
        "domicilio_operativo",
        "office_address",
        "premises",
        "leased_premises",
        "property",
        "property_address",
        "inmueble",
        "business_address",
        "place_of_business",
        "establishment_address",
    },
    "fiscal": {"fiscal_address", "domicilio_fiscal", "tax_address"},
    "residential": {
        "residential_address",
        "residence_address",
        "home_address",
        "domicilio_particular",
    },
    "unknown": {
        "address",
        "domicilio",
        "direccion",
        "service_address",
        "billing_address",
        "mailing_address",
        "supply_address",
    },
}
_ACTIVITY = {
    "activity",
    "activities",
    "objeto_social",
    "corporate_object",
    "economic_activity",
    "actividad_economica",
    "actividad_principal",
    "actividad",
    "business_activity",
    "nature_of_business",
    "line_of_business",
    "industry",
    "ciiu",
    "business",
    "principal_activity",
    "purpose",
    "objeto",
    "corporate_purpose",
}
_OPS_V = {
    "employees",
    "employee_count",
    "number_of_employees",
    "headcount",
    "staff",
    "empleados",
    "numero_de_empleados",
    "website",
    "web",
    "url",
    "web_site",
    "sitio_web",
    "pagina_web",
    "domain",
    "web_page",
}
_BANK = {
    "bank_account",
    "account",
    "account_number",
    "account_since",
    "account_opened",
    "bank",
    "bank_reference",
    "cuenta",
    "cuenta_corriente",
    "account_holder",
    "bank_relationship",
}
_ROLE = {
    "role",
    "board_role",
    "title",
    "position",
    "cargo",
    "capacity",
    "caracter",
    "signatory",
    "signatory_role",
    "director",
    "directors",
    "president",
    "secretary",
    "legal_representative",
    "legal_rep",
    "representante_legal",
    "registered_agent",
    "agente_residente",
    "resident_agent",
    "agente_registrado",
    "nominee",
    "nominee_director",
    "nominee_shareholder",
    "beneficiary",
    "beneficiario",
    "council_member",
    "founder",
    "fundador",
    "consejo",
    "officer",
    "council",
    "consejo_de_fundacion",
    "principal",
    "controller",
    "manager",
    "gerente",
    "ceo",
    "control",
}
_HOLD_TYPES = {
    "owns",
    "ownership",
    "shareholding",
    "holding",
    "beneficial_ownership",
    "participation",
}
_HOLD_ATTRS = {
    "shareholding",
    "participation",
    "participacion",
    "ownership",
    "beneficial_ownership",
    "ultimate_ownership",
    "indirect_ownership",
    "percentage",
    "pct",
    "percent",
    "holding",
    "ownership_pct",
    "shareholding_pct",
    "shareholding_percentage",
    "ownership_percentage",
    "porcentaje",
    "shares",
    "acciones",
}
_TRANSFER_WORDS = ("transfer", "transmision", "acquisition", "adquisicion", "cesion", "amendment")
_VIA = ("via", "through", "indirect", "a_traves_de")
_RELATIONS = ("holding", "role", "signature", "transfer")
_LEGAL_FORMS = {"sa", "sas", "ltd", "ltda", "limited", "llc", "inc", "corp", "corporation", "gmbh",
                "fze", "fzco", "plc", "srl", "cv", "saeca", "sae", "bv", "nv", "ag", "spa",
                "sarl", "lp", "llp", "pte", "pty"}  # fmt: skip
_PENDING = re.compile(
    r"pendiente|pending|por inscribir|to be registered|sin inscribir|"
    r"not yet registered|subject to registration"
)
_DOC_TITLE = (
    "declaracion",
    "declaration",
    "acta ",
    "certificado",
    "certificate",
    "constancia",
    "contrato",
    "contract",
    "libro",
    "register",
    "registro",
    "asiento",
    "escritura",
    "ultimate beneficial",
    "extracto",
    "extract",
    "total",
    "pagina",
    "page",
    "lease",
    "arrendamiento",
    "carta",
    "letter",
    "bank reference",
    "referencia",
    "recibo",
    "factura",
    "invoice",
    "minutes",
    "sesion",
    "folio",
)
_DOC_TYPES = (  # first match on the folded file name
    ("bank_reference", "bank_reference"),
    ("referencia_bancaria", "bank_reference"),
    ("certificacion_bancaria", "bank_reference"),
    ("financial", "financial_statement"),
    ("estados_financieros", "financial_statement"),
    ("balance", "financial_statement"),
    ("ubo", "ubo_declaration"),
    ("beneficiar", "ubo_declaration"),
    ("beneficial", "ubo_declaration"),
    ("titularidad", "ubo_declaration"),
    ("proof_of_address", "proof_of_address"),
    ("comprobante", "proof_of_address"),
    ("utility", "proof_of_address"),
    ("suministro", "proof_of_address"),
    ("lease", "lease"),
    ("arrendamiento", "lease"),
    ("register_of", "share_register"),
    ("shareholder", "share_register"),
    ("accionistas", "share_register"),
    ("members", "share_register"),
    ("acta_junta", "board_minutes"),
    ("minutes", "board_minutes"),
    ("asamblea", "board_minutes"),
    ("fideicomiso", "other"),
    ("trust_deed", "other"),
    ("constitutiva", "incorporation"),
    ("incorporation", "incorporation"),
    ("existencia", "incorporation"),
    ("vigencia", "incorporation"),
    ("acra", "incorporation"),
    ("escritura", "incorporation"),
    ("articles", "incorporation"),
    ("estatutos", "incorporation"),
    ("constancia", "tax_certificate"),
    ("rfc", "tax_certificate"),
    ("tributari", "tax_certificate"),
    ("tax", "tax_certificate"),
    ("bank", "bank_reference"),
)


def doc_type(document_name: str) -> str:
    name = akey(document_name.rsplit(".", 1)[0])
    return next((t for k, t in _DOC_TYPES if k in name), "other")


def _text(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(str(v) for v in value.values())
    return str(value)


def _dec(x: Any) -> Decimal | None:
    """A number as an exact, canonical Decimal ("55,00" → 55; "25.5" → 25.5)."""
    if isinstance(x, bool):
        return None
    if isinstance(x, int | float):
        raw = str(x)
    else:
        m = re.search(r"\d+(?:[.,]\d+)?", str(x))
        if not m:
            return None
        raw = m[0].replace(",", ".")
    try:
        d = Decimal(raw)
    except InvalidOperation:
        return None
    return d.quantize(Decimal(1)) if d == d.to_integral_value() else d.normalize()


def _pct(value: Any, attr: str) -> Decimal | None:
    if isinstance(value, dict):
        for k in ("pct", "percent", "percentage", "porcentaje", "ownership_pct", "participacion"):
            if k in value:
                return _dec(value[k])
        return None
    if isinstance(value, int | float) and not isinstance(value, bool):
        pct_like = canon_attr(attr) == "pct" or attr in _HOLD_ATTRS - {"shares", "acciones"}
        return _dec(value) if pct_like else None
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*%", str(value))
    if m:
        return _dec(m[1])
    if canon_attr(attr) == "pct" and re.fullmatch(r"\s*\d+(?:[.,]\d+)?\s*", str(value)):
        return _dec(value)
    return None


def _shares(value: Any, attr: str) -> int | None:
    raw: Any = None
    if isinstance(value, dict):
        raw = next((value[k] for k in ("shares", "acciones", "share_count") if k in value), None)
    elif canon_attr(attr) == "shares":
        raw = value
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    m = re.search(r"\d{1,3}(?:[.,]\d{3})+|\d+", str(raw))
    return int(re.sub(r"[.,]", "", m[0])) if m else None


def _role_kind(text: str) -> str:
    t = " ".join(akey(text).split("_"))
    rules = (
        (r"nominee|nominad|testaferro", None),
        (r"registered agent|agente registrado|agente residente|resident agent", "registered_agent"),
        (r"legal rep|representante legal|apoderad", "legal_rep"),
        (
            r"consejo de fundacion|foundation council|council member|miembro del consejo"
            r"|\bcouncil\b",
            "council_member",
        ),
        (r"presiden|chair", "president"),
        (r"secretar", "secretary"),
        (r"beneficiar", "beneficiary"),
        (r"director|consejer|administrador", "director"),
        (r"sign|firm", "signatory"),
    )
    for pattern, kind in rules:
        if re.search(pattern, t):
            if kind is None:
                return (
                    "nominee_shareholder"
                    if re.search(r"sharehold|accionist|member", t)
                    else "nominee_director"
                )
            return kind
    return "other"


def _looks_like_name(text: str) -> bool:
    """A party's name as documents write one: capitalised words, no digits."""
    if not text or re.search(r"\d", text) or len(text) > 120:
        return False
    if is_org(text):
        return True
    words = [w for w in re.split(r"[\s,]+", text.strip()) if w]
    connectors = {"de", "del", "la", "las", "los", "y", "da", "dos", "van", "von"}
    named = [w for w in words if fold(w).strip(".") not in connectors]
    return 2 <= len(named) <= 6 and all(w[0].isupper() for w in named if w[0].isalpha())


def _is_doc_title(subject: str) -> bool:
    s = base_norm(subject) + " "
    return any(s.startswith(p) for p in _DOC_TITLE)


# --- claims -----------------------------------------------------------------------------------


@dataclass
class _Claim:
    raw: dict
    doc: str
    eid: str
    order: int
    verified: bool
    typ: str
    attr: str
    kind: str = ""  # the class: doc, signature, transfer, holding, role, identity, address, ...

    @property
    def ref(self) -> EvidenceRef:
        return EvidenceRef(
            claim_id=self.eid, doc=self.doc, page=self.raw["page"], verified=self.verified
        )

    @property
    def subject(self) -> str:
        return str(self.raw.get("subject") or "").strip()

    @property
    def obj(self) -> str:
        return str(self.raw.get("object") or "").strip()

    @property
    def value(self) -> Any:
        return self.raw.get("value")


def _classify(c: _Claim) -> str:
    a, t = c.attr, c.typ
    if t == "integrity":
        return "integrity"
    if t == "signature" or a in _SIGNATURE:
        return "signature"
    if (
        t in ("transfer", "amendment")
        or canon_attr(a) == "transfer"
        or any(w in a for w in _TRANSFER_WORDS)
    ):
        return "transfer"
    if t != "capital" and (t in _HOLD_TYPES or a in _HOLD_ATTRS):
        return "holding"
    if a in _DOC_DATE or a in _SIGN_DATE or a in _VALIDITY or a in _DOC_OTHER:
        return "doc"
    if t == "role" or a in _ROLE:
        return "role"
    if a in _DOB | _NATIONALITY | _JURISDICTION | _INCORPORATED | _NAME or any(
        a in ids for ids in _IDS.values()
    ):
        return "identity"
    if t == "address" or any(a in s for s in _ADDR.values()):
        return "address"
    if a in _ACTIVITY or (t in ("corporate_activity", "activity") and a == t):
        return "activity"
    if a in _OPS_V:
        return "ops"
    if a in _BANK:
        return "bank"
    return "other"


# --- entities -----------------------------------------------------------------------------------


def _same_party(a: str, b: str) -> bool:
    oa, ob = is_org(a), is_org(b)
    if oa and ob:
        return bool(org_key(a)) and org_key(a) == org_key(b)
    if oa != ob:
        return base_norm(a) == base_norm(b)
    if base_norm(a) == base_norm(b):
        return True
    if len(base_norm(a).split()) < 2 or len(base_norm(b).split()) < 2:
        return False
    return subject_match(a, b) or subject_match(b, a)


def _abbreviated(name: str) -> bool:
    """A spelling with an initial ("A. Delgado V.", "E. Cifuentes")."""
    return any(len(t.strip(".,")) == 1 or re.fullmatch(r"(?:\w\.){2,}", t) for t in name.split())


@dataclass
class _Cluster:
    names: list[str]
    first: int
    docs: set[str] = field(default_factory=set)
    eid: str = ""
    name_docs: dict[str, set[str]] = field(default_factory=dict)


class _Parties:
    """Clusters of name spellings; `resolve` never creates one.

    Two spellings are never merged when the same document writes both in full ("Ana Perez" and
    "Ana Maria Perez" as two shareholder rows are two people), or when their verified identifiers
    disagree (a different date of birth, passport, national or tax id)."""

    def __init__(self, idents: dict[str, dict[str, set[str]]] | None = None) -> None:
        self.clusters: list[_Cluster] = []
        self.idents = idents or {}

    def _matches(self, name: str) -> list[_Cluster]:
        return [k for k in self.clusters if any(_same_party(name, n) for n in k.names)]

    def _separate(self, name: str, doc: str, k: _Cluster) -> bool:
        mine = self.idents.get(name, {})
        for n in k.names:
            if base_norm(n) == base_norm(name):
                continue
            persons = not (is_org(n) or is_org(name))
            full = not (_abbreviated(n) or _abbreviated(name))
            if persons and full and doc in k.name_docs.get(n, set()):
                return True  # two people written in full in one document (Codex r1 #3, r2 #6)
            theirs = self.idents.get(n, {})
            if any(mine[key] and theirs.get(key) and not mine[key] & theirs[key] for key in mine):
                return True
        return False

    def add(self, name: str, doc: str, order: int) -> None:
        if not name or ";" in name:
            return
        exact = [k for k in self.clusters if name in k.names]
        hits = exact or [k for k in self._matches(name) if not self._separate(name, doc, k)]
        if len(hits) == 1:
            k = hits[0]
            if name not in k.names:
                k.names.append(name)
        else:  # none, or ambiguous: a new party rather than a merge of two
            k = _Cluster(names=[name], first=order)
            self.clusters.append(k)
        k.docs.add(doc)
        k.name_docs.setdefault(name, set()).add(doc)

    def resolve(self, name: str) -> _Cluster | None:
        if not name or ";" in name:
            return None
        exact = [k for k in self.clusters if name in k.names]
        if len(exact) == 1:
            return exact[0]
        hits = self._matches(name)
        return hits[0] if len(hits) == 1 else None

    def mentioned_in(self, text: str) -> list[_Cluster]:
        """Parties free text names by a significant token (transfer context only)."""
        t = f" {base_norm(text)} "
        return [
            k
            for k in self.clusters
            if any(f" {tok} " in t for n in k.names for tok in significant_tokens(n))
        ]


def _slug(names: list[str]) -> str:
    best = max(names, key=lambda n: (len(base_norm(n).split()), len(n), base_norm(n)))
    toks = org_key(best).split() if is_org(best) else base_norm(best).split()
    return "-".join(toks)[:40].strip("-") or "party"


def _kind(names: list[str], has_person_facts: bool) -> str:
    toks = {t for n in names for t in merged_tokens(n)}
    if toks & FOUNDATION_MARKERS:
        return "foundation"
    if toks & TRUST_MARKERS:
        return "trust"
    if any(is_org(n) for n in names):
        return "company"
    if has_person_facts or any(_looks_like_name(n) for n in names):
        return "person"
    return "unknown"


# --- the derivation -----------------------------------------------------------------------------


@dataclass
class _E:
    """An entity under construction."""

    eid: str
    names: list[str]
    kind: str
    ids: dict[str, list[str]] = field(default_factory=dict)
    nationality: list[str] = field(default_factory=list)
    dob: list[str] = field(default_factory=list)
    jurisdiction: list[str] = field(default_factory=list)
    incorporated: list[str] = field(default_factory=list)
    addresses: list[tuple[str, str, list[_Claim]]] = field(default_factory=list)
    evidence: list[_Claim] = field(default_factory=list)


def _add(xs: list, x: Any) -> None:
    if x not in xs:
        xs.append(x)


def _refs(claims: list[_Claim]) -> list[EvidenceRef]:
    seen: dict[str, _Claim] = {}
    for c in sorted(claims, key=lambda c: c.order):
        seen.setdefault(c.eid, c)
    return [c.ref for c in seen.values()]


def derive_factset(
    case: str, run: str, as_of: str, docs: list[DocInput], *, materiality: Decimal
) -> FactSet:
    """The `FactSet` for one run. Deterministic: same inputs, same bytes. `materiality` is the
    ownership threshold in points, read from the policy (`materiality(policy_text)`)."""
    claims = _claims(docs)
    fields: dict[str, str] = {}

    def cite(c: _Claim, what: str) -> _Claim:
        fields.setdefault(c.eid, what)
        return c

    verified = [c for c in claims if c.verified]
    by_doc: dict[str, list[_Claim]] = defaultdict(list)
    for c in claims:
        by_doc[c.doc].append(c)
    dtype = {d.doc: doc_type(d.document_name) for d in docs}

    parties = _parties(verified)
    applicant = _applicant(parties)
    if applicant is None:
        raise ValueError(f"{case} {run}: no verified claim names an organisation (the applicant)")
    person_facts = {
        id(k)
        for c in verified
        if c.kind == "identity" and c.attr in _DOB | _NATIONALITY
        for k in [parties.resolve(c.subject)]
        if k
    }
    used: set[str] = set()
    ents: dict[int, _E] = {}
    for k in sorted(parties.clusters, key=lambda k: k.first):
        slug, n = _slug(k.names), 2
        eid = f"e-{slug}"
        while eid in used:
            eid, n = f"e-{slug}-{n}", n + 1
        used.add(eid)
        k.eid = eid
        ents[id(k)] = _E(eid=eid, names=list(k.names), kind=_kind(k.names, id(k) in person_facts))
    app = ents[id(applicant)]

    def ent(name: str) -> _E | None:
        k = parties.resolve(name) if name and not _is_doc_title(name) else None
        return ents[id(k)] if k else None

    principal = _principals(verified, ent)

    def foreign(c: _Claim, allowed: list[_E | None]) -> bool:
        """Does the excerpt name a party other than the allowed ones? Read on the intact
        excerpt: every known party's spelling is located as a token span and the longest span
        wins, so "Andes Agropecuaria SpA" inside "GRUPO ANDES AGROPECUARIA SpA" belongs to the
        longer name (Codex r4 #1). Foreign: a span of another party; a legal form outside every
        span; or an allowed company whose name is glued to a preceding capitalised word (a
        longer company name nobody registered)."""
        excerpt = c.raw.get("excerpt", "")
        toks = merged_tokens(excerpt)
        ids = {e.eid for e in allowed if e}
        kept = _spans(toks, ents.values())
        if any(eid not in ids for _, _, eid in kept):
            return True
        covered = {i for s0, s1, _ in kept for i in range(s0, s1)}
        if any(t in _LEGAL_FORMS for i, t in enumerate(toks) if i not in covered):
            return True
        orgs = [e for e in allowed if e and e.kind != "person"]
        return any(_glued(excerpt, n) for e in orgs for n in e.names)

    def bound(c: _Claim, subj: _E | None, name: str) -> _E | None | bool:
        """The party a relationship claim points at, if verification binds it; False if not.

        The witness checks subject and value, never the object. So a target is bound only when
        the verified excerpt quotes it, or it is the applicant or the document's principal party
        (the one organisation whose identity the document verifiably states) and the excerpt
        names no other party. An ownership claim with no object defaults to the applicant under
        the same proviso. Codex r2: document co-occurrence alone is not a binding."""
        if not name:
            return False if foreign(c, implicit(c, subj)) else None
        e = ent(name)
        if e is None:
            return False
        if quoted(c, e):
            return e
        if (e is app or e.eid == principal.get(c.doc)) and not foreign(c, [subj, e]):
            return e
        return False

    def quoted(c: _Claim, e: _E) -> bool:
        """The excerpt quotes the party: one of the token spans it resolves to (longest name
        first) is that party's. "Trusted Ltd" is not quoted by "Trusted Holdings Ltd" (Codex r3
        #2), and "Acme Ltd" is quoted by "Parent Acme Ltd owns 25% of Acme Ltd" (r4 #2)."""
        excerpt = c.raw.get("excerpt", "")
        if not any(eid == e.eid for _, _, eid in _spans(merged_tokens(excerpt), ents.values())):
            return False
        # a company quoted only as the tail of a longer name is not quoted (Codex r5 #1)
        return e.kind == "person" or any(_free(excerpt, n) for n in e.names)

    def implicit(c: _Claim, subj: _E | None) -> list[_E | None]:
        """The parties an object-less relationship is about, so its excerpt may name only them:
        the subject and the applicant it defaults to, or for `registered_agent: Apex` the
        subject and the holder its value names (Codex r3 #1)."""
        if c.kind in ("role",) and _holder_in_value(c):
            named = ent(str(c.value).strip())
            if named is not None:
                return [subj, named]
        return [subj, app]

    # a relationship nothing binds is not a verified fact: it reads as unverified downstream
    unbound: set[str] = set()
    for c in verified:
        if c.kind not in _RELATIONS or (c.kind == "transfer" and not c.obj):
            continue  # an object-less transfer names its counterparty by nature: context only
        subj = ent(c.subject)
        ok = bound(c, subj, c.obj) is not False
        v = c.value
        if ok and c.kind == "holding" and isinstance(v, dict):
            via = next((v[k] for k in _VIA if k in v), None)
            ok = via is None or bool(bound(c, subj, str(via)))
        if not ok:
            unbound.add(c.eid)
            c.verified = False
    verified = [c for c in verified if c.verified]

    # documents first: holdings take their as_of from them
    documents = [_document(d, by_doc[d.doc], dtype[d.doc], app, ent, cite) for d in docs]
    issued = {d.doc: d.issued for d in documents}

    holdings: list[dict] = []
    roles: dict[tuple[str, str, str], list[_Claim]] = {}
    context: dict[str, list[_Claim]] = defaultdict(list)
    corporate_object: list[str] = []
    subject_of: dict[str, str] = {}
    for c in verified:
        subj = ent(c.subject)
        if subj is not None:
            _add(subj.evidence, c)
            subject_of[c.eid] = subj.eid
        # only a relationship has a target, and only a bound one (checked above)
        target = (bound(c, subj, c.obj) or None) if c.obj and c.kind in _RELATIONS else None
        if target is not None:
            _add(target.evidence, c)
        if c.kind == "holding" and subj is not None:
            via = lambda n, c=c, s=subj: bound(c, s, n)  # noqa: E731
            _holding(c, subj, target or app, holdings, issued, via)
        elif c.kind == "transfer":
            for k in [subj, target, *(ents[id(x)] for x in parties.mentioned_in(_text(c.value)))]:
                if k is not None:
                    _add(context[k.eid], c)
            # a pending acquisition is recorded against the applicant only if its excerpt names
            # no company other than known parties (its counterparty is one by nature)
            stray = not c.obj and foreign(c, list(ents.values()))
            pending = _PENDING.search(fold(f"{c.attr} {_text(c.value)}"))
            if subj is not None and pending and not stray:
                pct = _pct(_text(c.value), "pct")
                holdings.append(
                    {
                        "owner": subj.eid,
                        "owned": (target or app).eid,
                        "pct": pct,
                        "shares": None,
                        "doc": c.doc,
                        "status": "pending",
                        "as_of": issued.get(c.doc),
                        "claims": [c],
                    }
                )
        elif c.kind in ("role", "signature") and subj is not None:
            _role(c, subj, target, app, ent, roles)
        elif c.kind == "identity" and subj is not None:
            _identity(c, subj, cite)
        elif c.kind == "address" and subj is not None:
            _address(c, subj, dtype[c.doc], cite)
        elif c.kind == "activity" and (subj is None or subj is app):
            _add(corporate_object, _text(c.value))
            cite(c, "corporate_object")
    _infer_jurisdiction(app, cite)

    hs = _finish_holdings(holdings, dtype, issued)
    for h in hs:
        for c in h["claims"]:
            cite(c, f"holding[{h['id']}].pct" if h["pct"] is not None else f"holding[{h['id']}]")
    role_list = []
    for i, ((holder, of, kind), cs) in enumerate(roles.items(), 1):
        role_list.append(
            Role(role_id=f"r-{i}", holder=holder, of=of, role=kind, evidence=_refs(cs))
        )
        for c in cs:
            cite(c, f"role[r-{i}]")
    agents = [r for r in role_list if r.role in ("registered_agent", "nominee_director")]
    by_eid = {e.eid: e for e in ents.values()}
    owners = {app.eid} | {h["owner"] for h in hs}
    entities = [
        _entity(e, agents, by_eid, _regions(e, cite, subject_of) if e.eid in owners else [])
        for e in sorted(ents.values(), key=lambda e: _first(e))
    ]
    coverage = {d.doc: d.coverage for d in docs}
    conflicts = _conflicts(hs, by_doc, coverage, parties, ents, context, app, materiality, unbound)
    operations = _operations(app, entities, documents, by_doc, dtype, docs, ent, parties)
    evidence = [
        EvidenceItem(
            evidence_id=c.eid,
            doc=c.doc,
            document_name=next(d.document_name for d in docs if d.doc == c.doc),
            page=c.raw["page"],
            bbox=tuple(c.raw["bbox"]),
            excerpt=c.raw["excerpt"],
            verified=c.verified,
            field=fields.get(c.eid),
        )
        for c in claims
    ]
    body = dict(
        case=case,
        run=run,
        as_of=as_of,
        applicant=app.eid,
        entities=entities,
        holdings=[
            Holding(
                holding_id=h["id"],
                owner=h["owner"],
                owned=h["owned"],
                pct=h["pct"],
                shares=h["shares"],
                as_of=h["as_of"],
                status=h["status"],
                evidence=_refs(h["claims"]),
            )
            for h in hs
        ],
        roles=role_list,
        documents=documents,
        conflicts=conflicts,
        operations=operations,
        corporate_object=corporate_object,
        evidence=evidence,
        unverified=[c.ref for c in claims if not c.verified],
        verification=[_verification(d) for d in docs],
    )
    draft = FactSet(fact_set_hash="", **body)
    digest = sha256_json(draft.model_dump(mode="json", exclude={"fact_set_hash"}))
    return draft.model_copy(update={"fact_set_hash": digest})


def fact_set_hash(fs: FactSet) -> str:
    return sha256_json(fs.model_dump(mode="json", exclude={"fact_set_hash"}))


def _claims(docs: list[DocInput]) -> list[_Claim]:
    seen: dict[str, int] = defaultdict(int)
    for d in docs:
        for c in d.claims:
            seen[c["claim_id"]] += 1
    out, order = [], 0
    for d in docs:
        for c in d.claims:
            cid = c["claim_id"]
            eid = cid if seen[cid] == 1 else f"{d.doc}:{cid}"  # ids are per document
            typ = akey(c.get("type"))
            cl = _Claim(
                raw=c,
                doc=d.doc,
                eid=eid,
                order=order,
                verified=bool(d.verdicts.get(cid, False)),
                typ=typ,
                attr=akey(c.get("attribute")) or typ,
            )
            cl.kind = _classify(cl)
            out.append(cl)
            order += 1
    return out


_IDENT_KEYS = {"dob": _DOB, **{k: v for k, v in _IDS.items() if k != "reg_no"}}


def _idents(verified: list[_Claim]) -> dict[str, dict[str, set[str]]]:
    """Per spelling, the identifiers verified claims give it: what keeps two parties apart."""
    out: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for c in verified:
        for key, attrs in _IDENT_KEYS.items():
            if c.kind == "identity" and c.attr in attrs:
                v = _date_of(c) if key == "dob" else re.sub(r"[^0-9a-z]", "", fold(_text(c.value)))
                if v:
                    out[c.subject][key].add(v)
    return out


def _parties(verified: list[_Claim]) -> _Parties:
    parties = _Parties(_idents(verified))
    subjects = defaultdict(list)
    for c in verified:
        subjects[c.doc].append(c.subject)
    for c in verified:
        if c.kind in ("doc", "integrity"):
            continue
        if c.subject and not _is_doc_title(c.subject):
            parties.add(c.subject, c.doc, c.order)
    for c in verified:
        if c.kind in ("doc", "integrity") or not c.obj or _is_doc_title(c.obj):
            continue
        # an object is a party only where verification binds it (see `bound`)
        if contains(c.raw.get("excerpt", ""), c.obj) or any(
            _same_party(c.obj, x) for x in subjects[c.doc] if x
        ):
            parties.add(c.obj, c.doc, c.order)
    for c in verified:
        if c.kind in ("doc", "integrity"):
            continue
        if c.kind == "identity" and c.attr in _NAME and isinstance(c.value, str):
            parties.add(c.value, c.doc, c.order)
        if c.kind == "role" and _holder_in_value(c):
            parties.add(str(c.value).strip(), c.doc, c.order)
    return parties


def _holder_in_value(c: _Claim) -> bool:
    """`registered_agent: Apex Corporate Services Ltd`: the attribute names the role and the
    value names who holds it."""
    v = c.value
    return (
        isinstance(v, str)
        and _role_kind(c.attr) != "other"
        and _role_kind(v) == "other"
        and _looks_like_name(v.strip())
    )


_IDENTITY_ATTRS = _INCORPORATED | _NAME | _JURISDICTION | {a for v in _IDS.values() for a in v}


def _forms(name: str) -> list[tuple[str, ...]]:
    """A party's spellings as token sequences: the full name, and (for an organisation of two
    or more words) its name without the legal form."""
    full = tuple(merged_tokens(name))
    key = tuple(org_key(name).split()) if is_org(name) else ()
    return [f for f in {full, key} if f and (f == full or len(f) > 1)]


def _spans(toks: list[str], ents) -> list[tuple[int, int, str]]:
    """Where known parties are named in a token list: every occurrence of every spelling,
    longest first, never overlapping. A one-word spelling counts only as a whole name."""
    found = []
    for e in ents:
        for n in e.names:
            for form in _forms(n):
                k = len(form)
                for i in range(len(toks) - k + 1):
                    if tuple(toks[i : i + k]) == form:
                        found.append((i, i + k, e.eid))
    kept: list[tuple[int, int, str]] = []
    for s0, s1, eid in sorted(found, key=lambda f: (-(f[1] - f[0]), f[0], f[2])):
        if all(s1 <= a or s0 >= b for a, b, _ in kept):
            kept.append((s0, s1, eid))
    return sorted(kept)


_CONNECTORS = {"of", "de", "del", "la", "el", "los", "las", "for", "by", "to", "a", "en", "para",
               "por", "y", "and", "the", "as", "at", "in", "from", "con", "with", "is", "es",
               "al", "su", "sus", "its", "on", "behalf", "nombre", "que", "le", "les", "des",
               "du", "et", "e", "o", "or", "u"}  # fmt: skip


def _occurrences(excerpt: str, name: str, *, complete: bool) -> list[bool]:
    """Positive support needs a complete form; a veto can also use the raw company stem.

    Locate every normalized start without losing the raw context. In particular, dropping
    `the` for identity matching must not hide `GRUPO` before `THE ANDES ...` (R7).
    """
    forms = _forms(name)
    if not complete and is_org(name):
        # Retain the article while allowing omitted legal suffixes. A bare first word
        # (e.g. the unrelated `the` in `Under the resolution`) is not a company span.
        forms = [*forms, tuple(strip_suffixes(merged_tokens(name)))]
    if not forms:
        return []
    starts = {form[0] for form in forms}
    raw = "".join(ch for ch in unicodedata.normalize("NFKD", excerpt)
                  if not unicodedata.combining(ch))  # fmt: skip
    out = []
    pattern = "|".join(re.escape(t) for t in sorted(starts))
    for m in re.finditer(rf"\b(?:{pattern})\b", raw, re.IGNORECASE):
        tail = merged_tokens(raw[m.start() :])
        if not any(tuple(tail[: len(form)]) == form for form in forms):
            continue
        before = raw[: m.start()]
        # org_key removes a leading article; restore it for the boundary check, rather
        # than treating it as a connector that could free a tail inside a foreign name.
        if m[0].lower() != "the":
            before = re.sub(r"\bthe\s+$", "", before, flags=re.IGNORECASE)
        prev = re.search(r"(\S+)\s+$", before)
        word = prev[1] if prev else ""
        out.append(word.isalpha() and word.lower() not in _CONNECTORS)
    return out


def _free(excerpt: str, name: str) -> bool:
    """At least one occurrence of the name stands on its own ("… of Acme Ltd" in "Parent Acme
    Ltd owns 25% of Acme Ltd")."""
    return any(not glued for glued in _occurrences(excerpt, name, complete=True))


def _glued(excerpt: str, name: str) -> bool:
    """Is an occurrence of `name` in the excerpt the tail of a longer name ("GRUPO Andes …",
    "grupo andes …", "Parent Acme Ltd")? Judged on the raw text: the word right before it
    carries no punctuation and is not a connector, whatever its case (Codex r5 #1). Labels end
    in punctuation ("Registered agent: Apex"), so they do not glue."""
    return any(_occurrences(excerpt, name, complete=False))


def _principals(verified: list[_Claim], ent) -> dict[str, str]:
    """Per document, the one organisation whose identity (name, registration, tax id,
    incorporation, jurisdiction) the document verifiably states: the party a register, a
    charter or a certificate is *about*. None when there are several."""
    seen: dict[str, set[str]] = defaultdict(set)
    for c in verified:
        e = ent(c.subject)
        if c.kind == "identity" and c.attr in _IDENTITY_ATTRS and e and e.kind != "person":
            seen[c.doc].add(e.eid)
    return {doc: next(iter(es)) for doc, es in seen.items() if len(es) == 1}


def _applicant(parties: _Parties) -> _Cluster | None:
    orgs = [k for k in parties.clusters if any(is_org(n) for n in k.names)]
    return min(orgs, key=lambda k: (-len(k.docs), k.first), default=None)


def _first(e: _E) -> int:
    return min((c.order for c in e.evidence), default=10**9)


def _date_of(c: _Claim) -> str | None:
    v = c.value
    if isinstance(v, dict):
        v = next((x for x in v.values() if parse_date(str(x))), "")
    return parse_date(str(v))


def _document(d: DocInput, cs: list[_Claim], dtype: str, app: _E, ent, cite) -> Document:
    ok = [c for c in cs if c.verified]
    issued = None
    for attr in _DOC_DATE:
        dates = {x for c in ok if c.attr == attr for x in [_date_of(c)] if x}
        if dates:
            issued = dates.pop() if len(dates) == 1 else None  # two dates: unknown, not a choice
            for c in ok:
                if c.attr == attr and _date_of(c):
                    cite(c, f"document[{d.doc}].issued")
            break
    used = [c for c in ok if c.attr in _DOC_DATE and _date_of(c)]
    is_extract = any(
        k in akey(d.document_name) for k in ("existencia", "extract", "extracto", "standing")
    ) or any(c.attr in _VALIDITY for c in cs)
    if issued is None and not used and dtype == "incorporation" and not is_extract:
        inc = [c for c in ok if c.attr in _INCORPORATED and _date_of(c) and ent(c.subject) is app]
        dates = {_date_of(c) for c in inc}
        if len(dates) == 1:
            issued = dates.pop()
            used = inc
            for c in inc:
                cite(c, f"document[{d.doc}].issued")
    signs = [c for c in ok if c.attr in _SIGN_DATE and _date_of(c)]
    sign_dates = {_date_of(c) for c in signs}
    signed_on = sign_dates.pop() if len(sign_dates) == 1 else None
    signatures = [c for c in ok if c.kind == "signature" and _affirms(c.value)]
    unsigned = [c for c in ok if c.kind == "signature" and not _affirms(c.value)]
    validity = [c for c in ok if c.attr in _VALIDITY]
    days = {x for c in validity for x in [_validity_days(_text(c.value))] if x is not None}
    for c in signs:
        cite(c, f"document[{d.doc}].signed_on")
    for c in validity:
        cite(c, f"document[{d.doc}].stated_validity_days")
    return Document(
        doc=d.doc,
        document_name=d.document_name,
        doc_type=dtype,
        issued=issued,
        # an explicit "not signed" is a fact too; both at once is unknown, never signed
        signed=None
        if (signs or signatures) and unsigned
        else True
        if (signs or signatures)
        else False
        if unsigned
        else None,
        signed_on=signed_on,
        stated_validity_days=days.pop() if len(days) == 1 else None,
        evidence=_refs(used + signs + signatures + unsigned + validity),
    )


_NEGATIVE = re.compile(r"^(?:no|none|ninguno|ninguna|sin\b.*|not\b.*|n/?a|false|0+|"
                       r"no\s.*|unsigned|nil|-)$")  # fmt: skip


def _affirms(value: Any) -> bool:
    """A value that states something rather than its absence: `False`, 0, "none", "sin firma",
    "no employees", "not signed" do not."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value > 0
    t = " ".join(fold(_text(value)).split()).strip(" .,;:")
    return bool(t) and not _NEGATIVE.match(t)


_EMPLOYEES = {"employees", "employee_count", "number_of_employees", "headcount", "staff",
              "empleados", "numero_de_empleados"}  # fmt: skip


_DENIAL = re.compile(
    r"\b(?:no|none|sin|not|without|nil|ningun[oa]?|nadie|closed|cerrad[oa]|inactive|"
    r"inactiv[oa]|discontinued|defunct|dad[oa] de baja|n/?a)\b"
)
_COUNT = re.compile(
    r"(\d{1,3}(?:[.,]\d{3})+|\d+(?:[.,]\d+)?)\s*(?:employees?|empleados?|staff|trabajadores|"
    r"colaboradores|personas|people|workers|headcount)\b"
)
_WEB = re.compile(r"(?:https?://)?(?:www\.)?[a-z0-9-]+(?:\.[a-z0-9-]+)*\.[a-z]{2,}(?::\d+)?")
_BARE = re.compile(r"\d{1,3}(?:[.,]\d{3})+|\d+(?:[.,]\d+)?")
_QUOTE_PAIRS = {"\"": "\"", "'": "'", "“": "”", "‘": "’", "«": "»", "‹": "›",
                "„": "“", "‚": "‘", "〝": "〞"}  # fmt: skip


def _web_spans(text: str) -> list[tuple[int, int]]:
    """URLs in prose, including ports, queries and fragments, bounded by context (RFC3986 C).

    Unmatched closing brackets and quotes delimit prose; balanced internal parentheses are
    URI data. Thus `(https://host.co?q=(no)):closed` retains only the outside denial.
    """
    spans = []
    for m in _WEB.finditer(text):
        if spans and m.start() < spans[-1][1]:
            continue
        end = m.end()
        quote = _QUOTE_PAIRS.get(text[: m.start()].rstrip()[-1:])
        if end < len(text) and text[end] in "/?#":
            stack = []
            pairs = {"(": ")", "[": "]", "{": "}"}
            while end < len(text):
                ch = text[end]
                if ch.isspace() or ch in '<>"' or ch == quote:
                    break
                if ch in pairs:
                    stack.append(pairs[ch])
                elif ch in ")]}":
                    if not stack or stack[-1] != ch:
                        break
                    stack.pop()
                end += 1
        spans.append((m.start(), end))
    return spans


def _count(raw: str) -> Decimal:
    grouped = re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", raw)
    return Decimal(re.sub(r"[.,]", "", raw) if grouped else raw.replace(",", "."))


def _ops_polarity(c: _Claim) -> bool | None:
    """Operational indicator (v) by attribute. A denial anywhere in the value ("no", "sin",
    "closed") is read first: it is a none, unless the value also states a positive count,
    which makes it unclear. An employee count is a number attached to "employees" (or the
    whole value); a year in "as of 2026-10-25" is not a count. A website must look like one
    and not be denied. Anything else is None: the indicator stays unknown."""
    if isinstance(c.value, bool):
        return c.value
    if isinstance(c.value, int | float):
        return c.value > 0
    text = " ".join(fold(_text(c.value)).split()).strip(" .,;:")
    # a denial is prose: URL and domain components ("andesindustrial.no") are not words
    web = _web_spans(text)
    prose = text
    for start, end in reversed(web):
        prose = prose[:start] + " " + prose[end:]
    denied = bool(_DENIAL.search(prose))
    if c.attr in _EMPLOYEES:
        m = _COUNT.search(text)
        raw = m[1] if m else (text if _BARE.fullmatch(text) else None)
        if raw is not None:
            n = _count(raw)
            return None if (denied and n > 0) else n > 0
        return False if denied else None
    if denied:
        return False
    if web:
        return True
    return None


def _validity_days(text: str) -> int | None:
    """An issuer's stated validity in days (spec): "válido por 30 días", "valid for 2 weeks".
    Months and years are not a fixed number of days, so they are left unknown."""
    t = fold(text)
    if m := re.search(r"(\d+)\s*(?:dias?|days?|d\b)", t):
        return int(m[1])
    if m := re.search(r"(\d+)\s*(?:semanas?|weeks?)", t):
        return 7 * int(m[1])
    return None


def _holding(c: _Claim, owner: _E, owned: _E, holdings: list[dict], issued: dict, bound) -> bool:
    """Record one holding; False when the claim names an intermediary nothing binds."""
    v = c.value
    if isinstance(v, dict) and any(k in v for k in _VIA):
        via = bound(str(next(v[k] for k in _VIA if k in v)))
        if via is False or via is None:
            return False
        if via is not owner:
            owned = via  # an indirect holding is recorded through its named intermediary
    # only the flag the witness checks (`struck-through` must be among the re-read's struck
    # values); a claim's document_date is the reader's, so the date is the document's own
    # exactly the flag the witness enforces (Codex r2 #2: a respelling would bypass it)
    status = "struck" if "struck-through" in (c.raw.get("flags") or []) else "current"
    as_of = issued.get(c.doc)
    holdings.append(
        {
            "owner": owner.eid,
            "owned": owned.eid,
            "pct": _pct(v, c.attr),
            "shares": _shares(v, c.attr),
            "doc": c.doc,
            "status": status,
            "as_of": as_of,
            "claims": [c],
        }
    )
    return True


def _finish_holdings(raw: list[dict], dtype: dict, issued: dict) -> list[dict]:
    """Merge one owner's statements within a document; mark superseded deeds historical."""
    merged: list[dict] = []
    for h in raw:
        same = next(
            (
                m
                for m in merged
                if (m["owner"], m["owned"], m["doc"], m["status"])
                == (h["owner"], h["owned"], h["doc"], h["status"])
                and (m["pct"] is None or h["pct"] is None or m["pct"] == h["pct"])
                and (m["shares"] is None or h["shares"] is None or m["shares"] == h["shares"])
            ),
            None,
        )
        if same is None:
            merged.append({**h, "claims": list(h["claims"])})
            continue
        same["pct"] = same["pct"] if same["pct"] is not None else h["pct"]
        same["shares"] = same["shares"] if same["shares"] is not None else h["shares"]
        same["claims"] += h["claims"]
    for h in merged:
        if h["status"] == "current" and dtype.get(h["doc"]) == "incorporation":
            mine = issued.get(h["doc"])
            later = [
                o
                for o in merged
                if o["owned"] == h["owned"]
                and o["doc"] != h["doc"]
                and o["status"] != "pending"
                and issued.get(o["doc"])
                and mine
                and issued[o["doc"]] > mine
            ]
            if later:
                h["status"] = "historical"
    for i, h in enumerate(merged, 1):
        h["id"] = f"h-{i}"
    return merged


def _role(c: _Claim, subj: _E, target: _E | None, app: _E, ent, roles: dict) -> None:
    if c.kind == "signature":
        holder, of, kind = subj, target or app, "signatory"
    else:
        kind = _role_kind(_text(c.value))
        if kind == "other":
            kind = _role_kind(c.attr)
        holder, of = subj, target or app
        if _holder_in_value(c):
            named = ent(str(c.value).strip())
            if named is not None and named is not subj:
                holder, of = named, target or subj
    if holder is of:
        return
    _add(holder.evidence, c)
    roles.setdefault((holder.eid, of.eid, kind), [])
    _add(roles[(holder.eid, of.eid, kind)], c)


def _identity(c: _Claim, e: _E, cite) -> None:
    a, v = c.attr, _text(c.value)
    if a in _DOB and (d := _date_of(c)):
        _add(e.dob, d)
        cite(c, f"entity[{e.eid}].dob")
    elif a in _NATIONALITY:
        for x in countries(v):
            _add(e.nationality, x)
        cite(c, f"entity[{e.eid}].nationality")
    elif a in _JURISDICTION:
        for x in countries(v)[:1]:
            _add(e.jurisdiction, x)
        cite(c, f"entity[{e.eid}].jurisdiction")
    elif a in _INCORPORATED and (d := _date_of(c)):
        _add(e.incorporated, d)
        cite(c, f"entity[{e.eid}].incorporated")
    elif a in _NAME:
        if isinstance(c.value, str) and c.value.strip():
            _add(e.names, c.value.strip())
        cite(c, f"entity[{e.eid}].names")
    else:
        for key, attrs in _IDS.items():
            if a in attrs and v.strip():
                _add(e.ids.setdefault(key, []), v.strip())
                cite(c, f"entity[{e.eid}].identifiers.{key}")


def _address(c: _Claim, e: _E, dtype: str, cite) -> None:
    kind = next((k for k, s in _ADDR.items() if c.attr in s), "unknown")
    if kind == "unknown":
        if dtype == "lease":
            kind = "operating"
        elif dtype == "tax_certificate" and e.kind != "person":
            kind = "fiscal"
        elif dtype == "incorporation" and e.kind != "person" and c.attr == "domicilio":
            kind = "registered"
        elif e.kind == "person":
            kind = "residential"
    text = _text(c.value).strip()
    if not text:
        return
    for i, (k, t, cs) in enumerate(e.addresses):
        if k == kind and same_address(t, text):
            e.addresses[i] = (k, t, [*cs, c])
            break
    else:
        e.addresses.append((kind, text, [c]))
    cite(c, f"entity[{e.eid}].addresses.{kind}")


def _infer_jurisdiction(app: _E, cite) -> None:
    """A company with no stated jurisdiction takes the country its registered (else fiscal)
    address names: "Ciudad de México" → MX. Cited to the address."""
    if app.jurisdiction or app.kind == "person":
        return
    for kind in ("registered", "fiscal"):
        for k, text, _cs in app.addresses:
            found = countries(text)
            if k == kind and found:
                app.jurisdiction.append(found[0])
                return


def _regions(e: _E, cite, subject_of: dict[str, str]) -> list[str]:
    """spec: the sub-national regions and free zones the party's addresses and incorporation
    statements name, verbatim (embargoed UA regions, AE free zones). Empty when none is stated."""
    out: list[str] = []
    for c in sorted(e.evidence, key=lambda c: c.order):
        if subject_of.get(c.eid) != e.eid:
            continue  # another party's statement that merely names this one (Codex r2 #7)
        statement = c.kind == "address" or (c.kind == "identity" and c.attr in _JURISDICTION)
        found = regions(_text(c.value)) if statement else []
        for r in found:
            _add(out, r)
        if found:
            cite(c, f"entity[{e.eid}].regions")
    return out


def _entity(e: _E, agents: list[Role], by_eid: dict[str, _E], named: list[str]) -> Entity:
    addresses = []
    for kind, text, cs in e.addresses:
        addresses.append(
            Address(
                kind=kind,
                text=text,
                evidence=_refs(cs),
                mass_registration=_mass_registration(e, text, cs, agents, by_eid),
            )
        )
    return Entity(
        entity_id=e.eid,
        kind=e.kind,
        names=e.names,
        identifiers=e.ids,
        nationality=e.nationality,
        dob=e.dob,
        jurisdiction=e.jurisdiction,
        incorporated=e.incorporated,
        addresses=addresses,
        evidence=_refs(e.evidence),
        regions=named,
    )


def _mass_registration(
    e: _E, text: str, cs: list[_Claim], agents: list[Role], by_eid: dict[str, _E]
) -> TriFact:
    """Shell indicator (i): present when the address is the entity's own registered agent's (every
    significant word of the agent's name is in it) or is given in care of another party; absent
    when the packet names no agent for the entity and the address is its own; else unknown."""
    t = f" {base_norm(text)} "
    mine = [r for r in agents if r.of == e.eid]
    for r in mine:
        agent = by_eid[r.holder]
        for n in agent.names:
            toks = significant_tokens(n)
            if toks and all(f" {x} " in t for x in toks):
                return TriFact(
                    state="present",
                    evidence=_refs(cs) + r.evidence,
                    note=f"the address is the registered agent's ({n})",
                )
    if re.search(r"\bc/o\b|care of|at the offices of|en las oficinas de", fold(text)):
        return TriFact(
            state="present",
            evidence=_refs(cs),
            note="the address is given in care of another party",
        )
    if not mine:
        return TriFact(
            state="absent",
            evidence=_refs(cs),
            note="the packet names no registered or nominee agent for this party, "
            "and the address is not given in care of another",
        )
    return TriFact(
        state="unknown",
        evidence=_refs(cs),
        note="an agent is named, but the address is not shown to be the agent's",
    )


# --- conflicts (spec) ----------------------------------------------------------------------


def _fmt(d: Decimal) -> str:
    return format(d, "f")


def _cv(value: str, cs: list[_Claim]) -> ConflictValue:
    return ConflictValue(value=value, evidence=_refs(cs))


def _ordered(values: dict[str, list[_Claim]]) -> list[ConflictValue]:
    return [
        _cv(v, cs) for v, cs in sorted(values.items(), key=lambda kv: min(c.order for c in kv[1]))
    ]


def _conflicts(
    hs, by_doc, coverage, parties, ents, context, app, materiality: Decimal, unbound: set[str]
) -> list[Conflict]:
    out: list[tuple[str, str, list[ConflictValue]]] = []
    stated = [h for h in hs if h["status"] != "pending"]
    pairs = sorted(
        {(h["owner"], h["owned"]) for h in stated},
        key=lambda p: min(
            c.order for h in stated if (h["owner"], h["owned"]) == p for c in h["claims"]
        ),
    )
    # one owner's stake stated differently (more than `materiality` points apart)
    for owner, owned in pairs:
        values: dict[str, list[_Claim]] = defaultdict(list)
        for h in stated:
            if (h["owner"], h["owned"]) == (owner, owned) and h["pct"] is not None:
                values[_fmt(h["pct"])] += h["claims"]
        nums = [Decimal(v) for v in values]
        if len(nums) > 1 and max(nums) - min(nums) > materiality:
            attr = "ownership_pct" if owned == app.eid else f"ownership_pct:{owned}"
            out.append((owner, attr, _ordered(values)))
    # an owner present in one shareholder listing and absent from another
    for owner, owned in pairs:
        present = [h for h in stated if (h["owner"], h["owned"]) == (owner, owned)]
        absent: list[_Claim] = []
        for doc in sorted({h["doc"] for h in stated if h["owned"] == owned}):
            if doc in {h["doc"] for h in present}:
                continue
            listed = [h for h in stated if h["owned"] == owned and h["doc"] == doc]
            if _absence_established(owner, listed, by_doc[doc], coverage[doc], parties, unbound):
                absent += [c for h in listed for c in h["claims"]]
        if present and absent:
            values = defaultdict(list)
            for h in present:
                values[_fmt(h["pct"]) if h["pct"] is not None else "present"] += h["claims"]
            values["absent"] += absent
            attr = "owner_presence" if owned == app.eid else f"owner_presence:{owned}"
            out.append((owner, attr, _ordered(values)))
    # registered addresses and registration / tax numbers of the same party
    for e in sorted(ents.values(), key=_first):
        regs: dict[str, list[_Claim]] = {}
        for kind, text, cs in e.addresses:
            if kind == "registered":
                regs[text] = cs
        if len(regs) > 1:
            out.append((e.eid, "registered_address", _ordered(regs)))
        for key in ("reg_no", "tax_id"):
            vals: dict[str, list[_Claim]] = defaultdict(list)
            for c in e.evidence:
                if c.kind == "identity" and c.attr in _IDS[key]:
                    vals[re.sub(r"[^0-9a-z]", "", base_norm(_text(c.value)))].append(c)
            vals.pop("", None)
            if len(vals) > 1:
                shown = {_text(cs[0].value).strip(): cs for cs in vals.values()}
                out.append((e.eid, key, _ordered(shown)))
    conflicts = []
    for i, (subject, attr, values) in enumerate(out, 1):
        conflicts.append(
            Conflict(
                conflict_id=f"x-{i}",
                subject=subject,
                attribute=attr,
                values=values,
                context=_refs(context.get(subject, [])),
            )
        )
    return conflicts


def _absence_established(
    owner: str,
    listed: list[dict],
    doc_claims: list[_Claim],
    coverage: list[dict],
    parties: _Parties,
    unbound: set[str],
) -> bool:
    """Is `owner` shown to be absent from this document's shareholder listing? Only when the
    listing is complete as far as verification can tell: no failed ownership claim that could
    be the owner's (a claim whose target nothing verifies counts as failed), and every uncovered
    ownership fact is about a holder the listing has."""
    holders = {h["owner"] for h in listed}

    def who(name: str) -> str | None:
        k = parties.resolve(str(name)) if name else None
        return k.eid if k else None

    for c in doc_claims:
        maybe_owner = who(c.subject) in (owner, None) or (c.obj and who(c.obj) == owner)
        failed = not c.verified or c.eid in unbound
        if failed and c.kind in ("holding", "transfer") and maybe_owner:
            return False
    for f in coverage:
        if f.get("status") != "uncovered" or f.get("kind") not in ("ownership", "amendment"):
            continue
        eids = {who(str(f.get(k) or "")) for k in ("subject", "value")} - {None}
        if owner in eids or not (eids & holders):
            return False
    return True


# --- operations and verification --------------------------------------------------------------


def _operations(
    app: _E,
    entities: list[Entity],
    documents: list[Document],
    by_doc,
    dtype,
    docs: list[DocInput],
    ent,
    parties: _Parties,
) -> Operations:
    """Shell indicators (ii)-(v) for the applicant only: a claim counts when its subject is the
    applicant; a gap the witness left (an uncovered fact, a failed claim) that could be the
    applicant's makes the indicator unknown rather than absent."""
    me = next(e for e in entities if e.entity_id == app.eid)
    claims = [c for cs in by_doc.values() for c in cs]

    def mine(c: _Claim) -> bool:
        return ent(c.subject) is app

    def maybe_mine(name: str) -> bool:
        k = parties.resolve(str(name)) if name and not _is_doc_title(str(name)) else None
        return k is None or k.eid == app.eid

    def gaps(kinds: tuple[str, ...]) -> list[dict]:
        return [f for d in docs for f in d.coverage
                if f.get("status") == "uncovered" and f.get("kind") in kinds
                and akey(f.get("attribute")) not in _DOC_OTHER
                and maybe_mine(f.get("subject"))]  # fmt: skip

    return Operations(
        distinct_operating_address=_op_address(me, claims, gaps(("address",)), maybe_mine),
        bank_reference_or_financials=_op_bank(
            documents, by_doc, claims, mine, gaps(("financial_reference",))
        ),
        employees_or_website=_op_people(claims, mine, maybe_mine, gaps(("operations",))),
    )


def _op_address(me: Entity, claims: list[_Claim], gaps: list[dict], maybe_mine) -> TriFact:
    reg = [a for a in me.addresses if a.kind == "registered"]
    ops = [a for a in me.addresses if a.kind == "operating"]
    every = [r for a in me.addresses for r in a.evidence]
    for a in ops:
        if not any(same_address(a.text, r.text) for r in reg):
            note = (
                f"operating address {a.text!r} differs from the registered address"
                if reg
                else f"operating address {a.text!r}; no registered address is stated"
            )
            return TriFact(
                state="present",
                evidence=a.evidence + [x for r in reg for x in r.evidence],
                note=note,
            )
    failed = [
        c
        for c in claims
        if not c.verified
        and c.kind == "address"
        and next((k for k, s in _ADDR.items() if c.attr in s), "unknown")
        in ("operating", "unknown")
        and maybe_mine(c.subject)
    ]
    if failed or gaps:
        return TriFact(
            state="unknown",
            evidence=[c.ref for c in failed],
            note="an address of the applicant was not verified (a failed claim or an "
            "uncovered address fact)",
        )
    known = [a for a in me.addresses if a.kind in ("registered", "fiscal", "operating")]
    loose = [
        a
        for a in me.addresses
        if a.kind == "unknown" and not any(same_address(a.text, k.text) for k in known)
    ]
    if loose:
        return TriFact(
            state="unknown",
            evidence=[x for a in loose for x in a.evidence],
            note="an address of unstated purpose differs from the registered one",
        )
    if me.addresses:
        kinds = sorted({a.kind for a in me.addresses})
        return TriFact(
            state="absent",
            evidence=every,
            note="no separate operating address beyond the registered one: the packet "
            f"gives only {', '.join(kinds)} address(es)",
        )
    return TriFact(state="unknown", note="no verified address of the applicant")


def _op_bank(documents: list[Document], by_doc, claims: list[_Claim], mine, gaps) -> TriFact:
    """Shell indicator (iv). Present only on positive evidence about the applicant: a verified
    claim in a bank reference or financial statement, or an affirmative bank claim elsewhere. A
    stated negative ("bank_reference: none") is absence with evidence; both at once is unknown
    (Codex r2 #3)."""
    kinds = ("bank_reference", "financial_statement")
    bank_docs = [d for d in documents if d.doc_type in kinds]
    in_docs = [c for d in bank_docs for c in by_doc[d.doc] if c.verified and mine(c)]
    banks = [c for c in claims if c.verified and c.kind == "bank" and mine(c)]
    yes = [c for c in in_docs if _affirms(c.value)]
    yes += [c for c in banks if _affirms(c.value) and c not in yes]
    no = [c for c in in_docs + banks if not _affirms(c.value)]
    if yes and no:
        return TriFact(
            state="unknown",
            evidence=_refs(yes + no)[:8],
            note="the packet evidences a bank reference and also states there is none",
        )
    if yes:
        return TriFact(
            state="present",
            evidence=_refs(yes)[:6],
            note="a bank reference or financial statement about the applicant is in the packet",
        )
    if bank_docs or gaps or [c for c in claims if c.kind == "bank" and not c.verified]:
        return TriFact(
            state="unknown",
            evidence=[c.ref for c in claims if c.kind == "bank"],
            note="a bank reference is in the packet but nothing verified ties it to the applicant",
        )
    if no:
        return TriFact(
            state="absent",
            evidence=_refs(no),
            note="the packet states the applicant has no bank reference and no financials",
        )
    if any(d.doc_type == "other" for d in documents):
        return TriFact(
            state="unknown", note="a document of unrecognised type could be a bank reference"
        )
    return TriFact(
        state="absent",
        note="the packet has neither a bank reference nor a financial "
        "statement: " + ", ".join(f"{d.doc} {d.doc_type}" for d in documents),
    )


def _op_people(claims: list[_Claim], mine, maybe_mine, gaps: list[dict]) -> TriFact:
    """Shell indicator (v): employees *or* a website. Each is read on its own; a statement that
    and has not (within one of them) is unknown; either one evidenced is presence."""
    stated = [c for c in claims if c.kind == "ops" and c.verified and mine(c)]
    by = {"employees": [c for c in stated if c.attr in _EMPLOYEES],
          "website": [c for c in stated if c.attr not in _EMPLOYEES]}  # fmt: skip
    yes = [c for cs in by.values() for c in cs if _ops_polarity(c) is True]
    no = [c for cs in by.values() for c in cs if _ops_polarity(c) is False]
    unclear = [c for c in stated if _ops_polarity(c) is None]
    torn = [cs for cs in by.values() if {_ops_polarity(c) for c in cs} >= {True, False}]
    if torn:
        return TriFact(
            state="unknown",
            evidence=_refs([c for cs in torn for c in cs]),
            note="the packet states employees (or a website) and also none",
        )
    if yes:
        return TriFact(
            state="present",
            evidence=_refs(yes),
            note="employees or a website are stated for the applicant",
        )
    failed = [c for c in claims if c.kind == "ops" and not c.verified and maybe_mine(c.subject)]
    if failed or gaps or unclear:
        return TriFact(
            state="unknown",
            evidence=[c.ref for c in failed] + _refs(unclear),
            note="an employees or website statement was not verified or does not say",
        )
    if no:
        return TriFact(
            state="absent",
            evidence=_refs(no),
            note="the packet states the applicant has no employees or website",
        )
    return TriFact(state="absent", note="no document in the packet states employees or a website")


def _verification(d: DocInput) -> DocVerification:
    uncovered = [
        UncoveredFact(
            block_id=str(f.get("block_id", "")),
            kind=str(f.get("kind", "")),
            subject=str(f.get("subject", "")),
            attribute=str(f.get("attribute", "")),
            value=str(f.get("value", "")),
        )
        for f in d.coverage
        if f.get("status") == "uncovered"
    ]
    findings: list[str] = []
    for k in d.findings:
        _add(findings, k)
    if any(f.get("status") == "conflict" for f in d.coverage):
        _add(findings, "coverage_conflict")
    return DocVerification(
        doc=d.doc,
        assurance=d.assurance,
        isolated=d.isolated,
        uncovered=uncovered,
        findings=findings,
    )
