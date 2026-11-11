"""B4 contracts (spec): the shapes the system's lanes build against.

Frozen for the sample. A lane that needs a change files an ask (`.control/asks/`); it does not edit
this file. Every record is written by a `kyb` action to the path the ontology registry names
(`src/kyb/workspace/template/kyb.ontology.toml`), catalogued and audited like every other record.

Flow (E-p, spec; spec):

    verified claims ─► FactSet ─► ScreeningResult ─► RuleResult[] ◄── PolicyModel
       + assurance       │             │                  ▲
       (verification)    └────────────►└─► PredicateInstance ─► PredicateAnswer (code | agent)
                                                              │
                     CorrectionRecord[] ─► JudgeOutcome[] (veto-only) ─► DecisionRecord
                                                                          └ KYBDecision (open-kyb's)

Only *verified* claims become facts. A claim that failed verification is listed in
`FactSet.unverified`; a material fact no claim covered is in `FactSet.verification`. Either way,
a rule that needed it reads as `unknown`, never as absent, and the judges cap the decision.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kyb.contracts.kyb_decision_schema import Citation, KYBDecision

CONTRACTS_VERSION = "b4-v1"

FACTS_REL = "gold/facts/{case}/{run}.json"  # derive_facts (L2)
SCREENING_REL = "gold/screening/{case}/{run}.json"  # screen (L2)
POLICY_REL = "gold/policy/{rules_hash}.json"  # load_policy (L3)
PREDICATES_REL = "gold/predicates/{case}/{run}.json"  # eval_predicates + propose_predicates (L4)
JUDGEMENTS_REL = "gold/judgements/{case}/{run}.json"  # run_judges (L4)
DECISION_REL = "ledger/decisions/{decision_id}.json"  # emit_decision (L4)
CORRECTION_REL = "ledger/corrections/{correction_id}.json"  # record_override / sign_decision (L5)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --- evidence ------------------------------------------------------------------------------------


class EvidenceRef(_Strict):
    """A pointer to one claim; `verified` is the witness's verdict (spec)."""

    claim_id: str
    doc: str
    page: int = Field(ge=1)
    verified: bool


class EvidenceItem(_Strict):
    """A resolved, pinned piece of evidence: what a predicates subagent reads instead of the
    document, and what becomes open-kyb's `Citation` (`citation()`)."""

    evidence_id: str  # the claim id
    doc: str  # "03"
    document_name: str  # "03_libro_registro_socios.pdf"
    page: int = Field(ge=1)
    bbox: tuple[int, int, int, int]  # 300-dpi pixels
    excerpt: str = Field(max_length=600)  # verbatim, verified
    verified: bool
    field: str | None = None  # the fact field it supports, e.g. "holding[h-2].pct"


def citation(case: str, item: EvidenceItem, field: str | None = None) -> Citation:
    """The one mapping from evidence to open-kyb's `Citation`."""
    x0, y0, x1, y1 = item.bbox
    return Citation(document_id=f"{case}/{item.document_name}",
                    locator=f"p.{item.page} bbox[{x0},{y0},{x1},{y1}]@300dpi",
                    excerpt=item.excerpt, field=field or item.field)  # fmt: skip


Tri = Literal["present", "absent", "unknown"]


class TriFact(_Strict):
    """A fact the policy reads as present / absent / unknown. `absent` needs a reason the packet
    shows it is absent (a field left empty, a document that should state it and does not);
    otherwise it is `unknown`."""

    state: Tri
    evidence: list[EvidenceRef] = []
    note: str = ""


# --- facts (L2) -------------------------------------------------------------------------------

EntityKind = Literal["company", "person", "foundation", "trust", "nominee", "agent", "unknown"]
HoldingStatus = Literal["current", "struck", "pending", "historical"]
RoleKind = Literal["director", "president", "secretary", "legal_rep", "nominee_director",
                   "nominee_shareholder", "registered_agent", "signatory", "beneficiary",
                   "council_member", "other"]  # fmt: skip
DocType = Literal["incorporation", "ubo_declaration", "proof_of_address", "tax_certificate",
                  "share_register", "board_minutes", "lease", "bank_reference",
                  "financial_statement", "other"]  # fmt: skip
AddressKind = Literal["registered", "operating", "fiscal", "residential", "unknown"]


class Address(_Strict):
    kind: AddressKind
    text: str  # verbatim
    mass_registration: TriFact  # shell indicator (i): a mass-registration or agent address
    evidence: list[EvidenceRef]


class Entity(_Strict):
    entity_id: str  # stable within a run: "e-<slug>"
    kind: EntityKind
    names: list[str]  # every spelling the documents use, verbatim
    identifiers: dict[
        str, list[str]
    ] = {}  # e.g. {"rfc": [...], "reg_no": [...], "passport": [...]}
    nationality: list[str] = []  # ISO-3166 alpha-2
    dob: list[str] = []  # ISO dates; more than one value is a conflict, not a choice
    jurisdiction: list[str] = []  # incorporation (companies) or residence (persons), alpha-2
    incorporated: list[str] = []  # ISO dates of incorporation (not a certificate's issue date)
    # sub-national region or free zone as the documents state it (e.g. "Crimea", "DMCC free zone"):
    # qualified jurisdiction rules (UA regions, AE free zones) read it; empty means not stated
    regions: list[str] = []
    addresses: list[Address] = []
    evidence: list[EvidenceRef]


class Holding(_Strict):
    holding_id: str
    owner: str  # entity_id
    owned: str  # entity_id
    pct: Decimal | None = None
    shares: int | None = None
    as_of: str | None = None
    status: HoldingStatus
    evidence: list[EvidenceRef]


class Role(_Strict):
    role_id: str
    holder: str  # entity_id
    of: str  # entity_id
    role: RoleKind
    evidence: list[EvidenceRef]


class Document(_Strict):
    doc: str  # the packet's document key ("01")
    document_name: str
    doc_type: DocType
    issued: str | None = None  # ISO date the issuer issued it (a certificate, a statement)
    signed: bool | None = None
    signed_on: str | None = None  # ISO date of signature; a lease is dated by it (spec)
    stated_validity_days: int | None = None  # an issuer's stated validity (spec)
    evidence: list[EvidenceRef]


class Operations(_Strict):
    """Operational indicators (ii)–(v) for the applicant, each present/absent/unknown."""

    distinct_operating_address: TriFact  # (ii)
    bank_reference_or_financials: TriFact  # (iv)
    employees_or_website: TriFact  # (v)


class ConflictValue(_Strict):
    value: str
    evidence: list[EvidenceRef]


class Conflict(_Strict):
    """spec: one conflict per (subject, attribute) with two or more material values."""

    conflict_id: str
    subject: str  # entity_id
    attribute: str  # "ownership_pct", "owner_presence", "address", "reg_no", ...
    values: list[ConflictValue] = Field(min_length=2)
    context: list[
        EvidenceRef
    ] = []  # explanations (later filings, pending transfers): not resolutions


class UncoveredFact(_Strict):
    block_id: str
    kind: str
    subject: str
    attribute: str
    value: str


class DocVerification(_Strict):
    """What verification established for one document, re-read through `assurance_now` (the
    chain as it stands), so a judge can cap the decision without re-deriving it."""

    doc: str
    assurance: Literal["W1", "W2"]
    isolated: bool  # the reader's submission is still eligible as isolated
    uncovered: list[UncoveredFact] = []  # material facts no verified claim matched
    findings: list[
        str
    ] = []  # finding kinds: integrity, layer_vs_pixel, coverage_unestablished, ...


class FactSet(_Strict):
    contracts_version: Literal["b4-v1"] = CONTRACTS_VERSION
    case: str
    run: str
    as_of: str
    fact_set_hash: str  # sha256 of the canonical JSON without this field
    applicant: str  # entity_id
    entities: list[Entity]
    holdings: list[Holding]
    roles: list[Role]
    documents: list[Document]
    conflicts: list[Conflict]
    operations: Operations
    corporate_object: list[str] = []  # every declared activity, verbatim (spec)
    evidence: list[EvidenceItem]  # every EvidenceRef above resolves here
    unverified: list[
        EvidenceRef
    ] = []  # claims that failed verification: rules read them as unknown
    verification: list[DocVerification]  # one per document


# --- screening (L2, spec + spec) --------------------------------------------------------------


class WatchlistEntry(_Strict):
    """A snapshot of the list entry as screened (versioned with the list), so the policy can read
    it without the list: PEP office dates drive the look-back window."""

    entry_id: str
    list_kind: Literal["sanctions", "pep"]
    name: str
    aliases: list[str] = []
    entity_type: str | None = None
    nationality: str | None = None
    jurisdiction: str | None = None
    dob: str | None = None
    registration_number: str | None = None
    program: str | None = None
    position: str | None = None
    in_office: bool | None = None
    left_office: str | None = None


class IdentifierCheck(_Strict):
    corroborating: list[str] = []  # e.g. ["reg_no", "jurisdiction+dob", "address"]
    conflicting: list[str] = []  # e.g. ["reg_no", "jurisdiction"]


class ScreeningCandidate(_Strict):
    """A candidate is a pair worth a judgement. L4 builds its `same_party` instance with the id
    `sp-<candidate_id>`; screening does not."""

    candidate_id: str
    entity_id: str
    entry: WatchlistEntry
    matched_name: str  # the list name or alias that scored highest
    jw_score: float = Field(ge=0, le=1)  # spec: normalized Jaro-Winkler, max over name + aliases
    alias_exact: bool
    triggers_fuzzy: bool  # the policy's fuzzy-match trigger: jw_score >= threshold or alias_exact
    identifiers: IdentifierCheck


class ScreeningResult(_Strict):
    contracts_version: Literal["b4-v1"] = CONTRACTS_VERSION
    case: str
    run: str
    fact_set_hash: str
    watchlist_version: str
    method: str  # "candidates-v1+jw-0.85" (recorded, so a method change is visible)
    candidates: list[ScreeningCandidate]


# --- policy (L3) --------------------------------------------------------------------------------

Severity = Literal["REJECT", "REVIEW", "SCORE"]
RuleKind = Literal[
    "sanctions_corroborated",
    "sanctions_fuzzy",
    "jurisdiction_prohibited",
    "jurisdiction_monitored",
    "ubo_unresolved",
    "ubo_high_risk_unresolved",
    "nominee_control",
    "nominee_sole_controller",
    "dormant_or_layered",
    "pep_active",
    "pep_former",
    "conflicting_filings",
    "required_documents",
    "shell_indicators",
    "shell_severe",
    "industry_prohibited",
    "industry_elevated",
    "conflict_multiplier",
    "document_age_penalty",
]


class RuleSpec(_Strict):
    rule_id: str  # policy-declared identifier, e.g. "SCR-01", "OWN-01", "MOD-01"
    kind: RuleKind | str = ""  # stable internal evaluator kind declared in the policy
    severity: Severity
    score_delta: int = 0
    text: str  # the policy text this rule was parsed from, verbatim


class PolicyModel(_Strict):
    contracts_version: Literal["b4-v1"] = CONTRACTS_VERSION
    rules_version: str  # the declared version ("2026.11-r1")
    content_hash: str  # sha256 of compliance_rules.md
    params: dict[str, Any]  # thresholds, windows, lists; keys documented in b4-contracts.md §3
    rules: list[RuleSpec]
    unrepresentable: list[str] = []  # policy text the parser cannot represent: a visible guard


class RuleResult(_Strict):
    rule_id: str
    severity: Severity
    fired: Literal["yes", "no", "unknown"]  # unknown → MANUAL_REVIEW for that rule (E-p)
    score_delta: int = 0
    predicates: list[str] = []  # instance ids the rule consumed
    evidence: list[EvidenceRef] = []
    description: str


# --- predicates (L4) ------------------------------------------------------------------------------

PredicateValue = bool | Literal["UNKNOWN"]


class SamePartyArgs(_Strict):
    """spec: escalate-only. `true` can raise a review; `false` never cancels a fuzzy match."""

    predicate: Literal["same_party"] = "same_party"
    candidate_id: str
    entity: Entity
    entry: WatchlistEntry


class DirectsBusinessArgs(_Strict):
    """Governance check: is a named individual shown to run the business? `true` is
    favourable (it clears sole-controller escalation), so J-SUPPORT and the evidence judges must
    pass for it to count."""

    predicate: Literal["natural_person_directs_business"] = "natural_person_directs_business"
    company: str  # entity_id
    persons: list[Entity]
    roles: list[Role]


class AmendmentEffectiveArgs(_Strict):
    """Documentary conflict context: is a struck, handwritten or pending amendment in effect?
    Either answer keeps the conflict reported; it informs the explanation, not the count."""

    predicate: Literal["amendment_effective"] = "amendment_effective"
    holding_ids: list[str]
    holdings: list[Holding]


class ProofOfAddressArgs(_Strict):
    """Required filings check: does this document prove a registered or operating address?
    `true` is favourable."""

    predicate: Literal["doc_is_proof_of_address"] = "doc_is_proof_of_address"
    document: Document


PredicateArgs = Annotated[SamePartyArgs | DirectsBusinessArgs | AmendmentEffectiveArgs
                          | ProofOfAddressArgs, Field(discriminator="predicate")]  # fmt: skip
FAVOURABLE = {"natural_person_directs_business": True, "doc_is_proof_of_address": True}
"""Model predicates whose `true` can remove a REVIEW trigger: the residual risk named in
ARCHITECTURE §4.3c. `same_party` and `amendment_effective` cannot lower scrutiny."""


MODEL_PREDICATES = frozenset({"same_party", "natural_person_directs_business",
                              "amendment_effective", "doc_is_proof_of_address"})  # fmt: skip


class PredicateInstance(_Strict):
    instance_id: str  # same_party: "sp-<candidate_id>"
    predicate: str  # catalogued
    kind: Literal["code", "model"]
    args: PredicateArgs | dict[str, Any]  # model: its typed variant; code: plain arguments
    evidence_universe: list[str]  # evidence ids the answer may cite (FactSet.evidence)

    @model_validator(mode="after")
    def _args_match(self) -> PredicateInstance:
        """A model predicate carries exactly its own typed arguments; a code predicate is never
        dispatched as a model one."""
        if self.kind == "model":
            if self.predicate not in MODEL_PREDICATES:
                raise ValueError(f"{self.predicate!r} is not a catalogued model predicate")
            if isinstance(self.args, dict) or self.args.predicate != self.predicate:
                raise ValueError(f"model predicate {self.predicate!r} needs its own typed args")
        elif self.predicate in MODEL_PREDICATES:
            raise ValueError(f"{self.predicate!r} is a model predicate")
        return self


class WorkOrder(_Strict):
    """What a `kyb-predicates` subagent reads (agent-driven-review §4): typed facts, pinned
    verified excerpts and the policy text. No pages, no text layers."""

    contracts_version: Literal["b4-v1"] = CONTRACTS_VERSION
    work_order_id: str
    case: str
    run: str
    fact_set_hash: str
    rules_version: str
    catalog_version: str
    instances: list[PredicateInstance]  # model predicates only
    evidence: list[EvidenceItem]  # every id in any evidence_universe, resolved
    policy_text: dict[str, str]  # rule_id → the rule's verbatim text


class PredicateAnswer(_Strict):
    instance_id: str
    value: PredicateValue
    confidence: float = Field(ge=0, le=1)
    evidence: list[str]  # must be within the instance's evidence_universe
    rationale: str = Field(max_length=600)


# --- corrections (L4 reads, L5 writes) -----------------------------------------------------------


class CorrectionRecord(_Strict):
    """An officer's attributed override. A comparable profile shares the incorporation
    jurisdiction and the fired-rule set (policy §12)."""

    correction_id: str
    decision_id: str
    officer: str
    from_status: Literal["APPROVED", "REJECTED", "MANUAL_REVIEW"]
    to_status: Literal["APPROVED", "REJECTED", "MANUAL_REVIEW"]
    rationale: str = Field(min_length=1)
    jurisdiction: str
    fired_rules: list[str]  # sorted
    recorded_at: str


def comparable_key(jurisdiction: str, fired_rules: list[str]) -> tuple[str, tuple[str, ...]]:
    return jurisdiction.upper(), tuple(sorted(set(fired_rules)))


# --- judges (L4, E-p + spec) -------------------------------------------------------------------


class JudgeOutcome(_Strict):
    """Judges are veto-only: `effect` can only make the outcome more cautious."""

    judge_id: str  # "J-SCHEMA", "J-ARITH", "J-PRECEDENCE", "J-EVIDENCE", "J-SUPPORT", ...
    target: str  # an instance_id, a rule_id or "decision"
    verdict: Literal["pass", "veto"]
    effect: Literal["none", "unknown", "manual_review"]
    reason: str = ""
    model: str | None = None  # J-SUPPORT: the concrete model version the API reported
    request_hash: str | None = None  # recorded call, replayable


# --- decision (L4) -----------------------------------------------------------------------------


class DecisionRecord(_Strict):
    contracts_version: Literal["b4-v1"] = CONTRACTS_VERSION
    decision_id: str  # deterministic from the fingerprint
    case: str
    run: str
    fingerprint: str  # fact_set_hash + policy content_hash + answers + corrections considered
    corrections_considered: list[str] = []  # correction ids (= officer_overrides_considered)
    kyb_decision: KYBDecision  # open-kyb's contract, validated, unchanged
    rule_results: list[RuleResult]
    judge_outcomes: list[JudgeOutcome]
    signature_state: Literal["pending_signature", "signed"] = "pending_signature"
