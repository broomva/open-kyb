"""Offline evaluation of the evidence loop: a finite synthetic matrix run against `plan_cycle`.

Labels and limits (spec Acceptance 2 and 4):

- Fixtures are generic and synthetic. Nothing is read from `datasets/`, supplied golden labels,
  sealed holdouts, witness answers or live output, and no model or provider is called.
- A fresh run (`after`) is a predeclared, simulated facts output. It measures how the planner
  handles that output, not whether a reader would produce it. Value is a delta on these pairs.
- The value oracle counts verified typed inputs straight from the facts, so the planner never
  grades itself. Expectations sit beside each fixture; their hashes go into the receipt.
- A missing core, or a stub that raises NotImplementedError, is `pending`, never a skipped pass.
  Action-layer cases stay pending here: the integrated action run is owned by the parent.
- No product PASS follows from a green matrix. `matrix_verdict` is never a release status.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from kyb.contracts.b4 import FactSet
from kyb.evals.scorecard import Case, Suite, pending
from kyb.facts.derive import DocInput
from kyb.facts.derive import fact_set_hash as derive_hash
from kyb.util.hashing import canonical_json, sha256_json

SCHEMA = "evidence-loop-eval-v1"
FIXTURE_VERSION = "2026-11-01.1"
CORE_NEEDS = "lane A kyb.loop.planner.plan_cycle and kyb.loop.models (LoopBudget, LoopPlan)"
ACTION_NEEDS = "parent-owned integrated run: kyb.loop.action over a real workspace"
DEFAULT_BUDGET = {"max_attempts": 2, "max_elapsed_seconds": 900}
# A same-name or series pair must surface as a candidate that is not linked. Requiring the
# candidate closes the vacuous pass where a planner that emits nothing "never supports" a pair.
NOT_LINKED = ("unresolved", "contradictory")
METRICS = ("tokens", "provider_calls", "elapsed_seconds", "cost_usd")
BILLED_PROVENANCE = ("billed_amount",)  # or a `rate:<name>@<version>` calculation

SUITES = (
    ("E-SAFE", "Safety: no false accept or false promotion", "safety"),
    ("E-PROG", "Progress: duplicate or renamed claims do not earn progress", "progress"),
    ("E-BUDGET", "Budget, usage and refusal", "budget"),
    ("E-VALUE", "Repair closure and verified-input gain", "value"),
    ("E-CONTROL", "Positive controls: true identity, ready, unknown cost", "control"),
    ("E-LINEAGE", "Determinism, lineage and action-layer lineage", "lineage"),
)

# Action-layer cases. The parent-owned integrated run owns these; they are never passed here.
ACTION_CASES = (
    "A01-stale-source-rejected",
    "A02-cross-case-plan-rejected",
    "A03-tampered-predecessor-rejected",
    "A04-identical-retry-returns-original",
    "A05-one-allocation-per-plan",
    "A06-refusal-writes-nothing",
    "A07-plan-bytes-repeat-on-disk",
    "A08-cli-mcp-agree",
)

# Open interface questions for lane A. Recorded so they are not silently resolved downstream.
INTERFACE_QUESTIONS = (
    "usage shape is unnamed; fixtures use {metric: {status, value, provenance}}",
    "max_attempts=0 terminal state is unnamed; the eval accepts budget_exhausted or manual_review",
    "explicit-nondisclosure carrier is unnamed; fixtures use verification findings "
    "'explicit_nondisclosure' beside an uncovered controller",
    "ReconciliationCandidate has no applied_to_factset field; absence fails the safety check",
    "semantic gap key is not frozen; the eval uses (kind, entity_ids, doc_ids, field)",
    "Gap.reason is free text and must never reach a reader prompt",
    "planner trusts caller-supplied usage and previous; the action layer must derive both",
    "Entity has no series field; same registry number across series must not merge",
)


class CoreMissing(RuntimeError):
    """The lane-A core is not importable here: planner cases are pending, not skipped."""


@dataclass(frozen=True)
class Core:
    plan_cycle: Callable[..., Any]
    budget: Callable[..., Any]
    plan: Any  # the LoopPlan model, or None for an injected double


@dataclass(frozen=True)
class Fixture:
    case_id: str
    group: str
    summary: str
    facts: dict[str, Any]
    after: dict[str, Any] | None = None
    prior: bool = False
    budget: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_BUDGET))
    usage: dict[str, Any] | None = None
    expect: dict[str, Any] = field(default_factory=dict)


def load_core() -> Core:
    try:
        planner = importlib.import_module("kyb.loop.planner")
        models = importlib.import_module("kyb.loop.models")
        return Core(planner.plan_cycle, models.LoopBudget, models.LoopPlan)
    except (ImportError, AttributeError) as exc:
        raise CoreMissing(f"{CORE_NEEDS}: {exc}") from exc


# --- synthetic fixture builders (generic names; every ref resolves to an evidence item) ---------

DOC_NAMES = {"01": "01_incorporation.pdf", "02": "02_declaracion_jurada_ubo.pdf"}
DOC_TYPES = {"01": "incorporation", "02": "ubo_declaration"}


def ev(cid: str, doc: str = "02", *, verified: bool = True, text: str | None = None) -> dict:
    return {
        "evidence_id": cid,
        "doc": doc,
        "document_name": DOC_NAMES[doc],
        "page": 1,
        "bbox": [10, 10, 400, 40],
        "excerpt": text or f"Synthetic statement {cid}.",
        "verified": verified,
        "field": None,
    }


def ref(cid: str, doc: str = "02", *, verified: bool = True) -> dict:
    return {"claim_id": cid, "doc": doc, "page": 1, "verified": verified}


def ent(eid: str, names: list[str], *, kind: str = "company", juris=(), ids=None, refs=()) -> dict:
    return {
        "entity_id": eid,
        "kind": kind,
        "names": list(names),
        "identifiers": dict(ids or {}),
        "nationality": [],
        "dob": [],
        "jurisdiction": list(juris),
        "incorporated": [],
        "regions": [],
        "addresses": [],
        "evidence": list(refs),
    }


def hold(hid: str, owner: str, owned: str, *, pct=None, shares=None, status="current", refs=()):
    return {
        "holding_id": hid,
        "owner": owner,
        "owned": owned,
        "pct": pct,
        "shares": shares,
        "as_of": None,
        "status": status,
        "evidence": list(refs),
    }


APPLICANT = ent(
    "e-app",
    ["Applicant Co S.A."],
    juris=["PA"],
    ids={"reg_no": ["REG-100"]},
    refs=[ref("c-app", "01")],
)
APP_EVIDENCE = [ev("c-app", "01", text="Applicant Co S.A. is incorporated in the synthetic PA.")]
OPS_EVIDENCE = [ev("c-ops", "01", text="Synthetic operating statement.")]


def _tri(state: str) -> dict:
    refs = [ref("c-ops", "01")] if state == "present" else []
    return {"state": state, "evidence": refs, "note": ""}


def _document(doc: str) -> dict:
    refs = [ref("c-app", "01")] if doc == "01" else []
    return {
        "doc": doc,
        "document_name": DOC_NAMES[doc],
        "doc_type": DOC_TYPES[doc],
        "issued": None,
        "signed": None,
        "signed_on": None,
        "stated_validity_days": None,
        "evidence": refs,
    }


def facts(
    case: str,
    *,
    entities=(),
    holdings=(),
    evidence=(),
    conflicts=(),
    uncovered=(),
    findings=(),
    ops: str = "present",
    assurance: str = "W2",
    isolated: bool = True,
    verification: bool = True,
) -> dict:
    """A FactSet dict that validates against the existing contract, with a derived hash."""
    body = {
        "case": f"SYN-{case}",
        "run": "syn-r1",
        "as_of": "2026-10-01",
        "fact_set_hash": "",
        "applicant": "e-app",
        "entities": [APPLICANT, *entities],
        "holdings": list(holdings),
        "roles": [],
        "documents": [_document("01"), _document("02")],
        "conflicts": list(conflicts),
        "operations": {
            "distinct_operating_address": _tri(ops),
            "bank_reference_or_financials": _tri(ops),
            "employees_or_website": _tri(ops),
        },
        "corporate_object": ["synthetic activity"],
        "evidence": [*APP_EVIDENCE, *OPS_EVIDENCE, *evidence],
        "unverified": [],
        "verification": [
            {
                "doc": doc,
                "assurance": assurance,
                "isolated": isolated,
                "uncovered": list(uncovered),
                "findings": list(findings),
            }
            for doc in ("01", "02")
        ]
        if verification
        else [],
    }
    model = FactSet.model_validate(body)
    return {**body, "fact_set_hash": derive_hash(model)}


def covered(*, juris=(), tag: str = "", long_text: bool = False, dup: bool = False) -> dict:
    """One verified holder owning 100 percent. `juris` adds a separately verified claim."""
    text = "Covered Holder S.A. holds the applicant shares."
    if long_text:
        text += " The same statement continues with further synthetic wording for the excerpt."
    refs = [ref(f"c-t1{tag}")]
    evs = [ev(f"c-t1{tag}", text=text)]
    if juris:
        refs.append(ref(f"c-t1-juris{tag}"))
        evs.append(ev(f"c-t1-juris{tag}", text="Covered Holder S.A. is incorporated in MX."))
    if dup:
        refs.append(ref(f"c-t1-dup{tag}"))
        evs.append(ev(f"c-t1-dup{tag}", text=text))
    holder = ent("e-t1", ["Covered Holder S.A."], juris=juris, ids={"reg_no": ["CH-1"]}, refs=refs)
    hold_ = hold("h-t1", "e-t1", "e-app", pct="100", refs=[ref(f"c-h1{tag}")])
    evs.append(ev(f"c-h1{tag}", text="Covered Holder S.A. owns one hundred percent."))
    return {"entities": [holder], "holdings": [hold_], "evidence": evs}


def _merge(*parts: dict) -> dict:
    out: dict[str, list] = {}
    for part in parts:
        for key, value in part.items():
            out.setdefault(key, []).extend(value)
    return out


def _two_holders(names, juris, ids, *, shares=None, pcts=("30", "20")) -> dict:
    ents, hs, evs = [], [], []
    for i, (name, jur, reg) in enumerate(zip(names, juris, ids, strict=True), start=1):
        eid, cid = f"e-p{i}", f"c-p{i}"
        ents.append(ent(eid, [name], juris=[jur] if jur else [], ids=reg, refs=[ref(cid)]))
        evs.append(ev(cid, text=f"{name} holds a stake in the applicant."))
        hs.append(
            hold(f"h-p{i}", eid, "e-app", pct=None if shares else pcts[i - 1], shares=shares,
                 refs=[ref(cid)])
        )  # fmt: skip
    return {"entities": ents, "holdings": hs, "evidence": evs}


def _fixtures() -> list[Fixture]:
    fx: list[Fixture] = []

    # --- safety: negatives that must never become supported, ready or applied -----------------
    same_name = _two_holders(
        ["Northwind Trading S.A.", "Northwind Trading S.A."], ["BR", "MX"], [{}, {}]
    )
    fx.append(
        Fixture(
            "S01-same-name-diff-country",
            "safety",
            "two identically named holders in BR and MX, no identifiers",
            facts("S01", **same_name),
            expect={
                "forbid_supported": [("e-p1", "e-p2")],
                "candidate_in": [(("e-p1", "e-p2"), NOT_LINKED)],
            },
        )
    )
    same_name_pa = _two_holders(
        ["Orion Partners S.A.", "Orion Partners S.A."], ["PA", "PA"], [{}, {}]
    )
    fx.append(
        Fixture(
            "S02-same-name-same-country-no-ids",
            "safety",
            "same name and country, no strong identifiers",
            facts("S02", **same_name_pa),
            expect={
                "forbid_supported": [("e-p1", "e-p2")],
                "candidate_in": [(("e-p1", "e-p2"), NOT_LINKED)],
            },
        )
    )
    conflicting = _two_holders(
        ["Vega Logistics S.A.", "Vega Logistics S.A."],
        ["PA", "PA"],
        [{"reg_no": ["V-1"]}, {"reg_no": ["V-2"]}],
    )
    fx.append(
        Fixture(
            "S03-conflicting-identifiers",
            "safety",
            "same name and country, different registry numbers",
            facts("S03", **conflicting),
            expect={
                "forbid_supported": [("e-p1", "e-p2")],
                "candidate_in": [(("e-p1", "e-p2"), NOT_LINKED)],
            },
        )
    )
    series = _two_holders(
        ["Helix Fund - Series A", "Helix Fund - Series B"],
        ["PA", "PA"],
        [{"reg_no": ["HF-1"]}, {"reg_no": ["HF-1"]}],
        shares=1000,
    )
    fx.append(
        Fixture(
            "S04-same-count-different-series",
            "safety",
            "same registry number and share count, distinct series qualifiers",
            facts("S04", **series),
            expect={
                "forbid_supported": [("e-p1", "e-p2")],
                "candidate_in": [(("e-p1", "e-p2"), NOT_LINKED)],
            },
        )
    )
    forged = _merge(
        covered(),
        {"holdings": [hold("h-f", "e-t1", "e-app", pct="50", refs=[ref("c-forged")])]},
    )
    fx.append(
        Fixture(
            "S05-forged-evidence-ref",
            "safety",
            "holding cites a claim id that no evidence item resolves",
            facts("S05", **forged),
            expect={"forbid_state": ["ready"], "state_in": ["blocked", "repair", "manual_review"]},
        )
    )
    unverified_ref = ref("c-u1", verified=False)
    unverified = {
        "entities": [ent("e-u1", ["Unverified Holder S.A."], juris=["PA"], refs=[unverified_ref])],
        "holdings": [hold("h-u", "e-u1", "e-app", pct="40", refs=[unverified_ref])],
        "evidence": [ev("c-u1", verified=False, text="Unverified Holder S.A. holds 40 percent.")],
    }
    fx.append(
        Fixture(
            "S06-unverified-claim-input",
            "safety",
            "holding rests on a claim the witness did not verify",
            facts("S06", **unverified),
            expect={"forbid_state": ["ready"]},
        )
    )
    nondisclosure = facts(
        "S07",
        **covered(juris=("PA",)),
        uncovered=[{"block_id": "b-ctrl", "kind": "controller", "subject": "e-app",
                    "attribute": "controller", "value": ""}],
        findings=["explicit_nondisclosure"],
    )  # fmt: skip
    fx.append(
        Fixture(
            "S07-explicit-nondisclosure-not-repairable",
            "safety",
            "controller stated as disclosed on request: unresolved business fact",
            nondisclosure,
            expect={
                "state": "manual_review",
                "forbid_repairable_entity": ["e-app"],
                "repair_admitted": False,
            },
        )
    )
    struck = _merge(
        {
            "entities": [ent("e-p1", ["Person One"], kind="person", refs=[ref("c-p1a")])],
            "holdings": [
                hold("h-c", "e-p1", "e-t1", pct="40", refs=[ref("c-p1a")]),
                hold("h-d", "e-p1", "e-t1", pct="25", status="struck", refs=[ref("c-p1b")]),
            ],
            "evidence": [
                ev("c-p1a", "01", text="Person One holds 40 percent."),
                ev("c-p1b", "01", text="Person One holds 25 percent (struck)."),
            ],
        },  # fmt: skip
        covered(juris=("PA",)),
    )
    struck["conflicts"] = [
        {
            "conflict_id": "k-1",
            "subject": "e-t1",
            "attribute": "ownership_pct",
            "values": [
                {"value": "40", "evidence": [ref("c-p1a", "01")]},
                {"value": "25", "evidence": [ref("c-p1b", "01")]},
            ],
            "context": [],
        }
    ]
    fx.append(
        Fixture(
            "S08-struck-current-conflict",
            "safety",
            "a struck holding does not resolve a printed-versus-current conflict",
            facts("S08", **struck),
            expect={
                "state_in": ["manual_review", "repair"],
                "forbid_state": ["ready"],
                "must_gap_entity": ["e-t1"],
            },
        )
    )
    pending_transfer = {
        "entities": [ent("e-p1", ["Person One"], kind="person", refs=[ref("c-p1c")])],
        "holdings": [
            hold("h-pend", "e-p1", "e-t1", pct="25", status="pending", refs=[ref("c-p1c")])
        ],
        "evidence": [ev("c-p1c", "01", text="Person One agreed to buy 25 percent.")],
    }
    fx.append(
        Fixture(
            "S09-pending-transfer-not-current",
            "safety",
            "a pending transfer must not count as current ownership",
            facts("S09", **pending_transfer),
            expect={"forbid_state": ["ready"]},
        )
    )
    helix = _two_holders(
        ["Helix Labs S.A.", "Helix Labs"], ["PA", "PA"], [{"reg_no": ["H-7"]}, {"reg_no": ["H-7"]}],
        pcts=("50", "50"),
    )  # fmt: skip
    fx.append(
        Fixture(
            "S10-uncovered-not-suppressed-by-link",
            "safety",
            "a supported identity link must not clear an uncovered finding on one side",
            facts(
                "S10",
                **helix,
                uncovered=[
                    {
                        "block_id": "b-h",
                        "kind": "holding",
                        "subject": "e-p1",
                        "attribute": "pct",
                        "value": "",
                    }
                ],
            ),  # fmt: skip
            expect={
                "require_supported": [("e-p1", "e-p2")],
                "forbid_state": ["ready"],
                "must_gap_entity": ["e-p1"],
            },
        )
    )
    fx.append(
        Fixture(
            "S11-unknown-is-not-absent",
            "safety",
            "an operating-address indicator that is unknown must not read as absent or ready",
            facts("S11", **covered(juris=("PA",)), ops="unknown"),
            expect={"forbid_state": ["ready"]},
        )
    )
    fx.append(
        Fixture(
            "S12-revoked-assurance-blocks",
            "safety",
            "a revoked (isolated=False, W1) verification blocks planning",
            facts("S12", **covered(juris=("PA",)), assurance="W1", isolated=False),
            expect={"state": "blocked"},
        )
    )
    fx.append(
        Fixture(
            "S13-missing-verification-blocks",
            "safety",
            "no verification record is an error, not an empty successful case",
            facts("S13", **covered(juris=("PA",)), verification=False),
            expect={"state": "blocked"},
        )
    )

    # --- progress: duplicate, renamed or broader claims earn nothing ---------------------------
    before = covered()
    fx.append(
        Fixture(
            "P01-duplicate-claims-no-progress",
            "progress",
            "same unresolved gap, plus duplicated claim ids on the same holder",
            facts("P01", **before),
            after=facts("P01b", **covered(dup=True)),
            prior=True,
            expect={"state": "no_progress"},
        )
    )
    fx.append(
        Fixture(
            "P02-renamed-broader-no-progress",
            "progress",
            "same unresolved gap after renaming claim ids and broadening an excerpt",
            facts("P02", **before),
            after=facts("P02b", **covered(tag="-v2", long_text=True)),
            prior=True,
            expect={"state": "no_progress"},
        )
    )
    fx.append(
        Fixture(
            "P03-real-progress-not-no-progress",
            "progress",
            "one gap closed and a different gap opened: the gap set changed",
            facts("P03", **before),
            after=facts(
                "P03b",
                **_merge(
                    covered(juris=("MX",)),
                    {"entities": [ent("e-t9", ["Second Holder S.A."], refs=[])]},
                ),
            ),
            prior=True,
            expect={"forbid_state": ["no_progress"]},
        )
    )

    # --- budget, usage and refusal -------------------------------------------------------------
    fx.append(
        Fixture(
            "B01-attempts-exhausted",
            "budget",
            "previous attempt used the whole attempt budget",
            facts("B01", **before),
            prior=True,
            budget={"max_attempts": 1},
            expect={"state": "budget_exhausted"},
        )
    )
    fx.append(
        Fixture(
            "B02-max-attempts-zero-inspect-only",
            "budget",
            "max_attempts=0 is inspect-only: a gap is reported but no repair is admitted",
            facts("B02", **before),
            budget={"max_attempts": 0},
            expect={
                "state_in": ["budget_exhausted", "manual_review"],
                "forbid_state": ["repair"],
                "repair_admitted": False,
            },
        )
    )
    fx.append(
        Fixture(
            "B03-elapsed-exceeded",
            "budget",
            "measured elapsed time is over the wall-clock budget",
            facts("B03", **before),
            prior=True,
            usage={
                "elapsed_seconds": {
                    "status": "measured",
                    "value": 1000,
                    "provenance": "loop-record",
                }
            },  # fmt: skip
            expect={"state": "budget_exhausted"},
        )
    )
    fx.append(
        Fixture(
            "B04-token-usage-unknown-blocked",
            "budget",
            "a token cap is configured but tokens are unknown: stop, never treat as free",
            facts("B04", **before),
            budget={"max_attempts": 2, "max_tokens": 10000},
            usage={"tokens": {"status": "unknown", "provenance": "not_supplied"}},
            expect={"state": "blocked", "reason_contains": "usage_unknown"},
        )
    )
    fx.append(
        Fixture(
            "B05-tokens-measured-exhausted",
            "budget",
            "measured tokens exceed the configured cap",
            facts("B05", **before),
            prior=True,
            budget={"max_attempts": 2, "max_tokens": 10000},
            usage={
                "tokens": {"status": "measured", "value": 12000, "provenance": "harness-report"}
            },  # fmt: skip
            expect={"state": "budget_exhausted"},
        )
    )
    fx.append(
        Fixture(
            "B06-provider-calls-unknown-blocked",
            "budget",
            "a provider-call cap is configured but the call count is unknown",
            facts("B06", **before),
            budget={"max_attempts": 2, "max_provider_calls": 5},
            usage={"provider_calls": {"status": "unknown", "provenance": "not_supplied"}},
            expect={"state": "blocked", "reason_contains": "usage_unknown"},
        )
    )
    fx.append(
        Fixture(
            "B07-cost-unknown-under-cap-blocked",
            "budget",
            "a currency cap is set but the flat-quota call has no billed amount",
            facts("B07", **before),
            budget={"max_attempts": 2, "max_cost_usd": 5.0},
            usage={"cost_usd": {"status": "unknown", "provenance": "flat_quota"}},
            expect={"state": "blocked", "reason_contains": "usage_unknown"},
        )
    )

    # --- value: repair closes a typed-input gap; ready is reached only by evidence -----------
    fx.append(
        Fixture(
            "V01-missing-jurisdiction-repairable",
            "value",
            "a corporate holder without a typed jurisdiction is repairable",
            facts("V01", **before),
            expect={
                "state": "repair",
                "must_gap_entity": ["e-t1"],
                "repair_admitted": True,
            },
        )
    )
    fx.append(
        Fixture(
            "V02-fresh-run-closes-gap",
            "value",
            "a predeclared fresh run adds a verified jurisdiction and closes the gap",
            facts("V02", **before),
            after=facts("V02b", **covered(juris=("MX",))),
            prior=True,
            expect={
                "state_in": ["ready", "repair"],
                "forbid_repairable_entity": ["e-t1"],
                "gain_min": 1,
            },
        )
    )

    # --- control: positive cases that must still pass -----------------------------------------
    helix_true = _two_holders(
        ["Helix Labs S.A.", "Helix Labs"], ["PA", "PA"], [{"reg_no": ["H-7"]}, {"reg_no": ["H-7"]}],
        pcts=("50", "50"),
    )  # fmt: skip
    fx.append(
        Fixture(
            "C01-true-supported-identity",
            "control",
            "equal verified registry number, same verified jurisdiction: supported, not applied",
            facts("C01", **helix_true),
            expect={"require_supported": [("e-p1", "e-p2")]},
        )
    )
    fx.append(
        Fixture(
            "C02-fully-covered-ready",
            "control",
            "a fully covered holder with verified inputs is ready (not APPROVED)",
            facts("C02", **covered(juris=("PA",))),
            expect={"state": "ready"},
        )
    )
    fx.append(
        Fixture(
            "C03-flat-quota-cost-unknown-not-zero",
            "control",
            "a flat-quota call has unknown cost: recorded as unknown, never zero, and no cap",
            facts("C03", **before),
            usage={
                "cost_usd": {"status": "unknown", "provenance": "flat_quota"},
                "provider_calls": {
                    "status": "measured",
                    "value": 2,
                    "provenance": "harness-report",
                },
            },  # fmt: skip
            expect={"state": "repair", "repair_admitted": True},
        )
    )

    # --- lineage: planner-level determinism and hash lineage ----------------------------------
    fx.append(
        Fixture(
            "L01-repair-plan-repeat-bytes",
            "lineage",
            "a repair plan repeats byte-identically and carries its input fact-set hash",
            facts("L01", **before),
            expect={"state": "repair"},
        )
    )
    return fx


FIXTURES: list[Fixture] = _fixtures()


# --- independent oracle and metric helpers -----------------------------------------------------


def verified_inputs(facts_dict: dict) -> set[tuple]:
    """Verified typed inputs, counted from facts. Independent of the planner."""
    ok_ids = {e["evidence_id"]: e["verified"] for e in facts_dict["evidence"]}

    def verified(refs: list[dict]) -> bool:
        return bool(refs) and all(r["verified"] and ok_ids.get(r["claim_id"]) is True for r in refs)

    out: set[tuple] = set()
    for h in facts_dict["holdings"]:
        if h["status"] == "current" and verified(h["evidence"]):
            out.add(("holding", h["owner"], h["owned"], str(h["pct"]), str(h["shares"])))
    for e in facts_dict["entities"]:
        if not verified(e["evidence"]):
            continue
        out.update(("jurisdiction", e["entity_id"], j) for j in e["jurisdiction"])
        out.update(
            ("identifier", e["entity_id"], ns, v) for ns, vs in e["identifiers"].items() for v in vs
        )
    return out


def gap_key(gap: dict) -> str:
    """Semantic key: kind, entities, documents and field. Ids and wording do not enter it."""
    return "|".join(
        [
            gap.get("kind") or "",
            ",".join(sorted(gap.get("entity_ids") or [])),
            ",".join(sorted(gap.get("doc_ids") or [])),
            gap.get("field") or "",
        ]
    )


def repair_admitted(plan: dict) -> bool:
    return plan.get("state") == "repair" and any(
        g.get("repairable") for g in plan.get("gaps") or []
    )


def usage_metric(usage: dict | None, key: str) -> dict[str, Any]:
    """Measured value or explicit unknown. A missing metric is unknown, never zero."""
    raw = (usage or {}).get(key)
    if not isinstance(raw, dict):
        return {"status": "unknown", "value": None, "provenance": "not_supplied"}
    if raw.get("status") != "measured" or not isinstance(raw.get("value"), int | float):
        return {
            "status": "unknown",
            "value": None,
            "provenance": raw.get("provenance", "unlabelled"),
        }
    return {"status": "measured", "value": raw["value"], "provenance": raw.get("provenance", "")}


def cost_metric(usage: dict | None) -> dict[str, Any]:
    """Currency is measured only when billed or a versioned rate calculation says so."""
    cost = usage_metric(usage, "cost_usd")
    prov = cost["provenance"]
    billed = prov in BILLED_PROVENANCE or prov.startswith("rate:")
    if cost["status"] == "measured" and billed:
        return cost
    return {"status": "unknown", "value": None, "provenance": prov or "not_supplied"}


def _plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def _model(kind: Any, value: dict | None) -> Any:
    if value is None or not hasattr(kind, "model_validate"):
        return value
    return kind.model_validate(value)


def _doc_input(doc: str) -> DocInput:
    return DocInput(
        doc=doc,
        document_name=DOC_NAMES[doc],
        claims=[],
        verdicts={},
        assurance="W2",
        isolated=True,
    )


def run_planner(core: Core, fx: Fixture) -> dict[str, Any]:
    """Drive the real planner (or an injected double) through the frozen signature."""
    docs = [_doc_input("01"), _doc_input("02")]
    before = FactSet.model_validate(fx.facts)
    prior = None
    if fx.prior:
        prior = _plain(
            core.plan_cycle(
                before, docs, budget=core.budget(**DEFAULT_BUDGET), previous=None, usage=None
            )
        )
    current_dict = fx.after or fx.facts
    current = FactSet.model_validate(current_dict)
    previous = _model(core.plan, prior)
    budget = core.budget(**fx.budget)
    plan = _plain(core.plan_cycle(current, docs, budget=budget, previous=previous, usage=fx.usage))
    again = _plain(core.plan_cycle(current, docs, budget=budget, previous=previous, usage=fx.usage))
    baseline = _plain(
        core.plan_cycle(before, docs, budget=core.budget(max_attempts=0), previous=None, usage=None)
    )
    return {
        "plan": plan,
        "again": again,
        "baseline": baseline,
        "before": fx.facts,
        "current": current_dict,
    }


def score(fx: Fixture, run: dict[str, Any]) -> Case:
    plan, exp = run["plan"], fx.expect
    state = plan.get("state")
    gaps = plan.get("gaps") or []
    candidates = plan.get("candidates") or []
    cands = {tuple(sorted((c.get("left"), c.get("right")))): c for c in candidates}
    fails: list[str] = []
    breach = None
    if "state" in exp and state != exp["state"]:
        fails.append(f"state {state!r} != {exp['state']!r}")
    if exp.get("state_in") and state not in exp["state_in"]:
        fails.append(f"state {state!r} not in {exp['state_in']}")
    if state in exp.get("forbid_state", []):
        fails.append(f"state {state!r} is forbidden here")
        breach = "false_accept" if state == "ready" else None
    for eid in exp.get("must_gap_entity", []):
        if not any(eid in (g.get("entity_ids") or []) for g in gaps):
            fails.append(f"no gap names {eid}")
    for eid in exp.get("forbid_repairable_entity", []):
        if any(g.get("repairable") and eid in (g.get("entity_ids") or []) for g in gaps):
            fails.append(f"repairable gap names {eid}")
    for pair in exp.get("forbid_supported", []):
        c = cands.get(tuple(sorted(pair)))
        if c and c.get("state") == "supported":
            fails.append(f"{pair} is supported")
            breach = "false_promotion"
    for pair in exp.get("require_supported", []):
        c = cands.get(tuple(sorted(pair)))
        if not c or c.get("state") != "supported":
            fails.append(f"{pair} not supported")
    for pair, states in exp.get("candidate_in", []):
        c = cands.get(tuple(sorted(pair)))
        if not c or c.get("state") not in states:
            fails.append(f"{pair} candidate not in {list(states)}")
    for c in candidates:
        if c.get("applied_to_factset") is not False:
            fails.append(f"candidate {c.get('left')}~{c.get('right')} not recorded applied=false")
            breach = "false_promotion"
    if "repair_admitted" in exp and repair_admitted(plan) != exp["repair_admitted"]:
        fails.append(f"repair_admitted != {exp['repair_admitted']}")
    if "reason_contains" in exp and exp["reason_contains"] not in canonical_json(plan).decode():
        fails.append(f"reason does not mention {exp['reason_contains']!r}")
    gain = len(verified_inputs(run["current"]) - verified_inputs(run["before"]))
    if "gain_min" in exp and gain < exp["gain_min"]:
        fails.append(f"verified-input gain {gain} < {exp['gain_min']}")
    if plan.get("fact_set_hash") != run["current"]["fact_set_hash"]:
        fails.append("lineage: plan fact_set_hash differs from its input")
    if canonical_json(plan) != canonical_json(run["again"]):
        fails.append("non-deterministic: repeat plan bytes differ")
    data = {"state": state, "gain": gain, "breach": breach, "gaps": len(gaps)}
    return Case(fx.case_id, "fail" if fails else "pass", "; ".join(fails), data=data)


def _row(fx: Fixture, run: dict[str, Any]) -> dict[str, Any]:
    plan, base = run["plan"], run["baseline"]
    before, current = verified_inputs(run["before"]), verified_inputs(run["current"])
    cost = cost_metric(fx.usage)
    return {
        "case": fx.case_id,
        "group": fx.group,
        "state_one_way": base.get("state"),
        "state_loop": plan.get("state"),
        "unresolved_before": sorted({gap_key(g) for g in base.get("gaps") or []}),
        "unresolved_after": sorted({gap_key(g) for g in plan.get("gaps") or []}),
        "verified_gain": len(current - before),
        "verified_lost": len(before - current),
        "repair_admitted": repair_admitted(plan),
        "usage": {k: usage_metric(fx.usage, k) for k in METRICS if k != "cost_usd"}
        | {"cost_usd": cost},
    }


def _lineage(fx: Fixture, run: dict[str, Any]) -> dict[str, Any]:
    return {
        "case": fx.case_id,
        "input_sha256": sha256_json(
            {
                "facts": fx.facts,
                "after": fx.after,
                "prior": fx.prior,
                "budget": fx.budget,
                "usage": fx.usage,
            }
        ),  # fmt: skip
        "expect_sha256": sha256_json(fx.expect),
        "fact_set_hash": run["current"]["fact_set_hash"],
        "plan_id": run["plan"].get("plan_id"),
        "plan_sha256": sha256_json(run["plan"]),
    }


def _bypass_core() -> Core:
    """Self-check double: always ready, no gaps. A green matrix must not survive this."""

    def always_ready(facts_model, documents, *, budget, previous=None, usage=None):
        return {
            "state": "ready",
            "gaps": [],
            "candidates": [],
            "fact_set_hash": facts_model.fact_set_hash,
            "plan_id": "bypass",
            "attempt": 1,
        }

    return Core(always_ready, lambda **kw: kw, None)


def _requires_non_ready(fx: Fixture) -> bool:
    exp = fx.expect
    if exp.get("state") not in (None, "ready"):
        return True
    if exp.get("candidate_in") or exp.get("require_supported"):
        return True  # a planner that emits no candidates cannot satisfy these
    if "ready" in exp.get("forbid_state", []):
        return True
    return bool(exp.get("state_in")) and "ready" not in exp["state_in"]


def run_matrix(
    core: Core | None = None,
    *,
    fixtures: list[Fixture] | None = None,
    code_version: str = "unrecorded",
    selfcheck: bool = True,
) -> dict[str, Any]:
    fixtures = FIXTURES if fixtures is None else fixtures
    core_status = "injected" if core is not None else "present"
    if core is None:
        try:
            core = load_core()
        except CoreMissing as exc:
            core_status = f"missing: {exc}"
    suites = {sid: Suite(sid, name, True) for sid, name, _ in SUITES}
    group_suite = {grp: sid for sid, _, grp in SUITES}
    rows, lineage = [], []
    for fx in fixtures:
        suite = suites[group_suite[fx.group]]
        if core is None:
            suite.cases.append(pending(fx.case_id, CORE_NEEDS, "planner not importable"))
            continue
        try:
            run = run_planner(core, fx)
        except NotImplementedError as exc:
            suite.cases.append(pending(fx.case_id, CORE_NEEDS, f"core stub: {exc}"))
            continue
        except Exception as exc:  # a real core that raises on a valid fixture is a failure
            suite.cases.append(
                Case(fx.case_id, "fail", f"planner raised {type(exc).__name__}: {exc}")
            )
            continue
        suite.cases.append(score(fx, run))
        rows.append(_row(fx, run))
        lineage.append(_lineage(fx, run))
    for case_id in ACTION_CASES:
        suites["E-LINEAGE"].cases.append(pending(case_id, ACTION_NEEDS))

    selfcheck_doc = {"ran": False}
    if selfcheck:  # runs without the core too: the matrix itself must not be vacuous
        bypass = run_matrix(_bypass_core(), fixtures=fixtures, selfcheck=False)
        must_fail = [f.case_id for f in fixtures if _requires_non_ready(f)]
        failing = {c["id"] for s in bypass["suites"] for c in s["cases"] if c["status"] == "fail"}
        missed = [cid for cid in must_fail if cid not in failing]
        selfcheck_doc = {"ran": True, "must_fail": len(must_fail), "missed": missed,
                         "ok": not missed}  # fmt: skip

    breaches = [c for s in suites.values() for c in s.cases if c.data.get("breach")]
    false_accept = sum(c.data.get("breach") == "false_accept" for c in breaches)
    false_promotion = sum(c.data.get("breach") == "false_promotion" for c in breaches)
    failing_suites = [s.suite_id for s in suites.values() if s.status == "fail"]
    pending_suites = [s.suite_id for s in suites.values() if s.status == "pending"]
    if failing_suites or false_accept or false_promotion or selfcheck_doc.get("ok") is False:
        verdict = "fail"
    elif pending_suites:
        verdict = "pending"
    else:
        verdict = "pass"
    return {
        "schema": SCHEMA,
        "fixture_version": FIXTURE_VERSION,
        "matrix_verdict": verdict,
        "product_pass": False,
        "release_gate": False,
        "headline": (
            f"{verdict}: {false_accept} false-accept · {false_promotion} false-promotion · "
            f"{len(pending_suites)} suites pending (matrix only, not a product PASS)"
        ),
        "false_accept": false_accept,
        "false_promotion": false_promotion,
        "failing_suites": failing_suites,
        "pending_suites": pending_suites,
        "selfcheck": selfcheck_doc,
        "inputs": {
            "core": core_status,
            "code_version": code_version,
            "fixtures_sha256": sha256_json([_lineage_input(f) for f in fixtures]),
            "expectations_sha256": sha256_json([f.expect for f in fixtures]),
            "simulated_fresh_runs": True,
            "budget_default": DEFAULT_BUDGET,
        },
        "suites": [s.as_dict() for s in suites.values()],
        "value": _value(rows),
        "cost": _cost(rows),
        "lineage": lineage,
        "action_pending": list(ACTION_CASES),
        "interface_questions": list(INTERFACE_QUESTIONS),
    }


def _lineage_input(fx: Fixture) -> dict[str, Any]:
    return {"case": fx.case_id, "input": sha256_json({"facts": fx.facts, "after": fx.after})}


def _value(rows: list[dict]) -> dict[str, Any]:
    return {
        "baseline": "one-way pipeline: the same planner with max_attempts=0 (no repair)",
        "rows": rows,
        "totals": {
            "verified_gain": sum(r["verified_gain"] for r in rows),
            "verified_lost": sum(r["verified_lost"] for r in rows),
            "unresolved_keys_one_way": sum(len(r["unresolved_before"]) for r in rows),
            "unresolved_keys_loop": sum(len(r["unresolved_after"]) for r in rows),
            "repair_admitted": sum(r["repair_admitted"] for r in rows),
        },
    }


def _cost(rows: list[dict]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in METRICS:
        values = [r["usage"][key] for r in rows]
        measured = [v["value"] for v in values if v["status"] == "measured"]
        out[key] = {
            "measured_total": sum(measured) if measured else None,
            "unknown_cases": sum(v["status"] == "unknown" for v in values),
        }
    gain = sum(r["verified_gain"] for r in rows)
    unknown_cost = out["cost_usd"]["unknown_cases"]
    if gain == 0:
        per_gain = {"value": None, "reason": "verified gain is zero"}
    elif unknown_cost:
        per_gain = {"value": None, "reason": f"currency unknown in {unknown_cost} case(s)"}
    else:
        per_gain = {"value": out["cost_usd"]["measured_total"] / gain, "reason": "measured"}
    out["cost_per_verified_gain"] = per_gain
    out["note"] = "unknown is never zero; flat-quota calls stay unknown currency"
    return out


def exit_code(doc: dict[str, Any]) -> int:
    return {"pass": 0, "fail": 1, "pending": 2}[doc["matrix_verdict"]]


def render_summary(doc: dict[str, Any]) -> str:
    lines = [
        f"evidence-loop matrix: {doc['matrix_verdict'].upper()} "
        "(not a product PASS; release_gate=false)",
        f"core: {doc['inputs']['core']}",
    ]
    for s in doc["suites"]:
        lines.append(f"  {s['id']:<11} {s['status']:<8} {s['summary']}")
    st = doc["selfcheck"]
    lines.append(
        f"false-accept={doc['false_accept']} false-promotion={doc['false_promotion']} "
        f"selfcheck={'ok' if st.get('ok') else ('n/a' if not st['ran'] else 'FAILED')}"
    )
    v = doc["value"]["totals"]
    lines.append(
        f"value (simulated fresh runs): verified gain={v['verified_gain']} "
        f"unresolved keys one-way={v['unresolved_keys_one_way']} loop={v['unresolved_keys_loop']}"
    )
    c = doc["cost"]
    lines.append(
        f"cost: tokens measured={c['tokens']['measured_total']} "
        f"currency unknown in {c['cost_usd']['unknown_cases']} case(s); "
        f"cost per gain={c['cost_per_verified_gain']['value']} "
        f"({c['cost_per_verified_gain']['reason']})"
    )
    lines.append(f"pending action-layer cases (parent-owned): {len(doc['action_pending'])}")
    return "\n".join(lines) + "\n"
