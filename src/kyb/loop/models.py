"""Versioned administrative records for the evidence loop."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

VERSION = "evidence-loop-v1"


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LoopBudget(Strict):
    max_attempts: int = Field(default=2, ge=0)
    max_elapsed_seconds: int = Field(default=900, ge=0)
    max_tokens: int | None = Field(default=None, ge=0)
    max_provider_calls: int | None = Field(default=None, ge=0)
    max_cost_usd: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class Gap(Strict):
    gap_id: str
    kind: str
    entity_ids: list[str]
    doc_ids: list[str]
    field: str
    repairable: bool
    reason: str
    evidence_ids: list[str]


class ReconciliationCandidate(Strict):
    left: str
    right: str
    state: Literal["supported", "unresolved", "contradictory"]
    reason: str
    evidence_ids: list[str]
    applied_to_factset: Literal[False] = False


class LoopPlan(Strict):
    version: Literal["evidence-loop-v1"] = VERSION
    plan_id: str
    case: str
    source_run: str
    input_fingerprint: str
    fact_set_hash: str
    previous_plan: str | None
    attempt: int = Field(ge=0)
    state: Literal["ready", "repair", "manual_review", "budget_exhausted", "no_progress", "blocked"]
    gaps: list[Gap]
    candidates: list[ReconciliationCandidate]
    next_actions: list[dict[str, Any]]
    metrics: dict[str, Any]
