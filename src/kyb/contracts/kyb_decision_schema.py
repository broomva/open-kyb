"""Decision and audit contracts for open-kyb."""

from __future__ import annotations

import datetime as dt
import typing as t

import pydantic as pd

type Status = t.Literal["MANUAL_REVIEW", "REJECTED", "APPROVED"]
type Severity = t.Literal["SCORE", "REVIEW", "REJECT"]


class Citation(pd.BaseModel):
    """Source span backing a derived fact or triggered rule."""

    excerpt: t.Annotated[str, pd.Field(max_length=600)]
    locator: str
    document_id: str
    field: str | None = None


class Violation(pd.BaseModel):
    """Triggered policy rule and its cited evidence."""

    evidence: list[Citation]
    description: str
    severity: Severity
    rule_id: str
    score_delta: int = 0


class AuditEvent(pd.BaseModel):
    """Single link in the tamper-evident workspace ledger."""

    hash: str
    prev_hash: str
    output_hash: str
    input_hash: str
    action: str
    component: str
    timestamp: dt.datetime
    seq: int
    detail: dict[str, t.Any] = pd.Field(default_factory=dict)
    cost_usd: float = 0.0
    tokens_out: int = 0
    tokens_in: int = 0


class KYBDecision(pd.BaseModel):
    """Signed decision payload emitted at the end of a review."""

    emitted_at: dt.datetime
    audit_trail: list[AuditEvent]
    rationale: t.Annotated[str, pd.Field(max_length=2000)]
    citations: list[Citation]
    violations: list[Violation]
    risk_score: t.Annotated[int, pd.Field(ge=0, le=100)]
    status: Status
    rules_version: str
    entity_name: str
    entity_id: str
    decision_id: str
    officer_overrides_considered: list[str] = pd.Field(default_factory=list)
