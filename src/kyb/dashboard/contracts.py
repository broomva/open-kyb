"""Versioned UI contracts shared by MCP and CLI adapters. Core records keep their own types."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kyb.contracts.b4 import EvidenceItem
from kyb.loop.models import LoopBudget, LoopPlan
from kyb.workspace.cases import CaseLifecycle

VERSION = "kyb-dashboard-v2"


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class WorkspaceContext(Contract):
    id: str
    label: str
    cli_path: str
    scope: Literal["one_workspace_many_cases"] = "one_workspace_many_cases"


class ProcessState(Contract):
    state: Literal["not_started", "needs_work", "review_complete", "signed", "unavailable"]
    step: str
    label: str
    next_action: str


class Recommendation(Contract):
    decision_id: str
    status: Literal["APPROVED", "REJECTED", "MANUAL_REVIEW"]
    entity_name: str
    risk_score: float
    rules_version: str
    signature_state: Literal["signed", "pending_signature"]
    emitted_at: str
    findings: list[dict[str, Any]]


class CaseSummary(Contract):
    case: str
    display_name: str
    runs: list[str]
    latest_run: str | None
    process: ProcessState
    lifecycle: CaseLifecycle
    recommendation: Recommendation | None = None


class CaseMembership(Contract):
    workspace_id: str
    case: str
    lifecycle: CaseLifecycle


class AgentObservation(Contract):
    doc: str
    document: str
    claims: int = Field(ge=0)
    verified: int = Field(ge=0)
    assurance: Literal["W1", "W2"]
    isolated: bool
    findings: list[str]
    acceptance: float | None
    accuracy: None = None
    drift: None = None
    reader_model: str | None
    witness_model: str | None
    comparator: str | None
    binding: str | None
    coverage_version: str | None
    coverage_total: int = Field(ge=0)
    coverage_matched: int = Field(ge=0)


class Allocation(Contract):
    version: Literal["evidence-loop-v1"]
    case: str
    plan_id: str
    run: str
    source_run: str
    as_of: str


class PlanSummary(Contract):
    plan_id: str
    state: str
    attempt: int
    budget: LoopBudget
    previous_plan: str | None
    allocation: Allocation | None


class CaseDetail(Contract):
    case: str
    run: str
    as_of: str
    historical: bool
    snapshot_id: str
    process: ProcessState
    recommendations: list[Recommendation]
    evidence: list[EvidenceItem]
    agents: list[AgentObservation]
    plans: list[PlanSummary]
    readiness: str
    loop: LoopPlan | None
    retry_budget: LoopBudget | None = None
    errors: list[str]


class TimelineEvent(Contract):
    seq: int
    at: str
    action: str
    label: str
    case: str | None
    run: str | None
    reference: str | None


class Timeline(Contract):
    chain_verified: bool
    events: list[TimelineEvent]
    total: int
    truncated: bool
    error: str | None = None


class Route(Contract):
    id: Literal["current-harness"]
    label: str
    note: str


class DashboardSnapshot(Contract):
    version: Literal["kyb-dashboard-v2"] = VERSION
    workspace: str
    workspace_context: WorkspaceContext
    observed_at: str
    cases: list[CaseSummary]
    queue: dict[str, Any]
    selected: CaseDetail | None
    timeline: Timeline
    errors: list[str]
    read_only: Literal[True] = True
    execution: Literal["harness_native_confirmation"]
    route: Route


class OpenSummary(Contract):
    """What `open_dashboard` returns: small enough for a host's tool-output limit. A full snapshot
    with evidence ran to 71k characters, and Claude Code shows the model the structured content,
    so the summary is all of it. The embedded app reads the pinned snapshot itself through
    `dashboard_snapshot`."""

    version: Literal["kyb-dashboard-open-v1"] = "kyb-dashboard-open-v1"
    workspace_context: WorkspaceContext
    case: str | None
    run: str | None
    process: ProcessState | None
    cases: list[CaseSummary]
    errors: list[str]
    read_only: Literal[True] = True
    execution: Literal["harness_native_confirmation"]
    presentation: Literal["inline", "inline_unavailable", "browser_link"]
    resource_uri: str
    browser_url: str
    capabilities: dict[str, Any]
    message: str


class RetryRequest(Contract):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    case: str
    run: str
    snapshot_id: str
    workspace_id: str | None = None
    max_attempts: int = Field(default=2, ge=1)
    max_elapsed_seconds: int = Field(default=900, ge=1)


class SnapshotQuery(Contract):
    case: str | None = None
    run: str | None = None
    workspace_id: str | None = None

    @model_validator(mode="after")
    def run_has_case(self):
        if self.run and not self.case:
            raise ValueError("Select a case before selecting a review attempt.")
        return self


class IntentRequest(Contract):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    case: str
    workspace_id: str
    run: str | None = None
    kind: Literal[
        "question", "continue_review", "review_recommendation", "archive_case", "restore_case"
    ]
    case_state_hash: str | None = None
    question: str = Field(default="", max_length=2000)
    snapshot_id: str | None = None


class ChatIntent(Contract):
    request_id: str
    workspace_id: str
    case: str
    run: str | None
    kind: Literal[
        "question", "continue_review", "review_recommendation", "archive_case", "restore_case"
    ]
    case_state_hash: str | None
    question: str
    snapshot_id: str | None
    message: str
    authorization: Literal["question_only", "native_question_required", "separate_officer_flow"]


class RepairHandoff(Contract):
    request_id: str
    workspace_id: str
    case: str
    run: str
    plan_id: str
    plan: LoopPlan
    budget: LoopBudget
    expires_at: str
    valid: bool
    allocation: Allocation | None
    state: Literal["allocated", "prepared", "stale", "expired"]
    authorization: Literal["native_question_required"]
    message: str
