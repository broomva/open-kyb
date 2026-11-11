"""Optional MCP Apps extension; no assumptions about a named harness's UI support."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlencode

from mcp.server.apps import client_supports_apps
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from kyb.dashboard import data
from kyb.dashboard.browser import app_html, serve
from kyb.dashboard.contracts import (
    CaseMembership,
    ChatIntent,
    DashboardSnapshot,
    OpenSummary,
    RepairHandoff,
)

URI = "ui://kyb/orchestration-dashboard"
# MCP Apps visibility: the embedded app may only inspect and prepare. Opening, archive/restore and
# every core action (kyb.mcp_server.MODEL_ONLY) stay with the model and its native confirmation.
APP = {"ui": {"visibility": ["model", "app"]}}
MODEL = {"ui": {"visibility": ["model"]}}
MIME = "text/html;profile=mcp-app"


def register(server: MCPServer, resolve_root: Callable[[], Path]) -> None:
    def root_or_refuse() -> Path:
        # "no workspace" is a RuntimeError, outside data.EXPECTED: unwrapped, the SDK masks it and
        # a new user sees a bare "Error executing tool" instead of "call init_workspace first"
        try:
            return resolve_root()
        except RuntimeError as exc:
            raise ToolError(str(exc)) from exc

    @server.resource(
        URI,
        name="kyb_dashboard",
        mime_type=MIME,
        meta={"ui": {"prefersBorder": False, "csp": {"connectDomains": [], "resourceDomains": []}}},
    )
    def dashboard_resource() -> str:
        return app_html()

    @server.tool(
        meta={"ui": {"resourceUri": URI, "visibility": ["model"]}},
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    )
    def open_dashboard(
        ctx: Context,
        case: str | None = None,
        run: str | None = None,
        surface: Literal["auto", "inline", "browser"] = "auto",
    ) -> OpenSummary:
        """Open open-kyb's responsive review dashboard. Read-only: no plan/run/provider/signature.

        Inline requires negotiated MCP Apps; otherwise return a local browser link.
        A link returned is not a browser opened. Only the server's discovered workspace is exposed.
        """
        try:
            root = root_or_refuse()
            snap = data.snapshot(root, case, run)
            inline = client_supports_apps(ctx)
            browser = serve(root).url
            args = {k: v for k, v in {"case": case, "run": run}.items() if v}
            if args:
                browser += "?" + urlencode(args)
            opened = {
                "presentation": "inline"
                if inline and surface != "browser"
                else "inline_unavailable"
                if surface == "inline"
                else "browser_link",
                "resource_uri": URI,
                "browser_url": browser,
                "capabilities": {
                    "inline": inline,
                    "native_confirmation": "harness_owned_not_inferred",
                },
                "message": "The dashboard cannot authorize work or sign. "
                "Prepared plans require a native chat confirmation.",
            }
            chosen = snap.get("selected") or {}
            return OpenSummary.model_validate(
                {
                    "workspace_context": snap["workspace_context"],
                    "case": chosen.get("case", case),
                    "run": chosen.get("run", run),
                    "process": chosen.get("process"),
                    "cases": snap["cases"],
                    "errors": snap["errors"],
                    "execution": snap["execution"],
                    **opened,
                }
            )
        except data.EXPECTED as exc:
            raise ToolError(str(exc)) from exc

    @server.tool(meta=APP, annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False))
    def dashboard_snapshot(
        case: str | None = None, run: str | None = None, workspace_id: str | None = None
    ) -> DashboardSnapshot:
        """Read receipted queue/evidence/current assurance for an explicit pinned run."""
        try:
            root = root_or_refuse()
            data.check_workspace(root, workspace_id)
            return DashboardSnapshot.model_validate(data.snapshot(root, case, run))
        except data.EXPECTED as exc:
            raise ToolError(str(exc)) from exc

    @server.tool(
        meta=APP,
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    )
    def prepare_dashboard_action(
        case: str,
        run: str,
        snapshot_id: str,
        max_attempts: Annotated[int, Field(strict=True, ge=1)] = 2,
        max_elapsed_seconds: Annotated[int, Field(strict=True, ge=1)] = 900,
        workspace_id: str | None = None,
    ) -> RepairHandoff:
        """Prepare a checked bounded repair plan only. No run, inference, execution or signing.

        Return a chat handoff. It is NOT authorization. Use kyb-dashboard to ask the user natively.
        """
        try:
            return RepairHandoff.model_validate(
                data.prepare(
                    root_or_refuse(),
                    case,
                    run,
                    snapshot_id,
                    max_attempts,
                    max_elapsed_seconds,
                    workspace_id,
                )
            )
        except data.EXPECTED as exc:
            raise ToolError(str(exc)) from exc

    @server.tool(meta=APP, annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False))
    def dashboard_handoff(
        case: str, plan_id: str, workspace_id: str | None = None
    ) -> RepairHandoff:
        """Re-read a staged request and allocation receipt. Never treats a UI message as assent.

        Follow kyb-dashboard: native AskUserQuestion first; on explicit assent revalidate this
        request, then existing checked actions. A stale or allocated request must not redispatch.
        """
        try:
            return RepairHandoff.model_validate(
                data.handoff(root_or_refuse(), case, plan_id, workspace_id)
            )
        except data.EXPECTED as exc:
            raise ToolError(str(exc)) from exc

    @server.tool(meta=APP, annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False))
    def dashboard_intent(
        case: str,
        workspace_id: str,
        kind: Literal[
            "question", "continue_review", "review_recommendation", "archive_case", "restore_case"
        ],
        run: str | None = None,
        question: str = "",
        snapshot_id: str | None = None,
        case_state_hash: str | None = None,
    ) -> ChatIntent:
        """Validate a case-bound question/action for the chat agent; no inference or execution.

        Questions use existing ask/record_answer. Work needs a native confirmation;
        signing retains its separate officer flow. Keep the exact workspace/case/run.
        """
        try:
            return ChatIntent.model_validate(
                data.intent(
                    root_or_refuse(),
                    case,
                    run,
                    workspace_id,
                    kind,
                    question,
                    snapshot_id,
                    case_state_hash,
                )
            )
        except data.EXPECTED as exc:
            raise ToolError(str(exc)) from exc

    @server.tool(
        meta=MODEL,
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    )
    def archive_case(
        case: str, workspace_id: str, expected_event_hash: str | None
    ) -> CaseMembership:
        """After explicit native Archive case assent, remove the case from active queues.

        Preserve every evidence/run/decision/signature and append audit history. Does not stop
        work. The UI message is NOT assent; follow kyb-dashboard and revalidate before this call.
        """
        try:
            return data.change_case_membership(
                root_or_refuse(), case, workspace_id, expected_event_hash, archived=True
            )
        except data.EXPECTED as exc:
            raise ToolError(str(exc)) from exc

    @server.tool(
        meta=MODEL,
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    )
    def restore_case(
        case: str, workspace_id: str, expected_event_hash: str | None
    ) -> CaseMembership:
        """After explicit native Restore case assent, return a case to active queues.

        Appends history only; starts no review and changes no recommendation or signature.
        """
        try:
            return data.change_case_membership(
                root_or_refuse(), case, workspace_id, expected_event_hash, archived=False
            )
        except data.EXPECTED as exc:
            raise ToolError(str(exc)) from exc
