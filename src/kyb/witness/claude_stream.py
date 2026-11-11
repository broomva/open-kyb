"""Inspect the entire Claude NDJSON stream before classifying a process/provider failure.

Only protocol metadata is retained: never response text, tool inputs or diagnostic contents.
Malformed lines do not discard earlier model/tool evidence or permit availability fallback.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from kyb.read.harness import model_family
from kyb.util.hashing import sha256_json
from kyb.witness.envelope import (
    EMPTY,
    LEAF,
    OPAQUE,
    Anomalies,
    Many,
    Map,
    Tagged,
    error_like_key,
    norm_key,
)
from kyb.witness.errors import (
    ROUTING_VERSION,
    WitnessAnomalous,
    WitnessRefused,
    WitnessUnavailable,
)
from kyb.witness.evidence import claude_terminal, diagnostic, diagnostics

_OUTPUT_TOOL = "StructuredOutput"
_TRANSIENT_HTTP = frozenset({429, 500, 503, 504, 529})
# The official SDK exposes assistant error codes and result.api_error_status. Generic
# error_during_execution/max-turns are not evidence of provider availability failure.
_AVAILABILITY_ERRORS = {"rate_limit", "server_error"}
_RESULTS = {"success", "error_during_execution", "error_max_turns", "error_max_budget_usd",
            "error_max_structured_output_retries"}  # fmt: skip

_SYSTEM_COMMON = frozenset({"type", "subtype", "uuid", "session_id"})
_API_RETRY_ALLOWED = frozenset({"error_status"})


def _ok_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def _benign_str(value: Any) -> bool:
    return isinstance(value, str) and len(value) <= 200


def _ui_event(value: Any) -> bool:
    # A dotted UI identifier ("ui.render"), never free text that could hold a reply.
    return isinstance(value, str) and re.fullmatch(r"ui(\.[a-z_]{1,32}){1,3}", value) is not None


_SYSTEM: dict[str, tuple[frozenset, dict]] = {
    "init": (
        frozenset(
            {
                "additional_directories",
                "agents",
                "analytics_disabled",
                "apiKeySource",
                "capabilities",
                "claude_code_version",
                "cwd",
                "fast_mode_disabled_reason",
                "fast_mode_state",
                "mcp_servers",
                "messaging_socket_path",
                "model",
                "original_cwd",
                "output_style",
                "per_turn_effort_active",
                "permissionMode",
                "plugins",
                "product_feedback_disabled",
                "skills",
                "slash_commands",
                "tools",
                "view_mode",
            }
        ),
        {"mcp_servers": lambda v: v == []},
    ),  # fmt: skip
    "status": (
        frozenset({"status", "permissionMode"}),
        {"status": lambda v: v in (None, "requesting", "compacting")},
    ),
    "thinking_tokens": (
        frozenset({"estimated_tokens", "estimated_tokens_delta"}),
        {"estimated_tokens": _ok_int, "estimated_tokens_delta": _ok_int},
    ),
    "hook_started": (
        frozenset({"hook_id", "hook_name", "hook_event"}),
        {"hook_id": _benign_str, "hook_name": _benign_str, "hook_event": _benign_str},
    ),
    "hook_response": (
        frozenset(
            {
                "hook_id",
                "hook_name",
                "hook_event",
                "outcome",
                "exit_code",
                "output",
                "stdout",
                "stderr",
            }
        ),
        {
            "hook_id": _benign_str,
            "hook_name": _benign_str,
            "hook_event": _benign_str,
            "outcome": lambda v: v == "success",
            "exit_code": lambda v: v == 0 and type(v) is int,
            "output": lambda v: v == "",
            "stdout": lambda v: v == "",
            "stderr": lambda v: v == "",
        },
    ),  # fmt: skip
    # Claude Code (2.1.314+) opens every stream with a UI repaint hint. It is emitted
    # before init, carries no model output, tool call or usage, and its only field is the
    # event identifier, so it is safe as a closed shape; any other key is still an anomaly.
    "ui_invalidate": (frozenset({"event"}), {"event": _ui_event}),
    "api_retry": (
        frozenset({"attempt", "max_retries", "retry_delay_ms", "error_status"}),
        {
            "attempt": _ok_int,
            "max_retries": _ok_int,
            "retry_delay_ms": lambda v: type(v) in (int, float) and v >= 0,
        },
    ),  # fmt: skip
}
_FRAME_KEYS: dict[str, frozenset] = {
    "system": _SYSTEM_COMMON,
    "assistant": frozenset({"type", "message", "parent_tool_use_id", "request_id", "session_id",
                            "thinking_display", "thinking_duration_ms", "timestamp", "uuid",
                            "wire_tool_inputs", "error", "error_message", "aborted"}),
    "user": frozenset({"type", "message", "parent_tool_use_id", "session_id", "timestamp",
                       "tool_use_result", "tool_result_meta", "uuid"}),
    "rate_limit_event": frozenset({"type", "rate_limit_info", "session_id", "uuid"}),
    "result": frozenset({"type", "subtype", "api_error_status", "duration_api_ms", "duration_ms",
                         "fast_mode_disabled_reason", "fast_mode_state", "first_content_frame_ms",
                         "is_error", "modelUsage", "model_usage_since_last_result", "num_turns",
                         "permission_denials", "queued_turn_count", "result", "result_index",
                         "safety_stops", "session_id", "stop_reason", "structured_output",
                         "subagent_stats", "terminal_reason", "time_to_request_ms",
                         "total_cost_usd", "ttft_ms", "ttft_stream_ms", "usage", "uuid",
                         "deferred_tool_use", "errors", "error", "error_code", "status_code",
                         "code", "status", "message", "error_message"}),
}  # fmt: skip
_RESULT_KEYS = _FRAME_KEYS["result"]
_MESSAGE_KEYS = frozenset({"stop_reason", "type", "usage", "context_management", "diagnostics",
                           "input_transformations", "container", "id", "content", "stop_details",
                           "model", "role", "stop_sequence", "error", "error_message"})  # fmt: skip
_RATE_KEYS = frozenset({"status", "resetsAt", "rateLimitType", "utilization", "overageStatus",
                        "overageDisabledReason", "isUsingOverage", "unifiedWindows",
                        "overageEnabled", "overage_status", "overage_disabled_reason",
                        "surpassedThreshold", "overageResetsAt"})  # fmt: skip
_OVERAGE_REASONS = frozenset({"overage_not_provisioned", "org_level_disabled",
                              "org_level_disabled_until", "seat_tier_level_disabled",
                              "member_level_disabled", "seat_tier_zero_credit_limit",
                              "group_zero_credit_limit", "member_zero_credit_limit",
                              "org_service_level_disabled", "org_service_zero_credit_limit",
                              "no_limits_configured", "unknown"})  # fmt: skip


_CACHE = {"ephemeral_5m_input_tokens": LEAF, "ephemeral_1h_input_tokens": LEAF}
_ITERATION = {
    "input_tokens": LEAF,
    "output_tokens": LEAF,
    "cache_read_input_tokens": LEAF,
    "cache_creation_input_tokens": LEAF,
    "cache_creation": _CACHE,
    "type": LEAF,
}
_USAGE = {"input_tokens": LEAF, "cache_creation_input_tokens": LEAF,
          "cache_read_input_tokens": LEAF, "cache_creation": _CACHE, "output_tokens": LEAF,
          "output_tokens_details": {"thinking_tokens": LEAF},
          "server_tool_use": {"web_search_requests": LEAF, "web_fetch_requests": LEAF},
          "service_tier": LEAF, "inference_geo": LEAF, "speed": LEAF,
          "iterations": Many(_ITERATION), "fallback_credit": LEAF}  # fmt: skip
_BLOCKS = Tagged("type", {
    "thinking": {"type": LEAF, "thinking": LEAF, "signature": LEAF},
    "redacted_thinking": {"type": LEAF, "data": LEAF},
    "text": {"type": LEAF, "text": LEAF},
    "tool_use": {"type": LEAF, "id": LEAF, "name": LEAF, "input": OPAQUE,
                 "caller": {"type": LEAF}},
})  # fmt: skip
_MODEL_USAGE = {k: LEAF for k in ("inputTokens", "outputTokens", "cacheReadInputTokens",
                                  "cacheCreationInputTokens", "webSearchRequests", "costUSD",
                                  "contextWindow", "maxOutputTokens", "thinkingTokens",
                                  "canonicalModel", "provider", "costBasis",
                                  # Per-TTL split of cacheCreationInputTokens, the
                                  # billing counters usage.cache_creation already carries; a
                                  # scalar leaf, never content (a nested value is still flagged)
                                  "cacheCreationEphemeral5mInputTokens",
                                  "cacheCreationEphemeral1hInputTokens")}  # fmt: skip
_COMMON = {"type": LEAF, "subtype": LEAF, "uuid": LEAF, "session_id": LEAF}
_INIT = {
    **{k: LEAF for k in _SYSTEM["init"][0]},
    "additional_directories": Many(LEAF), "agents": Many(LEAF), "capabilities": Many(LEAF),
    "skills": Many(LEAF), "slash_commands": Many(LEAF), "tools": Many(LEAF),
    "mcp_servers": Many(LEAF),
    "plugins": Many({"name": LEAF, "path": LEAF, "source": LEAF}),
}  # fmt: skip
_SYSTEM_SPECS: dict[str, dict] = {
    sub: {**_COMMON, **{k: LEAF for k in keys}} for sub, (keys, _) in _SYSTEM.items()
}
_SYSTEM_SPECS["init"] = {**_COMMON, **_INIT}
_OPEN = {k: OPAQUE for k in ("errors", "error", "error_code", "status_code", "code", "status",
                             "message", "error_message", "deferred_tool_use", "result",
                             "structured_output", "permission_denials")}  # fmt: skip
_FRAME_SPECS: dict[str, dict] = {
    "assistant": {
        **{k: LEAF for k in _FRAME_KEYS["assistant"]},
        "wire_tool_inputs": OPAQUE,
        "message": {
            "model": LEAF, "id": LEAF, "type": LEAF, "role": LEAF, "stop_reason": LEAF,
            "stop_sequence": LEAF, "usage": _USAGE, "content": Many(_BLOCKS),
            "container": EMPTY, "diagnostics": EMPTY, "context_management": EMPTY,
            "input_transformations": EMPTY, "stop_details": EMPTY,
            "error": LEAF, "error_message": LEAF,
        },
    },
    "user": {
        **{k: LEAF for k in _FRAME_KEYS["user"]},
        "message": {"role": LEAF, "content": Many({"tool_use_id": LEAF, "type": LEAF,
                                                   "content": LEAF, "is_error": LEAF})},
        "tool_result_meta": Many({"id": LEAF, "permission_decision": {"decision": LEAF,
                                                                      "source": LEAF}}),
    },
    "rate_limit_event": {
        **{k: LEAF for k in _FRAME_KEYS["rate_limit_event"]},
        "rate_limit_info": {
            **{k: LEAF for k in _RATE_KEYS},
            "unifiedWindows": Map({"utilization": LEAF, "resetsAt": LEAF, "status": LEAF,
                                   "surpassedThreshold": LEAF}),
        },
    },
    "result": {
        **{k: LEAF for k in _RESULT_KEYS},
        **_OPEN,
        "usage": _USAGE,
        "modelUsage": Map(_MODEL_USAGE),
        "model_usage_since_last_result": Map(_MODEL_USAGE),
        "subagent_stats": {
            "spawned": LEAF, "started_in_background": LEAF, "max_depth": LEAF,
            "spawned_by_subagents": LEAF, "completed": LEAF, "failed": LEAF,
            "requested": {"background": LEAF, "foreground": LEAF, "unset": LEAF},
            "killed": {"parent": LEAF, "user": LEAF, "system": LEAF},
            "refused": {"depth_limit": LEAF, "concurrency_limit": LEAF, "budget": LEAF},
            "by_type": Map(LEAF),
        },
    },
}  # fmt: skip


def stream_bytes(value: bytes | str | None) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    raise WitnessRefused("Claude stream has an invalid transport type")


def _object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate JSON key")
        out[key] = value
    return out


def _constant(_):
    raise ValueError("non-finite JSON number")


@dataclass
class ClaudeStream:
    schema: dict
    validate: Callable[[Any, dict], None]
    workdir_unchanged: bool
    workdir: dict = field(default_factory=dict)
    models: set[str] = field(default_factory=set)
    tools: list[str] | None = None
    called: list[str] = field(default_factory=list)
    provider_errors: list[str] = field(default_factory=list)
    assistant_diagnostics: list[dict] = field(default_factory=list)
    other_diagnostics: list[dict] = field(default_factory=list)
    result_metadata: list[dict] = field(default_factory=list)
    results: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    refusals: list[str] = field(default_factory=list)
    inits: int = 0
    actual_assistants: int = 0
    nonblank_lines: int = 0
    schema_values: int = 0
    schema_checks: list[dict] = field(default_factory=list)
    schema_tool_ids: set[str] = field(default_factory=set)
    rate_limit_events: list[dict] = field(default_factory=list)
    final_output_hash: str | None = None
    requested_model: str | None = None
    provider: str = "claude-p"
    anomalies: Anomalies = field(init=False)

    def __post_init__(self) -> None:
        self.anomalies = Anomalies(self.provider, self.requested_model)

    def refuse(self, reason: str) -> None:
        if reason not in self.refusals:
            self.refusals.append(reason)

    def schema_value(self, value: Any, *, source: str) -> None:
        self.schema_values += 1
        valid = True
        try:
            self.validate(value, self.schema)
        except Exception:
            valid = False
            if source == "tool":
                # The CLI rejects a schema-invalid StructuredOutput call and the model may submit
                # again. A superseded invalid attempt is recorded as an anomaly: the reply never
                # verifies and the port's bounded self-heal retries.
                self.anomalies.add("assistant.tool_use.StructuredOutput.input", "schema_invalid",
                                   sha256_json(value), "assistant", "unknown")  # fmt: skip
            else:
                self.refuse("schema_invalid")
        self.schema_checks.append(
            {"valid": valid, "value_hash": sha256_json(value), "source": source}
        )

    def consume(self, raw: bytes) -> None:
        for line in raw.splitlines():
            if not line.strip():
                continue
            self.nonblank_lines += 1
            try:
                event = json.loads(line.decode("utf-8"), object_pairs_hook=_object,
                                   parse_constant=_constant)  # fmt: skip
                # Escaped unpaired surrogates must not enter UTF-8 failure metadata.
                json.dumps(event, ensure_ascii=False, allow_nan=False).encode("utf-8")
            except Exception:
                self.refuse("stream_json_or_encoding_invalid")
                continue
            if not isinstance(event, dict):
                self.refuse("stream_event_not_object")
                continue
            try:
                self.observe(event)
            except Exception:
                # Keep inspecting after an invalid frame; no earlier proof is discarded.
                self.refuse("event_structure_invalid")
        if not self.workdir_unchanged:
            self.refuse("workdir_changed")
        if self.nonblank_lines:
            if self.inits != 1:
                self.refuse("init_evidence_incomplete")
            if len(self.results) != 1:
                self.refuse("result_evidence_incomplete")
        if len(self.models) > 1:
            self.refuse("mixed_assistant_models")
        # An invalid tool attempt is only a retry when a valid final output follows it.
        if any(not c["valid"] for c in self.schema_checks) and not any(
            c["valid"] and c["source"] == "final" for c in self.schema_checks
        ):
            self.refuse("schema_invalid")

    def error_keys(
        self,
        node: Any,
        *,
        system: bool,
        allowed: frozenset = frozenset(),
        path: str = "",
        frame: str = "",
    ) -> bool:
        """Any error-like key (normalized: strip, NFKC, zero-width removed, case-insensitive) at
        any depth; in system frames also any key equal to `message`. Records an `error`-severity
        anomaly and returns True when one is present."""
        if isinstance(node, list):
            return any([self.error_keys(i, system=system, allowed=allowed,
                                        path=f"{path}[{n}]", frame=frame)
                        for n, i in enumerate(node)])  # fmt: skip
        if not isinstance(node, dict):
            return False
        hit = False
        for key, value in node.items():
            if key not in allowed and (
                error_like_key(key) or (system and norm_key(key) == "message")
            ):
                hit = True
                self.other_diagnostics.extend(diagnostics([value]))
                self.anomalies.add(f"{path}.{key}", key, value, frame or path, "error")
            if self.error_keys(value, system=system, allowed=allowed,
                               path=f"{path}.{key}", frame=frame):  # fmt: skip
                hit = True
        return hit

    def scan(self, event: dict) -> None:
        """Record every unknown key/value of the frame (recursively) as an envelope anomaly."""
        kind, subtype = event.get("type"), event.get("subtype")
        if kind == "system":
            label = subtype if isinstance(subtype, str) else "system"
            spec = _SYSTEM_SPECS.get(subtype) if isinstance(subtype, str) else None
            if spec is None:
                self.anomalies.add(f"{label}.subtype", "subtype", subtype, label)
                spec = _COMMON
        else:
            label = kind if isinstance(kind, str) else "frame"
            spec = _FRAME_SPECS.get(kind) if isinstance(kind, str) else None
            if spec is None:
                self.anomalies.add(f"{label}.type", "type", kind, label)
                spec = {"type": LEAF}
        self.anomalies.scan(event, spec, label, label)
        if self.anomalies.has_error:
            self.refuse("explicit_error_signal")  # an error-like key/value is an error, as before

    def system_allowlist(self, event: dict) -> None:
        """Every system subtype has an explicit accepted shape; anything else refuses."""
        subtype = event.get("subtype")
        spec = _SYSTEM.get(subtype) if isinstance(subtype, str) else None
        if spec is None:
            if self.error_keys(event, system=True, path=str(subtype), frame=str(subtype)):
                self.refuse("system_frame_error_key")
            return
        _keys, values = spec
        if subtype == "api_retry":
            status = event.get("error_status")
            if status is not None and not (type(status) is int and status in _TRANSIENT_HTTP):
                self.other_diagnostics.extend(diagnostics({"error_status": status}))
                self.refuse("system_frame_error_status")
        if self.error_keys(
            event,
            system=True,
            allowed=_API_RETRY_ALLOWED if subtype == "api_retry" else frozenset(),
            path=subtype,
            frame=subtype,
        ):
            self.refuse("system_frame_error_key")
        for key, check in values.items():
            if key in event and not check(event[key]):
                if key == "mcp_servers":
                    self.refuse("system_frame_value_unlisted")  # confinement breach
                else:
                    self.anomalies.add(f"{subtype}.{key}", key, event[key], subtype)

    def observe(self, event: dict) -> None:
        kind = event.get("type")
        safe_kind = (
            kind if kind in ("system", "assistant", "result", "user", "rate_limit_event") else None
        )
        subtype = event.get("subtype")
        safe_subtype = (
            subtype
            if subtype
            in (
                *_RESULTS,
                "init",
                "status",
                "api_retry",
                "hook_started",
                "hook_response",
                "ui_invalidate",
            )
            else None
        )
        self.events.append({"type": safe_kind, "subtype": safe_subtype})
        if not isinstance(kind, str):
            self.refuse("event_type_invalid")
            return
        if kind not in ("assistant", "result"):
            for key in ("error", "errors", "error_code", "error_message"):
                if key in event:
                    self.other_diagnostics.extend(diagnostics({key: event[key]}))
                    self.refuse("unexpected_stream_diagnostic")
        self.scan(event)
        if kind == "system":
            if not isinstance(event.get("subtype"), str):
                self.refuse("system_subtype_invalid")
            self.system_allowlist(event)
            if event.get("subtype") == "init":
                self.inits += 1
                tools = event.get("tools")
                if not isinstance(tools, list) or any(not isinstance(t, str) for t in tools):
                    self.refuse("advertised_tools_invalid")
                else:
                    # Retain unsafe names even if a later init advertises a clean list.
                    self.tools = list(dict.fromkeys([*(self.tools or []), *tools]))
                    if set(tools) - {_OUTPUT_TOOL}:
                        self.refuse("data_tools_advertised")
            elif "tools" in event:
                self.refuse("unexpected_tools_declaration")
                tools = event["tools"]
                if isinstance(tools, list) and all(isinstance(t, str) for t in tools):
                    self.tools = list(dict.fromkeys([*(self.tools or []), *tools]))
            return
        if kind == "assistant":
            self.assistant(event)
        elif kind == "result":
            self.result(event)
        elif kind == "user":
            message = event.get("message")
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, list) or any(
                not isinstance(b, dict) or b.get("type") != "tool_result" for b in content
            ):
                self.refuse("user_stream_evidence_invalid")
            elif any(
                not isinstance(b.get("tool_use_id"), str)
                or b["tool_use_id"] not in self.schema_tool_ids
                for b in content
            ):
                self.refuse("unmatched_tool_result")
        elif kind == "rate_limit_event":
            info = event.get("rate_limit_info")
            if self.error_keys(info, system=False, path="rate_limit_event.rate_limit_info",
                               frame="rate_limit_event"):  # fmt: skip
                self.refuse("rate_limit_error_key")
            if isinstance(info, dict) and (
                info.get("overageDisabledReason", "overage_not_provisioned") not in _OVERAGE_REASONS
                or "overageStatus" in info
                and info["overageStatus"] not in (None, "allowed", "allowed_warning", "rejected")
            ):
                self.anomalies.add("rate_limit_event.rate_limit_info", "overage", info,
                                   "rate_limit_event")  # fmt: skip
            status = info.get("status") if isinstance(info, dict) else None
            overage = info.get("overage_status") if isinstance(info, dict) else None
            rate_diagnostics = (
                diagnostics([info["overage_disabled_reason"]])
                if isinstance(info, dict) and info.get("overage_disabled_reason") not in (None, "")
                else []
            )
            self.rate_limit_events.append(
                {
                    "status": status
                    if status in ("allowed", "allowed_warning", "rejected")
                    else None,
                    "hash": sha256_json(info),
                    "overage_status": overage
                    if overage in (None, "allowed", "allowed_warning", "rejected")
                    else None,
                    "diagnostics": rate_diagnostics,
                }
            )
            if status not in ("allowed", "allowed_warning", "rejected"):
                self.anomalies.add("rate_limit_event.rate_limit_info.status", "status", status,
                                   "rate_limit_event")  # fmt: skip
            if overage not in (None, "allowed", "allowed_warning", "rejected"):
                self.anomalies.add("rate_limit_event.rate_limit_info.overage_status",
                                   "overage_status", overage, "rate_limit_event")  # fmt: skip
        # Unknown frame types were recorded as anomalies by `scan`; they never verify anything.

    def assistant(self, event: dict) -> None:
        if self.inits != 1:
            self.refuse("assistant_before_single_init")
        if event.get("parent_tool_use_id") is not None:
            self.refuse("unexpected_nested_assistant")
        message = event.get("message")
        if not isinstance(message, dict):
            self.refuse("assistant_message_invalid")
            return
        for holder in (event, message):
            if "error_message" in holder:
                self.assistant_diagnostics.extend(diagnostics([holder["error_message"]]))
        model = message.get("model")
        if not isinstance(model, str) or not model.strip():
            self.refuse("assistant_model_missing_or_invalid")
        elif model != "<synthetic>":
            self.actual_assistants += 1
            self.models.add(model)
            if model_family(model) != "anthropic":
                self.refuse("assistant_model_family_invalid")
        errors = [e for e in (event.get("error"), message.get("error")) if e is not None]
        if len(errors) == 2 and errors[0] != errors[1]:
            self.refuse("conflicting_assistant_errors")
        for error in errors:
            self.assistant_diagnostics.append(diagnostic(error))
            if isinstance(error, str) and error in _AVAILABILITY_ERRORS:
                self.provider_errors.append(error)
            else:
                self.refuse("unrecognized_assistant_error")
        if model == "<synthetic>" and not errors:
            self.refuse("unrecognized_synthetic_assistant")
        if event.get("aborted") or message.get("stop_reason") not in (None, "end_turn", "tool_use"):
            self.refuse("assistant_incomplete_or_refused")
        content = message.get("content")
        if not isinstance(content, list):
            self.refuse("assistant_content_invalid")
            return
        for block in content:
            if not isinstance(block, dict):
                self.refuse("assistant_block_invalid")
                continue
            kind = block.get("type")
            if not isinstance(kind, str):
                self.refuse("assistant_block_type_invalid")
                continue
            if kind == "tool_use":
                name = block.get("name")
                if isinstance(name, str):
                    self.called.append(name)
                if name != _OUTPUT_TOOL:
                    self.refuse("data_tool_called")
                else:
                    self.schema_value(block.get("input"), source="tool")
                    identifier = block.get("id")
                    if not isinstance(identifier, str) or not identifier:
                        self.refuse("schema_tool_id_missing")
                    else:
                        self.schema_tool_ids.add(identifier)
            elif kind in {"text", "thinking", "redacted_thinking"}:
                key = {"text": "text", "thinking": "thinking", "redacted_thinking": "data"}[kind]
                if errors or model == "<synthetic>":
                    self.assistant_diagnostics.append(diagnostic(block.get(key)))
                if not isinstance(block.get(key), str):
                    self.refuse("assistant_text_block_invalid")
            # an unlisted block type is an envelope anomaly (recorded by `scan`), not a refusal

    def result(self, event: dict) -> None:
        if self.inits != 1:
            self.refuse("result_before_single_init")
        self.results.append(event)
        subtype = event.get("subtype")
        errors = event.get("errors", [])
        if errors is None and subtype == "success":
            errors = []
        summary = {
            k: event.get(k)
            for k in (
                "subtype",
                "is_error",
                "stop_reason",
                "terminal_reason",
                "api_error_status",
                "total_cost_usd",
                "usage",
            )
        }
        if summary["api_error_status"] is not None and type(summary["api_error_status"]) is not int:
            errors = (
                [*errors, {"invalid_api_status": summary["api_error_status"]}]
                if isinstance(errors, list)
                else errors
            )
            summary["api_error_status"] = None
        summary["errors"] = diagnostics(errors)
        result = event.get("result")
        summary["result_diagnostics"] = (
            [diagnostic(result)]
            if event.get("is_error") is True and result not in (None, "")
            else []
        )
        if result is not None and not isinstance(result, str):
            summary["result_diagnostics"].extend(diagnostics(result))
        for key in (
            "error",
            "error_code",
            "status_code",
            "code",
            "status",
            "message",
            "error_message",
        ):
            if key in event:
                summary["result_diagnostics"].extend(diagnostics({key: event[key]}))
        summary["final_output_hash"] = (
            sha256_json(event["structured_output"]) if "structured_output" in event else None
        )
        self.result_metadata.append(summary)
        if isinstance(subtype, str) and subtype not in _RESULTS:
            self.anomalies.add("result.subtype", "subtype", subtype, "result")
        if not isinstance(subtype, str) or type(event.get("is_error")) is not bool:
            self.refuse("result_status_invalid")
        if subtype in _RESULTS and subtype != "success" and event.get("is_error") is not True:
            self.refuse("result_status_inconsistent")
        if event.get("terminal_reason") not in (None, "completed", "model_error", "api_error"):
            self.refuse("result_not_completed")
        if event.get("is_error") is False and (
            event.get("terminal_reason") == "model_error"
            or event.get("api_error_status") is not None
        ):
            self.refuse("result_status_inconsistent")
        if event.get("stop_reason") not in (None, "end_turn", "tool_use"):
            self.refuse("result_incomplete_or_refused")
        if "structured_output" in event:
            self.schema_value(event["structured_output"], source="final")
            self.final_output_hash = sha256_json(event["structured_output"])
        elif event.get("subtype") == "success" and not event.get("is_error"):
            self.refuse("schema_output_missing")
        denials = event.get("permission_denials")
        if denials is None:
            denials = []
        if not isinstance(denials, list) or denials:
            self.refuse("permission_denial_evidence")
        if event.get("deferred_tool_use") is not None:
            self.refuse("deferred_tool_evidence")
        if event.get("api_error_status") is not None and type(event["api_error_status"]) is not int:
            self.refuse("provider_status_invalid")
        usage, cost = event.get("usage"), event.get("total_cost_usd")
        if usage is not None and not isinstance(usage, dict):
            self.refuse("usage_metadata_invalid")
        if cost is not None and (
            type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0
        ):
            self.refuse("cost_metadata_invalid")
        if isinstance(usage, dict):
            for key in (
                "input_tokens",
                "cache_read_input_tokens",
                "cache_creation_input_tokens",
                "output_tokens",
            ):
                if key in usage and (type(usage[key]) is not int or usage[key] < 0):
                    self.refuse("usage_metadata_invalid")

    def provenance(self) -> dict:
        out = self.results[0] if len(self.results) == 1 else {}
        api_status = (
            out.get("api_error_status") if type(out.get("api_error_status")) is int else None
        )
        return {"validation_version": ROUTING_VERSION, "tools": self.tools, "workdir": self.workdir,
                "called_tools": self.called, "reported_models": sorted(self.models),
                "reported_model": next(iter(self.models)) if len(self.models) == 1 else None,
                "no_data_tools": self.inits == 1 and self.tools is not None
                    and not (set(self.tools) | set(self.called)) - {_OUTPUT_TOOL}
                    and self.workdir_unchanged and not self.refusals,
                "emitted_events": self.events, "nonblank_lines": self.nonblank_lines,
                "refusal_evidence": self.refusals, "provider_errors": self.provider_errors,
                "assistant_diagnostics": self.assistant_diagnostics,
                "other_diagnostics": self.other_diagnostics,
                "schema_checks": self.schema_checks, "final_output_hash": self.final_output_hash,
                "rate_limit_events": self.rate_limit_events,
                "result_evidence": self.result_metadata,
                "schema_values_checked": self.schema_values, "session_id": out.get("session_id"),
                "cost_usd": out.get("total_cost_usd"), "usage": out.get("usage"),
                "api_error_status": api_status,
                "envelope_anomalies": self.anomalies.stamp(
                    next(iter(self.models)) if len(self.models) == 1 else None)}  # fmt: skip

    def finish(self, returncode: int | None, *, timed_out: bool = False,
               launch_unavailable: bool = False,
               transport_evidence: dict | None = None) -> tuple[dict, dict]:  # fmt: skip
        # No process/provider outcome can override any evidence gathered above.
        if self.refusals:
            raise WitnessRefused("Claude emitted refusal evidence: " + ", ".join(self.refusals))
        if self.anomalies.items:
            # Unknown shapes never verify a claim and are never plain availability.
            raise WitnessAnomalous("Claude stream carries unknown shapes", self.anomalies.items)
        outcome = claude_terminal(
            {
                **(transport_evidence or {}),
                **self.provenance(),
                "returncode": returncode,
                "timed_out": timed_out,
            }
        )
        if outcome == "unavailable":
            raise WitnessUnavailable("Claude reported consistent recognized no-answer availability")
        out = self.results[0]
        return out["structured_output"], out.get("usage") or {}
