"""Read native Codex 0.164 rollouts without promoting configured models to served models.

This summarizes schema evidence, not authorization. Tool confinement and native hook identity
must still be checked by the caller. Unknown context-bearing records fail closed; unrelated
configuration fields are ignored. Snapshots belong in the audit chain, not mutable path lookups.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

# Code-mode orchestration has no data access of its own. Its executed nested operations appear
# as native item_completed events and must ALSO be enforced by the PreToolUse hook.
WRAPPERS = {("functions", "exec"), ("functions", "wait")}
INERT_RECORDS = {"world_state", "inter_agent_communication_metadata"}
INERT_EVENTS = {"token_count", "thread_settings_applied"}
INERT_ITEMS = {"UserMessage", "AgentMessage", "Reasoning"}
USAGE_NAMES = {"input_tokens": "input_tokens", "output_tokens": "output_tokens",
               "cached_input_tokens": "cache_read_input_tokens",
               "cache_write_input_tokens": "cache_creation_input_tokens",
               "reasoning_output_tokens": "reasoning_output_tokens",
               "total_tokens": "total_tokens"}  # fmt: skip


def _lineage(meta: dict, diagnostics: list[str]) -> tuple[Any, Any, Any, bool]:
    source = meta.get("source")
    subagent = source.get("subagent") if isinstance(source, dict) else None
    spawn = subagent.get("thread_spawn") if isinstance(subagent, dict) else None
    spawn = spawn if isinstance(spawn, dict) else {}
    child = meta.get("id")
    parents = {v for v in (meta.get("parent_thread_id"), spawn.get("parent_thread_id")) if v}
    parent = next(iter(parents)) if len(parents) == 1 else None
    roles = {v for v in (meta.get("agent_type"), meta.get("agent_role"),
                         spawn.get("agent_type"), spawn.get("agent_role")) if v}  # fmt: skip
    role = next(iter(roles)) if len(roles) == 1 else None
    if not isinstance(child, str) or not child or not spawn:
        diagnostics.append("missing native child identity")
    if not isinstance(parent, str) or not parent or parent == child:
        diagnostics.append("missing or conflicting native parent identity")
    if not isinstance(role, str) or not role:
        diagnostics.append("missing or conflicting native agent types")
    history = meta.get("subagent_history_start_ordinal")
    fresh = (history is None or type(history) is int and history == 0) and not meta.get(
        "forked_from_id"
    )
    if not fresh:
        diagnostics.append("child inherited or forked prior history")
    return child, parent, role, fresh


def _input_image(block: Any) -> bool:
    return (isinstance(block, dict) and block.get("type") == "input_image"
            and isinstance(block.get("image_url"), str)
            and set(block) <= {"type", "image_url", "detail"})  # fmt: skip


def _bound_images(content: list, paths: list[str], diagnostics: list[str]) -> tuple[list, bool]:
    """Strip only the CLI's exact caller-bound, consecutive three-block image prefixes."""
    for index, path in enumerate(paths, 1):
        offset = (index - 1) * 3
        marker = f'<image name=[Image #{index}] path="{path}">'
        if (len(content) < offset + 3
                or content[offset] != {"type": "input_text", "text": marker}
                or not _input_image(content[offset + 1])
                or content[offset + 2] != {"type": "input_text", "text": "</image>"}):  # fmt: skip
            diagnostics.append("image wrapper differs from trusted path/index/order")
            return content, False
    # Retain the images for count/schema checks; remove only their verified text wrappers.
    return ([content[i * 3 + 1] for i in range(len(paths))]
            + content[len(paths) * 3:]), True  # fmt: skip


def _prompt(content: Any, diagnostics: list[str]) -> tuple[str | None, int]:
    if isinstance(content, str):
        return content, 0
    if not isinstance(content, list):
        diagnostics.append("invalid user content")
        return None, 0
    parts = []
    images = 0
    for block in content:
        if isinstance(block, dict) and "encrypted_content" in block:
            diagnostics.append("encrypted user content cannot be verified")
            continue
        if _input_image(block):
            images += 1
            continue
        if (not isinstance(block, dict) or not isinstance(block.get("type"), str)
                or block["type"] not in {"input_text", "text"}):  # fmt: skip
            diagnostics.append("unsupported user content block")
        elif not isinstance(block.get("text"), str):
            diagnostics.append("invalid user text")
        else:
            parts.append(block["text"])
    return "".join(parts), images


def _trusted_environment(prompt: str, trusted: str | None) -> bool:
    """Never exempt arbitrary user text on the strength of an environment-looking prefix."""
    if trusted is None or prompt != trusted:
        return False
    try:
        element = ET.fromstring(prompt)
    except ET.ParseError:
        return False
    return (element.tag == "environment_context" and not element.attrib
            and all(c.tag in {"cwd", "shell", "current_date", "timezone", "filesystem"}
                    for c in element))  # fmt: skip


def _direct_tool(payload: dict) -> str | None:
    namespace, name = payload.get("namespace"), payload.get("name")
    if (not isinstance(name, str) or not name
            or namespace is not None and not isinstance(namespace, str)):  # fmt: skip
        return "unknown_tool"
    if (
        (namespace, name) in WRAPPERS
        or namespace is None
        and name in {"functions.exec", "functions.wait"}
    ):
        return None
    return f"{namespace}.{name}" if namespace else name


def _item_tool(item: dict, diagnostics: list[str]) -> str | None:
    kind = item.get("type")
    if not isinstance(kind, str):
        diagnostics.append("invalid native item type")
        return None
    if kind in INERT_ITEMS:
        return None
    if kind == "McpToolCall":
        server, tool = item.get("server"), item.get("tool")
        if isinstance(server, str) and server and isinstance(tool, str) and tool:
            return f"mcp__{server}__{tool}"
    elif kind == "CommandExecution":
        return "exec_command"
    elif kind == "SubAgentActivity":
        return "spawn_agent"
    diagnostics.append(f"unsupported native item: {kind}")
    return None


def summarize_rollout(
    path: str | Path,
    *,
    expected_child: str | None = None,
    expected_parent: str | None = None,
    expected_agent_type: str | None = None,
    canonical: str | None = None,
    standalone: bool = False,
    trusted_environment_context: str | None = None,
    expected_image_count: int = 0,
    expected_image_paths: list[str | Path] | None = None,
) -> dict:
    """Return lifecycle/schema evidence, explicitly scoped to the harness's active model.

    At SubagentStop the harness may not have appended task_complete yet. A completed final
    assistant response is then terminal *response* evidence; later starts/aborts invalidate it.
    The caller must capture this snapshot from the trusted stop hook and revalidate lifecycle.
    """
    diagnostics: list[str] = []
    bound_paths = None
    if expected_image_paths is not None:
        if (not isinstance(expected_image_paths, list)
                or any(not isinstance(p, (str, Path)) or not str(p)
                       or not Path(p).is_absolute() for p in expected_image_paths)):  # fmt: skip
            diagnostics.append("invalid trusted image paths")
        else:
            bound_paths = [str(p) for p in expected_image_paths]
            if (type(expected_image_count) is not int
                    or expected_image_count not in {0, len(bound_paths)}):  # fmt: skip
                diagnostics.append("trusted image path count differs from launch count")
            expected_image_count = len(bound_paths)
    records = []
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if (not isinstance(record, dict) or not isinstance(record.get("payload"), dict)
                    or not isinstance(record.get("type"), str)):  # fmt: skip
                diagnostics.append("invalid rollout record")
            else:
                records.append(record)
    except OSError, ValueError, TypeError, UnicodeDecodeError:
        diagnostics.append("rollout unreadable or malformed")
    metas = [r["payload"] for r in records if r.get("type") == "session_meta"]
    if len(metas) != 1 or not records or records[0].get("type") != "session_meta":
        diagnostics.append("expected exactly one initial native session metadata record")
    meta = metas[0] if metas else {}
    if standalone:
        child, parent, role = meta.get("id"), None, expected_agent_type
        history = meta.get("subagent_history_start_ordinal")
        fresh = (meta.get("source") == "exec" and not meta.get("parent_thread_id")
                 and not meta.get("forked_from_id")
                 and (history is None or type(history) is int and history == 0))  # fmt: skip
        if not isinstance(child, str) or not child or not fresh:
            diagnostics.append("managed session is not a fresh standalone exec")
        if expected_parent is not None:
            diagnostics.append("standalone session cannot claim a native parent")
        if not isinstance(role, str) or not role or not isinstance(canonical, str) or not canonical:
            diagnostics.append("managed session needs trusted role and canonical launch")
    else:
        try:
            child, parent, role, fresh = _lineage(meta, diagnostics)
        except TypeError:
            child, parent, role, fresh = None, None, None, False
            diagnostics.append("invalid native lineage fields")
    fresh = fresh and len(metas) == 1
    provider = meta.get("model_provider")
    if provider != "openai":
        diagnostics.append("native provider is not openai")
    for label, actual, expected in (
        ("child", child, expected_child),
        ("parent", parent, expected_parent),
        ("agent type", role, expected_agent_type),
    ):
        if expected is not None and actual != expected:
            diagnostics.append(f"native {label} differs from expected identity")

    prompts: list[str] = []
    tools: list[str] = []
    tool_calls: list[dict] = []
    models: dict[str, set[str]] = {}
    usage_by_response: dict[str, tuple[str, dict]] = {}
    started, completed = [], []
    final_response = False
    image_count = environment_count = 0
    wrapper_bindings = 0
    for rec in records:
        kind, payload = rec.get("type"), rec["payload"]
        if kind == "turn_context":
            turn, model = payload.get("turn_id"), payload.get("model")
            if not isinstance(turn, str) or not turn or not isinstance(model, str) or not model:
                diagnostics.append("turn context lacks native model or turn identity")
            else:
                models.setdefault(turn, set()).add(model)
        elif kind == "response_item":
            item_type = payload.get("type")
            if not isinstance(item_type, str):
                diagnostics.append("invalid response item type")
                continue
            if item_type == "message":
                role_name = payload.get("role")
                if not isinstance(role_name, str):
                    diagnostics.append("invalid message role")
                    continue
                if role_name == "user":
                    content = payload.get("content")
                    if (bound_paths is not None and isinstance(content, list)
                            and any(isinstance(b, dict) and b.get("type") == "input_image"
                                    for b in content)):  # fmt: skip
                        content, bound = _bound_images(content, bound_paths, diagnostics)
                        wrapper_bindings += int(bound)
                    prompt, images = _prompt(content, diagnostics)
                    image_count += images
                    if prompt is not None:
                        if (
                            standalone
                            and not images
                            and _trusted_environment(prompt, trusted_environment_context)
                        ):
                            if prompts:
                                diagnostics.append("environment context follows launch")
                            environment_count += 1
                        else:
                            prompts.append(prompt)
                elif role_name == "assistant":
                    final_response |= (payload.get("phase") == "final"
                                       and payload.get("status") == "completed")  # fmt: skip
                elif role_name not in {"system", "developer"}:
                    diagnostics.append("unsupported message role")
            elif item_type in {"function_call", "custom_tool_call"}:
                if standalone and _direct_tool(payload) is None:
                    tools.append("code-mode-wrapper")
                    tool_calls.append({"tool": "code-mode-wrapper", "id": payload.get("call_id")})
                elif tool := _direct_tool(payload):
                    tools.append(tool)
                    tool_calls.append({"tool": tool, "id": payload.get("call_id")})
            elif item_type == "additional_tools" and payload.get("role") == "developer":
                if not isinstance(payload.get("tools"), list):
                    diagnostics.append("invalid native additional tools declaration")
            elif item_type not in {"function_call_output", "custom_tool_call_output", "reasoning"}:
                diagnostics.append(f"unsupported response item: {item_type}")
        elif kind == "token_usage_record":
            response, turn, usage = (payload.get("response_id"), payload.get("turn_id"),
                                     payload.get("usage"))  # fmt: skip
            if (payload.get("thread_id") != child or not isinstance(response, str) or not response
                    or not isinstance(turn, str) or not turn or not isinstance(usage, dict)
                    or not {"input_tokens", "output_tokens"} <= usage.keys()
                    or any(k not in USAGE_NAMES or type(v) is not int or v < 0
                           for k, v in usage.items())):  # fmt: skip
                diagnostics.append("invalid or foreign per-response usage")
                continue
            normalized = {USAGE_NAMES[k]: v for k, v in usage.items()}
            candidate = (turn, normalized)
            if response in usage_by_response and usage_by_response[response] != candidate:
                diagnostics.append("conflicting duplicate response usage")
            else:
                usage_by_response[response] = candidate
        elif kind == "event_msg":
            event = payload.get("type")
            if not isinstance(event, str):
                diagnostics.append("invalid lifecycle event type")
                continue
            if event == "task_started":
                started.append(payload.get("turn_id"))
            elif event == "task_complete":
                completed.append(payload.get("turn_id"))
            elif event == "item_completed":
                item = payload.get("item")
                if not isinstance(item, dict):
                    diagnostics.append("invalid native completed item")
                elif tool := _item_tool(item, diagnostics):
                    tools.append(tool)
                    tool_calls.append({"tool": tool, "id": item.get("id")})
            elif event not in INERT_EVENTS:
                diagnostics.append(f"unsupported or interrupted lifecycle event: {event}")
        elif kind != "session_meta" and kind not in INERT_RECORDS:
            diagnostics.append(f"unsupported rollout record: {kind}")

    if len(started) != 1 or not isinstance(started[0], str) or not started[0]:
        diagnostics.append("expected exactly one native started turn")
    active_turn = started[0] if len(started) == 1 else None
    if set(models) != {active_turn} or any(len(v) != 1 for v in models.values()):
        diagnostics.append("missing, foreign, or mixed active model evidence")
    if len(prompts) != 1 or canonical is not None and prompts != [canonical]:
        diagnostics.append("non-canonical launch or added user messages")
    if environment_count > 1:
        diagnostics.append("repeated environment context")
    if type(expected_image_count) is not int or image_count != expected_image_count:
        diagnostics.append("user images differ from trusted launch")
    if bound_paths and wrapper_bindings != 1:
        diagnostics.append("trusted launch image wrappers were not bound exactly once")
    if completed and completed != [active_turn]:
        diagnostics.append("foreign or repeated terminal turn")
    terminal = ("task-complete" if completed == [active_turn] and active_turn
                else "assistant-final-response" if final_response else None)  # fmt: skip
    if not terminal:
        diagnostics.append("no completed terminal response or turn")
    if standalone and terminal != "task-complete":
        diagnostics.append("managed standalone session has no completed native task")
    if not usage_by_response:
        diagnostics.append("no native per-response usage")
    turns = []
    for response, (turn, usage) in sorted(usage_by_response.items()):
        chosen = models.get(turn, set())
        if turn != active_turn or len(chosen) != 1:
            diagnostics.append("usage has no unambiguous active model/turn")
            continue
        turns.append({"message_id": response, "model": next(iter(chosen)), "usage": usage})
    return {
        "complete": not diagnostics,
        "prompts": prompts,
        "tools": tools,
        "tool_calls": tool_calls,
        "turns": turns,
        "model_evidence_scope": "harness-active",
        "models": sorted({m for values in models.values() for m in values}),
        "provider": provider,
        "child": child,
        "parent": parent,
        "agent_type": role,
        "agent_type_evidence_scope": "trusted-runner" if standalone else "native-metadata",
        "fresh": fresh,
        "standalone": standalone,
        "image_count": image_count,
        "image_wrapper_paths": bound_paths if wrapper_bindings == 1 else [],
        "image_wrappers_bound": wrapper_bindings == 1,
        "terminal_evidence": terminal,
        "diagnostics": sorted(set(diagnostics)),
    }
