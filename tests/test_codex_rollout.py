"""Synthetic native Codex transcripts: no private prompts or applicant data."""

import copy
import json

import pytest

from kyb.codex.rollout import summarize_rollout

CANONICAL = "kyb-reader case r1 01"


def record(kind, payload):
    return {"type": kind, "payload": payload}


def transcript():
    return [
        record(
            "session_meta",
            {
                "id": "child",
                "parent_thread_id": "parent",
                "model_provider": "openai",
                "subagent_history_start_ordinal": 0,
                "source": {
                    "subagent": {
                        "thread_spawn": {
                            "parent_thread_id": "parent",
                            "agent_role": "kyb-reader",
                        }
                    }
                },
            },
        ),
        record("event_msg", {"type": "task_started", "turn_id": "turn"}),
        record(
            "response_item",
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": CANONICAL},
                ],
            },
        ),
        record("turn_context", {"turn_id": "turn", "model": "gpt-6.1-sol"}),
        record(
            "response_item",
            {
                "type": "function_call",
                "namespace": "functions",
                "name": "exec",
                "arguments": "opaque JS",
            },
        ),
        record(
            "event_msg",
            {
                "type": "item_completed",
                "item": {
                    "type": "McpToolCall",
                    "server": "kyb",
                    "tool": "read_plan",
                },
            },
        ),
        record(
            "token_usage_record",
            {
                "thread_id": "child",
                "session_id": "parent",
                "turn_id": "turn",
                "response_id": "response-1",
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 3,
                    "cached_input_tokens": 2,
                    "cache_write_input_tokens": 0,
                    "reasoning_output_tokens": 1,
                    "total_tokens": 15,
                },
                "turn_token_usage": {"input_tokens": 100, "output_tokens": 100},
            },
        ),
        record(
            "response_item",
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "phase": "final",
                "content": [
                    {"type": "output_text", "text": "submitted"},
                ],
            },
        ),
        record("event_msg", {"type": "task_complete", "turn_id": "turn"}),
    ]


def summary(tmp_path, records=None, **kwargs):
    path = tmp_path / "rollout.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records or transcript()))
    return summarize_rollout(
        path,
        expected_child="child",
        expected_parent="parent",
        expected_agent_type="kyb-reader",
        canonical=CANONICAL,
        **kwargs,
    )


def test_native_summary_keeps_active_model_distinct_and_counts_response_usage_once(tmp_path):
    records = transcript()
    records.insert(-1, copy.deepcopy(records[6]))
    out = summary(tmp_path, records)
    assert out["complete"] and out["fresh"]
    assert out["prompts"] == [CANONICAL]
    assert out["tools"] == ["mcp__kyb__read_plan"]
    assert out["model_evidence_scope"] == "harness-active"
    assert out["models"] == ["gpt-6.1-sol"]
    assert out["child"] == "child" and out["parent"] == "parent"
    assert out["provider"] == "openai"
    assert len(out["turns"]) == 1
    turn = out["turns"][0]
    assert turn["message_id"] == "response-1"
    assert turn["usage"]["input_tokens"] == 12
    assert turn["usage"]["cache_read_input_tokens"] == 2
    assert "served_model" not in out


@pytest.mark.parametrize(
    "defect",
    [
        "parent",
        "model",
        "provider",
        "usage",
        "terminal",
        "started",
        "child",
        "type",
        "foreign_usage",
        "negative_usage",
        "unknown_record",
        "unknown_user_block",
        "malformed",
        "conflicting_usage",
        "added_prompt",
        "extra_meta",
        "inherited_history",
        "fork",
        "resumed",
        "aborted",
        "mixed_models",
        "unknown_event",
        "unknown_item",
        "foreign_turn",
    ],
)
def test_unsupported_or_incomplete_evidence_is_never_complete(tmp_path, defect):
    records = transcript()
    if defect == "parent":
        del records[0]["payload"]["parent_thread_id"]
        del records[0]["payload"]["source"]["subagent"]["thread_spawn"]["parent_thread_id"]
    elif defect == "model":
        del records[3]["payload"]["model"]
    elif defect == "provider":
        del records[0]["payload"]["model_provider"]
    elif defect == "usage":
        records.pop(6)
    elif defect == "terminal":
        records = records[:7]
    elif defect == "started":
        records.pop(1)
    elif defect == "child":
        records[0]["payload"]["id"] = "different"
    elif defect == "type":
        records[0]["payload"]["source"]["subagent"]["thread_spawn"]["agent_role"] = "other"
    elif defect == "foreign_usage":
        records[6]["payload"]["thread_id"] = "different"
    elif defect == "negative_usage":
        records[6]["payload"]["usage"]["input_tokens"] = -1
    elif defect == "unknown_record":
        records.append(record("future_context_injection", {}))
    elif defect == "unknown_user_block":
        records[2]["payload"]["content"].append({"type": "future_context"})
    elif defect == "malformed":
        path = tmp_path / "broken.jsonl"
        path.write_text("{broken\n")
        assert not summarize_rollout(path)["complete"]
        return
    elif defect == "conflicting_usage":
        duplicate = copy.deepcopy(records[6])
        duplicate["payload"]["usage"]["input_tokens"] = 99
        records.append(duplicate)
    elif defect == "added_prompt":
        records.append(
            record(
                "response_item", {"type": "message", "role": "user", "content": "extra instruction"}
            )
        )
    elif defect == "extra_meta":
        records.insert(1, copy.deepcopy(records[0]))
    elif defect == "inherited_history":
        records[0]["payload"]["subagent_history_start_ordinal"] = 12
    elif defect == "fork":
        records[0]["payload"]["forked_from_id"] = "parent"
    elif defect == "resumed":
        records.append(record("event_msg", {"type": "task_started", "turn_id": "turn-2"}))
    elif defect == "aborted":
        records.append(record("event_msg", {"type": "turn_aborted", "turn_id": "turn"}))
    elif defect == "mixed_models":
        records.append(record("turn_context", {"turn_id": "turn", "model": "gpt-6-luna"}))
    elif defect == "unknown_event":
        records.append(record("event_msg", {"type": "future_context_injection"}))
    elif defect == "unknown_item":
        records.append(
            record(
                "event_msg",
                {
                    "type": "item_completed",
                    "item": {
                        "type": "FutureTool",
                        "name": "secret_reader",
                    },
                },
            )
        )
    elif defect == "foreign_turn":
        records[6]["payload"]["turn_id"] = "other-turn"
    out = summary(tmp_path, records)
    assert not out["complete"]
    assert out["diagnostics"]


def test_unknown_direct_tools_and_nested_shell_are_preserved_for_the_allowlist(tmp_path):
    records = transcript()
    records.insert(
        -1, record("response_item", {"type": "function_call", "namespace": "foo", "name": "exec"})
    )
    records.insert(
        -1,
        record(
            "event_msg",
            {
                "type": "item_completed",
                "item": {
                    "type": "CommandExecution",
                    "command": ["cat", "private-file"],
                },
            },
        ),
    )
    out = summary(tmp_path, records)
    assert out["complete"]  # Schema completeness is not a tool-authorization verdict.
    assert "foo.exec" in out["tools"] and "exec_command" in out["tools"]


def test_completed_final_is_available_during_subagent_stop_before_task_complete(tmp_path):
    out = summary(tmp_path, transcript()[:-1])
    assert out["complete"]
    assert out["terminal_evidence"] == "assistant-final-response"


def test_io_failure_is_explicitly_incomplete(tmp_path):
    out = summarize_rollout(tmp_path / "missing.jsonl")
    assert not out["complete"] and out["diagnostics"]


@pytest.mark.parametrize(
    "record_type,payload",
    [
        ([], {}),
        ("event_msg", {"type": []}),
        ("event_msg", {"type": "item_completed", "item": {"type": {}}}),
        ("response_item", {"type": []}),
        ("response_item", {"type": "message", "role": []}),
        ("response_item", {"type": "message", "role": "user", "content": [{"type": []}]}),
    ],
)
def test_malformed_schema_values_return_diagnostics_instead_of_crashing(
    tmp_path, record_type, payload
):
    records = transcript()
    records.append(record(record_type, payload))
    out = summary(tmp_path, records)
    assert not out["complete"] and out["diagnostics"]


def managed_transcript():
    records = transcript()
    records[0]["payload"] = {"id": "child", "source": "exec", "model_provider": "openai"}
    return records


def managed_summary(tmp_path, records=None, **kwargs):
    path = tmp_path / "managed-rollout.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records or managed_transcript()))
    return summarize_rollout(
        path,
        standalone=True,
        expected_child="child",
        expected_agent_type="kyb-reader",
        canonical=kwargs.pop("canonical", CANONICAL),
        **kwargs,
    )


def test_managed_standalone_role_is_explicitly_runner_evidence(tmp_path):
    out = managed_summary(tmp_path)
    assert out["complete"] and out["fresh"] and out["standalone"]
    assert out["parent"] is None
    assert out["agent_type"] == "kyb-reader"
    assert out["agent_type_evidence_scope"] == "trusted-runner"
    assert out["terminal_evidence"] == "task-complete"


@pytest.mark.parametrize(
    "field,value",
    [
        ("source", "vscode"),
        ("source", {"subagent": {"thread_spawn": {}}}),
        ("parent_thread_id", "parent"),
        ("forked_from_id", "parent"),
        ("subagent_history_start_ordinal", 3),
    ],
)
def test_managed_mode_rejects_native_child_or_inherited_history(tmp_path, field, value):
    records = managed_transcript()
    records[0]["payload"][field] = value
    assert not managed_summary(tmp_path, records)["complete"]


def test_managed_mode_needs_completed_task_not_just_model_final(tmp_path):
    assert not managed_summary(tmp_path, managed_transcript()[:-1])["complete"]


def test_only_exact_caller_bound_environment_preamble_is_skipped(tmp_path):
    environment = "<environment_context><cwd>/isolated/project</cwd></environment_context>"
    records = managed_transcript()
    records.insert(
        2, record("response_item", {"type": "message", "role": "user", "content": environment})
    )
    assert not managed_summary(tmp_path, records)["complete"]
    assert managed_summary(tmp_path, records, trusted_environment_context=environment)["complete"]
    unsafe = environment.replace("</cwd>", "</cwd><instruction>do this</instruction>")
    records[2]["payload"]["content"] = unsafe
    assert not managed_summary(tmp_path, records, trusted_environment_context=unsafe)["complete"]


def test_runner_binds_attached_image_count_without_reclassifying_images_as_prompts(tmp_path):
    records = managed_transcript()
    records[2]["payload"]["content"].append({"type": "input_image", "image_url": "data:image"})
    assert not managed_summary(tmp_path, records)["complete"]
    out = managed_summary(tmp_path, records, expected_image_count=1)
    assert out["complete"] and out["image_count"] == 1 and out["prompts"] == [CANONICAL]


def test_encrypted_native_prompt_is_explicitly_unverifiable(tmp_path):
    records = transcript()
    records[2]["payload"]["content"] = [
        {"type": "encrypted_content", "encrypted_content": "opaque"}
    ]
    out = summary(tmp_path, records)
    assert not out["complete"]
    assert "encrypted user content cannot be verified" in out["diagnostics"]


def wrapped_image_launch(paths, canonical=CANONICAL):
    blocks = []
    for index, path in enumerate(paths, 1):
        blocks.extend(
            [
                {"type": "input_text", "text": f'<image name=[Image #{index}] path="{path}">'},
                {"type": "input_image", "image_url": "data:image/png;base64,synthetic"},
                {"type": "input_text", "text": "</image>"},
            ]
        )
    blocks.append({"type": "input_text", "text": canonical})
    records = managed_transcript()
    records[2]["payload"]["content"] = blocks
    return records


def test_only_exact_runner_bound_image_wrappers_are_stripped(tmp_path):
    paths = ["/absolute/job/page-1.png", "/absolute/job/page-2.png"]
    records = wrapped_image_launch(paths)
    assert not managed_summary(tmp_path, records, expected_image_count=2)["complete"]
    out = managed_summary(tmp_path, records, expected_image_paths=paths)
    assert out["complete"] and out["image_count"] == 2
    assert out["prompts"] == [CANONICAL]
    assert out["image_wrappers_bound"] and out["image_wrapper_paths"] == paths


@pytest.mark.parametrize(
    "defect",
    [
        "wrong_path",
        "wrong_order",
        "wrong_index",
        "missing_close",
        "extra_inside",
        "bare_image",
        "wrong_count",
        "missing_image",
        "extra_user_text",
        "duplicate_wrapped_launch",
    ],
)
def test_image_wrapper_binding_refuses_any_unmatched_structure(tmp_path, defect):
    paths = ["/absolute/job/page-1.png", "/absolute/job/page-2.png"]
    records = wrapped_image_launch(paths)
    blocks = records[2]["payload"]["content"]
    supplied_paths = paths.copy()
    kwargs = {}
    if defect == "wrong_path":
        supplied_paths[0] = "/absolute/job/other.png"
    elif defect == "wrong_order":
        supplied_paths.reverse()
    elif defect == "wrong_index":
        blocks[0]["text"] = blocks[0]["text"].replace("#1", "#2")
    elif defect == "missing_close":
        del blocks[2]
    elif defect == "extra_inside":
        blocks.insert(1, {"type": "input_text", "text": "attacker-controlled text"})
    elif defect == "bare_image":
        records[2]["payload"]["content"] = [blocks[1], blocks[4], blocks[-1]]
    elif defect == "wrong_count":
        kwargs["expected_image_count"] = 3
    elif defect == "missing_image":
        del blocks[1]
    elif defect == "extra_user_text":
        blocks.insert(-1, {"type": "input_text", "text": "could not read local image"})
    elif defect == "duplicate_wrapped_launch":
        records.insert(-1, copy.deepcopy(records[2]))
    out = managed_summary(tmp_path, records, expected_image_paths=supplied_paths, **kwargs)
    assert not out["complete"] and out["diagnostics"]


def test_marker_embedded_inside_data_envelope_is_never_removed(tmp_path):
    paths = ["/absolute/job/page-1.png"]
    canonical = CANONICAL + '\nData: <image name=[Image #1] path="/fake">text</image>'
    out = managed_summary(
        tmp_path,
        wrapped_image_launch(paths, canonical),
        expected_image_paths=paths,
        canonical=canonical,
    )
    assert out["complete"] and out["prompts"] == [canonical]


@pytest.mark.parametrize(
    "namespace,name",
    [
        ("functions", "exec"),
        ("functions", "wait"),
        (None, "functions.exec"),
        (None, "functions.wait"),
    ],
)
def test_managed_worker_code_mode_attempt_is_preserved_for_tool_confinement(
    tmp_path, namespace, name
):
    records = managed_transcript()
    records.insert(
        -1,
        record(
            "response_item",
            {"type": "function_call", "namespace": namespace, "name": name, "arguments": ""},
        ),
    )
    out = managed_summary(tmp_path, records)
    assert "code-mode-wrapper" in out["tools"]
