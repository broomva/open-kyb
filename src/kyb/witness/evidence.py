"""Shared live/replay evidence predicates. Recognition is affirmative, never absence of an error."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from kyb.read.harness import model_family
from kyb.util.hashing import sha256_json
from kyb.witness.errors import WitnessRefused

CLAUDE_HTTP = {429: "rate_limit", 500: "server_error", 504: "server_error", 529: "server_error"}
_CODES = {
    "rate_limit": "rate_limit",
    "rate_limit_error": "rate_limit",
    "rate limit exceeded": "rate_limit",
    "too many requests": "rate_limit",
    "server_error": "server_error",
    "internal_server_error": "server_error",
    "overloaded_error": "server_error",
    "internal server error": "server_error",
    "temporarily unavailable": "transient",
    "service unavailable": "transient",
    "resource_exhausted": "rate_limit",
    "unavailable": "transient",
    "deadline_exceeded": "transient",
    "internal": "server_error",
}


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise WitnessRefused(reason)


def hash_value(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value) is not None


def diagnostic(value: Any) -> dict:
    if not isinstance(value, str):
        return {"category": "malformed", "hash": sha256_json(value)}
    val = value.strip().lower()
    category = _CODES.get(val)
    if not category:
        if any(
            term in val
            for term in (
                "rate limit",
                "rate_limit",
                "quota exceeded",
                "quota_exceeded",
                "too many requests",
                "resource_exhausted",
                "resource exhausted",
            )
        ):
            category = "rate_limit"
        elif any(
            term in val
            for term in (
                "temporarily unavailable",
                "service unavailable",
                "unavailable",
                "deadline_exceeded",
                "deadline exceeded",
            )
        ):
            category = "transient"
        elif any(
            term in val
            for term in ("internal error", "server error", "server_error", "internal_server_error")
        ):
            category = "server_error"
        else:
            category = "unknown"
    return {"category": category, "hash": sha256_json(value)}


def diagnostics(values: Any) -> list[dict]:
    if not isinstance(values, list):
        return [{"category": "malformed", "hash": sha256_json(values)}]
    return [diagnostic(value) for value in values]


def diagnostic_kinds(records: Any) -> set[str]:
    require(isinstance(records, list), "diagnostic evidence is missing or malformed")
    kinds = set()
    for item in records:
        require(
            isinstance(item, dict) and set(item) == {"category", "hash"}, "diagnostic shape invalid"
        )
        kind = item.get("category")
        require(
            kind in ("rate_limit", "server_error", "transient") and hash_value(item.get("hash")),
            "unknown, malformed or refused diagnostic remains terminal",
        )
        kinds.add(kind)
    specific = kinds - {"transient"}
    require(len(specific) <= 1, "terminal diagnostics contradict each other")
    return specific


_ANSWER_KEYS = ("final_output_hash", "schema_text_hash", "schema_hash", "data_hash",
                "schema_valid", "reported_model", "reported_models", "schema_checks")  # fmt: skip


def require_no_answer_evidence(p: dict, label: str) -> None:
    """Shared by every provider: a no-answer (unavailable) outcome carries no answer evidence."""
    for key in _ANSWER_KEYS:
        require(p.get(key) in (None, [], {}), f"{label} no-answer outcome carries {key}")
    for result in p.get("result_evidence") or []:
        require(
            isinstance(result, dict) and result.get("final_output_hash") is None,
            f"{label} no-answer outcome carries a result answer hash",
        )


def claude_terminal(p: dict) -> str:
    """Same predicate for parsed live evidence and persisted replay metadata."""
    errors, results = p.get("provider_errors"), p.get("result_evidence")
    models, checks = p.get("reported_models"), p.get("schema_checks")
    require(
        p.get("other_diagnostics") == [], "Claude unexpected diagnostic evidence remains terminal"
    )
    require(hash_value(p.get("response_hash")), "Claude raw response hash missing")
    require(
        isinstance(errors, list) and all(e in ("rate_limit", "server_error") for e in errors),
        "Claude provider error codes are malformed or unknown",
    )
    require(
        isinstance(results, list) and isinstance(models, list) and isinstance(checks, list),
        "Claude terminal evidence incomplete",
    )
    kinds = diagnostic_kinds(p.get("assistant_diagnostics"))
    require(kinds == set(errors), "Claude parallel assistant error evidence conflicts")
    require(
        type(p.get("schema_values_checked")) is int and p["schema_values_checked"] == len(checks),
        "Claude schema evidence count conflicts",
    )
    require(
        type(p.get("timed_out")) is bool
        and (p.get("returncode") is None or type(p.get("returncode")) is int)
        and p.get("process_exception") in (None, "TimeoutExpired", "FileNotFoundError"),
        "Claude process evidence invalid",
    )
    tools, called, work = p.get("tools"), p.get("called_tools"), p.get("workdir")
    require(
        isinstance(called, list)
        and all(t == "StructuredOutput" for t in called)
        and (
            tools is None or isinstance(tools, list) and all(t == "StructuredOutput" for t in tools)
        )
        and isinstance(work, dict)
        and isinstance(work.get("before"), list)
        and work.get("before") == work.get("after"),
        "Claude tool/workdir evidence conflicts",
    )
    from kyb.witness.channels import validate_channels

    stderr_kinds, stderr_diagnostics = validate_channels(p)
    kinds |= stderr_kinds
    rates = p.get("rate_limit_events")
    require(
        isinstance(rates, list)
        and all(
            isinstance(r, dict)
            and set(r) == {"status", "hash", "overage_status", "diagnostics"}
            and r.get("status") in ("allowed", "allowed_warning", "rejected")
            and r.get("overage_status") in (None, "allowed", "allowed_warning", "rejected")
            and hash_value(r.get("hash"))
            for r in rates
        ),
        "Claude rate limit status evidence missing or unknown",
    )
    for rate in rates:
        kinds |= diagnostic_kinds(rate["diagnostics"])
    if any(r["status"] == "rejected" or r["overage_status"] == "rejected" for r in rates):
        kinds.add("rate_limit")
    require(len(kinds) <= 1, "Claude assistant/channel diagnostics conflict")
    if not results:
        require_no_answer_evidence(p, "Claude")
        require(
            not models
            and not checks
            and p.get("nonblank_lines") == 0
            and p.get("reported_model") is None
            and p.get("api_error_status") is None
            and tools is None
            and not called,
            "Claude partial evidence is not empty transport availability",
        )
        require(
            p.get("timed_out") is True
            or p.get("process_exception") == "FileNotFoundError"
            or type(p.get("returncode")) is int
            and p["returncode"] != 0,
            "Claude empty response lacks recognized transport failure",
        )
        return "unavailable"
    require(len(results) == 1 and isinstance(results[0], dict), "Claude result evidence incomplete")
    result = results[0]
    require(
        {
            "subtype",
            "is_error",
            "stop_reason",
            "terminal_reason",
            "api_error_status",
            "errors",
            "result_diagnostics",
        }
        <= result.keys(),
        "Claude result fields missing",
    )
    require(
        "api_error_status" in p and p["api_error_status"] == result["api_error_status"],
        "Claude parallel status evidence conflicts",
    )
    require(
        p.get("usage") == result.get("usage") and p.get("cost_usd") == result.get("total_cost_usd"),
        "Claude parallel result metadata conflicts",
    )
    kinds |= diagnostic_kinds(result["errors"]) | diagnostic_kinds(result["result_diagnostics"])
    require(len(kinds) <= 1, "Claude final diagnostics conflict with earlier evidence")
    status = result["api_error_status"]
    require(status is None or type(status) is int, "Claude status has invalid type")
    subtype, failed = result["subtype"], result["is_error"]
    require(
        type(failed) is bool and result["stop_reason"] in (None, "end_turn", "tool_use"),
        "Claude stop/error evidence invalid",
    )
    if subtype == "success" and failed is False:
        require(
            not kinds
            and not stderr_diagnostics
            and not result["errors"]
            and status is None
            and result["terminal_reason"] in (None, "completed"),
            "Claude error conflicts with success",
        )
        require(
            p.get("timed_out") is False
            and type(p.get("returncode")) is int
            and p["returncode"] == 0,
            "Claude process did not complete normally",
        )
        require(
            len(models) == 1
            and models[0] == p.get("reported_model")
            and model_family(models[0]) == "anthropic",
            "Claude actual model evidence incomplete",
        )
        require(
            p.get("no_data_tools") is True
            and checks
            and all(
                isinstance(c, dict)
                and c.get("valid") is True
                and hash_value(c.get("value_hash"))
                and c.get("source") in ("tool", "final")
                for c in checks
            ),
            "Claude schema/confinement evidence incomplete",
        )
        final = [c for c in checks if c["source"] == "final"]
        require(
            len(final) == 1
            and hash_value(p.get("final_output_hash"))
            and final[0]["value_hash"] == p["final_output_hash"]
            and result.get("final_output_hash") == p["final_output_hash"],
            "Claude final-output/schema evidence missing or contradictory",
        )
        return "complete"
    require_no_answer_evidence(p, "Claude")
    require(
        failed is True
        and not models
        and not checks
        and p.get("reported_model") is None
        and p.get("timed_out") is False
        and not called
        and result["stop_reason"] is None,
        "Claude partial/refused answer is not availability",
    )
    require(
        result["terminal_reason"] in (None, "api_error", "model_error"),
        "Claude unknown/aborted terminal reason",
    )
    if subtype == "success":
        require(
            not result["errors"] and status in CLAUDE_HTTP, "Claude API availability status invalid"
        )
        require(not kinds or kinds == {CLAUDE_HTTP[status]}, "Claude diagnostic/status conflict")
    else:
        # SDK API status belongs only to success. Earlier codes cannot override any later status.
        require(
            subtype == "error_during_execution" and status is None and bool(kinds),
            "Claude loop failure is not recognized consistent availability",
        )
    return "unavailable"


def safe_ratings(value: Any) -> list[dict]:
    require(isinstance(value, list), "Gemini safety evidence missing or malformed")
    for rating in value:
        require(
            isinstance(rating, dict) and set(rating) == {"category", "probability", "blocked"},
            "Gemini safety rating invalid",
        )
        require(
            rating["blocked"] is None or rating["blocked"] is False,
            "Gemini retained safety block remains terminal",
        )
        require(
            isinstance(rating["category"], str)
            and rating["category"].startswith("HARM_CATEGORY_")
            and rating["probability"]
            in ("HARM_PROBABILITY_UNSPECIFIED", "NEGLIGIBLE", "LOW", "MEDIUM", "HIGH"),
            "Gemini safety enum evidence unknown",
        )
    return value


def gemini_terminal(p: dict, outcome: str) -> None:
    require(
        p.get("reported_model") is None or isinstance(p["reported_model"], str),
        "Gemini actual model evidence invalid",
    )
    require(p.get("tools") == [], "Gemini tool evidence conflicts")
    require(
        {
            "terminal_evidence",
            "candidates",
            "prompt_block_reason",
            "prompt_safety",
            "error_status",
            "error_diagnostics",
            "finish_diagnostics",
        }
        <= p.keys(),
        "Gemini retained evidence missing",
    )
    require(p["prompt_block_reason"] is None, "Gemini retained prompt block remains terminal")
    prompt = safe_ratings(p["prompt_safety"])
    require(isinstance(p["candidates"], list), "Gemini candidate evidence malformed")
    expected = [("blocked", False) for r in prompt if r["blocked"] is False]
    for c in p["candidates"]:
        require(
            isinstance(c, dict) and set(c) == {"finish_reason", "safety"},
            "Gemini candidate shape invalid",
        )
        require(c["finish_reason"] == "STOP", "Gemini retained stop evidence remains terminal")
        expected.append(("finishReason", "STOP"))
        expected.extend(
            ("blocked", False) for r in safe_ratings(c["safety"]) if r["blocked"] is False
        )
    markers = p["terminal_evidence"]
    require(
        isinstance(markers, list)
        and all(
            isinstance(m, dict)
            and set(m) == {"field", "value"}
            and isinstance(m["field"], str)
            and type(m["value"]) in (str, bool, type(None))
            for m in markers
        ),
        "Gemini terminal marker shape invalid",
    )
    require(
        Counter((m["field"], m["value"]) for m in markers) == Counter(expected),
        "Gemini parallel terminal/block evidence conflicts",
    )
    kinds = diagnostic_kinds(p["error_diagnostics"])
    finish = p["finish_diagnostics"]
    require(isinstance(finish, list), "Gemini finish diagnostic evidence missing")
    require(
        all(
            isinstance(d, dict)
            and set(d) == {"category", "hash"}
            and d.get("category") == "normal"
            and hash_value(d.get("hash"))
            for d in finish
        ),
        "Gemini unknown or contradictory finish diagnostic remains terminal",
    )
    if outcome == "complete":
        require(hash_value(p.get("response_hash")), "Gemini raw response hash missing")
        require(
            p.get("no_data_tools") is True
            and not any(k in p for k in ("network_stage", "network_error_type")),
            "Gemini completion conflicts with transport evidence",
        )
        require(
            len(p["candidates"]) == 1
            and type(p.get("http_status")) is int
            and p["http_status"] == 200,
            "Gemini normal completion evidence incomplete",
        )
        require(
            p["error_status"] is None and not kinds and not p["error_diagnostics"],
            "Gemini retained error contradicts completion",
        )
        require(
            model_family(p.get("reported_model")) == "google"
            and p.get("schema_valid") is True
            and hash_value(p.get("schema_hash"))
            and hash_value(p.get("data_hash"))
            and hash_value(p.get("final_output_hash"))
            and hash_value(p.get("schema_text_hash"))
            and p["schema_hash"] == p["data_hash"] == p["final_output_hash"],
            "Gemini actual model/schema proof incomplete",
        )
    elif "http_status" in p:
        from kyb.witness.gemini_outcome import transient_error

        require_no_answer_evidence(p, "Gemini")
        require(not finish, "Gemini no-answer status conflicts with finish diagnostic")
        require(
            isinstance(p.get("error_status"), dict)
            and set(p["error_status"]) == {"code", "status"}
            and hash_value(p.get("response_hash")),
            "Gemini retained HTTP error evidence malformed",
        )

        require(
            not p["candidates"]
            and p["reported_model"] is None
            and not prompt
            and transient_error(p),
            "Gemini HTTP availability evidence conflicts",
        )
        require(
            not any(
                k in p
                for k in (
                    "schema_valid",
                    "schema_hash",
                    "data_hash",
                    "final_output_hash",
                    "schema_text_hash",
                    "network_stage",
                    "network_error_type",
                )
            ),
            "Gemini no-answer status conflicts with retained answer/transport evidence",
        )
        if p["http_status"] == 429:
            expected_kind = "rate_limit"
        elif p["http_status"] in (503, 504):
            expected_kind = "transient"
        else:
            expected_kind = "server_error"
        require(not kinds or kinds == {expected_kind}, "Gemini diagnostics conflict with status")
    else:
        require_no_answer_evidence(p, "Gemini")
        require(
            not finish
            and not p["candidates"]
            and p["reported_model"] is None
            and not prompt
            and p["error_status"] is None
            and not p["error_diagnostics"]
            and not any(
                k in p
                for k in (
                    "schema_valid",
                    "schema_hash",
                    "data_hash",
                    "final_output_hash",
                    "schema_text_hash",
                )
            )
            and p.get("network_stage") == "connect"
            and p.get("network_error_type")
            in (
                "TimeoutError",
                "ConnectionRefusedError",
                "ConnectionResetError",
                "ConnectionAbortedError",
                "BrokenPipeError",
                "gaierror",
            ),
            "Gemini network availability evidence incomplete",
        )
