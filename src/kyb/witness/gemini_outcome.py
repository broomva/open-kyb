"""Gemini terminal evidence, inspected before any HTTP availability classification.

Only allow-listed metadata and hashes leave this parser, never provider diagnostic text.
"""

from __future__ import annotations

import json
import re
from typing import Any

from kyb.read.harness import model_family
from kyb.util.hashing import sha256_bytes, sha256_json
from kyb.witness.envelope import LEAF, OPAQUE, Anomalies, Many, error_like_key
from kyb.witness.errors import (
    ROUTING_VERSION,
    WitnessAnomalous,
    WitnessRefused,
    WitnessUnavailable,
)
from kyb.witness.evidence import diagnostics, gemini_terminal

_TRANSIENT = {
    429: "RESOURCE_EXHAUSTED",
    500: "INTERNAL",
    503: "UNAVAILABLE",
    504: "DEADLINE_EXCEEDED",
}
_TOOL_PART_KEYS = frozenset({"functionCall", "functionResponse", "executableCode",
                             "codeExecutionResult", "inlineData", "fileData"})  # fmt: skip
_RATING = {"category": LEAF, "probability": LEAF, "blocked": LEAF, "probabilityScore": LEAF,
           "severity": LEAF, "severityScore": LEAF}  # fmt: skip
_CITATION = {"startIndex": LEAF, "endIndex": LEAF, "uri": LEAF, "title": LEAF, "license": LEAF,
             "publicationDate": {"year": LEAF, "month": LEAF, "day": LEAF}}  # fmt: skip
_DETAILS = Many({"modality": LEAF, "tokenCount": LEAF})
_ROOT_SPEC = {
    "candidates": Many({
        "content": {"role": LEAF, "parts": Many({"text": LEAF, "thought": LEAF,
                                                  "thoughtSignature": LEAF,
                                                  **dict.fromkeys(_TOOL_PART_KEYS, OPAQUE)})},
        "finishReason": LEAF, "finishMessage": LEAF, "index": LEAF,
        "safetyRatings": Many(_RATING),
        "citationMetadata": {"citations": Many(_CITATION), "citationSources": Many(_CITATION)},
        "avgLogprobs": LEAF, "tokenCount": LEAF,
        # tool/continuation evidence is a refusal (checked below), not an anomaly
        "groundingMetadata": OPAQUE, "urlContextMetadata": OPAQUE, "continuationToken": OPAQUE,
    }),
    "promptFeedback": {"blockReason": LEAF, "blockReasonMessage": LEAF,
                       "safetyRatings": Many(_RATING)},
    "usageMetadata": {
        "promptTokenCount": LEAF, "candidatesTokenCount": LEAF, "totalTokenCount": LEAF,
        "thoughtsTokenCount": LEAF, "cachedContentTokenCount": LEAF,
        "toolUsePromptTokenCount": LEAF, "promptTokensDetails": _DETAILS,
        "candidatesTokensDetails": _DETAILS, "cacheTokensDetails": _DETAILS,
        "toolUsePromptTokensDetails": _DETAILS, "serviceTier": LEAF, "trafficType": LEAF,
    },
    "modelVersion": LEAF, "responseId": LEAF, "createTime": LEAF,
}  # fmt: skip
_REFUSAL_TEXT = re.compile(
    r"safety|block(?:ed|list)|refus|schema|invalid.argument|authenticat|api.key|permission|"
    r"configur|unsupported|malformed",
    re.I,
)


def error_like_keys(node: Any) -> list[tuple[str, Any]]:
    """Every non-empty key whose normalized form (lowercase, no `_`/`-`) starts with `error`,
    at any depth of any dict/list, regardless of casing."""
    found: list[tuple[str, Any]] = []
    if isinstance(node, list):
        for item in node:
            found.extend(error_like_keys(item))
    elif isinstance(node, dict):
        for key, value in node.items():
            if error_like_key(key) and value not in (None, "", [], {}):
                found.append((str(key), value))
            found.extend(error_like_keys(value))
    return found


def transient_error(provenance: dict) -> bool:
    status, error = provenance.get("http_status"), provenance.get("error_status")
    return (
        type(status) is int
        and status in _TRANSIENT
        and isinstance(error, dict)
        and type(error.get("code")) is int
        and error["code"] == status
        and error.get("status") == _TRANSIENT[status]
    )


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate JSON key")
        out[key] = value
    return out


def decode(raw: bytes) -> Any:
    def constant(_: str) -> None:
        raise ValueError("nonfinite JSON")

    value = json.loads(raw, object_pairs_hook=_object, parse_constant=constant)
    json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return value


def token(value: Any) -> str | None:
    """Protocol enums/model IDs only; arbitrary provider text is represented by its hash."""
    return (
        value
        if isinstance(value, str) and re.fullmatch(r"[a-zA-Z0-9_.:/-]{1,150}", value)
        else None
    )


class GeminiOutcome:
    def __init__(self, raw: bytes, status: int, *, transport: str = "gemini-api",
                 route: dict[str, Any] | None = None,
                 requested_model: str | None = None) -> None:  # fmt: skip
        self.raw, self.status = raw, status
        self.anomalies = Anomalies(transport, requested_model)
        self.refusals: list[str] = []
        self.provenance = {
            "transport": transport,
            "family": "google",
            "tools": [],
            "no_data_tools": False,
            "validation_version": ROUTING_VERSION,
            "reported_model": None,
            "cost_usd": None,
            "http_status": status,
            "response_hash": sha256_bytes(raw),
            "refusal_evidence": self.refusals,
            "classification": "refused",
            "normal_terminal": False,
            "candidates": [],
            "prompt_block_reason": None,
            "prompt_safety": [],
            "error_status": None,
            "error_diagnostics": [],
            "finish_diagnostics": [],
            "envelope_anomalies": self.anomalies.items,
        }
        self.provenance["terminal_evidence"] = []
        if route is not None:
            self.provenance["route"] = dict(route)
        try:
            self.out = decode(raw)
        except Exception:
            self.out = None
            self.refuse("response_json_invalid")
        if not isinstance(self.out, dict):
            self.refuse("response_not_object")
            self.out = {}
        self.provenance["reported_model"] = token(self.out.get("modelVersion"))
        self.anomalies.served_model = self.provenance["reported_model"]
        self.inspect(self.out)

    def refuse(self, reason: str) -> None:
        if reason not in self.refusals:
            self.refusals.append(reason)

    def inspect(self, node: Any) -> None:
        """Do not let a conflicting failure/block marker hide inside an error or candidate."""
        if isinstance(node, list):
            for item in node:
                self.inspect(item)
        elif isinstance(node, dict):
            for key, value in node.items():
                if key == "finishMessage" and value not in (None, ""):
                    item = diagnostics([value])[0]
                    if value in ("STOP", "completed"):
                        item["category"] = "normal"
                    self.provenance["finish_diagnostics"].append(item)
                if key in {"finishReason", "blockReason", "blocked"}:
                    self.provenance["terminal_evidence"].append(
                        {"field": key, "value": value if type(value) is bool else token(value)}
                    )
                if key == "finishReason" and value != "STOP":
                    self.refuse("finish_not_stop")
                if key == "blockReason":
                    self.refuse("prompt_blocked")
                if key == "blocked" and type(value) is not bool:
                    self.refuse("blocked_marker_invalid")
                elif key == "blocked" and value:
                    self.refuse("safety_blocked")
                if key in {"refusal", "refusal_evidence"} and value:
                    self.refuse("explicit_refusal")
                self.inspect(value)
        # Unstructured error messages are inspected only in error(), never saved.

    def error(self) -> None:
        error = self.out.get("error")
        safe = (
            {
                "code": error.get("code") if type(error.get("code")) is int else None,
                "status": token(error.get("status")),
            }
            if isinstance(error, dict)
            else {}
        )
        self.provenance["error_status"] = safe
        # An unrecognised message dominates a transient status (a shaped 503 around a refusal
        # stays terminal). The one exception is a quota response, where HTTP 429 and
        # RESOURCE_EXHAUSTED agree: its message names a varying quota (in live runs: a hash
        # that matched no known phrasing), so the agreed status alone decides. A retry is
        # bounded and still needs a valid answer, so this can never accept anything (issue R4).
        diagnostics_list = []
        if isinstance(error, dict):
            status_str = error.get("status")
            if status_str:
                diagnostics_list.append(status_str)
            quota = error.get("code") == 429 and status_str == "RESOURCE_EXHAUSTED"
            if error.get("message") and not quota:
                diagnostics_list.append(error["message"])
        self.provenance["error_diagnostics"] = (
            diagnostics(diagnostics_list) if diagnostics_list else []
        )
        if not isinstance(error, dict) or set(self.out) != {"error"}:
            for extra in set(self.out) - {"error"}:
                self.anomalies.add(f"$.{extra}", extra, self.out[extra], "error_response", "error")
            self.refuse("error_envelope_invalid")
        elif set(error) - {"code", "status", "message", "details"}:
            self.refuse("error_metadata_unknown")
        if isinstance(error, dict):
            if type(error.get("code")) is not int or error["code"] != self.status:
                self.refuse("http_error_status_conflict")
            if not isinstance(error.get("message", ""), str):
                self.refuse("error_message_invalid")
            # These words are refusal evidence even when the numeric status is transient.
            if _REFUSAL_TEXT.search(json.dumps(error, ensure_ascii=True)):
                self.refuse("error_refusal_or_configuration")
            details = error.get("details", [])
            if not isinstance(details, list):
                self.refuse("error_details_invalid")
            else:
                for detail in details:
                    if not isinstance(detail, dict):
                        self.refuse("error_details_invalid")
                        continue
                    name = detail.get("@type")
                    if name not in {
                        "type.googleapis.com/google.rpc.RetryInfo",
                        "type.googleapis.com/google.rpc.QuotaFailure",
                        "type.googleapis.com/google.rpc.Help",
                    }:
                        self.refuse("error_details_unknown")
                    elif name.endswith("RetryInfo"):
                        if (
                            set(detail) != {"@type", "retryDelay"}
                            or not isinstance(detail.get("retryDelay"), str)
                            or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?s", detail["retryDelay"])
                        ):
                            self.refuse("retry_details_malformed")
                    elif name.endswith("QuotaFailure"):
                        violations = detail.get("violations")
                        if (
                            self.status != 429
                            or set(detail) != {"@type", "violations"}
                            or not isinstance(violations, list)
                        ):
                            self.refuse("quota_details_malformed")
                            continue
                        for violation in violations:
                            if (
                                not isinstance(violation, dict)
                                or set(violation) - {"subject", "description"}
                                or any(not isinstance(v, str) for v in violation.values())
                            ):
                                self.refuse("quota_diagnostic_malformed")
                            elif violation.get("description"):
                                self.provenance["error_diagnostics"].extend(
                                    diagnostics([violation["description"]])
                                )
                    else:
                        links = detail.get("links")
                        if set(detail) != {"@type", "links"} or not isinstance(links, list):
                            self.refuse("help_details_malformed")
                            continue
                        for link in links:
                            if (
                                not isinstance(link, dict)
                                or set(link) - {"url", "description"}
                                or any(not isinstance(v, str) for v in link.values())
                            ):
                                self.refuse("help_diagnostic_malformed")
                            elif link.get("description"):
                                self.provenance["error_diagnostics"].extend(
                                    diagnostics([link["description"]])
                                )

        if self.refusals:
            raise WitnessRefused("Gemini error evidence refused: " + ", ".join(self.refusals))
        if _TRANSIENT.get(self.status) == safe.get("status"):
            gemini_terminal(self.provenance, "unavailable")
            self.provenance["classification"] = "unavailable"
            raise WitnessUnavailable("Gemini reported a recognized transient no-answer error")
        self.refuse("error_not_recognized_transient")
        raise WitnessRefused("Gemini error is not recognized transient availability")

    def completion(self, schema: dict) -> tuple[dict, dict]:
        from kyb.witness.adapters import json_schema, validate

        if self.status != 200 or "error" in self.out:
            return self.error()
        self.anomalies.scan(self.out, _ROOT_SPEC, "$", "response")
        explicit = error_like_keys(self.out)
        if self.anomalies.has_error and not explicit:
            explicit = [(i["key"], i["value"]) for i in self.anomalies.items
                        if i["severity"] == "error"]  # fmt: skip
        if explicit:
            # STOP plus schema-valid JSON must not erase an observed failure at any depth.
            for _key, value in explicit:
                self.provenance["error_diagnostics"].extend(diagnostics([value]))
            self.refuse("explicit_error_metadata")
            raise WitnessRefused("Gemini explicit error metadata refused")
        model = self.provenance["reported_model"]
        if model_family(model) != "google":
            self.refuse("actual_model_missing_or_invalid")
        feedback = self.out.get("promptFeedback", {})
        if not isinstance(feedback, dict):
            self.refuse("prompt_feedback_invalid")
        else:
            self.provenance["prompt_block_reason"] = token(feedback.get("blockReason"))
            self.provenance["prompt_safety"] = self.ratings(
                feedback.get("safetyRatings", []), "prompt_safety"
            )
        candidates = self.out.get("candidates")
        texts = []
        if not isinstance(candidates, list) or len(candidates) != 1:
            self.refuse("candidate_evidence_incomplete")
        if isinstance(candidates, list):
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    self.refuse("candidate_invalid")
                    continue
                self.provenance["candidates"].append(
                    {
                        "finish_reason": token(candidate.get("finishReason")),
                        "safety": self.ratings(
                            candidate.get("safetyRatings", []), "candidate_safety"
                        ),
                    }
                )
                if candidate.get("finishReason") != "STOP":
                    self.anomalies.add(
                        "$.candidates[].finishReason",
                        "finishReason",
                        candidate.get("finishReason"),
                        "candidate",
                        "error",
                    )
                    self.refuse("finish_not_stop")
                if any(
                    k in candidate
                    for k in ("groundingMetadata", "urlContextMetadata", "continuationToken")
                ):
                    self.refuse("unexpected_tool_or_continuation_evidence")
                content = candidate.get("content")
                parts = content.get("parts") if isinstance(content, dict) else None
                if not isinstance(parts, list) or not parts:
                    self.refuse("content_invalid")
                    continue
                for part in parts:
                    if (
                        not isinstance(part, dict)
                        or set(part) & _TOOL_PART_KEYS
                        or not isinstance(part.get("text"), str)
                    ):
                        self.refuse("unexpected_content_or_tool")
                        continue
                    if "thought" in part and type(part["thought"]) is not bool:
                        self.refuse("thought_marker_invalid")
                    if part.get("thought") is not True:
                        texts.append(part["text"])
        data = None
        if len(texts) != 1:
            self.refuse("schema_answer_incomplete")
        else:
            try:
                data = decode(texts[0].encode("utf-8"))
                validate(data, json_schema(schema))
                self.provenance["schema_valid"] = True
            except Exception:
                self.refuse("schema_invalid")
                self.provenance["schema_valid"] = False
            self.provenance["schema_text_hash"] = sha256_bytes(texts[0].encode("utf-8"))
            self.provenance["schema_hash"] = sha256_json(data)
        usage = self.out.get("usageMetadata", {})
        if not isinstance(usage, dict) or any(
            type(v) is not int or v < 0 for k, v in usage.items() if k.endswith("TokenCount")
        ):
            self.refuse("usage_invalid")
        if self.refusals:
            raise WitnessRefused("Gemini completion refused: " + ", ".join(self.refusals))
        if self.anomalies.items:
            self.provenance["classification"] = "anomalous"
            raise WitnessAnomalous("Gemini envelope carries unknown shapes", self.anomalies.items)
        self.provenance["data_hash"] = sha256_json(data)
        self.provenance["final_output_hash"] = sha256_json(data)
        self.provenance["no_data_tools"] = True
        gemini_terminal(self.provenance, "complete")
        self.provenance["classification"] = "complete"
        self.provenance["normal_terminal"] = True
        self.provenance["no_data_tools"] = True
        return data, usage

    def ratings(self, ratings: Any, label: str) -> list[dict]:
        if not isinstance(ratings, list) or any(not isinstance(r, dict) for r in ratings):
            self.refuse(label + "_invalid")
            return []
        return [
            {
                "category": token(r.get("category")),
                "probability": token(r.get("probability")),
                "blocked": r.get("blocked") if type(r.get("blocked")) is bool else None,
            }
            for r in ratings
        ]
