"""Envelope anomalies: observe unknown or error-like shapes, never crash, never verify.

A spec is a nested description of the KNOWN shape of a provider envelope. Anything outside it is
recorded as an anomaly (`path`, `key`, `value` truncated, `severity`) instead of refusing:
- `severity: "error"` when the key or any value inside is error-like (an explicit failure signal);
- `severity: "unknown"` for any other unlisted key.
An anomalous attempt can never verify a claim or be classified as plain availability; the port
retries it (bounded) and otherwise leaves the claim CLAIMED with the anomalies attached.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from typing import Any

from kyb.util.hashing import sha256_json

MAX_VALUE = 200
ERROR_PREFIXES = ("err", "fault", "fail", "exception", "fatal")
ERROR_VALUES = frozenset({"error", "failed", "failure", "fatal", "fatalerror", "exception"})
# Free text/model output/opaque payloads that may legitimately contain the word "error".
FREE_TEXT = frozenset(
    {"text", "thinking", "data", "signature", "content", "result", "input", "structured_output",
     "message", "tool_use_result", "transcription"}
)  # fmt: skip


def norm_key(key: Any) -> str:
    """strip + NFKC + zero-width/format characters removed + lowercase, `_`/`-`/space dropped."""
    text = unicodedata.normalize("NFKC", str(key))
    text = "".join(c for c in text if unicodedata.category(c) not in ("Cf", "Cc", "Zs", "Zl", "Zp"))
    return text.strip().lower().replace("_", "").replace("-", "")


def error_like_key(key: Any) -> bool:
    n = norm_key(key)
    return n.startswith(ERROR_PREFIXES) or any(w in n for w in ("error", "fault", "exception",
                                                                 "failure"))  # fmt: skip


STATUS_KEYS = frozenset({"status", "state", "outcome", "code", "reason", "condition"})


def error_like_value(value: Any, key: Any = None) -> bool:
    """`error`/`failed`/... as a whole value anywhere; under a status-like key also any value
    containing a failure word (`authentication_failed`) or an HTTP error status (>= 400)."""
    if isinstance(value, str):
        n = norm_key(value)
        if n in ERROR_VALUES:
            return True
        return norm_key(key) in STATUS_KEYS and any(w in n for w in ERROR_PREFIXES + ("error",))
    return norm_key(key) in STATUS_KEYS and type(value) is int and 400 <= value < 600


def deep_error(node: Any, key: Any = None) -> bool:
    """Any error-like key, or error-word scalar outside free-text fields, at any depth."""
    if isinstance(node, dict):
        return any(error_like_key(k) or deep_error(v, k) for k, v in node.items())
    if isinstance(node, list):
        return any(deep_error(v, key) for v in node)
    return key not in FREE_TEXT and error_like_value(node, key)


def truncate(value: Any) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        text = repr(value)
    return text[:MAX_VALUE]


class Opaque:
    """Not inspected (the witness answer itself, or payloads validated by their own schema)."""


class Many:
    """A list whose items follow `spec`."""

    def __init__(self, spec: Any) -> None:
        self.spec = spec


class Map:
    """A dict with arbitrary keys (model ids, tool ids) whose values follow `spec`."""

    def __init__(self, spec: Any) -> None:
        self.spec = spec


class Tagged:
    """A dict whose `tag` value selects its spec; an unlisted tag value is one anomaly."""

    def __init__(self, tag: str, specs: dict[str, Any]) -> None:
        self.tag, self.specs = tag, specs


class Empty:
    """Known only when absent, null or empty; anything else is an anomaly."""


OPAQUE, EMPTY, LEAF = Opaque(), Empty(), None


@dataclass
class Anomalies:
    """Collector with the context every entry carries."""

    provider: str
    requested_model: str | None = None
    served_model: str | None = None

    def __post_init__(self) -> None:
        self.items: list[dict] = []

    def add(self, path: str, key: str, value: Any, frame: str, severity: str | None = None) -> None:
        if severity is None:
            empty = value in (None, "", [], {})
            severity = (
                "error"
                if (error_like_key(key) and not empty) or deep_error(value, key)
                else "unknown"
            )
        # Provider diagnostic text behind an error signal is kept as a hash only (R4/R5 rule);
        # bare error words and every `unknown` value are kept (truncated) for the drift report.
        shown = (
            sha256_json(value)
            if severity == "error" and not error_like_value(value, key)
            else truncate(value)
        )
        entry = {
            "path": path,
            "key": str(key)[:MAX_VALUE],
            "value": shown,
            "severity": severity,
            "frame": frame,
            "provider": self.provider,
            "requested_model": self.requested_model,
            "served_model": self.served_model,
        }
        if entry not in self.items:
            self.items.append(entry)

    def stamp(self, served_model: str | None) -> list[dict]:
        for item in self.items:
            if item["served_model"] is None:
                item["served_model"] = served_model
        return self.items

    @property
    def has_error(self) -> bool:
        return any(i["severity"] == "error" for i in self.items)

    def scan(self, node: Any, spec: Any, path: str, frame: str, key: str = "") -> None:
        if isinstance(spec, Opaque):
            return
        if isinstance(spec, Empty):
            if node not in (None, [], {}):
                self.add(path, key, node, frame)
            return
        if isinstance(spec, Tagged):
            if not isinstance(node, dict):
                if node is not None:
                    self.add(path, key, node, frame)
                return
            chosen = spec.specs.get(node.get(spec.tag)) if isinstance(
                node.get(spec.tag), str) else None  # fmt: skip
            if chosen is None:
                self.add(f"{path}.{spec.tag}", spec.tag, node.get(spec.tag), frame,
                         "error" if deep_error(node) else "unknown")  # fmt: skip
            else:
                self.scan(node, chosen, path, frame, key)
            return
        if isinstance(spec, Many):
            if not isinstance(node, list):
                if node is not None:
                    self.add(path, key, node, frame)
                return
            for i, item in enumerate(node):
                self.scan(item, spec.spec, f"{path}[{i}]", frame, key)
            return
        if isinstance(spec, Map):
            if not isinstance(node, dict):
                if node is not None:
                    self.add(path, key, node, frame)
                return
            for k, v in node.items():
                self.scan(v, spec.spec, f"{path}.{k}", frame, str(k))
            return
        if isinstance(spec, dict):
            if not isinstance(node, dict):
                if node is not None:
                    self.add(path, key, node, frame)
                return
            for k, v in node.items():
                if k in spec:
                    self.scan(v, spec[k], f"{path}.{k}", frame, k)
                else:
                    self.add(f"{path}.{k}", k, v, frame)
            return
        # leaf: a scalar is expected; a container where a scalar belongs is an unknown shape
        if isinstance(node, dict | list):
            self.add(path, key, node, frame)
        elif key not in FREE_TEXT and error_like_value(node, key):
            self.add(path, key, node, frame, "error")
