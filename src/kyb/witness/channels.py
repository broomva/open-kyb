"""Safe diagnostic-channel summaries. Unknown evidence is retained by category/hash, not text."""

from __future__ import annotations

from typing import Any

from kyb.util.hashing import sha256_bytes, sha256_json
from kyb.witness.evidence import CLAUDE_HTTP, diagnostic, diagnostic_kinds, hash_value, require
from kyb.witness.gemini_outcome import decode

_RESPONSE_TYPES = ("system", "assistant", "result", "user", "rate_limit_event")
_FIELDS = ("error", "errors", "message", "error_message", "error_code", "type")
_STATUS = ("status", "status_code", "code", "api_error_status")


def channel(raw: bytes, *, diagnostic_channel: bool = False) -> dict:
    records, statuses = [], []

    def malformed(value: Any) -> None:
        records.append({"category": "malformed", "hash": sha256_json(value)})

    def observe(value: Any) -> None:
        if isinstance(value, str):
            records.append(diagnostic(value))
        elif isinstance(value, list) and value:
            for item in value:
                observe(item)
        elif isinstance(value, dict) and value:
            if set(value) - set(_FIELDS + _STATUS):
                malformed(value)
            for key, item in value.items():
                if key == "type" and item == "error":
                    continue  # recognized error-envelope discriminator, never outcome evidence
                if key in _STATUS and type(item) is int:
                    statuses.append(item)
                elif key in _FIELDS + _STATUS:
                    observe(item)
        else:
            malformed(value)

    nonblank = 0
    for line in raw.splitlines():
        if not line.strip():
            continue
        nonblank += 1
        if not diagnostic_channel:
            continue
        try:
            text = line.decode("utf-8")
            if line.lstrip().startswith((b"{", b"[", b'"')):
                observe(decode(line))
            else:
                observe(text)
        except Exception:
            records.append({"category": "malformed", "hash": sha256_bytes(line)})
    # A nonempty JSON envelope containing only its discriminator is not availability.
    if diagnostic_channel and nonblank and not records and not statuses:
        records.append({"category": "malformed", "hash": sha256_bytes(raw)})
    return {
        "hash": sha256_bytes(raw),
        "bytes": len(raw),
        "nonblank_lines": nonblank,
        "diagnostics": records,
        "statuses": statuses,
    }


def response_lines(raw: bytes):
    for line in raw.splitlines():
        try:
            event = decode(line)
        except Exception:
            continue
        if isinstance(event, dict) and event.get("type") in _RESPONSE_TYPES:
            yield line


def stdout_diagnostics(provenance: dict) -> list[dict]:
    require(
        isinstance(provenance.get("assistant_diagnostics"), list)
        and isinstance(provenance.get("other_diagnostics"), list),
        "Claude stdout diagnostic metadata malformed",
    )
    out = [*provenance["assistant_diagnostics"], *provenance["other_diagnostics"]]
    for result in provenance.get("result_evidence", []):
        require(
            isinstance(result, dict)
            and isinstance(result.get("errors"), list)
            and isinstance(result.get("result_diagnostics"), list),
            "Claude stdout result diagnostics malformed",
        )
        out.extend(result["errors"])
        out.extend(result.get("result_diagnostics", []))
    return out


def validate_channels(p: dict) -> tuple[set[str], bool]:
    channels = p.get("channels")
    require(
        isinstance(channels, dict) and set(channels) == {"stdout", "stderr"},
        "Claude channel evidence missing",
    )
    for name, value in channels.items():
        require(
            isinstance(value, dict)
            and set(value) == {"hash", "bytes", "nonblank_lines", "diagnostics", "statuses"},
            "Claude channel summary shape invalid",
        )
        require(
            hash_value(value["hash"])
            and type(value["bytes"]) is int
            and value["bytes"] >= 0
            and type(value["nonblank_lines"]) is int
            and 0 <= value["nonblank_lines"] <= value["bytes"]
            and (value["bytes"] != 0 or value["hash"] == sha256_bytes(b""))
            and value["bytes"] == p.get(name + "_bytes")
            and isinstance(value["statuses"], list)
            and all(type(s) is int and s in CLAUDE_HTTP for s in value["statuses"]),
            "Claude channel counters/statuses invalid",
        )
        diagnostic_kinds(value["diagnostics"])
    stdout, stderr = channels["stdout"], channels["stderr"]
    require(
        stdout["nonblank_lines"] == p.get("nonblank_lines")
        and stdout["diagnostics"] == stdout_diagnostics(p)
        and stdout["statuses"]
        == [
            r["api_error_status"]
            for r in p["result_evidence"]
            if r.get("api_error_status") is not None
        ],
        "Claude stdout evidence conflicts with parsed terminal metadata",
    )
    require(
        p.get("response_hash") == sha256_json({"stdout": stdout["hash"], "stderr": stderr["hash"]}),
        "Claude channel hashes conflict with captured response identity",
    )
    require(
        bool(stderr["nonblank_lines"]) == bool(stderr["diagnostics"] or stderr["statuses"]),
        "Claude stderr evidence is incomplete or contradicts captured nonblank lines",
    )
    kinds = diagnostic_kinds(stderr["diagnostics"]) | {CLAUDE_HTTP[s] for s in stderr["statuses"]}
    require(len(kinds) <= 1, "Claude stderr diagnostics/statuses conflict")
    return kinds, bool(stderr["diagnostics"] or stderr["statuses"])
