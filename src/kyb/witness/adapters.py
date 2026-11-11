"""Dev/eval CLI transports for spec. Never infer confinement or served model from flags.

L6's schema conversion/environment allow-list and CLI output parsing are reused here in the
product port without importing the eval package. Claude attaches image blocks directly: its
built-in and MCP tool sets are empty. Codex and agy remain unavailable for verified routing
where the measured CLI interface cannot establish no-tools plus actual-model provenance.
Capability probes verify the Claude CLI adapter; Codex and agy remain declared
unavailable until their sandbox and schema guarantees are established.
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from kyb.util.hashing import sha256_bytes, sha256_json
from kyb.witness import gemini
from kyb.witness.channels import channel, response_lines, stdout_diagnostics
from kyb.witness.claude_stream import ClaudeStream, stream_bytes
from kyb.witness.errors import ROUTING_VERSION, WitnessAnomalous, WitnessRefused, WitnessUnavailable

ENV = ("PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL", "LC_CTYPE",
       "TMPDIR", "TERM", "CLAUDE_CONFIG_DIR", "CODEX_HOME")  # fmt: skip


def json_schema(node: Any) -> Any:
    """OpenAPI-style schema to closed JSON Schema; optional properties stay optional."""
    if isinstance(node, list):
        return [json_schema(x) for x in node]
    if not isinstance(node, dict):
        return node
    out = {k: json_schema(v) for k, v in node.items() if k != "nullable"}
    if "type" in out:
        out["type"] = out["type"].lower()
    if node.get("nullable"):
        out["type"] = [out["type"], "null"]
    if out.get("type") == "object":
        out["additionalProperties"] = False
    return out


def request(kind: str, spec: dict) -> tuple[str, dict]:
    """Construct only claim-blind requests, independent of reader ids and typed values."""
    if kind == "transcribe" and set(spec) == {"crop_ids"}:
        ids = spec["crop_ids"]
        if ids != [f"k{i}" for i in range(1, len(ids) + 1)]:
            raise ValueError("only opaque positional crop ids are accepted")
        return (gemini.TRANSCRIBE_PROMPT.format(guard=gemini._GUARD, ids=", ".join(ids)),
                json_schema(gemini.TRANSCRIBE_SCHEMA))  # fmt: skip
    if kind == "layout" and set(spec) == {"kinds", "doc_sha256"}:
        return (gemini.LAYOUT_PROMPT.format(guard=gemini._GUARD, kinds=", ".join(spec["kinds"])),
                json_schema(gemini.LAYOUT_SCHEMA))  # fmt: skip
    raise ValueError("unsupported or non-blind witness request")


def validate(data: Any, schema: dict) -> None:
    """Validate the small JSON-Schema subset the two witness schemas actually use."""
    kinds = schema.get("type")
    kinds = kinds if isinstance(kinds, list) else [kinds]
    types = {"object": dict, "array": list, "string": str, "integer": int,
             "boolean": bool, "null": type(None)}  # fmt: skip
    if not any(type(data) is types[k] for k in kinds):
        raise WitnessRefused("witness output has an invalid JSON type")
    if data is None:
        return
    if isinstance(data, dict):
        props = schema["properties"]
        if set(data) - set(props) or set(schema.get("required", [])) - set(data):
            raise WitnessRefused("witness output violates the closed schema")
        for key, value in data.items():
            validate(value, props[key])
    elif isinstance(data, list):
        for value in data:
            validate(value, schema["items"])


def served_model_matches(requested: str, served: str) -> bool:
    """Exact, or the requested alias with an Anthropic dated snapshot suffix (-YYYYMMDD)."""
    return served == requested or re.fullmatch(re.escape(requested) + r"-\d{8}", served) is not None


class ClaudeWitness:
    """Fresh Claude print process, directly attached images and no model-facing tools.

    Launch flags are documented and locally checked at 2.1.310. Successful live canary runs
    are separate evidence; unit fakes validate plumbing only. No retry spends are automatic.
    """

    transport = "claude-p"
    family = "anthropic"
    prompt_version = "witness-strata-v1"
    validation_version = ROUTING_VERSION

    def __init__(self, model: str, *, binary: str = "claude", timeout: int = 180) -> None:
        self.model, self.binary, self.timeout = model, binary, timeout
        self.last_provenance: dict[str, Any] = {}

    def command(self, schema: dict) -> list[str]:
        return [self.binary, "-p", "--model", self.model, "--restricted", "--safe-mode",
                "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
                "--tools", "", "--setting-sources", "", "--disable-slash-commands",
                "--disallowedTools", "mcp__*", "--no-chrome",
                "--no-session-persistence", "--input-format", "stream-json",
                "--output-format", "stream-json", "--verbose", "--max-turns", "3",
                "--json-schema", json.dumps(schema, separators=(",", ":"))]  # fmt: skip

    def check_served_model(self, evidence: ClaudeStream) -> None:
        """Served model is ONLY each assistant event's message.model (never modelUsage, which
        includes harness side calls, and never the init echo of the requested alias)."""
        if not evidence.models:
            if any(
                r.get("subtype") == "success" and not r.get("is_error") for r in evidence.results
            ):
                evidence.refuse("served_model_missing")
        elif any(not served_model_matches(self.model, m) for m in evidence.models):
            evidence.refuse("served_model_mismatch")

    def call(self, kind: str, images: list[bytes], spec: dict) -> tuple[dict, dict]:
        prompt, schema = request(kind, spec)
        return self.run(prompt, images, schema)

    def run(self, prompt: str, images: list[bytes], schema: dict) -> tuple[dict, dict]:
        """Shared with real canary probes: exactly the same command and confinement flags."""
        self.last_provenance = {}
        content = [{"type": "text", "text": prompt}]
        for image in images:
            content.append({"type": "image", "source": {"type": "base64",
                "media_type": "image/png", "data": base64.b64encode(image).decode()}})  # fmt: skip
        message = {"type": "user", "message": {"role": "user", "content": content},
                   "parent_tool_use_id": None}  # fmt: skip
        self.last_provenance = {
            "transport": self.transport,
            "family": self.family,
            "validation_version": self.validation_version,
            "reported_model": None,
        }
        with tempfile.TemporaryDirectory(prefix="kyb-witness-") as directory:
            work = Path(directory)
            for i, image in enumerate(images, 1):
                (work / f"k{i}.png").write_bytes(image)
            before = sorted(p.name for p in work.iterdir())
            timed_out, launch_unavailable, process_refused = False, False, False
            process_exception = None
            try:
                proc = subprocess.run(self.command(schema), cwd=work,
                    env={k: os.environ[k] for k in ENV if k in os.environ},
                    input=(json.dumps(message) + "\n").encode(), capture_output=True,
                    timeout=self.timeout)  # fmt: skip
                output, diagnostic, returncode = proc.stdout, proc.stderr, proc.returncode
            except Exception as exc:
                output, diagnostic, returncode = (
                    getattr(exc, "output", None),
                    getattr(exc, "stderr", None),
                    None,
                )
                process_exception = type(exc).__name__
                timed_out = isinstance(exc, subprocess.TimeoutExpired)
                launch_unavailable = (
                    isinstance(exc, FileNotFoundError) and not output and not diagnostic
                )
                process_refused = not timed_out and not launch_unavailable
            after = sorted(p.name for p in work.iterdir())
        channel_refusals = []
        channels = []
        for name, value in (("stdout", output), ("stderr", diagnostic)):
            if value is None and process_exception is None:
                channel_refusals.append(f"{name}_capture_missing")
            try:
                channels.append(stream_bytes(value))
            except WitnessRefused:
                channel_refusals.append(f"{name}_capture_invalid")
                channels.append(b"")
        raw, stderr = channels
        self.last_provenance |= {"workdir": {"before": before, "after": after},
            "response_hash": sha256_json({"stdout": sha256_bytes(raw),
                                          "stderr": sha256_bytes(stderr)}),
            "stdout_bytes": len(raw), "stderr_bytes": len(stderr),
            "returncode": returncode, "timed_out": timed_out,
            "process_exception": process_exception}  # fmt: skip
        evidence = ClaudeStream(
            schema,
            validate,
            before == after,
            workdir={"before": before, "after": after},
            requested_model=self.model,
        )
        evidence.consume(raw)
        self.check_served_model(evidence)
        # Inspect every stderr line, even beside a valid stdout result or on timeout.
        channels = {"stdout": channel(raw), "stderr": channel(stderr, diagnostic_channel=True)}
        for line in response_lines(stderr):
            evidence.refuse("response_on_diagnostic_channel")
            evidence.consume(line)
        for reason in channel_refusals:
            evidence.refuse(reason)
        if process_refused:
            evidence.refuse("unclassified_process_failure")
        parsed = evidence.provenance()
        channels["stdout"]["diagnostics"] = stdout_diagnostics(parsed)
        channels["stdout"]["statuses"] = [
            r["api_error_status"]
            for r in parsed["result_evidence"]
            if type(r.get("api_error_status")) is int
        ]
        self.last_provenance |= {**parsed, "channels": channels}
        try:
            data, usage = evidence.finish(
                returncode,
                timed_out=timed_out,
                launch_unavailable=launch_unavailable,
                transport_evidence=self.last_provenance,
            )
        except WitnessRefused:
            self.last_provenance["classification"] = "refused"
            self.last_provenance["normal_terminal"] = False
            raise
        except WitnessAnomalous:
            self.last_provenance["classification"] = "anomalous"
            self.last_provenance["normal_terminal"] = False
            raise
        except WitnessUnavailable:
            self.last_provenance["classification"] = "unavailable"
            self.last_provenance["normal_terminal"] = False
            raise
        self.last_provenance["classification"] = "complete"
        self.last_provenance["normal_terminal"] = True
        tokens = {"cost_usd": self.last_provenance.get("cost_usd")}
        input_keys = ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
        if any(k in usage for k in input_keys):
            tokens["promptTokenCount"] = sum(usage.get(k, 0) for k in input_keys)
        if "output_tokens" in usage:
            tokens["candidatesTokenCount"] = usage["output_tokens"]
        return data, tokens


class CodexWitness:
    """L6 codex-exec result parser with its actual provenance gap kept fail-closed.

    0.153.4 exec JSONL does not report the served model. read-only covers shell writes,
    not disk reads; disabling two shell features is not proof that every tool is absent.
    Do not invoke a paid call whose known interface cannot satisfy the boundary.
    Per CLI capability probes: no served model outside
    RUST_LOG=trace stderr, and --json hides tool events (residual code-mode tools).
    """

    transport, family, prompt_version = "codex-exec", "openai", "witness-strata-v1"
    validation_version = ROUTING_VERSION

    def __init__(self, model: str) -> None:
        self.model = model
        self.last_provenance = {"reported_model": None,
            "gap": "codex exec has no attested served-model/no-tools output"}  # fmt: skip

    def call(self, kind: str, images: list[bytes], spec: dict) -> tuple[dict, dict]:
        request(kind, spec)
        self.last_provenance |= {
            "transport": self.transport,
            "family": self.family,
            "classification": "refused",
            "normal_terminal": False,
            "validation_version": ROUTING_VERSION,
            "refusal_evidence": ["unsupported_cli_capability"],
        }
        raise WitnessRefused(self.last_provenance["gap"])


class AgyWitness(CodexWitness):
    """L6 agy transport is unavailable for verification: terminal sandbox is not no-tools,
    and the measured JSON response carries no served model. Never substitute its model pin.
    Per CLI capability probes: init echoes the requested
    model and run_command executed outside the workspace.
    """

    transport, family = "agy", "google"

    def __init__(self, model: str) -> None:
        super().__init__(model)
        self.last_provenance = {"reported_model": None,
            "gap": "agy reports no served model and exposes no no-tools switch"}  # fmt: skip
