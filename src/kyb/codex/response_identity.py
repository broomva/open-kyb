"""Extract ONLY response id/model from the pinned Codex WebSocket receive trace.

Codex 0.164.0 does not put served-model metadata in its rollout. Its native transport logs
the authenticated response events. Filter in memory: never persist or return raw trace lines,
headers, prompts, auth data or response text. Do not search nested/model-generated JSON.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")
PREFIX = re.compile(r"^\S+\s+TRACE tungstenite::protocol: Received message (\{.*\})$")


class Models:
    def __init__(self):
        self.created: dict[str, str] = {}
        self.completed: dict[str, str] = {}
        self.usage: dict[str, dict | None] = {}
        self.invalid = False

    def feed(self, line: str) -> None:
        match = PREFIX.fullmatch(ANSI.sub("", line.rstrip("\r\n")))
        if match is None:
            return
        try:
            event = json.loads(match[1])
            kind = event.get("type")
            if kind not in {"response.created", "response.completed"}:
                return
            response = event["response"]
            rid, model = response["id"], response["model"]
            if (not isinstance(rid, str) or not rid.startswith("resp_")
                    or not isinstance(model, str) or not model):  # fmt: skip
                raise ValueError("missing response identity")
            for records in (self.created, self.completed):
                if rid in records and records[rid] != model:
                    raise ValueError("served model changed within a response")
            records = self.created if kind == "response.created" else self.completed
            records[rid] = model
            if kind == "response.completed":
                usage = response.get("usage")
                valid = isinstance(usage, dict) and all(
                    type(usage.get(k)) is int and usage[k] >= 0
                    for k in ("input_tokens", "output_tokens")
                )
                self.usage[rid] = ({k: usage[k] for k in ("input_tokens", "output_tokens")}
                                   if valid else None)  # fmt: skip
        except KeyError, TypeError, ValueError:
            self.invalid = True


def execute(cmd: list[str], prompt: str, *, cwd: Path, env: dict, directory: Path,
            timeout: int) -> dict:  # fmt: skip
    models = Models()
    # Only this module's receive records are parsed. Other trace output is discarded as it
    # arrives, in bounded chunks. In exec mode this pinned version writes tracing to stderr.
    env = {**env, "RUST_LOG": "tungstenite::protocol=trace"}
    with (directory / "events.jsonl").open("w") as output:
        started = time.monotonic()
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.PIPE,
                                stdout=output, stderr=subprocess.PIPE, text=True,
                                start_new_session=True)  # fmt: skip

        def collect():
            at_start = True
            try:
                while chunk := proc.stderr.readline(1024 * 1024):
                    complete = chunk.endswith("\n")
                    if at_start and complete:
                        models.feed(chunk)
                    at_start = complete
            except OSError, UnicodeError:
                models.invalid = True

        reader = threading.Thread(target=collect, daemon=True)
        reader.start()
        input_errors = []

        def send():
            try:
                with proc.stdin as stream:
                    stream.write(prompt)
            except OSError, UnicodeError:
                input_errors.append(True)

        writer = threading.Thread(target=send, daemon=True)
        writer.start()
        try:
            proc.wait(timeout=max(0, timeout - (time.monotonic() - started)))
            writer.join(timeout=max(0, timeout - (time.monotonic() - started)))
            if writer.is_alive() or input_errors:
                raise subprocess.TimeoutExpired(cmd, timeout)
        except subprocess.TimeoutExpired:
            # The session owns this process group, including its MCP subprocesses. Killing
            # only the Codex parent can leave those children holding a blocked pipe open.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            raise RuntimeError("managed Codex process timed out or closed its input") from None
        finally:
            writer.join(timeout=1)
            reader.join(timeout=1)
            if reader.is_alive() or writer.is_alive():
                models.invalid = True
        if models.invalid:
            raise RuntimeError("Codex served-response metadata is malformed or incomplete")
    return {"exit": proc.returncode, "created": models.created, "completed": models.completed,
            "usage": models.usage}  # fmt: skip
