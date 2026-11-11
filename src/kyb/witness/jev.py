"""Jev adapter for J-SUPPORT (spec).

One System One call per question, recorded and replayed like the witness. The folder name is the
hash of the prompt version and the request body. The API key and the support threshold are not
part of that hash. Replay never calls the network.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from kyb.util.hashing import sha256_json
from kyb.witness.port import publish_exclusive_new

PROMPT_VERSION = "j-support-v2"
MODEL_ALIAS = "jev-latest"
ENDPOINT = "https://api.typesafe.ai/v1/systemone"
FAMILY = "typesafe-jev"
TIMEOUT_S = 60
RETRIES = 2


class JevUnavailable(LookupError):
    """No recorded response, or System One could not answer. Carries the hash when one exists."""

    def __init__(
        self,
        message: str,
        *,
        request_hash: str | None = None,
        model: str | None = None,
    ) -> None:
        super().__init__(message)
        self.request_hash = request_hash
        self.model = model


@dataclass(frozen=True)
class JevResponse:
    model: str
    answers: dict[str, Any]
    usage: dict[str, Any]
    request_hash: str
    replayed: bool


class JevAdapter(Protocol):
    def ask(self, body: dict[str, Any]) -> dict[str, Any]: ...


def _redact(text: str, secret: str) -> str:
    if secret and secret in text:
        return text.replace(secret, "[redacted]")
    return text


class JevClient:
    """HTTP adapter. Reads the key at construction, never at import, and never logs it."""

    def __init__(self, api_key: str | None = None) -> None:
        raw = api_key if api_key is not None else os.environ.get("TYPESAFE_API_KEY", "")
        self._key = raw.strip()
        if not self._key:
            raise RuntimeError("TYPESAFE_API_KEY is not set; use replay mode for recorded runs")

    def ask(self, body: dict[str, Any]) -> dict[str, Any]:
        payload = json.dumps(body).encode()
        last = "no attempt"
        for attempt in range(RETRIES + 1):
            request = urllib.request.Request(
                ENDPOINT,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": f"Bearer {self._key}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
                    parsed = json.load(response)
            except urllib.error.HTTPError as exc:
                detail = self._http_detail(exc)
                last = f"HTTP {exc.code}: {detail}"
                if (exc.code != 429 and exc.code < 500) or attempt == RETRIES:
                    raise JevUnavailable(last) from None
                time.sleep(min(2**attempt, 8))
                continue
            except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                last = _redact(f"{type(exc).__name__}: {exc}", self._key)
                if attempt == RETRIES:
                    raise JevUnavailable(last) from None
                time.sleep(min(2**attempt, 8))
                continue
            if not isinstance(parsed, dict) or not isinstance(parsed.get("model"), str):
                raise JevUnavailable("systemone response missing a model version")
            if not isinstance(parsed.get("answers"), dict):
                raise JevUnavailable("systemone response missing answers", model=parsed["model"])
            return parsed
        raise JevUnavailable(last)

    def _http_detail(self, exc: urllib.error.HTTPError) -> str:
        try:
            raw = exc.read().decode("utf-8", "replace")
        except OSError:
            raw = str(exc.reason)
        return _redact(raw, self._key)[:300]


class JevPort:
    """Content-addressed record/replay. `root` is the recordings directory."""

    family = FAMILY

    def __init__(self, root: Path, adapter: JevAdapter | None, *, mode: str = "auto") -> None:
        if mode not in ("auto", "replay", "live"):
            raise ValueError(f"jev mode {mode!r}")
        self.root = root
        self.adapter = adapter
        self.mode = mode

    def ask(self, body: dict[str, Any]) -> JevResponse:
        key = sha256_json({"v": PROMPT_VERSION, "body": body})
        folder = _folder(self.root, key)
        pinned = folder / "0.json"
        if self.mode != "live" and pinned.is_file():
            return _load(pinned, key)
        if self.mode == "replay" or self.adapter is None:
            raise JevUnavailable(f"no recorded jev response ({key})", request_hash=key)
        try:
            raw = self.adapter.ask(body)
        except JevUnavailable as exc:
            if exc.request_hash is None:
                raise JevUnavailable(str(exc), request_hash=key, model=exc.model) from None
            raise
        reported = raw.get("model") if isinstance(raw, dict) else None
        answers = raw.get("answers") if isinstance(raw, dict) else None
        if (
            not isinstance(reported, str)
            or not reported.strip()
            or reported == MODEL_ALIAS
            or not isinstance(answers, dict)
        ):
            raise JevUnavailable("jev adapter returned no model or answers", request_hash=key)
        usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
        record = {
            "alias": body.get("model"),
            "data": {"answers": answers, "model": reported},
            "model": reported,
            "prompt_version": PROMPT_VERSION,
            "request": key,
            "request_body": body,
            "usage": usage,
        }
        blob = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
        revision = 0
        while not publish_exclusive_new(folder / f"{revision}.json", blob):
            revision += 1
            if revision > 100:
                raise JevUnavailable("recording revisions exhausted", request_hash=key)
        return JevResponse(reported, answers, usage, key, replayed=False)


def _folder(root: Path, key: str) -> Path:
    from kyb.util.fsutil import contained

    return contained(root, key.split(":", 1)[1])


def _load(path: Path, key: str) -> JevResponse:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        data = record["data"]
        model = record["model"]
        answers = data["answers"]
        usage = record.get("usage") or {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise JevUnavailable(
            f"recorded jev response is unreadable: {type(exc).__name__}",
            request_hash=key,
        ) from None
    if (
        not isinstance(model, str)
        or not model.strip()
        or model == MODEL_ALIAS
        or not isinstance(answers, dict)
        or not isinstance(usage, dict)
        or record.get("request") != key
        or record.get("prompt_version") != PROMPT_VERSION
        or data.get("model") != model
        or sha256_json({"v": PROMPT_VERSION, "body": record.get("request_body")}) != key
    ):
        raise JevUnavailable("recorded jev response is malformed", request_hash=key)
    return JevResponse(model, answers, usage, key, replayed=True)
