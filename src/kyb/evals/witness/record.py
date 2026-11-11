"""Record and replay witness calls per backend, so the backend eval runs offline in `make verify`.

A recording is keyed by what the witness sees and is asked (kind, prompt version, spec, and the
SHA-256 of every image), never by a claim. Each backend has its own directory, so the same request
to two backends gives two recordings. A recording keeps the requested model, the model the backend
reported (or null with the reason), latency, usage and the backend's own provenance fields.

`seed` directories hold responses recorded by the product itself (`silver/witness/raw/`, keyed the
way `kyb.witness.port.Witness` keys them): for the pinned API backend, the run's own recorded
answer is the baseline, so it is replayed, not re-asked.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kyb.util.hashing import sha256_bytes, sha256_json


@dataclass(frozen=True)
class Reply:
    """What `verify_document` reads from a witness answer (the port's `Response` shape)."""

    data: dict[str, Any]
    usage: dict[str, Any]
    model: str
    request_hash: str
    replayed: bool
    latency_s: float | None = None
    provenance: dict[str, Any] | None = None


class Recorder:
    """A witness for one backend: replay a recording, or (live) call the adapter and record it."""

    def __init__(
        self,
        directory: Path,
        backend: dict[str, Any],
        adapter=None,
        *,
        live: bool = False,
        seeds: list[Path] = (),
    ) -> None:
        self.dir, self.backend, self.adapter, self.live = directory, backend, adapter, live
        self.seeds = list(seeds)
        self.mode = "auto" if live else "replay"  # verify_case refuses "live" on a done run
        self.calls: list[dict[str, Any]] = []

    @property
    def prompt_version(self) -> str:
        from kyb.evals.witness import prompts

        return self.backend.get("prompt_version") or prompts.PROMPT_VERSION

    def key(self, kind: str, images: list[bytes], spec: dict) -> str:
        return sha256_json(
            {
                "kind": kind,
                "prompt": self.prompt_version,
                "spec": spec,
                "images": [sha256_bytes(i) for i in images],
            }
        )

    def _seeded(self, kind: str, images: list[bytes], spec: dict) -> dict | None:
        port_key = sha256_json(
            {
                "kind": kind,
                "model": self.backend["model"],
                "prompt": self.prompt_version,
                "spec": spec,
                "images": [sha256_bytes(i) for i in images],
            }
        )
        for seed in self.seeds:
            path = seed / port_key.split(":", 1)[1] / "0.json"
            if path.exists():
                rec = json.loads(path.read_text("utf-8"))
                return {
                    "backend": self.backend["id"],
                    "requested_model": self.backend["model"],
                    "reported_model": rec["model"],
                    "prompt_version": rec["prompt_version"],
                    "request": port_key,
                    "data": rec["data"],
                    "usage": rec["usage"],
                    "latency_s": None,
                    "provenance": {
                        "source": "product recording (silver/witness/raw), replayed as the baseline"
                    },
                }
        return None

    @staticmethod
    def _call(key: str, source: str, rec: dict) -> dict[str, Any]:
        """What one consumed reply cost and reported, whatever its source: `live` (spent now),
        `eval recording` (spent when it was recorded) or `product seed` (spent by the product)."""
        prov = rec.get("provenance") or {}
        return {
            "request": key,
            "source": source,
            "latency_s": rec.get("latency_s"),
            "usage": rec.get("usage") or {},
            "reported_model": rec.get("reported_model"),
            "gaps": sorted(v for k, v in prov.items() if k.endswith("_gap")),
        }

    def ask(self, kind: str, images: list[bytes], spec: dict[str, Any]) -> Reply:
        key = self.key(kind, images, spec)
        path = self.dir / f"{key.split(':', 1)[1]}.json"
        # a recording is never re-asked: live mode only fills what is missing (delete one to redo)
        rec = json.loads(path.read_text("utf-8")) if path.exists() else None
        source = "eval recording"
        if rec is None:
            rec, source = self._seeded(kind, images, spec), "product seed"
        if rec is not None:
            self.calls.append(self._call(key, source, rec))
            return Reply(
                rec["data"],
                rec["usage"],
                rec["reported_model"] or rec["requested_model"],
                key,
                True,
                rec["latency_s"],
                rec.get("provenance"),
            )
        if not self.live:
            raise LookupError(f"{self.backend['id']}: no recording for {kind} {key} (run live)")
        if self.adapter is None:  # built only on a miss: replay never needs a key or a CLI
            from kyb.evals.witness.backends import adapter

            self.adapter = adapter(self.backend)
        if self.adapter.prompt_version != self.prompt_version:
            raise ValueError("adapter prompt version differs from the recording key")
        t0 = time.monotonic()
        data, usage = self.adapter.call(kind, images, spec)
        latency = round(time.monotonic() - t0, 2)
        prov = dict(getattr(self.adapter, "last_provenance", None) or {})
        reported = prov.pop("reported_model", None)
        rec = {
            "backend": self.backend["id"],
            "requested_model": self.backend["model"],
            "reported_model": reported,
            "prompt_version": self.prompt_version,
            "request": key,
            "kind": kind,
            "spec": spec,
            "images": [sha256_bytes(i) for i in images],
            "data": data,
            "usage": usage,
            "latency_s": latency,
            "provenance": prov,
        }
        self.dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n", "utf-8")
        self.calls.append(self._call(key, "live", rec))
        return Reply(data, usage, reported or self.backend["model"], key, False, latency, prov)
