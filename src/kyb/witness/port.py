"""The witness port: the only model calls the system makes itself (spec, spec).

Two requests, both blind:
- `layout(pages, kinds)`: a claim-blind pass over whole pages; it sees the page images only, never
  the text layer or any claim, and returns blocks plus material facts of the given kinds;
- `transcribe(crops)`: one request per document with every crop; it sees the crop images only.

Every response is recorded, content-addressed by the request, under `silver/witness/raw/`, so a
run replays with zero calls. `mode="replay"` refuses to call a model at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from kyb.read.harness import model_family
from kyb.util.fsutil import contained, publish_exclusive
from kyb.util.hashing import sha256_json
from kyb.witness.errors import (
    ROUTING_VERSION,
    WitnessAnomalous,
    WitnessRefused,
    WitnessUnavailable,
)
from kyb.witness.evidence import claude_terminal, diagnostic_kinds, gemini_terminal, require


def read_record(path: Path) -> dict:
    from kyb.witness.gemini_outcome import decode

    try:
        record = decode(path.read_bytes())
    except Exception:
        raise WitnessRefused("witness recording is malformed") from None
    if not isinstance(record, dict):
        raise WitnessRefused("witness recording is not an object")
    return record


@dataclass(frozen=True)
class Response:
    data: dict[str, Any]
    usage: dict[str, Any]
    model: str
    request_hash: str
    replayed: bool
    provenance: dict[str, Any] = field(default_factory=dict)


class Adapter(Protocol):
    model: str
    prompt_version: str

    def call(self, kind: str, images: list[bytes], spec: dict[str, Any]) -> tuple[dict, dict]: ...


class Witness:
    def __init__(self, root: Path, adapter: Adapter | None, *, mode: str = "auto",
                 model: str | None = None, prompt_version: str | None = None,
                 attempt: int = 0, sleep: Any = None) -> None:  # fmt: skip
        """`model` and `prompt_version` pin what a replay must match when there is no adapter.
        `attempt` > 0 is a self-heal retry: its own request identity, recorded and replayable."""
        import time

        if mode not in ("auto", "replay", "live"):
            raise ValueError(f"witness mode {mode!r}")
        from kyb.witness import gemini

        self.root, self.adapter, self.mode, self.attempt = root, adapter, mode, attempt
        self.model = adapter.model if adapter else (model or gemini.MODEL)
        self.prompt_version = (
            adapter.prompt_version if adapter else (prompt_version or gemini.PROMPT_VERSION)
        )
        self._sleep = sleep or time.sleep

    def _request(self, kind: str, images: list[bytes], spec: dict) -> tuple[str, Path]:
        from kyb.util.hashing import sha256_bytes

        identity = {"kind": kind, "model": self.model, "prompt": self.prompt_version,
                    "spec": spec, "images": [sha256_bytes(i) for i in images]}  # fmt: skip
        if transport := getattr(self.adapter, "transport", None):
            from kyb.witness.adapters import request

            prompt, schema = request(kind, spec)
            identity |= {"transport": transport, "schema": sha256_json(schema),
                         "prompt_hash": sha256_json(prompt),
                         "validation_version": getattr(self.adapter, "validation_version",
                                                       ROUTING_VERSION),
                         "record_version": ROUTING_VERSION}  # fmt: skip
            if route := getattr(self.adapter, "route_identity", None):
                identity["route"] = route
            if self.attempt:
                identity["attempt"] = self.attempt
        key = sha256_json(identity)
        return key, contained(self.root, f"silver/witness/raw/{key.split(':', 1)[1]}")

    def ask(self, kind: str, images: list[bytes], spec: dict[str, Any]) -> Response:
        """Replay the pinned response (revision 0) or call the model.

        Every live answer is kept as its own revision (`0.json`, `1.json`, ...), so a live re-read
        for drift never collides with the pinned answer and is itself replayable.
        """
        integrity_error = WitnessRefused if hasattr(self.adapter, "transport") else PermissionError
        key, folder = self._request(kind, images, spec)
        pinned = folder / "0.json"
        if pinned.exists() and self.mode != "live":
            if hasattr(self.adapter, "transport") and (folder / "failed" / "0.json").exists():
                prior = read_record(folder / "failed" / "0.json")
                if prior.get("request") != key or prior.get("content_hash") != sha256_json(
                    {k: v for k, v in prior.items() if k != "content_hash"}
                ):
                    raise WitnessRefused("prior witness failure integrity is invalid")
                self._versions(prior)
                self._outcome(prior)
                if prior["failure_kind"] == "refused":
                    raise WitnessRefused("recorded refusal evidence remains terminal")
            rec = read_record(pinned)
            if rec.get("request") != key:
                raise integrity_error("witness recording names a different request")
            if hasattr(self.adapter, "transport") and not rec.get("content_hash"):
                raise integrity_error("new witness recording is missing its content hash")
            if hasattr(self.adapter, "transport") and rec.get("status") != "complete":
                raise integrity_error("new witness response has no complete status")
            if hasattr(self.adapter, "transport"):
                self._versions(rec)
            if "content_hash" in rec and rec["content_hash"] != sha256_json(
                {k: v for k, v in rec.items() if k != "content_hash"}
            ):
                raise integrity_error("witness recording content hash mismatch")
            if hasattr(self.adapter, "transport"):
                self._outcome(rec)
            if rec.get("status") == "failed":
                raise RuntimeError("recorded witness attempt failed; explicit live retry required")
            return Response(rec["data"], rec["usage"], rec["model"], key, replayed=True,
                            provenance=rec.get("provenance") or {})  # fmt: skip
        if (folder / "failed" / "0.json").exists() and self.mode != "live":
            failed = read_record(folder / "failed" / "0.json")

            # 1. Check request identity and content_hash integrity
            if (
                failed.get("request") != key
                or not failed.get("content_hash")
                or failed["content_hash"]
                != sha256_json({k: v for k, v in failed.items() if k != "content_hash"})
            ):
                raise integrity_error("witness failure recording integrity mismatch")

            # 2. Check versions and outcome if transport exists
            if hasattr(self.adapter, "transport"):
                self._versions(failed)
                self._outcome(failed)

            # 3. Check typed-status check
            if failed.get("status") != "failed" or (
                failed.get("failure_kind"),
                failed.get("failure_type"),
            ) not in (
                ("unavailable", "WitnessUnavailable"),
                ("refused", "WitnessRefused"),
                ("anomalous", "WitnessAnomalous"),
            ):
                raise WitnessRefused("recorded failure has incomplete typed refusal status")

            try:
                prov = failed.get("provenance") or {}
                kinds = diagnostic_kinds(prov.get("error_diagnostics") or [])
            except Exception:
                kinds = set()
            is_retryable = failed.get("failure_kind") == "unavailable" and bool(
                {"transient", "rate_limit"} & kinds
            )
            failed_count = 0
            while (folder / "failed" / f"{failed_count}.json").exists():
                failed_count += 1

            if is_retryable and failed_count <= 2:
                if self.mode != "replay" and self.adapter is not None:
                    prov = failed.get("provenance") or {}
                    retry_after = prov.get("retry_after") or prov.get("Retry-After")
                    if retry_after is not None:
                        try:
                            # a recorded Retry-After is honoured but bounded like the backoff
                            delay = min(max(float(retry_after), 0.0), 60.0)
                        except ValueError, TypeError:
                            delay = None
                    else:
                        delay = None

                    if delay is None:
                        n = failed_count - 1
                        delay = min(15.0 * (2**n), 60.0)

                    self._sleep(delay)
            else:
                recorded = (failed.get("provenance") or {}).get("envelope_anomalies") or []
                message = "recorded attempt failed; explicit live retry required"
                if failed.get("failure_kind") == "anomalous":
                    raise WitnessAnomalous(message, recorded)
                failure = (
                    WitnessUnavailable
                    if failed.get("failure_kind") == "unavailable"
                    else WitnessRefused
                )
                exc = failure(message)
                exc.anomalies = recorded
                raise exc
        if self.mode == "replay" or self.adapter is None:
            missing = WitnessUnavailable if hasattr(self.adapter, "transport") else LookupError
            raise missing(f"no recorded witness response for {kind} ({key}) in replay mode")
        try:
            data, usage = self.adapter.call(kind, images, spec)
            error = None
        except Exception as exc:
            if isinstance(exc, WitnessRefused | WitnessUnavailable | WitnessAnomalous):
                error = exc
            else:
                error = WitnessRefused(f"witness response failed ({type(exc).__name__})")
            data, usage = None, {}
        source_provenance = getattr(self.adapter, "last_provenance", {})
        provenance = dict(source_provenance) if isinstance(source_provenance, dict) else {}
        model = (provenance.get("reported_model") or "unknown"
                 if hasattr(self.adapter, "transport") else self.adapter.model)  # fmt: skip
        rec = {"request": key, "kind": kind, "model": model,
               "requested_model": self.adapter.model, "provenance": provenance,
               "prompt_version": self.adapter.prompt_version, "data": data,
               "usage": usage, "status": "failed" if error else "complete"}  # fmt: skip
        if hasattr(self.adapter, "transport"):
            rec |= {
                "record_version": ROUTING_VERSION,
                "validation_version": getattr(self.adapter, "validation_version", ROUTING_VERSION),
            }
        if error:
            rec["failure_type"] = type(error).__name__
            rec["failure_kind"] = (
                "unavailable"
                if isinstance(error, WitnessUnavailable)
                else "anomalous"
                if isinstance(error, WitnessAnomalous)
                else "refused"
            )
        if hasattr(self.adapter, "transport"):
            try:
                self._outcome(rec)
            except WitnessRefused:
                # Missing/contradictory adapter markers never become availability.
                error = WitnessRefused("witness adapter outcome contract refused")
                provenance |= {
                    "classification": "refused",
                    "refusal_evidence": [
                        *(
                            provenance.get("refusal_evidence")
                            if isinstance(provenance.get("refusal_evidence"), list)
                            else []
                        ),
                        "outcome_contract_invalid",
                    ],
                    "normal_terminal": False,
                }
                rec |= {
                    "status": "failed",
                    "failure_kind": "refused",
                    "failure_type": "WitnessRefused",
                    "data": None,
                    "usage": {},
                }
        rec["content_hash"] = sha256_json(rec)
        body = (json.dumps(rec, indent=2, ensure_ascii=False) + "\n").encode()
        n = 0
        destination = folder / "failed" if error else folder
        while not publish_exclusive_new(destination / f"{n}.json", body):
            n += 1
        if error:
            error.anomalies = list(provenance.get("envelope_anomalies") or [])
            raise error
        return Response(data, usage, model, key, replayed=False, provenance=provenance)

    def _versions(self, record: dict) -> None:
        if record.get("record_version") != ROUTING_VERSION or record.get(
            "validation_version"
        ) != getattr(self.adapter, "validation_version", ROUTING_VERSION):
            raise WitnessRefused("witness recording validation identity changed; new run required")

    def _outcome(self, record: dict) -> None:
        required = {
            "request": str,
            "kind": str,
            "model": str,
            "requested_model": str,
            "prompt_version": str,
            "usage": dict,
            "provenance": dict,
            "status": str,
            "record_version": str,
            "validation_version": str,
        }
        require(
            all(type(record.get(k)) is t for k, t in required.items()),
            "witness required cache fields missing or invalid",
        )
        require(
            record["kind"] in ("layout", "transcribe")
            and bool(record["model"])
            and record["requested_model"] == self.model
            and record["prompt_version"] == self.prompt_version,
            "witness cached identity is incomplete or contradictory",
        )
        provenance = record.get("provenance")
        if not isinstance(provenance, dict):
            raise WitnessRefused("recorded outcome has invalid provenance")
        if (
            provenance.get("transport") != self.adapter.transport
            or provenance.get("validation_version") != record.get("validation_version")
            or provenance.get("family") != self.adapter.family
        ):
            raise WitnessRefused("recorded outcome provenance identity is incomplete")
        if provenance.get("route") != getattr(self.adapter, "route_identity", None):
            raise WitnessRefused("recorded outcome route/project identity differs from adapter")
        classification = provenance.get("classification")
        if "data" not in record or not isinstance(record.get("usage"), dict):
            raise WitnessRefused("recorded outcome payload metadata is incomplete")
        evidence = provenance.get("refusal_evidence")
        if not isinstance(evidence, list) or any(not isinstance(e, str) for e in evidence):
            raise WitnessRefused("recorded outcome has missing or invalid refusal evidence")
        if evidence and classification != "refused":
            raise WitnessRefused("recorded refusal evidence remains terminal")
        anomalies = provenance.get("envelope_anomalies", [])
        if not isinstance(anomalies, list) or any(
            not isinstance(a, dict) or a.get("severity") not in ("unknown", "error")
            for a in anomalies
        ):
            raise WitnessRefused("recorded envelope anomalies are malformed")
        if any(k in provenance for k in ("failure_kind", "failure_type", "status")):
            raise WitnessRefused("provenance contains contradictory outcome markers")
        if record.get("status") == "complete":
            require(
                record["model"] == provenance.get("reported_model")
                and model_family(record["model"]) == self.adapter.family,
                "witness actual model evidence missing or contradictory",
            )
            if (
                classification != "complete"
                or provenance.get("normal_terminal") is not True
                or anomalies
                or evidence
                or "failure_kind" in record
                or "failure_type" in record
                or not isinstance(record.get("data"), dict)
            ):
                raise WitnessRefused("complete witness outcome conflicts with failure evidence")
            self._provider_evidence(provenance, "complete")
            payload_hash = sha256_json(record["data"])
            require(
                provenance.get("final_output_hash") == payload_hash,
                "witness final-output hash conflicts with replayed payload",
            )
            if self.adapter.transport == "claude-p":
                final = [c for c in provenance["schema_checks"] if c["source"] == "final"]
                require(
                    final[0]["value_hash"] == payload_hash,
                    "Claude final schema hash conflicts with payload",
                )
            else:
                require(
                    provenance["data_hash"] == provenance["schema_hash"] == payload_hash,
                    "Gemini normalized schema/final hash conflicts with payload",
                )
            from kyb.witness import gemini
            from kyb.witness.adapters import json_schema, validate

            schema = (
                gemini.LAYOUT_SCHEMA if record["kind"] == "layout" else gemini.TRANSCRIBE_SCHEMA
            )
            validate(record["data"], json_schema(schema))
            return
        pair = (record.get("failure_kind"), record.get("failure_type"))
        if (
            record.get("status") != "failed"
            or pair
            not in (
                ("refused", "WitnessRefused"),
                ("unavailable", "WitnessUnavailable"),
                ("anomalous", "WitnessAnomalous"),
            )
            or classification != record.get("failure_kind")
            or (classification == "anomalous" and (not anomalies or evidence))
            or (classification == "unavailable" and anomalies)
            or provenance.get("normal_terminal") is not False
            or record.get("data") is not None
        ):
            raise WitnessRefused("recorded failure has inconsistent typed refusal status")
        if classification == "unavailable":
            self._provider_evidence(provenance, "unavailable")

    def _provider_evidence(self, provenance: dict, outcome: str) -> None:
        if self.adapter.transport == "claude-p":
            require(
                claude_terminal(provenance) == outcome,
                "Claude retained terminal evidence conflicts",
            )
        elif self.adapter.transport in ("gemini-api", "vertex-gcp"):
            gemini_terminal(provenance, outcome)
        else:
            raise WitnessRefused("unsupported witness transport cannot establish an outcome")


def publish_exclusive_new(path: Path, body: bytes) -> bool:
    """Publish `body` at `path` if nothing is there yet; False if the slot is already taken."""
    if path.exists():
        return False
    try:
        publish_exclusive(path, body)
    except PermissionError:
        return False
    return True


def make_witness(root: Path, mode: str):
    """kyb's own spec router. the harness never supplies a product witness."""
    from kyb.witness.strata import RoutedWitness

    return RoutedWitness(root, mode=mode)
