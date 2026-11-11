"""spec routing and enforcement. Unknown attestation or actual served model fails closed.

The independent pre-read layout uses the L6-selected backend without claiming a stratum.
Transcriptions route only after the document reader has stopped and can be attested. B is
exact-model W1; A enforces a different actual family; A+B retains both independent records.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from kyb.read.harness import model_family
from kyb.witness.adapters import AgyWitness, ClaudeWitness, CodexWitness, request, validate
from kyb.witness.errors import WitnessAnomalous, WitnessRefused, WitnessUnavailable
from kyb.witness.gemini import GeminiWitness
from kyb.witness.port import Witness
from kyb.witness.vertex import VertexWitness

# L6 historical benchmark: Pro 14/14, 0/22 negatives, 49/205 full-run; Flash 13/14,
# 0/22, 40/205; Claude 14/14, 0/22, 26/205. Prompt/comparator coverage limits in l8.yaml.
# CLI provenance gaps preclude ranking Codex/agy as successful fallbacks.
DEFAULT_A = ("vertex-gcp:gemini-3.1-pro-preview", "vertex-gcp:gemini-3.8-flash",
             "claude-p:claude-opus-5-5")  # fmt: skip


class RoutingRefused(WitnessRefused):
    """No permissible route. Verification must record the refusal and retain claimed status."""


def adapter(backend: str):
    transport, sep, model = backend.partition(":")
    if not sep or not model:
        raise RoutingRefused("backend must name transport:exact-model")
    if transport == "gemini-api":
        return GeminiWitness(model=model, retries=0)
    if transport == "vertex-gcp":
        return VertexWitness(model=model, retries=0)
    cls = {"claude-p": ClaudeWitness, "codex-exec": CodexWitness, "agy": AgyWitness}.get(transport)
    if cls is None:
        raise RoutingRefused("unknown witness transport")
    return cls(model)


@dataclass(frozen=True)
class Config:
    stratum: str = "B"
    a_backends: tuple[str, ...] = DEFAULT_A

    def __post_init__(self) -> None:
        if self.stratum not in {"B", "A", "A+B"} or not self.a_backends:
            raise RoutingRefused("stratum must be B, A or A+B with explicit A backends")

    @classmethod
    def environment(cls):
        selected = os.environ.get("KYB_WITNESS_A_BACKENDS")
        return cls(os.environ.get("KYB_WITNESS_STRATUM", "B"),
                   tuple(selected.split(",")) if selected else DEFAULT_A)  # fmt: skip


class CheckedWitness:
    """A transport may not promote its requested model pin into actual-model evidence."""

    def __init__(self, port: Witness, stratum: str, reader: dict) -> None:
        self.port, self.stratum, self.reader = port, stratum, reader
        self.mode = port.mode

    def ask(self, kind: str, images: list[bytes], spec: dict):
        reply = self.port.ask(kind, images, spec)
        actual = reply.provenance.get("reported_model")
        if (
            not isinstance(actual, str)
            or not actual
            or actual != reply.model
            or model_family(actual) == "unknown"
        ):
            raise RoutingRefused("witness actual served model is missing or unknown")
        if model_family(actual) != self.port.adapter.family:
            raise RoutingRefused("witness actual family conflicts with its declared transport")
        if (
            self.port.adapter.transport == "claude-p"
            and reply.provenance.get("no_data_tools") is not True
        ):
            raise RoutingRefused("Claude runtime no-data-tools evidence is missing")
        if self.stratum == "B" and actual != self.reader["reader_model"]:
            raise RoutingRefused("B witness actual model differs from the reader's actual model")
        if self.stratum == "A" and model_family(actual) == self.reader["reader_family"]:
            raise RoutingRefused("A witness actual family equals the reader family")
        validate(reply.data, request(kind, spec)[1])
        return reply


class FallbackWitness:
    """Only transport failures can advance the evaluated fallback order. Every failure is kept.

    Self-heal is bounded: an anomalous attempt (unknown envelope shapes) is retried ONCE on the
    same backend; if the retry is also anomalous, ONE attempt is made on the next configured
    backend of this stratum. If that is anomalous too, the claim stays CLAIMED with the anomalies
    attached (`WitnessAnomalous`). Every attempt is its own recorded request, so REPLAY follows the
    same path with zero calls.
    """

    def __init__(self, router, backends: tuple[str, ...], reader: dict, stratum: str):
        self.router, self.backends, self.reader, self.stratum = router, backends, reader, stratum
        self.mode = router.mode
        self.attempts: list[dict] = []

    def ask(self, kind: str, images: list[bytes], spec: dict):
        anomalous: list[dict] = []
        exhausted = 0  # backends that stayed anomalous after their allowed attempts
        for backend in self.backends:
            if exhausted >= 2:
                break
            try:
                candidate = self.router.factory(backend)
            except WitnessRefused:
                self.attempts.append({"backend": backend, "status": "assurance_refused"})
                raise
            except Exception as exc:
                self.attempts.append({"backend": backend, "status": "assurance_refused"})
                raise RoutingRefused(
                    f"witness route has invalid evidence ({type(exc).__name__})"
                ) from None
            if self.stratum == "A" and candidate.family == self.reader["reader_family"]:
                self.attempts.append({"backend": backend, "status": "same_family_refused"})
                continue
            allowed = 2 if not anomalous else 1
            healed_here = False
            for attempt in range(allowed):
                try:
                    port = Witness(self.router.root, candidate, mode=self.mode, attempt=attempt)
                    result = CheckedWitness(port, self.stratum, self.reader).ask(kind, images, spec)
                    self.attempts.append(
                        {"backend": backend, "status": "complete", "attempt": attempt,
                         "request": result.request_hash}
                    )  # fmt: skip
                    return result
                except WitnessAnomalous as exc:
                    self.attempts.append(
                        {"backend": backend, "status": "anomalous", "attempt": attempt,
                         "anomalies": exc.anomalies}
                    )  # fmt: skip
                    anomalous.extend(exc.anomalies)
                    healed_here = True
                except WitnessRefused as exc:
                    self.attempts.append({"backend": backend, "status": "assurance_refused",
                                          "attempt": attempt,
                                          "anomalies": getattr(exc, "anomalies", [])})  # fmt: skip
                    raise
                except WitnessUnavailable as exc:
                    self.attempts.append({"backend": backend, "status": "unavailable",
                                          "attempt": attempt, "reason": str(exc),
                                          "anomalies": getattr(exc, "anomalies", [])})  # fmt: skip
                    break
                except Exception as exc:
                    self.attempts.append({"backend": backend, "status": "assurance_refused"})
                    raise RoutingRefused(
                        f"witness route has invalid evidence ({type(exc).__name__})"
                    ) from None
            if healed_here:
                exhausted += 1
        if anomalous:
            raise WitnessAnomalous(
                "witness envelope anomalies persisted after bounded self-heal", anomalous
            )
        raise RoutingRefused("no permissible witness backend completed")


class RoutedWitness:
    def __init__(self, root: Path, *, mode: str = "auto", config: Config | None = None,
                 factory: Callable = adapter) -> None:  # fmt: skip
        self.root, self.mode = root, mode
        self.config, self.factory = config or Config.environment(), factory
        if mode not in {"auto", "live", "replay"}:
            raise ValueError("unknown witness mode")

    def ask(self, kind: str, images: list[bytes], spec: dict):
        if kind != "layout":
            raise RoutingRefused("transcription routing requires a document's reader attestation")
        # Layout precedes the reader, so no W1/W2 assertion is made here.
        reader = {"reader_model": None, "reader_family": "unknown"}
        return FallbackWitness(self, self.config.a_backends, reader, "layout").ask(
            kind, images, spec
        )

    def for_document(self, sub: dict) -> dict[str, FallbackWitness]:
        from kyb.witness.verify import assurance

        reader = assurance(self.root, sub, "unknown")
        managed = reader.get("reader_origin") == "codex-managed"
        if managed and self.config.stratum != "A":
            raise RoutingRefused("managed Codex supports only independent stratum A")
        if (
            reader["context_now"] != "isolated"
            or not reader["reader_model"]
            or reader["reader_family"] == "unknown"
        ):
            raise RoutingRefused("reader actual model or eligible isolated attestation is missing")
        routes = {}
        if self.config.stratum in {"B", "A+B"}:
            transport = {"anthropic": "claude-p", "openai": "codex-exec",
                         "google": "vertex-gcp"}[reader["reader_family"]]  # fmt: skip
            routes["B"] = FallbackWitness(
                self, (f"{transport}:{reader['reader_model']}",), reader, "B"
            )
        if self.config.stratum in {"A", "A+B"}:
            routes["A"] = FallbackWitness(self, self.config.a_backends, reader, "A")
        return routes
