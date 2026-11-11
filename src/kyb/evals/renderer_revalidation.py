"""Collect an origin revalidation candidate; never admit a legacy binding.

The fixed native-job entrypoint owns host/source admission. Its context callback
must verify the actual job before and after this operation. The private scope is
not an MCP/CLI flag or an attestation against malicious Python/repository writers.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from collections.abc import Callable, Iterator
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from kyb.evals.paths import datasets
from kyb.evals.renderer import Comparison, RendererIncomparable, check_historical_maps, compare
from kyb.evals.renderer_runtime import capture_runtime_identity
from kyb.evals.witness import comparator, fullrun
from kyb.evals.witness.evaluate import recordings
from kyb.evals.witness.record import Recorder

_SCOPE: ContextVar[tuple[Path, Any] | None] = ContextVar(
    "renderer_revalidation_scope", default=None
)


class _RecordedOnly(Recorder):
    """A cache miss is terminal; even an accidental live-mode mutation refuses."""

    def __init__(self, run_dir: Path, backend: dict):
        super().__init__(
            recordings(backend["id"]), backend, seeds=fullrun.seeds(run_dir), live=False
        )
        self.run_dir = run_dir.resolve()

    def ask(self, kind, images, spec):
        if not _permitted(self.run_dir, self):
            raise RendererIncomparable("unscoped or non-replay historical recorder")
        reply = super().ask(kind, images, spec)
        if not reply.replayed or self.calls[-1]["source"] != "product seed":
            raise ValueError("historical revalidation requires original product seeds")
        return reply


def _permitted(run_dir: Path, witness: Any) -> bool:
    scope = _SCOPE.get()
    return (
        scope is not None
        and type(witness) is _RecordedOnly
        and scope[0] == run_dir.resolve() == witness.run_dir
        and scope[1] is witness
        and witness.live is False
        and witness.adapter is None
        and witness.mode == "replay"
    )


@contextlib.contextmanager
def _candidate_scope(run_dir: Path, witness: _RecordedOnly) -> Iterator[None]:
    if _SCOPE.get() is not None or type(witness) is not _RecordedOnly:
        raise RendererIncomparable("invalid or nested historical scope")
    token = _SCOPE.set((run_dir.resolve(), witness))
    try:
        if not _permitted(run_dir, witness):
            raise RendererIncomparable("historical recorder is not offline")
        yield
    finally:
        _SCOPE.reset(token)


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _inputs(root: Path) -> dict[str, str]:
    """Hash all original run records and packet PDFs; reject alternate corpora."""
    root = root.resolve()
    if datasets().resolve() != root / "datasets" or fullrun.PACKETS.resolve() != (
        root / "examples/packets"
    ):
        raise ValueError("historical corpus root differs from checked source")
    runs = fullrun.runs()
    if {r.name for r in runs} != set(fullrun._BASELINES) or len(runs) != len(fullrun._BASELINES):
        raise ValueError("historical corpus is incomplete")
    files = set()
    for run in runs:
        if run.is_symlink() or run.parent.resolve() != root / "datasets/witness/runs":
            raise ValueError("unexpected historical run root")
        for path in run.rglob("*"):
            if path.is_symlink():
                raise ValueError("historical symlink is not an immutable input")
            if path.is_file():
                files.add(path)
            elif not path.is_dir():
                raise ValueError("historical input is not a regular file or directory")
        case, _ = fullrun._case_run(run)
        packet = fullrun.PACKETS / case
        if packet.is_symlink():
            raise ValueError("historical packet symlink")
        pdfs = list(packet.glob("*.pdf"))
        if not pdfs or any(p.is_symlink() for p in pdfs):
            raise ValueError("historical packet input unavailable")
        files.update(pdfs)
    return {str(p.relative_to(root)): _hash(p) for p in sorted(files)}


def create_candidate(root: Path, context: Callable[[], dict]) -> dict:
    """Execute the same pinned product reverify/maps with no provider fallback.

    Called only after the fixed native entrypoint validates the actual approved
    renderer job; context is checked again afterward. The output is evidence,
    not an accepted sidecar or a scorecard promotion. Original pytest partition
    execution and the receiver's independent provenance/admission remain separate.
    """
    job = context()
    before = capture_runtime_identity()
    if compare(before, before).state is not Comparison.MATCH:
        raise RendererIncomparable("incomplete revalidation identity", category="profile-schema")
    inputs = _inputs(root)
    versions = comparator.versions()
    maps, call_counts, errors = {}, {}, []
    provider_calls = 0
    for run in fullrun.runs():
        recorder = _RecordedOnly(run, fullrun.recording_backend(run))
        try:
            with _candidate_scope(run, recorder):
                result = fullrun.reverify(run, recorder)
            maps[run.name] = result["verdicts"]
            if (
                not recorder.calls
                or any(c["source"] != "product seed" for c in recorder.calls)
                or result["claims"] != len(result["verdicts"])
                or result["verified"] != sum(result["verdicts"].values())
            ):
                raise ValueError("historical replay evidence incomplete")
            call_counts[run.name] = len(recorder.calls)
        except LookupError, ValueError, RendererIncomparable, OSError, TypeError:
            errors.append({"run": run.name, "reason": "replay-or-original-seed-evidence-failed"})
        provider_calls += sum(c.get("source") == "live" for c in recorder.calls)
    map_evidence = {"schema": "kyb-renderer-verdicts-v1", "comparator": versions, "verdicts": maps}
    map_check = None
    try:
        map_check = check_historical_maps(map_evidence)
    except ValueError:
        errors.append({"reason": "complete-original-verdict-maps-not-reproduced"})
    after = capture_runtime_identity()
    comparison = compare(before, after)
    if _inputs(root) != inputs or comparator.versions() != versions or context() != job:
        raise ValueError("historical inputs/source/context changed")
    encoded = json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema": "kyb-renderer-revalidation-candidate-v1",
        "kind": "revalidation-candidate",
        "status": "failed"
        if errors
        else "revalidated"
        if comparison.state is Comparison.MATCH
        else "incomparable",
        "job": job,
        "profile_before": before,
        "profile_after": after,
        "renderer_comparison": comparison.state.value,
        "incomparability_stage": "drift" if comparison.state is not Comparison.MATCH else None,
        "input_manifest": inputs,
        "input_manifest_sha256": hashlib.sha256(encoded).hexdigest(),
        "map_evidence": map_evidence,
        "map_check": map_check,
        "product_seed_requests": call_counts,
        "new_provider_calls": provider_calls,
        "errors": errors,
        "recording_time_identity": "unknown",
        "legacy_binding": "candidate-unadmitted",
        "pytest_renderer_partition": "not executed by this candidate collector",
        "product_pass": False,
    }
