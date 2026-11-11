"""Renderer profile comparison primitives; a match is not a legacy binding.

File fingerprints and profile comparisons do not admit historical replays.
Runtime capture lives in renderer_runtime; callers cannot infer admission from a match.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class Comparison(Enum):
    MATCH = "MATCH"
    MISMATCH = "MISMATCH"
    UNKNOWN = "UNKNOWN"


class RendererIncomparable(RuntimeError):
    """Historical replay has no admissible renderer identity."""

    def __init__(self, message: str, *, category: str = "unavailable"):
        super().__init__(message)
        allowed = {
            "unavailable",
            "context",
            "native-probe",
            "file-fingerprint",
            "drift",
            "profile-schema",
        }
        self.category = category if category in allowed else "unavailable"


# Keys are intentionally explicit: silently dropping a missing library is unsafe.
COMPONENTS = (
    "platform",
    "architecture",
    "poppler",
    "cairo",
    "freetype",
    "fontconfig",
    "libpng",
    "lcms",
    "openjpeg",
    "pixman",
    "harfbuzz",
    "executable_sha256",
    "font_set_sha256",
    "dpi",
    "render_flags",
    "font_config_sha256",
)
_HASH = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ComparisonResult:
    state: Comparison
    components: tuple[str, ...]
    actual_hash: str | None
    expected_hash: str | None


def _valid(profile: Any) -> bool:
    if not isinstance(profile, dict) or set(profile) != set(COMPONENTS):
        return False
    if any(not isinstance(profile[k], str) or not profile[k] for k in COMPONENTS):
        return False
    return all(
        _HASH.fullmatch(profile[k])
        for k in ("executable_sha256", "font_set_sha256", "font_config_sha256")
    )


def _hash(profile: dict[str, str]) -> str:
    raw = json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def compare(actual: Any, expected: Any) -> ComparisonResult:
    """Compare complete profiles, exposing component names and hashes only.

    Unknown data never becomes a match, including extra fields whose semantics
    this version cannot understand. A profile match alone cannot admit replay.
    """
    av, ev = _valid(actual), _valid(expected)
    ah = _hash(actual) if av else None
    eh = _hash(expected) if ev else None
    if not av or not ev:
        return ComparisonResult(Comparison.UNKNOWN, (), ah, eh)
    changed = tuple(k for k in COMPONENTS if actual[k] != expected[k])
    return ComparisonResult(Comparison.MISMATCH if changed else Comparison.MATCH, changed, ah, eh)


def _file_state(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def capture_file_identity(
    renderers: dict[str, Path],
    *,
    fonts: Sequence[Path],
    configs: Sequence[Path],
    dependencies: Sequence[Path] = (),
) -> dict[str, str]:
    """Hash observed files, without exporting paths, names or raw contents.

    The caller must enumerate the *active* font/config files in the same runtime
    that renders; this function cannot attest that enumeration's completeness.
    These three fields are not a complete profile and cannot admit a binding.
    Actual host enumeration and both pre/post-render captures remain mandatory.
    """
    try:
        if set(renderers) != {"pdftoppm", "pdftocairo"} or not fonts or not configs:
            raise ValueError
        groups = [*renderers.values(), *fonts, *configs, *dependencies]
        if any(not isinstance(path, Path) or not path.is_absolute() for path in groups):
            raise ValueError
        # Keep original aliases only in memory so a symlink change is observable.
        resolved = {path: path.resolve(strict=True) for path in groups}
        states: dict[Path, tuple[int, ...]] = {}
        hashes: dict[Path, str] = {}
        for path in sorted(set(resolved.values())):
            before = path.stat()
            if not stat.S_ISREG(before.st_mode):
                raise ValueError
            with path.open("rb") as stream:
                if _file_state(os.fstat(stream.fileno())) != _file_state(before):
                    raise ValueError
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if _file_state(os.fstat(stream.fileno())) != _file_state(before):
                    raise ValueError
            states[path] = _file_state(before)
            hashes[path] = digest
        # Recheck all files after the entire observation, including original
        # aliases, rather than accepting a file that changed after its own read.
        for alias, path in resolved.items():
            if alias.resolve(strict=True) != path or _file_state(path.stat()) != states[path]:
                raise ValueError

        def aggregate(paths: Sequence[Path]) -> str:
            values = sorted(hashes[path] for path in {resolved[p] for p in paths})
            return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()

        binary_hashes = {role: hashes[resolved[path]] for role, path in renderers.items()}
        if dependencies:
            binary_hashes["dependencies_sha256"] = aggregate(dependencies)
        return {
            "executable_sha256": hashlib.sha256(
                json.dumps(binary_hashes, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "font_set_sha256": aggregate(fonts),
            "font_config_sha256": aggregate(configs),
        }
    except OSError, ValueError, RuntimeError, TypeError, AttributeError:
        # Native file errors include home/font filenames; never persist them.
        raise RendererIncomparable("renderer file identity unavailable or changed") from None


def require_legacy_binding(run_dir: Path | None = None, witness: Any = None) -> None:
    """Normal replay refuses; one private native revalidation scope may collect evidence.

    A scope authorizes only its exact run and offline recorder, not a compatible
    binding or ordinary evaluator PASS. No environment flag enables this path.
    """
    if run_dir is not None and witness is not None:
        from kyb.evals.renderer_revalidation import _permitted

        if _permitted(run_dir, witness):
            return
    raise RendererIncomparable("renderer identity and legacy binding unavailable")


def check_historical_maps(evidence: Any) -> dict[str, Any]:
    """Check companion verdicts against the independently recomputed pinned oracle.

    This checks evidence consistency only. It neither executes a replay nor
    admits a renderer/binding, and cannot resolve an incomparable scorecard case.
    Source, host, job provenance and partition completeness are separate gates.
    In particular JSON numbers must not compare equal to boolean verdicts.
    """
    from kyb.evals.witness import comparator, fullrun

    if type(evidence) is not dict or set(evidence) != {"schema", "comparator", "verdicts"}:
        raise ValueError("invalid historical verdict evidence fields")
    if evidence["schema"] != "kyb-renderer-verdicts-v1":
        raise ValueError("unknown historical verdict evidence schema")
    versions = comparator.versions()
    if type(evidence["comparator"]) is not dict or evidence["comparator"] != versions:
        raise ValueError("historical comparator identity mismatch")

    # A missing corpus directory must not shrink the required set of runs.
    runs = fullrun.runs()
    names = [run.name for run in runs]
    if len(set(names)) != len(names) or set(names) != set(fullrun._BASELINES):
        raise ValueError("historical corpus differs from the pinned run set")
    expected = {run.name: fullrun.expected_verdicts(run) for run in runs}
    maps = evidence["verdicts"]
    if type(maps) is not dict or set(maps) != set(expected):
        raise ValueError("historical verdict run set mismatch")
    for name, oracle in expected.items():
        actual = maps[name]
        if type(actual) is not dict or set(actual) != set(oracle):
            raise ValueError("historical verdict claim set mismatch")
        if any(type(value) is not bool for value in actual.values()):
            raise ValueError("historical verdicts must be JSON booleans")
        if actual != oracle:
            raise ValueError("historical verdict disagrees with the pinned oracle")

    # Return only safe identities/counts. An outer checksum or a claimed pass is
    # never used as the oracle, and no untrusted diagnostic text is copied.
    raw = json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema": "kyb-renderer-map-check-v1",
        "comparator": versions,
        "maps_sha256": hashlib.sha256(raw).hexdigest(),
        "run_counts": {name: len(expected[name]) for name in sorted(expected)},
        "claim_count": sum(len(values) for values in expected.values()),
        "replay_admitted": False,
    }
