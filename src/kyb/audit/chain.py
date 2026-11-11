"""Append-only, hash-chained audit log, built on open-kyb's `AuditEvent` contract.

The contract defines each event's `hash` over its fields and `prev_hash`, without fixing the
serialisation. We pin one, in one place:
- canonical JSON (sorted keys, compact separators) over exactly the eleven named fields;
- `timestamp` canonicalised to ISO-8601 UTC with microseconds and `Z`, whether it arrives as a
  datetime or as any ISO string the contract model emits;
- `cost_usd` hashed exactly as stored; `append` rounds it to 6 dp before storing.

A prefix of a valid chain is itself valid, so deleting the newest events is invisible to the chain
alone. Checkpoints (`ledger/checkpoints/cp-<seq>.json`, written once at the end of every action)
pin heads. `verify_workspace` checks **every** retained checkpoint, requires the init checkpoint,
and, at an action boundary, requires the tail to be pinned. `workspace_action` runs each action
between two such checks, so a new checkpoint can never paper over a broken older one. Locally a
checkpoint is only as trustworthy as the filesystem (deleting a middle checkpoint is not
detectable); hosted, checkpoints are signed and anchored in WORM storage (spec §8).
"""

from __future__ import annotations

import contextlib
import functools
import json
import math
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Lazy (PEP 810): a hook that never touches the chain must not pay for pydantic (spec, spec).
lazy from pydantic import TypeAdapter

lazy from kyb.contracts.kyb_decision_schema import AuditEvent
lazy from kyb.util.fsutil import action_lock, locked, publish_exclusive
lazy from kyb.util.hashing import sha256_json

GENESIS = "GENESIS"
HASHED_FIELDS = (
    "seq",
    "timestamp",
    "component",
    "action",
    "input_hash",
    "output_hash",
    "tokens_in",
    "tokens_out",
    "cost_usd",
    "detail",
    "prev_hash",
)


# pydantic's ValidationError is a ValueError, so naming it here would only force the import early
_BAD_RECORD = (ValueError, TypeError, KeyError, AttributeError, OverflowError)


@functools.cache
def _datetime_adapter() -> TypeAdapter[datetime]:
    return TypeAdapter(datetime)


class AuditCorruption(RuntimeError):
    """The log's framing or contents are damaged; appending would hide it."""


_TS_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
_CANONICAL_TS = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}Z")


def _plain_ts(value: object) -> str | None:
    """`value` if it is already a canonical timestamp (what `append` writes), else None.

    The stdlib parse must reproduce the string exactly, so a value it reads differently from the
    contract model (such as `24:00`) is never taken on this path: it goes to pydantic.
    """
    if not isinstance(value, str) or not _CANONICAL_TS.fullmatch(value):
        return None
    try:
        out = datetime.fromisoformat(value).astimezone(UTC).strftime(_TS_FORMAT)
    except ValueError, OverflowError:
        return None
    return value if out == value else None


def canonical_ts(value: datetime | str | int | float) -> str:
    """Any timestamp the contract accepts (datetime, ISO string, epoch number), as UTC with µs."""
    if (plain := _plain_ts(value)) is not None:
        return plain
    dt = value if isinstance(value, datetime) else _datetime_adapter().validate_python(value)
    if dt.tzinfo is None:
        raise ValueError("audit timestamps must be timezone-aware")
    try:
        return dt.astimezone(UTC).strftime(_TS_FORMAT)
    except OverflowError as exc:
        raise ValueError(f"timestamp out of range: {value!r}") from exc


_STR_FIELDS = ("component", "action", "input_hash", "output_hash", "prev_hash", "hash")
_INT_FIELDS = ("seq", "tokens_in", "tokens_out")


def _plainly_valid(ev: object) -> bool:
    """True when `ev` has exactly the shape `append` writes: exact JSON types, every field present,
    a canonical timestamp. The contract model accepts every such record (its lax mode coerces a
    superset), so only other shapes need pydantic."""
    return (
        isinstance(ev, dict)
        and all(type(ev.get(k)) is str for k in _STR_FIELDS)
        and all(type(ev.get(k)) is int for k in _INT_FIELDS)
        and type(ev.get("cost_usd")) is float  # `append` stores a float; an int may not fit one
        and type(ev.get("detail")) is dict
        and _plain_ts(ev.get("timestamp")) is not None
    )


def check_event(ev: object) -> None:
    """The contract gate: `ev` must validate as `AuditEvent`. A record of the shape `append`
    writes is admitted without importing pydantic (the hook fast lane, spec); anything else is
    validated by the model itself, so the verdict is the model's in every case."""
    if not _plainly_valid(ev):
        AuditEvent.model_validate(ev)


def event_hash(fields: dict[str, Any]) -> str:
    payload = {k: fields[k] for k in HASHED_FIELDS}
    payload["timestamp"] = canonical_ts(payload["timestamp"])
    cost = float(payload["cost_usd"])
    if not math.isfinite(cost):
        raise ValueError("cost_usd must be finite")
    payload["cost_usd"] = cost
    return sha256_json(payload)


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    events: int
    head: str
    error: str | None = None


def _scan(raw: bytes) -> tuple[VerifyResult, list[str]]:
    """Verify a chain snapshot. Returns the result and the hash of every valid event, by seq."""
    hashes: list[str] = []
    prev = GENESIS
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return VerifyResult(False, 0, prev, f"log is not UTF-8 ({exc})"), hashes
    for lineno, line in enumerate(text.splitlines(), start=1):
        n = len(hashes)
        try:
            ev = json.loads(line)
            check_event(ev)
            good_hash = event_hash(ev) == ev["hash"]
        except _BAD_RECORD as exc:
            return VerifyResult(False, n, prev, f"line {lineno}: unreadable record ({exc})"), hashes
        if ev["seq"] != n:
            return VerifyResult(False, n, prev, f"line {lineno}: seq {ev['seq']} != {n}"), hashes
        if ev["prev_hash"] != prev:
            return VerifyResult(
                False, n, prev, f"line {lineno}: prev_hash breaks the chain"
            ), hashes
        if not good_hash:
            return VerifyResult(
                False, n, prev, f"line {lineno}: hash does not match its fields"
            ), hashes
        prev = ev["hash"]
        hashes.append(prev)
    return VerifyResult(True, len(hashes), prev), hashes


class AuditLog:
    """One JSONL file; one line per `AuditEvent`."""

    def __init__(self, path: Path, clock: Callable[[], datetime] | None = None) -> None:
        self.path = path
        self._lock = path.with_name(path.name + ".lock")
        self._clock = clock or (lambda: datetime.now(UTC))

    def _tail_checked(self) -> tuple[int, str, bool]:
        """(last seq, last hash, needs_newline). Raises AuditCorruption on a damaged tail."""
        if not self.path.exists() or self.path.stat().st_size == 0:
            return -1, GENESIS, False
        raw = self.path.read_bytes()
        needs_newline = not raw.endswith(b"\n")
        last = raw.rstrip(b"\n").rsplit(b"\n", 1)[-1]
        try:
            ev = json.loads(last)
            check_event(ev)
            if event_hash(ev) != ev["hash"]:
                raise AuditCorruption("last event's hash does not match its fields")
        except _BAD_RECORD as exc:
            raise AuditCorruption(f"refusing to append after a damaged record: {exc}") from exc
        return ev["seq"], ev["hash"], needs_newline

    def append(
        self,
        component: str,
        action: str,
        *,
        input_hash: str,
        output_hash: str,
        detail: dict[str, Any] | None = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost_usd: float = 0.0,
    ) -> AuditEvent:
        fields = self.append_record(
            component,
            action,
            input_hash=input_hash,
            output_hash=output_hash,
            detail=detail,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
        )
        return AuditEvent.model_validate(fields)

    def append_record(
        self,
        component: str,
        action: str,
        *,
        input_hash: str,
        output_hash: str,
        detail: dict[str, Any] | None = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost_usd: float = 0.0,
    ) -> dict[str, Any]:
        """`append`, returning the stored record as a dict: the hooks use this, so recording a
        tool call needs hashing and canonical JSON only, never the pydantic model."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with locked(self._lock):
            seq, prev, needs_newline = self._tail_checked()
            fields: dict[str, Any] = {
                "seq": seq + 1,
                "timestamp": canonical_ts(self._clock()),
                "component": component,
                "action": action,
                "input_hash": input_hash,
                "output_hash": output_hash,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "cost_usd": round(float(cost_usd), 6),
                "detail": detail or {},
                "prev_hash": prev,
            }
            fields["hash"] = event_hash(fields)
            check_event(fields)  # the contract is the gate
            line = json.dumps(fields, ensure_ascii=False, sort_keys=True) + "\n"
            with self.path.open("a", encoding="utf-8") as f:
                f.write(("\n" if needs_newline else "") + line)
                f.flush()
        return fields

    def snapshot(self) -> bytes:
        if not self.path.exists():
            return b""
        with locked(self._lock, shared=True):
            return self.path.read_bytes()

    def verify(self) -> VerifyResult:
        return _scan(self.snapshot())[0]

    def event_at(self, seq: int) -> dict[str, Any] | None:
        for line in self.snapshot().decode("utf-8", "replace").splitlines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("seq") == seq:
                return ev
        return None


def _checkpoint_dir(root: Path) -> Path:
    return root / "ledger" / "checkpoints"


def _checkpoints(root: Path) -> list[dict[str, Any]]:
    """Every retained checkpoint, oldest first. Raises AuditCorruption on a malformed one."""
    out = []
    for path in sorted(_checkpoint_dir(root).glob("cp-*.json")):
        try:
            cp = json.loads(path.read_bytes().decode("utf-8"))
            ok = isinstance(cp["seq"], int) and cp["seq"] >= 0
            ok = ok and path.name == f"cp-{cp['seq']:08d}.json"
            ok = ok and isinstance(cp["head"], str)
        except (*_BAD_RECORD, json.JSONDecodeError) as exc:
            raise AuditCorruption(f"{path.name}: unreadable checkpoint ({exc})") from exc
        if not ok:
            raise AuditCorruption(f"{path.name}: checkpoint name and contents disagree")
        out.append(cp)
    return out


def latest_checkpoint(root: Path) -> dict[str, Any] | None:
    cps = _checkpoints(root)
    return cps[-1] if cps else None


def verify_workspace(
    root: Path, *, at_boundary: bool = False, bootstrap: bool = False
) -> VerifyResult:
    """The chain verifies, starts with the workspace's init, and agrees with every checkpoint.

    Checkpoints are read before the chain: a checkpoint only ever pins an event that already
    exists, and the chain only grows, so a concurrent writer cannot cause a false failure.
    `at_boundary` (no action in progress: the caller holds the action lock) additionally requires
    the newest event to be pinned, so a lost tail is detected too. An initialised workspace must
    keep its init checkpoint; the one exception is `bootstrap`, used only by `write_checkpoint`
    to pin the init event itself.
    """
    try:
        cps = _checkpoints(root)
    except AuditCorruption as exc:
        return VerifyResult(False, 0, GENESIS, str(exc))
    raw = AuditLog(root / "ledger" / "audit.jsonl").snapshot()
    result, hashes = _scan(raw)
    if not result.ok:
        return result
    n, head = result.events, result.head

    def fail(msg: str) -> VerifyResult:
        return VerifyResult(False, n, head, msg)

    if (root / "template.lock").exists():
        first = raw.split(b"\n", 1)[0]
        if n == 0 or json.loads(first).get("action") != "workspace.init":
            return fail("history before init is missing")
        pinning_init = bootstrap and not cps and n == 1
        if not pinning_init and (not cps or cps[0]["seq"] != 0):
            return fail("the init checkpoint (cp-00000000) is missing")
    for cp in cps:
        if cp["seq"] >= n:
            return fail(f"chain ends at seq {n - 1}, checkpoint expects {cp['seq']}")
        if hashes[cp["seq"]] != cp["head"]:
            return fail(f"event {cp['seq']} disagrees with its checkpoint (rewritten history)")
    if at_boundary and n and (not cps or cps[-1]["seq"] != n - 1):
        return fail(f"events after seq {cps[-1]['seq'] if cps else -1} are not checkpointed")
    return result


def write_checkpoint(root: Path) -> dict[str, Any] | None:
    """Pin the current head, after re-checking every existing checkpoint. Under the action lock.

    An empty log has nothing to pin, so nothing is written (a failed first init stays retryable).
    """
    result = verify_workspace(root, bootstrap=True)
    if not result.ok:
        raise AuditCorruption(f"cannot checkpoint a broken workspace: {result.error}")
    if result.events == 0:
        return None
    seq = result.events - 1
    cp = {"seq": seq, "head": result.head}
    path = _checkpoint_dir(root) / f"cp-{seq:08d}.json"
    publish_exclusive(path, (json.dumps(cp, sort_keys=True) + "\n").encode("utf-8"))
    return cp


@contextlib.contextmanager
def workspace_action(root: Path) -> Iterator[None]:
    """Run one action: take the lock, refuse a workspace that does not verify at a boundary, and
    pin whatever the action appended, even if it failed part-way (those events happened)."""
    with action_lock(root):
        before = verify_workspace(root, at_boundary=True)
        if not before.ok:
            raise AuditCorruption(
                f"refusing to act on a workspace that does not verify: {before.error}"
            )
        try:
            yield
        finally:
            write_checkpoint(root)


def chain_events(root: Path) -> list[dict[str, Any]]:
    """Every event in the workspace chain, oldest first. Verify the workspace before trusting it."""
    raw = AuditLog(root / "ledger" / "audit.jsonl").snapshot().decode("utf-8")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]
