"""Filesystem safety for records: validated ids, contained paths, atomic and exclusive publication.

Records are written once (spec). Every write in the core goes through `publish_exclusive` or
`atomic_replace`, every path through `contained`, and every id through `safe_id`, so a rejected
operation never leaves partial or replaced bytes behind.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import re
import tempfile
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,80}")


class UnsafePath(ValueError):
    """A path or identifier that could escape its subtree."""


def safe_id(value: str, what: str = "id") -> str:
    if not _ID.fullmatch(value) or ".." in value:
        raise UnsafePath(f"invalid {what}: {value!r}")
    return value


def safe_rel(rel: str) -> PurePosixPath:
    p = PurePosixPath(rel)
    if p.is_absolute() or not p.parts or any(part in ("", ".", "..") for part in p.parts):
        raise UnsafePath(f"not a clean relative path: {rel!r}")
    return p


def contained(root: Path, rel: str | PurePosixPath) -> Path:
    """`root/rel`, refusing traversal, absolute paths and symlinks anywhere below `root`.

    This is a check, not a capability: a parent directory swapped for a symlink between this call
    and the write can still escape (TOCTOU). That is accepted for spec's trusted local mode; the
    hosted tiers and hardened local mode put the boundary in the OS (mounts, a separate user).
    """
    root = root.resolve()
    rp = safe_rel(str(rel))
    cur = root
    for part in rp.parts:
        cur = cur / part
        if cur.is_symlink():
            raise UnsafePath(f"symlink in record path: {cur}")
    if not cur.resolve().is_relative_to(root):
        raise UnsafePath(f"{rel} escapes {root}")
    return cur


def _tmp_in(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=directory, prefix=".tmp-")
    os.close(fd)
    return Path(name)


def _write_synced(path: Path, data: bytes) -> None:
    with path.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def publish_exclusive(path: Path, data: bytes) -> bool:
    """Create `path` with `data` exactly once.

    Returns True if created, False if an identical file already existed. Raises PermissionError if
    a different file exists. The target never holds partial bytes.
    """
    if path.exists():
        if path.read_bytes() == data:
            return False
        raise PermissionError(f"{path}: records are written once; different bytes already exist")
    tmp = _tmp_in(path.parent)
    try:
        _write_synced(tmp, data)
        try:
            os.link(tmp, path)  # atomic, and fails if another writer got there first
        except FileExistsError:
            if path.read_bytes() != data:
                raise PermissionError(
                    f"{path}: written concurrently with different bytes"
                ) from None
            return False
        return True
    finally:
        tmp.unlink(missing_ok=True)


def atomic_replace(path: Path, data: bytes) -> None:
    """Replace a mutable index file (catalog) atomically: readers see old or new, never partial."""
    tmp = _tmp_in(path.parent)
    try:
        _write_synced(tmp, data)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


@contextlib.contextmanager
def locked(lockfile: Path, *, shared: bool = False) -> Iterator[None]:
    """A stable lockfile (never replaced), so locks survive atomic replacement of the data file."""
    lockfile.parent.mkdir(parents=True, exist_ok=True)
    with lockfile.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def action_lock(root: Path) -> contextlib.AbstractContextManager[None]:
    """One action at a time per workspace (the local stand-in for spec's case lease)."""
    return locked(root / ".kyb.lock")
