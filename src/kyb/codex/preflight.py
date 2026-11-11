"""One-document, validation-only MCP callback for a code-owned managed Codex job.

Scope comes only from the private runner file. Proposed claims never enter the workspace or
the receipt log; the log retains hashes, fixed error categories and the bounded attempt count.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import stat
from pathlib import Path

from pydantic import ValidationError

from kyb.audit.chain import AuditCorruption, workspace_action
from kyb.read.schema import ClaimsSubmission
from kyb.read.submit import _check
from kyb.util.fsutil import safe_id
from kyb.util.hashing import sha256_json

SERVER_NAME = "kyb_preflight"
TOOL_NAME = "validate_claims"
MAX_ATTEMPTS = 3
SCOPE_FILE = "preflight-scope.json"
RECEIPTS_FILE = "preflight-receipts.jsonl"
SCOPE_KEYS = {"version", "role", "root", "case", "run", "doc", "doc_sha256"}
RECEIPT_KEYS = {"attempt", "claims_hash", "valid", "errors", "scope_hash", "role"}
ERROR_CODES = {"schema", "excerpt_missing", "read_plan", "document_hash", "page", "bbox",
               "text_layer", "layout_block", "linked_blocks", "core_validation",
               "workspace_unavailable", "budget_exhausted"}  # fmt: skip


def _unique_keys(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate JSON configuration key")
        out[key] = value
    return out


def _decode(raw: str):
    def invalid_constant(_):
        raise ValueError("invalid JSON constant")

    return json.loads(raw, object_pairs_hook=_unique_keys, parse_constant=invalid_constant)


def _private(path: Path, mode: int) -> None:
    info = path.lstat()
    expected = stat.S_ISDIR(info.st_mode) if mode == 0o700 else stat.S_ISREG(info.st_mode)
    if not expected or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != mode:
        raise PermissionError("preflight runtime path is not private and runner-owned")


def load_scope(job: Path) -> dict:
    """Load the exact runner-only scope. No role, root, or target is accepted from a model."""
    if not job.is_absolute():
        raise ValueError("preflight job directory must be absolute")
    _private(job, 0o700)
    path = job / SCOPE_FILE
    _private(path, 0o600)
    scope = _decode(path.read_text(encoding="utf-8"))
    # A model cannot smuggle a different target, role, or extra runtime setting into this file.
    if (not isinstance(scope, dict) or set(scope) != SCOPE_KEYS
            or type(scope.get("version")) is not int or scope["version"] != 1
            or scope.get("role") != "kyb-reader"
            or any(not isinstance(scope.get(k), str)
                   for k in SCOPE_KEYS - {"version"})):  # fmt: skip
        raise ValueError("invalid preflight scope configuration")
    root = Path(scope["root"])
    if not root.is_absolute() or root.resolve() != root or not (root / "template.lock").is_file():
        raise ValueError("preflight scope root is not an absolute workspace")
    for key in ("case", "doc"):
        safe_id(scope[key], key)
    if (not re.fullmatch(r"r[1-9]\d*", scope["run"])
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", scope["doc_sha256"])):  # fmt: skip
        raise ValueError("invalid preflight run or document hash")
    return scope


@contextlib.contextmanager
def _job_lock(job: Path):
    path = job / "preflight.lock"
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        _private(path, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _read_receipts(job: Path, scope_hash: str) -> list[dict]:
    path = job / RECEIPTS_FILE
    if not path.exists():
        return []
    _private(path, 0o600)
    records = [_decode(line) for line in path.read_text().splitlines() if line.strip()]
    for attempt, record in enumerate(records, 1):
        if (not isinstance(record, dict) or set(record) != RECEIPT_KEYS
                or type(record.get("attempt")) is not int or record["attempt"] != attempt
                or record.get("scope_hash") != scope_hash or record.get("role") != "kyb-reader"
                or type(record.get("valid")) is not bool
                or not isinstance(record.get("claims_hash"), str)
                or not re.fullmatch(r"sha256:[0-9a-f]{64}", record["claims_hash"])
                or not isinstance(record.get("errors"), list)
                or any(not isinstance(e, str) or e not in ERROR_CODES for e in record["errors"])
                or record["valid"] != (not record["errors"])):  # fmt: skip
            raise ValueError("invalid preflight receipt history")
    return records


def _error_code(error) -> str:
    if isinstance(error, dict):
        return "excerpt_missing" if error.get("code") == "excerpt_missing" else "core_validation"
    for fragment, code in (
        ("not in the read plan", "read_plan"),
        ("doc_sha256", "document_hash"),
        ("schema:", "schema"),
        ("bbox", "bbox"),
        ("text_layer", "text_layer"),
        ("linked blocks", "linked_blocks"),
        ("layout block", "layout_block"),
        ("page", "page"),
    ):
        if fragment in error:
            return code
    return "core_validation"


def _append(job: Path, record: dict) -> None:
    path = job / RECEIPTS_FILE
    fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    try:
        _private(path, 0o600)
        with os.fdopen(fd, "a", closefd=False) as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)


def validate_claims(claims: list[dict]) -> dict:
    """Check proposed claims against this process's one code-bound document; never submit."""
    claims_hash = sha256_json(claims)
    refused = {"valid": False, "errors": ["preflight scope unavailable or invalid"],
               "claims_hash": claims_hash, "attempt": 0}  # fmt: skip
    try:
        directory = os.environ.get("KYB_CODEX_RECEIPT_DIR")
        if not directory:
            return refused
        job = Path(directory)
        if not job.is_absolute():
            return refused
        _private(job, 0o700)
        with _job_lock(job):
            scope = load_scope(job)
            scope_hash = sha256_json(scope)
            attempt = len(_read_receipts(job, scope_hash)) + 1
            if attempt > MAX_ATTEMPTS:
                errors = ["budget_exhausted"]
                codes = errors
            else:
                root = Path(scope["root"])
                try:
                    payload = {k: scope[k] for k in ("case", "run", "doc", "doc_sha256")}
                    payload["claims"] = claims
                    with workspace_action(root):
                        sub = ClaimsSubmission.model_validate(payload)
                        errors = _check(sub, root)
                    codes = [_error_code(e) for e in errors]
                except ValidationError as exc:
                    errors = [f"schema: {e['loc']}: {e['msg']}" for e in exc.errors()]
                    codes = ["schema"]
                except OSError, ValueError, LookupError, PermissionError, AuditCorruption:
                    errors, codes = ["workspace validation unavailable"], ["workspace_unavailable"]
            record = {"attempt": attempt, "claims_hash": claims_hash, "valid": not errors,
                      "errors": codes, "scope_hash": scope_hash, "role": scope["role"]}  # fmt: skip
            _append(job, record)
            return {"valid": not errors, "errors": errors, "claims_hash": claims_hash,
                    "attempt": attempt}  # fmt: skip
    except OSError, ValueError, TypeError:
        return refused


def create_server():
    from mcp.server.mcpserver import MCPServer

    server = MCPServer(name=SERVER_NAME)
    server.tool()(validate_claims)
    return server


def main() -> None:
    create_server().run("stdio")


if __name__ == "__main__":
    main()
