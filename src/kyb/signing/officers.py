"""Officers and their tokens. Only an enrolled officer's token signs (spec, `auth: officer_token`).

A token is 256 random bits shown once, at enrolment; the ledger keeps only its hash, so reading
the ledger (which an agent may do) yields nothing that signs. The agent has no token and the
hook denies its attempt before it reaches the action; this check is the second, independent
denial, for a harness that has no hooks. Locally that is spec's trusted mode: the token proves
the officer to the action, it does not stop a process that can read the officer's environment.

Enrolment is open until the first officer exists. From then on a new officer is enrolled only by
presenting an enrolled officer's token, so an agent cannot enrol itself after the officer did.
"""

from __future__ import annotations

import hmac
import secrets
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kyb.signing.ledger import read_record, unreceipted, write_record
from kyb.util.fsutil import contained, safe_id
from kyb.util.hashing import sha256_bytes
from kyb.workspace.catalog import Catalog

OFFICER_REL = "ledger/officers/{officer}.json"
_PREFIX = "ledger/officers/"
DENIED = "officer_token is missing or does not match an enrolled officer"


def token_digest(token: str) -> str:
    return sha256_bytes(token.encode("utf-8"))


def officer_rel(officer: str) -> str:
    return OFFICER_REL.format(officer=safe_id(officer, "officer id"))


def enrolled(root: Path) -> list[str]:
    return sorted(Path(r).stem for r in Catalog(root).entries() if r.startswith(_PREFIX)
                  and r.endswith(".json"))  # fmt: skip


def authenticate(root: Path, officer: str, token: str | None) -> str:
    """The officer's id if `token` is theirs; otherwise a PermissionError that says no more than
    that, so it does not reveal which officers exist."""
    try:
        rel = officer_rel(officer)
    except ValueError:
        raise PermissionError(DENIED) from None
    if not token or Catalog(root).get(rel) is None:
        raise PermissionError(DENIED)
    want = read_record(root, rel)["token_sha256"]
    if not hmac.compare_digest(want, token_digest(token)):
        raise PermissionError(DENIED)
    return officer


def enroll(root: Path, officer: str, *, by: str | None = None,
           by_token: str | None = None) -> dict[str, Any]:  # fmt: skip
    """Enrol `officer` and return `{officer, token}` (the only time the token exists in clear).
    Call under `workspace_action`. A second officer needs an enrolled officer's `by` and token."""
    rel = officer_rel(officer)
    if enrolled(root):
        authenticate(root, by or "", by_token)
    if Catalog(root).get(rel) is not None:
        raise FileExistsError(f"officer {officer!r} is already enrolled")
    stale = unreceipted(root, rel)  # an interrupted enrolment: its token was never shown
    if stale is not None:
        contained(root, rel).unlink()
    token = secrets.token_urlsafe(32)
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    write_record(root, rel, {"officer": officer, "token_sha256": token_digest(token),
                             "enrolled_at": now}, action="enroll_officer", tier="decided",
                 as_of=now[:10], inputs=[], detail={"officer": officer, "by": by})  # fmt: skip
    return {"officer": officer, "token": token}
