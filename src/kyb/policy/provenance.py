"""Which policy text is approved, and refusing any other (cross-review of PR #80, F3).

The live `policy/compliance_rules.md` has exactly two legitimate sources: the text `kyb init`
installed (`template.lock` `policy_at_init`, renewed by a forced re-init) and the `new_hash` of
the latest `policy.apply` after it (an apply is approved by the harness prompt or an officer).
A file replaced any other way (a shell trick, a hand edit) is not the approved policy, so the
decision path refuses it instead of deciding under it. To change policy, propose and apply.
"""

from __future__ import annotations

import json
from pathlib import Path

from kyb.audit.chain import chain_events
from kyb.util.fsutil import contained
from kyb.util.hashing import sha256_bytes

POLICY_SRC = "policy/compliance_rules.md"
APPLY_EVENT = "policy.apply"


def approved_hash(root: Path) -> str | None:
    """The hash of the approved policy text, or None if the workspace records none."""
    events = chain_events(root)
    last_init = max((i for i, e in enumerate(events) if e.get("action") == "workspace.init"),
                    default=-1)  # fmt: skip
    applies = [e for e in events[last_init + 1 :]
               if e.get("component") == "policy" and e.get("action") == APPLY_EVENT]  # fmt: skip
    if applies:
        return applies[-1]["detail"]["new_hash"]
    try:
        return json.loads((root / "template.lock").read_text("utf-8")).get("policy_at_init")
    except OSError, ValueError:
        return None


def approved_policy(root: Path) -> bytes:
    """The live policy's bytes, if they are the approved text; otherwise PermissionError."""
    raw = contained(root, POLICY_SRC).read_bytes()
    live, want = sha256_bytes(raw), approved_hash(root)
    if live != want:
        raise PermissionError(
            f"{POLICY_SRC} ({live}) is not the approved policy ({want or 'none recorded'}: the "
            "text kyb init installed or the latest policy.apply). It was changed outside "
            "apply_policy, so nothing is decided under it. Restore the approved text, or "
            "propose the change and have it applied (propose_policy, apply_policy)."
        )
    return raw
