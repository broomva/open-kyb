"""Changing a test workspace's policy the ways kyb accepts (PR #80 review, F3).

A policy file written directly is not the approved policy, and the decision path refuses it. A
test that needs another policy goes through one of the two legitimate sources:

- `apply_as_officer`: propose the text and apply it as an enrolled officer (a representable edit);
- `reinstall`: `kyb init --force --officer` with the new text (wording the engine has to be
  taught, which ships with a code change; `propose_policy` refuses it). A re-init that changes
  the approved policy is an officer's act too.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from kyb.audit.chain import workspace_action
from kyb.policy.change import apply_policy, propose_policy
from kyb.signing import officers
from kyb.workspace.init import init_workspace

_TOKENS: dict[Path, str] = {}  # the policy officer's token per workspace (shown once at enrolment)
OFFICER = "policy-officer"


def _token(root: Path) -> str:
    if root not in _TOKENS:
        with workspace_action(root):
            _TOKENS[root] = officers.enroll(root, OFFICER)["token"]
    return _TOKENS[root]


def apply_as_officer(root: Path, text: str) -> dict:
    token = _token(root)
    with workspace_action(root):
        out = propose_policy(root, text)
        return apply_policy(root, out["proposal_id"], out["current_hash"], officer=OFFICER,
                            officer_token=token)  # fmt: skip


def reinstall(root: Path, text: str, scratch: Path) -> dict:
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / "compliance_rules.md").write_text(text, "utf-8")
    watchlist = root / "policy/watchlists.json"
    if watchlist.exists():
        shutil.copyfile(watchlist, scratch / "watchlists.json")
    tenant = json.loads((root / "template.lock").read_text())["tenant"]
    return init_workspace(root, tenant=tenant, policy_src=scratch / "compliance_rules.md",
                          force=True, officer=OFFICER, officer_token=_token(root))  # fmt: skip
