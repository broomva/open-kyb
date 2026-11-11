"""The `load_policy` action: parse the workspace's policy file and record it at
`gold/policy/<hash>.json` (b4 `POLICY_REL`). Call it inside `workspace_action`, as every action is.

The record is content-addressed by the policy file's hash, so re-loading an unchanged policy
writes nothing new, and an edited policy is a new record beside the old one, never a replacement.
"""

from __future__ import annotations

from pathlib import Path

from kyb import __version__
from kyb.audit.chain import AuditLog
from kyb.contracts.b4 import POLICY_REL, PolicyModel
from kyb.policy.parse import load_policy, parse_policy
from kyb.policy.provenance import POLICY_SRC, approved_policy
from kyb.util.fsutil import contained, publish_exclusive
from kyb.util.hashing import canonical_json, sha256_bytes
from kyb.workspace.catalog import Catalog


def record_policy(root: Path, *, as_of: str, src_rel: str = POLICY_SRC) -> tuple[str, PolicyModel]:
    # the live policy is decided under only if it is the approved text (PR #80 review, F3)
    if src_rel == POLICY_SRC:
        model = parse_policy(approved_policy(root))
    else:
        model = load_policy(contained(root, src_rel))
    rel = POLICY_REL.format(rules_hash=model.content_hash.removeprefix("sha256:"))
    data = canonical_json(model.model_dump(mode="json")) + b"\n"
    created = publish_exclusive(contained(root, rel), data)
    if created:
        Catalog(root).register(
            rel, tier="derived", as_of=as_of, inputs=[f"{src_rel}#{model.content_hash}"],
            produced_by={"action": "load_policy", "kyb": __version__},
        )  # fmt: skip
        AuditLog(root / "ledger" / "audit.jsonl").append(
            "policy", "tool.call", input_hash=model.content_hash, output_hash=sha256_bytes(data),
            detail={"action": "load_policy", "rules_version": model.rules_version,
                    "unrepresentable": len(model.unrepresentable)},
        )  # fmt: skip
    return rel, model
