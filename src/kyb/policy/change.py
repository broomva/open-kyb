"""`propose_policy` and `apply_policy`: the checked way an agent changes policy.

Policy proposals require explicit human approval through the workspace hook.

`policy/` is hook-protected, so no agent writes the policy file. Instead:

1. **propose** takes the complete new text and validates it with the same `parse_policy` that
   decides. Any `unrepresentable` reason refuses the proposal, every reason verbatim, so the agent
   can fix and resubmit. A valid text is recorded, never applied: an immutable, content-addressed
   proposal (parameter diff, changed rules, unified diff) beside both raw texts.
2. **apply** replaces the live file with a recorded proposal, only on a human's approval:
   - an agent's call: the hook answers `permissionDecision: "ask"` (the signing pattern, spec)
     and adds a one-time token to the call's input (`updatedInput`, the attestation pattern,
     spec), recording only its hash in `harness.policy_ask`. The harness runs the call, token
     and all, only if the human approves its prompt; a denied prompt delivers no token, so its
     record is inert. Apply consumes a token once. A call without one (no hooks, Codex, bypass
     modes, an unrecognised command form) is refused;
   - an officer at a terminal: `kyb policy apply … --officer <id>`, authenticated by the officer's
     token typed at a hidden prompt (`kyb.signing.officers`), never read from the environment.

Invariant 3 holds: the file stays the configuration and every decision still records its
`rules_version`. Nothing here re-decides: `decide` sees the new hash and demands `kyb run` again.
Invariant 5: the previous raw text is kept, content-addressed, so history is never lost.
"""

from __future__ import annotations

import difflib
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kyb import __version__
from kyb.audit.chain import AuditLog, chain_events
from kyb.contracts.b4 import RuleSpec
from kyb.policy.parse import parse_policy, section_texts
from kyb.policy.provenance import APPLY_EVENT, POLICY_SRC, approved_hash
from kyb.run.records import read_record
from kyb.util.fsutil import atomic_replace, contained, publish_exclusive
from kyb.util.hashing import canonical_json, sha256_bytes, sha256_json
from kyb.workspace.catalog import Catalog

PROPOSAL_REL = "ledger/policy/proposals/{proposal_id}.json"
TEXT_REL = "ledger/policy/texts/{text_hash}.md"  # raw policy texts, content-addressed
MAX_BYTES = 256_000
SUMMARY_DIFF_CHARS = 4_000  # the reply must stay well under a host's tool-output limit
ASK_EVENT = "harness.policy_ask"
CONFIRMATION_FIELD = "kyb_confirmation"  # the hook's token, added to the MCP call's input
# parameters that existing runs already consumed (screening, facts): a change needs a fresh run
RUN_BOUND_PARAMS = ("name_similarity_threshold", "conflict_ownership_points")
MAX_REASONS_SHOWN = 40
_PROPOSAL_ID = re.compile(r"pp-[0-9a-f]{16}")
_HASH = re.compile(r"sha256:[0-9a-f]{64}")


class PolicyRefused(ValueError):
    """The candidate does not validate: `reasons` are `parse_policy`'s, verbatim."""

    def __init__(self, reasons: list[str]) -> None:
        self.reasons = reasons
        shown = reasons[:MAX_REASONS_SHOWN]
        more = len(reasons) - len(shown)
        tail = f"\n- … and {more} more" if more else ""
        super().__init__("the proposed policy does not validate:\n- " + "\n- ".join(shown) + tail)


def apply_input(proposal_id: str, expected_current_hash: str) -> dict[str, str]:
    """The exact input a confirmation covers: the hook and the action hash the same dict."""
    return {"proposal_id": proposal_id, "expected_current_hash": expected_current_hash}


def confirmation_hash(token: str) -> str:
    return sha256_json({"kyb_policy_confirmation": token})


def _today() -> str:
    return datetime.now(UTC).date().isoformat()


def _keep_text(root: Path, raw: bytes, *, action: str) -> str:
    """Store a raw policy text at its content address (idempotent). Returns its path."""
    rel = TEXT_REL.format(text_hash=sha256_bytes(raw).removeprefix("sha256:"))
    # register also when a crash left the file without its receipt
    if publish_exclusive(contained(root, rel), raw) or Catalog(root).get(rel) is None:
        Catalog(root).register(rel, tier="none", as_of=_today(),
                               produced_by={"action": action, "kyb": __version__})  # fmt: skip
    return rel


def _param_diff(old: dict[str, Any], new: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {k: {"old": old.get(k), "new": new.get(k)} for k in sorted(old.keys() | new.keys())
            if old.get(k) != new.get(k)}  # fmt: skip


def _rule_diff(old: list[RuleSpec], new: list[RuleSpec]) -> dict[str, dict[str, Any]]:
    """Severity and points per rule id: they live in `PolicyModel.rules`, not in `params`, so the
    parameter diff alone hides them (PR #80 review, F2)."""
    a = {r.rule_id: {"severity": r.severity, "score_delta": r.score_delta} for r in old}
    b = {r.rule_id: {"severity": r.severity, "score_delta": r.score_delta} for r in new}
    return {k: {"old": a.get(k), "new": b.get(k)} for k in sorted(a.keys() | b.keys())
            if a.get(k) != b.get(k)}  # fmt: skip


def _weight(rule: dict[str, Any] | None) -> str:
    if rule is None:
        return "absent"
    delta = rule["score_delta"]
    return rule["severity"] + (f" +{delta}" if delta else "")


def _show(value: Any) -> str:
    return ", ".join(map(str, value)) if isinstance(value, list) else str(value)


def _changed_sections(old: bytes, new: bytes) -> list[str]:
    a, b = section_texts(old), section_texts(new)
    return [k for k in a.keys() | b.keys() if a.get(k) != b.get(k)]


def summary(record: dict[str, Any]) -> dict[str, Any]:
    """What an agent shows the human: compact, with the parameter diff on one line each."""
    diff = record["text_diff"]
    return {
        "proposal_id": record["proposal_id"],
        "current_hash": record["current_hash"],
        "proposed_hash": record["proposed_hash"],
        "current_version": record["current_version"],
        "proposed_version": record["proposed_version"],
        "parameter_diff": [
            f"{k} {_show(v['old'])} → {_show(v['new'])}" for k, v in record["param_diff"].items()
        ],
        "rule_diff": [
            f"{k} {_weight(v['old'])} → {_weight(v['new'])}" for k, v in record["rule_diff"].items()
        ],
        "changed_rules": record["changed_rules"],
        "changed_sections": record["changed_sections"],
        "text_diff": diff[:SUMMARY_DIFF_CHARS],
        "text_diff_truncated": len(diff) > SUMMARY_DIFF_CHARS,
        "record": PROPOSAL_REL.format(proposal_id=record["proposal_id"]),
        "next": ("show the human the parameter and rule diff and ask for an explicit yes; then "
                 f"apply_policy('{record['proposal_id']}', '{record['current_hash']}')"),
    }  # fmt: skip


def propose_policy(root: Path, text: str) -> dict[str, Any]:
    """Validate the complete candidate text and record it as a proposal. Never changes the live
    policy. Raises `PolicyRefused` with every `unrepresentable` reason."""
    raw = text.encode("utf-8")
    if len(raw) > MAX_BYTES:
        raise ValueError(f"the proposed policy is {len(raw)} bytes; the limit is {MAX_BYTES}")
    live = contained(root, POLICY_SRC).read_bytes()
    if raw == live:
        raise ValueError("the proposed text is the live policy: nothing to change")
    new = parse_policy(raw)
    if new.unrepresentable:
        raise PolicyRefused(list(new.unrepresentable))
    old = parse_policy(live)
    old_rules = {r.rule_id: r.text for r in old.rules}
    sections = sorted(_changed_sections(live, raw))
    record = {
        "proposal_id": "pp-" + sha256_json([old.content_hash, new.content_hash])[7:23],
        "current_hash": old.content_hash,
        "proposed_hash": new.content_hash,
        "current_version": old.rules_version,
        "proposed_version": new.rules_version,
        "param_diff": _param_diff(old.params, new.params),
        "rule_diff": _rule_diff(old.rules, new.rules),
        "changed_rules": sorted(r.rule_id for r in new.rules if old_rules.get(r.rule_id) != r.text),
        "changed_sections": sections,
        "text_diff": "".join(difflib.unified_diff(
            live.decode("utf-8").splitlines(keepends=True),
            text.splitlines(keepends=True), "compliance_rules.md (live)",
            "compliance_rules.md (proposed)")),
    }  # fmt: skip
    texts = [_keep_text(root, b, action="propose_policy") for b in (live, raw)]
    rel = PROPOSAL_REL.format(proposal_id=record["proposal_id"])
    body = canonical_json(record) + b"\n"
    if publish_exclusive(contained(root, rel), body) or Catalog(root).get(rel) is None:
        Catalog(root).register(
            rel, tier="claimed", as_of=_today(),
            inputs=[f"{t}#{sha256_bytes(contained(root, t).read_bytes())}" for t in texts],
            produced_by={"action": "propose_policy", "kyb": __version__},
        )  # fmt: skip
        AuditLog(root / "ledger" / "audit.jsonl").append(
            "policy", "policy.propose", input_hash=new.content_hash, output_hash=sha256_bytes(body),
            detail={"action": "propose_policy", "proposal_id": record["proposal_id"],
                    "current_hash": old.content_hash, "proposed_hash": new.content_hash,
                    "changed_rules": record["changed_rules"],
                    "params_changed": sorted(record["param_diff"]),
                    "rules_reweighted": sorted(record["rule_diff"])},
        )  # fmt: skip
    return {**summary(record), "live_policy_approved": old.content_hash == approved_hash(root)}


def load_proposal(root: Path, proposal_id: str) -> dict[str, Any]:
    if not isinstance(proposal_id, str) or not _PROPOSAL_ID.fullmatch(proposal_id):
        raise ValueError(f"not a policy proposal id: {proposal_id!r}")
    return read_record(root, PROPOSAL_REL.format(proposal_id=proposal_id))


def _consume_token(root: Path, wanted: dict[str, str], token: str | None) -> dict[str, str]:
    """The hook's record of the prompt whose approval delivered `token` for exactly this input,
    not used by an earlier apply. Anything else means no human approved this call: refuse."""
    refusal = PermissionError(
        "apply_policy needs the human's approval: this call carries no unused confirmation from "
        "the harness's permission prompt for this proposal and hash. Ask the human, then call "
        "apply_policy through a harness whose kyb hook shows that prompt (Claude Code), or let "
        "the officer run `kyb policy apply <id> --expected-current-hash <hash> --officer <id>` "
        "in their own terminal."
    )
    if not isinstance(token, str) or not token:
        raise refusal
    th = confirmation_hash(token)
    events = chain_events(root)
    asks = [e["detail"] for e in events if e.get("component") == "harness"
            and e.get("action") == ASK_EVENT and e["detail"].get("token_hash") == th]  # fmt: skip
    used = any(e["detail"].get("confirmation", {}).get("token_hash") == th for e in events
               if e.get("action") in (APPLY_EVENT, "policy.apply_refused"))  # fmt: skip
    if not asks or used or asks[0].get("input_hash") != sha256_json(wanted):
        raise refusal
    ask = asks[0]
    if problems := _provenance_problems(events, ask):
        raise PermissionError(f"{refusal} The prompt's record is not a harness hook's: "
                              f"{'; '.join(problems)}")  # fmt: skip
    return {"via": "harness-ask", "tool_use_id": ask.get("tool_use_id", ""),
            "session": ask.get("session", ""), "token_hash": th}  # fmt: skip


def _provenance_problems(events: list[dict], ask: dict) -> list[str]:
    """Why an ask record may not come from the harness's own hook (as `harness.attest` checks a
    reader's token): the MCP tool, a tool_use_id and a session the harness's SessionStart or
    PostToolUse hooks had already seen, delivered by one hook registration (PR #80 review, F1)."""
    from kyb.read.harness import registration_problem

    session = ask.get("session") or ""
    problems = []
    if ask.get("tool") != "apply_policy":
        problems.append(f"it was asked for {ask.get('tool')!r}, not the MCP apply_policy")
    if not ask.get("tool_use_id") or not session:
        problems.append("it has no tool_use_id or session")
    at = next(i for i, e in enumerate(events) if e.get("detail") is ask)
    seen = [e for e in events[:at] if e.get("component") == "harness"
            and e.get("action") in ("harness.session_start", "tool.call")
            and e["detail"].get("session") == session]  # fmt: skip
    if session and not seen:
        problems.append(f"session {session!r} was never seen by the harness's SessionStart or "
                        "PostToolUse hooks before it asked")  # fmt: skip
    if session and (problem := registration_problem(events, session)):
        problems.append(problem)
    return problems


def apply_policy(
    root: Path,
    proposal_id: str,
    expected_current_hash: str,
    *,
    confirmation: str | None = None,
    officer: str | None = None,
    officer_token: str | None = None,
) -> dict[str, Any]:
    """Replace the live policy with a recorded, still-valid proposal. Call inside
    `workspace_action`. Approved by `confirmation` (the hook's token, delivered only when the
    human approved the harness prompt) or by an enrolled `officer` and their token."""
    if not isinstance(expected_current_hash, str) or not _HASH.fullmatch(expected_current_hash):
        raise ValueError("expected_current_hash must be the live policy's sha256:<64 hex>")
    if officer is not None:
        from kyb.signing.officers import authenticate

        approval = {"via": "officer", "officer": authenticate(root, officer, officer_token)}
    else:
        wanted = apply_input(proposal_id, expected_current_hash)
        approval = _consume_token(root, wanted, confirmation)
    try:
        return _apply(root, proposal_id, expected_current_hash, approval)
    except Exception as exc:
        if approval.get("token_hash") and not isinstance(exc, _Unrecorded):
            # an approved call that fails still spends its approval: it is never replayed later
            AuditLog(root / "ledger" / "audit.jsonl").append_record(
                "policy", "policy.apply_refused", input_hash=sha256_json(approval),
                output_hash=sha256_json({"refusal": str(exc)[:500]}),
                detail={"action": "apply_policy", "proposal_id": proposal_id,
                        "confirmation": approval, "refusal": str(exc)[:500]},
            )  # fmt: skip
        raise exc.__cause__ if isinstance(exc, _Unrecorded) else exc from None


class _Unrecorded(Exception):
    """The audit append failed before its record was written; the live file was restored."""


def _apply(root: Path, proposal_id: str, expected_current_hash: str,
           approval: dict[str, Any]) -> dict[str, Any]:  # fmt: skip
    record = load_proposal(root, proposal_id)
    live_path = contained(root, POLICY_SRC)
    live = live_path.read_bytes()
    live_hash = sha256_bytes(live)
    if live_hash != expected_current_hash:
        raise PermissionError(
            f"stale: the live policy is {live_hash}, not the expected {expected_current_hash}; "
            "propose again against the live text"
        )
    if record["current_hash"] != live_hash:
        raise PermissionError(f"stale: proposal {proposal_id} was made against "
                              f"{record['current_hash']}, the live policy is {live_hash}; "
                              "propose again")  # fmt: skip
    text_rel = TEXT_REL.format(text_hash=record["proposed_hash"].removeprefix("sha256:"))
    raw = contained(root, text_rel).read_bytes()
    if sha256_bytes(raw) != record["proposed_hash"]:
        raise PermissionError(f"{text_rel}: bytes differ from the proposal's hash")
    new = parse_policy(raw)  # the engine may have changed since the proposal: validate again
    if new.unrepresentable:
        raise PolicyRefused(list(new.unrepresentable))
    previous = _keep_text(root, live, action="apply_policy")  # history is never lost
    detail = {"action": "apply_policy", "proposal_id": proposal_id, "old_hash": live_hash,
              "new_hash": new.content_hash, "old_version": record["current_version"],
              "new_version": new.rules_version, "previous_text": previous,
              "confirmation": approval}  # fmt: skip
    detail["live_was_approved"] = live_hash == approved_hash(root)
    log = AuditLog(root / "ledger" / "audit.jsonl")
    atomic_replace(live_path, raw)
    try:
        log.append_record("policy", APPLY_EVENT, input_hash=live_hash,
                          output_hash=new.content_hash, detail=detail)  # fmt: skip
    except BaseException as exc:
        try:
            written = any(e.get("action") == APPLY_EVENT and e["detail"] == detail
                          for e in chain_events(root)[-3:])  # fmt: skip
        except Exception:
            written = False
        if written:  # the record is in the chain: the change stands with it
            raise
        atomic_replace(live_path, live)  # an unrecorded change is no change
        raise _Unrecorded() from exc
    return {
        "applied": proposal_id,
        "old_hash": live_hash,
        "new_hash": new.content_hash,
        "rules_version": f"{new.rules_version}+{new.content_hash}",
        "previous_text": previous,
        "confirmation": approval,
        "next": _next_step(record),
    }


def _next_step(record: dict[str, Any]) -> str:
    bound = [k for k in RUN_BOUND_PARAMS if k in record["param_diff"]]
    if not bound:
        return "decisions made under the old policy are not re-used: run_review, then decide"
    from kyb.workspace.runs import FRESH_RUN

    return (
        f"{', '.join(bound)} changed: runs already screened or derived under the old value "
        f"cannot be re-run. {FRESH_RUN}"
    )
