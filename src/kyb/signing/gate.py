"""The hook's signing gate: an agent call signs only through the officer's two approvals (spec,
enforced by the workspace hook guard).

An agent session cannot tell an officer's prompt from its own, so the agent never signs on its
own. A sign call may proceed only after (1) the officer's recorded answer to the skill's
AskUserQuestion ("Sign decision X as officer Y?" -> Sign), and (2) this hook returning
`permissionDecision: "ask"`, which makes the harness show its own permission prompt: the
officer's second, harness-level approval. The officer's token (3) stays out of the agent's reach:
`kyb sign` and the MCP `sign_decision` read `KYB_OFFICER_TOKEN` from the workspace environment
themselves, and `officers.py` still checks it, so a harness without hooks gains nothing.

Only an `assent` through `sign_decision` / `kyb sign <id> assent --officer <id>` can be asked.
Everything else is denied outright: `record_override` / `kyb override` / a `sign ... override`
verdict, `enroll_officer` / `kyb officer enroll` (CLI only, l5-2), any command or tool input that
carries or names the token or the signing internals, and any sign call from an event that is not
recognisably Claude Code in a mode where the prompt is shown (Codex, unknown harnesses, bypass
modes): the "ask" cannot be expressed there, so it fails closed. The Bash check is a best-effort
pattern match, not an OS boundary (spec). Imports only `re`: it runs on the hook fast path (spec).

Applying a policy change takes the same native prompt, and only in modes where a person answers
it (default, acceptEdits, plan): kyb's MCP `apply_policy` with exactly a
proposal id and the live hash is asked; a malformed call is denied. The guard adds a one-time
token to the asked call, which reaches the action only if the human approves, and the action
refuses any call without one (`kyb.policy.change`). The agent's shell never applies policy and
never runs the hook entry points that mint tokens (`kyb.workspace.shell`, PR #80 review F1).
"""

from __future__ import annotations

import re
from typing import Any

DENY = "deny"
ASK = "ask"
DENIED_TOOLS = frozenset({"record_override", "enroll_officer"})
_SIGN_TOOL = "sign_decision"
_SUBCOMMAND = re.compile(r"(?:^|[\s'\"])(sign|override|sign_decision|record_override)(?=[\s'\"]|$)")
_ENROLL = re.compile(r"\bofficer\s+enroll\b")
_INTERNALS = re.compile(
    r"kyb\.signing|kyb\.policy|kyb\.hook\b|kyb\.workspace\.guard|KYB_OFFICER_TOKEN|officer_token"
    r"|--officer-token",
    re.IGNORECASE,
)
_KYB = re.compile(r"(?:^|[\s/'\"=;&|(])(?:uv\s+run\s+|uvx\s+|python3?\s+-m\s+)?kyb(?=[\s'\".]|$)")
# the one command line an officer's approval can cover: a plain, single `kyb sign <id> assent`
_ID = r"[A-Za-z0-9][A-Za-z0-9._-]*"
# spaces and tabs only: a newline separates shell commands
_SIGN_COMMAND = re.compile(
    rf"[ \t]*(?:uv[ \t]+run[ \t]+|uvx[ \t]+|python3?[ \t]+-m[ \t]+)?kyb"
    rf"(?:[ \t]+(?:-w|--workspace)[ \t]+[^\s;&|`$()<>]+)?"
    rf"[ \t]+sign[ \t]+({_ID})[ \t]+assent[ \t]+--officer[ \t]+({_ID})[ \t]*",
    re.ASCII,
)
_ID_FULL = re.compile(rf"{_ID}", re.ASCII)
_POLICY_TOOL = "apply_policy"
_POLICY_KEYS = frozenset({"proposal_id", "expected_current_hash"})
_CONFIRMATION = "kyb_confirmation"  # the hook's own token: overwritten on every ask
# the modes where a person answers the harness prompt (auto and SDK hosts may answer it alone)
_HUMAN_MODES = frozenset({"default", "acceptEdits", "plan"})
_PROPOSAL = r"pp-[0-9a-f]{16}"
_HASH = r"sha256:[0-9a-f]{64}"
POLICY_WHY = (
    "a policy change is the human's to approve. The agent proposes with propose_policy, shows "
    "the human the parameter and rule diff, asks with AskUserQuestion, and only on an explicit "
    "yes calls apply_policy(proposal_id, expected_current_hash); the harness then asks the human "
    "again. An officer can instead run `kyb policy apply … --officer <id>` in their own terminal."
)
# the MCP call an approval can cover: exactly these keys, ids that are plain identifiers (the
# officer's prompt quotes them, so a newline or markup could misstate what is being approved)
_SIGN_KEYS = frozenset({"decision_id", "verdict", "officer", "rationale", "status"})
_RATIONALE_MAX = 2000
_UNSAFE_MODES = frozenset({"bypassPermissions", "dontAsk"})
_CODEX_FIELDS = ("turn_id", "model")  # Codex hook events carry these; Claude Code's do not
WHY = (
    "signing is the officer's act. The agent never signs on its own: after review_queue and "
    'explain it asks the officer with AskUserQuestion, and only on an explicit "Sign" does it '
    "call sign_decision (assent) or `kyb sign <id> assent --officer <id>`; the harness then "
    "asks the officer again, and the officer token comes from the workspace environment, never "
    "from the agent. Overrides and officer enrolment are not available to the agent."
)


def _can_ask(event: dict[str, Any]) -> bool:
    """True only for a Claude Code PreToolUse event whose mode shows the permission prompt."""
    return (
        event.get("hook_event_name") == "PreToolUse"
        and bool(event.get("tool_use_id"))
        and bool(event.get("session_id"))
        and not any(event.get(f) for f in _CODEX_FIELDS)
        and event.get("permission_mode") not in (None, "", *_UNSAFE_MODES)
    )


def policy_input(base: str, tool_input: dict) -> dict[str, str] | None:
    """The `{proposal_id, expected_current_hash}` an askable policy apply covers, else None."""
    if base == _POLICY_TOOL:
        if set(tool_input) - {_CONFIRMATION} != _POLICY_KEYS:
            return None
        pid, h = tool_input.get("proposal_id"), tool_input.get("expected_current_hash")
        ok = isinstance(pid, str) and isinstance(h, str)
        if not (ok and re.fullmatch(_PROPOSAL, pid) and re.fullmatch(_HASH, h)):
            return None
        return {"proposal_id": pid, "expected_current_hash": h}
    return None


def policy_check(base: str, tool_input: dict, event: dict[str, Any]) -> tuple[str, str] | None:
    """`("ask", …)` for an exact policy apply, `("deny", …)` for a malformed kyb `apply_policy`
    call or a harness that cannot ask, None otherwise."""
    tool = str(event.get("tool_name") or "")
    kyb_tool = base == _POLICY_TOOL and (tool == base or "kyb" in tool.rsplit("__", 1)[0])
    wanted = policy_input(base, tool_input)
    if wanted is None:
        if kyb_tool:
            return DENY, ("apply_policy takes exactly a proposal id (pp-…) and the live policy's "
                          f"sha256 hash, nothing else: {POLICY_WHY}")  # fmt: skip
        return None
    if base == _POLICY_TOOL and not kyb_tool:
        return None
    if not _can_ask(event) or event.get("permission_mode") not in _HUMAN_MODES:
        return DENY, (f"this harness cannot show the human's approval prompt, so a policy "
                      f"change fails closed. {POLICY_WHY}")  # fmt: skip
    return ASK, (f"human approval: apply policy proposal {wanted['proposal_id']} over the live "
                 f"policy {wanted['expected_current_hash'][:19]}…")  # fmt: skip


def _ask(event: dict[str, Any], decision: str, officer: str) -> tuple[str, str] | None:
    if not _can_ask(event):
        return DENY, (f"this harness cannot show the officer's approval prompt, so signing "
                      f"fails closed. {WHY}")  # fmt: skip
    return ASK, f"officer approval: sign decision {decision} (assent) as officer {officer}"


def check(base: str, tool_input: dict, event: dict[str, Any]) -> tuple[str, str] | None:
    """`("deny", reason)`, `("ask", reason)` or None for a call that is none of the signing paths.

    `base` is the tool name without the MCP prefix."""
    if verdict := policy_check(base, tool_input, event):
        return verdict
    if base in DENIED_TOOLS:
        return DENY, f"{base} is officer-only: {WHY}"
    if base == _SIGN_TOOL:
        if any("token" in str(k).lower() for k in tool_input):
            return DENY, f"the agent must not pass the officer token: {WHY}"
        if set(tool_input) - _SIGN_KEYS or tool_input.get("status") is not None:
            return DENY, f"sign_decision takes only decision_id, verdict, officer, rationale: {WHY}"
        if tool_input.get("verdict") != "assent":
            return DENY, f"only an assent can be signed from the harness: {WHY}"
        decision, officer = tool_input.get("decision_id"), tool_input.get("officer")
        if not all(isinstance(v, str) and _ID_FULL.fullmatch(v) for v in (decision, officer)):
            return DENY, f"decision and officer ids must be plain identifiers: {WHY}"
        rationale = tool_input.get("rationale", "")
        if not isinstance(rationale, str) or len(rationale) > _RATIONALE_MAX:
            return DENY, f"the rationale must be short text: {WHY}"
        return _ask(event, decision, officer)
    if base != "Bash":
        return None
    command = str(tool_input.get("command") or "")
    if _INTERNALS.search(command):
        return DENY, ("the command touches officer credentials or the signing, hook or "
                      f"policy-change internals: {WHY}")  # fmt: skip
    if not (_KYB.search(command) and (_SUBCOMMAND.search(command) or _ENROLL.search(command))):
        return None
    if m := _SIGN_COMMAND.fullmatch(command):
        return _ask(event, m.group(1), m.group(2))
    return DENY, f"the command signs, overrides or enrols an officer: {WHY}"
