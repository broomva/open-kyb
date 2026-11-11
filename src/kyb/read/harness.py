"""Harness attestation (spec): who called an action, as reported by the harness's own hooks.

Hooks append events to the chain from the harness's hook input, never from the model's output:
- `harness.session_start` (SessionStart: the session and the model the harness reports);
- `harness.launch` (PreToolUse on `Agent`: a kyb-reader or kyb-predicates launch, whose prompt
  must be the canonical one, or the launch is denied);
- `harness.subagent_start` (SubagentStart: agent_id and agent_type; it fires again on a resume);
- `harness.reader_bind` (the reader's first `read_plan` call, which must match an unbound launch
  of this session for the same document);
- `harness.attestation` (PreToolUse on a submit tool: the hook mints a one-time token, injects it
  into the call's input, and records only the token's hash with the input hash);
- `harness.attestation_consumed` (a submission used that token).

A submission is `isolated` only if its token matches an unconsumed attestation for exactly this
input, made inside a subagent that was opened once in this session, of the right type, launched
canonically for this target, that has not submitted for another target, in a session where only
one kyb hook registration (plugin or workspace) recorded events. Isolation is checked again
whenever assurance is read (`eligibility`), because disqualifying evidence can arrive later.
Everything else is recorded as `unattested`, and the judges cap any decision that consumes it.
Attestation trusts the harness: that is spec's trusted-local stance, not a cryptographic proof.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Lazy (PEP 810): the hooks import this module on every tool call (spec, spec).
lazy from kyb.audit.chain import chain_events
lazy from kyb.util.hashing import sha256_json

READER_TYPES = frozenset({"kyb-reader", "kyb:kyb-reader"})
PREDICATES_TYPES = frozenset({"kyb-predicates", "kyb:kyb-predicates"})
TOKEN_FIELD = "kyb_attestation"
READER_PROMPT = re.compile(r"kyb-reader ([A-Za-z0-9][A-Za-z0-9_.-]*) (r[1-9]\d*) ([A-Za-z0-9_.-]+)")
PREDICATES_PROMPT = re.compile(r"kyb-predicates ([A-Za-z0-9][A-Za-z0-9_.-]*)")


def token_hash(token: str) -> str:
    return sha256_json({"kyb_attestation_token": token})


def split_token(payload: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """The call's arguments without the hook's token, and the token."""
    clean = {k: v for k, v in payload.items() if k != TOKEN_FIELD}
    token = payload.get(TOKEN_FIELD)
    return clean, token if isinstance(token, str) and token else None


def model_family(model: str | None) -> str:
    """The vendor family of a harness-reported model id; `unknown` when not reported."""
    m = (model or "").lower()
    if not m:
        return "unknown"
    for prefix, fam in (("claude", "anthropic"), ("gpt", "openai"), ("o1", "openai"),
                        ("o3", "openai"), ("o4", "openai"), ("codex", "openai"),
                        ("gemini", "google")):  # fmt: skip
        if m.startswith(prefix):
            return fam
    return "unknown"


@dataclass(frozen=True)
class Attestation:
    context: str  # "isolated" | "unattested"
    agent_id: str | None = None
    agent_type: str | None = None
    tool_use_id: str | None = None
    session: str | None = None
    token_hash: str | None = None
    launch_id: str | None = None
    reasons: list[str] = field(default_factory=list)
    model: str | None = None

    @property
    def family(self) -> str:
        return model_family(self.model)

    def identity(self) -> dict[str, Any]:
        return {"context": self.context, "agent_id": self.agent_id, "agent_type": self.agent_type,
                "tool_use_id": self.tool_use_id, "session": self.session,
                "token_hash": self.token_hash, "launch_id": self.launch_id, "model": self.model,
                "family": self.family, "unattested_reasons": self.reasons}  # fmt: skip


def by_action(events: list[dict], action: str) -> list[dict]:
    return [
        e["detail"] for e in events if e.get("component") == "harness" and e["action"] == action
    ]


def bound_document(events: list[dict], session: str, agent_id: str) -> dict | None:
    binds = [
        d for d in by_action(events, "harness.reader_bind")
        if d.get("agent_id") == agent_id and d.get("session") == session
    ]  # fmt: skip
    return binds[0] if binds else None


def bound_predicates(events: list[dict], session: str, agent_id: str) -> dict | None:
    binds = [
        d
        for d in by_action(events, "harness.predicate_bind")
        if d.get("session") == session and d.get("agent_id") == agent_id
    ]
    return binds[0] if binds else None


# Events a hook writes (as opposed to ones the core writes, e.g. attestation_consumed)
HOOK_ACTIONS = frozenset({"harness.session_start", "harness.launch", "harness.subagent_start",
                          "harness.subagent_stop", "harness.reader_bind", "harness.attestation",
                          "harness.predicate_bind", "harness.policy_ask",
                          "tool.call"})  # fmt: skip
HOOK_SOURCES = frozenset({"plugin", "workspace"})


def hook_sources(events: list[dict], session: str) -> set[str]:
    """The registrations that delivered hook events in this session. An event with no source
    (a hook from before `--source`) counts as `untagged`, so it cannot hide behind a tagged one."""
    return {e["detail"].get("hook_source") or "untagged" for e in events
            if e.get("component") == "harness" and e.get("action") in HOOK_ACTIONS
            and e["detail"].get("session") == session}  # fmt: skip


def registration_problem(events: list[dict], session: str) -> str | None:
    """Why this session's hook evidence is ambiguous, or None: one known registration only."""
    sources = hook_sources(events, session)
    bad = sources - HOOK_SOURCES - {"untagged"}
    if bad:
        return f"unsupported hook source(s) {sorted(bad)} in this session"
    if len(sources) > 1:
        return ("kyb hooks are registered more than once in this session "
                f"({', '.join(sorted(sources))}): remove one (the plugin, or the workspace's "
                ".claude/settings.json hooks)")  # fmt: skip
    return None


def eligibility(events: list[dict], session: str | None, agent_id: str | None) -> list[str]:
    """Evidence recorded since a submission that withdraws its isolation. Checked again whenever
    assurance is read: a second registration or a resume can be discovered later."""
    if not session or not agent_id:
        return ["no session or agent recorded"]
    from kyb.codex.managed import identity

    managed = identity(events, session, agent_id)
    if managed is not None:
        return managed["reasons"]
    reasons = []
    problem = registration_problem(events, session)
    if problem:
        reasons.append(problem)
    starts = [d for d in by_action(events, "harness.subagent_start")
              if d.get("agent_id") == agent_id and d.get("session") == session]  # fmt: skip
    if len(starts) > 1:
        reasons.append("the subagent was resumed (more than one SubagentStart)")
    return reasons


def stopped_model(events: list[dict], session: str | None, agent_id: str | None) -> str | None:
    """The one model the harness transcripts show for this subagent's turns, over every
    SubagentStop snapshot recorded (a stop can be vetoed and the agent continue). None when the
    agent never stopped, ran on more than one model, or any snapshot was incomplete (a missing,
    unreadable or partly unparsable transcript leaves the model unknown)."""
    if not session or not agent_id:
        return None
    from kyb.codex.managed import identity

    managed = identity(events, session, agent_id)
    if managed is not None:
        models = set(managed.get("served_response_models", {}).values())
        return next(iter(models)) if not managed["reasons"] and len(models) == 1 else None
    stops = [d for d in by_action(events, "harness.subagent_stop")
             if d.get("agent_id") == agent_id and d.get("session") == session]  # fmt: skip
    if not stops or any(d.get("complete") is not True or not d.get("models") for d in stops):
        return None
    models = {m for d in stops for m in d["models"]}
    return models.pop() if len(models) == 1 else None


def free_launch(events: list[dict], session: str, target: dict) -> dict | None:
    """The oldest canonical reader launch for `target` in this session not yet bound to an agent."""
    taken = {d.get("launch_id") for d in by_action(events, "harness.reader_bind")}
    for d in by_action(events, "harness.launch"):
        if (d.get("session") == session and d.get("subagent_type") in READER_TYPES
                and all(d.get(k) == v for k, v in target.items())
                and d.get("launch_id") not in taken):  # fmt: skip
            return d
    return None


def attest(
    root: Path, *, tool: str, input_hash: str, token: str | None,
    allowed_types: frozenset[str], target: dict,
) -> Attestation:  # fmt: skip
    """Check the hook attestation that this exact execution carries."""
    if token is None:
        return Attestation("unattested", reasons=["the call carries no hook attestation token"])
    events = chain_events(root)
    from kyb.codex.managed import attest as managed_attest

    managed = managed_attest(events, token=token, tool=tool, input_hash=input_hash,
                             allowed_types=allowed_types, target=target)  # fmt: skip
    if managed is not None:
        return managed
    th = token_hash(token)
    matches = [d for d in by_action(events, "harness.attestation") if d.get("token_hash") == th]
    if not matches:
        return Attestation("unattested", reasons=["the token matches no recorded hook call"])
    a = matches[0]
    session, agent_id = a.get("session") or "", a.get("agent_id") or ""
    agent_type = a.get("agent_type") or ""
    sessions = [
        d for d in by_action(events, "harness.session_start")
        if d.get("session") == session and d.get("model")
    ]  # fmt: skip
    binding_action = (
        "harness.reader_bind" if allowed_types == READER_TYPES else "harness.predicate_bind"
    )
    bind_fn = bound_document if allowed_types == READER_TYPES else bound_predicates
    bind = bind_fn(events, session, agent_id) if agent_id else None
    base = {"agent_id": agent_id or None, "agent_type": agent_type or None,
            "tool_use_id": a.get("tool_use_id"), "session": session or None, "token_hash": th,
            "launch_id": (bind or {}).get("launch_id"),
            "model": sessions[-1]["model"] if sessions else None}  # fmt: skip
    reasons = []
    if any(d.get("token_hash") == th for d in by_action(events, "harness.attestation_consumed")):
        reasons.append("this attestation was already consumed")
    if a.get("tool") != tool or a.get("input_hash") != input_hash:
        reasons.append("the executed input differs from what the hook attested")
    if not a.get("tool_use_id"):
        reasons.append("the hook call had no tool_use_id")
    if not session:
        reasons.append("the hook call had no session")
    if not agent_id:
        reasons.append("not called from a subagent (no agent_id)")
    starts = [
        d for d in by_action(events, "harness.subagent_start")
        if d.get("agent_id") == agent_id and d.get("session") == session
    ]  # fmt: skip
    if agent_id and not starts:
        reasons.append("agent_id was not opened by SubagentStart in this session")
    if len(starts) > 1:
        reasons.append("the subagent was resumed (more than one SubagentStart)")
    if agent_type not in allowed_types:
        reasons.append(f"agent_type {agent_type!r} is not one of {sorted(allowed_types)}")
    if starts and starts[0].get("agent_type") != agent_type:
        reasons.append("agent_type differs from the one SubagentStart recorded")
    if bind is not None:
        binds = by_action(events, binding_action)
        owners = {d.get("agent_id") for d in binds if d.get("launch_id") == bind.get("launch_id")}
        mine = [d for d in binds if d.get("agent_id") == agent_id and d.get("session") == session]
        if len(owners) > 1 or len(mine) > 1:
            reasons.append("the binding is ambiguous (a launch or an agent bound more than once)")
    if allowed_types == READER_TYPES:
        if bind is None:
            reasons.append("the reader never bound a canonically launched document")
        elif any(bind.get(k) != v for k, v in target.items()):
            reasons.append(f"this reader is bound to {bind.get('doc')}, not {target.get('doc')}")
    elif bind is None:
        reasons.append("predicates agent never bound a canonical work order")
    elif any(bind.get(k) != v for k, v in target.items()):
        reasons.append("predicates agent is bound to another work order")
    prior = [
        d for d in by_action(events, "harness.attestation_consumed")
        if d.get("agent_id") == agent_id and d.get("session") == session
        and d.get("target") != target
    ]  # fmt: skip
    if agent_id and prior:
        reasons.append("this agent already submitted for another target")
    problem = registration_problem(events, session) if session else None
    if problem:
        reasons.append(problem)
    if reasons:
        return Attestation("unattested", reasons=reasons, **base)
    return Attestation("isolated", **base)
