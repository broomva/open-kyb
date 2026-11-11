"""Harness transcript summaries. No model client and no trust in model-reported identity.

The hook snapshots these facts at every SubagentStop. Judges consume the chained snapshot,
never a mutable transcript at its original path. This module stays stdlib-only for hooks.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# Claude Code appends this to a subagent transcript as an `isMeta` user record each time a tool
# result carries an image (a Read of a page image, render_crop). It is the harness's own
# coordinate note: digits only, no free text, so it cannot carry an instruction. It is not a
# launch or an added message, and the judge skips exactly this shape and nothing near it.
HARNESS_IMAGE_NOTE = re.compile(
    r"\[Image: original \d+x\d+, displayed at \d+x\d+\. "
    r"Multiply coordinates by \d+(?:\.\d+)? to map to original image\.\]"
)


def transcript_summary(path: str) -> dict:
    prompts, tools, turns = [], [], {}
    complete = True
    try:
        raw = Path(path).read_bytes() if path else b""
        if not raw:
            raise ValueError("empty transcript")
        for line in raw.decode("utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            message = rec.get("message") or {}
            content = message.get("content", [])
            if rec.get("type") == "user":
                if isinstance(content, str):
                    prompts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if block.get("type") == "text":
                            prompts.append(block.get("text", ""))
                        elif block.get("type") != "tool_result":
                            complete = False
                else:
                    complete = False
            elif rec.get("type") == "assistant":
                for block in content if isinstance(content, list) else []:
                    if block.get("type") == "tool_use":
                        tools.append(block.get("name", ""))
                model = message.get("model", "")
                if model.startswith("<"):
                    continue
                usage = message.get("usage")
                mid = message.get("id") or rec.get("uuid")
                if (
                    not mid
                    or not model
                    or not isinstance(usage, dict)
                    or any(
                        type(usage.get(k)) is not int or usage[k] < 0
                        for k in ("input_tokens", "output_tokens")
                    )
                ):
                    complete = False
                else:
                    turns[mid] = {"message_id": mid, "model": model, "usage": usage}
    except OSError, ValueError, TypeError, UnicodeDecodeError:
        complete = False
    return {
        "complete": complete and bool(turns),
        "prompts": prompts,
        "tools": tools,
        "turns": [turns[k] for k in sorted(turns)],
    }


def lifecycle_reasons(
    events: list[dict], who: dict, canonical: str, allowed_tools: set[str]
) -> list[str]:
    from kyb.read.harness import by_action, eligibility
    from kyb.util.hashing import sha256_json

    reasons = eligibility(events, who.get("session"), who.get("agent_id"))
    if who.get("context") != "isolated":
        reasons.append("submission is unattested")
    stops = [
        d
        for d in by_action(events, "harness.subagent_stop")
        if d.get("session") == who.get("session") and d.get("agent_id") == who.get("agent_id")
    ]
    from kyb.codex.managed import identity

    managed = identity(events, who.get("session"), who.get("agent_id"))
    if managed is not None:
        stops = [managed]
        if managed.get("canonical") != canonical:
            reasons.append("managed Codex work order differs from its launch")
    if not stops:
        reasons.append("no stopped transcript")
    for stop in stops:
        snap = stop.get("lifecycle") or {}
        if not snap.get("complete"):
            reasons.append("transcript or usage incomplete")
        for turn in snap.get("turns", []):
            usage = turn["usage"]
            receipt = sha256_json(
                {"session": who.get("session"), "agent_id": who.get("agent_id"), "turn": turn}
            )
            if not any(
                e["action"] == "model.call"
                and e["component"] == "harness"
                and e["input_hash"] == receipt
                and e["output_hash"] == sha256_json(usage)
                and e["tokens_in"] == usage["input_tokens"]
                and e["tokens_out"] == usage["output_tokens"]
                and e["detail"].get("cost_basis") == "transcript_usage"
                and e["detail"].get("model") == turn["model"]
                for e in events
            ):
                reasons.append("audit: incomplete (missing transcript usage receipt)")
        prompts = [
            p
            for p in snap.get("prompts") or []
            if not (isinstance(p, str) and HARNESS_IMAGE_NOTE.fullmatch(p))
        ]
        if prompts != [canonical]:
            reasons.append("transcript contains non-canonical launch or added messages")
        effective_tools = (
            {"validate_claims"}
            if managed is not None and managed.get("agent_type") == "kyb-reader"
            else allowed_tools
        )
        if any(t.split("__")[-1] not in effective_tools for t in snap.get("tools", [])):
            reasons.append("transcript contains tools outside the allow-list")
    return sorted(set(reasons))
