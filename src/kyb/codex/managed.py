"""Closed, one-shot Codex processes, distinct from native/Claude hook attestation.

The runner owns inputs, process completion and the immutable transcript snapshot. No process
handle, token or document output is exposed to the parent model. Native Codex launches whose
prompts are encrypted are deliberately NOT promoted through this path.
"""

from __future__ import annotations

from kyb.read.harness import Attestation, by_action, model_family, token_hash
from kyb.util.hashing import sha256_json

RUN = "harness.codex_managed_run"
ATTEST = "harness.codex_managed_attestation"


def validation_bound(record: dict) -> bool:
    """Only the reader's code-scoped, fully accounted preflight can extend tool-free jobs."""
    snap = record.get("lifecycle", {})
    proof = record.get("preflight")
    if not snap.get("tools"):
        return not proof
    if not isinstance(proof, dict) or record.get("agent_type") != "kyb-reader":
        return False
    scope, checks, hooks = proof.get("scope", {}), proof.get("checks", []), proof.get("hooks", [])
    calls = proof.get("native_calls", [])
    validations = [c for c in calls if c.get("tool") == "mcp__kyb_preflight__validate_claims"]
    wrappers = [c for c in calls if c.get("tool") in {"code-mode-wrapper", "exec"}]
    ids = {c.get("id") for c in validations}
    n = len(checks)
    return (
        scope.get("role") == "kyb-reader"
        and scope.get("version") == 1
        and record.get("target") == {k: scope.get(k) for k in ("case", "run", "doc")}
        and record.get("canonical")
        == f"kyb-reader {scope.get('case')} {scope.get('run')} {scope.get('doc')}"
        and 1 <= n <= 3
        and len(validations) == n
        and len(wrappers) == n
        and len(calls) == 2 * n
        and len(ids) == n
        and None not in ids
        and len(hooks) == n
        and {h.get("call_id") for h in hooks} == ids
        and all(
            h.get("allowed") is True and h.get("session_id") == snap.get("child") for h in hooks
        )
        and snap.get("native_tools") == [c["tool"] for c in calls]
        and snap.get("tools") == ["validate_claims"] * n
        and all(
            c.get("attempt") == i
            and c.get("scope_hash") == sha256_json(scope)
            and c.get("role") == "kyb-reader"
            for i, c in enumerate(checks, 1)
        )
        and checks[-1].get("valid") is True
        and checks[-1].get("claims_hash")
        == proof.get("final_claims_hash")
        == record.get("output_hash")
    )


def identity(events: list[dict], session: str | None, agent: str | None) -> dict | None:
    matches = [d for d in by_action(events, RUN)
               if d.get("session") == session and d.get("agent_id") == agent]  # fmt: skip
    if not matches:
        return None
    if len(matches) != 1:
        return {"reasons": ["ambiguous managed Codex process receipts"]}
    record = matches[0]
    snap = record.get("lifecycle", {})
    reasons = []
    if (record.get("version") != "codex-managed-v1" or not record.get("closed")
            or record.get("exit_code") != 0 or not record.get("guard_observed")
            or snap.get("complete") is not True or not validation_bound(record)
            or not snap.get("fresh") or snap.get("provider") != "openai"
            or snap.get("model_evidence_scope") != "harness-active"
            or len(snap.get("models", [])) != 1
            or not record.get("deny_hook_confirmed")):  # fmt: skip
        reasons.append("managed Codex process lacks complete scoped runtime evidence")
    if snap.get("prompts") != [record.get("canonical")]:
        reasons.append("managed Codex process has non-canonical context")
    served = record.get("served_response_models") or {}
    ids = {t.get("message_id") for t in snap.get("turns", [])}
    if (
        record.get("served_model_scope") != "authenticated-response-completed"
        or not ids
        or not ids <= set(served)
        or len(set(served.values())) != 1
        or any(model_family(m) != "openai" for m in served.values())
    ):
        reasons.append("managed Codex process lacks bound served-response models")
    return {**record, "reasons": reasons}


def attest(events: list[dict], *, token: str, tool: str, input_hash: str,
           allowed_types: frozenset[str], target: dict) -> Attestation | None:  # fmt: skip
    matches = [d for d in by_action(events, ATTEST) if d.get("token_hash") == token_hash(token)]
    if not matches:
        return None
    call = matches[0]
    run = identity(events, call.get("session"), call.get("agent_id")) or {}
    reasons = list(run.get("reasons", []))
    if len(matches) != 1 or not run:
        reasons.append("missing or ambiguous managed Codex attestation")
    if (call.get("tool") != tool or call.get("input_hash") != input_hash
            or call.get("target") != target or run.get("agent_type") not in allowed_types
            or call.get("target") != run.get("target")
            or run.get("submission_hash") != input_hash):  # fmt: skip
        reasons.append("managed Codex attestation differs from the executed input or role")
    if any(d.get("token_hash") == call.get("token_hash")
           for d in by_action(events, "harness.attestation_consumed")):  # fmt: skip
        reasons.append("this attestation was already consumed")
    return Attestation(
        "unattested" if reasons else "isolated",
        agent_id=call.get("agent_id"),
        agent_type=run.get("agent_type"),
        tool_use_id=call.get("tool_use_id"),
        session=call.get("session"),
        token_hash=call.get("token_hash"),
        launch_id=call.get("session"),
        reasons=reasons,
        model=(next(iter(set(run["served_response_models"].values()))) if not reasons else None),
    )
