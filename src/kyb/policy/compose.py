"""`compose`: rule results → status and risk score. Code sets status (E-p).

Pure and total. Precedence: a fired REJECT → REJECTED; otherwise a fired REVIEW, any `unknown`
result, or any judge veto → MANUAL_REVIEW; otherwise the score thresholds. Judges are veto-only:
a veto can turn APPROVED or REJECTED into MANUAL_REVIEW and nothing else.

A `POLICY-GUARD` result (a global policy section changed in a way the engine cannot apply)
acts like a veto: when the rules of decision themselves are not understood, the engine does not
reject either.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict

from kyb.contracts.b4 import JudgeOutcome, PolicyModel, RuleResult

Status = Literal["APPROVED", "REJECTED", "MANUAL_REVIEW"]
SCORE_REASON = "score above the auto-approval band"
GUARD_ID = "POLICY-GUARD"


class Composition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Status
    risk_score: int  # after the cap
    uncapped_score: int
    fired: list[str]  # rule ids fired `yes`, in result order
    unknown: list[str]  # rule ids that read `unknown`
    reasons: list[str]  # why the status is what it is, most decisive first


def compose(rule_results: Sequence[RuleResult], policy: PolicyModel,
            judge_outcomes: Iterable[JudgeOutcome] = ()) -> Composition:  # fmt: skip
    ids = [r.rule_id for r in rule_results]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate rule results: {sorted({i for i in ids if ids.count(i) > 1})}")
    fired = [r for r in rule_results if r.fired == "yes"]
    unknown = [r.rule_id for r in rule_results if r.fired == "unknown"]
    raw = sum(r.score_delta for r in fired)
    cap = policy.params.get("score_cap")
    threshold = policy.params.get("auto_approve_max_score")
    score = min(raw, int(cap)) if cap is not None else raw

    rejects = [r.rule_id for r in fired if r.severity == "REJECT"]
    reviews = [r.rule_id for r in fired if r.severity == "REVIEW"]
    vetoes = [j for j in judge_outcomes if j.verdict == "veto" and j.effect == "manual_review"]
    guarded = [r for r in rule_results if r.rule_id == GUARD_ID and r.fired != "no"]
    reasons: list[str] = []
    status: Status
    if rejects:
        status = "REJECTED"
        reasons.append(f"REJECT: {', '.join(rejects)}")
    elif reviews or unknown:
        status = "MANUAL_REVIEW"
        if reviews:
            reasons.append(f"REVIEW: {', '.join(reviews)}")
        if unknown:
            reasons.append(f"cannot be established: {', '.join(unknown)}")
    elif cap is None or threshold is None:
        status = "MANUAL_REVIEW"
        reasons.append("the policy's score cap or threshold could not be read")
    elif score > int(threshold):
        status = "MANUAL_REVIEW"
        reasons.append(f"{SCORE_REASON} ({score} > {int(threshold)})")
    else:
        status = "APPROVED"
        reasons.append(f"no REJECT or REVIEW trigger and score {score} ≤ {int(threshold)}")
    if vetoes or guarded:
        if status != "MANUAL_REVIEW":
            why = "judge veto" if vetoes else "unrepresentable policy text"
            reasons.insert(0, f"{why} overrides {status}")
        status = "MANUAL_REVIEW"
        reasons += [f"veto {j.judge_id} on {j.target}: {j.reason}" for j in vetoes]
        reasons += [r.description for r in guarded]
    return Composition(
        status=status, risk_score=score, uncapped_score=raw,
        fired=[r.rule_id for r in fired], unknown=unknown, reasons=reasons,
    )  # fmt: skip
