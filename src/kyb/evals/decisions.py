"""Suite 4 (decisions, E-p): golden and synthetic cases through L3's `evaluate` and `compose`.

The cases start from fixture FactSets, so `evaluate` + `compose` are proven against the locked
labels; the same golden and synthetic cases then go through L4's real decision path (`kyb decide`
for the goldens, the decision assembly with attested predicate answers for the variants), and
packet → FactSet → screening → decision (L2 + L4) is run on the recorded claims. The headline
number is false-APPROVE: an APPROVED where the label is anything else, counted on every path.
Calibration needs model confidences and stays pending (`LIVE=1`).
"""

from __future__ import annotations

from functools import cache
from typing import Any

from kyb.contracts import b4
from kyb.evals import cases
from kyb.evals.paths import POLICY
from kyb.evals.scorecard import Case, Suite, check, failed, pending
from kyb.policy import Composition, Evaluation, compose, evaluate, parse_policy


@cache
def base_policy() -> b4.PolicyModel:
    return parse_policy(POLICY.read_bytes())


def decide(p: cases.Packet, policy: b4.PolicyModel | None = None) -> tuple[Evaluation, Composition]:
    pol = policy or base_policy()
    ev = evaluate(p.facts, p.screening, pol, p.answers)
    return ev, compose(ev.rule_results, pol)


def fired(ev: Evaluation) -> set[str]:
    return {r.rule_id for r in ev.rule_results if r.fired == "yes"}


def judge(case_id: str, expected: dict[str, Any], ev: Evaluation, comp: Composition) -> Case:
    got = fired(ev)
    problems = []
    if comp.status != expected["status"]:
        problems.append(f"status {comp.status} ≠ {expected['status']}")
    if "risk_score" in expected and comp.risk_score != expected["risk_score"]:
        problems.append(f"score {comp.risk_score} ≠ {expected['risk_score']}")
    if "fired" in expected and got != set(expected["fired"]):
        problems.append(f"fired {sorted(got)} ≠ {expected['fired']}")
    if missing := set(expected.get("includes", [])) - got:
        problems.append(f"missing fired {sorted(missing)}")
    if wrong := set(expected.get("not_fired", [])) & got:
        problems.append(f"fired but must not: {sorted(wrong)}")
    unknown = {r.rule_id for r in ev.rule_results if r.fired == "unknown"}
    if missing := set(expected.get("unknown", [])) - unknown:
        problems.append(f"not unknown: {sorted(missing)}")
    unevidenced = [r.rule_id for r in ev.rule_results if r.fired == "yes" and not r.evidence]
    if unevidenced:
        problems.append(f"fired without evidence (§0.3): {unevidenced}")
    return check(
        case_id,
        not problems,
        "; ".join(problems),
        status=comp.status,
        risk_score=comp.risk_score,
        fired=sorted(got),
    )


def all_cases() -> list[tuple[str, dict[str, Any], cases.Packet]]:
    out = [(g["id"], g["expected"], cases.golden_packet(k)) for k, g in cases.golden().items()]
    out += [(v["id"], v["expected"], p) for v, p in cases.variants()]
    return out


def suite() -> tuple[Suite, int]:
    s = Suite(
        "4",
        "Decisions (E-p): golden + synthetic",
        True,
        threshold="false-APPROVE = 0 · status 100% on golden + synthetic",
    )
    tp = fp = fn = false_approve = false_reject = agree = 0
    by_outcome: dict[str, int] = {}
    for cid, exp, packet in all_cases():
        ev, comp = decide(packet)
        s.cases.append(judge(cid, exp, ev, comp))
        want = exp["status"]
        by_outcome[want] = by_outcome.get(want, 0) + 1
        agree += comp.status == want
        false_approve += comp.status == "APPROVED" and want != "APPROVED"
        false_reject += comp.status == "REJECTED" and want != "REJECTED"
        if "fired" in exp:  # exact labels only
            got, lab = fired(ev), set(exp["fired"])
            tp, fp, fn = tp + len(got & lab), fp + len(got - lab), fn + len(lab - got)
    n = len(s.cases)
    s.metrics = {
        "cases": n,
        "by_expected_outcome": dict(sorted(by_outcome.items())),
        "status_agreement": round(agree / n, 4) if n else None,
        "false_approve": false_approve,
        "false_reject": false_reject,
        "fired_rule_precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "fired_rule_recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "source": "fixture FactSets (L3 evaluate + compose); `kyb-decide:`, `end-to-end-*` cases "
        "run the same cases through L4 and from recorded claims",
    }
    false_approve += _decide_path(s)
    s.metrics["false_approve"] = false_approve
    s.cases.append(
        pending(
            "calibration",
            "needs a labelled predicate set: the live kyb-predicates subagent's confidences are "
            "recorded (datasets/verify/recordings/predicates, 8 answers on C), but only 1 of them "
            "(poa-06) has a golden label, so a Brier / ECE would be one point, not a measure",
            "Brier / ECE of predicate confidence against labels",
        )
    )
    return s, false_approve


def _decide_path(s: Suite) -> int:
    """Golden + synthetic through L4, and packet → facts → decision from the recorded claims."""
    from kyb.evals import agentpath as ap

    false_approve = 0
    for letter, d in ap.golden_decided().items():
        exp = cases.golden()[letter]["expected"]
        k = d.record.kyb_decision
        fired_now = {r.rule_id for r in d.record.rule_results if r.fired == "yes"}
        want_status = "MANUAL_REVIEW" if letter == "A" else exp["status"]
        bad = [
            f"{what} {got} ≠ {want}"
            for what, got, want in (
                ("status", k.status, want_status),
                ("score", k.risk_score, exp["risk_score"]),
                ("fired", sorted(fired_now), sorted(exp["fired"])),
            )
            if got != want
        ]
        false_approve += k.status == "APPROVED" and exp["status"] != "APPROVED"
        s.cases.append(
            check(f"kyb-decide:golden-{letter}", not bad, "; ".join(bad), status=k.status)
        )
    for label, packet in cases.variants():
        want = label["expected"]
        results = {}
        for mode in ("healthy", "absent", "w1"):
            rec, _, _ = ap.synthetic_decision(label, packet, mode)
            results[mode] = rec.kyb_decision
            false_approve += rec.kyb_decision.status == "APPROVED" and want["status"] != "APPROVED"
            false_approve += mode == "w1" and rec.kyb_decision.status == "APPROVED"
        healthy = results["healthy"]
        bad = []
        if healthy.status != want["status"]:
            bad.append(f"status {healthy.status} ≠ {want['status']}")
        if "risk_score" in want and healthy.risk_score != want["risk_score"]:
            bad.append(f"score {healthy.risk_score} ≠ {want['risk_score']}")
        if results["absent"].status not in (want["status"], "MANUAL_REVIEW"):
            bad.append(f"support-unavailable status {results['absent'].status}")
        if results["w1"].status == "APPROVED":
            bad.append("approved on W1-only verification (spec)")
        s.cases.append(
            check(f"kyb-decide:{label['id']}", not bad, "; ".join(bad), status=healthy.status)
        )
    for letter, d in ap.derived_decided().items():
        exp = cases.golden()[letter]["expected"]
        k = d.record.kyb_decision
        false_approve += k.status == "APPROVED" and exp["status"] != "APPROVED"
        want_status = "MANUAL_REVIEW" if letter == "A" else exp["status"]
        s.cases.append(
            check(
                f"end-to-end-status:{letter}",
                k.status == want_status,
                f"packet → facts → screening → decide gives {k.status}, label {want_status}",
                score=k.risk_score,
            )
        )
        fired_now = sorted(r.rule_id for r in d.record.rule_results if r.fired == "yes")
        exact = (
            k.status == want_status
            and k.risk_score == exp["risk_score"]
            and fired_now == sorted(exp["fired"])
        )
        if exact:
            s.cases.append(check(f"end-to-end-golden:{letter}", True, status=k.status))
        else:
            # measured and not met: a failure, not a wait (it fails safe, see the reason)
            s.cases.append(
                failed(
                    f"end-to-end-golden:{letter}",
                    f"{_E2E_NEEDS[letter]}. From the recorded claims: status {k.status}, "
                    f"score {k.risk_score}, fired {fired_now}; label: score {exp['risk_score']}, "
                    f"fired {sorted(exp['fired'])}",
                    status=k.status,
                )
            )
    return false_approve


_E2E_NEEDS = {
    "A": "recorded claims for packet A do not reproduce the golden label",
    "B": "MEASURED, NOT MET: recorded live agent-path runs do not derive the full conflict and "
    "unresolved-ownership rule set because handwritten and degraded document spans leave dependent "
    "rules unknown. It fails safe to MANUAL_REVIEW: unreadable facts are never taken as clean",
    "C": "recorded live answers for C do not reproduce the golden label",
}
