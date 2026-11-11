"""Suite 8 (policy change): scripted edits to `compliance_rules.md`, re-decided.

Each edit is applied to the policy text, re-parsed by L3 and run on named cases. The suite checks
the expected decision delta, that the content hash changed (a new `rules_version`), and the guard:
an edit the engine cannot represent must be listed in `unrepresentable` and must never yield a
silent APPROVED or REJECTED.
"""

from __future__ import annotations

from kyb.evals.decisions import all_cases, base_policy, decide, judge
from kyb.evals.paths import POLICY, load
from kyb.evals.scorecard import Suite, check
from kyb.policy import parse_policy
from kyb.policy.rules import GUARD_ID


def edited_text(spec: dict) -> str:
    text = POLICY.read_text("utf-8")
    for e in spec["edits"]:
        if e["old"] not in text:
            raise ValueError(f"{spec['id']}: edit anchor not in the policy: {e['old']!r}")
        text = text.replace(e["old"], e["new"])
    if cut := spec.get("remove_section"):
        text = text[: text.index(cut[0])] + text[text.index(cut[1]) :]
    return text + spec.get("append", "")


def suite() -> Suite:
    s = Suite(
        "8",
        "Policy change",
        True,
        threshold="100%: expected deltas; guard on edits the engine cannot represent",
    )
    named = {cid: packet for cid, _, packet in all_cases()}
    base = base_policy()
    data = load("policy_edits/edits.json")
    for spec in data["edits"]:
        pol = parse_policy(edited_text(spec).encode("utf-8"))
        guarded = bool(pol.unrepresentable)
        s.cases.append(
            check(
                f"{spec['id']}:hash",
                pol.content_hash != base.content_hash,
                "content hash unchanged after an edit",
            )
        )
        if "guard" in spec:
            s.cases.append(
                check(
                    f"{spec['id']}:guard",
                    guarded == spec["guard"],
                    f"unrepresentable={pol.unrepresentable}",
                )
            )
        for k, v in spec.get("params", {}).items():
            s.cases.append(
                check(
                    f"{spec['id']}:param:{k}",
                    pol.params.get(k) == v,
                    f"{k}={pol.params.get(k)!r}, expected {v!r}",
                )
            )
        for c in spec["cases"]:
            cid = f"{spec['id']}:{c['case']}"
            exp, packet = c["expected"], named[c["case"]]
            ev, comp = decide(packet, pol)
            if exp.get("same_as_base"):
                ev0, comp0 = decide(packet, base)
                s.cases.append(
                    check(
                        cid,
                        ev.rule_results == ev0.rule_results and comp == comp0,
                        "a cosmetic edit changed the result",
                    )
                )
                continue
            case = judge(cid, exp, ev, comp)
            if guarded and comp.status != "MANUAL_REVIEW":
                case = check(cid, False, f"guarded edit yielded {comp.status}")
            s.cases.append(case)
            if spec.get("guard") and not any(r.fired == "unknown" for r in ev.rule_results):
                s.cases.append(
                    check(
                        f"{cid}:visible",
                        any(r.rule_id == GUARD_ID for r in ev.rule_results),
                        "guard is not visible",
                    )
                )
    s.metrics = {
        "edits": len(data["edits"]),
        "guarded": sum(bool(e.get("guard")) for e in data["edits"]),
    }
    return s
