"""Suite 7 (copilot Q&A, L5), the agent-path gates (spec) and the spec benchmarks (screening:
L2 fuzzy, the agent's same_party, Jev; support: Jev J-SUPPORT, L4).

The landed producers are called for real, offline, from recorded inputs (`evals/copilot_qa.py`,
`evals/agentpath.py`, `evals/screening_eval.py`). What still needs a live model is a pending case
with a reason that starts `needs LIVE=1:` (or `gold-label review` where the label itself is
disputed): never a pass.
"""

from __future__ import annotations

import importlib.util
import json
from collections import Counter

from kyb.evals import recorded
from kyb.evals.paths import REPO, load
from kyb.evals.scorecard import Case, Suite, check, failed, pending
from kyb.evals.witness import fullrun


def _landed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def qa() -> Suite:
    from kyb.evals import copilot_qa

    s = Suite(
        "7",
        "Copilot Q&A",
        True,
        threshold="0 fabricated answers on the unanswerable set · ≥ 90% correct",
    )
    data = load("qa/questions.json")["questions"]
    unanswerable = [q for q in data if not q["answerable"]]
    s.cases.append(
        check(
            "dataset-shape",
            len(unanswerable) >= 8 and all(q["citations"] for q in data if q["answerable"]),
            "needs ≥ 8 unanswerable and a gold citation for every answerable question",
        )
    )
    cases, st = copilot_qa.run()
    s.cases += cases
    wrong = recorded.qa_summary()["wrong_unanswerable"] if recorded.qa_sessions() else 0
    s.cases.append(
        check(
            "fabricated-answers",
            st["fabricated"] == 0 and wrong == 0,
            f"record_answer accepted {st['fabricated']} forged statements; a recorded session "
            f"answered {wrong} unanswerable question(s)",
            probes=st["forged_probes"],
            questions=len(data),
        )
    )
    s.cases.append(
        check(
            "answers-code-rendered",
            st["answers_not_code_rendered"] == 0,
            "an accepted answer's text is not the receipted, code-rendered one",
            answers=st["answers_checked"],
        )
    )
    agent = recorded.qa_summary() if recorded.qa_sessions() else None
    total = st["answerable"]
    if agent and not any(r[0] == "none" for k, r in agent["rows"].items()):
        s.cases.append(
            check(
                "correct-rate",
                agent["correct"] / total >= 0.9,
                f"{agent['correct']}/{total} answerable questions correct "
                f"({agent['correct'] / total:.0%}) in the recorded live sessions: the rest were "
                "answered 'not in the documents' or without the gold (see each question)",
                agent=agent["sessions"],
            )
        )
    elif st["offline_decided"] == total:
        s.cases.append(
            check("correct-rate", st["correct"] / total >= 0.9, f"{st['correct']}/{total} correct")
        )
    else:
        s.cases.append(
            pending(
                "correct-rate",
                f"needs `make verify-record`: an agent answering through ask + record_answer. "
                f"Offline, the reference answerer (no model) settled {st['offline_decided']}/"
                f"{total} answerable questions (all correct); the other "
                f"{total - st['offline_decided']} need the agent's reformulation or a read of "
                "the packet",
                "≥ 90% of answerable questions correct, with valid citations",
            )
        )
    s.metrics = {
        "questions": len(data),
        "unanswerable": len(unanswerable),
        "languages": dict(Counter(q["lang"] for q in data)),
        "reference_answerer": st,
        "recorded_sessions": (
            {k: v for k, v in agent.items() if k != "rows"} if agent else "none recorded"
        ),
        "gold_evidence_in_recorded_record": f"{st['evidence_in_record']}/{st['answerable']}",
        "gold_evidence_retrieved_into_bundle": (
            f"{st['retrieved_in_bundle'] + st['correct']}/{st['answerable']}"
        ),
    }
    return s


def agent_gates() -> Suite:
    """spec agent-path gates. They bind the live receipt; recorded runs report today's numbers."""
    s = Suite(
        "G",
        "Agent-path gates (spec)",
        True,
        threshold="unverified_answer = 0 · coverage of every material fact · isolation",
    )
    s.cases.append(_display())
    s.cases.append(
        recorded.unverified_answer_case()
        or pending(
            "unverified_answer",
            "needs `make verify-record qa`: the live Q&A sessions' Stop-hook display check",
            "no factual text shown to the officer that is not a code-rendered "
            "record_answer from that turn",
        )
    )
    cov, iso = {}, {}
    for rd in fullrun.runs():
        case, run = fullrun._case_run(rd)
        facts = matched = 0
        for f in sorted((rd / "silver" / "witness" / case / run).glob("*.json")):
            rec = json.loads(f.read_text("utf-8"))
            listed = [x for x in rec["coverage"] if x.get("kind") != "claim"]
            facts += len(listed)
            matched += sum(x["status"] == "matched" for x in listed)
        cov[rd.name] = {
            "material_facts": facts,
            "matched": matched,
            "recall": round(matched / facts, 4) if facts else None,
        }
        subs = [
            json.loads(f.read_text("utf-8"))
            for f in sorted((rd / "silver" / "claims" / case / run).glob("*.json"))
        ]
        iso[rd.name] = {
            "submissions": len(subs),
            "attested_isolated": sum(
                x["produced_by"]["reader"].get("context") == "isolated" for x in subs
            ),
        }
    s.cases.append(_coverage())
    s.cases.append(_isolation_reads())
    s.cases.append(_isolation_predicates())
    live_iso = recorded.isolation_predicates()
    if live_iso is not None:
        s.cases.append(live_iso)
    s.cases.append(_isolation(live_iso))
    for name, v in iso.items():
        s.cases.append(
            check(
                f"reader-submissions-isolated:{name}",
                v["attested_isolated"] == v["submissions"],
                json.dumps(v),
            )
        )
    s.cases.append(_adversarial_through_agent(live_iso))
    s.metrics = {
        "coverage_recorded_runs": cov,
        "isolation_recorded_runs": iso,
        "dogfood_latest_b": _dogfood_uncovered(),
        "coverage_live_abc": (recorded._read("agent/coverage.json") or {}).get("runs"),
    }
    return s


def _coverage() -> Case:
    """Every material fact matched, for A, B and C: measured on the live post-L1 runs."""
    rec = recorded._read("agent/coverage.json")
    if rec is None:
        gaps = _dogfood_uncovered()
        return pending(
            "coverage",
            "needs `make verify-record coverage`: a post-L1 agent-path run of A, B and C. "
            f"Recorded: runs of B only, leaving {gaps['uncovered']} of {gaps['facts']} material "
            "facts uncovered",
            "every material fact of the claim-blind layout matched by a verified claim, for "
            "A, B and C",
        )
    per = {
        k: (
            sum(d["matched"] for d in docs.values()),
            sum(d["material_facts"] for d in docs.values()),
        )
        for k, docs in rec["runs"].items()
    }
    shown = ", ".join(f"{k} {m}/{n}" for k, (m, n) in per.items())
    if all(m == n for m, n in per.values()):
        return check("coverage", True, shown)
    # measured, and the gate is not met: a failure. It fails safe: the product's
    # `coverage_incomplete` finding already sends every such run to MANUAL_REVIEW.
    return failed(
        "coverage",
        f"MEASURED, NOT MET: live post-L1 runs of A, B and C (run {rec['source_run']}) match "
        f"{shown} material facts; the reader leaves the rest uncovered, so each document carries "
        "coverage_incomplete. It fails safe to MANUAL_REVIEW (spec). Gate: every material fact "
        "of the claim-blind layout matched by a verified claim, for A, B and C",
        matched=dict(per),
    )


def _isolation(live: Case | None) -> Case:
    if live is None:
        return pending(
            "isolation",
            "needs `make verify-record predicates`: a live predicates attestation. Offline, "
            "recorded reads and the predicates hook boundary pass above",
            "every read and predicate attested as its subagent, no resume, no injected messages",
        )
    reads = _isolation_reads()
    return check(
        "isolation",
        live.status == "pass" and reads.status == "pass",
        f"reads: {reads.status}; the recorded live predicates subagent: {live.status}",
    )


def _adversarial_through_agent(live_iso: Case | None) -> Case:
    visible, injected = recorded.visible_case(), recorded.injection_case()
    if visible is None or injected is None:
        return pending(
            "adversarial-through-agent",
            "needs `make verify-record`: suite 6's model-side cases "
            "(adv-visible-page-instruction, adv-predicates-subagent-injection)",
            "suite 6 runs through the agent path, reader and predicates included",
        )
    ok = visible.status == injected.status == "pass" and (
        live_iso is None or live_iso.status == "pass"
    )
    return check(
        "adversarial-through-agent",
        ok,
        f"reader (injected packet, live): {visible.status}; predicates (injected excerpt, live "
        f"subagent + Jev): {injected.status}",
    )


def _dogfood() -> list[dict]:
    root = REPO / "artifacts" / "dogfood"
    out = (
        [json.loads(p.read_text("utf-8")) for p in sorted(root.glob("*.json"))]
        if root.exists()
        else []
    )
    return [d for d in out if (d.get("workspace") or {}).get("documents")]


def _dogfood_uncovered() -> dict:
    claude = [d for d in _dogfood() if d["harness"] == "claude" and d["gate"] == "pass"]
    docs = claude[-1]["workspace"]["documents"] if claude else {}
    return {
        "run": claude[-1]["run"] if claude else None,
        "uncovered": sum(d.get("uncovered") or 0 for d in docs.values()),
        "facts": sum((d.get("uncovered") or 0) + (d.get("verified") or 0) for d in docs.values()),
    }


def _display() -> Case:
    """The display check, on the offline copilot: a rendered answer shown verbatim is clean, a
    paraphrase of it is flagged `unverified_answer`."""
    from kyb.audit.chain import workspace_action
    from kyb.copilot.display import ingest_session
    from kyb.evals import agentpath as ap
    from kyb.evals import copilot_qa

    d = ap.golden_decided()["C"]
    case, run = d.packet.facts.case, d.packet.facts.run
    out = copilot_qa._answer(
        d.ws,
        case,
        run,
        "why?",
        [{"type": "explain", "decision_id": d.record.decision_id, "rule_ids": []}],
    )

    def line(kind, content):
        return json.dumps({"type": kind, "message": {"role": kind, "content": content}})

    tool = "mcp__plugin_kyb_kyb__record_answer"
    call = [{"type": "tool_use", "id": "u1", "name": tool, "input": {}}]
    result = [
        {
            "type": "tool_result",
            "tool_use_id": "u1",
            "content": [{"type": "text", "text": json.dumps({"answer_id": out["answer_id"]})}],
        }
    ]
    clean = [
        line("user", "why?"),
        line("assistant", call),
        line("user", result),
        line("assistant", [{"type": "text", "text": out["text"]}]),
    ]
    para = [
        line("user", "and the entity?"),
        line(
            "assistant",
            [{"type": "text", "text": f"{d.record.kyb_decision.entity_name} looks fine to me."}],
        ),
    ]
    path = d.ws.parent / "session.jsonl"
    path.write_text("\n".join(clean) + "\n")
    with workspace_action(d.ws):
        shown = ingest_session(d.ws, "eval", str(path))
    path.write_text("\n".join([*clean, *para]) + "\n")
    with workspace_action(d.ws):
        flagged = ingest_session(d.ws, "eval", str(path))
    return check(
        "display-check-offline",
        shown["unverified_answer"] == 0 and flagged["unverified_answer"] == 1,
        f"rendered answer flagged: {shown}; paraphrase flagged: {flagged} (want 0 and 1)",
    )


def _isolation_reads() -> Case:
    bad = []
    for d in _dogfood():
        if d["harness"] == "claude" and d["gate"] == "pass":
            for doc, v in d["workspace"]["documents"].items():
                if v["context"] != "isolated" or v.get("context_now", "isolated") != "isolated":
                    bad.append(f"{d['run']}/{doc}")
    return check(
        "isolation-reads-recorded",
        not bad,
        f"reads not isolated: {bad}",
        receipts=sum(d["harness"] == "claude" and d["gate"] == "pass" for d in _dogfood()),
    )


def _isolation_predicates() -> Case:
    """Predicates through the real hook boundary: attested and stopped passes; a missing
    lifecycle (no stopped transcript) is vetoed, so an unattested subagent clears nothing."""
    from kyb.contracts.b4 import PredicateAnswer
    from kyb.evals import agentpath as ap
    from kyb.evals.cases import variants
    from kyb.judges.core import predicate_judges

    td, _ = ap.fixtures()
    label, packet = next(v for v in variants() if v[0]["id"] == "syn-a-former-pep")
    tmp = ap._fresh("isolation")
    ws, packet, work = td.fixture_run(tmp, "synthetic", packet)
    proposal, events = td.attested_fixture_answers(ws, packet, work, tmp)
    answers = [PredicateAnswer.model_validate(a) for a in proposal["answers"]]

    def verdicts(evts: list) -> set[str]:
        _, judges = predicate_judges(
            packet.facts,
            work,
            answers,
            who=proposal["produced_by"]["predicates"],
            events=evts,
            sources=proposal["sources"],
        )
        return {j.verdict for j in judges if j.judge_id == "J-PRED-ISOLATION"}

    ok = verdicts(events) == {"pass"} and verdicts([]) == {"veto"}
    return check(
        "isolation-predicates-attested",
        ok,
        f"attested lifecycle: {verdicts(events)}; no lifecycle: {verdicts([])} (want pass / veto)",
    )


def screening() -> Suite:
    data = load("screening/cases.json")["pairs"]
    s = Suite(
        "B-screening",
        "Benchmark: screening eval set (spec)",
        False,
        threshold="report-only: recall, precision, false clears per method",
    )
    s.cases.append(
        check(
            "dataset-shape",
            {"same", "different"} == {p["label"] for p in data}
            and {"transliteration", "reorder", "alias", "near_miss", "format_shifted_identifier"}
            <= {p["category"] for p in data},
            "every spec category and both labels",
        )
    )
    from kyb.evals import screening_eval

    report = screening_eval.fuzzy()
    s.cases.append(
        check(
            "method:fuzzy",
            report["different_corroborated"] == 0 and not report["false_clears"],
            f"false clears {report['false_clears']}; a different party corroborated: "
            f"{report['different_corroborated']}",
            recall=report["recall"],
            precision=report["precision"],
        )
    )
    hooks = {
        "agent-same-party": "needs LIVE=1: the kyb-predicates subagent answering same_party",
        "jev": "needs LIVE=1: a recorded Jev same_party judgement per pair; "
        "datasets/screening has no recordings",
    }
    for method, hook in hooks.items():
        s.cases.append(
            pending(
                f"method:{method}", hook, "recall / precision / false clears on the screening set"
            )
        )
    s.metrics = {
        "pairs": len(data),
        "by_category": dict(Counter(p["category"] for p in data)),
        "by_label": dict(Counter(p["label"] for p in data)),
        "fuzzy": {k: v for k, v in report.items() if k != "rows"},
    }
    return s


def _jev_support() -> dict:
    """J-SUPPORT on the recorded Jev judgements (no adapter is reachable), plus the eval's own
    fault: a port that calls everything supported must fail the gate."""
    from kyb.judges.support import (
        RECORDINGS,
        FixtureSupportPort,
        evaluate_support_set,
        load_target_cases,
        passes_false_pass_gate,
    )
    from kyb.witness.jev import JevPort, JevResponse

    class Closed:
        family = "typesafe-jev"

        def ask(self, body):
            raise AssertionError("offline replay called the adapter")

    class Always:
        family = "typesafe-jev"
        model = "jev-fault-always-supported"

        def answerer_model(self, instance, answer):
            return "gpt-6.1-sol"

        def ask(self, body):
            return JevResponse(
                model=self.model,
                answers={"supported": {"type": "noul", "noul": 1.0}},
                usage={"input_tokens": 1, "output_tokens": 1},
                request_hash="sha256:" + "ab" * 32,
                replayed=False,
            )

    fault = evaluate_support_set(Always())
    fault_caught = not passes_false_pass_gate(fault)
    if not (RECORDINGS / "v2" / "target-controls.json").exists():
        return {"recorded": False, "fault_caught": fault_caught}
    cases = load_target_cases()
    port = FixtureSupportPort(JevPort(RECORDINGS / "v2", Closed(), mode="replay"), cases)
    report = evaluate_support_set(port, cases)
    return {
        **report,
        "recorded": True,
        "gate": passes_false_pass_gate(report),
        "cases": len(report["cases"]),
        "fault_caught": fault_caught,
    }


def support() -> Suite:
    data = load("support/cases.json")["pairs"]
    s = Suite(
        "B-support",
        "Benchmark: support eval set (spec, J-SUPPORT)",
        False,
        threshold="report-only: agreement with labels; favourable-unsupported passes = 0",
    )
    fav = [
        p
        for p in data
        if p["value"] is True
        and p["predicate"] in ("natural_person_directs_business", "doc_is_proof_of_address")
    ]
    s.cases.append(
        check(
            "dataset-shape",
            {"supported", "unsupported"} == {p["label"] for p in data}
            and any(p["label"] == "unsupported" for p in fav),
            "both labels, and favourable values labelled unsupported",
        )
    )
    report = _jev_support()
    s.cases.append(
        check(
            "jev-j-support-fault-caught",
            report["fault_caught"],
            "an always-supported port is caught by the false-pass gate",
        )
    )
    if report.get("recorded"):
        s.cases.append(
            check(
                "jev-j-support",
                report["gate"] and report["fault_caught"],
                f"false passes {report['false_passes']} (want 0)",
                cases=report["cases"],
            )
        )
    else:
        s.cases.append(
            pending(
                "jev-j-support",
                (
                    "needs LIVE=1: no recorded Jev judgements under "
                    "examples/evals/support/recordings/v2"
                ),
                "agreement with labels; favourable-unsupported passes = 0",
            )
        )
    s.metrics = {
        "pairs": len(data),
        "by_predicate": dict(Counter(p["predicate"] for p in data)),
        "favourable_true": len(fav),
        "jev_replay": (
            {
                k: report[k]
                for k in (
                    "false_passes",
                    "veto_precision",
                    "veto_recall",
                    "support_threshold",
                    "model",
                    "cases",
                )
            }
            if report.get("recorded")
            else "none recorded"
        ),
    }
    return s


def benchmarks() -> Suite:
    """EVALS.md report-only benchmarks beyond the witness and spec sets: listed with their
    hooks so the scorecard shows what is not measured yet."""
    s = Suite(
        "B-adapters",
        "Benchmarks: extract/decide adapters, sources, retrieval (spec, spec, spec)",
        False,
        threshold="report-only: accuracy, calibration, cost, latency",
    )
    from kyb.evals import copilot_qa

    _, st = copilot_qa.run()
    s.cases.append(
        check(
            "d032-retrieval:bm25",
            True,
            "report-only",
            answerable=st["answerable"],
            gold_in_record=st["evidence_in_record"],
            gold_in_bundle=st["retrieved_in_bundle"] + st["correct"],
            correct_at_head=st["correct"],
        )
    )
    for cid, needs, what in (
        (
            "d032-retrieval:embeddings",
            "no embeddings retriever exists in kyb.copilot (BM25 only)",
            "Q&A correctness and citation validity with +embeddings",
        ),
        (
            "d031-extract-decide",
            "no Clef adapter exists, and no Jev extract/decide adapter "
            "(kyb.witness.jev serves J-SUPPORT only)",
            "Claude vs Clef/Jev on the golden + synthetic sets: status agreement, false-APPROVE",
        ),
        (
            "d030-sources",
            "no +verified / +open-context source packs exist for L2 facts",
            "bundle vs +verified vs +open-context: decision deltas",
        ),
    ):
        s.cases.append(pending(cid, needs, what))
    return s
