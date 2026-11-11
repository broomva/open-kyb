"""Load pinned inputs, run veto-only judges, emit an immutable DecisionRecord."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from kyb.audit.chain import AuditLog, chain_events
from kyb.contracts.b4 import (
    DECISION_REL,
    FACTS_REL,
    SCREENING_REL,
    DecisionRecord,
    FactSet,
    PredicateAnswer,
    ScreeningResult,
)
from kyb.corrections import comparable
from kyb.decide.core import decision_record
from kyb.decide.profile import resolve_profile
from kyb.judges.core import (
    FIXTURE_INPUT,
    STRICT,
    fact_judges,
    live_rules,
    outcome,
    predicate_judges,
    profile_outcome,
    witness_judges,
)
from kyb.judges.dispatch import checker_identity, support_checker
from kyb.policy import compose, evaluate, parse_policy
from kyb.policy.provenance import approved_policy
from kyb.predicates.lifecycle import lifecycle_reasons
from kyb.predicates.submit import load_answers
from kyb.run.action import amendment_claims, build_work_order, load_work_order, work_order_rel
from kyb.run.records import read_record, write_record
from kyb.util.fsutil import contained, safe_id
from kyb.util.hashing import sha256_bytes, sha256_json
from kyb.workspace.catalog import Catalog
from kyb.workspace.runs import FRESH_RUN, safe_run_id


def fixture_inputs(root: Path, case: str, run: str) -> list[str]:
    """Read trusted producer receipts for the two decision inputs; missing is unknown."""
    fixtures = []
    for template in (FACTS_REL, SCREENING_REL):
        rel = template.format(case=case, run=run)
        entry = Catalog(root).get(rel)
        if entry is None:
            raise LookupError(f"{rel}: no input producer receipt")
        if entry["produced_by"]["action"] in ("fixture_facts", "fixture_screen"):
            fixtures.append(rel)
    return fixtures


def inputs_for_decision(root: Path, case: str, run: str):
    # The explicit fixture adapter has its own producer; it cannot silently replace live facts.
    rel = FACTS_REL.format(case=case, run=run)
    entry = Catalog(root).get(rel)
    if entry and entry["produced_by"]["action"] == "fixture_facts":
        facts = FactSet.model_validate(read_record(root, rel))
    else:
        from kyb.facts.action import load_facts

        facts = load_facts(root, case, run)
    if sha256_json(facts.model_dump(mode="json", exclude={"fact_set_hash"})) != facts.fact_set_hash:
        raise PermissionError("FactSet hash differs from content")
    screening = ScreeningResult.model_validate(
        read_record(root, SCREENING_REL.format(case=case, run=run))
    )
    if (screening.case, screening.run, screening.fact_set_hash) != (case, run, facts.fact_set_hash):
        raise PermissionError("screening targets different facts")
    watchlist = contained(root, "policy/watchlists.json")
    screening_entry = Catalog(root).get(SCREENING_REL.format(case=case, run=run))
    fixture = screening_entry["produced_by"]["action"] == "fixture_screen"
    # Label-only fixtures have no live source pin; they are separately barred from approval
    # and dogfood. Hash-pinned fixtures still invalidate when that source changes.
    if not fixture or "#sha256:" in screening.watchlist_version:
        if not watchlist.exists():
            raise LookupError("screening watchlist source is missing; fresh screening is required")
        if screening.watchlist_version.rsplit("#", 1)[-1] != sha256_bytes(watchlist.read_bytes()):
            raise LookupError(f"watchlist changed: fresh screening is required. {FRESH_RUN}")
    policy = parse_policy(approved_policy(root))  # never an unapproved text (PR #80, F3)
    if not fixture and "name_similarity_threshold" in policy.params:
        from kyb.screening.screen import method

        if screening.method != method(float(policy.params["name_similarity_threshold"])):
            raise LookupError(
                f"screening method/threshold changed: fresh screening is required. {FRESH_RUN}"
            )
    work = build_work_order(facts, screening, policy, flagged_claims=amendment_claims(root, facts))
    if load_work_order(root, work.work_order_id) != work:
        raise PermissionError("work order differs from current inputs; run kyb run again")
    return facts, screening, policy, work


def _reader_judges(root: Path, case: str, run: str, docs: list[str], events: list[dict]):
    judges = []
    for doc in sorted(docs):
        rel = f"silver/claims/{case}/{run}/{doc}.json"
        if Catalog(root).get(rel) is None:
            judges.append(outcome("J-READER-ISOLATION", doc, "reader transcript unavailable"))
            continue
        record = read_record(root, rel)
        who = record["produced_by"]["reader"]
        reasons = lifecycle_reasons(
            events,
            who,
            f"kyb-reader {case} {run} {doc}",
            {"Read", "Glob", "read_plan", "read_document", "render_crop"},
        )
        judges.append(outcome("J-READER-ISOLATION", doc, "; ".join(reasons)))
    return judges


def _record_profile(root: Path, case: str, run: str, state: str, profile: str, why: str) -> None:
    """Chain event for a decision made under a non-default profile (once per state), so every
    approval under it is distinguishable in the audit trail and in the decision's own trail."""
    detail = {"action": "decide.profile", "case": case, "run": run, "state": state,
              "profile": profile, "summary": why}  # fmt: skip
    if any(e["detail"] == detail for e in chain_events(root) if e["component"] == "decision-core"):
        return
    AuditLog(root / "ledger/audit.jsonl").append(
        "decision-core",
        "tool.call",
        input_hash=sha256_json({"case": case, "run": run, "state": state}),
        output_hash=sha256_json({"profile": profile}),
        detail=detail,
    )


def witness_now(root: Path, case: str, run: str, docs: list[str]) -> dict[str, dict | None]:
    """Each document's L8 witness state through `assurance_now`, never the record's snapshot.
    A record whose bytes differ from its catalog receipt is refused."""
    from kyb.witness.verify import assurance_now

    catalog = Catalog(root)
    out: dict[str, dict | None] = {}
    for doc in sorted(docs):
        rel = f"silver/witness/{case}/{run}/{doc}.json"
        entry = catalog.get(rel)
        if entry is None:
            out[doc] = None
            continue
        if sha256_bytes(contained(root, rel).read_bytes()) != entry["content_hash"]:
            raise PermissionError(f"{rel}: bytes differ from its catalog receipt")
        now = assurance_now(root, case, run, doc)
        out[doc] = {k: now[k] for k in ("assurance", "self_referential", "anomalies")}
    return out


def _witness_judges(root: Path, case: str, run: str, docs: list[str]):
    return witness_judges(witness_now(root, case, run, docs))


def _notes(root: Path, case: str) -> dict:
    out = {}
    for kind in ("notes", "proposals"):
        directory = contained(root, f"cases/{case}/{kind}")
        for path in sorted(directory.rglob("*")):
            rel = str(path.relative_to(root))
            path = contained(root, rel)
            if path.is_file():
                out[rel] = {
                    "hash": sha256_bytes(path.read_bytes()),
                    "text": path.read_text("utf-8"),
                }
    return out


def revision_state(
    root: Path,
    case: str,
    run: str,
    work,
    proposal=None,
    *,
    support_identity: dict | None = None,
    profile: str | None = None,
) -> tuple[str, dict]:
    """The read-only state key used by review_status. A later lifecycle finding invalidates it.
    A decision under a non-default profile (spec) is a different state from the strict one."""
    from kyb.judges.core import JUDGES_VERSION
    from kyb.read.harness import hook_sources

    entries = Catalog(root).entries()
    actors = set()
    if proposal:
        for source in proposal.get("sources", {}).values():
            who = source["who"]
            if who.get("agent_id"):
                actors.add((who.get("session"), who["agent_id"]))
    for rel in entries:
        if rel.startswith(f"silver/claims/{case}/{run}/") and rel.endswith(".json"):
            who = read_record(root, rel)["produced_by"]["reader"]
            if who.get("agent_id"):
                actors.add((who.get("session"), who["agent_id"]))
    provenance = {
        "work_order": work.work_order_id,
        "proposal": proposal,
        "support": support_identity
        if support_identity is not None
        else checker_identity(support_checker(root)),
        "support_records": {
            rel: {"hash": entry["content_hash"], "record": read_record(root, rel)}
            for rel, entry in sorted(entries.items())
            if rel.startswith(f"gold/judgements/support-links/{work.work_order_id}/")
        },
        "notes": _notes(root, case),
        "inputs": {
            rel: entry
            for rel, entry in sorted(entries.items())
            if (
                rel.startswith(f"bronze/packets/{case}/")
                or rel.startswith(f"silver/witness/{case}/{run}/")
                or rel.startswith(f"silver/layout/{case}/{run}/")
                or rel
                in (FACTS_REL.format(case=case, run=run), SCREENING_REL.format(case=case, run=run))
            )
        },
        "lifecycle": [
            {"action": e["action"], "detail": e["detail"]}
            for e in chain_events(root)
            if e["action"]
            in (
                "harness.subagent_start",
                "harness.subagent_stop",
                "harness.reader_bind",
                "harness.predicate_bind",
            )
            and (e["detail"].get("session"), e["detail"].get("agent_id")) in actors
        ],
        "registrations": {
            session: sorted(hook_sources(chain_events(root), session))
            for session in sorted({s for s, _ in actors if s})
        },
    }
    profile = resolve_profile(profile)
    if profile != STRICT:  # absent for the default, so strict decisions stay byte-identical
        provenance["decision_profile"] = profile
    state = sha256_json(
        {
            "provenance": provenance,
            "judges": JUDGES_VERSION,
            "corrections": {
                rel: e["content_hash"]
                for rel, e in sorted(entries.items())
                if rel.startswith("ledger/corrections/")
            },
        }
    )
    return state.removeprefix("sha256:"), provenance


def state_prefix(case: str, run: str, state: str) -> str:
    return f"gold/judgements/{case}/{run}/state-{state}/"


def state_rel(case: str, run: str, state: str, support_revision: str) -> str:
    return state_prefix(case, run, state) + support_revision + ".json"


def decision_status(root: Path, case: str, run: str) -> dict:
    base = {"case": case, "run": run, "next": "decide", "available": True}
    if not Catalog(root).get(FACTS_REL.format(case=case, run=run)):
        return {**base, "stage": "run", "command": f"kyb run {case}"}
    try:
        _, _, _, work = inputs_for_decision(root, case, run)
    except LookupError, PermissionError:
        return {**base, "stage": "run", "command": f"kyb run {case}"}
    proposal = load_answers(root, work.work_order_id)
    state, _ = revision_state(root, case, run, work, proposal)
    prefix = state_prefix(case, run, state)
    current = {rel for rel in Catalog(root).entries() if rel.startswith(prefix)}
    # Explicit new support revisions append a new receipt, never overwrite an older result.
    ordered = [
        e["detail"].get("record")
        for e in chain_events(root)
        if e["detail"].get("action") == "decision_state"
    ]
    rel = next((rel for rel in reversed(ordered) if rel in current), None)
    if rel:
        link = read_record(root, rel)
        record = DecisionRecord.model_validate(read_record(root, link["decision"]))
        return {
            **base,
            "next": "decided",
            "decision": link["decision"],
            "decision_id": record.decision_id,
            "status": record.kyb_decision.status,
            "risk_score": record.kyb_decision.risk_score,
            "signature_state": record.signature_state,
        }
    if work.instances and (proposal is None or proposal.get("pending")):
        return {
            **base,
            "stage": "predicates",
            "work_order_id": work.work_order_id,
            "launch": f"kyb-predicates {work.work_order_id}",
        }
    return {**base, "stage": "decide", "command": f"kyb decide {case}"}


def decide_case(
    root: Path, case: str, run: str, *, support_port=None, profile: str | None = None
) -> DecisionRecord:
    case, run = safe_id(case, "case id"), safe_run_id(run)
    profile = resolve_profile(profile)
    facts, screening, policy, work = inputs_for_decision(root, case, run)
    proposal = load_answers(root, work.work_order_id)
    answers = [PredicateAnswer.model_validate(a) for a in proposal["answers"]] if proposal else []
    who = proposal["produced_by"]["predicates"] if proposal else None
    events = chain_events(root)
    checker = support_checker(root, port=support_port)
    if checker is not None:
        from kyb.decide.support import answerer_resolver

        checker.work_order_id = work.work_order_id
        checker.answerer_model = answerer_resolver(root, work, proposal)
    checked, judges = predicate_judges(
        facts,
        work,
        answers,
        who=who,
        events=events,
        sources=proposal.get("sources") if proposal else None,
        support=checker,
    )
    rules = evaluate(facts, screening, policy, checked).rule_results
    judges += fact_judges(facts, None if profile == STRICT else live_rules(rules, policy))
    fixtures = fixture_inputs(root, case, run)
    judges.append(
        outcome(FIXTURE_INPUT, "inputs", "fixture-produced inputs: " + ", ".join(fixtures))
        if fixtures
        else outcome(FIXTURE_INPUT, "inputs")
    )
    judges += _reader_judges(root, case, run, [d.doc for d in facts.documents], events)
    judges += _witness_judges(root, case, run, [d.doc for d in facts.documents])
    fired = compose(rules, policy).fired
    app = next(e for e in facts.entities if e.entity_id == facts.applicant)
    corrections = {
        c.correction_id: c for jur in app.jurisdiction for c in comparable(root, jur, fired)
    }
    state, provenance = revision_state(
        root, case, run, work, proposal, support_identity=checker_identity(checker), profile=profile
    )
    notes = provenance["notes"]
    if notes:
        judges.append(outcome("J-AGENT-CONTEXT", "decision", "agent notes/proposals considered"))
    if profile != STRICT:
        judges.append(profile_outcome(profile, judges))
        _record_profile(root, case, run, state, profile, judges[-1].reason)
    catalog = Catalog(root)
    record = decision_record(
        facts,
        screening,
        policy,
        checked,
        judges,
        [corrections[k] for k in sorted(corrections)],
        provenance=provenance,
        emitted_at=datetime.now(UTC),
        audit_trail=chain_events(root),
    )
    rel = DECISION_REL.format(decision_id=record.decision_id)
    path = contained(root, rel)
    if path.exists():
        # Recover an interrupted emission with its original clock and audit envelope.
        # The semantic body must match before an unreceipted file can be completed.
        prior = DecisionRecord.model_validate_json(path.read_bytes())
        exclude = {"kyb_decision": {"audit_trail", "emitted_at"}}
        if prior.model_dump(mode="json", exclude=exclude) != record.model_dump(
            mode="json", exclude=exclude
        ):
            raise PermissionError("published decision differs from current deterministic body")
        if (
            prior.kyb_decision.audit_trail
            != record.kyb_decision.audit_trail[: len(prior.kyb_decision.audit_trail)]
        ):
            raise PermissionError("published decision audit trail is not a prefix of this chain")
        record = prior
    # Exclude clocks and audit tail from the identity, but retain them in the immutable envelope.
    judgement_rel = f"gold/judgements/{case}/{run}/{record.decision_id}.json"
    inputs = [catalog.ref(work_order_rel(work.work_order_id))]
    write_record(
        root,
        judgement_rel,
        {
            "decision_id": record.decision_id,
            "judge_outcomes": [j.model_dump(mode="json") for j in record.judge_outcomes],
        },
        action="run_judges",
        tier="judged",
        as_of=facts.as_of,
        inputs=inputs,
    )
    write_record(
        root,
        rel,
        record.model_dump(mode="json"),
        action="emit_decision",
        tier="decided",
        as_of=facts.as_of,
        inputs=[*inputs, catalog.ref(judgement_rel)],
    )
    write_record(
        root,
        state_rel(
            case,
            run,
            state,
            sha256_json(
                [
                    j.model_dump(mode="json")
                    for j in record.judge_outcomes
                    if j.judge_id in ("J-SUPPORT", "j-support-unavailable")
                ]
            ).removeprefix("sha256:"),
        ),
        {"decision": rel, "provenance": provenance},
        action="decision_state",
        tier="decided",
        as_of=facts.as_of,
        inputs=[catalog.ref(rel)],
    )
    return record
