"""The landed producers (L2 facts, L4 decide/judges, L5 copilot, L8 witness) driven from recorded
inputs, for the suites that used to be pending on them.

Nothing here makes a model call: reads are the recorded runs and hand fixtures, predicate answers
are the golden code labels submitted through the real hook and submission boundary, and the
support (J-SUPPORT) port is pinned to the spec states the product itself defines (healthy
stand-in, unavailable). The builders are the L4/L5 test fixtures (`tests/test_decide.py`,
`tests/test_facts.py`, `tests/decision_fixtures.py`), loaded from the checkout, so what the suites
run is what the product's own tests run.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import shutil
import sys
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any

from kyb.evals.paths import POLICY, REPO

TESTS = REPO / "tests"
WATCHLIST = POLICY.parent / "watchlists.json"
FACT_FIXTURES = {
    "A": "cl-andes-agro",
    "B": "es-iberia-software",
    "C": "sg-strait-capital",
}


@cache
def scratch() -> Path:
    """One scratch directory for the whole run, removed at exit."""
    root = Path(tempfile.mkdtemp(prefix="kyb-eval-agentpath-")).resolve()
    atexit.register(shutil.rmtree, root, True)
    return root


def _fresh(label: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=f"{label}-", dir=scratch()))


def workspace_root(label: str) -> Path:
    """An initialised workspace under the product's policy, in the scratch directory."""
    from kyb.workspace.init import init_workspace

    root = _fresh(label) / "ws"
    init_workspace(root, tenant="kyb-demo", policy_src=POLICY)
    return root


@cache
def fixtures() -> tuple[Any, Any]:
    """`tests/test_decide` and `tests/test_facts`: the product's own fixture builders."""
    if str(TESTS) not in sys.path:
        sys.path.append(str(TESTS))
    import test_decide
    import test_facts

    return test_decide, test_facts


@contextlib.contextmanager
def offline_support() -> Iterator[None]:
    """Pin the spec unavailable support state: no API key is read, no model is called."""
    from kyb.decide import action

    real = action.support_checker
    action.support_checker = lambda *a, **kw: None
    try:
        yield
    finally:
        action.support_checker = real


@dataclass(frozen=True)
class Decided:
    ws: Path
    packet: Any
    work: Any
    record: Any


def decide_fixture(letter: str, packet=None, *, label: str | None = None) -> Decided:
    """A golden (or given) packet through the real boundary: run → predicates → `decide_case`."""
    from kyb.audit.chain import workspace_action
    from kyb.decide.action import decide_case

    td, _ = fixtures()
    tmp = _fresh(label or letter)
    ws, packet, work = td.fixture_run(tmp, letter, packet)
    td.submit_fixture(ws, packet, work)
    with offline_support(), workspace_action(ws):
        record = decide_case(ws, packet.facts.case, packet.facts.run)
    return Decided(ws, packet, work, record)


@cache
def golden_decided() -> dict[str, Decided]:
    return {letter: decide_fixture(letter) for letter in "ABC"}


def derived_packet(letter: str):
    """Packet → FactSet from the recorded claims (L2 `derive_factset`) → screening (L2
    `screen_factset`, the product's watchlist and policy threshold), with the golden answers."""
    from kyb.evals.cases import Packet, golden_packet
    from kyb.screening.screen import policy_threshold, screen_factset

    _, tf = fixtures()
    facts = tf.facts(FACT_FIXTURES[letter])
    threshold = policy_threshold(POLICY.read_text("utf-8"))
    screening = screen_factset(
        facts,
        json.loads(WATCHLIST.read_bytes()),
        watchlist_version="eval-recorded",
        threshold=threshold,
    )
    return Packet(facts, screening, golden_packet(letter).answers)


@cache
def derived_decided() -> dict[str, Decided]:
    from kyb.evals.recorded import packet_with_live_answers

    return {
        letter: decide_fixture(
            letter,
            packet_with_live_answers(letter, derived_packet(letter))[0],
            label=f"{letter}-e2e",
        )
        for letter in "ABC"
    }


def synthetic_decision(label: dict, packet, mode: str):
    """A synthetic variant through L4's decision assembly with attested, code-labelled answers
    (the L4 test's path). `mode`: `healthy` (J-SUPPORT stand-in passes) or `absent` (the spec
    unavailable state: a favourable answer cannot approve), `w1` (healthy support, but every
    document verified by W1 only: the witness judges read that assurance as the product does)."""
    from kyb.audit.chain import workspace_action
    from kyb.contracts.b4 import DocVerification, PredicateAnswer
    from kyb.decide.core import decision_record
    from kyb.evals.cases import Packet
    from kyb.judges.core import fact_judges, outcome, predicate_judges, witness_judges
    from kyb.policy import load_policy

    td, _ = fixtures()
    facts = packet.facts.model_copy(
        update={
            "verification": [
                DocVerification(doc=d.doc, assurance="W1" if mode == "w1" else "W2", isolated=True)
                for d in packet.facts.documents
            ]
        }
    )
    tmp = _fresh("syn")
    ws, packet, work = td.fixture_run(
        tmp, "synthetic", Packet(facts, packet.screening, packet.answers)
    )
    proposal, events = td.attested_fixture_answers(ws, packet, work, tmp)
    support = (lambda i, a, e: outcome("J-SUPPORT", i.instance_id)) if mode != "absent" else None
    with workspace_action(ws):
        answers, judges = predicate_judges(
            packet.facts,
            work,
            [PredicateAnswer.model_validate(a) for a in proposal["answers"]],
            who=proposal["produced_by"]["predicates"],
            events=events,
            sources=proposal["sources"],
            support=support,
        )
    now = {
        d.doc: {"assurance": "W1" if mode == "w1" else "W2", "self_referential": False,
                "anomalies": {"count": 0, "severities": {}}}
        for d in packet.facts.documents
    }  # fmt: skip
    extra = witness_judges(now) if mode == "w1" else []
    record = decision_record(
        packet.facts,
        packet.screening,
        load_policy(POLICY),
        answers,
        [*judges, *fact_judges(packet.facts), *extra],
        [],
        provenance={"labels": "code-generated evaluation"},
        emitted_at=datetime(2026, 10, 26, tzinfo=UTC),
        audit_trail=events,
    )
    return record, judges, proposal
