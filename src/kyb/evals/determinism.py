"""Suite 5 (determinism and drift): the same inputs give the same bytes, N times.

- policy parse, and evaluate + compose on every golden and synthetic case, N=5: one hash each;
- the witness: each vendored run re-verified by replaying its recorded answers, N=5, gives the
  product's recorded verdicts every time (crops re-cut by the product from the packet PDF, so this
  also pins the renderer: a poppler change that moved a pixel would miss the recording).

- the decision: each golden packet through `kyb decide` N=5 times in fresh workspaces, and once
  more in the same workspace, gives one decision_id and a byte-identical body (the clock and the
  audit envelope are outside the identity), with the network closed, so zero calls.

The live drift run (re-read, re-witness, re-answer without the cache: `make verify LIVE=1`,
spec) is pending.
"""

from __future__ import annotations

import contextlib
import json
import socket
from collections.abc import Iterator

from kyb.evals.decisions import all_cases, decide
from kyb.evals.paths import POLICY, datasets
from kyb.evals.renderer import RendererIncomparable
from kyb.evals.scorecard import Case, Suite, check, incomparable, pending
from kyb.evals.witness import fullrun
from kyb.evals.witness.record import Recorder
from kyb.policy import parse_policy
from kyb.util.hashing import sha256_json

N = 5


def decision_hash(packet) -> str:
    ev, comp = decide(packet)
    return sha256_json(
        {
            "rules": [r.model_dump(mode="json") for r in ev.rule_results],
            "composition": comp.model_dump(mode="json"),
        }
    )


def suite(n: int = N, witness_runs: int = N) -> Suite:
    s = Suite("5", "Determinism & drift", True, threshold=f"replay N={n}: one fingerprint")
    raw = POLICY.read_bytes()
    pols = {sha256_json(parse_policy(raw).model_dump(mode="json")) for _ in range(n)}
    s.cases.append(check("policy-parse", len(pols) == 1, f"{len(pols)} distinct parses"))
    split = []
    for cid, _, packet in all_cases():
        if len({decision_hash(packet) for _ in range(n)}) != 1:
            split.append(cid)
    s.cases.append(
        check(
            "decide-replay", not split, f"non-deterministic: {split}", cases=len(all_cases()), n=n
        )
    )
    for rd in fullrun.runs():
        pin = fullrun.recording_backend(rd)  # the run's own witness, replayed
        want = fullrun.expected_verdicts(rd)
        try:
            got = [
                fullrun.reverify(
                    rd,
                    Recorder(
                        datasets() / "witness" / "recordings" / pin["id"],
                        pin,
                        seeds=fullrun.seeds(rd),
                    ),
                )["verdicts"]
                for _ in range(witness_runs)
            ]
        except RendererIncomparable as exc:
            s.cases.append(incomparable(f"witness-replay:{rd.name}", str(exc)))
            continue
        except LookupError as exc:
            s.cases.append(
                pending(
                    f"witness-replay:{rd.name}",
                    "a recording for this prompt version (make verify LIVE=1)",
                    str(exc),
                )
            )
            continue
        s.cases.append(
            check(
                f"witness-replay:{rd.name}",
                all(g == want for g in got),
                "replayed verdicts differ from the version-bound expected verdicts",
                n=witness_runs,
                claims=len(want),
            )
        )
    s.cases.append(_fingerprint(n))
    s.cases.append(
        pending(
            "live-drift",
            "genuinely live-only: it compares a fresh live recording against the replay, so it "
            "needs a second live agent-path run (spec) taken after the recorded one, "
            "`make verify LIVE=1`. Replaying a recording against itself would pass by "
            "construction and measure nothing",
            "re-read, re-witness and re-answer without the cache; field and "
            "decision flips on the critical set = 0",
        )
    )
    return s


_BODY = {"kyb_decision": {"audit_trail", "emitted_at"}}


def _fingerprint(n: int) -> Case:
    from kyb.audit.chain import workspace_action
    from kyb.decide.action import decide_case
    from kyb.evals import agentpath as ap

    split, calls = [], 0
    with _network_closed() as attempts:
        for letter in "ABC":
            runs = [ap.decide_fixture(letter, label=f"{letter}-n{i}") for i in range(n)]
            first = runs[0]
            with ap.offline_support(), workspace_action(first.ws):
                again = decide_case(first.ws, first.packet.facts.case, first.packet.facts.run)
            bodies = {
                json.dumps(r.record.model_dump(mode="json", exclude=_BODY), sort_keys=True)
                for r in runs
            }
            ids = {r.record.decision_id for r in runs} | {again.decision_id}
            if len(bodies) != 1 or len(ids) != 1 or again.fingerprint != first.record.fingerprint:
                split.append(letter)
        calls = len(attempts)
    return check(
        "decision-fingerprint",
        not split and calls == 0,
        f"non-deterministic: {split}; network attempts: {calls}",
        packets=3,
        n=n,
    )


@contextlib.contextmanager
def _network_closed() -> Iterator[list]:
    """Within the block any attempt to open a connection is refused and counted."""
    attempts: list = []
    real = socket.socket.connect

    def refuse(self, address):
        attempts.append(address)
        raise OSError("network closed: offline replay makes no calls")

    socket.socket.connect = refuse
    try:
        yield attempts
    finally:
        socket.socket.connect = real
