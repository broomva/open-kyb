"""The scorecard: cases → suites → one JSON document with a single exit status.

A suite fails on any failure, then is incomparable on unresolved renderer evidence, then pending
on missing work. Incomparable always blocks; strict mode also blocks pending. The
document holds no wall-clock time, so the same inputs give the same bytes (EVALS: offline,
deterministic).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

CaseStatus = Literal["pass", "fail", "pending", "incomparable"]
SCHEMA = "kyb-verify-scorecard-v2"
STATES = frozenset(("pass", "fail", "pending", "incomparable"))


@dataclass
class Case:
    case_id: str
    status: CaseStatus
    detail: str = ""
    needs: str = ""  # pending: the lane and hook it waits on, e.g. "L4 kyb.decide.emit_decision"
    data: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        if self.status not in STATES:
            raise ValueError(f"unknown case status: {self.status!r}")
        out: dict[str, Any] = {"id": self.case_id, "status": self.status}
        if self.detail:
            out["detail"] = self.detail
        if self.needs:
            out["needs"] = self.needs
        if self.data:
            out["data"] = self.data
        return out


def passed(case_id: str, detail: str = "", **data: Any) -> Case:
    return Case(case_id, "pass", detail, data=data)


def failed(case_id: str, detail: str, **data: Any) -> Case:
    return Case(case_id, "fail", detail, data=data)


def pending(case_id: str, needs: str, detail: str = "") -> Case:
    return Case(case_id, "pending", detail, needs=needs)


def incomparable(case_id: str, detail: str, **data: Any) -> Case:
    return Case(case_id, "incomparable", detail, data=data)


def check(case_id: str, ok: bool, detail: str = "", **data: Any) -> Case:
    return Case(case_id, "pass" if ok else "fail", "" if ok else detail, data=data)


@dataclass
class Suite:
    suite_id: str
    name: str
    blocking: bool
    cases: list[Case] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    threshold: str = ""

    @property
    def status(self) -> str:
        states = {c.status for c in self.cases}
        unknown = states - STATES
        if unknown:
            raise ValueError(f"unknown case statuses: {sorted(unknown)!r}")
        if "fail" in states:
            return "fail"
        if "incomparable" in states:
            return "incomparable"
        if "pending" in states or not self.cases:
            return "pending"
        return "pass"

    def counts(self) -> dict[str, int]:
        return {
            s: sum(c.status == s for c in self.cases)
            for s in ("pass", "fail", "pending", "incomparable")
        }

    def as_dict(self) -> dict[str, Any]:
        n = self.counts()
        return {
            "id": self.suite_id,
            "name": self.name,
            "blocking": self.blocking,
            "status": self.status,
            "threshold": self.threshold,
            "summary": f"{n['pass']}/{n['pass'] + n['fail']} pass · {n['fail']} fail · "
            f"{n['pending']} pending · {n['incomparable']} incomparable",
            "counts": n,
            "metrics": self.metrics,
            "cases": [c.as_dict() for c in self.cases],
        }


def build(
    suites: list[Suite], *, false_approve: int, inputs: dict[str, Any], strict: bool, live: bool
) -> dict[str, Any]:
    blocking = [s for s in suites if s.blocking]
    failing = [s.suite_id for s in blocking if s.status == "fail"]
    waiting = [s.suite_id for s in blocking if s.status == "pending"]
    unresolved = [s.suite_id for s in suites if any(c.status == "incomparable" for c in s.cases)]
    cases = [c for s in blocking for c in s.cases]
    n_pass = sum(c.status == "pass" for c in cases)
    n_run = sum(c.status in ("pass", "fail") for c in cases)
    n_incomparable = sum(c.status == "incomparable" for s in suites for c in s.cases)
    ok = not failing and not unresolved and false_approve == 0 and not (strict and waiting)
    return {
        "schema": SCHEMA,
        "verdict": "pass" if ok else "fail",
        "headline": f"{n_pass}/{n_run} blocking cases pass · {false_approve} false-APPROVE · "
        f"{len(waiting)} blocking suites pending · {n_incomparable} cases incomparable",
        "false_approve": false_approve,
        "blocking_failed": failing,
        "blocking_pending": waiting,
        "incomparable_suites": unresolved,
        "mode": {"live": live, "strict": strict},
        "inputs": inputs,
        "suites": [s.as_dict() for s in suites],
    }


def write(card: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(card, indent=1, ensure_ascii=False, sort_keys=False) + "\n", "utf-8")
