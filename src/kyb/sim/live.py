"""`LiveDecider`: replay of real end-to-end pipeline runs on rendered synthetic packets.

`make dogfood` runs the whole pipeline (Claude readers, the Vertex witness, L4 `decide`) on one
packet and writes a summary under artifacts/dogfood/. `record` turns such a run into a small,
names-free recording (datasets/synth/live/<case_id>.json): the decided status, score, fired rules,
reasons, the verified/claims counts and the cost. `LiveDecider` replays those recordings, so
`make sim DECIDER=live` makes zero calls. A run that stopped before a decision is recorded as a
fail-closed MANUAL_REVIEW (`reached: false`), never as a decision the pipeline made. A case
with no recording raises `LiveUnavailable`; the simulation of a live decider runs on the recorded
sample only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kyb.contracts import b4
from kyb.run.records import read_record
from kyb.sim.decide import Decision
from kyb.sim.l4 import decision_from
from kyb.synth.case import CaseSpec

ROOT = Path(__file__).resolve().parents[3]
RECORDINGS = ROOT / "datasets" / "synth" / "live"
VERSION = "live-v1"


class LiveUnavailable(LookupError):
    """No recorded live run for this case."""


def record(summary_path: Path, out: Path = RECORDINGS) -> Path:
    """Write the recording for one dogfood summary (a run of a synthetic packet, to `decided`)."""
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    ws = summary["workspace"]
    docs = ws.get("documents") or {}
    if ws.get("decision"):
        project = Path(summary["run_dir"]) / "project"
        rec = b4.DecisionRecord.model_validate(
            read_record(project / ws["workspace"], ws["decision"]["record"])
        )
        d = decision_from(rec)
        decided: dict[str, Any] = {
            "reached": True,
            "status": d.status,
            "score": d.score,
            "fired_rules": list(d.fired_rules),
            "reasons": list(d.reasons),
            "evidence": list(d.evidence),
            "decision_id": rec.decision_id,
            "judge_vetoes": sorted(
                {f"{j.judge_id}:{j.effect}" for j in rec.judge_outcomes if j.verdict == "veto"}
            ),
        }
    else:  # the pipeline failed closed before `decide`: an officer would get the case
        decided = {
            "reached": False,
            "status": "MANUAL_REVIEW",
            "score": 0,
            "fired_rules": [],
            "reasons": [f"pipeline stopped before a decision (next step: {ws.get('next')})"],
            "evidence": [],
        }
    body: dict[str, Any] = {
        "version": VERSION,
        "case_id": summary["packet"],
        "run": summary["run"],
        "git_head": summary["git_head"],
        "stratum": summary["stratum"],
        "gate": summary["gate"],
        "failures": summary["failures"],
        "decision": decided,
        "documents": len(docs),
        "claims": sum(x["claims"] for x in docs.values()),
        "verified": sum(x["verified"] or 0 for x in docs.values()),
        "assurance": {k: x["assurance"] for k, x in sorted(docs.items())},
        "next": ws.get("next"),
        "cost_usd": summary["agent"].get("cost_usd"),
        "seconds": summary["seconds"],
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{body['case_id']}.json"
    path.write_text(json.dumps(body, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_recordings(root: Path = RECORDINGS) -> dict[str, dict]:
    out = {}
    for p in sorted(root.glob("*.json")) if root.is_dir() else []:
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec.get("version") == VERSION:
            out[rec["case_id"]] = rec
    return out


class LiveDecider:
    """Replays recorded live runs: zero calls. `calls` stays 0 by construction."""

    name = "live-pipeline-replay"
    calls = 0

    def __init__(self, root: Path = RECORDINGS) -> None:
        self.recordings = load_recordings(root)

    def decide(self, case: CaseSpec) -> Decision:
        rec = self.recordings.get(case.case_id)
        if rec is None:
            raise LiveUnavailable(f"no recorded live run for {case.case_id}")
        d = rec["decision"]
        return Decision(
            status=d["status"],
            score=d["score"],
            fired_rules=tuple(d["fired_rules"]),
            reasons=tuple(d["reasons"]),
            evidence=tuple(d["evidence"]),
        )
