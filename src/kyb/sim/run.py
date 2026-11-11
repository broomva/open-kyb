"""`make sim`: run the autonomy-progression simulation over a built case set.

uv run python -m kyb.sim.run --run smoke --decider label --officer sim
uv run python -m kyb.sim.run --run fixture --decider fixture --officer sim
uv run python -m kyb.sim.run --run l4-jev --decider l4 --officer jev   # datasets/synth/jev
uv run python -m kyb.sim.run --run live-sim --decider live --officer sim   # datasets/synth/live
uv run python -m kyb.sim.run --run l4-jev --decider l4 --officer jev --jev record   # live Jev
"""

from __future__ import annotations

import argparse
from pathlib import Path

from kyb.sim.decide import Decider, FaultyDecider, FixtureDecider, LabelDecider
from kyb.sim.engine import SimConfig, by_week, simulate, totals
from kyb.sim.l4 import L4Decider
from kyb.sim.live import LiveDecider
from kyb.sim.officer import JevOfficer, SimOfficer, record_fixture
from kyb.sim.report import write
from kyb.synth.build import load_cases
from kyb.synth.label import label
from kyb.synth.sources import cache_root
from kyb.witness.jev import JevClient, JevPort

ROOT = Path(__file__).resolve().parents[3]
JEV_RECORDINGS = ROOT / "datasets" / "synth" / "jev"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="smoke")
    ap.add_argument(
        "--cases", type=Path, default=None, help="dir with cases.jsonl (default: cache)"
    )
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument(
        "--decider", choices=("label", "fixture", "l4", "live", "faulty"), default="label"
    )
    ap.add_argument("--officer", choices=("sim", "jev"), default="sim")
    ap.add_argument("--weeks", type=int, default=8)
    ap.add_argument("--per-week", type=int, default=10)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--audit-rate", type=float, default=0.25)
    ap.add_argument("--officer-error", type=float, default=0.0)
    ap.add_argument("--faulty-rate", type=float, default=0.15)
    ap.add_argument(
        "--jev",
        choices=("replay", "record", "scripted"),
        default="replay",
        help="replay: recorded answers only (zero calls); record: call Jev for what is not "
        "recorded (needs TYPESAFE_API_KEY); scripted: offline gold-label fixtures (not Jev)",
    )
    ap.add_argument("--jev-recordings", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "sim")
    a = ap.parse_args(argv)

    path = a.cases or cache_root() / "cases" / f"seed-{a.seed}-n{a.n}"
    if not (path / "cases.jsonl").is_file():
        ap.error(f"no cases at {path}: run `make synth-build` first")
    pool = load_cases(path)
    base: Decider = {
        "label": LabelDecider,
        "fixture": FixtureDecider,
        "l4": L4Decider,
        "live": LiveDecider,
        "faulty": FixtureDecider,
    }[a.decider]()
    decider = FaultyDecider(base, a.faulty_rate, a.seed) if a.decider == "faulty" else base
    if isinstance(base, LiveDecider):  # a live decider is defined on the recorded sample only
        pool = [c for c in pool if c.case_id in base.recordings]
        if not pool:
            ap.error("no live recordings: run scripts/sim/record_live.py first")
    gold = {c.case_id: label(c) for c in pool}

    if a.officer == "jev":
        if a.jev == "scripted":  # offline: replay fixtures from the gold labels (not Jev)
            root = a.jev_recordings or cache_root() / "jev-recordings" / f"seed-{a.seed}"
            if not root.is_dir():
                decisions = {c.case_id: (c, decider.decide(c)) for c in pool}
                gl = {k: v.status for k, v in gold.items()}
                record_fixture(root, decisions, gl, seed=a.seed)
            port = JevPort(root, None, mode="replay")
        else:
            root = a.jev_recordings or JEV_RECORDINGS
            adapter = JevClient() if a.jev == "record" else None
            port = JevPort(root, adapter, mode="auto" if adapter else "replay")
        officer = JevOfficer(port)
    else:
        officer = SimOfficer(error_rate=a.officer_error, seed=a.seed)

    cfg = SimConfig(weeks=a.weeks, per_week=a.per_week, k=a.k, audit_rate=a.audit_rate, seed=a.seed)
    res = simulate(pool, decider, officer, cfg, gold)
    out = write(res, a.run, a.out)
    t = totals(res)
    print(
        f"false approvals: {t['false_approvals']}  |  auto rate by week: "
        + " ".join(f"{100 * r['auto_rate']:.0f}%" for r in by_week(res))
    )
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
