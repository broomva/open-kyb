"""Run the witness backend eval: every backend on the labelled set and the vendored runs, the
fault self-test, and the table (report-only backend comparison).

Replay by default. `live=True` records what is missing (the API key or `agy` is needed then); a
recording is never re-asked, so a live run is resumable and its cost is paid once.
"""

from __future__ import annotations

import json
import statistics
import subprocess
from pathlib import Path
from typing import Any

from kyb.evals.paths import datasets
from kyb.evals.renderer import RendererIncomparable
from kyb.evals.witness import backends, comparator, faults, fullrun, labelled
from kyb.evals.witness.record import Recorder


def recordings(backend_id: str) -> Path:
    return datasets() / "witness" / "recordings" / backend_id


def _recorder(b: dict, live: bool, seeds: list[Path] = ()) -> Recorder:
    # no adapter up front: it is built on the first recording miss, so a backend whose key or
    # CLI is unavailable still replays, and a live miss it cannot serve is recorded as missing.
    # A legacy (prohibited-route) backend is replay-only: a live miss stays missing, it never
    # builds the refused adapter.
    live = live and "legacy_route" not in b
    return Recorder(recordings(b["id"]), b, None, live=live, seeds=seeds)


def _tokens(u: dict) -> tuple[int, int]:
    """(in, out) from a recorded usage: Gemini's `usageMetadata` names, which the CLI adapters
    map onto."""
    t_out = int(u.get("candidatesTokenCount", 0)) + int(u.get("thoughtsTokenCount", 0))
    return int(u.get("promptTokenCount", 0)), t_out


def _usage(calls: list[dict]) -> dict[str, Any]:
    """Usage over every consumed reply, product seeds included, split by who spent it."""
    by: dict[str, list[int]] = {}
    cost = 0.0
    for c in calls:
        t_in, t_out = _tokens(c["usage"])
        acc = by.setdefault(c["source"], [0, 0, 0])
        acc[0], acc[1], acc[2] = acc[0] + t_in, acc[1] + t_out, acc[2] + 1
        cost += float(c["usage"].get("cost_usd") or 0)
    lat = [c["latency_s"] for c in calls if c.get("latency_s") is not None]
    return {
        "tokens_in": sum(v[0] for v in by.values()),
        "tokens_out": sum(v[1] for v in by.values()),
        "by_source": {
            k: {"tokens_in": v[0], "tokens_out": v[1], "requests": v[2]}
            for k, v in sorted(by.items())
        },
        "cost_usd": round(cost, 4),
        "requests": len(calls),
        "timed_requests": len(lat),
        "reported_models": sorted({c["reported_model"] for c in calls if c["reported_model"]}),
        "provenance_gaps": sorted({g for c in calls for g in c["gaps"]}),
        "latency_s_median": round(statistics.median(lat), 1) if lat else None,
        "latency_s_max": max(lat) if lat else None,
    }


def run_backend(b: dict, *, live: bool = False, only: str = "all") -> dict[str, Any]:
    """One backend: labelled set + every vendored run. A missing recording makes the part
    `pending` (named in `missing`), never a pass."""
    out: dict[str, Any] = {"backend": b, "missing": [], "incomparable": []}
    data = labelled.labels()
    rec = _recorder(b, live)
    if only in ("all", "labelled"):
        try:
            responses = labelled.witness_all(rec, data)
            out["labelled"] = labelled.score(data, responses)
            out["faults"] = {
                n: _caught(out["labelled"], labelled.score(data, labelled.transform(responses, f)))
                for n, f in faults.FAULTS.items()
            }
        except (LookupError, RuntimeError, subprocess.TimeoutExpired) as exc:
            # no recording (replay) or a failed live call: the part is missing, never a pass
            out["missing"].append(f"labelled: {exc}")
    calls = list(rec.calls)
    out["runs"] = {}
    if only in ("all", "runs"):
        for rd in fullrun.runs():
            # a run's own recorded answers seed the backend that produced them (same model)
            r = _recorder(b, live, fullrun.seeds(rd))
            try:
                out["runs"][rd.name] = fullrun.reverify(rd, r)
            except RendererIncomparable as exc:
                out["incomparable"].append(f"{rd.name}: {exc}")
            except (LookupError, RuntimeError, subprocess.TimeoutExpired) as exc:
                out["missing"].append(f"{rd.name}: {str(exc)[:300]}")
            calls += r.calls
    out["usage"] = _usage(calls)
    out["calls"] = calls
    return out


def _caught(base: dict, faulted: dict) -> bool:
    """The eval catches a fault injected into this backend's own answers when the faulted
    answers fail the gate and score worse than the backend did (fewer crops read correctly, or
    more false accepts). Failing a gate the backend already failed, unchanged, is not a catch."""
    worse = faulted["correct"] < base["correct"] or faulted["false_accepts"] > base["false_accepts"]
    return worse and not faulted["gate"]


def self_test() -> dict[str, Any]:
    """The eval's check on itself: the oracle (the labels as a perfect witness) passes the gate,
    and each injected fault fails it."""
    data = labelled.labels()
    o = labelled.oracle(data)
    base = labelled.score(data, o)
    faulted = {n: labelled.score(data, labelled.transform(o, f)) for n, f in faults.FAULTS.items()}
    return {
        "oracle_passes": base["gate"],
        "oracle": _brief(base),
        "faults": {n: {"fails_gate": not s["gate"], **_brief(s)} for n, s in faulted.items()},
    }


def _brief(s: dict) -> dict:
    return {
        k: s[k]
        for k in (
            "correct_value_rate",
            "positives_verified",
            "positives",
            "false_accepts",
            "false_accept_ids",
        )
    }


def isolation(live: bool) -> list[dict[str, Any]]:
    path = datasets() / "witness" / "isolation.json"
    if live:
        img = labelled.crop_bytes(labelled.labels()["crops"][0])
        res = [backends.isolation_probe(b, img) for b in backends.BACKENDS]
        path.write_text(json.dumps(res, indent=1) + "\n", "utf-8")
    if not path.exists():
        return []
    return json.loads(path.read_text("utf-8"))


def evaluate(*, live: bool = False, only_backends: list[str] | None = None) -> dict[str, Any]:
    chosen = [b for b in backends.BACKENDS if not only_backends or b["id"] in only_backends]
    with fullrun.render_cache():
        results = [run_backend(b, live=live) for b in chosen]
    pin = next((r for r in results if r["backend"].get("pinned")), None)
    rows = [_row(r, pin) for r in results]
    rows += [combined_row(a, b, pin) for a, b in pairs(results)]
    return {
        "comparator": comparator.versions(),
        "self_test": self_test(),
        "rows": rows,
        "isolation": isolation(live),
        "results": [_slim(r) for r in results],
    }


def _verdicts(r: dict) -> dict[str, bool]:
    out = {
        f"{run}:{k}": v for run, x in r.get("runs", {}).items() for k, v in x["verdicts"].items()
    }
    if "labelled" in r:
        out |= {f"probe:{k}": v for k, v in labelled.verdicts(r["labelled"]).items()}
    return out


def _row(r: dict, pin: dict | None) -> dict[str, Any]:
    b, lab = r["backend"], r.get("labelled")
    runs = r.get("runs", {})
    claims = sum(x["claims"] for x in runs.values())
    verified = sum(x["verified"] for x in runs.values())
    u = r["usage"]
    row = {
        "backend": b["id"],
        "transport": b["transport"],
        "family": b["family"],
        "stratum": backends.stratum(b),
        "requested_model": b["model"],
        "reported_model": ", ".join(u["reported_models"]) or None,
        "pinned": bool(b.get("pinned")),
        "correct_value_rate": lab["correct_value_rate"] if lab else None,
        "correct": f"{lab['correct']}/{lab['crops']}" if lab else None,
        "positives_verified": f"{lab['positives_verified']}/{lab['positives']}" if lab else None,
        "false_accepts": f"{lab['false_accepts']}/{lab['negatives']}" if lab else None,
        "false_accept_ids": lab["false_accept_ids"] if lab else None,
        "verified_rate_full_runs": round(verified / claims, 4) if claims else None,
        "verified_full_runs": f"{verified}/{claims}" if claims else None,
        "agreement_with_pin": (
            fullrun.agreement(_verdicts(r), _verdicts(pin)) if pin and pin is not r else None
        ),
        "latency_s_median_per_request": u["latency_s_median"],
        "latency_s_max": u["latency_s_max"],
        "requests": u["requests"],
        "tokens_in_out": [u["tokens_in"], u["tokens_out"]],
        "cost": _cost(b, u),
        "provenance_gaps": u["provenance_gaps"],
        "faults_caught": r.get("faults"),
        "missing": r["missing"],
        "incomparable": r.get("incomparable", []),
    }
    if "legacy_route" in b:
        row["legacy_route"] = b["legacy_route"]
    return row


def pairs(results: list[dict]) -> list[tuple[dict, dict]]:
    """spec A+B pairs for a Claude reader: every cross-family (A) backend with every
    reader-family (B) backend, where both have complete recordings."""
    done = [r for r in results if not r["missing"] and not r.get("incomparable")]
    a = [r for r in done if r["backend"]["family"] != backends.READER_FAMILY]
    b = [r for r in done if r["backend"]["family"] == backends.READER_FAMILY]
    return [(x, y) for x in a for y in b]


def combined_row(ra: dict, rb: dict, pin: dict | None) -> dict[str, Any]:
    """A+B on the same fixed inputs. The combination rule: a claim verifies only when **both**
    witnesses verify it (A gives W2, B the self-referential check); every claim on which they
    disagree is a recorded finding and stays unverified. A crop is read correctly only when both
    read it correctly. Cost adds up; latency is the slower witness (the two calls run in
    parallel)."""
    va, vb = _verdicts(ra), _verdicts(rb)
    both = {k: va[k] and vb[k] for k in sorted(set(va) & set(vb))}
    disagree = sorted(k for k in both if va[k] != vb[k])
    la, lb = ra["labelled"], rb["labelled"]
    read = sum(
        x["correct"] and y["correct"] for x, y in zip(la["per_crop"], lb["per_crop"], strict=True)
    )
    probes = {p["id"]: p["polarity"] for c in la["per_crop"] for p in c["probes"]}
    pos = [k for k, pol in probes.items() if pol == "positive"]
    neg = [k for k, pol in probes.items() if pol == "negative"]
    fa = [k for k in neg if both[f"probe:{k}"]]
    claims = [k for k in both if not k.startswith("probe:")]
    verified = sum(both[k] for k in claims)
    ua, ub = ra["usage"], rb["usage"]
    lat = paired_latency(ra["calls"], rb["calls"])
    row = {
        "backend": f"{ra['backend']['id']} + {rb['backend']['id']}",
        "transport": "pair",
        "family": f"{ra['backend']['family']} + {rb['backend']['family']}",
        "stratum": "A+B (both must verify; a disagreement is a finding)",
        "requested_model": None,
        "reported_model": None,
        "pinned": False,
        "correct_value_rate": round(read / la["crops"], 4),
        "correct": f"{read}/{la['crops']}",
        "positives_verified": f"{sum(both[f'probe:{k}'] for k in pos)}/{len(pos)}",
        "false_accepts": f"{len(fa)}/{len(neg)}",
        "false_accept_ids": fa,
        "verified_rate_full_runs": round(verified / len(claims), 4) if claims else None,
        "verified_full_runs": f"{verified}/{len(claims)}" if claims else None,
        "disagreements": {
            "full_runs": sum(not k.startswith("probe:") for k in disagree),
            "probes": sum(k.startswith("probe:") for k in disagree),
        },
        "agreement_with_pin": fullrun.agreement(both, _verdicts(pin)) if pin else None,
        "latency_s_median_per_request": lat["median"],
        "latency_s_max": lat["max"],
        "latency_pairs": {"timed": lat["timed"], "untimed": lat["untimed"]},
        "requests": ua["requests"] + ub["requests"],
        "tokens_in_out": [ua["tokens_in"] + ub["tokens_in"], ua["tokens_out"] + ub["tokens_out"]],
        "cost": f"{_cost(ra['backend'], ua)} + {_cost(rb['backend'], ub)}",
        "provenance_gaps": sorted({*ua["provenance_gaps"], *ub["provenance_gaps"]}),
        "faults_caught": None,
        "missing": [],
    }
    if "legacy_route" in ra["backend"] or "legacy_route" in rb["backend"]:
        row["legacy_route"] = "api-key (prohibited since spec)"
    return row


def paired_latency(a: list[dict], b: list[dict]) -> dict[str, Any]:
    """The A+B wall time per request: the two witnesses run in parallel on the same request
    (the request key is what both saw), so each pair costs its slower call. A pair with either
    call untimed (a product seed) is counted, not guessed."""
    ta = {c["request"]: c["latency_s"] for c in a}
    tb = {c["request"]: c["latency_s"] for c in b}
    both = sorted(set(ta) & set(tb))
    pairs = [max(ta[k], tb[k]) for k in both if ta[k] is not None and tb[k] is not None]
    return {
        "median": round(statistics.median(pairs), 1) if pairs else None,
        "max": max(pairs) if pairs else None,
        "timed": len(pairs),
        "untimed": len(both) - len(pairs),
    }


def _cost(b: dict, u: dict) -> str:
    """What the backend reports, never a guess: list USD where the CLI states it, tokens where
    the API or CLI counts them, the quota otherwise."""
    if b["transport"] == "claude":
        return f"USD {u['cost_usd']:.2f} (claude -p list cost)"
    if b["transport"] == "agy":
        return "flat-rate quota (no usage reported)"
    if b["transport"] == "codex":
        return f"flat-rate quota; {u['tokens_in']}/{u['tokens_out']} tokens"
    seed = u["by_source"].get("product seed", {})
    of_which = (
        f", of which {seed['tokens_in']}/{seed['tokens_out']} spent by the product runs"
        if seed
        else ""
    )
    return f"{u['tokens_in']}/{u['tokens_out']} API tokens{of_which} (price table pending)"


def _slim(r: dict) -> dict[str, Any]:
    out = {
        "backend": r["backend"]["id"],
        "missing": r["missing"],
        "incomparable": r.get("incomparable", []),
    }
    if "labelled" in r:
        out["labelled"] = [
            {k: c[k] for k in ("id", "correct", "misses", "probes")}
            for c in r["labelled"]["per_crop"]
        ]
    out["runs"] = {k: v["documents"] for k, v in r.get("runs", {}).items()}
    return out


def table(report: dict[str, Any]) -> str:
    """The PR table (markdown)."""
    runs = len(fullrun.runs())
    approved_rows = []
    legacy_rows = []
    for r in report["rows"]:
        if r.get("legacy_route"):
            legacy_rows.append(r)
        else:
            approved_rows.append(r)

    def format_rows(rows_list):
        lines = []
        head = (
            "| backend | family · spec stratum | reported model | "
            "read correctly (labelled crops) | false accepts (negatives) | "
            f"positives bound | faults caught | verified ({runs} recorded B runs) | "
            "agreement w/ pin | latency / request (median, max) | cost |"
        )
        lines.append(head)
        lines.append("|" + "---|" * 11)

        def cell(v):
            return "—" if v is None else str(v)

        for r in rows_list:
            lat = (
                f"{r['latency_s_median_per_request']} s, {r['latency_s_max']} s"
                if r["latency_s_median_per_request"] is not None
                else "replayed"
            )
            caught = r["faults_caught"]
            faults = f"{sum(caught.values())}/{len(caught)}" if caught else "—"
            pin = " (pin)" if r["pinned"] else ""

            r_backend = f"`{r['backend']}`{pin}"
            r_fam_strat = f"{r['family']} · {r['stratum']}"
            r_rep = cell(r["reported_model"])
            r_corr = cell(r["correct"])
            r_fa = cell(r["false_accepts"])
            r_pos = cell(r["positives_verified"])
            r_full = cell(r["verified_full_runs"])
            r_agree = cell(r["agreement_with_pin"])

            lines.append(
                f"| {r_backend} | {r_fam_strat} | {r_rep} | {r_corr} | {r_fa} "
                f"| {r_pos} | {faults} | {r_full} | {r_agree} | {lat} | {r['cost']} |"
            )
        return "\n".join(lines)

    md = "### Approved Backends\n\n" + format_rows(approved_rows)
    if legacy_rows:
        legacy_title = "\n\n### Historical Legacy Backends (Prohibited since spec)\n\n"
        md += legacy_title + format_rows(legacy_rows)
    return md
