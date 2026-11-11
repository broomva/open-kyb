"""`make dogfood`: a live agent runs the kyb review the way an officer would (spec).

A fresh project holds only the sample assets. The harness gets the kyb plugin (Claude Code) or
the kyb skills and MCP server (Codex), pointed at this checkout, and one prompt: review the packet
with the kyb-review skill. Nothing else drives it. Afterwards the workspace is read back and the
run is gated:

- every document was read and submitted, and verification ran;
- the audit chain verifies;
- Claude Code: every submission is `isolated` (hook-attested) and every document reaches W2;
- Codex: managed scoped sessions with bound served-model/runtime receipts are required.

The run directory (project, workspace, transcript) stays under ~/.cache/kyb/dogfood/ (or
$KYB_DOGFOOD_DIR), outside any checkout, so workspace discovery cannot climb into another
workspace; it holds applicant data and is never committed. A summary is written to
artifacts/dogfood/<run>.json, the committed record lanes cite in their PRs. The verification
rate is reported, not gated, until the comparator hardening lands (docs/analysis/b3-dogfood.md).

    uv run python scripts/dogfood.py --harness claude --packet es-iberia-software
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from kyb.witness.strata import DEFAULT_A

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "examples"
PROMPT = ("Run a KYB review on the packet at examples/packets/{packet}, "
          "with as_of {as_of}. Use the kyb-review skill.")  # fmt: skip
CLAUDE_TOOLS = ["mcp__plugin_kyb_kyb__*", "Agent", "Read", "Glob", "Skill"]
# kyb's own MCP tools, pre-approved for a Codex run (codex exec cannot answer an approval prompt)
KYB_TOOLS = [
    "review_status",
    "codex_runtime",
    "codex_read_document",
    "codex_predicates",
    "init_workspace",
    "ingest_packet",
    "pages",
    "layout_pages",
    "read_plan",
    "render_crop",
    "read_document",
    "verify_quotes",
    "ask",
    "record_answer",
    "review_queue",
    "run_review",
    "predicate_work_order",
    "propose_predicates",
    "decide",
    "audit_verify",
]


RUNS = Path(os.environ.get("KYB_DOGFOOD_DIR") or Path.home() / ".cache" / "kyb" / "dogfood")


def _env(project: Path) -> dict[str, str]:
    """The harness environment: this checkout as the kyb source, no inherited workspace or
    project pin, and workspace discovery bounded to the run's own project (KYB_DISCOVERY_ROOT),
    so neither the agent nor a hook can act on an enclosing workspace."""
    env = {**os.environ, "KYB_DEV_REPO": str(REPO), "KYB_DISCOVERY_ROOT": str(project.resolve())}
    for var in ("KYB_WORKSPACE", "CLAUDE_PROJECT_DIR"):
        env.pop(var, None)
    return env


def _project(run: Path) -> Path:
    project = run / "project"
    project.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ASSETS, project / "examples")
    (project / "examples/packets/expected_labels.json").unlink(missing_ok=True)
    return project


def _claude(project: Path, run: Path, prompt: str, max_turns: int) -> dict:
    cmd = ["claude", "-p", prompt, "--plugin-dir", str(REPO / "plugin"),
           "--allowedTools", *CLAUDE_TOOLS, "--output-format", "json",
           "--max-turns", str(max_turns)]  # fmt: skip
    proc = subprocess.run(cmd, cwd=project, env=_env(project), capture_output=True, text=True)
    (run / "stderr.log").write_text(proc.stderr)
    (run / "result.json").write_text(proc.stdout)
    try:
        out = json.loads(proc.stdout)
    except ValueError:
        return {"exit": proc.returncode, "error": proc.stdout[-2000:] or proc.stderr[-2000:]}
    return {"exit": proc.returncode, "cost_usd": out.get("total_cost_usd"),
            "turns": out.get("num_turns"), "subtype": out.get("subtype"),
            "report": out.get("result"), "session": out.get("session_id")}  # fmt: skip


# Codex starts MCP servers with a filtered environment: forward by name only, never by value
CODEX_FORWARDED_ENV = ["TYPESAFE_API_KEY", "KYB_DISCOVERY_ROOT",
                       "KYB_WITNESS_STRATUM", "KYB_WITNESS_A_BACKENDS"]  # fmt: skip


def _codex_cmd(run: Path, prompt: str) -> list[str]:
    """The `codex exec` command for a run: kyb's MCP server for this run only (-c), its tools
    pre-approved, and the variables it needs forwarded through Codex's environment filter
    (the witness keys, and KYB_DISCOVERY_ROOT so discovery stays inside the run's project)."""
    args = json.dumps(["run", "--quiet", "--project", str(REPO), "kyb-mcp"])
    cmd = ["codex", "exec", "--json", "--skip-git-repo-check", "-s", "workspace-write",
           "-c", 'mcp_servers.kyb.command="uv"', "-c", f"mcp_servers.kyb.args={args}",
           "-c", "mcp_servers.kyb.tool_timeout_sec=900",
           "-c", f"mcp_servers.kyb.env_vars={json.dumps(CODEX_FORWARDED_ENV)}",
           "-c", 'mcp_servers.kyb.default_tools_approval_mode="approve"',
           *[x for t in KYB_TOOLS
             for x in ("-c", f'mcp_servers.kyb.tools.{t}.approval_mode="approve"')],
           "-o", str(run / "last-message.md"), prompt]  # fmt: skip
    return cmd


def _codex(project: Path, run: Path, prompt: str) -> dict:
    """Codex with the skills in the project's .agents/skills and the MCP server set for this run
    only (-c), so the user's Codex config is left alone."""
    skills = project / ".agents" / "skills"
    for skill in (REPO / "plugin" / "skills").iterdir():
        shutil.copytree(skill, skills / skill.name)
    shutil.copy(REPO / "plugin" / "agents" / "kyb-reader.md", skills / "kyb-review" / "reader.md")
    shutil.copy(
        REPO / "plugin" / "agents" / "kyb-predicates.md", skills / "kyb-review" / "predicates.md"
    )
    cmd = _codex_cmd(run, prompt)
    proc = subprocess.run(cmd, cwd=project, env=_env(project), capture_output=True, text=True)
    (run / "stderr.log").write_text(proc.stderr)
    (run / "events.jsonl").write_text(proc.stdout)
    usage: dict = {}
    for line in proc.stdout.splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if isinstance(ev, dict) and isinstance(ev.get("usage"), dict):
            usage = ev["usage"]
    last = run / "last-message.md"
    return {"exit": proc.returncode, "usage": usage,
            "report": last.read_text() if last.exists() else None}  # fmt: skip


def _inspect(project: Path, packet: str) -> dict:
    """Read the workspace back: what was submitted, attested and verified."""
    sys.path.insert(0, str(REPO / "src"))
    from kyb.audit.chain import verify_workspace
    from kyb.read.status import review_status
    from kyb.witness.verify import assurance_now
    from kyb.workspace.guard import discover

    root, why = discover(project)
    if root is None:
        return {"workspace": None, "why": why}
    if not root.resolve().is_relative_to(project.resolve()):
        return {"workspace": None, "why": f"resolved a workspace outside the run: {root}"}
    status = review_status(root, packet)
    run = status.get("run")
    docs: dict = {}
    for claims in sorted((root / "silver" / "claims" / packet / str(run)).glob("*.json")):
        sub = json.loads(claims.read_text())
        w = root / "silver" / "witness" / packet / str(run) / claims.name
        wit = json.loads(w.read_text()) if w.exists() else None
        now = assurance_now(root, packet, str(run), claims.stem) if wit else {}
        docs[claims.stem] = {
            "claims": len(sub["claims"]),
            "context": sub["produced_by"]["reader"]["context"],
            "verified": sum(r["verified"] for r in wit["results"]) if wit else None,
            "assurance": _assurance(root, packet, str(run), claims.stem) if wit else None,
            "reader_model": now.get("reader_model"),
            "context_now": now.get("context_now"),
            "witness_model": wit.get("model") if wit else None,
            "reported_model": (wit.get("provenance") or {}).get("reported_model") if wit else None,
            "stratum": wit.get("stratum") if wit else None,
            "self_referential": wit.get("self_referential") if wit else None,
            "routing_refused": wit.get("routing_refused", False) if wit else None,
            "self_referential_claims": all(r.get("self_referential") is True
                for r in wit["results"] if r["verified"]) if wit else False,
            "strata": {k: {"model": r.get("model"), "assurance": r.get("assurance"),
                "self_referential": r.get("self_referential"),
                    "self_referential_claims": all(x.get("self_referential") is True
                        for x in r["results"] if x["verified"]),
                "reported_model": (r.get("provenance") or {}).get("reported_model"),
                "routing_refused": r.get("routing_refused", False)}
                for k, r in (wit.get("strata") or {}).items()} if wit else {},
            "uncovered": sum(x["status"] == "uncovered" for x in wit["coverage"]) if wit else None,
            "conflicts": sum(x["status"] == "conflict" for x in wit["coverage"]) if wit else None,
            "findings": [f["kind"] for f in wit["findings"]] if wit else None,
        }  # fmt: skip
    decision = None
    if status.get("next") == "decided":
        from kyb.contracts.b4 import DecisionRecord
        from kyb.decide.action import fixture_inputs
        from kyb.run.records import read_record

        record = DecisionRecord.model_validate(read_record(root, status["decision"]))
        if (record.case, record.run) != (packet, run):
            raise ValueError("emitted decision belongs to another case/run")
        decision = {
            "record": status["decision"],
            "valid": True,
            "decision_id": record.decision_id,
            "status": record.kyb_decision.status,
            "risk_score": record.kyb_decision.risk_score,
            "signature_state": record.signature_state,
            "fixture_inputs": fixture_inputs(root, packet, run),
        }
    chain = verify_workspace(root, at_boundary=True)
    return {"workspace": str(root.relative_to(project)), "next": status.get("next"), "run": run,
            "stratum": os.environ.get("KYB_WITNESS_STRATUM", "B"),
            "documents": docs, "decision": decision,
            "audit_ok": chain.ok, "audit_events": chain.events}  # fmt: skip


def _assurance(root: Path, case: str, run: str, doc: str) -> str:
    """Assurance as of now (re-derived from the chain), not the witness record's snapshot."""
    try:
        from kyb.witness.verify import assurance_now
    except ImportError:  # before PR #13's assurance_now
        wit = json.loads((root / "silver" / "witness" / case / run / f"{doc}.json").read_text())
        return wit["assurance"]
    return assurance_now(root, case, run, doc)["assurance"]


def _agent_failures(harness: str, agent: dict) -> list[str]:
    """The harness run itself must have completed: a run cut short (max turns, an error, a
    crash) fails the gate even when the workspace looks finished."""
    fails = []
    if agent.get("error"):
        fails.append(f"the harness failed: {str(agent['error'])[:200]}")
    if agent.get("exit") != 0:
        fails.append(f"the harness exited {agent.get('exit')}")
    if harness == "claude" and agent.get("subtype") != "success":
        fails.append(f"the agent did not finish (result subtype {agent.get('subtype')!r})")
    return fails


def _gate(harness: str, expected_docs: int, ws: dict, agent: dict) -> list[str]:
    fails = _agent_failures(harness, agent)
    docs = ws.get("documents") or {}
    if ws.get("workspace") is None:
        return [*fails, f"no workspace was created ({ws.get('why')})"]
    if len(docs) != expected_docs:
        fails.append(f"{len(docs)}/{expected_docs} documents submitted")
    if any(d["verified"] is None for d in docs.values()) or ws.get("next") not in (
        "decide",
        "decided",
    ):
        fails.append(f"verification did not complete (next: {ws.get('next')})")
    if ws.get("next") == "decided":
        decision = ws.get("decision") or {}
        if (
            decision.get("valid") is not True
            or decision.get("signature_state") != "pending_signature"
        ):
            fails.append("decided run has no valid pending-signature DecisionRecord")
        if decision.get("fixture_inputs") != []:
            fails.append("decided run has fixture-produced inputs or missing producer evidence")
    if not ws.get("audit_ok"):
        fails.append("the audit chain does not verify")
    want = "isolated"
    wrong = [k for k, d in docs.items() if d["context"] != want]
    if wrong:
        fails.append(f"submissions not {want}: {wrong}")
    if harness in {"claude", "codex"}:
        # Historical test/receipt inputs omit stratum and retain the original W2 gate.
        # Real inspections always declare the requested stratum; B has its own strict W1 gate.
        stratum = ws.get("stratum", "A")
        wanted = "W1" if stratum == "B" else "W2"
        wrong = [k for k, d in docs.items() if d["assurance"] != wanted]
        if wrong:
            fails.append(f"documents not {wanted}: {wrong}")
    if "stratum" in ws:
        bad = [k for k, d in docs.items() if not _stratum_evidence(ws["stratum"], d)]
        if bad:
            fails.append(f"documents lack required stratum/model evidence: {bad}")
    return fails


def _stratum_evidence(stratum: str, doc: dict) -> bool:
    from kyb.read.harness import model_family

    reader, actual = doc.get("reader_model"), doc.get("witness_model")
    if (
        stratum not in {"A", "B", "A+B"}
        or doc.get("routing_refused")
        or doc.get("stratum") != stratum
        or doc.get("context") != "isolated"
        or doc.get("context_now") != "isolated"
        or model_family(reader) == "unknown"
        or model_family(actual) == "unknown"
        or actual != doc.get("reported_model")
    ):
        return False
    if stratum == "B":
        return (
            doc.get("assurance") == "W1"
            and actual == reader
            and doc.get("self_referential") is True
            and doc.get("self_referential_claims") is True
        )
    if doc.get("assurance") != "W2" or model_family(actual) == model_family(reader):
        return False
    if stratum == "A+B":
        a, b = doc.get("strata", {}).get("A", {}), doc.get("strata", {}).get("B", {})
        return (
            a.get("model") == a.get("reported_model") == actual
            and a.get("assurance") == "W2"
            and not a.get("routing_refused")
            and b.get("model") == b.get("reported_model") == reader
            and b.get("assurance") == "W1"
            and b.get("self_referential") is True
            and b.get("self_referential_claims") is True
            and not b.get("routing_refused")
            and "witness_strata_disagreement" not in (doc.get("findings") or [])
        )
    return True


def _git_head() -> str:
    """The short head, marked `+wip` when code differs from it: a gate receipt (spec) only accepts
    a summary that ran exactly at the reviewed head."""

    def git(*a: str) -> str:
        return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()

    dirty = git(
        "status", "--porcelain", "--", "src", "plugin", "scripts", "pyproject.toml", "uv.lock"
    )
    return git("rev-parse", "--short", "HEAD") + ("+wip" if dirty else "")


def _precheck(env: Mapping[str, str] | None = None) -> str | None:
    """Validate prerequisites for the configured witness stratum before running."""
    e = os.environ if env is None else env
    stratum = e.get("KYB_WITNESS_STRATUM", "B")
    if stratum in {"A", "A+B"}:
        selected = e.get("KYB_WITNESS_A_BACKENDS")
        a_backends = tuple(selected.split(",")) if selected else DEFAULT_A
        if any(b.startswith("vertex-gcp:") for b in a_backends) and not shutil.which("gcloud"):
            return (
                "gcloud not found: the stratum-A witness uses the approved Vertex route "
                "(application-default credentials)"
            )
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--harness", choices=["claude", "codex"], default="claude")
    ap.add_argument("--packet", default="es-iberia-software")
    ap.add_argument("--packet-dir", type=Path, default=None,
                    help="a packet folder outside the assets (e.g. a synthetic one); only its "
                         "PDFs are copied in, under its folder name")  # fmt: skip
    ap.add_argument("--as-of", default="2026-10-25")
    ap.add_argument("--max-turns", type=int, default=80)
    a = ap.parse_args(argv)
    if a.packet_dir is not None:
        if not a.packet_dir.is_dir():
            ap.error(f"no packet folder {a.packet_dir}")
        a.packet = a.packet_dir.name
    packet_src = a.packet_dir or ASSETS / "packets" / a.packet
    if not packet_src.is_dir():
        ap.error(f"no packet {a.packet}")
    if err := _precheck():
        ap.error(err)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    name = f"{stamp}-{a.harness}-{a.packet}"
    run = RUNS / name
    if run.exists():  # two runs started in the same second: never share a directory
        name = f"{name}-{os.getpid()}"
        run = RUNS / name
    project = _project(run)
    if a.packet_dir is not None:  # PDFs only: a manifest next to them could carry the label
        dest = project / "examples" / "packets" / a.packet
        dest.mkdir(parents=True)
        for pdf in sorted(packet_src.glob("*.pdf")):
            shutil.copy2(pdf, dest / pdf.name)
    prompt = PROMPT.format(packet=a.packet, as_of=a.as_of)
    t0 = time.monotonic()
    try:
        agent = (_claude(project, run, prompt, a.max_turns) if a.harness == "claude"
                 else _codex(project, run, prompt))  # fmt: skip
    except Exception as exc:  # a launch failure is a failed receipt, never a crash or a pass
        agent = {"exit": None, "error": f"{type(exc).__name__}: {exc}"}
    seconds = round(time.monotonic() - t0, 1)
    try:
        ws = _inspect(project, a.packet)
    except Exception as exc:
        ws = {"workspace": None, "why": f"inspection failed: {type(exc).__name__}: {exc}"}
    expected = len([p for p in packet_src.iterdir() if p.suffix == ".pdf"])
    fails = _gate(a.harness, expected, ws, agent)
    summary = {"run": name, "harness": a.harness, "packet": a.packet, "as_of": a.as_of,
               "stratum": os.environ.get("KYB_WITNESS_STRATUM", "B"),
               "seconds": seconds, "git_head": _git_head(),
               "agent": agent, "workspace": ws, "gate": "pass" if not fails else "fail",
               "failures": fails, "run_dir": str(run)}  # fmt: skip
    out = REPO / "artifacts" / "dogfood" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    docs = ws.get("documents") or {}
    verified = sum(d["verified"] or 0 for d in docs.values())
    claims = sum(d["claims"] for d in docs.values())
    print(f"{name}: gate {summary['gate']} · {len(docs)} docs · {verified}/{claims} verified · "
          f"{seconds}s · cost {agent.get('cost_usd')}")  # fmt: skip
    if decision := ws.get("decision"):
        print(
            f"  decision: {decision['status']} · score {decision['risk_score']} · "
            f"{decision['signature_state']}"
        )
    for f in fails:
        print(f"  FAIL {f}")
    print(f"  summary: {out.relative_to(REPO)}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
