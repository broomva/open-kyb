"""Foundation slice: contract, audit chain, workspace init, guards."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from kyb import ontology
from kyb.audit.chain import (
    GENESIS,
    AuditCorruption,
    AuditLog,
    event_hash,
    verify_workspace,
)
from kyb.contracts.kyb_decision_schema import AuditEvent
from kyb.util.hashing import sha256_file
from kyb.workspace import guard
from kyb.workspace.init import init_workspace

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "examples"


def test_contract_models_validate():
    from kyb.contracts.kyb_decision_schema import AuditEvent, Citation, KYBDecision, Violation

    assert all(m is not None for m in (AuditEvent, Citation, KYBDecision, Violation))


@pytest.mark.parametrize("name", ["compliance_rules.md", "watchlists.json"])
def test_shipped_sample_files_are_kybs_byte_for_byte(name):
    # init_workspace() without a policy uses these copies; they must stay open-kyb's files
    assert sha256_file(REPO / "src/kyb/workspace/defaults" / name) == sha256_file(
        ASSETS / "policy" / name
    )


def _clock(start=datetime(2026, 10, 26, 15, 0, tzinfo=UTC)):
    t = start
    while True:
        yield t
        t += timedelta(seconds=1)


def _log(path: Path) -> AuditLog:
    ticks = _clock()
    return AuditLog(path, clock=lambda: next(ticks))


FIXED = {
    "seq": 0, "timestamp": "2026-10-25T15:00:00.000000Z", "component": "kyb",
    "action": "workspace.init", "input_hash": "sha256:a", "output_hash": "sha256:b",
    "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0, "detail": {}, "prev_hash": GENESIS,
}  # fmt: skip


def test_audit_hash_is_pinned_to_a_literal():
    # Changing the canonical serialisation changes every hash; this makes that a deliberate act.
    assert event_hash(FIXED) == (
        "sha256:4d011ecab5f660b8cda580a86f38d072272157ffa890a81802996ec424e07c8a"
    )


def test_audit_hash_covers_every_field_exactly():
    base = event_hash(FIXED)
    assert base != event_hash({**FIXED, "detail": {"x": 1}})
    assert base != event_hash({**FIXED, "tokens_in": 1})
    assert event_hash({**FIXED, "cost_usd": 0.123456}) != event_hash(
        {**FIXED, "cost_usd": 0.1234561}
    )


def test_audit_chain_appends_and_verifies(tmp_path):
    log = _log(tmp_path / "audit.jsonl")
    e0 = log.append("kyb", "workspace.init", input_hash="sha256:a", output_hash="sha256:b")
    e1 = log.append("ingest", "tool.call", input_hash="sha256:c", output_hash="sha256:d",
                    detail={"files": 5}, tokens_in=10, cost_usd=0.0123)  # fmt: skip
    assert e0.seq == 0 and e0.prev_hash == GENESIS
    assert e1.seq == 1 and e1.prev_hash == e0.hash
    r = log.verify()
    assert r.ok and r.events == 2 and r.head == e1.hash


def test_contract_model_round_trip_still_verifies(tmp_path):
    # A whole-second timestamp serialises without microseconds through the contract model.
    ev = _log(tmp_path / "a.jsonl").append("c", "a", input_hash="sha256:i", output_hash="sha256:o")
    dumped = json.loads(AuditEvent.model_validate(ev.model_dump()).model_dump_json())
    assert event_hash(dumped) == dumped["hash"]


@pytest.mark.parametrize("tamper", ["edit", "delete", "reorder", "cost"])
def test_audit_chain_detects_tampering(tmp_path, tamper):
    path = tmp_path / "audit.jsonl"
    log = _log(path)
    for i in range(3):
        log.append("c", "a", input_hash=f"sha256:{i}", output_hash="sha256:o", detail={"i": i},
                   cost_usd=0.123456)  # fmt: skip
    lines = path.read_text().splitlines()
    if tamper == "edit":
        ev = json.loads(lines[1])
        ev["detail"] = {"i": 99}
        lines[1] = json.dumps(ev)
    elif tamper == "cost":
        ev = json.loads(lines[1])
        ev["cost_usd"] = 0.1234561
        lines[1] = json.dumps(ev)
    elif tamper == "delete":
        del lines[1]
    else:
        lines[0], lines[1] = lines[1], lines[0]
    path.write_text("\n".join(lines) + "\n")
    assert not AuditLog(path).verify().ok


def test_unterminated_last_record_is_framed_and_garbage_is_refused(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = _log(path)
    log.append("c", "a", input_hash="sha256:i", output_hash="sha256:o")
    path.write_bytes(path.read_bytes().rstrip(b"\n"))  # valid record, missing newline
    log.append("c", "b", input_hash="sha256:i", output_hash="sha256:o")
    assert log.verify().ok and log.verify().events == 2
    with path.open("a") as f:
        f.write('{"seq": 2, "broken')
    with pytest.raises(AuditCorruption):
        log.append("c", "c", input_hash="sha256:i", output_hash="sha256:o")
    r = log.verify()  # returns a structured failure instead of raising
    assert not r.ok and "unreadable" in r.error


def test_init_materialises_the_template(tmp_path):
    root = tmp_path / "ws"
    lock = init_workspace(root, tenant="kyb-demo", policy_src=ASSETS / "policy/compliance_rules.md")
    for rel in ("CLAUDE.md", ".mcp.json", ".claude/settings.json", "template.lock", "catalog.json",
                "ontology/kyb.ontology.toml", "policy/compliance_rules.md"):  # fmt: skip
        assert (root / rel).exists(), rel
    onto = ontology.workspace_ontology(root)
    assert len(onto.actions) == 23 and len(onto.reads) == 12
    # spec: the agent reads and answers predicates; the witness is the only system-side model call
    assert onto.actions["read_document"].invoker == ["agent"]
    assert onto.actions["propose_predicates"].invoker == ["agent"]
    assert [a for a, d in onto.actions.items() if d.model] == ["layout_pages", "verify_quotes"]
    mcp = sorted([k for k, v in onto.actions.items() if "mcp" in v.transport]
                 + [k for k, v in onto.reads.items() if "mcp" in v.transport])  # fmt: skip
    assert mcp == ["apply_policy", "ask", "audit_verify", "explain_violation", "get_decision",
                   "get_facts", "ingest_packet", "inspect_loop", "layout_pages", "list_entities",
                   "pages", "plan_loop", "predicate_work_order", "propose", "propose_policy",
                   "propose_predicates", "read_document", "read_plan", "record_answer",
                   "record_override", "render_crop",
                   "review_queue", "review_status", "search_documents", "sign_decision",
                   "start_loop_repair", "verify_quotes"]  # fmt: skip
    hooks = json.dumps(json.loads((root / ".claude/settings.json").read_text())["hooks"])
    # works without an activated venv, through the hooks' fast lane (spec)
    assert f"{sys.executable} -m kyb.hook pre --source workspace" in hooks
    assert lock["tenant"] == "kyb-demo" and lock["ontology_path"] == "ontology/kyb.ontology.toml"
    assert lock["ontology"] == sha256_file(root / "ontology/kyb.ontology.toml")
    assert verify_workspace(root).ok
    with pytest.raises(FileExistsError):
        init_workspace(root, tenant="kyb-demo", policy_src=ASSETS / "policy/compliance_rules.md")


def test_suffix_deletion_and_erased_history_are_detected(tmp_path):
    root = tmp_path / "ws"
    init_workspace(root, tenant="t", policy_src=ASSETS / "policy/compliance_rules.md")
    log = AuditLog(root / "ledger/audit.jsonl")
    log.append("c", "a", input_hash="sha256:i", output_hash="sha256:o")
    from kyb.audit.chain import write_checkpoint

    write_checkpoint(root)
    lines = log.path.read_text().splitlines()
    log.path.write_text("\n".join(lines[:-1]) + "\n")  # drop the newest event
    r = verify_workspace(root)
    assert not r.ok and "checkpoint" in r.error
    log.path.write_text("")  # erase everything
    assert not verify_workspace(root).ok


@pytest.fixture
def ws(tmp_path) -> Path:
    root = tmp_path / "ws"
    init_workspace(root, tenant="t", policy_src=ASSETS / "policy/compliance_rules.md")
    return root


def test_guard_allows_only_agent_writable(ws):
    assert guard.check_write(ws, "cases/B/notes/investigation-001.md") is None
    assert guard.check_write(ws, "cases/B/proposals/p-001.json") is None
    assert guard.check_write(ws, "gold/facts/B/r1.json")
    assert guard.check_write(ws, "cases/B/../../ledger/audit.jsonl")
    assert guard.check_write(ws, "/etc/passwd")


def test_guard_resolves_relative_paths_against_the_tool_cwd(ws):
    (ws / "cases/A").mkdir(parents=True)
    assert guard.check_write(ws, "notes/n.md", cwd=ws / "cases/A") is None
    assert guard.check_write(ws, "cases/A/notes/x.json", cwd=ws / "gold")


def test_guard_bash(ws):
    assert guard.check_bash("pdftoppm -r 600 05.pdf cases/B/scratch/crop", ws, ws) is None
    assert guard.check_bash("pdftotext in.pdf ledger/audit.jsonl", ws, ws)
    assert guard.check_bash("pdftoppm -png in.pdf gold/x", ws, ws)
    assert guard.check_bash("echo x > .claude/settings.json", ws, ws)
    assert guard.check_bash("echo x > ledger/audit.jsonl", ws, ws)
    assert guard.check_bash("rm -rf gold", ws, ws)
    assert guard.check_bash("curl https://example.com", ws, ws)


def test_pre_tool_use_exit_codes(ws):
    deny = guard.pre_tool_use({"cwd": str(ws), "tool_name": "Write",
                               "tool_input": {"file_path": "gold/x.json"}})  # fmt: skip
    allow = guard.pre_tool_use({"cwd": str(ws), "tool_name": "Write",
                                "tool_input": {"file_path": "cases/A/notes/n.md"}})  # fmt: skip
    assert deny[0] == 2 and allow == (0, "")


def test_template_registry_validates():
    onto = ontology.template()
    fields = {"actions", "reads", "agent_writable"}
    template_registry = onto.model_dump(include=fields, by_alias=True)
    validated = ontology.Registry.model_validate(template_registry)
    assert onto.actions.keys() == validated.actions.keys()
