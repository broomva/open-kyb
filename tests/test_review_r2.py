"""Regressions for the PR #3 round-2 review (N1–N9)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from kyb.audit.chain import AuditCorruption, AuditLog, verify_workspace, workspace_action
from kyb.cli import main
from kyb.docintel.ingest import ingest_packet
from kyb.docintel.pages import build_pages
from kyb.workspace.catalog import Catalog
from kyb.workspace.guard import append_hook, check_bash
from kyb.workspace.init import init_workspace
from kyb.workspace.runs import start_run

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "examples"
PACKET_A = ASSETS / "packets" / "cl-andes-agro"
CASE = "cl-andes-agro"
POLICY = ASSETS / "policy/compliance_rules.md"
needs_poppler = pytest.mark.skipif(shutil.which("pdftotext") is None, reason="poppler missing")


@pytest.fixture
def ws(tmp_path) -> Path:
    root = tmp_path / "ws"
    init_workspace(root, tenant="t", policy_src=POLICY)
    return root


def _log(root: Path) -> Path:
    return root / "ledger" / "audit.jsonl"


def _hook(root: Path) -> None:
    code, msg = append_hook({"cwd": str(root), "tool_name": "Read", "tool_input": {}})
    assert code == 0, msg


def test_n1_new_actions_cannot_conceal_a_broken_older_checkpoint(ws):
    for _ in range(4):
        _hook(ws)  # each hook event is an action: appended, then pinned
    assert verify_workspace(ws, at_boundary=True).ok
    lines = _log(ws).read_text().splitlines()
    _log(ws).write_text("\n".join(lines[:2]) + "\n")  # truncate, keep every checkpoint
    assert not verify_workspace(ws).ok
    with pytest.raises(AuditCorruption), workspace_action(ws):  # refuses, never re-anchors
        pass
    assert append_hook({"cwd": str(ws), "tool_name": "Read"})[0] == 2
    assert not verify_workspace(ws).ok


def test_n1_an_unpinned_tail_or_missing_checkpoints_fail_at_a_boundary(ws):
    _hook(ws)
    AuditLog(_log(ws)).append("x", "y", input_hash="sha256:i", output_hash="sha256:o")
    r = verify_workspace(ws, at_boundary=True)
    assert not r.ok and "not checkpointed" in r.error
    assert verify_workspace(ws).ok  # mid-action, an unpinned tail is legitimate
    for cp in (ws / "ledger" / "checkpoints").glob("cp-*.json"):
        cp.unlink()
    assert not verify_workspace(ws, at_boundary=True).ok


@needs_poppler
def test_n2_pages_refuses_a_manifest_that_disagrees_with_the_catalog(ws):
    ingest_packet(ws, PACKET_A, as_of="2026-10-25")
    run = start_run(ws, CASE, as_of="2026-10-25", reason="test")
    mpath = ws / f"bronze/packets/{CASE}/manifest.json"
    m = json.loads(mpath.read_text())
    m["files"], m["as_of"] = m["files"][:1], "2099-01-01"
    mpath.chmod(0o644)
    mpath.write_text(json.dumps(m))
    with pytest.raises(PermissionError, match="manifest"):
        build_pages(ws, CASE, run)
    assert not (ws / f"silver/pages/{CASE}").exists()


@needs_poppler
def test_n2_pages_refuses_a_run_whose_as_of_differs_from_the_packet(ws):
    ingest_packet(ws, PACKET_A, as_of="2026-10-25")
    run = start_run(ws, CASE, as_of="2026-01-01", reason="test")
    with pytest.raises(PermissionError, match="as_of"):
        build_pages(ws, CASE, run)


def test_n3_checkpoint_published_after_the_chain_read_is_not_false_corruption(ws, monkeypatch):
    import kyb.audit.chain as chain

    real = chain._checkpoints

    def racing(root):
        cps = real(root)  # the verifier's checkpoint snapshot is taken first...
        monkeypatch.setattr(chain, "_checkpoints", real)  # (one interleaving only)
        with chain.action_lock(root):  # ...then a writer appends and pins before the chain read
            AuditLog(_log(root)).append("x", "y", input_hash="sha256:i", output_hash="sha256:o")
            chain.write_checkpoint(root)
        return cps

    monkeypatch.setattr(chain, "_checkpoints", racing)
    assert chain.verify_workspace(ws).ok


@needs_poppler
def test_n4_every_accepted_filename_can_be_paged(ws, tmp_path):
    long_ok, too_long = tmp_path / "ok", tmp_path / "bad"
    long_ok.mkdir(), too_long.mkdir()
    src = next(PACKET_A.glob("01_*.pdf"))
    shutil.copy(src, long_ok / ("a" * 81 + ".pdf"))
    shutil.copy(src, too_long / ("a" * 82 + ".pdf"))
    with pytest.raises(ValueError):
        ingest_packet(ws, too_long, as_of="2026-10-25", case="bad")
    ingest_packet(ws, long_ok, as_of="2026-10-25", case="ok")
    run = start_run(ws, "ok", as_of="2026-10-25", reason="test")
    assert len(build_pages(ws, "ok", run)) == 1


@pytest.mark.parametrize("line", [
    b"\xff\xfe not utf-8\n",
    None,  # numeric timestamp, filled below
    None,  # timestamp that overflows on UTC conversion
])  # fmt: skip
def test_n5_damaged_or_boundary_input_is_a_structured_failure(tmp_path, line, request):
    log = AuditLog(tmp_path / "audit.jsonl")
    log.append("c", "a", input_hash="sha256:i", output_hash="sha256:o")
    ev = json.loads(log.path.read_text())
    idx = request.node.callspec.indices["line"]
    if idx == 1:
        ev["timestamp"] = 1793026800
    elif idx == 2:
        ev["timestamp"] = "0001-01-01T00:00:00+01:00"
    raw = line if line is not None else (json.dumps(ev) + "\n").encode()
    log.path.write_bytes(raw)
    r = log.verify()
    assert not r.ok and r.error
    with pytest.raises(AuditCorruption):
        log.append("c", "a", input_hash="sha256:i", output_hash="sha256:o")


def test_n5_numeric_timestamps_hash_like_their_iso_form(tmp_path):
    from kyb.audit.chain import canonical_ts

    assert canonical_ts(1792940400) == canonical_ts("2026-10-25T15:00:00Z")


def test_n9_concurrent_init_without_force_cannot_overwrite(tmp_path):
    root = tmp_path / "ws"
    results: list[str] = []

    def go(tenant: str) -> None:
        try:
            init_workspace(root, tenant=tenant, policy_src=POLICY)
            results.append(tenant)
        except FileExistsError:
            results.append("refused")

    threads = [threading.Thread(target=go, args=(t,)) for t in ("one", "two")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results.count("refused") == 1
    inits = [json.loads(x)["action"] for x in _log(root).read_text().splitlines()]
    assert inits.count("workspace.init") == 1


@needs_poppler
def test_n6_conflicting_catalog_provenance_refuses_before_any_write(ws):
    name = next(PACKET_A.glob("02_*.pdf")).name
    rel = f"bronze/packets/{CASE}/{name}"
    target = ws / rel
    target.parent.mkdir(parents=True)
    shutil.copy(PACKET_A / name, target)
    Catalog(ws).register(rel, tier="untrusted", produced_by={"action": "other"}, as_of="2020-01-01")
    with pytest.raises(PermissionError, match="provenance"):
        ingest_packet(ws, PACKET_A, as_of="2026-10-25")
    assert sorted(p.name for p in target.parent.iterdir()) == [name]


def test_n7_option_values_are_not_operands(ws):
    assert check_bash("pdftotext -f 1 cases/A/scratch/in.pdf ledger/new.txt", ws, ws)
    assert (
        check_bash("pdftotext -f 1 -l 1 cases/A/scratch/in.pdf cases/A/notes/p1.txt", ws, ws)
        is None
    )
    assert check_bash("pdftoppm -r 600 -x 1 -y 2 -W 3 -H 4 -png in.pdf gold/x", ws, ws)


def test_cli_audit_verify_is_checked_at_a_boundary(ws, capsys):
    assert main(["-w", str(ws), "audit", "verify"]) == 0
    AuditLog(_log(ws)).append("x", "y", input_hash="sha256:i", output_hash="sha256:o")
    assert main(["-w", str(ws), "audit", "verify"]) == 1
    assert "not checkpointed" in capsys.readouterr().out


def test_hooks_run_from_a_bare_path(ws):
    cmd = [sys.executable, "-m", "kyb.cli", "audit", "append-hook"]
    p = subprocess.run(cmd, input=json.dumps({"cwd": str(ws)}), text=True, capture_output=True,
                       env={"PATH": "/usr/bin:/bin"}, cwd=ws)  # fmt: skip
    assert p.returncode == 0, p.stderr
    assert verify_workspace(ws, at_boundary=True).ok


# --- round 3 ---------------------------------------------------------------------------------


def test_r3_1_failed_first_init_stays_retryable(tmp_path):
    root = tmp_path / "ws"
    with pytest.raises(FileNotFoundError):
        init_workspace(root, tenant="t", policy_src=tmp_path / "missing.md")
    assert not list((root / "ledger" / "checkpoints").glob("cp-*.json"))  # nothing to pin
    assert verify_workspace(root, at_boundary=True).ok
    init_workspace(root, tenant="t", policy_src=POLICY)
    assert verify_workspace(root, at_boundary=True).ok
    init_workspace(root, tenant="t", policy_src=POLICY, force=True)
    assert verify_workspace(root, at_boundary=True).ok


def test_r3_1_invalid_checkpoint_sequence_is_a_structured_failure(ws):
    cps = ws / "ledger" / "checkpoints"
    (cps / "cp--0000001.json").write_text('{"head": "GENESIS", "seq": -1}\n')
    r = verify_workspace(ws)
    assert not r.ok and "checkpoint" in r.error


def test_r3_n1_an_initialised_workspace_without_its_init_checkpoint_never_verifies(ws):
    for cp in (ws / "ledger" / "checkpoints").glob("cp-*.json"):
        cp.unlink()
    assert not verify_workspace(ws).ok
    assert not verify_workspace(ws, at_boundary=True).ok


@needs_poppler
def test_r3_n6_any_catalog_difference_refuses_before_any_write(ws):
    name = next(PACKET_A.glob("02_*.pdf")).name
    rel = f"bronze/packets/{CASE}/{name}"
    (ws / rel).parent.mkdir(parents=True)
    shutil.copy(PACKET_A / name, ws / rel)
    from kyb import __version__
    from kyb.docintel import poppler

    produced_by = {
        "action": "ingest_packet",
        "pin": f"kyb@{__version__}",
        "tool": poppler.version(),
    }
    Catalog(ws).register(rel, tier="claimed", produced_by=produced_by, as_of="2026-10-25")
    with pytest.raises(PermissionError, match="provenance"):
        ingest_packet(ws, PACKET_A, as_of="2026-10-25")
    assert sorted(p.name for p in (ws / rel).parent.iterdir()) == [name]


@pytest.mark.skipif(shutil.which("pdftotext") is None, reason="poppler missing")
def test_r3_n7_guard_reads_options_from_the_tool_and_follows_derived_outputs(ws):
    assert check_bash("pdftotext -colspacing 0.7 cases/A/scratch/in.pdf ledger/new.txt", ws, ws)
    assert check_bash("pdftotext bronze/packets/A/in.pdf", ws, ws)  # writes bronze/…/in.txt
    assert "unsupported option" in check_bash("pdftotext -bogus x cases/A/scratch/in.pdf", ws, ws)
    ok = "pdftotext -colspacing 0.7 -f 1 cases/A/scratch/in.pdf cases/A/notes/p1.txt"
    assert check_bash(ok, ws, ws) is None
    assert check_bash("pdftotext -layout bronze/packets/A/in.pdf -", ws, ws) is None  # stdout
    assert check_bash("pdftotext cases/A/scratch/in.pdf", ws, ws) is None


# --- round 4 ---------------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("pdftotext") is None, reason="poppler missing")
def test_r4_dash_is_stdout_only_for_pdftotext(ws):
    assert check_bash("pdftotext cases/A/scratch/in.pdf -", ws, ws) is None
    dash_files = (
        "pdftoppm -r 10 -png cases/A/scratch/in.pdf -",
        "pdfimages cases/A/scratch/in.pdf -",
        "pdfseparate cases/A/scratch/in.pdf -",
    )
    for cmd in dash_files:
        assert check_bash(cmd, ws, ws), cmd  # `-` is a file at the workspace root


@pytest.mark.skipif(shutil.which("pdftocairo") is None, reason="poppler missing")
def test_r4_pdftocairo_derives_its_output_in_the_working_directory(ws):
    scratch = ws / "cases" / "A" / "scratch"
    scratch.mkdir(parents=True)
    assert check_bash("pdftocairo -r 10 -png ../../../bronze/packets/A/in.pdf", ws, scratch) is None
    assert check_bash("pdftocairo -r 10 -png cases/A/scratch/s.pdf", ws, ws)  # would land at root
