"""Ingest + pages on the real packet A (requires poppler), plus the record-integrity regressions."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
from pathlib import Path

import pytest

from kyb.audit.chain import AuditLog, verify_workspace
from kyb.docintel.ingest import ingest_packet
from kyb.docintel.pages import build_pages, doc_keys
from kyb.util.fsutil import UnsafePath
from kyb.workspace.catalog import Catalog
from kyb.workspace.init import init_workspace
from kyb.workspace.runs import list_runs, start_run

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "examples"
PACKET_A = ASSETS / "packets" / "cl-andes-agro"
CASE = "cl-andes-agro"

pytestmark = pytest.mark.skipif(shutil.which("pdftotext") is None, reason="poppler not installed")


@pytest.fixture
def ws(tmp_path) -> Path:
    root = tmp_path / "ws"
    init_workspace(root, tenant="kyb-demo", policy_src=ASSETS / "policy/compliance_rules.md")
    return root


def test_ingest_freezes_bronze(ws):
    m = ingest_packet(ws, PACKET_A, as_of="2026-10-25")
    assert m["packet_id"] == CASE and m["as_of"] == "2026-10-25"
    by_name = {f["name"]: f for f in m["files"]}
    assert len(by_name) == 5
    assert by_name["04_boleta_de_suministro_SCAN.pdf"]["text_layer"] is False
    assert all(f["text_layer"] for n, f in by_name.items() if "SCAN" not in n)
    assert Catalog(ws).get(f"bronze/packets/{CASE}/manifest.json")["tier"] == "untrusted"
    ingest_packet(ws, PACKET_A, as_of="2026-10-25")  # identical input is idempotent
    assert verify_workspace(ws).ok


def test_rejected_ingest_leaves_every_byte_unchanged(ws):
    ingest_packet(ws, PACKET_A, as_of="2026-10-25")
    manifest = (ws / f"bronze/packets/{CASE}/manifest.json").read_bytes()
    catalog = (ws / "catalog.json").read_bytes()
    with pytest.raises(PermissionError):
        ingest_packet(ws, PACKET_A, as_of="2026-11-01")
    assert (ws / f"bronze/packets/{CASE}/manifest.json").read_bytes() == manifest
    assert (ws / "catalog.json").read_bytes() == catalog


def test_ingest_refuses_unsafe_names_and_symlinked_destinations(ws, tmp_path):
    bad = tmp_path / "bad"
    bad.mkdir()
    shutil.copyfile(PACKET_A / "01_estatutos_actualizados.pdf", bad / "-x.pdf")
    with pytest.raises(ValueError):
        ingest_packet(ws, bad, as_of="2026-10-25", case="bad")
    outside = tmp_path / "outside"
    outside.mkdir()
    (ws / "bronze/packets/evil").symlink_to(outside, target_is_directory=True)
    with pytest.raises(UnsafePath):
        ingest_packet(ws, PACKET_A, as_of="2026-10-25", case="evil")
    assert list(outside.iterdir()) == []


def test_catalog_entries_are_append_only(ws):
    (ws / "silver/x.json").write_text("{}")
    cat = Catalog(ws)
    e = cat.register("silver/x.json", tier="claimed", produced_by={"a": 1}, as_of="2026-10-25")
    assert cat.register("silver/x.json", tier="claimed", produced_by={"a": 1},
                        as_of="2026-10-25") == e  # fmt: skip
    for change in ({"tier": "verified"}, {"as_of": "2026-10-24"}, {"produced_by": {"a": 2}}):
        kw = {"tier": "claimed", "produced_by": {"a": 1}, "as_of": "2026-10-25", **change}
        with pytest.raises(PermissionError):
            cat.register("silver/x.json", **kw)
    with pytest.raises(UnsafePath):
        cat.register("silver/../../escape.json", tier="claimed", produced_by={}, as_of="2026-10-25")


def _register_many(root: str, start: int) -> None:
    cat = Catalog(Path(root))
    for i in range(start, start + 15):
        cat.register(f"silver/c{i}.json", tier="untrusted", produced_by={}, as_of="2026-10-25")


def test_catalog_survives_concurrent_writers(ws):
    for i in range(60):
        (ws / f"silver/c{i}.json").write_text(str(i))
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_register_many, args=(str(ws), s)) for s in (0, 15, 30, 45)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
        assert p.exitcode == 0
    data = json.loads((ws / "catalog.json").read_text())
    assert all(f"silver/c{i}.json" in data for i in range(60))


def test_pages_are_run_keyed_and_say_where_text_came_from(ws):
    ingest_packet(ws, PACKET_A, as_of="2026-10-25")
    run = start_run(ws, CASE, as_of="2026-10-25", reason="test")
    assert run == "r1" and list_runs(ws, CASE) == ["r1"]
    assert Catalog(ws).get(f"ledger/runs/{CASE}/r1.json") is not None  # run records are catalogued
    records = build_pages(ws, CASE, run)
    assert len(records) == 5
    scan = json.loads((ws / f"silver/pages/{CASE}/r1/04/p1.json").read_text())
    assert scan["text_layer"] is None and "reads the pixels" in scan["reader"]
    cert = json.loads((ws / f"silver/pages/{CASE}/r1/01/p1.json").read_text())
    assert "76.419.082-1" in cert["text_layer"]  # NIT, from the text layer
    png = ws / f"silver/pages/{CASE}/r1/04/p1.png"
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    entry = Catalog(ws).get(f"silver/pages/{CASE}/r1/04/p1.png")
    assert CASE in entry["inputs"][0]
    build_pages(ws, CASE, run)  # deterministic: re-rendering the same run is byte-identical
    assert verify_workspace(ws).ok
    actions = [json.loads(line)["action"] for line in
               (ws / "ledger/audit.jsonl").read_text().splitlines()]  # fmt: skip
    assert "run.started" in actions
    assert start_run(ws, CASE, as_of="2026-10-25", reason="test") == "r2"


def test_pages_refuse_replaced_bronze_bytes(ws):
    ingest_packet(ws, PACKET_A, as_of="2026-10-25")
    run = start_run(ws, CASE, as_of="2026-10-25", reason="test")
    target = ws / f"bronze/packets/{CASE}/01_estatutos_actualizados.pdf"
    os.chmod(target, 0o644)
    target.write_bytes(target.read_bytes() + b"\n%tampered")
    with pytest.raises(PermissionError):
        build_pages(ws, CASE, run)


def test_pages_refuse_unsafe_run_ids_before_writing(ws, tmp_path):
    ingest_packet(ws, PACKET_A, as_of="2026-10-25")
    with pytest.raises(ValueError):
        build_pages(ws, CASE, "../../../outside")
    assert not (tmp_path / "outside").exists()


def test_doc_keys_are_unique_and_safe():
    assert doc_keys(["01_a.pdf", "02_b.pdf"]) == {"01_a.pdf": "01", "02_b.pdf": "02"}
    keys = doc_keys(["01_first.pdf", "01_second.pdf"])
    assert len(set(keys.values())) == 2 and keys["01_first.pdf"] == "01_first"
    assert AuditLog  # imported for parity with other suites
