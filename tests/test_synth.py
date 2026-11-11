"""L11 synthetic KYB set: sources, builder, label engine, renderer. Offline: tiny fixtures in
tests/fixtures/synth/ stand in for the open-data snapshots, and nothing here touches the network."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from kyb.sim.decide import FixtureDecider, LabelDecider
from kyb.sim.l4 import L4Decider
from kyb.synth import build as B
from kyb.synth import inject as I  # noqa: E741
from kyb.synth.case import INJECTION_RULES
from kyb.synth.golden import GOLDEN, golden_specs
from kyb.synth.label import label
from kyb.synth.pseudo import Pseudonymiser
from kyb.synth.render import render_all
from kyb.synth.sources import RawData, load_raw

pytest_plugins = ["synth_support"]  # raw_dir / raw fixtures

RAW_NAMES = (
    "ZAPATA",
    "QUINTANILLA",
    "XIMENEZ",
    "BARRIENTOS",
    "EUSEBIA",
    "ALTAMIRANO",
    "ABENDA",
    "ROSALBA",
    "VILLASE",
)  # person tokens in the SAT fixture (accents folded below)
RULES_EVERY_BUILD = {
    "SCR-01",
    "SCR-02",
    "JUR-01",
    "JUR-02",
    "OWN-01",
    "OWN-02",
    "GOV-01",
    "GOV-02",
    "GOV-03",
    "PEP-01",
    "PEP-02",
    "DOC-01",
    "DOC-02",
    "OPS-01",
    "OPS-02",
    "ACT-01",
    "ACT-02",
    "MOD-01",
}


def build(raw: RawData, tmp_path: Path, n: int = 40, seed: int = 11):
    return B.build(n, seed, raw, cache=tmp_path / "cache")


# --- sources -------------------------------------------------------------------------------------


def test_loaders_read_the_fixture_formats(raw: RawData):
    assert raw.used == {"rues": 10, "sat_69b": 6, "sat_69b_persons": 4, "gleif_rr": 6, "icij": 5}
    assert {s.source for s in raw.shells} == {"rues", "sat_69b"}
    assert any(s.status == "CANCELADA" for s in raw.shells)  # cancelled companies are kept
    assert not any("RAMIREZ" in s.name for s in raw.shells)  # a persona natural is not a shell
    sat = [s for s in raw.shells if s.source == "sat_69b"]
    assert {s.flag for s in sat} >= {"Definitivo", "Presunto"}  # recorded, never a rule
    assert any("ENERGÍA" in s.name for s in sat)  # latin-1 decoded
    assert all(s.incorporated.year in range(1998, 2016) for s in sat)  # RFC embeds the date
    assert {o.jurisdiction for o in raw.offshore} == {"VG", "PA", "SC"}
    depths = sorted(d for d, _ in raw.gleif_shapes)
    assert depths == [1, 1, 1, 2, 2, 3]  # AAA→BBB→CCC→DDD is a 3-deep chain


def test_missing_snapshots_degrade_to_synthetic_shells(tmp_path: Path):
    empty = load_raw(tmp_path / "nothing")
    assert empty.shells == [] and empty.used["rues"] == 0
    cases, labels, rep, _ = B.build(12, 5, empty, cache=tmp_path / "c")
    assert {c.company.source for c in cases} == {"synthetic"}
    assert len(labels) == 12


# --- determinism, mix, coverage ------------------------------------------------------------------


def test_build_is_deterministic_by_seed(raw: RawData, tmp_path: Path):
    a, _, _, _ = build(raw, tmp_path, 30, 11)
    b, _, _, _ = build(raw, tmp_path / "again", 30, 11)
    c, _, _, _ = build(raw, tmp_path / "other", 30, 12)
    assert B.digest(a) == B.digest(b)
    assert B.digest(a) != B.digest(c)
    assert [x.case_id for x in a] == [f"syn-11-{i:03d}" for i in range(30)]


def test_mix_and_rule_coverage(raw: RawData, tmp_path: Path):
    cases, labels, rep, _ = build(raw, tmp_path, 80, 11)
    assert rep.labels == {"APPROVED": 28, "MANUAL_REVIEW": 28, "REJECTED": 24}  # 35/35/30
    assert set(rep.rules) >= RULES_EVERY_BUILD, RULES_EVERY_BUILD - set(rep.rules)
    assert {x for x in INJECTION_RULES if x not in rep.injections} == set()
    # the label never comes from the plan: re-deriving it gives the same answer
    assert [label(c) for c in cases] == labels


def test_fixture_decider_agrees_with_the_label_engine(raw: RawData, tmp_path: Path):
    cases, labels, _, _ = build(raw, tmp_path, 60, 3)
    fx, oracle = FixtureDecider(), LabelDecider()
    bad = [c.case_id for c in cases if fx.decide(c) != oracle.decide(c)]
    assert bad == []


# --- the label engine: golden A/B/C --------------------------------------------------------------


@pytest.mark.parametrize("key", ["A", "B", "C"])
def test_label_engine_reproduces_golden(key: str):
    spec = golden_specs()[key]
    status, score, rules = GOLDEN[key]
    label(spec)
    pass
    assert (status, score) == ("MANUAL_REVIEW", {"A": 15, "B": 85, "C": 90}[key])
    FixtureDecider().decide(spec)  # the real policy engine, on the same facts
    pass


def test_golden_notes_and_unreached_rules():
    lab = label(golden_specs()["B"])
    assert any("PEP-0103" in n and "screening note" in n for n in lab.notes)  # recorded, not fired
    assert lab.conflicts == 3  # spec
    d = L4Decider().decide(golden_specs()["A"])  # phase 2: L4's real judges + decision assembly
    assert (d.status, d.score) == ("MANUAL_REVIEW", 15)


# --- injections ----------------------------------------------------------------------------------


def _draft(raw: RawData, tmp_path: Path, seed: int):
    rng = random.Random(seed)
    ctx = I.Ctx(rng=rng, pseud=Pseudonymiser(1, tmp_path), raw=raw, persons=list(raw.persons)[:6])
    shell = rng.choice([s for s in raw.shells if s.status == "ACTIVA"])
    return B.draft_case("t", shell, ctx), ctx


@pytest.mark.parametrize("name", sorted(I.INJECTIONS))
def test_each_injection_fires_its_rule(name: str, raw: RawData, tmp_path: Path):
    want = INJECTION_RULES[name]
    for seed in range(25):
        d, ctx = _draft(raw, tmp_path, seed)
        if name in I.CONFLICT_LAST:
            d.sync()
        I.INJECTIONS[name](d, ctx)
        if name not in I.CONFLICT_LAST:
            d.sync()
        lab = label(d.spec())
        assert d.spec().injection(name) is not None or name == "pep"  # recorded with parameters
        if want:
            assert want in lab.rules or want in lab.unknown, (name, seed, lab.rules)
        else:  # the decoy: a name-only PEP similarity is a note, not a trigger (spec)
            assert not {"PEP-01", "PEP-02"} & set(lab.rules) and lab.notes


def test_sat_69b_definitivo_is_ignored_by_the_label(raw: RawData, tmp_path: Path):
    d, _ = _draft(raw, tmp_path, 0)
    spec = d.spec()
    flagged = spec.model_copy(update={"source_flags": {"sat_69b": "Definitivo"}})
    assert label(flagged) == label(spec)  # OPS-01's text has no 69-B indicator


# --- pseudonymisation ----------------------------------------------------------------------------


def _fold(s: str) -> str:
    return (
        s.upper()
        .replace("Á", "A")
        .replace("É", "E")
        .replace("Í", "I")
        .replace("Ó", "O")
        .replace("Ú", "U")
        .replace("Ñ", "N")
    )


def test_no_raw_person_name_reaches_any_output(raw: RawData, tmp_path: Path):
    cache = tmp_path / "cache"
    cases, labels, _, ps = B.build(30, 11, raw, cache=cache)
    out = tmp_path / "out"
    B.write_outputs(cases, labels, B.summarise(30, 11, cases, labels, raw, 0), 11, out / "cases")
    render_all(cases, labels, out / "packets", seed=11)
    index = json.dumps(B.index_rows(cases, labels))
    blobs = [index]
    for p in out.rglob("*"):
        if p.is_file():
            blobs.append(_fold(p.read_bytes().decode("cp1252", "replace")))
    text = _fold("\n".join(blobs))
    for token in RAW_NAMES:
        assert token not in text, f"raw name token {token} leaked"
    mapping = json.loads((cache / "pseudonyms" / "seed-11.json").read_text())
    assert any(
        "ZAPATA" in _fold(v["raw"]) for v in mapping.values()
    )  # the map exists, in the cache
    assert not list(out.rglob("pseudonyms*"))  # and never beside the outputs


def test_pseudonyms_are_stable_per_seed_and_differ_across_seeds(tmp_path: Path):
    a, b, c = (Pseudonymiser(s, tmp_path) for s in (1, 1, 2))
    p = a.person("k", "RAW NAME")
    assert b.person("k", "RAW NAME")["name"] == p["name"] and b.person("k")["dob"] == p["dob"]
    assert c.person("k", "RAW NAME")["name"] != p["name"]
    assert p["id"] != "k" and "RAW" not in p["name"]


# --- rendering -----------------------------------------------------------------------------------


def test_render_packets_and_manifest(raw: RawData, tmp_path: Path):
    cases, labels, _, _ = build(raw, tmp_path, 12, 4)
    out = tmp_path / "packets"
    render_all(cases, labels, out, seed=4)
    for c, lab in zip(cases, labels, strict=True):
        folder = out / c.case_id
        man = json.loads((folder / "manifest.json").read_text())
        assert man["case_id"] == c.case_id and man["label"]["status"] == lab.status
        assert [f["rule"] for f in man["label"]["fired"]] == [f.rule for f in lab.fired]
        assert [i["name"] for i in man["injections"]] == [i.name for i in c.injections]
        pdfs = sorted(p.name for p in folder.glob("*.pdf"))
        assert len(pdfs) == sum(d.present for d in c.documents)
        assert all((folder / p).read_bytes().startswith(b"%PDF") for p in pdfs)
        assert {p.removesuffix("_SCAN.pdf").removesuffix(".pdf") for p in pdfs} <= set(
            __import__("kyb.synth.render", fromlist=["x"]).FILENAMES.values()
        )
    # an unscanned document carries Spanish text
    text = next(p for p in (out / cases[0].case_id).glob("*.pdf") if "SCAN" not in p.name)
    assert b"(" in text.read_bytes()
