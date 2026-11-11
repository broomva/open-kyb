"""Shared fixtures for the synthetic-set tests: tiny raw snapshots in tests/fixtures/synth/."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import pytest

from kyb.synth.sources import RawData, load_raw

FIX = Path(__file__).parent / "fixtures" / "synth"


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    d = tmp_path / "raw"
    d.mkdir()
    for name in ("rues_sociedades.json", "sat_69b.csv"):
        shutil.copy(FIX / name, d / name)
    with zipfile.ZipFile(d / "gleif_rr.zip", "w") as z:
        z.write(FIX / "gleif_rr.xml", "gleif_rr.xml")
    with zipfile.ZipFile(d / "icij_offshoreleaks.zip", "w") as z:
        z.write(FIX / "icij_entities.csv", "nodes-entities.csv")
    return d


@pytest.fixture
def raw(raw_dir: Path) -> RawData:
    return load_raw(raw_dir)
