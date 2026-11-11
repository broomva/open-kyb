"""Loaders for the raw open-data snapshots (fetched by scripts/synth/fetch_sources.py).

Each loader reads a snapshot from the cache directory and returns plain rows; a missing snapshot
yields an empty list, so the builder can fall back to synthetic shells (and says so). Loaders stop
after `limit` rows so a build never walks a whole 200 MB file.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import xml.etree.ElementTree as ET
import zipfile
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict


def cache_root() -> Path:
    return Path(os.environ.get("KYB_SYNTH_CACHE", Path.home() / ".cache" / "kyb" / "synth"))


def raw_dir() -> Path:
    return cache_root() / "raw"


class Shell(BaseModel):
    """A company skeleton from a registry row."""

    model_config = ConfigDict(frozen=True)
    source: str  # "rues" | "sat_69b"
    name: str
    jurisdiction: str
    incorporated: date
    status: str
    ciiu: str | None = None
    city: str | None = None
    flag: str | None = None  # SAT 69-B situation, recorded and never a rule


class RawPerson(BaseModel):
    """A real natural person's registry row: pseudonymised before it reaches any case."""

    model_config = ConfigDict(frozen=True)
    source: str
    name: str
    key: str  # the registry id (RFC): used only to derive the pseudonym


class OffshoreEntity(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    jurisdiction: str  # ISO alpha-2 where known


class RawData(BaseModel):
    model_config = ConfigDict(frozen=True)
    shells: list[Shell] = []
    persons: list[RawPerson] = []
    gleif_shapes: list[tuple[int, int]] = []  # (chain depth, direct stake %)
    offshore: list[OffshoreEntity] = []
    used: dict[str, int] = {}  # source → rows used, for the build report


ICIJ_ISO = {
    "BVI": "VG",
    "PMA": "PA",
    "SEY": "SC",
    "BAH": "BS",
    "ANG": "AI",
    "NEV": "KN",
    "NIUE": "NU",
    "SAM": "WS",
    "UK": "GB",
    "HK": "HK",
    "UY": "UY",
    "IOM": "IM",
}
COMPANY_MARK = re.compile(
    r"\b(S\.?\s?A\.?|S\.?\s?C\.?|S\. DE R\.L\.|S\.?\s?A\.?P\.?I\.?|SAS|LTDA)\b"
)


def load_rues(path: Path, limit: int = 4000) -> list[Shell]:
    if not path.is_file():
        return []
    out = []
    for r in json.loads(path.read_text(encoding="utf-8")):
        d = r.get("fecha_matricula", "")
        if len(d) != 8 or not r.get("razon_social") or r.get("clase_identificacion") != "NIT":
            continue
        try:
            inc = date(int(d[:4]), int(d[4:6]), int(d[6:]))
        except ValueError:
            continue
        out.append(
            Shell(
                source="rues",
                name=r["razon_social"].strip(),
                jurisdiction="CO",
                incorporated=inc,
                status=r.get("estado_matricula", "ACTIVA"),
                ciiu=r.get("cod_ciiu_act_econ_pri"),
                city=r.get("camara_comercio"),
            )
        )
        if len(out) >= limit:
            break
    return out


def _sat_rows(path: Path) -> list[list[str]]:
    """The SAT file is latin-1 with two preamble rows; the header row starts with 'No,RFC'."""
    text = path.read_bytes().decode("latin-1")
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith("No,RFC,")), None)
    if start is None:
        return []
    return list(csv.reader(lines[start + 1 :]))


def _rfc_date(rfc: str) -> date | None:
    m = re.match(r"^[A-ZÑ&]{3,4}(\d{2})(\d{2})(\d{2})", rfc)
    if not m:
        return None
    yy, mm, dd = (int(x) for x in m.groups())
    try:
        return date((1900 if yy > 26 else 2000) + yy, mm, dd)
    except ValueError:
        return None


def load_sat(path: Path, limit: int = 3000) -> tuple[list[Shell], list[RawPerson]]:
    """Personas morales (12-char RFC) become shells; personas físicas (13) become raw persons."""
    if not path.is_file():
        return [], []
    shells, persons = [], []
    for row in _sat_rows(path):
        if len(row) < 4 or not row[1].strip():
            continue
        rfc, name, situation = row[1].strip(), row[2].strip(), row[3].strip()
        if len(rfc) == 12 and len(shells) < limit:
            inc = _rfc_date(rfc)
            if inc:
                shells.append(
                    Shell(
                        source="sat_69b",
                        name=name,
                        jurisdiction="MX",
                        incorporated=inc,
                        status="ACTIVA",
                        flag=situation,
                    )
                )
        elif len(rfc) == 13 and len(persons) < limit:
            persons.append(RawPerson(source="sat_69b", name=name, key=rfc))
    return shells, persons


def load_gleif_shapes(path: Path, limit: int = 40000) -> list[tuple[int, int]]:
    """Chain depth and direct stake from active, directly-consolidated relationship records:
    (depth of the parent chain above the child, stake %). Names are not in the RR file, so only the
    shape is used; every entity name in a case is synthetic."""
    if not path.is_file():
        return []
    ns = "{http://www.gleif.org/data/schema/rr/2016}"
    parent: dict[str, tuple[str, int]] = {}
    seen = 0
    with zipfile.ZipFile(path) as z, z.open(z.namelist()[0]) as fh:
        for _, el in ET.iterparse(fh, events=("end",)):
            if el.tag != ns + "RelationshipRecord":
                continue
            rel = el.find(ns + "Relationship")
            if (
                rel is not None
                and rel.findtext(ns + "RelationshipType") == ("IS_DIRECTLY_CONSOLIDATED_BY")
                and rel.findtext(ns + "RelationshipStatus") == "ACTIVE"
            ):
                child = rel.findtext(f"{ns}StartNode/{ns}NodeID")
                par = rel.findtext(f"{ns}EndNode/{ns}NodeID")
                amount = rel.findtext(f".//{ns}QuantifierAmount")
                try:
                    stake = int(round(float(amount))) if amount else 100
                except ValueError:
                    stake = 100
                if child and par:
                    parent[child] = (par, max(1, min(100, stake)))
            el.clear()
            seen += 1
            if seen >= limit:
                break
    shapes = []
    for child in sorted(parent):
        depth, node, guard = 0, child, set()
        while node in parent and node not in guard:
            guard.add(node)
            node = parent[node][0]
            depth += 1
        shapes.append((min(depth, 4), parent[child][1]))
    return shapes


def load_icij(path: Path, limit: int = 4000, scan: int = 120000) -> list[OffshoreEntity]:
    """Offshore entity names from nodes-entities.csv: up to `limit` of the first `scan` rows."""
    if not path.is_file():
        return []
    out = []
    with zipfile.ZipFile(path) as z:
        name = next((n for n in z.namelist() if n.endswith("nodes-entities.csv")), None)
        if name is None:
            return []
        with io.TextIOWrapper(z.open(name), encoding="utf-8", newline="") as fh:
            for i, row in enumerate(csv.DictReader(fh)):
                if i >= scan or len(out) >= limit:
                    break
                iso = ICIJ_ISO.get(row.get("jurisdiction", ""))
                nm = (row.get("name") or "").strip()
                if iso and nm and nm.isascii():
                    out.append(
                        OffshoreEntity(name=nm.title() if nm.isupper() else nm, jurisdiction=iso)
                    )
    return out


def load_raw(root: Path | None = None, *, limit: int = 4000) -> RawData:
    d = Path(root) if root else raw_dir()
    rues = load_rues(d / "rues_sociedades.json", limit)
    sat, persons = load_sat(d / "sat_69b.csv", limit)
    shapes = load_gleif_shapes(d / "gleif_rr.zip")
    off = load_icij(d / "icij_offshoreleaks.zip", limit)
    return RawData(
        shells=[*rues, *sat],
        persons=persons,
        gleif_shapes=shapes,
        offshore=off,
        used={
            "rues": len(rues),
            "sat_69b": len(sat),
            "sat_69b_persons": len(persons),
            "gleif_rr": len(shapes),
            "icij": len(off),
        },
    )
