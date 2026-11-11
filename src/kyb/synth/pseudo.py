"""Deterministic pseudonymisation of real natural persons (names, dates of birth, ids).

A raw person (a registry row) maps to a synthetic one by hashing (seed, key): the same seed gives
the same pseudonym, a different seed a different one. The raw → pseudonym map is written only under
the cache directory (never the repo) so a run can be audited by whoever holds the cache. Synthetic
people, who have no raw source, take the same path with a synthetic key.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

from kyb.synth.sources import cache_root

FIRST = (
    "Camila",
    "Mateo",
    "Valentina",
    "Santiago",
    "Isabela",
    "Sebastián",
    "Daniela",
    "Nicolás",
    "Mariana",
    "Andrés",
    "Lucía",
    "Tomás",
    "Paula",
    "Felipe",
    "Natalia",
    "Esteban",
    "Sara",
    "Julián",
    "Renata",
    "Ignacio",
    "Gabriela",
    "Emilio",
    "Ximena",
    "Rodrigo",
    "Catalina",
    "Joaquín",
    "Fernanda",
    "Hernán",
    "Pilar",
    "Bruno",
    "Adriana",
    "Leandro",
    "Marta",
    "Dario",
    "Teresa",
    "Ivan",
    "Rosa",
    "Nestor",
    "Olga",
    "Ramiro",
)
LAST = (
    "Aranguren",
    "Bellido",
    "Cardozo",
    "Dueñas",
    "Echeverri",
    "Fontalvo",
    "Galindo",
    "Hurtado",
    "Izquierdo",
    "Jaramillo",
    "Kuri",
    "Lleras",
    "Montoya",
    "Narváez",
    "Ospina",
    "Pedraza",
    "Quintero",
    "Rangel",
    "Sarmiento",
    "Tobón",
    "Urrutia",
    "Villamizar",
    "Wiesner",
    "Yepes",
    "Zuluaga",
    "Arismendi",
    "Betancur",
    "Calderón",
    "Diazgranados",
    "Escallón",
    "Ferreira",
    "Guerrero",
    "Holguín",
    "Iragorri",
    "Lozano",
    "Mutis",
    "Ordóñez",
    "Pizarro",
    "Herrera-Vélez",
    "Salgado",
)


class Pseudonymiser:
    def __init__(self, seed: int, root: Path | None = None) -> None:
        self.seed = seed
        self.path = (root or cache_root()) / "pseudonyms" / f"seed-{seed}.json"
        self.map: dict[str, dict[str, str]] = {}
        self._used: set[str] = set()

    def _digest(self, key: str) -> int:
        return int.from_bytes(hashlib.sha256(f"{self.seed}|{key}".encode()).digest()[:8], "big")

    def person(self, key: str, raw_name: str = "") -> dict[str, str]:
        """{'name', 'dob', 'id'} for the person behind `key`; stable per (seed, key)."""
        if key in self.map:
            return self.map[key]
        h = self._digest(key)
        for bump in range(1000):
            x = h + bump
            first, last1, last2 = FIRST[x % 40], LAST[(x // 40) % 40], LAST[(x // 1600) % 40]
            name = f"{first} {last1} {last2}"
            if name not in self._used:
                break
        self._used.add(name)
        dob = date(1955, 1, 1) + timedelta(days=self._digest("dob|" + key) % (50 * 365))
        pid = f"{self._digest('id|' + key) % 10**10:010d}"
        self.map[key] = {"name": name, "dob": dob.isoformat(), "id": pid, "raw": raw_name}
        return self.map[key]

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.map, indent=1, sort_keys=True, ensure_ascii=False))
        return self.path
