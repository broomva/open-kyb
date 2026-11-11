"""Decision cases as data: a golden FactSet + screening + recorded answers, and the mutations that
derive the synthetic variants from it (EVALS datasets: "generated with the mutation that defines
the expected delta").

A mutation is a small JSON op over the b4 records, so every variant reads as the delta it tests.
Evidence written as `"A02-05"` expands to a verified EvidenceRef on that document, the same
shorthand as the L3 fixtures. After the ops, the evidence table is completed and both
`fact_set_hash`es are recomputed, so a variant is a well-formed FactSet in its own right.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from kyb.contracts import b4
from kyb.evals.paths import load
from kyb.util.hashing import sha256_json


@dataclass(frozen=True)
class Packet:
    facts: b4.FactSet
    screening: b4.ScreeningResult
    answers: dict[str, b4.PredicateAnswer]


def ref(short: str, page: int = 1, verified: bool = True) -> dict[str, Any]:
    """`"A02-05"` → the EvidenceRef of claim `c-A02-05` on document `02`."""
    return {"claim_id": f"c-{short}", "doc": short[1:3], "page": page, "verified": verified}


def _expand(node: Any) -> Any:
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k in ("evidence", "context") and isinstance(v, list):
                out[k] = [ref(x) if isinstance(x, str) else _expand(x) for x in v]
            else:
                out[k] = _expand(v)
        return out
    if isinstance(node, list):
        return [_expand(x) for x in node]
    return node


def _refs(node: Any) -> list[dict]:
    if isinstance(node, dict):
        own = [node] if "claim_id" in node and "doc" in node else []
        return own + [r for k, v in node.items() if k != "evidence_items" for r in _refs(v)]
    if isinstance(node, list):
        return [r for v in node for r in _refs(v)]
    return []


def _by(items: list[dict], key: str, value: str) -> dict:
    for it in items:
        if it[key] == value:
            return it
    raise KeyError(f"no {key}={value!r}")


def _set(target: dict, changes: dict) -> None:
    for k, v in changes.items():
        if v is None and k in target and isinstance(target[k], list):
            target[k] = []
        else:
            target[k] = _expand(v)


def apply(facts: dict, screening: dict, answers: dict, ops: list[dict]) -> None:
    """Apply mutation ops in place. Unknown ops raise: a typo must not pass as a no-op."""
    for op in ops:
        kind = op["op"]
        if kind == "doc":
            _set(_by(facts["documents"], "doc", op["doc"]), op["set"])
        elif kind == "drop_doc":
            facts["documents"] = [d for d in facts["documents"] if d["doc"] != op["doc"]]
        elif kind == "entity":
            _set(_by(facts["entities"], "entity_id", op["id"]), op["set"])
        elif kind == "add_entity":
            facts["entities"].append(_expand(op["entity"]))
        elif kind == "holding":
            _set(_by(facts["holdings"], "holding_id", op["id"]), op["set"])
        elif kind == "add_holding":
            facts["holdings"].append(_expand(op["holding"]))
        elif kind == "drop_holding":
            facts["holdings"] = [h for h in facts["holdings"] if h["holding_id"] != op["id"]]
        elif kind == "add_role":
            facts["roles"].append(_expand(op["role"]))
        elif kind == "drop_role":
            facts["roles"] = [r for r in facts["roles"] if r["role_id"] != op["id"]]
        elif kind == "operations":
            _set(facts["operations"], op["set"])
        elif kind == "corporate_object":
            facts["corporate_object"] = list(op["set"])
        elif kind == "add_conflict":
            facts["conflicts"].append(_expand(op["conflict"]))
        elif kind == "drop_conflicts":
            facts["conflicts"] = []
        elif kind == "unverified":
            facts["unverified"].append(ref(op["ref"], verified=False))
        elif kind == "verification":
            facts["verification"].append(_expand(op["item"]))
        elif kind == "add_candidate":
            c = _expand(op["candidate"])
            c.setdefault("matched_name", c["entry"]["name"])
            c.setdefault("alias_exact", False)
            c.setdefault("identifiers", {"corroborating": [], "conflicting": []})
            c.setdefault("triggers_fuzzy", c["jw_score"] >= 0.85 or c["alias_exact"])
            screening["candidates"].append(c)
        elif kind == "candidate":
            _set(_by(screening["candidates"], "candidate_id", op["id"]), op["set"])
        elif kind == "answer":
            answers[op["id"]] = {
                "instance_id": op["id"],
                "value": op["value"],
                "confidence": 0.9,
                "evidence": [],
                "rationale": "fixture",
            }
        elif kind == "drop_answer":
            answers.pop(op["id"], None)
        else:
            raise ValueError(f"unknown mutation op {kind!r}")


def seal(facts: dict, screening: dict) -> tuple[b4.FactSet, b4.ScreeningResult]:
    """Complete the evidence table for every ref, recompute the hashes, validate both records."""
    have = {e["evidence_id"] for e in facts["evidence"]}
    body = {k: v for k, v in facts.items() if k != "evidence"}
    for r in _refs(body):
        if r["claim_id"] not in have:
            have.add(r["claim_id"])
            facts["evidence"].append(
                {
                    "evidence_id": r["claim_id"],
                    "doc": r["doc"],
                    "document_name": f"{r['doc']}.pdf",
                    "page": r["page"],
                    "bbox": [0, 0, 10, 10],
                    "excerpt": r["claim_id"],
                    "verified": r["verified"],
                }
            )
    facts["evidence"].sort(key=lambda e: e["evidence_id"])
    facts["fact_set_hash"] = sha256_json({k: v for k, v in facts.items() if k != "fact_set_hash"})
    screening["fact_set_hash"] = facts["fact_set_hash"]
    return b4.FactSet.model_validate(facts), b4.ScreeningResult.model_validate(screening)


def golden() -> dict[str, dict]:
    """The golden cases as raw JSON, keyed by packet letter."""
    return {c["packet"]: c for c in load("decisions/golden.json")["cases"]}


def build(base: dict, ops: list[dict] = ()) -> Packet:
    facts, screening = copy.deepcopy(base["facts"]), copy.deepcopy(base["screening"])
    answers = {a["instance_id"]: copy.deepcopy(a) for a in base["answers"]}
    apply(facts, screening, answers, list(ops))
    fs, sr = seal(facts, screening)
    return Packet(fs, sr, {k: b4.PredicateAnswer.model_validate(v) for k, v in answers.items()})


def golden_packet(letter: str) -> Packet:
    """A golden case exactly as recorded: its hashes are the fixture's, not recomputed."""
    g = golden()[letter]
    return Packet(
        b4.FactSet.model_validate(g["facts"]),
        b4.ScreeningResult.model_validate(g["screening"]),
        {a["instance_id"]: b4.PredicateAnswer.model_validate(a) for a in g["answers"]},
    )


def variants(rel: str = "decisions/synthetic.json") -> list[tuple[dict, Packet]]:
    """Every variant of a dataset with its built packet. A variant may name another variant as its
    base (`"base": "syn-a-clean"`), so shared preparation is written once."""
    data = load(rel)
    gold = golden()
    built: dict[str, dict] = {}
    out = []
    for v in data["cases"]:
        base = v["base"]
        if base in gold:
            src = gold[base]
            prior: list[dict] = []
        else:
            src = gold[built[base]["root"]]
            prior = built[base]["ops"]
        ops = [*prior, *v.get("ops", [])]
        built[v["id"]] = {"root": base if base in gold else built[base]["root"], "ops": ops}
        out.append((v, build(src, ops)))
    return out
