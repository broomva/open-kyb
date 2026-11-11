"""The labelled crop set (`datasets/witness/labels.json`): hand-checked crops, scored per backend.

A crop is **read correctly** when its response transcribes every labelled token, marks the
labelled struck values struck and the handwriting as handwriting, states none of the values that
are not on it (another party's value, a swapped one), and states no re-read value it did not read
itself (an invented one). That is the witness's own quality, independent of the comparator.

Binding is reported apart: each probe is a claim judged by the production comparator against the
response. A positive probe that does not bind is a lost verification (often the witness-v3 schema's
single value slot, not a misreading). A negative probe that binds is a **false accept**: the number
that matters most, because it is how another person's value would become a fact.

Crops are requested per source document, in label order, with opaque positional ids (`k1`, …),
exactly as `verify_document` sends them.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kyb.evals.paths import PACKETS, datasets, load
from kyb.evals.witness import comparator
from kyb.util.hashing import sha256_bytes
from kyb.witness import norm

GATE_CORRECT = 0.9  # the eval's own pass line: used to prove a faulty witness fails it


def labels() -> dict[str, Any]:
    return load("witness/labels.json")


def crop_path(crop_id: str) -> Path:
    return datasets() / "witness" / "crops" / f"{crop_id}.png"


def crop_bytes(c: dict) -> bytes:
    data = crop_path(c["id"]).read_bytes()
    if sha256_bytes(data) != c["png_sha256"]:
        raise ValueError(f"{c['id']}: crop bytes differ from the label's png_sha256")
    return data


def build_crops() -> list[str]:
    """Re-render every crop from its packet PDF at 300 dpi and pin its hash in the labels."""
    import json

    from kyb.docintel import poppler

    data = labels()
    out = []
    for c in data["crops"]:
        pdf = PACKETS / c["packet"] / c["document"]
        path = crop_path(c["id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        poppler.render_region(pdf, c["page"], tuple(c["bbox"]), path)
        c["png_sha256"] = sha256_bytes(path.read_bytes())
        out.append(c["id"])
    (datasets() / "witness" / "labels.json").write_text(
        json.dumps(data, indent=1, ensure_ascii=False) + "\n", "utf-8"
    )
    return out


def groups(data: dict) -> list[list[dict]]:
    """Crops grouped by source document, in label order: one witness request per group."""
    by: dict[tuple[str, str], list[dict]] = {}
    for c in data["crops"]:
        by.setdefault((c["packet"], c["document"]), []).append(c)
    return list(by.values())


def witness_all(recorder, data: dict) -> dict[str, dict]:
    """Ask the recorder for every group; return each label id's response crop."""
    out: dict[str, dict] = {}
    for group in groups(data):
        ids = [f"k{i}" for i in range(1, len(group) + 1)]
        reply = recorder.ask("transcribe", [crop_bytes(c) for c in group], {"crop_ids": ids})
        got = {r.get("crop_id"): r for r in reply.data.get("crops", []) if isinstance(r, dict)}
        for k, c in zip(ids, group, strict=True):
            out[c["id"]] = got.get(k, {})
    return out


def oracle(data: dict) -> dict[str, dict]:
    """The labels in response form: a perfect witness-v3 answer per crop."""
    return {c["id"]: {"crop_id": c["id"], "readable": True, **c["oracle"]} for c in data["crops"]}


def _has_struck(crop: dict, value: str) -> bool:
    return any(
        norm.norm(value) == norm.norm(s) or norm.contains(s, value)
        for s in comparator.struck_values(crop)
    )


_HAND = ("handwrit", "manuscrit", "a mano", "hand-writ", "ink", "tinta")


def _annotated(crop: dict, value: str) -> bool:
    """The handwritten value is observed and marked: an annotation carries the value itself, or
    the value is in the response (transcription or re-read) and an annotation says handwriting.
    A bare "handwritten" note with the value nowhere is not an observation."""
    notes = [str(a) for a in (crop.get("reread") or {}).get("annotations") or []]
    if any(norm.contains(a, value) for a in notes):
        return True
    shown = "\n".join([crop.get("transcription") or "", *_reread_values(crop)])
    return norm.contains(shown, value) and any(h in norm.norm(a) for a in notes for h in _HAND)


def _reread_values(crop: dict) -> list[str]:
    rr = crop.get("reread") or {}
    vals = [str(rr["value"])] if rr.get("value") else []
    return vals + [str(t.get("value")) for t in rr.get("values") or [] if isinstance(t, dict)]


# a value's tokens: words, numbers and identifiers as written, separators inside kept
# (3184920, PY-0833921, 1.250, 50,000, 64,00%, 1971-10-15, 3/8)
# a percentage keeps its % (46% is not the 18 of "Lot 18"), a number its sign (-46% ≠ 46%)
_TOKEN = re.compile(r"(?:(?<![0-9a-z])[-+](?=\d))?[0-9a-z]+(?:[.,/\-][0-9a-z]+)*%?")
# words that describe a reading rather than state a value; a re-read may use them freely
COMMENTARY = frozenset(
    {
        "handwritten",
        "handwriting",
        "printed",
        "struck",
        "through",
        "correction",
        "value",
        "date",
        "note",
        "manuscrito",
        "manuscrita",
        "anotado",
        "anotada",
        "tachado",
        "tachada",
        "impreso",
        "valor",
        "fecha",
        "nota",
    }
)
# grammar a re-read may add around a value; never a negation, never a value word
GRAMMAR = frozenset(
    {"is", "of", "the", "and", "a", "an", "in", "on", "el", "la", "los", "las", "de", "del", "y",
     "en", "por", "se"}
)  # fmt: skip


def _canon(token: str) -> str:
    """A decimal mark with zeros only is the same number (55,00 = 55.00 = 55); a thousands
    group is not (50,000 is not 50, 256.000 is not 220)."""
    return re.sub(r"[.,]0{1,2}(?=%?$)", "", token)


def _tokens(text: str) -> set[str]:
    return {_canon(t) for t in _TOKEN.findall(norm.norm(text))}


def truth_text(c: dict) -> str:
    """What is on the crop, from its label: the oracle's reading of it."""
    o = c["oracle"]
    rr = o["reread"]
    return " ".join(
        [
            o["transcription"],
            rr.get("subject", ""),
            rr.get("value", ""),
            *rr.get("struck", []),
            *rr.get("annotations", []),
        ]
    )


def _ungrounded(crop: dict, truth: str | None = None) -> list[str]:
    """Re-read values that state something not on the crop. Against the label's truth (or, with
    no label, the response's own reading): every number or identifier token in the value must be
    a whole token of the truth (887 is not 3184920, ZZ-0944999 is not PY-0833921, 50 is not
    50,000, -46% is not 46%), and every word, however short, must be in the truth or be reading
    commentary or grammar ("beneficial owner", "not", "N/A" and "UBO" are not on the crop). A
    witness that states what is not there is confabulating, however close the words."""
    if truth is None:
        rr = crop.get("reread") or {}
        truth = " ".join(
            [
                crop.get("transcription") or "",
                *comparator.struck_values(crop),
                *(str(a) for a in rr.get("annotations") or []),
            ]
        )
    have = _tokens(truth)
    out = []
    for v in _reread_values(crop):
        for t in _tokens(v):
            numeric = any(ch.isdigit() for ch in t)
            # every token counts, short ones included (N/A, UBO, not): only a word on the crop,
            # reading commentary or plain grammar may appear; a number must be on the crop
            if t not in have and (numeric or t not in COMMENTARY | GRAMMAR):
                out.append(v)
                break
    return out


def score_crop(c: dict, resp: dict) -> dict[str, Any]:
    exp, text = c["expect"], resp.get("transcription") or ""
    misses = [f"untranscribed {t!r}" for t in exp["transcribed"] if not norm.contains(text, t)]
    misses += [f"not struck {v!r}" for v in exp["struck"] if not _has_struck(resp, v)]
    misses += [
        f"handwriting not annotated {v!r}" for v in exp["handwritten"] if not _annotated(resp, v)
    ]
    stated = comparator.stated(resp)
    misses += [f"states {v!r}, not on the crop" for v in exp["absent"] if norm.contains(stated, v)]
    misses += [f"re-read {v!r} is not on the crop" for v in _ungrounded(resp, truth_text(c))]
    # binding is the comparator's verdict, reported apart from reading: a witness-v3 answer has
    # one (subject, value) slot, so a correct reading of a many-valued row can still not bind
    probes = []
    for p in c["probes"]:
        ok = comparator.verifies(p["claim"], resp)
        probes.append({"id": p["id"], "polarity": p["polarity"], "verified": ok})
    unbound = [p["id"] for p in probes if p["polarity"] == "positive" and not p["verified"]]
    if not comparator.readable(resp):
        misses.insert(0, "unreadable or missing")
    return {
        "id": c["id"],
        "category": c["category"],
        "correct": not misses,
        "misses": misses,
        "unbound_positives": unbound,
        "probes": probes,
    }


def score(data: dict, responses: dict[str, dict]) -> dict[str, Any]:
    per = [score_crop(c, responses.get(c["id"], {})) for c in data["crops"]]
    probes = [p for r in per for p in r["probes"]]
    pos = [p for p in probes if p["polarity"] == "positive"]
    neg = [p for p in probes if p["polarity"] == "negative"]
    fa = [p["id"] for p in neg if p["verified"]]
    correct = sum(r["correct"] for r in per)
    return {
        "crops": len(per),
        "correct": correct,
        "correct_value_rate": round(correct / len(per), 4) if per else None,
        "positives": len(pos),
        "positives_verified": sum(p["verified"] for p in pos),
        "negatives": len(neg),
        "false_accepts": len(fa),
        "false_accept_ids": fa,
        "gate": gate(correct / len(per) if per else 0.0, len(fa)),
        "per_crop": per,
    }


def gate(correct_rate: float, false_accepts: int) -> bool:
    return false_accepts == 0 and correct_rate >= GATE_CORRECT


def verdicts(scored: dict) -> dict[str, bool]:
    return {p["id"]: p["verified"] for r in scored["per_crop"] for p in r["probes"]}


def transform(responses: dict[str, dict], fn: Callable[[dict], dict]) -> dict[str, dict]:
    return {k: fn(v) for k, v in responses.items()}
