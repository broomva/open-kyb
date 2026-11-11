"""Name normalisers and Jaro-Winkler for screening (spec, spec).

Two roles, kept apart on purpose:
- **The policy's trigger (spec):** `d044` (lowercase, accents stripped, punctuation to spaces,
  whitespace collapsed) compared by `jaro_winkler` (prefix scale 0.1, max prefix 4). Only this
  score is recorded as `jw_score` and only it decides `triggers_fuzzy`. After the punctuation
  pass, runs of single letters are joined so dotted and spaced legal-form abbreviations
  ("S.L." vs "SA DE CV") collapse to one canonical string.
- **Recall (spec):** `variants` adds normalisers for transliteration, legal-form suffixes, token
  order, spacing and spelling, so a probable match the policy's comparison scores low still
  becomes a candidate for the `same_party` predicate. Recall can add candidates, never triggers.
"""

from __future__ import annotations

import re

from kyb.facts.text import SUFFIX_TOKENS, base_norm, fold, merged_tokens, org_key
from kyb.witness.verify import subject_match

# Cyrillic and Greek to Latin (ISO 9 / ELOT, simplified): enough that a list name written in
# another script normalises to the spelling the packets use. Unmapped characters pass through.
_TRANSLIT = str.maketrans({
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh", "з": "z",
    "и": "i", "й": "i", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r",
    "с": "s", "т": "t", "у": "u", "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh",
    "щ": "shch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya", "і": "i", "ї": "i",
    "є": "ye", "ґ": "g",
    "α": "a", "β": "v", "γ": "g", "δ": "d", "ε": "e", "ζ": "z", "η": "i", "θ": "th", "ι": "i",
    "κ": "k", "λ": "l", "μ": "m", "ν": "n", "ξ": "x", "ο": "o", "π": "p", "ρ": "r", "σ": "s",
    "ς": "s", "τ": "t", "υ": "y", "φ": "f", "χ": "ch", "ψ": "ps", "ω": "o",
})  # fmt: skip


def transliterate(text: str) -> str:
    """Accents off first (so "Ά" is "α" before it becomes "a"), then Cyrillic/Greek to Latin."""
    return fold(str(text)).translate(_TRANSLIT)


NORMALISATION = "d054"


def d044(name: str) -> str:
    """The policy's normalisation: spec as amended by spec (abbreviation letters joined,
    "s a de c v" → "sa de cv")."""
    return " ".join(merged_tokens(transliterate(name)))


def d044_literal(name: str) -> str:
    """spec before spec (no joining): kept as a recall normaliser."""
    return base_norm(transliterate(name))


def jaro(a: str, b: str) -> float:
    if a == b:
        return 1.0 if a else 0.0
    if not a or not b:
        return 0.0
    window = max(max(len(a), len(b)) // 2 - 1, 0)
    used_b = [False] * len(b)
    match_a = []
    for i, ch in enumerate(a):
        lo, hi = max(0, i - window), min(len(b), i + window + 1)
        for j in range(lo, hi):
            if not used_b[j] and b[j] == ch:
                used_b[j] = True
                match_a.append(ch)
                break
    m = len(match_a)
    if not m:
        return 0.0
    match_b = [b[j] for j in range(len(b)) if used_b[j]]
    transpositions = sum(x != y for x, y in zip(match_a, match_b, strict=True)) / 2
    return (m / len(a) + m / len(b) + (m - transpositions) / m) / 3


def jaro_winkler(a: str, b: str, *, prefix_scale: float = 0.1, max_prefix: int = 4) -> float:
    j = jaro(a, b)
    prefix = 0
    for x, y in zip(a[:max_prefix], b[:max_prefix], strict=False):
        if x != y:
            break
        prefix += 1
    return j + prefix * prefix_scale * (1 - j)


def policy_score(a: str, b: str) -> float:
    """spec's score for one pair of names, rounded so the recorded value is stable."""
    return round(jaro_winkler(d044(a), d044(b)), 6)


def _phonetic_token(tok: str) -> str:
    """A spelling skeleton: common transliteration pairs folded, doubles collapsed, vowels after
    the first letter dropped ("Tariq" and "Tareq" → "trq"; "Volkoff" and "Volkov" →
    "ptrf")."""
    t = tok
    for a, b in (("ph", "f"), ("ck", "k"), ("kh", "h"), ("sch", "s"), ("sh", "s"), ("th", "t"),
                 ("dh", "d"), ("ou", "u"), ("oo", "u"), ("ee", "i"), ("q", "k"), ("w", "v"),
                 ("v", "f"), ("y", "i"), ("z", "s"), ("x", "ks")):  # fmt: skip
        t = t.replace(a, b)
    t = re.sub(r"c(?=[aou])", "k", t)
    t = re.sub(r"(.)\1+", r"\1", t)
    return t[:1] + re.sub(r"[aeiou]", "", t[1:]) if t else t


def variants(name: str) -> dict[str, str]:
    """Every recall normaliser's form of a name."""
    base = d044(name)
    stripped = org_key(transliterate(name)) or base
    # out of order, a legal form can sit anywhere ("Grupo S.A. Fluvial"): drop it wherever it is
    unordered = sorted(t for t in stripped.split() if t not in SUFFIX_TOKENS) or stripped.split()
    return {
        "d044": base,
        "d044_literal": d044_literal(name),
        "suffix": stripped,
        "token_order": " ".join(unordered),
        "compact": stripped.replace(" ", ""),
        "phonetic": " ".join(_phonetic_token(t) for t in stripped.split()),
        "phonetic_order": " ".join(sorted(_phonetic_token(t) for t in stripped.split())),
    }


def contains_tokens(shorter: str, longer: str) -> bool:
    """Every token of one name (two or more, initials allowed) appears in order in the other:
    "Valeria Ortega" in "Valeria Ortega Méndez", "S. Paredes" in "Sebastián Paredes Vega"."""
    a, b = d044(shorter), d044(longer)
    if len(a.split()) < 2 or len(a.split()) > len(b.split()):
        return False
    return subject_match(a, b)


def recall_score(a: str, b: str) -> float:
    """The best score any recall normaliser gives the pair (1.0 for token containment)."""
    if contains_tokens(a, b) or contains_tokens(b, a):
        return 1.0
    va, vb = variants(a), variants(b)
    return max(jaro_winkler(va[k], vb[k]) for k in va if va[k] and vb[k])
