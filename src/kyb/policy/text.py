"""Text normalisation shared by the policy parser and the industry predicate.

Two different jobs, kept apart on purpose:
- `cosmetic` folds what a compliance editor can change without changing meaning (emphasis, list
  markers, case, whitespace, arrow and comparison spellings), so a reformatted rule is not a
  reworded one.
- `fold` folds what a document can vary without changing the activity it names (accents,
  punctuation), so "inversión en bienes inmuebles" matches "inversion en bienes inmuebles".
"""

from __future__ import annotations

import re
import unicodedata

_SPELLINGS = (
    ("->", "→"), (">=", "≥"), ("<=", "≤"), ("—", "-"), ("–", "-"),
    ("“", '"'), ("”", '"'), ("‘", "'"), ("’", "'"),
)  # fmt: skip
_LIST_MARKER = re.compile(r"(?m)^\s*(?:[-*+]|\d+\.)\s+")

NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}  # fmt: skip
NUM = r"(?:\d+|" + "|".join(NUMBER_WORDS) + r")"


def number(token: str) -> int:
    t = token.strip().lower()
    return NUMBER_WORDS[t] if t in NUMBER_WORDS else int(t)


def cosmetic(text: str) -> str:
    s = unicodedata.normalize("NFKC", text)
    for a, b in _SPELLINGS:
        s = s.replace(a, b)
    s = _LIST_MARKER.sub("", s)
    s = s.replace("*", "").replace("_", "").replace("`", "")
    return re.sub(r"\s+", " ", s).strip().lower()


def fold(text: str) -> str:
    s = unicodedata.normalize("NFKD", text)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", s)).strip()


# Cues that negate or exclude what a phrase names. Folded text; shared by the industry and region
# readers, which add their own (imitation; proximity and former status).
NEGATIONS = ("not", "no", "never", "nor", "without", "except", "excluding", "excluded",
             "other than", "prohibited from", "ni", "nunca", "sin", "excepto", "salvo", "excluye",
             "excluyendo", "con exclusion de", "distinto de", "distintas de",
             "distintos de")  # fmt: skip

# A lexicon term "a & b" names its category only when both parts occur within this many words of
# each other ("funding of export cargoes"), so a deed listing "exportación de
# mercancías" in one breath and "obtener financiamiento" in another does not read as trade finance.
NEAR = 3


def _spans(term: str, words: list[str]) -> list[tuple[int, int]]:
    """Where `term` occurs in folded `words`; a trailing "*" matches any suffix of its last word."""
    wild = term.endswith("*")
    stem = fold(term.removesuffix("*")).split()
    n = len(stem)
    if not n:
        return []
    out = []
    for i in range(len(words) - n + 1):
        last = words[i + n - 1]
        if words[i : i + n - 1] == stem[:-1] and (
            last.startswith(stem[-1]) if wild else last == stem[-1]
        ):
            out.append((i, i + n))
    return out


def matches(term: str, text: str) -> bool:
    """Does a lexicon term occur in `fold`ed text, at word boundaries? "a & b": both, near;
    "a|b": either (inside a part)."""
    words = text.split()
    parts = [p.strip() for p in term.split("&")]
    if len(parts) > 2:
        raise ValueError(f"a lexicon term joins at most two parts: {term!r}")
    spans = [[sp for alt in part.split("|") for sp in _spans(alt.strip(), words)] for part in parts]
    if not all(spans):
        return False
    if len(spans) == 1:
        return True
    return any(max(b0 - a1, a0 - b1) <= NEAR for a0, a1 in spans[0] for b0, b1 in spans[1])
