"""`norm-v2`: the one comparison every quote check uses (spec).

NFKC, casefold, typographic quotes and dashes folded, whitespace collapsed. Containment is exact
after normalisation: no fuzzy acceptance, so digits, dates and identifiers must match character
for character.

v2 (L1) folds two text-layer artifacts that are not content, on both sides alike:
- a lone separator mark between spaces (`·`, `.`, `•`, `|`): a layer's `A · B` is a scan's `A . B`;
- a `%` glued to the next cell's word (`46%Transmisión`): a layer that drops the column gap.
"""

from __future__ import annotations

import re
import unicodedata

VERSION = "norm-v2"
_FOLD = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "−": "-",
                       " ": " "})  # fmt: skip
_WS = re.compile(r"\s+")
# a lone mark between two numbers stays: "1 . 250" is not "1 250"
_LONE_MARK = re.compile(r"(?:(?<=\s)|^)(?<!\d\s)[.·•|](?=\s|$)(?!\s\d)")
_GLUED_PCT = re.compile(r"%(?=[^\W\d_])")


def norm(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).casefold().translate(_FOLD)
    t = _GLUED_PCT.sub("% ", _WS.sub(" ", t))
    return _WS.sub(" ", _LONE_MARK.sub(" · ", t)).strip()


def contains(haystack: str, needle: str) -> bool:
    """Exact containment after `norm`, on token boundaries: "46%" is not found inside "146%"."""
    n, h = norm(needle), norm(haystack)
    if not n:
        return False
    # a number is not cut from a larger one, decimals and digit grouping included ("150.46%")
    start = r"(?<!\w)(?<!\d[.,])" if n[0].isalnum() else ""
    end = r"(?!\w)(?![.,]\d)" if n[-1].isalnum() else ""
    return re.search(start + re.escape(n) + end, h) is not None


_GAP = re.compile(r"\S+(?: \S+)*")  # a layer cell: words joined by single spaces


def _cells(line: str) -> list[tuple[int, str]]:
    return [(m.start(), m.group()) for m in _GAP.finditer(line)]


def contains_wrapped(haystack: str, needle: str, layer: str) -> bool:
    """`contains`, or the needle with the text layer's wrapped cell continuations put back.

    A layer table row whose cell wraps ("Lucas Navarro   2.300   46%" over "Solís") is read
    from the pixels cell by cell ("Lucas Navarro Solís 2.300 46%"). The continuation is taken
    only from the layer line right below the needle's own line, and only for the needle's own
    cells, by column; the pixels must then contain the rebuilt row exactly.
    """
    if contains(haystack, needle):
        return True
    n = needle.strip()
    lines = layer.splitlines()
    for i, line in enumerate(lines[:-1]):
        at = line.find(n)
        if at < 0:
            continue
        cells = [(at + o, c) for o, c in _cells(n)]
        starts = [o for o, _ in cells]
        rebuilt = dict(cells)
        for o, c in _cells(lines[i + 1]):
            if at <= o < at + len(n):  # only a continuation of one of the needle's own cells
                col = max(s for s in starts if s <= o)
                rebuilt[col] = f"{rebuilt[col]} {c}"
        if rebuilt != dict(cells) and contains(haystack, " ".join(rebuilt[o] for o in starts)):
            return True
    return False


_MONTHS = {m: i for i, ms in enumerate([
    ("enero", "january", "jan", "ene"), ("febrero", "february", "feb"), ("marzo", "march", "mar"),
    ("abril", "april", "apr", "abr"), ("mayo", "may"), ("junio", "june", "jun"),
    ("julio", "july", "jul"), ("agosto", "august", "aug", "ago"),
    ("septiembre", "setiembre", "september", "sep", "sept"), ("octubre", "october", "oct"),
    ("noviembre", "november", "nov"), ("diciembre", "december", "dec", "dic"),
], start=1) for m in ms}  # fmt: skip


def parse_date(text: str) -> str | None:
    """An ISO date from the supported forms; None for anything else (never a guess).

    Supported: ISO `YYYY-MM-DD`; `D/M/YYYY` (the LATAM default) unless the text carries an explicit
    `MM/DD/YYYY` label; `19 de agosto de 2026` / `15 August 2026`; `August 15, 2026`.
    """
    t = norm(text)
    us = "mm/dd/yyyy" in t
    if m := re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", t):
        y, mo, d = int(m[1]), int(m[2]), int(m[3])
    elif m := re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", t):
        a, b, y = int(m[1]), int(m[2]), int(m[3])
        mo, d = (a, b) if us else (b, a)
    elif m := re.search(r"\b(\d{1,2}) (?:de )?([a-z]+)\.? (?:de |del )?(\d{4})\b", t):
        d, mo, y = int(m[1]), _MONTHS.get(m[2], 0), int(m[3])
    elif m := re.search(r"\b([a-z]+)\.? (\d{1,2}),? (\d{4})\b", t):
        mo, d, y = _MONTHS.get(m[1], 0), int(m[2]), int(m[3])
    else:
        return None
    if not (1 <= mo <= 12 and 1 <= d <= 31):
        return None
    return f"{y:04d}-{mo:02d}-{d:02d}"
