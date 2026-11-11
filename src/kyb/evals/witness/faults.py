"""Fault-injected witnesses: the eval's check on itself. A witness that swaps digits, invents a
value, or ignores strike-through must make the eval fail; an eval that passes any of them is
wrong.

Each fault rewrites a response crop (witness-v3 or v4 shape) the way that failure would.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable


def _swap(text: str) -> str:
    """In every run of digits, swap the first adjacent pair that differs: 3184920 → 1241887."""

    def one(m: re.Match[str]) -> str:
        s = m.group(0)
        for i in range(len(s) - 1):
            if s[i] != s[i + 1]:
                return s[:i] + s[i + 1] + s[i] + s[i + 2 :]
        return s

    return re.sub(r"\d{2,}", one, text)


def _invented(value: str) -> str:
    v = str(value)
    if "%" in v:
        return "37%"
    if re.fullmatch(r"\s*\d{4}-\d{2}-\d{2}\s*", v):
        return "1999-12-31"
    if re.search(r"\d", v):
        return re.sub(r"\d", "7", v) if set(re.sub(r"\D", "", v)) != {"7"} else "4242"
    return "Lorem Ipsum Dolor"


def _map_values(crop: dict, fn: Callable[[str], str], text_too: bool) -> dict:
    out = copy.deepcopy(crop)
    if text_too and "transcription" in out:
        out["transcription"] = fn(out["transcription"])
    rr = out.get("reread") or {}
    if "value" in rr:
        rr["value"] = fn(str(rr["value"]))
    for t in rr.get("values") or []:
        if isinstance(t, dict) and "value" in t:
            t["value"] = fn(str(t["value"]))
    if text_too:
        rr["struck"] = [fn(str(s)) for s in rr.get("struck") or []]
        rr["annotations"] = [fn(str(a)) for a in rr.get("annotations") or []]
    return out


def swap_digits(crop: dict) -> dict:
    """A misread: every number on the crop has two digits transposed, everywhere it is stated."""
    return _map_values(crop, _swap, text_too=True)


def invent_value(crop: dict) -> dict:
    """A confabulation: the re-read states values the crop does not show."""
    return _map_values(crop, _invented, text_too=False)


def ignore_strike(crop: dict) -> dict:
    """Strike-through unseen: a struck value is read as the current one, and nothing is struck."""
    out = copy.deepcopy(crop)
    rr = out.get("reread") or {}
    struck = list(rr.get("struck") or [])
    struck += [
        t["value"] for t in rr.get("values") or [] if isinstance(t, dict) and t.get("struck")
    ]
    if struck:
        rr["value"] = struck[0]
    rr["struck"] = []
    for t in rr.get("values") or []:
        if isinstance(t, dict):
            t["struck"] = False
    return out


FAULTS: dict[str, Callable[[dict], dict]] = {
    "swap-digits": swap_digits,
    "invent-value": invent_value,
    "ignore-strike": ignore_strike,
}
