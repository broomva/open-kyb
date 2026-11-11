"""Sub-national regions and offshore free zones, read from `Entity.regions`.

The policy qualifies some list entries: "the Crimea/Donetsk/Luhansk regions of `UA`", "`AE` free
zones". The parser keeps each qualifier as written (`prohibited_regions`, `high_risk_regions`:
code → qualifier); the region names come from that text, never from here. This module only knows
how documents spell them, and how they spell what lies outside them.

A stated region reads:
- `yes` when it names what the policy lists (Crimea, a free zone) and nothing outside it;
- `no` only when it *recognisably* names somewhere else (Kyiv; "mainland") and nothing listed;
- `unknown` otherwise (MANUAL_REVIEW): unrecognised text ("Donets'k" misspelt, "N/A"), both kinds
  at once, a negation or exclusion cue ("Kyiv (not Donetsk)"), or a non-Latin script. An entity
  in that country with no stated region is `unknown` too. Text the lexicon does not know never
  clears a rule, and a negated mention never fires one. Words surrounding a matched name must
  fit a full affirmative form; unsupported wording, questions and raw status marks remain unknown.

A qualifier whose names have no lexicon entry cannot be matched; the parser reports it as
unrepresentable (`unmatched`), so the rule is `unknown` instead of silently never firing.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

from kyb.policy.text import fold, matches

# Spellings per policy name, matched with `text.matches` on folded text (the name itself included).
PLACES: dict[str, tuple[str, ...]] = {
    "crimea": ("crimea*", "krym*", "krim", "krimea", "crimee", "sevastopol*", "sebastopol*",
               "simferopol*", "yalta", "kerch", "yevpatori*", "evpatori*", "feodosi*"),
    "donetsk": ("donetsk*", "donetska", "donezk", "donieck", "donbas*", "mariupol*",
                "kramatorsk*", "sloviansk*", "slavyansk*", "horlivk*", "gorlovk*", "makiivk*",
                "makeevk*", "bakhmut*"),
    "luhansk": ("luhansk*", "luhanska", "lugansk*", "luhanshchyna", "donbas*",
                "sievierodonetsk*", "severodonetsk*", "lysychansk*", "alchevsk*", "stakhanov*"),
    "zaporizhzhia": ("zaporizh*", "zaporozh*", "melitopol*", "berdiansk*", "enerhodar*"),
    "kherson": ("kherson*", "kakhovk*", "skadovsk*", "henichesk*"),
}  # fmt: skip
CATEGORIES: dict[str, tuple[str, ...]] = {
    "free zones": ("free zone", "free zones", "freezone", "freezones", "free trade zone",
                   "free trade zones", "zona franca", "zonas francas",
                   "dmcc", "jafz", "jafza", "dafz", "dafza", "rakez", "ifza", "ajman free zone",
                   "uaq ftz", "saif zone", "hamriyah", "kizad", "adgm",
                   "difc", "dubai silicon oasis", "dubai internet city", "dubai media city",
                   "masdar city", "twofour54", "jlt", "jumeirah lakes towers"),
}  # fmt: skip
# What recognisably lies outside: for a closed list, the country's other regions (keyed by code);
# for an open category, statements that a place is not in it. A name the policy lists always wins
# over this table, so listing Kherson tomorrow cannot be cleared by it.
OUTSIDE: dict[str, tuple[str, ...]] = {
    "UA": ("kyiv", "kiev", "kyyiv", "lviv", "lvov", "lwow", "kharkiv*", "kharkov*", "odesa*",
           "odessa*", "dnipro*", "dnepr*", "zaporizh*", "zaporozh*", "kherson*", "mykolai*",
           "nikolaev*", "poltav*", "sumy", "sumska", "chernihiv*", "chernigov*", "zhytomyr*",
           "vinnyts*", "vinnits*", "khmelnyts*", "ternopil*", "ivano frankivsk*", "rivne*",
           "volyn*", "lutsk", "zakarpat*", "uzhhorod", "chernivts*", "cherkas*", "kirovohrad*",
           "kropyvnyts*", "bila tserkva", "kryvyi rih"),
    "free zones": ("mainland", "onshore"),
}  # fmt: skip
# Raw input is tokenized before any folding. Only explicitly grammatical separators are
# accepted; symbols, brackets, slashes and other marks never disappear into a confident result.
_APOSTROPHES = re.compile(r"[\'‘’‛ʼʻʹˈ`´′＇]")
_TOKENS = re.compile(r"[^\W_]+(?:'[^\W_]+)*|[(),.:-]")
_PUNCTUATION = frozenset("(),.:-")
# Former wildcard stems have only bounded spelling endings in this grammar. Arbitrary text
# appended to a place name is not a transliteration (e.g. DonetskUnverified stays unknown).
_ENDINGS = ("", "a", "an", "n", "s", "ia", "iya", "ska", "skaya", "skyi", "sky")
_COUNTRY_LABELS = {
    "UA": ("UA", "Ukraine"),
    "AE": ("AE", "UAE", "United Arab Emirates", "Dubai", "Abu Dhabi", "Sharjah", "Ajman",
           "Fujairah", "Ras Al Khaimah", "Umm Al Quwain"),
}  # fmt: skip

_NAMED = re.compile(r"^(?P<names>.+?)\s+(?:regions?|zones?)\s+of$")


def targets(qualifier: str) -> tuple[list[str], bool]:
    """A qualifier → (the names it lists, folded; whether they are a closed list of places)."""
    q = " ".join(qualifier.lower().split())
    if q in ("free zones of", "free zone of"):
        return ["free zones"], False
    if m := _NAMED.match(q):
        return [fold(n) for n in m.group("names").split("/") if fold(n)], True
    return [fold(q)], False


def unmatched(qualifier: str) -> list[str]:
    """The names in a qualifier the lexicon cannot match (the parser's guard)."""
    names, closed = targets(qualifier)
    lexicon = PLACES if closed else CATEGORIES
    return [n for n in names if n not in lexicon]


def _input(region: str) -> str | None:
    """Consume all raw characters, retaining punctuation as tokens before normalization."""
    for c in region:
        if c in " \t\u00a0" or c in _PUNCTUATION or _APOSTROPHES.fullmatch(c):
            continue
        # Accept ASCII words and ordinary accented Latin letters, never compatibility symbols
        # that NFKC could turn into letters/numbers (unchecked boxes, circled marks, etc.).
        plain = "".join(x for x in unicodedata.normalize("NFD", c) if not unicodedata.combining(x))
        if not (c.isascii() and c.isalnum()) and not (
            unicodedata.category(c) in {"Ll", "Lu"} and plain.isascii() and plain.isalpha()
        ):
            return None
    source = _APOSTROPHES.sub("'", region)
    tokens, end = [], 0
    for m in _TOKENS.finditer(source):
        if source[end : m.start()].strip():
            return None
        token = m.group()
        tokens.append(token if token in _PUNCTUATION else fold(token.replace("'", "")))
        end = m.end()
    if source[end:].strip() or not tokens:
        return None
    return " ".join(tokens)


def _term(term: str) -> str:
    """A configured spelling as a whole lexical form, with explicit internal hyphens allowed."""
    words = fold(term.removesuffix("*")).split()
    if not words:
        raise ValueError("empty region spelling")
    last = words[-1]
    if term.endswith("*"):
        tail = "(?:" + "|".join(re.escape(last + ending) for ending in _ENDINGS) + ")"
    else:
        tail = re.escape(last)
    return r"(?: | - )".join([*(re.escape(w) for w in words[:-1]), tail])


def _alternatives(terms: Iterable[str]) -> str:
    return "(?:" + "|".join(_term(t) for t in terms) + ")"


def _accepted(
    code: str, terms: list[str], closed: bool, names: list[str], text: str, *, target: bool
) -> bool:
    """Anchored positive forms only; every token and punctuation mark must be consumed."""
    if not terms:
        return False
    place = _alternatives(terms)
    if closed:
        place = (
            rf"(?:autonomous republic of )?{place}(?: (?:oblast|region)| \( (?:oblast|region) \))?"
        )
    elif target:
        category = _alternatives([n.removesuffix("s") for n in names] + names)
        place = rf"{place}(?: {category}| \( {category} \))?"
    country = _alternatives(_COUNTRY_LABELS.get(code, (code,)))
    # Named location, optional country/emirate qualifiers; no free-form context is discarded.
    location = (
        rf"(?:{place}(?: , {country}){{0,2}}|{place} \( {country} \)|"
        rf"{country}(?: ,)? {place}|{country} \( {place} \))"
    )
    subject = r"(?:the )?(?:applicant|entity|company|owner|person|ubo|shareholder)"
    predicate = r"(?:registered|incorporated|located|resident|domiciled|based)"
    address = r"(?:the )?(?:(?:registered|operating) (?:address|office)|residence|jurisdiction)"
    forms = [
        location,
        rf"(?:{subject} (?:is|are) )?{predicate} (?:in|at) {location}",
        rf"{address}(?: is (?:in|at)| (?:in|at)| :) {location}",
        rf"{location} is {address}",
    ]
    if not closed:
        # Explicit positive category assertion (e.g. Dubai is a free zone / has free-zone status).
        forms += [rf"{country} is (?:a |an |the )?{place}", rf"{country} has {place} status"]
    return any(re.fullmatch(rf"(?:{form})(?: \.)?", text) for form in forms)


def read(code: str, qualifier: str, region: str) -> str:
    """ "yes" | "no" | "unknown" for one fully recognized affirmative region statement."""
    names, closed = targets(qualifier)
    lexicon = PLACES if closed else CATEGORIES
    if any(n not in lexicon for n in names):
        return "unknown"
    text = _input(region)
    if text is None:
        return "unknown"
    spellings = [t for n in names for t in (n, *lexicon[n])]
    outside = [t for t in OUTSIDE.get(code if closed else names[0], ())
               if not any(matches(t, fold(x.removesuffix("*"))) for x in spellings)]  # fmt: skip
    listed = _accepted(code, spellings, closed, names, text, target=True)
    other = _accepted(code, outside, closed, names, text, target=False)
    if listed != other:
        return "yes" if listed else "no"
    return "unknown"


def classify(code: str, qualifier: str, regions: Iterable[str]) -> str:
    """ "yes" | "no" | "unknown": does any stated region of an entity in `code` fall in what the
    qualifier names? One `yes` decides; otherwise any `unknown` (or none stated) is `unknown`."""
    states = [read(code, qualifier, r) for r in regions]
    if "yes" in states:
        return "yes"
    return "no" if states and set(states) == {"no"} else "unknown"
