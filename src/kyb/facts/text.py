"""Text folding shared by facts and screening: accents, organisation names, countries, addresses.

Pure functions, no model and no I/O. Everything here only *compares* or *labels* values that a
verified claim already carries; nothing here creates a value the documents did not state.
"""

from __future__ import annotations

import re
import unicodedata

# Latin letters NFKD does not decompose. The spec normalisation calls for unidecode, which is
# not a dependency (spec keeps the runtime set small); for Latin script these folds plus NFKD
# give the same ASCII. Other scripts are handled by `screening.normalize.transliterate`.
_SPECIAL = str.maketrans(
    {
        "ß": "ss",
        "ø": "o",
        "Ø": "O",
        "æ": "ae",
        "Æ": "AE",
        "œ": "oe",
        "Œ": "OE",
        "đ": "d",
        "Đ": "D",
        "ł": "l",
        "Ł": "L",
        "þ": "th",
        "Þ": "TH",
        "ð": "d",
        "Ð": "D",
        "ı": "i",
        "ŀ": "l",
    }
)


def fold(text: str) -> str:
    """Lowercase, accents stripped."""
    t = unicodedata.normalize("NFKD", str(text).translate(_SPECIAL))
    return "".join(ch for ch in t if not unicodedata.combining(ch)).casefold()


def base_norm(text: str) -> str:
    """spec: lowercase, strip accents, punctuation to spaces, collapse whitespace."""
    return " ".join(re.sub(r"[\W_]+", " ", fold(text)).split())


def akey(attribute: str | None) -> str:
    """A label as a snake_case key: "Fecha de expedición" → fecha_de_expedicion."""
    return re.sub(r"[^a-z0-9]+", "_", fold(attribute or "")).strip("_")


# --- organisations -------------------------------------------------------------------------


def merged_tokens(text: str) -> list[str]:
    """Base tokens with runs of single letters joined: "S.L." → ["sa", "de", "cv"]."""
    out: list[str] = []
    run = ""
    for tok in base_norm(text).split():
        if len(tok) == 1 and tok.isalpha():
            run += tok
            continue
        if run:
            out.append(run)
            run = ""
        out.append(tok)
    if run:
        out.append(run)
    return out


# Legal-form suffixes stripped from the end of an organisation's name, longest phrase first.
SUFFIXES: tuple[tuple[str, ...], ...] = tuple(
    sorted(
        {
            ("sa", "de", "cv"),
            ("sapi", "de", "cv"),
            ("s", "de", "rl"),
            ("free", "zone", "establishment"),
            ("free", "zone", "est"),
            ("free", "zone", "company"),
            ("fz", "llc"),
            ("fz", "co"),
            ("sa",),
            ("sas",),
            ("saeca",),
            ("sae",),
            ("cv",),
            ("ltd",),
            ("ltda",),
            ("limited",),
            ("llc",),
            ("inc",),
            ("incorporated",),
            ("corp",),
            ("corporation",),
            ("co",),
            ("company",),
            ("gmbh",),
            ("ag",),
            ("bv",),
            ("nv",),
            ("plc",),
            ("fze",),
            ("fzco",),
            ("fzc",),
            ("est",),
            ("establishment",),
            ("srl",),
            ("spa",),
            ("lp",),
            ("llp",),
            ("pte",),
            ("pty",),
            ("sarl",),
        },
        key=lambda p: (-len(p), p),
    )
)
SUFFIX_TOKENS = {t for p in SUFFIXES for t in p}
ORG_MARKERS = frozenset(
    {
        "sa",
        "sl",
        "sas",
        "saeca",
        "sae",
        "cv",
        "ltd",
        "ltda",
        "limited",
        "llc",
        "inc",
        "corp",
        "corporation",
        "gmbh",
        "fze",
        "fzco",
        "plc",
        "srl",
        "holdings",
        "holding",
        "group",
        "grupo",
        "inversiones",
        "trading",
        "services",
        "servicios",
        "company",
        "bank",
        "banco",
        "foundation",
        "fundacion",
        "trust",
        "fideicomiso",
        "establishment",
        "bv",
        "nv",
        "ag",
        "investments",
        "capital",
        "ventures",
        "partners",
        "logistica",
        "inmobiliaria",
        "sociedad",
        "compania",
        "spa",
        "sarl",
    }
)
FOUNDATION_MARKERS = frozenset({"foundation", "fundacion", "stiftung", "fondation"})
TRUST_MARKERS = frozenset({"trust", "fideicomiso"})


def strip_suffixes(tokens: list[str]) -> list[str]:
    toks = list(tokens)
    changed = True
    while changed and len(toks) > 1:
        changed = False
        for phrase in SUFFIXES:
            n = len(phrase)
            if len(toks) > n and tuple(toks[-n:]) == phrase:
                toks = toks[:-n]
                changed = True
                break
    return toks


def is_org(name: str) -> bool:
    return any(t in ORG_MARKERS for t in merged_tokens(name))


def org_key(name: str) -> str:
    """An organisation's identity key: folded, legal-form suffixes and a leading "the" dropped."""
    toks = merged_tokens(name)
    if toks and toks[0] == "the":
        toks = toks[1:]
    return " ".join(strip_suffixes(toks))


def significant_tokens(name: str) -> list[str]:
    """Tokens that can identify a party in free text (no legal forms, no generic words)."""
    generic = SUFFIX_TOKENS | {
        "de",
        "del",
        "la",
        "las",
        "los",
        "el",
        "y",
        "and",
        "of",
        "the",
        "s",
        "a",
        "grupo",
        "group",
        "holdings",
        "holding",
        "inversiones",
        "services",
        "servicios",
        "trading",
        "corporate",
        "company",
    }
    return [t for t in base_norm(name).split() if len(t) >= 4 and t not in generic]


# --- countries -----------------------------------------------------------------------------

# Names and demonyms (es/en) for the countries the policy lists and the packets' region. A value no
# entry matches yields no country: the fact is left empty rather than guessed.
_COUNTRIES: dict[str, tuple[str, ...]] = {
    "MX": ("mexico", "mexicano", "mexicana", "mexican", "cdmx", "ciudad de mexico"),
    "CO": ("colombia", "colombiano", "colombiana", "colombian", "bogota"),
    "PY": ("paraguay", "paraguayo", "paraguaya", "paraguayan", "asuncion"),
    "WS": ("samoa", "samoano", "samoana", "samoan", "apia"),
    "SG": ("singapore", "singapur", "singaporean"),
    "KE": ("kenya", "kenia", "kenyan", "nairobi"),
    "VG": (
        "british virgin islands",
        "virgin islands",
        "bvi",
        "islas virgenes britanicas",
        "tortola",
        "road town",
    ),
    "KY": ("cayman islands", "islas caiman", "caymanian", "cayman"),
    "BZ": ("belize", "belice", "beliceno", "belicena", "belizean", "belmopan"),
    "VE": ("venezuela", "venezolano", "venezolana", "venezuelan"),
    "AE": (
        "united arab emirates",
        "emiratos arabes unidos",
        "uae",
        "emirati",
        "dubai",
        "abu dhabi",
    ),
    "SC": ("seychelles",),
    "MH": ("marshall islands", "islas marshall"),
    "KP": ("north korea", "corea del norte", "dprk", "democratic people s republic of korea"),
    "IR": ("iran", "irani", "iranian"),
    "SY": ("syria", "siria", "syrian", "sirio", "siria"),
    "CU": ("cuba", "cubano", "cubana", "cuban"),
    "UA": (
        "ukraine",
        "ucrania",
        "ukrainian",
        "ucraniano",
        "ucraniana",
        "crimea",
        "donetsk",
        "luhansk",
    ),
    "RU": ("russia", "rusia", "russian", "ruso", "rusa", "russian federation"),
    "ES": ("spain", "espana", "spanish", "espanol", "espanola"),
    "US": ("united states", "estados unidos", "usa", "american", "estadounidense"),
    "AR": ("argentina", "argentino", "argentine"),
    "BR": ("brazil", "brasil", "brazilian", "brasileno", "brasilena"),
    "CL": ("chile", "chileno", "chilena", "chilean"),
    "PE": ("peru", "peruano", "peruana", "peruvian"),
    "EC": ("ecuador", "ecuatoriano", "ecuatoriana", "ecuadorian"),
    "GB": ("united kingdom", "reino unido", "british", "britanico", "britanica", "england"),
    "CR": ("costa rica", "costarricense"),
    "GT": ("guatemala", "guatemalteco", "guatemalteca"),
    "UY": ("uruguay", "uruguayo", "uruguaya", "uruguayan"),
    "BO": ("bolivia", "boliviano", "boliviana", "bolivian"),
    "DO": ("dominican republic", "republica dominicana", "dominicano", "dominicana"),
    "SV": ("el salvador", "salvadoreno", "salvadorena"),
    "HN": ("honduras", "hondureno", "hondurena"),
    "NI": ("nicaragua", "nicaraguense"),
    "CA": ("canada", "canadian", "canadiense"),
    "DE": ("germany", "alemania", "german", "aleman", "alemana"),
    "FR": ("france", "francia", "french", "frances", "francesa"),
    "IT": ("italy", "italia", "italian", "italiano", "italiana"),
    "CH": ("switzerland", "suiza", "swiss", "suizo"),
    "NL": ("netherlands", "paises bajos", "holanda", "dutch"),
    "CN": ("china", "chinese", "chino", "china popular"),
}
_COUNTRY_PATTERNS = sorted(
    ((name, code) for code, names in _COUNTRIES.items() for name in names),
    key=lambda nc: (-len(nc[0]), nc[0]),
)
ISO2 = frozenset(_COUNTRIES)


def countries(text: str) -> list[str]:
    """Every country a value names, in order of first mention: "Belmopan, Belice" → [BZ].
    A bare alpha-2 code counts only when it is the whole value."""
    raw = str(text).strip()
    if re.fullmatch(r"[A-Z]{2}", raw) and raw in ISO2:
        return [raw]
    t = f" {base_norm(raw)} "
    found: list[tuple[int, str]] = []
    taken: list[tuple[int, int]] = []
    for name, code in _COUNTRY_PATTERNS:
        for m in re.finditer(rf" {re.escape(name)} ", t):
            span = (m.start(), m.end())
            if any(s < span[1] and span[0] < e for s, e in taken):
                continue  # inside a longer name already matched ("ciudad de mexico")
            taken.append(span)
            found.append((m.start(), code))
    seen: list[str] = []
    for _, code in sorted(found):
        if code not in seen:
            seen.append(code)
    return seen


# --- addresses -----------------------------------------------------------------------------

_ADDR_ABBR = {
    "cr": "carrera",
    "cra": "carrera",
    "kr": "carrera",
    "kra": "carrera",
    "cl": "calle",
    "cll": "calle",
    "av": "avenida",
    "ave": "avenida",
    "avda": "avenida",
    "of": "oficina",
    "ofc": "oficina",
    "ofi": "oficina",
    "office": "oficina",
    "fl": "piso",
    "floor": "piso",
    "col": "colonia",
    "edif": "edificio",
    "ed": "edificio",
    "building": "edificio",
    "dg": "diagonal",
    "tv": "transversal",
    "st": "street",
    "rd": "road",
    "po": "po",
    "apdo": "po",
}
_ADDR_DROP = {
    "no",
    "nro",
    "num",
    "numero",
    "n",
    "de",
    "del",
    "la",
    "el",
    "the",
    "d",
    "c",
    "dc",
    "y",
    "and",
    "cp",
    "c p",
}


def address_key(text: str) -> tuple[tuple[str, ...], frozenset[str]]:
    """(the address's number tokens in order, its words): abbreviations expanded, filler dropped.
    `93A-21` gives the numbers ("93a", "21")."""
    nums: list[str] = []
    words: set[str] = set()
    for tok in base_norm(text).split():
        if any(ch.isdigit() for ch in tok):
            nums.append(tok)
        elif tok not in _ADDR_DROP:
            words.add(_ADDR_ABBR.get(tok, tok))
    return tuple(nums), frozenset(words)


def same_address(a: str, b: str) -> bool:
    """The same place written two ways ("CR 15 93A 21 OF 304, BOGOTA D.C." and "Carrera 15 No.
    93A-21, Oficina 304, Bogotá D.C., Colombia"): every number agrees, in order, and one's words
    are a subset of the other's. Without numbers only identical words count."""
    (na, wa), (nb, wb) = address_key(a), address_key(b)
    if na != nb:
        return False
    if not na:
        return wa == wb and bool(wa)
    return wa <= wb or wb <= wa


# --- sub-national regions and free zones (spec) ---------------------------------------------

_ZONE_WORDS = r"(?i:free\s*(?:trade\s+)?zone|freezone|zona\s+franca|zona\s+libre)"
_REGIONS = [
    # the embargoed UA regions named by the policy, in English, Spanish and transliterated forms
    r"(?i:\b(?:(?:Autonomous\s+)?Republic\s+of\s+)?Crimea\b|\bCrimean\b|\bKrym\b|\bSevastopol\b)",
    r"(?i:\bDonetsk(?:\s+(?:Oblast|region|regi[oó]n|People's\s+Republic))?\b)",
    r"(?i:\bLu[hg]ansk(?:\s+(?:Oblast|region|regi[oó]n|People's\s+Republic))?\b)",
    # named UAE free zones, with an optional "Free Zone" / "Authority" after the name
    r"\b(?:DMCC|JAFZA|DIFC|ADGM|DAFZA|RAKEZ|KIZAD|DSO|"
    r"(?i:Jebel\s+Ali|SAIF|RAK\s+FTZ|Dubai\s+Silicon\s+Oasis|Dubai\s+Internet\s+City|"
    r"Dubai\s+Media\s+City|Dubai\s+Multi\s+Commodities\s+Centre|Hamriyah|Ajman|Fujairah|"
    r"Sharjah\s+Airport|Masdar\s+City|Umm\s+Al\s+Quwain))(?:\s+" + _ZONE_WORDS
    + r"(?:\s+(?i:Authority|Establishment))?)?\b",
    # any other free zone, with up to three capitalised words naming it
    r"(?:\b[A-Z][\w&'.-]*\s+){0,3}" + _ZONE_WORDS + r"(?:\s+(?:de|of)\s+[A-Z][\w-]*)?\b",
]  # fmt: skip
_REGION_RE = [re.compile(p) for p in _REGIONS]
_NAMED_ZONE_ONLY = re.compile(r"(?i:ajman|fujairah|hamriyah|saif|jebel\s+ali|umm\s+al\s+quwain|"
                              r"sharjah\s+airport)")  # fmt: skip


def regions(text: str) -> list[str]:
    """The sub-national regions and free zones a text names, verbatim and in order: the UA
    embargoed regions and UAE free zones. An emirate's name alone is a place,
    not a free zone, so it counts only with "Free Zone" after it."""
    spans: list[tuple[int, int]] = []
    for rx in _REGION_RE:
        for m in rx.finditer(str(text)):
            s, e = m.span()
            frag = m[0].strip(" ,.;:-")
            if not frag or (_NAMED_ZONE_ONLY.fullmatch(frag)):
                continue
            s = m.start() + m[0].find(frag)
            spans.append((s, s + len(frag)))
    out: list[tuple[int, int]] = []
    for s, e in sorted(spans, key=lambda se: (se[0], -(se[1] - se[0]))):
        if any(s < oe and os_ < e for os_, oe in out):
            continue  # inside a longer match already kept
        out.append((s, e))
    seen: list[str] = []
    for s, e in sorted(out):
        frag = str(text)[s:e]
        if frag not in seen:
            seen.append(frag)
    return seen
