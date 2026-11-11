"""Terms and BM25 for retrieval. Lexical on purpose: the same question gives the same bundle.

Accents and case are folded (a question for "duracion" finds "Duración"), and English and
Spanish function words are dropped, because the packets are Spanish and the officers' questions
often are not.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence

_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_STOPWORDS = """
a about al algo all also an and any are as at be been being but by can como con could
cual cuales cuando de del did do does donde el ella ellos en entre era es esta este esto
for from fue ha han has have hay how if in into is it its la las le les lo los mas me mi
muy no nos not o of on or para pero por que quien quienes se should si sin sobre su sus
te than that the their then there these they this those tiene tienen to un una uno unos
very was were what when where which who whom whose why will with would y ya
"""
STOP = frozenset(re.findall(r"\w+", _STOPWORDS))
K1, B = 1.5, 0.75
WHERE_WEIGHT = 0.3  # a match in the cited document's name, against one in the content itself


def fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text).casefold()
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def terms(text: str) -> list[str]:
    """Content terms, in order, with repeats: folded words minus function words."""
    return [w for w in _WORD.findall(fold(text)) if w not in STOP and (len(w) > 1 or w.isdigit())]


# Words that frame a question without naming what it asks about: dropped from a retrieval query
# only (an abstention's topic check keeps `terms` exactly).
_FRAMING = frozenset(
    re.findall(
        r"\w+",
        """according say says said tell give list show many much please mention mentioned state
        states stated document documents record""",
    )
)

# Bilingual equivalence classes for the vocabulary of corporate KYB packets: the packets are
# Spanish and the questions often English (or the reverse), so a question word reaches the
# document word that means the same. Each row is one concept; match is on the stemmed form.
_GLOSSARY = (
    "minutes minute acta actas",
    "board consejo directorio",
    "shareholder shareholders stockholder stockholders accionista accionistas socio socios",
    "register registry registro ledger libro book",
    "share shares stock accion acciones stake participation participacion "
    "shareholding percentage percent porcentaje pct",
    "hold holds held holder own owns owned owner owners ownership propietario propietarios "
    "dueno titular posee",
    "beneficial beneficiary beneficiario beneficiarios ubo",
    "ultimate last latest recent ultimo ultima final",
    "lien liens encumbrance encumbrances pledge pledges gravamen gravamenes prenda prendas "
    "restriction restrictions restriccion restricciones",
    "authorise authorize authorised authorized authorizes authorises autoriza autorizo autorizar "
    "authorization authorisation autorizacion",
    "handwritten handwriting manuscrito manuscrita annotation annotations anotacion note notes "
    "nota "
    "notas",
    "struck strike strikethrough crossed tachado tachada",
    "replace replaced replaces replacement reemplaza reemplazo sustituye sustituido",
    "leave left leaves resigned resign ceased cease cesa cese renuncia renuncio dejo",
    "office cargo position puesto",
    "public publico publica",
    "address domicilio direccion home residence",
    "capital",
    "duration duracion plazo term vigencia",
    "date fecha",
    "signed signature firma firmado firmada sign",
    "update updated actualizo actualizacion actualizada",
    "company sociedad empresa applicant solicitante",
    "director directors administrador administradores",
    "president presidente chairman",
    "account accounts cuenta cuentas",
    "bank banco",
    "object objeto purpose giro activity actividad",
    "employee employees empleado empleados staff payroll nomina workforce headcount",
)


def stem(word: str) -> str:
    """A light, deterministic stem for folded Spanish and English words: plurals, -ing/-ed and
    -acion/-ation endings, a last vowel. Never below four letters; digits and short words stay."""
    if len(word) <= 4 or word.isdigit():
        return word
    for suffix in ("ies", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            word = word[: -len(suffix)] + ("y" if suffix == "ies" else "")
            break
    for suffix in ("acion", "ation", "ing", "ed", "ista", "mente", "ment"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            word = word[: -len(suffix)]
            break
    if word[-1] in "aoe" and len(word) > 4:
        word = word[:-1]
    return word


def stems(text: str, *, query: bool = False) -> list[str]:
    """`terms`, stemmed; for a question, also without its framing words."""
    return [stem(t) for t in terms(text) if not (query and t in _FRAMING)]


def _classes() -> dict[str, frozenset[str]]:
    out: dict[str, frozenset[str]] = {}
    for row in _GLOSSARY:
        group = frozenset(stem(fold(w)) for w in row.split())
        for member in group:
            out[member] = out.get(member, frozenset()) | group
    return out


_CONCEPT = _classes()


def concepts(query: Sequence[str]) -> list[frozenset[str]]:
    """A stemmed question as its distinct concepts: each stem with its glossary equivalents."""
    seen, out = set(), []
    for t in dict.fromkeys(query):
        group = _CONCEPT.get(t, frozenset()) | {t}
        if group not in seen:
            seen.add(group)
            out.append(group)
    return out


def concept_bm25(query: Sequence[str], docs: Sequence[Sequence[str]]) -> list[float]:
    """Okapi BM25 where a concept counts as present through any of its equivalents."""
    if not docs:
        return []
    n = len(docs)
    avg = sum(len(d) for d in docs) / n or 1.0
    cons = concepts(query)
    df = [sum(1 for d in docs if group & set(d)) for group in cons]
    out = []
    for d in docs:
        tf, score = Counter(d), 0.0
        for group, f in zip(cons, df, strict=True):
            c = sum(tf[t] for t in group)
            if c:
                idf = math.log(1 + (n - f + 0.5) / (f + 0.5))
                score += idf * c * (K1 + 1) / (c + K1 * (1 - B + B * len(d) / avg))
        out.append(round(score, 6))
    return out


def concept_matches(query: Sequence[str], doc: Sequence[str],
                    entity: frozenset[str] = frozenset()) -> tuple[int, int]:  # fmt: skip
    """(topical concepts of the question the document has, topical concepts in the question).

    A concept is topical unless every one of its words is part of an entity's name (`entity`):
    "Lucas Navarro" says whom the question is about, not what it asks, so a document that
    only names him does not answer "what is his phone number". A question that is only names
    counts its names."""
    have, cons = set(doc), concepts(query)
    topical = [g for g in cons if not g <= entity] or cons
    hit = [bool(g & have) for g in topical]
    # A two-concept question naming one attribute as modifier + head ("email address", "blood
    # type"): a document with the head but not the modifier is about a different attribute (R2,
    # issue). Only the modifier-first miss counts, so "shares transferred" or "NIT company"
    # still match a document holding their first concept.
    if len(topical) == 2 and not hit[0] and hit[1]:
        return 0, 2
    return sum(hit), len(topical)


def concept_enough(matched: int, wanted: int) -> bool:
    """A hit must share at least half of a short question's topical concepts (a third of a long
    one's, where the framing words that survive outnumber the subject): a lone shared word like
    "company" never makes a document an answer."""
    if wanted <= 0 or matched <= 0:
        return False
    need = math.ceil(wanted / 2) if wanted <= 4 else math.ceil(wanted / 3)
    return matched >= max(1, need)


def bm25(query: Sequence[str], docs: Sequence[Sequence[str]]) -> list[float]:
    """Okapi BM25 of each document for the query's distinct terms (0.0 when none occur)."""
    if not docs:
        return []
    n = len(docs)
    avg = sum(len(d) for d in docs) / n or 1.0
    df = Counter(t for d in docs for t in set(d))
    out = []
    for d in docs:
        tf, score = Counter(d), 0.0
        for t in dict.fromkeys(query):
            if t in tf:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                score += idf * tf[t] * (K1 + 1) / (tf[t] + K1 * (1 - B + B * len(d) / avg))
        out.append(round(score, 6))
    return out


def covers(query: Sequence[str], doc: Sequence[str]) -> int:
    """How many of the query's distinct terms the document has."""
    have = set(doc)
    return sum(1 for t in dict.fromkeys(query) if t in have)


def enough(matched: int, wanted: int) -> bool:
    """A hit must share at least half of the question's distinct terms (one, for a short one):
    a lone shared word like "company" does not make a document an answer."""
    return wanted > 0 and matched >= max(1, math.ceil(wanted / 2))
