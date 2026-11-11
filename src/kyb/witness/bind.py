"""`bind-v6`: does the witness's blind re-read of a crop say what the claim says? (spec)

The witness never sees the claim. It lists every `{subject, attribute, value}` it reads in a crop
(`witness-v4`; a `witness-v3` record holds one main pair, read the same way). A claim splits into
components: one for a scalar value, one per field of a structured value (amount + currency stay
together as money). The claim binds when every component binds to some re-read tuple of its crop:

- **subject:** the tuple names the claim's subject (ordered, one-to-one name tokens); or the tuple
  is a field label ("Fecha", "Estatus") whose canonical attribute is the claim's, and the claim's
  subject is the document's own entity, and no named party in the crop or in the blind layout of
  that block holds that attribute instead;
- **roles:** `Lucas Navarro Solís -> Presidente` and `Presidente -> Lucas Navarro Solís` are
  the same statement, so a role claim binds in either orientation;
- **attribute:** when the tuple's attribute is a known canonical attribute, it must be compatible
  with the component's (a share count never binds a nominal value, a birth date never an issue
  date);
- **value:** typed equality by kind (`value_match`); text after trimming punctuation; a
  structured value compares by component, each found in its role (`5.000 acciones`, `MXN 1.000`,
  `Serie A`).

A struck value never binds an unflagged claim, and a flagged claim binds only a struck value.
Every relaxation here has a negative test in `tests/test_witness.py`: the same row with another
party's value still fails, because a false accept is worse than a false reject.
"""

from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Any

from kyb.witness.norm import contains, norm, parse_date

BIND = "bind-v8"
_LEGAL = {
    "s.a.",
    "s.a",
    "sa",
    "de",
    "c.v.",
    "cv",
    "s.a.s.",
    "sas",
    "ltd",
    "ltda",
    "inc",
    "llc",
    "s.l.",
    "sl",
}
# a whole number: grouped (1.250.000) or plain with an optional decimal part (25,5)
_PCT = r"\s*%"
_NUM = r"(?<![\d.,])(?:\d{1,3}(?:[.,]\d{3}){2,}|\d+(?:[.,]\d+)?)(?![\d.,])"

# Canonical attributes: true synonyms only (field labels as documents print them included).
# Distinct facts (a meeting date and an issue date, a registered and an operating address) stay
# distinct even when their values coincide.
_ATTR = {
    "pct": ("pct", "percent", "percentage", "shareholding_percentage", "shareholding_pct",
            "ownership_pct", "ownership_percentage", "porcentaje", "participacion",
            "participation"),
    "shares": ("shares", "share_count", "number_of_shares", "acciones", "shares_held", "count",
               "total_shares"),
    "shares_acquired": ("shares_acquired", "acquired_shares", "acciones_adquiridas"),
    "series": ("series", "serie", "share_series"),
    "par_value": ("par_value", "valor_nominal", "nominal_value"),
    "amount": ("amount", "monto", "importe"),
    "capital": ("capital", "capital_social", "capital_social_fijo", "fixed_share_capital",
                "share_capital", "fixed_capital"),
    "dob": ("dob", "date_of_birth", "birth_date", "fecha_de_nacimiento", "nacido", "nacida"),
    "date": ("date", "fecha"),  # a bare date label: compatible with every date attribute
    "issue_date": ("issue_date", "issued_on", "date_issued", "fecha_de_emision",
                   "fecha_de_expedicion", "document_date"),
    "meeting_date": ("meeting_date", "fecha_de_sesion", "session_date"),
    "signing_date": ("signing_date", "signed_on", "fecha_de_firma"),
    "incorporation_date": ("incorporation_date", "incorporated_on", "fecha_de_constitucion"),
    "operations_start_date": ("operations_start_date", "fecha_de_inicio_de_operaciones",
                              "start_of_operations"),
    "registration_date": ("registration_date", "fecha_de_inscripcion", "inscrito_el"),
    "entry_date": ("entry_date", "fecha_del_asiento"),
    "validity_period": ("stated_validity", "validity_period", "vigencia", "valid_for"),
    "valid_until": ("valid_until", "expiry_date", "expires_on", "fecha_de_vencimiento"),
    "deed_number": ("deed_number", "numero"),
    "tax_id": ("tax_id", "rfc", "nit", "rut", "cuit", "tin"),
    "registry_id": ("registry_id", "folio_mercantil", "folio_mercantil_electronico",
                    "registration_number", "company_number", "matricula"),
    "legal_name": ("legal_name", "denominacion", "denominada", "denominacion_social",
                   "razon_social", "company_name", "entity_name"),
    "legal_form": ("legal_form", "tipo_de_sociedad", "forma_juridica", "company_type"),
    "management_body": ("management_body", "administracion", "organo_de_administracion",
                        "governing_body"),
    "registered_address": ("registered_address", "domicilio_social", "registered_domicile"),
    "operating_address": ("operating_address", "domicilio_operativo"),
    "tax_address": ("tax_address", "domicilio_fiscal", "fiscal_address"),
    "residential_address": ("residential_address", "home_address", "domicilio_particular"),
    # a bare address label: compatible with every address
    "address": ("address", "domicilio", "domicile"),
    "activity": ("activity", "activities", "objeto_social", "corporate_object",
                 "corporate_purpose", "declared_corporate_object", "economic_activity",
                 "actividad_economica", "actividad"),
    "tax_status": ("tax_status", "estatus", "status", "estatus_en_el_padron"),
    "tax_regime": ("tax_regime", "regimen", "regimen_fiscal"),
    "duration": ("duration", "duracion"),
    "nationality": ("nationality", "nacionalidad"),
    "role": ("role", "board_role", "cargo", "caracter", "puesto", "position"),
    "transfer": ("transfer", "pending_transfer", "share_transfer", "share_acquisition",
                 "pending_share_acquisition", "transmision"),
}  # fmt: skip
_CANON = {v: k for k, vs in _ATTR.items() for v in vs}
KNOWN = set(_ATTR)
_DATES = {"dob", "date", "issue_date", "meeting_date", "signing_date", "incorporation_date",
          "operations_start_date", "registration_date", "entry_date", "valid_until"}  # fmt: skip
_ADDRESSES = {"address", "registered_address", "operating_address", "tax_address",
              "residential_address"}  # fmt: skip
_GENERIC = {"date": _DATES, "address": _ADDRESSES}
_IDS = {"tax_id", "registry_id"}
_NUMBERS = {"pct", "shares", "shares_acquired"}
_MONEY_KEYS = {"amount", "par_value", "capital"}
_PARTS = {"pct", "shares", "shares_acquired", "series", "par_value", "amount", "capital"}
_HOLDING = _PARTS
# company-only facts. An address subject is a company too, unless the blind layout gives it a
# birth date: it also calls a person's domicile a "registered address" (a nationality is not
# person-only: the layout reads "Nacionalidad: Belice" for a company)
_ENTITY_ONLY = {"activity", "capital", "legal_form", "management_body"}
_ADDRESS_OWNER = _ADDRESSES - {"residential_address"}
_PERSON_ONLY = {"dob"}
_DOC_LABELS = {
    "escritura_publica",
    "escritura",
    "poder",
    "acta",
    "poliza",
    "documento",
}
_DOC_LEVEL = _DATES | {"deed_number"}
_CUR = {"mxn": "mxn", "usd": "usd", "eur": "eur", "cop": "cop", "us$": "usd", "$": "$"}
_CUR_RE = r"(mxn|usd|eur|cop|us\$|\$)"


def _fold(s: str) -> str:
    """`norm`, then accents dropped: `Iberia` and `IBERIA` are one name."""
    return "".join(ch for ch in unicodedata.normalize("NFKD", norm(s))
                   if not unicodedata.combining(ch))  # fmt: skip


def canon_attr(attribute: str | None) -> str:
    """The canonical attribute; a label's trailing preposition ("domicilio en") is not part of
    the fact."""
    a = re.sub(r"[^a-z0-9]+", "_", _fold(attribute or "")).strip("_")
    return _CANON.get(a) or _CANON.get(re.sub(r"_(?:en|el|la|de|del|por)$", "", a), a)


def compatible(a: str, b: str) -> bool:
    """Two canonical attributes name the same fact: equal, or one is the bare generic label."""
    return a == b or b in _GENERIC.get(a, ()) or a in _GENERIC.get(b, ())


def claim_attr(c: dict) -> str:
    if c.get("attribute"):
        return canon_attr(c["attribute"])
    return {"owns": "pct", "amendment": "transfer", "issued_on": "issue_date"}.get(
        c["type"], canon_attr(c["type"])
    )


def _is_date(attr: str) -> bool:
    return attr in _DATES or attr.endswith("_date")


# --- subjects ------------------------------------------------------------------------------------


def _tokens(s: str) -> list[str]:
    return [t for t in re.split(r"[^\w.]+", _fold(s)) if t and t not in _LEGAL]


def _tok_match(a: str, b: str) -> bool:
    a, b = a.rstrip("."), b.rstrip(".")
    if not a or not b:
        return False
    if len(a) == 1 or len(b) == 1:
        return a[0] == b[0]
    return a == b


def _given_first(name: str) -> str:
    """'Ortega Méndez, Valeria' -> 'Valeria Ortega Méndez' (an explicit surname-first form)."""
    if name.count(",") == 1:
        surnames, given = (p.strip() for p in name.split(","))
        if surnames and given and not re.search(r"\d", given):
            return f"{given} {surnames}"
    return name


def subject_match(claim: str, reread: str) -> bool:
    """Claimed name tokens match distinct re-read tokens in order. A full token may skip re-read
    tokens (a missing middle name); an initial must match the token at its own position."""
    ca, rb = _tokens(_given_first(claim)), _tokens(_given_first(reread))
    i = 0
    for t in ca:
        if len(t.rstrip(".")) == 1:
            if i >= len(rb) or not _tok_match(t, rb[i]):
                return False
        else:
            while i < len(rb) and not _tok_match(t, rb[i]):
                i += 1
            if i == len(rb):
                return False
        i += 1
    return bool(ca)


def label_like(t: dict) -> bool:
    """A tuple whose subject is a field label (or empty), not a named person or company."""
    s = (t.get("subject") or "").strip()
    if not s:
        return True
    cs = canon_attr(s)
    return cs in KNOWN or cs in _DOC_LABELS or cs == canon_attr(t.get("attribute"))


def _label(t: dict) -> str:
    s = (t.get("subject") or "").strip()
    cs = canon_attr(s)
    if cs in _DOC_LABELS:
        return canon_attr(t.get("attribute"))
    return cs if s and cs in KNOWN else canon_attr(t.get("attribute"))


# --- values --------------------------------------------------------------------------------------


def value_text(v: Any) -> str:
    if isinstance(v, dict):
        return " ".join(str(x) for x in v.values())
    return str(v)


def _alnum(s: str) -> str:
    return re.sub(r"[^0-9a-z]", "", _fold(s))


def _decimals(text: str, grouping: bool, marker: str = "") -> list[Decimal]:
    """The numbers written in `text`. Counts treat `.`/`,` as grouping (1.250.000 acciones);
    other numbers treat one `,` or `.` as the decimal mark (25,5% = 25.5%)."""
    out, t = [], norm(text)
    for m in re.finditer(_NUM, t):
        raw = m.group()
        if marker and not re.match(marker, t[m.end() :]):
            continue  # only the numbers in that role: `50 %`, `2.300 acciones`
        many = raw.count(".") + raw.count(",") > 1
        digits = re.sub(r"[.,]", "", raw) if grouping or many else raw.replace(",", ".")
        signed = _unary_minus(t[: m.start()])
        try:
            out.append(-Decimal(digits) if signed else Decimal(digits))
        except InvalidOperation:
            continue
    return out


def _typed_numbers(v: Any) -> list[Decimal] | None:
    """Numbers already typed in JSON keep their exact value; no document grouping rules."""
    vals = list(v.values()) if isinstance(v, dict) else [v]
    if vals and all(isinstance(x, int | float) and not isinstance(x, bool) for x in vals):
        return [Decimal(str(x)) for x in vals]
    return None


def _unary_minus(before: str) -> bool:
    """Is the `-` right before a number its sign? A free-standing minus ("-46%", "total -46%",
    ":-46%") is; a date's or an id's hyphen ("2017-05", "N-2017") and a range's dash ("10 -20%")
    are not."""
    m = re.search(r"(\S?)(\s*)-\s?$", before)
    if not m:
        return False
    prev, gap = m.group(1), m.group(2)
    if not prev:
        return True
    if prev.isdigit() or prev == "%":
        return False  # a range: "10-20%", "10 -20%", "10% - 20%"
    return bool(gap) or not prev.isalnum()


_RANGE = re.compile(r"(" + _NUM + r")\s*%?\s*(?:-|a|al|to|y)\s*(" + _NUM + r")\s*%")


def _role_numbers(text: str, attr: str) -> list[Decimal]:
    """The numbers in their role: for a percentage, only numbers marked with `%`. A range
    ("10-20%", "10 a 20%") is both its ends, so no single percentage binds it."""
    if attr == "pct" and "%" in text:
        ends = [d for m in _RANGE.finditer(norm(text))
                for d in _decimals(f"{m[1]} {m[2]}", False)]  # fmt: skip
        return ends + [d for d in _decimals(text, False, _PCT) if not ends or d not in ends]
    return _decimals(text, attr in ("shares", "shares_acquired"))


def _full_date(text: str) -> bool:
    d = parse_date(text)
    return bool(d) and not re.sub(r"[\d/\-., ]|de|del", "", norm(text)).strip()


def _text_key(x: str) -> str:
    """Text for equality: punctuation (a trailing period included) is not content."""
    return re.sub(r"[^\w%+\-]+", " ", norm(x)).strip()


def value_match(claim_value: Any, other: str, attr: str = "") -> bool:
    """Typed equality, dispatched by the attribute's kind first. Never a bag of digits.
    - dates (date attributes, or two values that are wholly dates): both must parse, and agree;
    - identifiers: every letter and digit;
    - percentages and counts: the numbers in their role, as decimals (typed JSON numbers exact);
    - everything else, transfers included: equal in full after `norm`, punctuation aside."""
    a, b = value_text(claim_value), str(other)
    if _is_date(attr):
        da, db = parse_date(a), parse_date(b)
        return bool(da) and da == db
    if attr in _IDS:
        return bool(_alnum(a)) and _alnum(a) == _alnum(b)
    if attr in _NUMBERS:
        na = _typed_numbers(claim_value) or _role_numbers(a, attr)
        return bool(na) and sorted(na) == sorted(_role_numbers(b, attr))
    if _full_date(a) and _full_date(b):
        return parse_date(a) == parse_date(b)
    typed = _typed_numbers(claim_value)
    if typed is not None:
        return typed == _decimals(b, False)
    return bool(_text_key(a)) and _text_key(a) == _text_key(b)


def _amount(raw: str) -> Decimal | None:
    """A written amount: `6.000.000` and `1.000` are grouped; `1.000,50` has a decimal part."""
    if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", raw):
        return Decimal(re.sub(r"[.,]", "", raw))
    if m := re.fullmatch(r"(\d{1,3}(?:([.,])\d{3})*)([.,])(\d{1,2})", raw):
        if m[2] is None or m[2] != m[3]:
            return Decimal(re.sub(r"[.,]", "", m[1]) + "." + m[4])
        return None
    if re.fullmatch(r"\d+[.,]\d{1,2}", raw):
        return Decimal(raw.replace(",", "."))
    return Decimal(raw) if raw.isdigit() else None


def _claimed_amount(x: Any) -> Decimal | None:
    """A claimed amount, sign kept: `-6000000`, `(6.000.000)` and `1000.50` as written."""
    if isinstance(x, int | float) and not isinstance(x, bool):
        return Decimal(str(x))
    raw = re.sub(r"\s+", "", str(x))
    m = re.fullmatch(r"(-?)(\d[\d.,]*)|\((\d[\d.,]*)\)", raw)
    if not m:
        return None  # "(5000000", "5000000-": not an amount as written
    amt = _amount(m[2] or m[3])
    return None if amt is None else (-amt if m[1] or m[3] else amt)


def _money(text: str) -> list[tuple[Decimal, str]]:
    t, out = norm(text), []
    num = r"(\d[\d.,]*\d|\d)"
    for m in re.finditer(_CUR_RE + r"\s*" + num + r"|" + num + r"\s*" + _CUR_RE + r"(?![a-z])", t):
        cur, raw = (m[1], m[2]) if m[1] else (m[4], m[3])
        if (amt := _amount(raw)) is not None:
            # a minus sign or accounting parentheses make it negative, never the positive amount
            before, after = t[: m.start()].rstrip(), t[m.end() :].lstrip()
            negative = before.endswith("-") or (before.endswith("(") and after.startswith(")"))
            out.append((-amt if negative else amt, _CUR[cur]))
    return out


def _money_range(text: str) -> bool:
    """Currency may occur at either or both ends; a range never states a scalar amount."""
    number = r"(?:\d[\d.,]*\d|\d)"
    currency = r"(?:mxn|usd|eur|cop|us\$|\$)"
    # Parentheses affect a scalar's sign in _money, but cannot hide a range endpoint.
    # Removing them here is rejection-only, including nested or ambiguous parentheses.
    text = re.sub(r"[()]", "", norm(text))
    signs = r"(?:[-+]\s*)*"
    return bool(re.search(number + rf"\s*(?:{currency})?\s*(?:-|al|a|to|y)\s*"
                          + signs + rf"(?:{currency}\s*)?" + signs + number, text))  # fmt: skip


def _share_counts(text: str) -> list[Decimal]:
    return _decimals(text, True, r"\s*(?:acciones|shares|acc\.)")


def _number(x: Any) -> Decimal | None:
    if isinstance(x, int | float) and not isinstance(x, bool):
        return Decimal(str(x))
    d = _decimals(str(x), True)
    return d[0] if len(d) == 1 else None


def _key_ok(key: str, attr: str) -> bool:
    """A field of a structured value may refine the claim's attribute, never contradict it: a
    holding is a percentage and a count, a capital an amount; a birth date is never an
    incorporation date."""
    if key not in KNOWN or attr not in KNOWN or compatible(key, attr):
        return True
    return key in _PARTS and attr in _HOLDING


def components(c: dict) -> list[tuple[str, Any]] | None:
    """The claim's value as `(canonical attribute, value)` parts, or None when a field
    contradicts the claim's attribute. A structured value splits by field; an amount and its
    currency stay one `money:<field>` part."""
    v, attr = c["value"], claim_attr(c)
    keys = [canon_attr(k) for k in v] if isinstance(v, dict) else []
    if not all(_key_ok(k, attr) for k in keys if k != "currency"):
        return None
    if not isinstance(v, dict) or len(v) <= 1:
        key = keys[0] if keys else ""
        if attr in KNOWN and compatible(key, attr):
            return [(attr, v)]  # the claim's own (more specific) attribute
        return [(key if key in KNOWN else attr, v)]
    cur = v.get("currency")
    out: list[tuple[str, Any]] = []
    for k, x in v.items():
        ck = canon_attr(k)
        if k == "currency":
            continue
        if cur and (ck in _MONEY_KEYS or k == "value"):
            # a bare `amount` is the amount of the claimed attribute (capital, a price)
            role = attr if ck in ("amount", "value") and attr in KNOWN else ck
            out.append((f"money:{role}", (x, str(cur))))
        else:
            out.append((ck, x))
    if cur and not any(k.startswith("money:") for k, _ in out):
        out.append(("currency", cur))
    return out


def _part_value(part: str, x: Any, t: dict) -> bool:
    """Is component `x` stated by tuple `t`, in its role?"""
    text = str(t.get("value") or "")
    if part.startswith("money:"):
        amount, cur = x
        n = _claimed_amount(amount)
        found = _money(text)
        # A currency may label only one end of a range. Counting currency-bearing matches
        # alone accepted that endpoint, even reading the binary dash as a negative sign.
        # Other numbers can state separate components, such as shares plus par value.
        return (n is not None and not _money_range(text) and len(found) == 1
                and found[0] == (n, _CUR.get(norm(cur), norm(cur))))  # fmt: skip
    if part == "currency":
        return any(c == _CUR.get(norm(str(x)), norm(str(x))) for _, c in _money(text))
    if part == "series":
        series = re.findall(r"\bseries? ([a-z0-9]+)\b", norm(f"{t.get('attribute', '')} {text}"))
        return series == [norm(str(x))] or _text_key(text) == _text_key(str(x))
    if part == "shares":
        counts = _share_counts(text)
        if counts:
            return len(counts) == 1 and counts[0] == _number(x)
        if _money(text) or "%" in text:
            return False  # a nominal value or a percentage is not a share count
        return value_match(x, text, "shares")
    return value_match(x, text, part)


def _attr_ok(part: str, t: dict) -> bool:
    """A tuple labelled with a different known fact never binds."""
    ta, pa = canon_attr(t.get("attribute")), part.removeprefix("money:")
    return not (ta in KNOWN and pa in KNOWN and not compatible(ta, pa))


# --- binding -------------------------------------------------------------------------------------


def rereads(rr: dict) -> list[dict]:
    """The re-read tuples of one crop: every `{subject, attribute, value}` (witness-v4), or the
    one main pair of a witness-v3 record."""
    out = [{"subject": str(x.get("subject") or ""), "attribute": str(x.get("attribute") or ""),
            "value": str(x.get("value") or ""), "struck": bool(x.get("struck"))}
           for x in rr.get("values") or [] if isinstance(x, dict)]  # fmt: skip
    if rr.get("subject") or rr.get("value"):
        out.append({"subject": str(rr.get("subject") or ""), "attribute": "",
                    "value": str(rr.get("value") or ""), "struck": False})  # fmt: skip
    return out + [d for t in out if (d := _label_prefixed(t))]


def _label_prefixed(t: dict) -> dict | None:
    """`Folio Mercantil Digital N-2017101543` states its own label: the same tuple, read as
    that label and the rest. An extra candidate, never a replacement."""
    if canon_attr(t["attribute"]) in KNOWN:
        return None  # an explicit label is never overwritten by the value's first words
    words = t["value"].split()
    for k in range(min(5, len(words) - 1), 0, -1):
        label = " ".join(words[:k])
        # one word is a label only when the document marks it so ("Fecha: ...")
        if canon_attr(label) in KNOWN and (k > 1 or label.endswith(":")):
            return {**t, "attribute": label, "value": " ".join(words[k:])}
    return None


def same_company(a: str, b: str) -> bool:
    """Two written names of one company: the same words, legal form and accents aside."""
    ta, tb = {t.rstrip(".") for t in _tokens(a)}, {t.rstrip(".") for t in _tokens(b)}
    return bool(ta) and ta == tb


def _is_role(c: dict) -> bool:
    return c["type"] == "role" or claim_attr(c) == "role"


class Scope:
    """What a document says about its own entity, read blind: the layout's headings and its
    material facts, and every crop's re-read tuples."""

    def __init__(self, headings: list[str], facts: list, crops: dict[str, dict]) -> None:
        self.facts = facts
        # Every legal-name read counts, including a name at a crop edge. Treating a short
        # name as a cut of another company hid real rivals (l1-a5, rounds 4–6).
        companies = []
        persons = [f.subject for f in facts if canon_attr(f.attribute) in _PERSON_ONLY]
        for f in facts:
            if canon_attr(f.attribute) == "legal_name":
                companies += [f.subject, f.value]
            elif (
                canon_attr(f.attribute) in _ENTITY_ONLY
                or canon_attr(f.attribute) in _ADDRESS_OWNER
                and not any(same_company(f.subject, n) for n in persons)
            ):
                companies.append(f.subject)
        companies = [n for n in companies if n]
        for r in crops.values():
            for t in rereads(r.get("reread") or {}):
                if t["value"] and (canon_attr(t["attribute"]) == "legal_name"
                                   or (label_like(t) and _label(t) == "legal_name")):  # fmt: skip
                    companies.append(t["value"])
        self.companies = companies
        self.titles = [h for h in headings if h]
        self.entities = self.titles + self.companies

    def is_entity(self, subject: str) -> bool:
        """The document's own entity: named by its title or headings, and the only company it
        names. A document that names two companies has no unambiguous own entity; company names
        are one company only when their words are the same (`Acme Holdings` is not `Acme
        Holdings Colombia`)."""
        return any(subject_match(subject, e) for e in self.entities) and all(
            same_company(subject, n) for n in self.companies
        )

    def owner_ok(self, c: dict, part: str, x: Any) -> bool:
        """Is the claim's subject the one the document gives this attribute in this block?
        Its own entity, or the subject the blind layout names for that very value; and no other
        named party holds the attribute there."""
        pa = part.removeprefix("money:")
        same = [f for f in self.facts if f.block_id == c["block_id"]
                and compatible(canon_attr(f.attribute), pa)]  # fmt: skip
        others = [f for f in same if not subject_match(c["subject"], f.subject)]
        if others:
            return False
        return self.is_entity(c["subject"]) or any(
            subject_match(c["subject"], f.subject) and _part_value(part, x, {"value": f.value})
            for f in same
        )


def _subject_ok(c: dict, part: str, x: Any, t: dict, crop: list[dict], scope: Scope | None) -> bool:
    if subject_match(c["subject"], t["subject"]):
        return True
    if scope is None or not label_like(t):
        return False
    pa = part.removeprefix("money:")

    s = (t.get("subject") or "").strip()
    is_doc_label = canon_attr(s) in _DOC_LABELS
    if is_doc_label and pa not in _DOC_LEVEL:
        return False

    if is_doc_label:
        label_attr = _label(t)
        if label_attr and not compatible(label_attr, pa):
            return False
    else:
        if not (pa in KNOWN and compatible(_label(t), pa)):
            return False
    # another named party in the crop that holds this attribute, or this very value, owns it
    rivals = [u for u in crop if not label_like(u)
              and not subject_match(c["subject"], u["subject"])
              and (compatible(canon_attr(u["attribute"]), pa)
                   or _part_value(part, x, u))]  # fmt: skip
    return not rivals and scope.owner_ok(c, part, x)


def _role_binds(c: dict, crop: list[dict], struck: list[str]) -> bool:
    """Either orientation: `person -> role` or `role -> person` (the role as subject or label).
    A struck line naming both the person and the role withdraws it."""
    role = _text_key(value_text(c["value"]))
    if any(subject_match(c["subject"], s) and contains(s, value_text(c["value"])) for s in struck):
        return False
    for t in crop:
        if t["struck"] or not role:
            continue

        if subject_match(c["subject"], t["subject"]) and _text_key(t["value"]) == role:
            return True
        if subject_match(c["subject"], t["value"]) and role in (
            _text_key(t["subject"]),
            _text_key(t["attribute"]),
        ):
            return True
    return False


def _pct_marked(text: str) -> list[Decimal]:
    return _role_numbers(text, "pct") if "%" in text else []


def transfer_binding(c: dict, rr: dict, transcription: str) -> bool:
    """A transfer is a relationship: its claimed text must be quoted verbatim from the witnessed
    crop (direction, counterparty, sign and status included). With every tuple re-read
    (witness-v4), a tuple of the claim's own subject must state it, with the same percentages;
    a witness-v3 record's one value must carry the transfer's (non-empty) numbers (bind-v5)."""
    text = value_text(c["value"])
    if not contains(transcription, text):
        return False
    crop = [t for t in rereads(rr) if not t["struck"]]
    named = [t for t in crop if c.get("subject") and subject_match(c["subject"], t["subject"])]
    if "values" in rr:  # every tuple re-read: the claimant's own tuple must state the transfer
        return any(contains(f"{t['subject']} {t['value']}", text)
                   and _pct_marked(t["value"]) == _pct_marked(text) for t in named)  # fmt: skip
    want = _role_numbers(text, "pct")  # a witness-v3 record: its one value carries the numbers
    return bool(want) and any(_role_numbers(t["value"], "pct") == want for t in (named or crop))


def binding(c: dict, rr: dict, transcription: str = "", scope: Scope | None = None) -> bool:
    crop = rereads(rr)
    attr = claim_attr(c)
    struck = [str(x) for x in rr.get("struck") or []]
    flagged = "struck-through" in c.get("flags", [])

    if attr == "legal_name" and not flagged:
        text = value_text(c["value"])
        if contains(transcription, text) and same_company(c["subject"], text):
            rivals = [
                u
                for u in crop
                if not label_like(u)
                and not subject_match(c["subject"], u["subject"])
                and (
                    canon_attr(u.get("attribute")) == "legal_name" or value_match(text, u["value"])
                )
            ]
            if (
                not any(value_match(text, s) for s in struck)
                and not rivals
                and any(not t["struck"] and same_company(text, t["subject"]) for t in crop)
            ):
                return True

    if attr == "transfer":
        text = value_text(c["value"])
        mine = [t for t in crop if subject_match(c["subject"], t["subject"])]
        struck_here = any(contains(s, text) for s in struck) or any(
            t["struck"] and contains(f"{t['subject']} {t['value']}", text) for t in mine
        )
        if flagged:  # a struck transfer binds only the claimant's own struck evidence
            return contains(transcription, text) and any(
                t["struck"] and contains(f"{t['subject']} {t['value']}", text)
                and _pct_marked(t["value"]) == _pct_marked(text)
                for t in mine
            )  # fmt: skip
        return bool(mine) and not struck_here and transfer_binding(c, rr, transcription)
    role = value_text(c["value"])
    if (
        _is_role(c)
        and not flagged
        and (
            any(subject_match(c["subject"], s) and contains(s, role) for s in struck)
            or _role_binds(c, [{**t, "struck": False} for t in crop if t["struck"]], [])
        )
    ):
        return False  # a struck line or tuple naming the person and the role withdraws it
    parts = components(c)
    if parts is None:
        return False
    if flagged:
        # a witness-v3 record: the crop's struck list, under its one re-read naming the subject
        if (
            "values" not in rr
            and any(subject_match(c["subject"], t["subject"]) for t in crop)
            and all(any(value_match(x, s, p) for s in struck) for p, x in parts)
        ):
            return True
        return all(any(t["struck"] and _attr_ok(p, t) and _part_value(p, x, t)
                       and _subject_ok(c, p, x, t, crop, scope) for t in crop)
                   for p, x in parts)  # fmt: skip
    for p, x in parts:  # a struck value presented as current
        if any(value_match(x, s, p) for s in struck):
            return False
        if any(t["struck"] and _part_value(p, x, t) and _subject_ok(c, p, x, t, crop, scope)
               for t in crop):  # fmt: skip
            return False
    if _is_role(c) and len(parts) == 1 and _role_binds(c, crop, struck):
        return True
    if len(parts) > 1 and any(  # one tuple states the whole structured value, each part in role
        not t["struck"] and _attr_ok(attr, t) and _subject_ok(c, attr, c["value"], t, crop, scope)
        and all(_part_value(p, x, t) for p, x in parts) for t in crop
    ):  # fmt: skip
        return True
    return all(any(not t["struck"] and _attr_ok(p, t) and _part_value(p, x, t)
                   and _subject_ok(c, p, x, t, crop, scope) for t in crop)
               for p, x in parts)  # fmt: skip
