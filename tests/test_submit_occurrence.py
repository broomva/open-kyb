"""Submission occurrence is necessary; the blind witness still decides subject/value binding."""

from __future__ import annotations

import functools
import itertools
import sys
import unicodedata

import pytest

from kyb.read.schema import Claim
from kyb.read.submit import _excerpt_errors
from kyb.witness.bind import binding
from kyb.witness.norm import norm


def claim(**over) -> Claim:
    return Claim.model_validate(
        {
            "claim_id": "c-B04-01",
            "type": "owns",
            "subject": "Valeria Ortega Méndez",
            "attribute": "pct",
            "value": {"pct": 25},
            "excerpt": "Valeria Ortega Méndez 25%",
            "page": 1,
            "bbox": [0, 0, 100, 100],
            "block_id": "04-b1",
            "source": "pixels",
            "confidence": 0.9,
            **over,
        }
    )


@pytest.mark.parametrize(
    "over, missing",
    [
        (
            {
                "value": {"pct": 25, "shares": 1250, "series": "A"},
                "excerpt": "Valeria Ortega Méndez 1.250 25%",
            },
            "series",
        ),
        (
            {
                "type": "role",
                "attribute": "role",
                "subject": "Lucas Navarro Solís",
                "value": "Presidente del Consejo de Administración",
                "excerpt": "Presidente: Lucas Navarro Solís",
            },
            "role",
        ),
        (
            {
                "attribute": "nationality",
                "subject": "Lucas Navarro Solís",
                "value": "mexicana",
                "excerpt": "LUCAS NAVARRO SOLÍS (mexicano, nacido el 1969-10-15)",
            },
            "nationality",
        ),
        (
            {"subject": "Total", "value": {"shares": 5000, "pct": 100}, "excerpt": "5.000 100%"},
            None,
        ),
        (
            {
                "subject": "Libro de registro de accionistas",
                "attribute": "entry_date",
                "value": "2026-10-25",
                "excerpt": "Inscripcion No. 8 · Fecha del asiento: 2026-10-25",
            },
            None,
        ),
        ({"subject": "Valeria Ortega Mendez"}, None),  # accents are content in quote normalization
        ({"subject": "Valeria Ortega Méndez", "excerpt": "Valeria Ortega 25%"}, None),
    ],
)
def test_actual_reader_failure_classes(over, missing):
    errors = _excerpt_errors(claim(**over))
    assert errors
    assert any(
        e["part"] == ("value" if missing else "subject") and e["component"] == missing
        for e in errors
    )
    assert all(e["claim_id"] == "c-B04-01" and e["code"] == "excerpt_missing" for e in errors)
    assert all(e["normalization"] == "norm-v2" and "re-read" in e["message"] for e in errors)


@pytest.mark.parametrize(
    "over",
    [
        {},
        {"excerpt": "VALERIA ORTEGA MÉNDEZ\n 25 %"},
        {"value": {"pct": 25, "shares": 1250}, "excerpt": "Valeria Ortega Méndez 1.250 25%"},
        {"value": {"pct": 25.5}, "excerpt": "Valeria Ortega Méndez 25,5%"},
        {
            "type": "role",
            "attribute": "role",
            "value": "Secretario",
            "excerpt": "Secretario: Valeria Ortega Méndez.",
        },
        {
            "attribute": "nationality",
            "value": "mexicana",
            "excerpt": "Valeria Ortega Méndez (mexicana)",
        },
        {"subject": "Total", "value": {"pct": 100, "shares": 5000}, "excerpt": "Total 5.000 100%"},
        {
            "subject": "Fecha",
            "attribute": "issue_date",
            "value": "2026-10-25",
            "excerpt": "Fecha: 25 de octubre de 2026",
        },
        {
            "attribute": "dob",
            "value": "1976-10-15",
            "excerpt": "Valeria Ortega Méndez, nacida el 15/10/1976",
        },
        {
            "attribute": "tax_id",
            "value": "MADE761015M55",
            "excerpt": "Valeria Ortega Méndez RFC: MADE761015M55",
        },
        {
            "attribute": "capital",
            "value": {"amount": 6000000, "currency": "MXN"},
            "excerpt": "Valeria Ortega Méndez capital MXN 6.000.000",
        },
        {
            "value": {"shares": 1250, "pct": 25, "series": "A"},
            "excerpt": "Valeria Ortega Méndez 1.250 acciones Serie A 25%",
        },
        {
            "attribute": "residential_address",
            "value": "Av. Chapultepec 480, Piso 9",
            "excerpt": "Valeria Ortega Méndez: Av. Chapultepec 480, Piso 9",
        },
        {
            "type": "amendment",
            "attribute": "transfer",
            "subject": "L. Navarro",
            "value": "recibe 10% extra de Altiplano el 3/8 — pendiente de asentar",
            "excerpt": ("L. Navarro recibe 10% extra de Altiplano el 3/8 — pendiente de asentar"),
        },
    ],
)
def test_valid_controls(over):
    assert _excerpt_errors(claim(**over)) == []


@pytest.mark.parametrize(
    "value, excerpt",
    [
        ({"pct": 25}, "Valeria Ortega Méndez 125%"),
        ({"pct": 25}, "Valeria Ortega Méndez 25,5%"),
        ({"pct": 25}, "Valeria Ortega Méndez 25 acciones"),
        ({"pct": 25}, "Valeria Ortega Méndez N-25"),
        ({"pct": 25}, "Valeria Ortega Méndez -25%"),
        ({"pct": 25}, "Valeria Ortega Méndez 10-25%"),
        ({"pct": -25}, "Valeria Ortega Méndez 10 -25%"),
        ({"pct": 25}, "Valeria Ortega Méndez 10 - 25%"),
        ({"pct": 25}, "Valeria Ortega Méndez - 25%"),
        ({"pct": 25}, "Valeria Ortega Méndez 10 a 25%"),
        ({"pct": 25}, "Valeria Ortega Méndez (125%)"),
        ({"shares": 1250}, "Valeria Ortega Méndez 121.100 acciones"),
        ({"shares": 1250}, "Valeria Ortega Méndez N-1250"),
        ({"shares": 1250}, "Valeria Ortega Méndez 1.250,5 acciones"),
        ({"amount": 6000000, "currency": "USD"}, "Valeria Ortega Méndez MXN 6.000.000"),
        ({"amount": 6000000, "currency": "MXN"}, "Valeria Ortega Méndez MXN -6.000.000"),
        ({"series": "A"}, "Valeria Ortega Méndez A"),
        ({"pct": True}, "Valeria Ortega Méndez true 1%"),
        ({"pct": [25]}, "Valeria Ortega Méndez 25%"),
        ({}, "Valeria Ortega Méndez 25%"),
    ],
)
def test_typed_components_cannot_take_other_tokens(value, excerpt):
    assert _excerpt_errors(claim(value=value, excerpt=excerpt))


def test_identifier_is_whole_and_date_has_all_components():
    assert _excerpt_errors(
        claim(
            attribute="tax_id",
            value="MADE761015M55",
            excerpt="Valeria Ortega Méndez XMADE761015M550",
        )
    )
    assert _excerpt_errors(
        claim(attribute="dob", value="1976-10-15", excerpt="Valeria Ortega Méndez 30/03/1977")
    )


def test_another_persons_value_in_the_same_row_never_verifies():
    c = claim(
        subject="Lucas Navarro Solís",
        value={"pct": 25},
        excerpt="Lucas Navarro Solís 46% | Valeria Ortega Méndez 25%",
    )
    # Both tokens occur: this preflight must never replace blind semantic binding.
    assert _excerpt_errors(c) == []
    reread = {
        "values": [
            {"subject": "Lucas Navarro Solís", "attribute": "pct", "value": "46%"},
            {"subject": "Valeria Ortega Méndez", "attribute": "pct", "value": "25%"},
        ]
    }
    assert not binding(c.model_dump(), reread, c.excerpt)
    assert binding(c.model_copy(update={"value": {"pct": 46}}).model_dump(), reread, c.excerpt)


@pytest.mark.parametrize(
    "value, excerpt",
    [
        ({"shares": 2026}, "Irene 2026-10-25"),
        ({"shares": 10}, "Irene 10-25%"),
        ({"shares": 10}, "Irene 10 - 25%"),
        ({"pct": 10}, "Irene 10% - 25%"),
        ({"amount": 5000, "currency": "MXN"}, "Irene XMXN 5.000"),
        ({"amount": 5000, "currency": "MXN"}, "Irene MXN 5.000ABC"),
        ({"amount": 5000, "currency": "MXN"}, "Irene MXN 5.000-ABC"),
        ({"amount": 5000, "currency": None}, "Irene 5000"),
        ({"pct": 25, "currency": False}, "Irene 25%"),
        ({"amount": 5000, "currency": ""}, "Irene 5000"),
        ({"amount": 5000, "currency": 0}, "Irene 5000"),
    ],
)
def test_round_one_boundary_and_unsupported_component_regressions(value, excerpt):
    assert _excerpt_errors(claim(subject="Irene", value=value, excerpt=excerpt))


@pytest.mark.parametrize(
    "attribute, value, excerpt",
    [
        ("pct", {"pct": 25}, "Irene 25%."),
        ("pct", {"pct": 25}, "Irene 25%, según acta"),
        ("pct", {"shares": 1250}, "Irene 1.250."),
        ("tax_id", "MADE761015M55", "Irene RFC:MADE761015M55"),
        ("registry_id", "N-2017101543", "Irene Folio:N-2017101543"),
        ("capital", {"amount": 5000, "currency": "MXN"}, "Irene MXN 5.000."),
    ],
)
def test_round_one_punctuation_controls(attribute, value, excerpt):
    assert (
        _excerpt_errors(claim(subject="Irene", attribute=attribute, value=value, excerpt=excerpt))
        == []
    )


@pytest.mark.parametrize(
    "excerpt", ["Irene -MXN 5.000", "Irene - MXN 5.000", "Irene (5.000 MXN)", "Irene (MXN 5.000)"]
)
def test_money_sign_and_accounting_context_are_preserved(excerpt):
    assert (
        _excerpt_errors(
            claim(
                subject="Irene",
                attribute="capital",
                value={"amount": -5000, "currency": "MXN"},
                excerpt=excerpt,
            )
        )
        == []
    )
    assert _excerpt_errors(
        claim(
            subject="Irene",
            attribute="capital",
            value={"amount": 5000, "currency": "MXN"},
            excerpt=excerpt,
        )
    )


def test_series_can_keep_its_printed_label():
    assert (
        _excerpt_errors(claim(value={"series": "Serie A"}, excerpt="Valeria Ortega Méndez Serie A"))
        == []
    )


@pytest.mark.parametrize(
    "value, excerpt",
    [
        ({"amount": -5000, "currency": "MXN"}, "Irene capital: 1.000 - 5.000 MXN"),
        ({"amount": 1000, "currency": "MXN"}, "Irene MXN 1.000 a MXN 5.000"),
        ({"amount": 5000, "currency": "MXN"}, "Irene MXN 1.000 a MXN 5.000"),
        ({"amount": 1000, "currency": "MXN"}, "Irene 1.000 MXN - 5.000 MXN"),
        ({"amount": -5000, "currency": "MXN"}, "Irene 1.000 MXN - 5.000 MXN"),
        ({"amount": 1000, "currency": "MXN"}, "Irene MXN 1.000 to 5.000 MXN"),
        ({"amount": 5000, "currency": "MXN"}, "Irene MXN 1.000 to 5.000 MXN"),
        ({"amount": 1000, "currency": "MXN"}, "Irene 1.000 MXN y 5.000 MXN"),
        ({"amount": 5000, "currency": "MXN"}, "Irene 1.000 MXN y 5.000 MXN"),
    ],
)
def test_monetary_ranges_never_supply_a_scalar_component(value, excerpt):
    assert _excerpt_errors(
        claim(subject="Irene", attribute="capital", value=value, excerpt=excerpt)
    )


@pytest.mark.parametrize(
    "value, excerpt",
    [
        ({"pct": 25}, "Irene (25%)"),
        ({"shares": 25}, "Irene (25 acciones)"),
        ({"pct": -25}, "Irene - 25%"),
        ({"pct": -25}, "Irene -25%"),
        ({"shares": -25}, "Irene - 25 acciones"),
    ],
)
def test_parenthesized_holdings_and_spaced_negative_literals(value, excerpt):
    assert _excerpt_errors(claim(subject="Irene", value=value, excerpt=excerpt)) == []


@pytest.mark.parametrize("connector", ["-", "–", "a", "al", "to", "y"])
@pytest.mark.parametrize("written", ["MXN 1.000 {c} MXN 5.000", "1.000 MXN {c} 5.000 MXN"])
def test_money_range_endpoints_across_printed_currency_forms(connector, written):
    excerpt = "Irene " + written.format(c=connector)
    for amount in (1000, 5000, -5000):
        assert _excerpt_errors(
            claim(
                subject="Irene",
                attribute="capital",
                value={"amount": amount, "currency": "MXN"},
                excerpt=excerpt,
            )
        )


@pytest.mark.parametrize(
    "excerpt",
    [
        "Irene capital: 1.000 - (5.000 MXN)",
        "Irene capital: 1.000 - ((5.000 MXN))",
        "Irene capital: 1.000-(5.000 MXN)",
        "Irene capital: (1.000 MXN) a (5.000 MXN)",
        "Irene capital: (MXN 1.000) - (MXN 5.000)",
    ],
)
def test_r3_parenthesized_monetary_endpoint_is_not_a_scalar(excerpt):
    for amount in (-5000, 5000):
        assert _excerpt_errors(
            claim(
                subject="Irene",
                attribute="capital",
                value={"amount": amount, "currency": "MXN"},
                excerpt=excerpt,
            )
        )


@pytest.mark.parametrize("operator", ["-", "a", "al", "to", "y"])
@pytest.mark.parametrize(
    "written, component, amounts",
    [
        ("10% {op} 25%", "pct", (10, 25)),
        ("10 acciones {op} 25 acciones", "shares", (10, 25)),
        ("(10%) {op} (25%)", "pct", (10, 25)),
        ("(10 acciones) {op} (25 acciones)", "shares", (10, 25)),
    ],
)
def test_r3_complete_marked_range_endpoints_never_become_scalars(
    operator, written, component, amounts
):
    excerpt = "Irene " + written.format(op=operator)
    for amount in amounts:
        assert _excerpt_errors(claim(subject="Irene", value={component: amount}, excerpt=excerpt))


@pytest.mark.parametrize(
    "excerpt", ["Irene 10%a25%", "Irene 10%al25%", "Irene 10%to25%", "Irene 10accionesa25acciones"]
)
def test_attached_range_operator_cannot_hide_a_marked_endpoint(excerpt):
    component = "pct" if "%" in excerpt else "shares"
    for amount in (10, 25):
        assert _excerpt_errors(claim(subject="Irene", value={component: amount}, excerpt=excerpt))


@pytest.mark.parametrize(
    "excerpt",
    [
        "Irene 25% y 1.250 acciones",
        "Irene 1.250 acciones y 25%",
        "Irene (25%) y (1.250 acciones)",
        "Irene 1.250 acciones, 25%",
        "Irene 25% y 1250 shares",
    ],
)
def test_r3_unlike_units_are_independent_ownership_components(excerpt):
    c = claim(subject="Irene", value={"pct": 25, "shares": 1250}, excerpt=excerpt)
    assert _excerpt_errors(c) == []
    rr = {
        "values": [
            {"subject": "Irene", "attribute": "pct", "value": "25%"},
            {"subject": "Irene", "attribute": "shares", "value": "1.250 acciones"},
        ]
    }
    assert binding(c.model_dump(), rr, c.excerpt)


def test_share_count_and_nominal_money_are_not_one_range():
    assert (
        _excerpt_errors(
            claim(
                subject="Irene",
                value={"shares": 1250, "par_value": 1000, "currency": "MXN"},
                excerpt="Irene 1.250 acciones y MXN 1.000",
            )
        )
        == []
    )


def test_scalar_amount_component_keeps_accounting_sign():
    assert (
        _excerpt_errors(
            claim(subject="Irene", attribute="capital", value=-5000, excerpt="Irene (MXN 5.000)")
        )
        == []
    )
    assert _excerpt_errors(
        claim(subject="Irene", attribute="capital", value=5000, excerpt="Irene (MXN 5.000)")
    )


# --- review round 5: grammar invariants, checked as metamorphic relations ---------------------
#
# Each relation transforms a written quantity and states what that transformation must do to
# the components it may supply, independently of how the lexer is built.


def _supplies(attribute, value, written):
    return not _excerpt_errors(claim(subject="Irene", attribute=attribute, value=value,
                                     excerpt="Irene " + written))  # fmt: skip


@functools.cache
def _dashes() -> tuple[str, ...]:
    """Unicode dash punctuation (Pd), plus the other Dash/Hyphen-property code points
    (SOFT HYPHEN, SWUNG DASH, the superscript, subscript and plain MINUS SIGN)."""
    pd = (chr(c) for c in range(sys.maxunicode + 1) if unicodedata.category(chr(c)) == "Pd")
    return tuple(sorted({*pd, "\u00ad", "\u2053", "\u207b", "\u208b", "\u2212"}))


HOLDINGS = [("pct", "25%"), ("shares", "25 acciones"), ("shares", "25 shares"), ("shares", "25")]


@pytest.mark.parametrize("part, literal", HOLDINGS)
@pytest.mark.parametrize("wrap", ["{}", "({})", "( {} )", "(({}))", "((( {} )))"])
def test_metamorphic_a_sign_applies_to_the_whole_wrapped_holding(part, literal, wrap):
    written = wrap.format(literal)
    assert _supplies(part, {part: 25}, written) and not _supplies(part, {part: -25}, written)
    for sign in ("-", "- "):
        outside, inside = sign + written, wrap.format(sign + literal)
        for form in (outside, inside):
            assert _supplies(part, {part: -25}, form), form
            assert not _supplies(part, {part: 25}, form), form
    for sign in ("+", "+ "):
        for form in (sign + written, wrap.format(sign + literal)):
            assert _supplies(part, {part: 25}, form), form
            assert not _supplies(part, {part: -25}, form), form


@pytest.mark.parametrize("part, literal", HOLDINGS + [("capital", "MXN 5.000")])
@pytest.mark.parametrize("form", ["--{}", "-+{}", "+-{}", "-(-{})", "+(-{})", "-(+{})"])
def test_metamorphic_two_signs_on_one_quantity_supply_nothing(part, literal, form):
    written = form.format(literal)
    if part == "capital":
        values = [{"amount": n, "currency": "MXN"} for n in (5000, -5000)]
    else:
        values = [{part: 25}, {part: -25}]
    for value in values:
        assert not _supplies(part, value, written), written


@pytest.mark.parametrize("written", ["MXN 5.000", "5.000 MXN"])
def test_metamorphic_money_accounting_wrapper_negates_once_and_never_with_a_sign(written):
    def money(n):
        return {"amount": n, "currency": "MXN"}

    assert _supplies("capital", money(5000), written)
    assert _supplies("capital", money(-5000), f"({written})")
    assert not _supplies("capital", money(5000), f"({written})")
    for ambiguous in (f"(({written}))", f"-({written})", f"- ({written})", f"+({written})",
                      f"(-{written})", f"(+{written})"):  # fmt: skip
        for n in (5000, -5000):
            assert not _supplies("capital", money(n), ambiguous), ambiguous


def test_metamorphic_a_dash_is_a_minus_exactly_when_norm_folds_it_to_one():
    for d in _dashes():
        written = f"{d}25%"
        if norm(d) == "-":
            assert _supplies("pct", {"pct": -25}, written), hex(ord(d))
            assert not _supplies("pct", {"pct": 25}, written), hex(ord(d))
        else:  # a hyphen is not a minus sign, and its sign must not vanish either
            for n in (25, -25):
                assert not _supplies("pct", {"pct": n}, written), hex(ord(d))
                assert not _supplies("pct", {"pct": n}, f"{d} 25%"), hex(ord(d))


@pytest.mark.parametrize(
    "template, attribute, value",
    [
        ("123{d}456XYZ", "shares", {"shares": 123}),
        ("123{d}456XYZ", "shares", {"shares": 456}),
        ("N{d}123", "shares", {"shares": 123}),
        ("N{d}123{d}456", "shares", {"shares": 456}),
        ("123{d}(456XYZ)", "shares", {"shares": 123}),
        ("N{d}(123)", "shares", {"shares": 123}),
        ("2026{d}07{d}30", "shares", {"shares": 30}),
        ("N{d}2026-10-25", "dob", "2026-10-25"),
        ("2026-10-25{d}ABC", "dob", "2026-10-25"),
        ("MXN 5.000{d}123ABC", "capital", {"amount": 5000, "currency": "MXN"}),
    ],
)
def test_metamorphic_every_dash_joins_an_identifier_like_ascii_hyphen(template, attribute, value):
    assert not _supplies(attribute, value, template.format(d="-"))
    for d in _dashes():
        assert not _supplies(attribute, value, template.format(d=d)), hex(ord(d))


@pytest.mark.parametrize(
    "template, attribute, values",
    [
        ("10{op}25%", "pct", ({"pct": 10}, {"pct": 25}, {"pct": -25})),
        ("10% {op} 25%", "pct", ({"pct": 10}, {"pct": 25}, {"pct": -25})),
        ("10 acciones {op} 25 acciones", "shares", ({"shares": 10}, {"shares": 25})),
        ("1.000 MXN{op}5.000 MXN", "capital",
         ({"amount": 1000, "currency": "MXN"}, {"amount": 5000, "currency": "MXN"},
          {"amount": -5000, "currency": "MXN"})),
        ("MXN 1.000 {op} MXN 5.000", "capital",
         ({"amount": 1000, "currency": "MXN"}, {"amount": 5000, "currency": "MXN"})),
    ],
)  # fmt: skip
def test_metamorphic_no_range_endpoint_is_a_scalar_whatever_the_dash(template, attribute, values):
    for op in (*_dashes(), "a", "al", "to", "y"):
        if op.isalpha() and "{op}" in template.replace(" {op} ", ""):
            continue  # an attached word operator is a word boundary case, tested separately
        written = template.format(op=op)
        for value in values:
            assert not _supplies(attribute, value, written), (written, value)


@pytest.mark.parametrize(
    "glued", ["N-{a} a {b}", "--{a} a {b}", "{a} a {b}-", "x{a} y {b}", "({a} a {b}", "{a} a {b}x"]
)
def test_metamorphic_a_glued_outer_edge_never_releases_the_other_endpoint(glued):
    """R6: range membership does not depend on the endpoints' outer boundaries."""
    for a, b, attribute, value in (
        ("10", "25%", "pct", {"pct": 25}),
        ("10%", "25%", "pct", {"pct": 10}),
        ("10%", "25%", "pct", {"pct": 25}),
        ("10 acciones", "25 acciones", "shares", {"shares": 10}),
    ):
        written = glued.format(a=a, b=b)
        assert not _supplies(attribute, value, written), written


UNIT_MARKERS = {"pct": ["%"], "shares": ["acciones", "shares", "acc."],
                "money": ["MXN", "USD", "$", "US$", "EUR", "COP"]}  # fmt: skip


@pytest.mark.parametrize(
    "literal, dimension",
    [("25%", "pct"), ("25 %", "pct"), ("25 acciones", "shares"), ("25 shares", "shares"),
     ("MXN 25", "money"), ("MXN25", "money"), ("25 MXN", "money"), ("25MXN", "money"),
     ("$25", "money"), ("USD 25", "money")],
)  # fmt: skip
def test_metamorphic_a_further_unit_marker_never_vanishes(literal, dimension):
    """R5 #3: a second unit after a complete literal blocks every component of either unit."""
    claims = [("pct", {"pct": 25}), ("shares", {"shares": 25}), ("capital", 25),
              ("capital", {"amount": 25}), ("capital", {"amount": 25, "currency": "MXN"}),
              ("capital", {"amount": 25, "currency": "USD"}),
              ("capital", {"amount": 25, "currency": "$"})]  # fmt: skip
    for other, markers in UNIT_MARKERS.items():
        for marker in markers:
            if other == dimension and dimension != "money":
                continue  # a repeated count word or percent sign is not a conflict of units
            for sep in ("", " "):
                written = f"{literal}{sep}{marker}"
                for attribute, value in claims:
                    assert not _supplies(attribute, value, written), (written, value)


@pytest.mark.parametrize("currency", ["MXN", "USD", "$", "US$"])
@pytest.mark.parametrize("amount", ["-5.000", "- 5.000", "+5.000", "(5.000)"])
def test_metamorphic_a_currency_stranded_before_a_quantity_is_not_dropped(currency, amount):
    written = f"{currency} {amount}"
    for value in (
        -5000,
        5000,
        {"amount": -5000},
        {"amount": 5000},
        {"amount": -5000, "currency": currency},
        {"amount": 5000, "currency": currency},
    ):
        assert not _supplies("capital", value, written), (written, value)  # fmt: skip


@pytest.mark.parametrize(
    "written, attribute, value",
    [
        ("25% según acta", "pct", {"pct": 25}),
        ("25% transmisión de 1.100 acciones", "pct", {"pct": 25}),
        ("25% coparticipación", "pct", {"pct": 25}),
        ("25% shareholders", "pct", {"pct": 25}),
        ("25% europeos", "pct", {"pct": 25}),
        ("25% MXN 1.000", "pct", {"pct": 25}),
        ("25% MXN 1.000", "capital", {"amount": 1000, "currency": "MXN"}),
        ("1.250 acciones 25%", "pct", {"pct": 25}),
        ("1.250 acciones 25%", "shares", {"shares": 1250}),
        ("5.000 acciones Serie C de MXN 1.000", "shares", {"shares": 5000}),
        (
            "5.000 acciones Serie C de MXN 1.000",
            "par_value",
            {"par_value": 1000, "currency": "MXN"},
        ),
        ("participación:25%", "pct", {"pct": 25}),
        ("capital: MXN 6.000.000", "capital", {"amount": 6000000, "currency": "MXN"}),
    ],
)
def test_metamorphic_a_neighbouring_word_or_quantity_is_not_a_unit_continuation(
    written, attribute, value
):
    assert _supplies(attribute, value, written)


# --- review round 6: span ownership, checked as metamorphic relations -------------------------
#
# Whitespace inside a literal is optional in the grammar, so spacing never changes a verdict;
# a parse that fails part-way still owns the span it read, so no inner scalar escapes it.

LITERALS = [("pct", "25%", {"pct": 25}, {"pct": -25}),
            ("shares", "25 acciones", {"shares": 25}, {"shares": -25}),
            ("capital", "MXN 5.000", {"amount": 5000, "currency": "MXN"},
             {"amount": -5000, "currency": "MXN"})]  # fmt: skip


def _spaced(form: str) -> list[str]:
    """Every filling of the optional-whitespace slots `_` with nothing or one space."""
    slots = form.count("_")
    out = []
    for mask in range(2**slots):
        filled = form
        for k in range(slots):
            filled = filled.replace("_", " " if mask >> k & 1 else "", 1)
        out.append(filled)
    return out


@pytest.mark.parametrize("attribute, literal, plus, minus", LITERALS)
@pytest.mark.parametrize(
    "form",
    ["{a}_{b}_{x}", "{a}_{b}_(_{x}_)", "{a}_(_{b}_{x}_)", "{a}_(_{b}_(_{x}_)_)", "{a}_{b}_{c}_{x}"],
)
def test_metamorphic_a_sign_chain_is_one_span_whatever_its_spacing(
    attribute, literal, plus, minus, form
):
    """R6 #1: any two or more signs on one quantity, spaced or not, at any wrapper level."""
    for a, b, c in itertools.product("+-", repeat=3):
        for written in _spaced(form.format(a=a, b=b, c=c, x=literal)):
            for value in (plus, minus):
                assert not _supplies(attribute, value, written), written


@pytest.mark.parametrize(
    "form, sign",
    [("-_25%", -1), ("-_(_25%_)", -1), ("(_-_25%_)", -1), ("-_(_(_25%_)_)", -1), ("+_25%", 1),
     ("+_(_25%_)", 1), ("(_+_25%_)", 1), ("(_25_%_)", 1), ("-_25_acciones", -1)],
)  # fmt: skip
def test_metamorphic_a_single_sign_keeps_its_value_whatever_its_spacing(form, sign):
    attribute = "shares" if "acciones" in form else "pct"
    for written in _spaced(form):
        assert _supplies(attribute, {attribute: 25 * sign}, written), written
        assert not _supplies(attribute, {attribute: -25 * sign}, written), written


@pytest.mark.parametrize("operator", ["a", "al", "to", "y", "-", "\N{HYPHEN}", "\N{EN DASH}"])
@pytest.mark.parametrize(
    "first, second, attribute, values",
    [
        ("10%", "25%", "pct", ({"pct": 10}, {"pct": 25})),
        ("10 acciones", "25 acciones", "shares", ({"shares": 10}, {"shares": 25})),
        ("1.000 MXN", "5.000 MXN", "capital",
         ({"amount": 1000, "currency": "MXN"}, {"amount": 5000, "currency": "MXN"})),
    ],
)  # fmt: skip
def test_metamorphic_a_leftover_marker_never_releases_the_other_endpoint(
    operator, first, second, attribute, values
):
    """R6 #2: wherever a duplicate or conflicting marker sits relative to the wrappers of one
    endpoint, the other endpoint stays a range endpoint."""
    for extra in ("%", " %", " MXN", " acciones", "$"):
        for wrap in ("{}", "({})", "( {} )", "(({}))"):
            placements = (
                (first, wrap.format(second + extra)),  # the marker inside the wrapper
                (first, wrap.format(second) + extra),  # or right after it
                (wrap.format(first + extra), second),
                (wrap.format(first) + extra, second),
            )
            for left, right in placements:
                written = f"{left} {operator} {right}"
                kept = values[0] if right != second else values[1]
                assert not _supplies(attribute, kept, written), (written, kept)


@pytest.mark.parametrize("operator", ["a", "al", "to", "y", "-", "\N{HYPHEN}"])
@pytest.mark.parametrize("opened", ["-_(_{x}", "+_(_{x}", "-_(_(_{x}_)", "- +_(_{x}"])
def test_metamorphic_a_signed_unclosed_endpoint_never_releases_the_other_one(operator, opened):
    """The wrapper a sign opens and nothing closes is owned with the sign, so the operator
    meets that whole endpoint whatever the spacing."""
    for form in _spaced(opened):
        written = f"10% {operator} " + form.format(x="25%") + " de las acciones"
        assert not _supplies("pct", {"pct": 10}, written), written


@pytest.mark.parametrize(
    "owner",
    ["- +({x})", "+ -({x})", "- -{x}", "-({x}", "- ({x}", "-(({x})", "({x}%)", "({x} MXN)",
     "(({x}) %)", "- +({x}) de las acciones", "-({x} de las acciones)"],
)  # fmt: skip
@pytest.mark.parametrize("attribute, literal, plus, minus", LITERALS)
def test_metamorphic_no_inner_scalar_of_an_owned_span_is_exposed(
    owner, attribute, literal, plus, minus
):
    written = owner.format(x=literal)
    for value in (plus, minus):
        assert not _supplies(attribute, value, written), written


@pytest.mark.parametrize(
    "written, attribute, value",
    [
        ("(25% de las acciones)", "pct", {"pct": 25}),
        ("( 25% de las acciones )", "pct", {"pct": 25}),
        ("(-25% de las acciones)", "pct", {"pct": -25}),
        ("((25%) según acta)", "pct", {"pct": 25}),
        ("(socia, 25%)", "pct", {"pct": 25}),
        ("(25% y 1.250 acciones)", "shares", {"shares": 1250}),
        ("(25% y 1.250 acciones)", "pct", {"pct": 25}),
        ("( MXN 5.000 de capital )", "capital", {"amount": 5000, "currency": "MXN"}),
    ],
)
def test_metamorphic_an_unsigned_unclosed_parenthesis_stays_prose(written, attribute, value):
    assert _supplies(attribute, value, written)


# --- review round 7: structural extent, checked as metamorphic relations ----------------------
#
# Putting an endpoint inside any parenthetical (its own wrapper, prose around it, a malformed
# unit, a sign, nesting, a group left open) never changes whether an operator joins it to a
# same-unit neighbour; without a neighbour, the parenthetical changes nothing at all.

ENDPOINTS = [("pct", "10%", "25%", "%", {"pct": 10}, {"pct": 25}),
             ("shares", "10 acciones", "25 acciones", " acciones", {"shares": 10}, {"shares": 25}),
             ("capital", "1.000 MXN", "5.000 MXN", " MXN", {"amount": 1000, "currency": "MXN"},
              {"amount": 5000, "currency": "MXN"})]  # fmt: skip
BALANCED = ["({x})", "({x}{e})", "({x} de las acciones)", "({x}{e} según acta)",
            "(({x}) de las acciones)", "(({x}{e}) según acta)", "( ( {x} ) según acta )",
            "-({x} de las acciones)", "-({x}{e} según acta)", "- ( {x} según acta )",
            "(socia, {x})"]  # fmt: skip
# A group left open faces away from the other endpoint: an unmatched `(` runs to the end of the
# quote and an unmatched `)` from its start, so on the left the open side is the `)`.
GROUPS = [(form, form) for form in BALANCED] + [
    ("({x}", "{x})"), ("({x}{e}", "{x}{e})"), ("({x} de las acciones", "{x} de las acciones)"),
    ("-({x} según acta", "-({x} según acta"),
]  # fmt: skip


@pytest.mark.parametrize("operator", ["a", "al", "to", "y", "-", "\N{HYPHEN}", "\N{EN DASH}"])
@pytest.mark.parametrize("right_form, left_form", GROUPS)
@pytest.mark.parametrize("attribute, first, second, extra, low, high", ENDPOINTS)
def test_metamorphic_a_parenthetical_never_hides_the_range_operator(
    operator, right_form, left_form, attribute, first, second, extra, low, high
):
    for written in (
        f"{first} {operator} " + right_form.format(x=second, e=extra),  # the group on the right
        left_form.format(x=first, e=extra) + f" {operator} {second}",  # or on the left
    ):
        for value in (low, high):
            assert not _supplies(attribute, value, written), (written, value)


@pytest.mark.parametrize(
    "group", ["({x})", "({x} de las acciones)", "(({x}) según acta)", "( {x} según acta )",
              "(socia, {x})", "({x} de las acciones", "({x}) según acta"],
)  # fmt: skip
@pytest.mark.parametrize("attribute, first, second, extra, low, high", ENDPOINTS)
def test_metamorphic_a_standalone_parenthetical_keeps_its_value(
    group, attribute, first, second, extra, low, high
):
    if attribute == "capital" and "({x})" in group:  # money's own wrapper: accounting (R2)
        high = {**high, "amount": -high["amount"]}
    assert _supplies(attribute, high, group.format(x=second))


@pytest.mark.parametrize(
    "written",
    ["10% y (1.250 acciones de la serie A)", "(10% de las acciones) y 1.250 acciones",
     "(1.250 acciones según acta) y 10%", "10% y ((1.250 acciones) según acta)",
     "(10% según acta) y (1.250 acciones de la serie A)", "10% a (1.250 acciones de x"],
)  # fmt: skip
def test_metamorphic_unlike_units_stay_independent_through_parentheses(written):
    assert _supplies("pct", {"pct": 10}, written)
    assert _supplies("shares", {"shares": 1250}, written)


def _supplies_exactly(attribute, value, excerpt):
    return not _excerpt_errors(claim(subject="Irene", attribute=attribute, value=value,
                                     excerpt=excerpt))  # fmt: skip


# R8: where the quote starts must not matter. The subject may come first or last, and leading
# or trailing whitespace normalises away, so the same value may sit at offset 0 or later.
PLACEMENTS = ["Irene {core}", "{core} Irene", "  {core} Irene", "\t{core}\nIrene",
              "Irene {core}  ", "{core}, según Irene"]  # fmt: skip
RANGE_CORES = ["{lo} de las acciones) {op} {hi}", "{lo}{e} de las acciones) {op} {hi}",
               "({lo}) según acta) {op} {hi}", "{lo} {op} {hi} de las acciones)",
               "({lo} de las acciones) {op} {hi}", "{lo} {op} ({hi} de las acciones)",
               "{lo} {op} ({hi}{e} según acta", "{lo} {op} {hi}"]  # fmt: skip


@pytest.mark.parametrize("operator", ["a", "al", "to", "y", "-", "\N{HYPHEN}", "\N{EN DASH}"])
@pytest.mark.parametrize("core", RANGE_CORES)
@pytest.mark.parametrize("attribute, first, second, extra, low, high", ENDPOINTS)
def test_metamorphic_a_range_verdict_does_not_depend_on_where_the_quote_starts(
    operator, core, attribute, first, second, extra, low, high
):
    body = core.format(lo=first, hi=second, e=extra, op=operator)
    for placement in PLACEMENTS:
        excerpt = placement.format(core=body)
        for value in (low, high):
            assert not _supplies_exactly(attribute, value, excerpt), (excerpt, value)


@pytest.mark.parametrize(
    "core, attribute, value",
    [
        ("25% de las acciones)", "pct", {"pct": 25}),
        ("(25% de las acciones)", "pct", {"pct": 25}),
        ("(-25% de las acciones)", "pct", {"pct": -25}),
        ("25%", "pct", {"pct": 25}),
        ("10% de las acciones) y 1.250 acciones", "pct", {"pct": 10}),
        ("10% de las acciones) y 1.250 acciones", "shares", {"shares": 1250}),
        ("(1.250 acciones según acta) y 10%", "pct", {"pct": 10}),
        ("1.250 acciones según acta) y 10%", "shares", {"shares": 1250}),
        ("(MXN 5.000)", "capital", {"amount": -5000, "currency": "MXN"}),
        ("MXN 5.000 según acta)", "capital", {"amount": 5000, "currency": "MXN"}),
    ],
)
def test_metamorphic_a_standalone_or_unlike_unit_value_does_not_depend_on_where_the_quote_starts(
    core, attribute, value
):
    for placement in PLACEMENTS:
        excerpt = placement.format(core=core)
        assert _supplies_exactly(attribute, value, excerpt), excerpt


@pytest.mark.parametrize(
    "core, attribute, value, supplied",
    [
        ("25%", "pct", {"pct": 25}, True),  # whole-token edges at the quote's start and end
        ("x25%", "pct", {"pct": 25}, False),
        ("123-456XYZ", "shares", {"shares": 123}, False),
        ("N\N{HYPHEN}123", "shares", {"shares": 123}, False),
        ("MXN -5.000", "capital", {"amount": -5000}, False),  # the stranded-currency floor
        ("-MXN 5.000", "capital", {"amount": -5000, "currency": "MXN"}, True),
        ("2026-10-25", "dob", "2026-10-25", True),  # the date token edges
        ("N\N{HYPHEN}2026-10-25", "dob", "2026-10-25", False),
        ("- +25%", "pct", {"pct": 25}, False),  # a sign chain, the trailing-marker loop
        ("25%%", "pct", {"pct": 25}, False),
        ("-(25% de las acciones", "pct", {"pct": 25}, False),  # the closing-parenthesis search
        ("(25% de las acciones", "pct", {"pct": 25}, True),
        ("10% a 25%", "pct", {"pct": 10}, False),  # the range gap and its dash test
        ("10%-25%", "pct", {"pct": 25}, False),
        ("10% -25%", "pct", {"pct": 10}, False),
    ],
)
def test_metamorphic_every_boundary_site_treats_the_quote_start_like_any_other_position(
    core, attribute, value, supplied
):
    """R8's audit: each site that reads a position or an interval gives the same verdict
    whether the value starts the quote (offset 0 after normalisation) or follows the subject."""
    for placement in PLACEMENTS:
        excerpt = placement.format(core=core)
        assert _supplies_exactly(attribute, value, excerpt) is supplied, excerpt


def test_metamorphic_a_signed_group_owns_its_contents_and_an_unsigned_one_does_not():
    unsigned, signed = "(10% de 25% de las acciones)", "-(10% de 25% de las acciones)"
    assert _supplies("pct", {"pct": 25}, unsigned) and _supplies("pct", {"pct": 10}, unsigned)
    for value in ({"pct": 25}, {"pct": -25}, {"pct": 10}, {"pct": -10}):
        assert not _supplies("pct", value, signed), value
