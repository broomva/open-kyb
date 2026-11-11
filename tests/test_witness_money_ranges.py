"""A scalar amount cannot bind an endpoint of a monetary range (L7 R2 repro)."""

import pytest

from kyb.witness.bind import binding


def money(amount, text):
    claim = {"type": "attribute", "subject": "Irene", "attribute": "capital",
             "value": {"amount": amount, "currency": "MXN"}}  # fmt: skip
    reread = {"values": [{"subject": "Irene", "attribute": "capital", "value": text}]}
    return binding(claim, reread, "Irene capital: " + text)


def test_l7_exact_binary_dash_is_not_a_negative_scalar():
    assert not money(-5000, "1.000 - 5.000 MXN")


@pytest.mark.parametrize("text", ["1.000 - (5.000 MXN)", "MXN1.000 - (MXN5.000)"])
def test_parenthesized_endpoint_remains_a_range(text):
    assert not money(-5000, text)
    assert not money(5000, text)


@pytest.mark.parametrize("separator", ["-", "–", "a", "al", "to", "y"])
@pytest.mark.parametrize(
    "template",
    [
        "1.000 {s} -5.000 MXN",
        "1.000 MXN {s} -5.000",
        "1.000 {s} ((5.000 MXN))",
        "1.000 MXN {s} ((-MXN5.000))",
        "(-1.000 MXN) {s} (-5.000 MXN)",
    ],
)
@pytest.mark.parametrize("amount", [-5000, 1000, 5000])
def test_r1_signed_and_nested_parenthesized_range_operands_fail_closed(separator, template, amount):
    assert not money(amount, template.format(s=separator))


@pytest.mark.parametrize("amount", [1000.50, "1000.50", "1.000,50"])
def test_decimal_grouping_is_one_monetary_operand(amount):
    assert money(amount, "MXN 1.000,50")


def test_separate_share_count_and_par_value_still_bind():
    claim = {"type": "capital", "subject": "Irene", "attribute": "total_shares",
             "value": {"shares": 5000, "series": "C", "par_value": 1000,
                       "currency": "MXN"}}  # fmt: skip
    reread = {"values": [{"subject": "Irene", "attribute": "Acciones",
                          "value": "5.000 acciones Serie C de MXN 1.000"}]}  # fmt: skip
    assert binding(claim, reread)


@pytest.mark.parametrize("separator", ["-", "–", "—", "a", "al", "to", "y"])
@pytest.mark.parametrize(
    "template",
    [
        "MXN 1.000 {s} 5.000",
        "1.000 {s} 5.000 MXN",
        "MXN 1.000 {s} MXN 5.000",
        "1.000 MXN {s} 5.000 MXN",
        "MXN1.000{s}MXN5.000",
        "1.000MXN{s}5.000MXN",
    ],
)
@pytest.mark.parametrize("amount", [1000, 5000, -5000])
def test_currency_placement_cannot_hide_another_range_operand(separator, template, amount):
    assert not money(amount, template.format(s=separator))


@pytest.mark.parametrize(
    "text",
    [
        "-MXN5000",
        "- MXN5000",
        "-MXN 5.000",
        "- MXN 5.000",
        "(5000MXN)",
        "(MXN5000)",
        "(5.000 MXN)",
        "(MXN 5.000)",
    ],
)
def test_genuine_negative_money_still_binds(text):
    assert money(-5000, text)
    assert not money(5000, text)


@pytest.mark.parametrize(
    ("value", "attribute", "text"),
    [
        ({"pct": 25}, "pct", "(25%)"),
        ({"pct": -25}, "pct", "- 25%"),
        ({"shares": 25}, "shares", "(25 acciones)"),
    ],
)
def test_holding_parentheses_and_unary_minus_are_unchanged(value, attribute, text):
    claim = {"type": "owns", "subject": "Irene", "value": value}
    reread = {"values": [{"subject": "Irene", "attribute": attribute, "value": text}]}
    assert binding(claim, reread, "Irene " + text)
