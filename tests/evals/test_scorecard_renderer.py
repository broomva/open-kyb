"""The v2 scorecard must not turn unresolved renderer evidence into success."""

import pytest

from kyb.evals.scorecard import Case, Suite, build, failed, incomparable, passed, pending


def card(cases, *, strict=False, blocking=True):
    return build(
        [Suite("s", "test", blocking, cases)],
        false_approve=0,
        inputs={},
        strict=strict,
        live=False,
    )


@pytest.mark.parametrize(
    ("cases", "state"),
    [
        ([passed("p")], "pass"),
        ([pending("p", "wait")], "pending"),
        ([pending("p", "wait"), incomparable("i", "unknown")], "incomparable"),
        ([failed("f", "bad"), incomparable("i", "unknown")], "fail"),
    ],
)
def test_suite_priority(cases, state):
    assert Suite("s", "test", True, cases).status == state


def test_incomparable_blocks_non_strict_and_is_excluded_from_executed():
    result = card([passed("p"), pending("w", "wait"), incomparable("i", "unknown")])
    assert result["schema"] == "kyb-verify-scorecard-v2"
    assert result["verdict"] == "fail"
    assert result["headline"].startswith("1/1 blocking cases pass")
    assert result["suites"][0]["counts"]["incomparable"] == 1
    assert result["incomparable_suites"] == ["s"]


def test_report_only_incomparability_also_blocks():
    assert card([incomparable("i", "unknown")], blocking=False)["verdict"] == "fail"


def test_strict_pending_still_blocks():
    assert card([pending("p", "wait")], strict=True)["verdict"] == "fail"
    assert card([pending("p", "wait")], strict=False)["verdict"] == "pass"


def test_unknown_status_refused_even_when_other_case_fails():
    with pytest.raises(ValueError, match="unknown case status"):
        card([failed("f", "bad"), Case("bad", "unknown")])
