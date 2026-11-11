"""Pure, offline profile comparisons do not claim replay admission."""

import pytest

from kyb.evals.renderer import COMPONENTS, Comparison, compare


def profile():
    result = {key: "observed" for key in COMPONENTS}
    for key in ("executable_sha256", "font_set_sha256", "font_config_sha256"):
        result[key] = "a" * 64
    return result


def test_identical_complete_profiles_match():
    result = compare(profile(), profile())
    assert result.state is Comparison.MATCH
    assert result.actual_hash == result.expected_hash


@pytest.mark.parametrize("component", COMPONENTS)
def test_every_component_difference_is_a_mismatch(component):
    expected, actual = profile(), profile()
    actual[component] = "b" * 64 if component.endswith("sha256") else "different"
    result = compare(actual, expected)
    assert result.state is Comparison.MISMATCH
    assert result.components == (component,)
    assert result.actual_hash != result.expected_hash


@pytest.mark.parametrize("bad", [None, {}, {"unexpected": "data"}])
def test_missing_or_unrecognized_profile_is_unknown(bad):
    assert compare(bad, profile()).state is Comparison.UNKNOWN


def test_malformed_hash_is_unknown():
    actual = profile()
    actual["font_set_sha256"] = "unmeasured"
    assert compare(actual, profile()).state is Comparison.UNKNOWN
