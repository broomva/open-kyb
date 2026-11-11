"""Real temporary-file observations, with no renderer execution or host binding."""

import json
from pathlib import Path

import pytest

from kyb.evals.renderer import (
    Comparison,
    RendererIncomparable,
    capture_file_identity,
    compare,
)


@pytest.fixture
def files(tmp_path):
    paths = []
    for name, content in [
        ("pdftoppm", b"binary-one"),
        ("pdftocairo", b"binary-two"),
        ("private-font-name.otf", b"font-one"),
        ("other-font.ttf", b"font-two"),
        ("private-fonts.conf", b"active config"),
    ]:
        path = tmp_path / name
        path.write_bytes(content)
        paths.append(path)
    return {
        "renderers": dict(zip(("pdftoppm", "pdftocairo"), paths[:2], strict=True)),
        "fonts": paths[2:4],
        "configs": paths[4:],
    }


def test_real_file_identity_is_order_stable_private_and_incomplete(files, tmp_path):
    result = capture_file_identity(**files)
    reordered = {**files, "fonts": list(reversed(files["fonts"]))}
    assert capture_file_identity(**reordered) == result
    rendered = json.dumps(result)
    assert str(tmp_path) not in rendered
    assert "private-font" not in rendered
    assert "font-one" not in rendered
    assert all(len(value) == 64 for value in result.values())
    assert compare(result, result).state is Comparison.UNKNOWN


@pytest.mark.parametrize(
    ("category", "index", "field"),
    [
        ("renderers", "pdftoppm", "executable_sha256"),
        ("renderers", "pdftocairo", "executable_sha256"),
        ("fonts", 0, "font_set_sha256"),
        ("fonts", 1, "font_set_sha256"),
        ("configs", 0, "font_config_sha256"),
    ],
)
def test_every_observed_file_changes_its_identity(files, category, index, field):
    before = capture_file_identity(**files)
    files[category][index].write_bytes(b"different bytes")
    after = capture_file_identity(**files)
    assert before[field] != after[field]
    assert all(before[key] == after[key] for key in before if key != field)


def test_binary_role_swap_is_detected(files):
    before = capture_file_identity(**files)
    a, b = files["renderers"].values()
    files["renderers"] = {"pdftoppm": b, "pdftocairo": a}
    assert capture_file_identity(**files)["executable_sha256"] != before["executable_sha256"]


def test_font_symlink_alias_is_not_counted_as_another_file(files, tmp_path):
    before = capture_file_identity(**files)
    alias = tmp_path / "alias.otf"
    alias.symlink_to(files["fonts"][0])
    files["fonts"].append(alias)
    assert capture_file_identity(**files) == before


@pytest.mark.parametrize("category", ["fonts", "configs", "renderers"])
def test_absent_required_file_groups_refuse(files, category):
    files[category] = {} if category == "renderers" else []
    with pytest.raises(RendererIncomparable):
        capture_file_identity(**files)


def test_unknown_renderer_role_refuses(files):
    files["renderers"]["unreviewed"] = files["fonts"][0]
    with pytest.raises(RendererIncomparable):
        capture_file_identity(**files)


@pytest.mark.parametrize("kind", ["missing", "directory", "relative", "not-path"])
def test_invalid_file_refuses_without_exposing_private_paths(files, tmp_path, kind):
    files["fonts"][0] = {
        "missing": tmp_path / "private-missing.otf",
        "directory": tmp_path,
        "relative": Path("private-relative.otf"),
        "not-path": "private-file.otf",
    }[kind]
    with pytest.raises(RendererIncomparable) as exc:
        capture_file_identity(**files)
    assert str(tmp_path) not in str(exc.value)
    assert "private-" not in str(exc.value)


def test_read_error_is_redacted(files, monkeypatch):
    def denied(*args, **kwargs):
        raise PermissionError("private-home/private-font-name.otf")

    monkeypatch.setattr(Path, "open", denied)
    with pytest.raises(RendererIncomparable) as exc:
        capture_file_identity(**files)
    assert "private" not in str(exc.value)


def test_changed_earlier_file_during_observation_refuses(files, monkeypatch):
    import hashlib

    original = hashlib.file_digest
    calls = 0

    def changing(stream, algorithm):
        nonlocal calls
        calls += 1
        result = original(stream, algorithm)
        if calls == 2:
            files["fonts"][1].write_bytes(b"changed during observation")
        return result

    monkeypatch.setattr(hashlib, "file_digest", changing)
    with pytest.raises(RendererIncomparable):
        capture_file_identity(**files)
