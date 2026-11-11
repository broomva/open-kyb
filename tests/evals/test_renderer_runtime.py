"""Mocked native observations and real tiny file hashes; no host-renderer proof."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from kyb.evals import renderer_runtime as runtime
from kyb.evals.renderer import Comparison, RendererIncomparable, compare, require_legacy_binding


@pytest.fixture
def origin(tmp_path, monkeypatch):
    for key in list(runtime.os.environ):
        if key.startswith(("DYLD_", "LD_")):
            monkeypatch.delenv(key)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("GEMINI_API_KEY", "sentinel-never-forwarded")
    monkeypatch.setattr(runtime.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(runtime.platform, "release", lambda: "25.0.0")
    monkeypatch.setattr(runtime.platform, "mac_ver", lambda: ("26.0", (), ""))
    monkeypatch.setattr(runtime.platform, "machine", lambda: "arm64")
    for name in ["pdftoppm", "pdftocairo", "fc-list", "fc-conflist", "font", "config", "library"]:
        (tmp_path / name).write_bytes(name.encode())
    monkeypatch.setattr(runtime.shutil, "which", lambda name, **kw: str(tmp_path / name))
    observed = {
        "versions": {name: "1.2.3" for name in runtime._LIBS},
        "poppler": "25.05.0",
        "fonts": [tmp_path / "font"],
        "configs": [tmp_path / "config"],
        "dependencies": [tmp_path / "library"],
        "system": [("/usr/lib/system.dylib", "A-B-C")],
    }

    def observe(tools, env):
        assert "GEMINI_API_KEY" not in env
        assert env["HOME"] == str(tmp_path)
        return deepcopy(observed)

    monkeypatch.setattr(runtime, "_observe", observe)
    return observed


def test_observed_profile_is_complete_private_and_does_not_admit_replay(origin, tmp_path):
    result = runtime.capture_runtime_identity()
    assert compare(result, result).state is Comparison.MATCH
    assert str(tmp_path) not in repr(result)
    assert "sentinel" not in repr(result)
    assert runtime.capture_runtime_identity() == result
    with pytest.raises(RendererIncomparable):
        require_legacy_binding()


@pytest.mark.parametrize("system", ["Linux", "Windows", "unknown"])
def test_unsupported_host_refuses_before_any_probe(origin, monkeypatch, system):
    monkeypatch.setattr(runtime.platform, "system", lambda: system)
    monkeypatch.setattr(runtime, "_observe", lambda *args: pytest.fail("unexpected probe"))
    with pytest.raises(RendererIncomparable):
        runtime.capture_runtime_identity()


@pytest.mark.parametrize("variable", ["DYLD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES", "LD_PRELOAD"])
def test_loader_overrides_refuse(origin, monkeypatch, variable):
    monkeypatch.setenv(variable, "/private/override")
    with pytest.raises(RendererIncomparable):
        runtime.capture_runtime_identity()


@pytest.mark.parametrize("missing", ["HOME", "PATH"])
def test_missing_runtime_context_refuses(origin, monkeypatch, missing):
    monkeypatch.delenv(missing)
    with pytest.raises(RendererIncomparable):
        runtime.capture_runtime_identity()


def test_missing_tool_refuses(origin, monkeypatch):
    monkeypatch.setattr(runtime.shutil, "which", lambda *args, **kwargs: None)
    with pytest.raises(RendererIncomparable):
        runtime.capture_runtime_identity()


@pytest.mark.parametrize("component", ["versions", "poppler", "fonts", "configs", "system"])
def test_observation_drift_refuses(origin, monkeypatch, component, tmp_path):
    second = deepcopy(origin)
    second[component] = {"cairo": "changed"} if component == "versions" else "changed"
    observations = iter([origin, second])
    monkeypatch.setattr(runtime, "_observe", lambda *args: next(observations))
    with pytest.raises(RendererIncomparable):
        runtime.capture_runtime_identity()


def test_dependency_content_changes_profile(origin, tmp_path):
    first = runtime.capture_runtime_identity()
    (tmp_path / "library").write_bytes(b"different linked library")
    second = runtime.capture_runtime_identity()
    assert first["executable_sha256"] != second["executable_sha256"]


def test_ephemeral_job_paths_do_not_change_an_identical_observed_stack(origin, monkeypatch):
    monkeypatch.setattr(runtime, "_observe", lambda *args: deepcopy(origin))
    first = runtime.capture_runtime_identity()
    monkeypatch.setenv("HOME", "/another/job/home")
    monkeypatch.setenv("PATH", "/another/job/path")
    monkeypatch.setenv("TMPDIR", "/another/job/tmp")
    assert runtime.capture_runtime_identity() == first


@pytest.mark.parametrize("setting", ["LANG", "LC_ALL", "FREETYPE_PROPERTIES"])
def test_semantic_render_setting_changes_identity(origin, monkeypatch, setting):
    first = runtime.capture_runtime_identity()
    monkeypatch.setenv(setting, "changed-setting")
    second = runtime.capture_runtime_identity()
    assert first["font_config_sha256"] != second["font_config_sha256"]


def test_probe_failure_is_redacted(origin, monkeypatch):
    def failed(*args):
        raise OSError("/private/customer/font.otf")

    monkeypatch.setattr(runtime, "_observe", failed)
    with pytest.raises(RendererIncomparable) as exc:
        runtime.capture_runtime_identity()
    assert "private" not in str(exc.value)


def test_loader_trace_requires_matching_executable_and_single_version(tmp_path, monkeypatch):
    tool = tmp_path / "pdftoppm"
    tool.write_bytes(b"synthetic")
    good = f"dyld[12]: <A-B-C> {tool}\npdftoppm version 25.05.0\n"
    monkeypatch.setattr(runtime, "_command", lambda *args: good)
    images, version = runtime._loaded(tool, {})
    assert images == {tool: "A-B-C"} and version == "25.05.0"
    for bad in [
        "pdftoppm version 25.05.0",
        good + "pdftoppm version 26.0\n",
        good.replace("pdftoppm version", "pdftocairo version"),
        good + f"dyld[12]: <D-E-F> {tool}\n",
    ]:
        monkeypatch.setattr(runtime, "_command", lambda *args, value=bad: value)
        with pytest.raises(ValueError):
            runtime._loaded(tool, {})


@pytest.mark.parametrize(
    ("name", "symbol", "returned", "expected"),
    [
        ("cairo", "cairo_version_string", b"1.18.4", "1.18.4"),
        ("libpng", "png_get_libpng_ver", b"1.6.58", "1.6.58"),
        ("openjpeg", "opj_version", b"2.5.4", "2.5.4"),
        ("fontconfig", "FcGetVersion", 21701, "2.17.1"),
        ("lcms", "cmsGetEncodedCMMversion", 2180, "encoded:2180"),
    ],
)
def test_native_version_api_result_is_used_without_package_versions(
    monkeypatch, name, symbol, returned, expected
):
    def fn(*args):
        return returned

    monkeypatch.setattr(runtime.ctypes, "CDLL", lambda path: SimpleNamespace(**{symbol: fn}))
    assert runtime._library_version(name, Path("/synthetic/library")) == expected


def test_freetype_library_lifecycle_version_and_cleanup(monkeypatch):
    calls = []

    def init(handle):
        handle._obj.value = 123
        return 0

    def version(handle, major, minor, patch):
        major._obj.value, minor._obj.value, patch._obj.value = 2, 14, 3

    def done(handle):
        calls.append("done")
        return 0

    monkeypatch.setattr(
        runtime.ctypes,
        "CDLL",
        lambda path: SimpleNamespace(
            FT_Init_FreeType=init, FT_Library_Version=version, FT_Done_FreeType=done
        ),
    )
    assert runtime._library_version("freetype", Path("/synthetic/library")) == "2.14.3"
    assert calls == ["done"]


@pytest.fixture
def native_probe(tmp_path, monkeypatch):
    tools = {name: tmp_path / name for name in ["pdftoppm", "pdftocairo", "fc-list", "fc-conflist"]}
    libraries = [tmp_path / (prefix + "1.dylib") for prefix in runtime._LIBS.values()]
    font, config = tmp_path / "font.otf", tmp_path / "active.conf"
    for file in [*tools.values(), *libraries, font, config]:
        file.write_bytes(file.name.encode())
    images = {path: f"A-B-{index}" for index, path in enumerate(libraries)}

    def loaded(tool, env):
        return {**images, tool: "ROOT"}, "25.05.0"

    def command(args, env):
        if args[0] == str(tools["fc-list"]):
            assert args[1:] == ["--format", "%{file}\\n"]
            return f"{font}\n{font}\n"
        assert args == [str(tools["fc-conflist"])]
        return f"+ {config}: active\n- /not/active.conf: disabled\n"

    monkeypatch.setattr(runtime, "_loaded", loaded)
    monkeypatch.setattr(runtime, "_command", command)
    monkeypatch.setattr(runtime, "_library_version", lambda *args: "1.2.3")
    return tools, images, font, config


def test_observer_uses_unique_active_fonts_configs_and_loaded_versions(native_probe):
    tools, images, font, config = native_probe
    result = runtime._observe(tools, {})
    assert result["fonts"] == [font]
    assert result["configs"] == [config]
    assert result["versions"] == dict.fromkeys(runtime._LIBS, "1.2.3")
    assert set(images) <= set(result["dependencies"])


@pytest.mark.parametrize(
    "component", ["cairo", "freetype", "fontconfig", "libpng", "lcms", "openjpeg", "pixman"]
)
def test_observer_missing_required_library_refuses(native_probe, component):
    tools, images, _, _ = native_probe
    path = next(path for path in images if path.name.startswith(runtime._LIBS[component]))
    del images[path]
    with pytest.raises(ValueError):
        runtime._observe(tools, {})


def test_observer_ambiguous_loaded_library_refuses(native_probe):
    tools, images, font, _ = native_probe
    duplicate = font.parent / "libpng16.other.dylib"
    duplicate.write_bytes(b"another loaded version")
    images[duplicate] = "OTHER"
    with pytest.raises(ValueError):
        runtime._observe(tools, {})


def test_observer_conflicting_poppler_versions_refuse(native_probe, monkeypatch):
    tools, images, _, _ = native_probe
    monkeypatch.setattr(
        runtime,
        "_loaded",
        lambda tool, env: (
            {**images, tool: "ROOT"},
            "25.05.0" if tool.name == "pdftoppm" else "26.0.0",
        ),
    )
    with pytest.raises(ValueError):
        runtime._observe(tools, {})


def test_unloaded_harfbuzz_is_reported_without_inventing_an_installed_version(
    native_probe, monkeypatch
):
    tools, images, _, _ = native_probe
    hb = next(path for path in images if path.name.startswith("libharfbuzz."))
    del images[hb]
    called = []

    def version(name, path):
        called.append(name)
        return "1.2.3"

    monkeypatch.setattr(runtime, "_library_version", version)
    observed = runtime._observe(tools, {})
    assert observed["versions"]["harfbuzz"] == "not-in-startup-images"
    assert "harfbuzz" not in called
    assert "pixman" in called


def test_ambiguous_harfbuzz_remains_blocking(native_probe):
    tools, images, font, _ = native_probe
    another = font.parent / "libharfbuzz.other.dylib"
    another.write_bytes(b"another image")
    images[another] = "OTHER-HB"
    with pytest.raises(ValueError):
        runtime._observe(tools, {})


def test_unreadable_observed_harfbuzz_cannot_be_relabeled_as_absent(native_probe):
    tools, images, _, _ = native_probe
    hb = next(path for path in images if path.name.startswith("libharfbuzz."))
    hb.unlink()
    with pytest.raises(OSError):
        runtime._observe(tools, {})


def test_absent_startup_harfbuzz_is_distinct_from_a_loaded_version(origin):
    loaded = runtime.capture_runtime_identity()
    origin["versions"]["harfbuzz"] = "not-in-startup-images"
    absent = runtime.capture_runtime_identity()
    assert compare(loaded, absent).state is Comparison.MISMATCH
