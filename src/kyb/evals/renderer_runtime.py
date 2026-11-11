"""Observe the origin Mac renderer without exporting paths or granting replay.

Only the admitted runner may perform heavy capture/revalidation. This observer
uses native loader traces and library version APIs, never package-manager claims.
Foreign/unsupported platforms explicitly remain incomparable.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

from kyb.docintel import poppler
from kyb.evals.renderer import Comparison, RendererIncomparable, capture_file_identity, compare

_LIBS = {
    "cairo": "libcairo.",
    "freetype": "libfreetype.",
    "fontconfig": "libfontconfig.",
    "libpng": "libpng16.",
    "lcms": "liblcms2.",
    "openjpeg": "libopenjp2.",
    "pixman": "libpixman-1.",
    "harfbuzz": "libharfbuzz.",
}
_TRACE = re.compile(r"^dyld\[\d+\]: <([0-9A-Fa-f-]+)> (/.+)$", re.M)
_VERSION = re.compile(r"^(pdftoppm|pdftocairo) version (\d+(?:\.\d+)+)$", re.M)
_ENV = frozenset(("HOME", "PATH", "LANG", "TMPDIR", "FREETYPE_PROPERTIES"))


def _command(args: list[str], env: dict[str, str]) -> str:
    result = subprocess.run(args, env=env, capture_output=True, text=True, check=True, timeout=20)
    return result.stdout + "\n" + result.stderr


def _loaded(tool: Path, env: dict[str, str]) -> tuple[dict[Path, str], str]:
    output = _command(
        [str(tool), "-v" if tool.name.startswith("pdfto") else "-V"],
        {**env, "DYLD_PRINT_LIBRARIES": "1"},
    )
    images: dict[Path, str] = {}
    for image_id, raw in _TRACE.findall(output):
        path = Path(raw)
        if path in images and images[path] != image_id:
            raise ValueError
        images[path] = image_id
    if tool.resolve(strict=True) not in images:
        raise ValueError
    matches = _VERSION.findall(output)
    if tool.name.startswith("pdfto"):
        if len(matches) != 1 or matches[0][0] != tool.name:
            raise ValueError
        return images, matches[0][1]
    return images, ""


def _library_version(name: str, path: Path) -> str:
    library = ctypes.CDLL(str(path))
    if name == "freetype":
        handle = ctypes.c_void_p()
        library.FT_Init_FreeType.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        library.FT_Init_FreeType.restype = ctypes.c_int
        library.FT_Done_FreeType.argtypes = [ctypes.c_void_p]
        library.FT_Done_FreeType.restype = ctypes.c_int
        library.FT_Library_Version.argtypes = [
            ctypes.c_void_p,
            *([ctypes.POINTER(ctypes.c_int)] * 3),
        ]
        library.FT_Library_Version.restype = None
        if library.FT_Init_FreeType(ctypes.byref(handle)) != 0 or not handle.value:
            raise ValueError
        parts = [ctypes.c_int() for _ in range(3)]
        try:
            library.FT_Library_Version(handle, *(ctypes.byref(p) for p in parts))
        finally:
            if library.FT_Done_FreeType(handle) != 0:
                raise ValueError
        value = ".".join(str(p.value) for p in parts)
    elif name in {"fontconfig", "lcms"}:
        fn = getattr(library, "FcGetVersion" if name == "fontconfig" else "cmsGetEncodedCMMversion")
        fn.argtypes, fn.restype = [], ctypes.c_int
        number = fn()
        if number <= 0:
            raise ValueError
        value = (
            f"{number // 10000}.{number // 100 % 100}.{number % 100}"
            if name == "fontconfig"
            else f"encoded:{number}"
        )
    else:
        symbol = {
            "cairo": "cairo_version_string",
            "libpng": "png_get_libpng_ver",
            "openjpeg": "opj_version",
            "pixman": "pixman_version_string",
            "harfbuzz": "hb_version_string",
        }[name]
        fn = getattr(library, symbol)
        fn.argtypes = [ctypes.c_void_p] if name == "libpng" else []
        fn.restype = ctypes.c_char_p
        raw = fn(None) if name == "libpng" else fn()
        if not raw:
            raise ValueError
        value = raw.decode("ascii")
    if not re.fullmatch(r"(?:encoded:)?\d+(?:\.\d+)*", value):
        raise ValueError
    return value


def _observe(tools: dict[str, Path], env: dict[str, str]) -> dict:
    images: dict[Path, str] = {}
    versions = {}
    font_library = None
    for role, tool in tools.items():
        loaded, version = _loaded(tool, env)
        for path, image_id in loaded.items():
            if path in images and images[path] != image_id:
                raise ValueError
            images[path] = image_id
        if role in {"pdftoppm", "pdftocairo"}:
            versions[role] = version
        candidates = {p.resolve(strict=True) for p in loaded if p.name.startswith("libfontconfig.")}
        if len(candidates) != 1 or (font_library is not None and candidates != {font_library}):
            raise ValueError
        font_library = candidates.pop()
    if versions["pdftoppm"] != versions["pdftocairo"]:
        raise ValueError
    libraries = {}
    library_versions = {}
    for name, prefix in _LIBS.items():
        candidates = {p.resolve(strict=True) for p in images if p.name.startswith(prefix)}
        if name == "harfbuzz" and not candidates:
            # The traced Poppler/font tools need not link HarfBuzz. An installed
            # package version would invent a runtime dependency. Record only
            # bounded startup absence; this does not prove absence of dlopen
            # during rendering or admit any historical binding.
            library_versions[name] = "not-in-startup-images"
            continue
        if len(candidates) != 1:
            raise ValueError
        libraries[name] = candidates.pop()
    library_versions.update(
        {name: _library_version(name, path) for name, path in libraries.items()}
    )
    fonts = {
        Path(line)
        for line in _command([str(tools["fc-list"]), "--format", "%{file}\\n"], env).splitlines()
        if line
    }
    configs = set()
    for line in _command([str(tools["fc-conflist"])], env).splitlines():
        if not line:
            continue
        if not line.startswith(("+ /", "- /")) or ": " not in line:
            raise ValueError
        if line.startswith("+ "):
            configs.add(Path(line[2:].split(": ", 1)[0]))
    # Apple shared-cache images can lack readable standalone files. Bind their
    # actual loader UUIDs plus the OS build, rather than inventing file hashes.
    system = sorted(
        (str(p), ident)
        for p, ident in images.items()
        if str(p).startswith(("/usr/lib/", "/System/Library/"))
    )
    dependencies = {p for p in images if not str(p).startswith(("/usr/lib/", "/System/Library/"))}
    return {
        "versions": library_versions,
        "poppler": versions["pdftoppm"],
        "fonts": sorted(fonts),
        "configs": sorted(configs),
        "dependencies": sorted(dependencies),
        "system": system,
    }


def capture_runtime_identity() -> dict[str, str]:
    """Observe a complete origin profile; no compatible binding or PASS follows.

    Invoke inside the exact admitted render job, both before and after rendering.
    The pure tests mock native probes; they do not establish origin compatibility.
    """
    stage = "context"
    try:
        if platform.system() != "Darwin" or any(
            key.startswith(("DYLD_", "LD_")) for key in os.environ
        ):
            raise ValueError
        env = {
            k: os.environ[k]
            for k in os.environ
            if k in _ENV or k.startswith(("LC_", "XDG_", "FONTCONFIG_"))
        }
        if not env.get("HOME") or not env.get("PATH"):
            raise ValueError
        tools = {}
        for name in ("pdftoppm", "pdftocairo", "fc-list", "fc-conflist"):
            found = shutil.which(name, path=env["PATH"])
            if not found:
                raise ValueError
            tools[name] = Path(found).absolute()
        stage = "native-probe"
        first = _observe(tools, env)
        renderers = {name: tools[name] for name in ("pdftoppm", "pdftocairo")}

        def files(observation):
            return capture_file_identity(
                renderers,
                fonts=observation["fonts"],
                configs=observation["configs"],
                dependencies=observation["dependencies"],
            )

        stage = "file-fingerprint"
        fingerprints = files(first)
        stage = "native-probe"
        second = _observe(tools, env)
        stage = "drift"
        if first != second or fingerprints != files(second):
            raise ValueError
        # HOME/PATH/cache locations select observed files but are not rendering
        # semantics themselves. Hashing ephemeral job paths would make identical
        # measured stacks incompatible on every fresh runner job.
        settings = {
            k: v
            for k, v in env.items()
            if k in {"LANG", "FREETYPE_PROPERTIES"} or k.startswith("LC_")
        }
        env_hash = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
        config_hash = hashlib.sha256(
            (fingerprints["font_config_sha256"] + env_hash).encode()
        ).hexdigest()
        system_hash = hashlib.sha256(
            json.dumps(first["system"], sort_keys=True).encode()
        ).hexdigest()
        profile = {
            **first["versions"],
            **fingerprints,
            "font_config_sha256": config_hash,
            "platform": f"Darwin:{platform.release()}:{platform.mac_ver()[0]}:{system_hash}",
            "architecture": platform.machine(),
            "poppler": first["poppler"],
            "dpi": str(poppler.DPI),
            "render_flags": "pdftoppm:png,singlefile,page(f,l),region(x,y,W,H)",
        }
        stage = "profile-schema"
        if compare(profile, profile).state is not Comparison.MATCH:
            raise ValueError
        return profile
    except (
        OSError,
        ValueError,
        RuntimeError,
        TypeError,
        AttributeError,
        KeyError,
        subprocess.SubprocessError,
    ):
        raise RendererIncomparable(
            "runtime renderer identity unavailable or changed", category=stage
        ) from None
