"""poppler is plumbing, not OCR (spec): page renders, the text layer, and metadata.

Untrusted PDFs are only ever passed to these binaries as file arguments, with no shell.
"""

from __future__ import annotations

import re
import subprocess
from functools import cache
from pathlib import Path

DPI = 300
_TIMEOUT = 120


def _run(args: list[str]) -> str:
    return subprocess.run(args, check=True, capture_output=True, timeout=_TIMEOUT).stdout.decode(
        "utf-8", "replace"
    )


@cache
def version() -> str:
    out = subprocess.run(["pdftotext", "-v"], capture_output=True, timeout=10)
    first = (out.stderr or out.stdout).decode().splitlines()[0]
    return "poppler " + first.split()[-1]


def info(pdf: Path) -> dict[str, str]:
    """`pdfinfo` as a dict. Keys we rely on: Pages, Producer, JavaScript, Encrypted, Form."""
    fields: dict[str, str] = {}
    for line in _run(["pdfinfo", str(pdf)]).splitlines():
        key, sep, value = line.partition(":")
        if sep:
            fields[key.strip()] = value.strip()
    return fields


def page_count(pdf: Path) -> int:
    return int(info(pdf)["Pages"])


def text_layer(pdf: Path, page: int) -> str:
    return _run(["pdftotext", "-layout", "-f", str(page), "-l", str(page), str(pdf), "-"])


_LINE = re.compile(
    r'<line xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">(.*?)</line>', re.S
)
_WORD = re.compile(r"<word[^>]*>(.*?)</word>", re.S)


def line_boxes(pdf: Path, page: int) -> list[tuple[str, tuple[int, int, int, int]]]:
    """The text layer's lines with their boxes, in pixels of the 300-dpi render: (text, bbox).
    Position comes from the layer, so a hidden or white-on-white line is still located."""
    import html

    xml = _run(["pdftotext", "-bbox-layout", "-f", str(page), "-l", str(page), str(pdf), "-"])
    k, out = DPI / 72, []
    for m in _LINE.finditer(xml):
        text = " ".join(html.unescape(w) for w in _WORD.findall(m[5])).strip()
        x0, y0, x1, y1 = (float(v) * k for v in m.groups()[:4])
        if text:
            out.append((text, (int(x0), int(y0), int(x1) + 1, int(y1) + 1)))
    return out


def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))


def render(pdf: Path, page: int, out_png: Path, dpi: int = DPI) -> Path:
    """Render one page to `out_png` (the path must end in .png)."""
    if out_png.suffix != ".png":
        raise ValueError("render target must be a .png path")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    prefix = out_png.with_suffix("")
    _run(["pdftoppm", "-r", str(dpi), "-f", str(page), "-l", str(page), "-singlefile", "-png",
          str(pdf), str(prefix)])  # fmt: skip
    return out_png


def render_region(pdf: Path, page: int, box: tuple[int, int, int, int], out_png: Path,
                  dpi: int = DPI) -> Path:  # fmt: skip
    """Render the pixel box [x0, y0, x1, y1] (at `dpi`) of one page to `out_png`."""
    if out_png.suffix != ".png":
        raise ValueError("render target must be a .png path")
    x0, y0, x1, y1 = box
    out_png.parent.mkdir(parents=True, exist_ok=True)
    _run(["pdftoppm", "-r", str(dpi), "-f", str(page), "-l", str(page), "-x", str(x0),
          "-y", str(y0), "-W", str(x1 - x0), "-H", str(y1 - y0), "-singlefile", "-png",
          str(pdf), str(out_png.with_suffix(""))])  # fmt: skip
    return out_png


def png_size(data: bytes) -> tuple[int, int]:
    """(width, height) from a PNG's IHDR chunk."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
