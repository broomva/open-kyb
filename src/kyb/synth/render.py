"""Render a CaseSpec into a packet folder shaped like examples/packets/*.

Five Spanish PDFs (Helvetica text, no third-party PDF library), a `manifest.json` with the case id,
label, fired rules and injections. About 20% of the documents (`Doc.scan`) are degraded into
image-only scans: rasterised, then rotated, noised and blurred with ImageMagick. If `pdftoppm` or
`magick` is missing the document stays a text PDF and the manifest says so. Output goes under the
cache (`~/.cache/kyb/synth/packets/<case_id>/`), never the repo.
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

from kyb.synth.case import CaseSpec, Doc
from kyb.synth.label import Label

FILENAMES = {
    "incorporation": "01_estatutos_actualizados",
    "ubo_declaration": "02_ubo_declaration",
    "share_register": "03_registro_de_accionistas",
    "proof_of_address": "04_proof_of_address",
    "bank_reference": "05_bank_reference",
}
WRAP = 96
LINES_PER_PAGE = 52


# --- a minimal PDF writer ------------------------------------------------------------------------


def _esc(s: str) -> bytes:
    raw = s.encode("cp1252", "replace")
    return raw.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")


def _wrap(text: str, width: int = WRAP) -> list[str]:
    out, cur = [], ""
    for word in text.split():
        if len(cur) + len(word) + 1 > width:
            out.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    return [*out, cur] if cur or not out else out


def pdf_bytes(lines: list[tuple[str, bool]], title: str) -> bytes:
    """(text, bold) lines → a PDF, paginated. Standard fonts only (Helvetica, WinAnsi)."""
    wrapped: list[tuple[str, bool]] = []
    for text, bold in lines:
        wrapped += [(w, bold) for w in (_wrap(text) if text else [""])]
    pages = [wrapped[i : i + LINES_PER_PAGE] for i in range(0, len(wrapped), LINES_PER_PAGE)] or [
        []
    ]
    objs: list[bytes] = [
        b"",
        b"",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>",
    ]  # 1 catalog, 2 pages, 3-4 fonts
    kids = []
    for page in pages:
        ops = [b"BT", b"/F1 10 Tf", b"13 TL", b"50 800 Td"]
        for text, bold in page:
            ops.append(b"/F2 10 Tf" if bold else b"/F1 10 Tf")
            ops.append(b"(" + _esc(text) + b") Tj T*")
        ops.append(b"ET")
        stream = b"\n".join(ops)
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        c_id = len(objs)
        objs.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font "
            b"<< /F1 3 0 R /F2 4 0 R >> >> /Contents %d 0 R >>" % c_id
        )
        kids.append(len(objs))
    objs[0] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objs[1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (
        b" ".join(b"%d 0 R" % k for k in kids),
        len(kids),
    )
    objs.append(b"<< /Title (%s) /Producer (open-kyb synth) >>" % _esc(title))
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R /Info %d 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objs) + 1,
        len(objs),
        xref,
    )
    return bytes(out)


# --- document text (Spanish, in the real packets' structure) -------------------------------------


def _holders_table(case: CaseSpec, stated: dict[str, int] | None) -> list[tuple[str, bool]]:
    rows = [("#  Titular · participación · domicilio · beneficiario final", True)]
    for i, h in enumerate(case.holders, start=1):
        if stated is not None and h.name not in stated:
            continue
        pct = stated[h.name] if stated is not None else h.pct
        who = (
            f"{h.person.name}, {h.person.nationality}, nacido el {h.person.dob.isoformat()}"
            if h.person is not None
            else "no identificado"
        )
        via = f" (a través de {' → '.join(h.chain)})" if h.chain else ""
        kind = {
            "natural": "persona natural",
            "legal": "persona jurídica",
            "foundation": "fundación",
            "trust": "fideicomiso",
        }[h.kind]
        reg = f", reg. {h.reg_no}" if h.reg_no else ""
        rows.append(
            (
                f"{i}  {h.name} ({kind}{reg}) · {pct}% · {h.domicile}{via} · "
                f"beneficiario final: {who}",
                False,
            )
        )
    return rows


def lines_for(case: CaseSpec, doc: Doc) -> list[tuple[str, bool]]:
    c = case.company
    issued = doc.issued.isoformat() if doc.issued else "s/f"
    head = f"{c.name} · Identificación {c.tax_id} · {c.jurisdiction}"
    if doc.kind == "incorporation":
        legal = next((o.name for o in case.officers if o.role in ("legal_rep", "director")), "—")
        addr = doc.claims.address or c.registered_address
        out = [
            ("CERTIFICADO DE EXISTENCIA Y REPRESENTACIÓN LEGAL", True),
            (f"Fecha de expedición: {issued}", False),
            ("", False),
            (f"Razón social: {c.name}", False),
            (f"Identificación: {doc.claims.reg_no or c.tax_id} · Estado: {c.status}", False),
            (f"Domicilio principal: {addr}", False),
            (f"Fecha de constitución: {c.incorporated.isoformat()}", False),
            ("Objeto social: " + "; ".join(c.activities), False),
            (f"Representación legal: {legal}", False),
        ]
        if doc.claims.holders:
            out += [("Socios fundadores:", True)] + [
                (f"  {n}: {p}%", False) for n, p in doc.claims.holders.items()
            ]
        if doc.stated_validity_days:
            out += [
                ("", False),
                (
                    f"Este certificado es válido por {doc.stated_validity_days} días "
                    "contados desde su expedición.",
                    False,
                ),
            ]
        return out
    if doc.kind == "ubo_declaration":
        sig = "Firmado" if doc.signed else "SIN FIRMA"
        return [
            ("DECLARACIÓN DE BENEFICIARIOS FINALES", True),
            (head, False),
            (f"Fecha: {issued}", False),
            ("", False),
            (
                "Declaro bajo juramento que las personas naturales que poseen o controlan el 25% "
                "o más de la entidad son:",
                False,
            ),
            *_holders_table(case, doc.claims.holders),
            ("", False),
            (f"Nota: número de registro declarado {doc.claims.reg_no or c.tax_id}", False),
            (
                f"{sig}: {next((o.name for o in case.officers if o.person), 'Representante')}"
                f" — {issued}",
                False,
            ),
        ]
    if doc.kind == "share_register":
        return [
            ("LIBRO DE REGISTRO DE ACCIONISTAS", True),
            (head, False),
            (f"Extracto expedido el {issued}", False),
            ("", False),
            *_holders_table(case, doc.claims.holders),
        ]
    if doc.kind == "proof_of_address":
        return [
            ("COMPROBANTE DE DOMICILIO", True),
            (head, False),
            (f"Fecha: {issued}", False),
            ("", False),
            (f"Se certifica el domicilio: {doc.claims.address or c.registered_address}", False),
            (
                "Domicilio operativo: "
                + (c.operating_address or "no declarado (solo domicilio fiscal)"),
                False,
            ),
        ]
    web = (
        "Reporta empleados en nómina y sitio web."
        if c.employees_or_website
        else ("No reporta empleados ni sitio web.")
    )
    return [
        ("CERTIFICACIÓN BANCARIA", True),
        (f"{c.jurisdiction}, {issued}", False),
        (
            f"Certificamos que {c.name}, {c.tax_id}, es titular de una cuenta activa con manejo "
            f"normal. {web}",
            False,
        ),
    ]


# --- scans ---------------------------------------------------------------------------------------


def degrade(pdf: Path, rng: random.Random) -> bool:
    """Replace `pdf` with an image-only scan (rasterise, rotate, noise, blur). False if the tools
    are missing or fail; the text PDF is left in place."""
    ppm, magick = shutil.which("pdftoppm"), shutil.which("magick") or shutil.which("convert")
    if not (ppm and magick):
        return False
    with tempfile.TemporaryDirectory() as td:
        try:
            subprocess.run(
                [ppm, "-r", "110", "-png", str(pdf), f"{td}/p"],
                check=True,
                capture_output=True,
                timeout=60,
            )
            pages = sorted(Path(td).glob("p*.png"))
            scans = []
            for k, png in enumerate(pages):
                out = Path(td) / f"s{k}.png"
                angle = rng.choice((-1, 1)) * rng.uniform(0.6, 1.8)
                subprocess.run(
                    [
                        magick,
                        str(png),
                        "-seed",
                        str(rng.randrange(1 << 30)),
                        "-background",
                        "white",
                        "-rotate",
                        f"{angle:.2f}",
                        "-attenuate",
                        "0.25",
                        "+noise",
                        "Gaussian",
                        "-blur",
                        "0x0.7",
                        "-colorspace",
                        "Gray",
                        "-quality",
                        "70",
                        str(out),
                    ],
                    check=True,
                    capture_output=True,
                    timeout=60,
                )
                scans.append(str(out))
            subprocess.run([magick, *scans, str(pdf)], check=True, capture_output=True, timeout=60)
        except subprocess.SubprocessError, OSError:
            return False
    return True


# --- a packet ------------------------------------------------------------------------------------


def render_case(case: CaseSpec, lab: Label, out_root: Path, *, seed: int = 0) -> Path:
    folder = out_root / case.case_id
    folder.mkdir(parents=True, exist_ok=True)
    rng = random.Random(f"scan/{seed}/{case.case_id}")
    files, degraded, absent = [], [], []
    for doc in case.documents:
        if not doc.present:
            absent.append(doc.kind)
            continue
        name = FILENAMES[doc.kind]
        path = folder / f"{name}.pdf"
        path.write_bytes(pdf_bytes(lines_for(case, doc), f"{name} {case.case_id}"))
        if doc.scan and degrade(path, rng):
            scanned = folder / f"{name}_SCAN.pdf"
            path.replace(scanned)
            path = scanned
            degraded.append(path.name)
        files.append(path.name)
    manifest = {
        "case_id": case.case_id,
        "as_of": case.as_of.isoformat(),
        "policy": "2026.11-r1",
        "label": {
            "status": lab.status,
            "score": lab.score,
            "fired": [
                {"rule": f.rule, "severity": f.severity, "points": f.points, "reason": f.reason}
                for f in lab.fired
            ],
            "notes": lab.notes,
        },
        "injections": [{"name": i.name, **i.params} for i in case.injections],
        "files": files,
        "scans": degraded,
        "absent_documents": absent,
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    return folder


def render_all(cases: list[CaseSpec], labels: list[Label], out_root: Path, *, seed: int = 0) -> int:
    for c, lab in zip(cases, labels, strict=True):
        render_case(c, lab, out_root, seed=seed)
    return len(cases)
