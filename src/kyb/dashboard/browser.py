"""Loopback-only browser fallback. A random capability scopes it to one workspace.

Only inspection and plan preparation are exposed. There is no execution/signing endpoint.
"""

from __future__ import annotations

import base64
import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from kyb.dashboard import data
from kyb.dashboard.contracts import DashboardSnapshot, IntentRequest, RetryRequest, SnapshotQuery

_SERVERS: dict[Path, DashboardServer] = {}
_LOCK = threading.Lock()


def app_html(
    preview_snapshots: list[DashboardSnapshot] | None = None, *, inline: bool = False
) -> str:
    assets = files("kyb.dashboard").joinpath("assets")
    html = (
        assets.joinpath("index.html")
        .read_text()
        .replace("/* DASHBOARD_CSS */", assets.joinpath("dashboard.css").read_text())
        .replace("/* DASHBOARD_JS */", assets.joinpath("dashboard.js").read_text())
        .replace("/* MATERIAL_JS */", assets.joinpath("materials.js").read_text())
        .replace("/* ORB_SHADER */", assets.joinpath("orb.glsl").read_text())
    )
    for key, name in (
        ("RALEWAY", "Raleway-300"),
        ("INTER", "Inter-400"),
        ("MONO", "JetBrainsMono-400"),
    ):
        font = assets.joinpath("fonts", name + ".woff2").read_bytes()
        html = html.replace(f"/* {key}_FONT */", base64.b64encode(font).decode())
    if preview_snapshots is not None:
        payload = json.dumps(
            {"snapshots": [s.model_dump(mode="json") for s in preview_snapshots]}
        ).replace("<", "\\u003c")
        html = html.replace(
            '<script id="orb-source"',
            f'<script id="preview-data" type="application/json">{payload}</script>'
            '<script id="orb-source"',
            1,
        )
    if inline:
        start, end = html.index("<style>") + 7, html.index("</style>")
        css = html[start:end]
        for old, new in {
            "--bg": "--background",
            "--ink": "--foreground",
            "--secondary": "--secondary-foreground",
            "--meta": "--muted-foreground",
            "--action": "--primary",
            "--surface": "--card",
            "--line": "--border",
            "--good": "--success",
            "--warn": "--warning",
            "--bad": "--destructive",
            "--metal": "--ring",
            "--accent": "--accent-foreground",
        }.items():
            css = css.replace(old, new)
        # Keep the frame's side gutters inline; only the top margin collapses into the chat.
        css += "body{background:transparent!important}#app{padding-top:var(--s1)!important}"
        html = html[:start] + css + html[end:]
    return html


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.token = secrets.token_urlsafe(32)
        super().__init__(("127.0.0.1", 0), Handler)

    @property
    def origin(self) -> str:
        return f"http://127.0.0.1:{self.server_port}"

    @property
    def url(self) -> str:
        return f"{self.origin}/{self.token}/"


class Handler(BaseHTTPRequestHandler):
    server: DashboardServer

    def log_message(self, *_args: object) -> None:
        pass  # The URL contains a local capability: never log it.

    def _send(self, status: int, body: bytes, mime: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", mime + "; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; "
            "script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
            "img-src data:; connect-src 'self'; font-src data:; "
            "frame-ancestors 'none'; base-uri 'none'; form-action 'none'",
        )
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _route(self) -> tuple[str, dict]:
        url = urlsplit(self.path)
        if self.headers.get("Host") != self.server.origin.removeprefix(
            "http://"
        ) or not url.path.startswith(f"/{self.server.token}/"):
            raise PermissionError("unavailable local dashboard")
        origin = self.headers.get("Origin")
        if origin and origin != self.server.origin:
            raise PermissionError("cross-origin dashboard access refused")
        return url.path[len(self.server.token) + 2 :], parse_qs(url.query)

    def do_GET(self) -> None:
        try:
            route, query = self._route()
            if not route:
                self._send(200, app_html().encode(), "text/html")
                return
            if route != "snapshot":
                self._send(404, b'{"error":"not found"}')
                return
            if any(len(values) != 1 for values in query.values()):
                raise ValueError("Each context field must have one value.")
            args = SnapshotQuery.model_validate({key: values[0] for key, values in query.items()})
            data.check_workspace(self.server.root, args.workspace_id)
            result = data.snapshot(self.server.root, args.case, args.run)
            self._send(200, json.dumps(result, ensure_ascii=False).encode())
        except (*data.EXPECTED, KeyError, TypeError) as exc:
            self._send(400, json.dumps({"error": str(exc)}).encode())

    def do_POST(self) -> None:
        try:
            route, _ = self._route()
            if route not in {"prepare", "intent"}:
                self._send(404, b'{"error":"not found"}')
                return
            if (
                self.headers.get("Origin") != self.server.origin
                or self.headers.get("Content-Type") != "application/json"
            ):
                raise PermissionError("same-origin JSON request required")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 16384:
                raise ValueError("invalid request size")
            body = self.rfile.read(length)
            if route == "prepare":
                args = RetryRequest.model_validate_json(body)
                result = data.prepare(self.server.root, **args.model_dump())
            else:
                args = IntentRequest.model_validate_json(body)
                result = data.intent(self.server.root, **args.model_dump())
            self._send(200, json.dumps(result, ensure_ascii=False).encode())
        except (*data.EXPECTED, KeyError, TypeError) as exc:
            self._send(400, json.dumps({"error": str(exc)}).encode())


def serve(root: Path) -> DashboardServer:
    root = root.resolve()
    with _LOCK:
        if root not in _SERVERS:
            instance = DashboardServer(root)
            threading.Thread(
                target=instance.serve_forever, daemon=True, name="kyb-dashboard"
            ).start()
            _SERVERS[root] = instance
        return _SERVERS[root]
