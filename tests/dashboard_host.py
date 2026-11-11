"""Offline MCP Apps protocol probe, not a production host or native-question implementation.

Runs the real stdio server and HTML resource. Only dashboard reads/preparation are proxied;
UI messages are captured, never executed by an agent. No provider calls or shared config.
"""

import asyncio
import json
import os
import secrets
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO = Path(__file__).resolve().parents[1]
# Filled from the server's advertised `ui.visibility`, as a conforming Apps host would.
ALLOWED: set[str] = set()


async def main(root: Path) -> None:
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "kyb.mcp_server"],
        cwd=str(REPO),
        env={**os.environ, "KYB_WORKSPACE": str(root)},
    )
    async with (
        stdio_client(parameters) as (read, write),
        ClientSession(
            read,
            write,
            extensions={"io.modelcontextprotocol/ui": {"mimeTypes": ["text/html;profile=mcp-app"]}},
        ) as session,
    ):
        await session.discover()
        ALLOWED.update(
            tool.name
            for tool in (await session.list_tools()).tools
            if "app" in ((tool.meta or {}).get("ui") or {}).get("visibility", ["model", "app"])
        )
        resource = await session.read_resource("ui://kyb/orchestration-dashboard")
        app_html = resource.contents[0].text
        initial = await session.call_tool("open_dashboard", {})
        initial_json = json.dumps(initial.model_dump(mode="json", by_alias=True, exclude_none=True))
        loop = asyncio.get_running_loop()
        token = secrets.token_urlsafe(24)

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def send(self, body: bytes, mime="text/html", status=200):
                self.send_response(status)
                self.send_header("Content-Type", mime + "; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if not self.path.startswith(f"/{token}/"):
                    self.send(b"Not found", status=404)
                    return
                if self.path.endswith("resource"):
                    self.send(app_html.encode())
                    return
                page = (
                    (REPO / "tests/fixtures/dashboard-host.html")
                    .read_text()
                    .replace("INITIAL_RESULT", initial_json.replace("<", "\\u003c"))
                )
                self.send(page.encode())

            def do_POST(self):
                expected = f"http://127.0.0.1:{self.server.server_port}"
                if (
                    self.path != f"/{token}/rpc"
                    or self.headers.get("Origin") != expected
                    or self.headers.get("Content-Type") != "application/json"
                ):
                    self.send(b'{"error":"Refused"}', "application/json", 403)
                    return
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length < 16384:
                    self.send(b'{"error":"Request too large"}', "application/json", 400)
                    return
                args = json.loads(self.rfile.read(length))
                if args["name"] not in ALLOWED:
                    self.send(b'{"error":"Tool not allowed in probe"}', "application/json", 403)
                    return
                future = asyncio.run_coroutine_threadsafe(
                    session.call_tool(args["name"], args.get("arguments", {})), loop
                )
                result = future.result(timeout=20)
                self.send(
                    json.dumps(
                        result.model_dump(mode="json", by_alias=True, exclude_none=True)
                    ).encode(),
                    "application/json",
                )

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"http://127.0.0.1:{server.server_port}/{token}/", flush=True)
        try:
            await asyncio.Event().wait()
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1])))
