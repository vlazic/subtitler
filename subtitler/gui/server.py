"""The only module that binds a socket or talks to a browser.

Everything above it is a pure function of a request, so this file stays small enough to
read: bind loopback, mint a token, hand the URL to the browser, and translate between
`BaseHTTPRequestHandler` and `GuiApp.handle`.
"""

from __future__ import annotations

import secrets
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from subtitler import __version__
from subtitler.gui.app import GuiApp, Response

DEFAULT_HOST = "127.0.0.1"
TOKEN_HEADER = "X-Subtitler-Token"

_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "0.0.0.0"})


def is_local_host_header(value: str | None) -> bool:
    """Guard against DNS rebinding, which is the one real risk of a local HTTP server.

    A page on the internet cannot read a cross-origin response, but it *can* make the
    browser resolve its own hostname to 127.0.0.1 and then talk to us same-origin. The
    token stops that on its own; rejecting a `Host` we did not expect is the cheap second
    lock, and it costs a string comparison.

    IPv6 is why this is not a one-liner: the header is `[::1]:53211`, brackets and all,
    and a bare `rsplit(":")` on it yields `[::1`.
    """
    if not value:
        return False
    host = value.strip()
    if host.startswith("["):
        host = host[1:].partition("]")[0]
    elif host.count(":") == 1:
        host = host.rsplit(":", 1)[0]
    return host.lower() in _LOCAL_HOSTS


class _Handler(BaseHTTPRequestHandler):
    server_version = f"subtitler/{__version__}"
    protocol_version = "HTTP/1.1"
    app: GuiApp  # set on the subclass built in `serve`

    def do_GET(self) -> None:  # BaseHTTPRequestHandler's naming, not ours
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def _dispatch(self, method: str) -> None:
        if not is_local_host_header(self.headers.get("Host")):
            self._send(Response(403, b'{"error":"non-local Host header"}'))
            return
        parts = urlsplit(self.path)
        query = {k: v[0] for k, v in parse_qs(parts.query).items()}
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        token = self.headers.get(TOKEN_HEADER) or query.get("t")
        # Lowercased, because HTTP header names are case-insensitive and `GuiApp` is a
        # plain mapping lookup away from the wire.
        incoming = {key.lower(): value for key, value in self.headers.items()}
        try:
            response = self.app.handle(
                method, parts.path, query, body, token=token, headers=incoming
            )
        except Exception as exc:  # a 500 with a message beats a dead socket
            response = Response(500, f'{{"error":"{type(exc).__name__}: {exc}"}}'.encode())
        self._send(response)

    def _send(self, response: Response) -> None:
        self.send_response(response.status)
        self.send_header("Content-Type", response.content_type)
        self.send_header("Content-Length", str(len(response.body)))
        for name, value in response.headers:
            self.send_header(name, value)
        # The page is regenerated on every launch during development and must never be
        # served from a stale cache after an upgrade.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(response.body)

    def log_message(self, fmt: str, *args: Any) -> None:
        """Silence the per-request access log.

        The page polls once a second; the default handler would bury the one line the user
        actually needs, which is the URL.
        """


def build_server(
    *,
    host: str = DEFAULT_HOST,
    port: int = 0,
    app: GuiApp | None = None,
    token: str | None = None,
) -> tuple[ThreadingHTTPServer, GuiApp]:
    """A bound server and its app, without starting the loop or opening a browser.

    Split out so tests can drive the real thing over real HTTP on an ephemeral port.
    """
    gui = app or GuiApp(token=token if token is not None else secrets.token_urlsafe(16))
    handler = type("SubtitlerHandler", (_Handler,), {"app": gui})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server, gui


def serve(
    *,
    host: str = DEFAULT_HOST,
    port: int = 0,
    open_browser: bool = True,
    home: Path | None = None,
) -> int:
    token = secrets.token_urlsafe(16)
    try:
        server, _gui = build_server(host=host, port=port, app=GuiApp(token=token, home=home))
    except OSError as exc:
        print(f"error: cannot listen on {host}:{port} ({exc})", file=sys.stderr)
        return 1

    bound_host, bound_port = server.server_address[0], server.server_address[1]
    display_host = "127.0.0.1" if bound_host in {"0.0.0.0", "::"} else bound_host
    url = f"http://{display_host}:{bound_port}/?t={token}"

    if host not in _LOCAL_HOSTS or host == "0.0.0.0":
        print(
            "warning: this server can browse and write files anywhere this user can. "
            f"You bound it to {host}, which is reachable from other machines.",
            file=sys.stderr,
        )

    print(f"subtitler gui is running.\n\n    {url}\n", file=sys.stderr)
    print("Leave this window open while you use it. Press Ctrl-C to stop.", file=sys.stderr)

    if open_browser and not _open_browser(url):
        print(
            "\nCould not open a browser automatically. Copy the address above into one.",
            file=sys.stderr,
        )

    thread = threading.Thread(target=server.serve_forever, name="subtitler-gui", daemon=True)
    thread.start()
    try:
        thread.join()
    except KeyboardInterrupt:
        print("\nstopping.", file=sys.stderr)
    finally:
        server.shutdown()
        server.server_close()
    return 0


def _open_browser(url: str) -> bool:
    """`webbrowser` raises on a machine with no browser at all; that is not a crash.

    On a headless box, or over SSH with no DISPLAY, the right outcome is the URL printed
    above and a sentence telling the user to open it themselves.
    """
    try:
        return webbrowser.open(url)
    except (webbrowser.Error, OSError):
        return False
