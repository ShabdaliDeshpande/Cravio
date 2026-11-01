"""OAuth login for Swiggy MCP: dynamic client registration, PKCE, phone + OTP in the browser.

The MCP SDK's OAuthClientProvider does the protocol work. This module supplies the two
pieces it leaves to the host app: somewhere to keep tokens, and a localhost page that
catches the redirect after the user signs in.
"""

import asyncio
import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from mcp.client.auth import OAuthClientProvider
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken

# Swiggy matches redirect URIs exactly, so the port is fixed. The URI is registered on
# first login; changing it later needs a new registration (delete the token file).
CALLBACK_HOST = "localhost"
CALLBACK_PORT = int(os.getenv("SWIGGY_OAUTH_CALLBACK_PORT", "3030"))
CALLBACK_PATH = "/callback"
REDIRECT_URI = f"http://{CALLBACK_HOST}:{CALLBACK_PORT}{CALLBACK_PATH}"

# Holds a live token for the user's Swiggy account. Kept out of git via .gitignore.
TOKEN_FILE = Path(os.getenv("SWIGGY_TOKEN_FILE", ".cravio/swiggy_oauth.json"))

LOGIN_TIMEOUT_SECONDS = 300


class FileTokenStorage:
    """Keeps the registered client and its tokens in one JSON file.

    All Swiggy servers share one OAuth resource (https://mcp.swiggy.com), so one
    login serves Food, Instamart and Dineout.
    """

    def __init__(self, path: Path = TOKEN_FILE):
        self.path = path

    def _read(self) -> dict:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, key: str, value: dict) -> None:
        data = self._read()
        data[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    async def get_tokens(self) -> OAuthToken | None:
        raw = self._read().get("tokens")
        return OAuthToken.model_validate(raw) if raw else None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        self._write("tokens", tokens.model_dump(mode="json", exclude_none=True))

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        raw = self._read().get("client_info")
        return OAuthClientInformationFull.model_validate(raw) if raw else None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        self._write("client_info", client_info.model_dump(mode="json", exclude_none=True))


class _CallbackCatcher:
    """One-shot localhost server that receives the authorization code."""

    def __init__(self) -> None:
        self._done = threading.Event()
        self._params: dict[str, str] = {}
        self._server: HTTPServer | None = None

    def start(self) -> None:
        # Reset so a second login in the same process (e.g. a revoked refresh token) works.
        self._done.clear()
        self._params = {}
        catcher = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                url = urlparse(self.path)
                if url.path != CALLBACK_PATH:
                    self.send_error(404)
                    return
                catcher._params = {k: v[0] for k, v in parse_qs(url.query).items()}
                ok = "code" in catcher._params
                self.send_response(200 if ok else 400)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                message = "Signed in to Swiggy. You can close this tab." if ok else "Sign-in failed."
                self.wfile.write(f"<p>{message}</p>".encode())
                catcher._done.set()

            def log_message(self, *args) -> None:
                pass

        self._server = HTTPServer((CALLBACK_HOST, CALLBACK_PORT), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    async def wait(self) -> tuple[str, str | None]:
        try:
            arrived = await asyncio.to_thread(self._done.wait, LOGIN_TIMEOUT_SECONDS)
        finally:
            if self._server:
                self._server.shutdown()
                self._server.server_close()
        if not arrived:
            raise TimeoutError(f"No Swiggy sign-in within {LOGIN_TIMEOUT_SECONDS}s.")
        if "code" not in self._params:
            raise RuntimeError(f"Swiggy sign-in failed: {self._params.get('error', 'no code returned')}")
        return self._params["code"], self._params.get("state")


def build_oauth_provider(server_url: str) -> OAuthClientProvider:
    """httpx auth for one Swiggy MCP server. Opens the browser only when no valid token exists."""
    catcher = _CallbackCatcher()

    async def open_browser(authorization_url: str) -> None:
        # Listen before the browser can redirect back.
        catcher.start()
        print(f"\nSign in to Swiggy in your browser (phone + OTP).\nIf it didn't open: {authorization_url}\n")
        webbrowser.open(authorization_url)

    return OAuthClientProvider(
        server_url=server_url,
        client_metadata=OAuthClientMetadata(
            client_name="Cravio (dev)",
            redirect_uris=[REDIRECT_URI],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="none",
            scope="mcp:tools",
        ),
        storage=FileTokenStorage(),
        redirect_handler=open_browser,
        callback_handler=catcher.wait,
        timeout=LOGIN_TIMEOUT_SECONDS,
    )
