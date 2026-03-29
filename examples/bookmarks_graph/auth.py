"""OAuth 2.0 PKCE flow for Twitter API v2."""

import base64
import hashlib
import json
import secrets
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

CONFIG_PATH = Path.home() / ".bookmarks_graph.json"
TOKEN_URL = "https://api.twitter.com/2/oauth2/token"
AUTH_URL = "https://twitter.com/i/oauth2/authorize"
REDIRECT_URI = "http://localhost:8080/callback"
SCOPES = "tweet.read users.read bookmark.read offline.access"


def _generate_pkce():
    code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()
    code_challenge = base64.urlsafe_b64encode(
        hashlib.sha256(code_verifier.encode()).digest()
    ).rstrip(b"=").decode()
    return code_verifier, code_challenge


def _save_config(data: dict):
    existing = {}
    if CONFIG_PATH.exists():
        existing = json.loads(CONFIG_PATH.read_text())
    existing.update(data)
    CONFIG_PATH.write_text(json.dumps(existing, indent=2))


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    return json.loads(CONFIG_PATH.read_text())


class _CallbackHandler(BaseHTTPRequestHandler):
    code = None

    def do_GET(self):
        params = parse_qs(urlparse(self.path).query)
        _CallbackHandler.code = params.get("code", [None])[0]
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"<html><body><h2>Auth complete. Return to your terminal.</h2></body></html>")

    def log_message(self, *args):
        pass  # suppress server logs


def authenticate(client_id: str) -> str:
    """Run OAuth 2.0 PKCE flow, save token, return access token."""
    code_verifier, code_challenge = _generate_pkce()
    state = secrets.token_urlsafe(16)

    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPES,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    auth_url = f"{AUTH_URL}?{urlencode(params)}"

    print(f"\nOpening browser for Twitter authorization...")
    print(f"If browser doesn't open, visit:\n{auth_url}\n")
    webbrowser.open(auth_url)

    server = HTTPServer(("localhost", 8080), _CallbackHandler)
    server.handle_request()  # blocks until one request arrives

    code = _CallbackHandler.code
    if not code:
        raise RuntimeError("No authorization code received from Twitter.")

    resp = httpx.post(
        TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": REDIRECT_URI,
            "client_id": client_id,
            "code_verifier": code_verifier,
        },
    )
    resp.raise_for_status()
    token_data = resp.json()

    _save_config(
        {
            "client_id": client_id,
            "access_token": token_data["access_token"],
            "refresh_token": token_data.get("refresh_token"),
        }
    )
    return token_data["access_token"]


def refresh_access_token(client_id: str, refresh_tok: str) -> str:
    resp = httpx.post(
        TOKEN_URL,
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_tok,
            "client_id": client_id,
        },
    )
    resp.raise_for_status()
    token_data = resp.json()
    _save_config(
        {
            "access_token": token_data["access_token"],
            "refresh_token": token_data.get("refresh_token", refresh_tok),
        }
    )
    return token_data["access_token"]


def get_valid_token() -> str:
    """Return a valid access token, attempting refresh if needed."""
    config = load_config()
    token = config.get("access_token")
    if not token:
        raise RuntimeError("Not authenticated. Run: bookmarks auth")

    # Try to refresh proactively if we have a refresh token
    refresh_tok = config.get("refresh_token")
    client_id = config.get("client_id")
    if refresh_tok and client_id:
        try:
            return refresh_access_token(client_id, refresh_tok)
        except Exception:
            pass  # fall back to existing token

    return token
