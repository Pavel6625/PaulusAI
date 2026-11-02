"""OAuth sign-in for MCP servers (mcp_auth.py): the token file, the full flow
against a fake Notion, refreshing across restarts and processes, and what a
server that isn't signed in reports."""
import asyncio
import json
import os
import time
from urllib.parse import parse_qs, urlparse

import pytest

pytest.importorskip("mcp")
import httpx2  # noqa: E402

from paulus import config, mcp_auth, mcp_client  # noqa: E402

URL = "https://mcp.example.com/mcp"
REDIRECT = mcp_auth.DEFAULT_REDIRECT


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    mcp_client.shutdown()
    yield tmp_path
    mcp_client.shutdown()


class FakeNotion:
    """An MCP endpoint behind an OAuth server that rotates refresh tokens and,
    like Notion, revokes the grant when a retired one comes back."""

    def __init__(self, expires_in=3600):
        self.expires_in = expires_in
        self.registrations = 0
        self.issued = 0
        self.valid_access = set()
        self.refresh = None          # the one live refresh token
        self.revoked = False
        self.codes = {}

    def _tokens(self):
        self.issued += 1
        access, self.refresh = f"at{self.issued}", f"rt{self.issued}"
        self.valid_access = {access}
        return httpx2.Response(200, json={"access_token": access, "token_type": "bearer",
                                          "expires_in": self.expires_in,
                                          "refresh_token": self.refresh})

    def __call__(self, request):
        path, base = request.url.path, "https://mcp.example.com"
        if path == "/.well-known/oauth-protected-resource/mcp":
            return httpx2.Response(200, json={"resource": URL, "authorization_servers": [base]})
        if path == "/.well-known/oauth-authorization-server":
            return httpx2.Response(200, json={
                "issuer": base, "authorization_endpoint": f"{base}/authorize",
                "token_endpoint": f"{base}/token", "registration_endpoint": f"{base}/register",
                "response_types_supported": ["code"],
                "code_challenge_methods_supported": ["S256"]})
        if path == "/register":
            self.registrations += 1
            body = json.loads(request.content)
            return httpx2.Response(201, json={**body, "client_id": "paulus-client"})
        if path == "/token":
            form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
            if form["grant_type"] == "authorization_code" and form["code"] in self.codes:
                return self._tokens()
            if form["grant_type"] == "refresh_token" and not self.revoked:
                if form["refresh_token"] == self.refresh:
                    return self._tokens()
                self.revoked = True          # a replayed refresh token kills the grant
            return httpx2.Response(400, json={"error": "invalid_grant"})
        if path == "/mcp":
            token = request.headers.get("authorization", "").removeprefix("Bearer ")
            if token in self.valid_access and not self.revoked:
                return httpx2.Response(200, json={"ok": True})
            return httpx2.Response(401)
        return httpx2.Response(404)

    def approve(self, link):
        """What the browser does: approve, and land on the redirect URI."""
        query = {k: v[0] for k, v in parse_qs(urlparse(link).query).items()}
        assert query["code_challenge_method"] == "S256" and query["redirect_uri"] == REDIRECT
        code = f"code{len(self.codes)}"
        self.codes[code] = query
        return f"{REDIRECT}?code={code}&state={query['state']}"


def _get(auth, notion):
    async def go():
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(notion), auth=auth) as c:
            return (await c.get(URL)).status_code
    return asyncio.run(go())


def _sign_in(notion):
    landed = []

    async def show_link(link):
        landed.append(notion.approve(link))

    async def read_answer():
        return mcp_auth.parse_redirect(landed.pop())

    mcp_auth.FileStorage("notion").forget_tokens(REDIRECT)
    auth = mcp_auth.provider("notion", {"url": URL}, show_link, read_answer)
    assert _get(auth, notion) == 200


# --- the flow ----------------------------------------------------------------

def test_sign_in_registers_once_and_saves_a_private_token_file(data_dir):
    notion = FakeNotion()
    _sign_in(notion)
    path = mcp_auth.token_path("notion")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["tokens"]["refresh_token"] == "rt1"
    assert data["client_info"]["client_id"] == "paulus-client"
    assert data["expires_at"] == pytest.approx(time.time() + 3600, abs=60)
    if os.name == "posix":
        assert path.stat().st_mode & 0o777 == 0o600

    _sign_in(notion)                        # signing in again reuses the registration
    assert notion.registrations == 1


def test_signed_in_server_works_without_a_person(data_dir):
    notion = FakeNotion()
    _sign_in(notion)
    assert _get(mcp_auth.provider("notion", {"url": URL}), notion) == 200


def test_restart_after_expiry_refreshes_instead_of_asking(data_dir):
    notion = FakeNotion(expires_in=1)
    _sign_in(notion)
    time.sleep(1.1)
    # A fresh provider is a restarted gateway: the expiry comes from the file.
    assert _get(mcp_auth.provider("notion", {"url": URL}), notion) == 200
    assert notion.issued == 2
    assert json.loads(mcp_auth.token_path("notion").read_text())["tokens"]["refresh_token"] == "rt2"


def test_two_processes_never_replay_a_retired_refresh_token(data_dir):
    notion = FakeNotion(expires_in=1)
    _sign_in(notion)
    gateway = mcp_auth.provider("notion", {"url": URL})
    terminal = mcp_auth.provider("notion", {"url": URL})
    assert _get(gateway, notion) == 200            # holds rt1 in memory
    time.sleep(1.1)
    assert _get(terminal, notion) == 200           # rotates rt1 -> rt2 on disk
    time.sleep(1.1)
    assert _get(gateway, notion) == 200            # refreshes with rt2, not its stale rt1
    assert not notion.revoked


def test_a_dead_grant_drops_the_tokens_and_asks_for_a_sign_in(data_dir):
    notion = FakeNotion(expires_in=1)
    _sign_in(notion)
    notion.revoked = True
    time.sleep(1.1)
    with pytest.raises(mcp_auth.SignInNeeded):
        _get(mcp_auth.provider("notion", {"url": URL}), notion)
    assert not mcp_auth.FileStorage("notion").signed_in()


def test_a_server_error_on_refresh_keeps_the_tokens(data_dir):
    notion = FakeNotion(expires_in=1)
    _sign_in(notion)
    time.sleep(1.1)
    down = lambda request: httpx2.Response(503)  # noqa: E731
    assert _get(mcp_auth.provider("notion", {"url": URL}), down) == 503
    assert mcp_auth.FileStorage("notion").signed_in()


# --- pieces ------------------------------------------------------------------

def test_parse_redirect():
    got = mcp_auth.parse_redirect(f" {REDIRECT}?code=abc&state=xyz&iss=https%3A%2F%2Fa \n")
    assert (got.code, got.state, got.iss) == ("abc", "xyz", "https://a")
    with pytest.raises(ValueError, match="refused: nope"):
        mcp_auth.parse_redirect(f"{REDIRECT}?error=access_denied&error_description=nope")
    with pytest.raises(ValueError, match="no \\?code="):
        mcp_auth.parse_redirect("https://example.com/")


def test_new_redirect_uri_means_a_new_registration(data_dir):
    _sign_in(FakeNotion())
    storage = mcp_auth.FileStorage("notion")
    storage.forget_tokens("http://localhost:9999/cb")
    assert not storage.signed_in()
    assert asyncio.run(storage.get_client_info()) is None


def test_unsigned_server_reports_how_to_sign_in(data_dir, tmp_path):
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"mcpServers": {"notion": {"url": URL, "auth": "oauth"}}}))
    assert mcp_client.init(str(path)) == 0
    assert "run paulus-mcp-login notion" in mcp_client.status()


def test_config_checks_auth(tmp_path):
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"mcpServers": {
        "typo": {"url": URL, "auth": "oath"},
        "stdio": {"command": "x", "auth": "oauth"},
    }}))
    confs = mcp_client.load_config(str(path))
    assert "can only be" in confs["typo"]["error"]
    assert "needs a 'url'" in confs["stdio"]["error"]


def test_login_command_explains_bad_arguments(tmp_path, monkeypatch, capsys):
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"mcpServers": {"notion": {"url": URL, "auth": "oauth"},
                                               "gh": {"command": "x"}}}))
    monkeypatch.setattr(config, "MCP_CONFIG", str(path))
    assert mcp_auth.main([]) == 2
    assert "notion" in capsys.readouterr().out
    assert mcp_auth.main(["nope"]) == 1
    assert mcp_auth.main(["gh"]) == 1
    assert "doesn't sign in with OAuth" in capsys.readouterr().out
