"""The web UI adapter: Telegram login, sessions, and the chat WebSocket."""
import asyncio
import base64
import hashlib
import hmac
import time

import pytest

pytest.importorskip("aiohttp")

from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

from paulus.gateway.base import AdapterState, SessionSource  # noqa: E402
from paulus.gateway.platforms import webui  # noqa: E402
from paulus.gateway.runner import GatewayRunner  # noqa: E402

TOKEN = "123456:TEST-TOKEN"


@pytest.fixture(autouse=True)
def web_env(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    monkeypatch.setenv("TELEGRAM_BOT_USERNAME", "PaulusTestBot")
    # Chat is open to all; only user 42 may approve.
    monkeypatch.setenv("TELEGRAM_TRUSTED_USERS", "42")
    for name in ("TELEGRAM_ALLOWED_USERS", "WEB_PUBLIC_URL"):
        monkeypatch.delenv(name, raising=False)


def _signed(fields: dict, token: str = TOKEN) -> dict:
    """A login payload signed the way Telegram's widget signs it."""
    fields = {k: str(v) for k, v in fields.items()}
    check = "\n".join(f"{k}={fields[k]}" for k in sorted(fields))
    secret = hashlib.sha256(token.encode()).digest()
    fields["hash"] = hmac.new(secret, check.encode(), hashlib.sha256).hexdigest()
    return fields


def _login(uid="42", **extra) -> dict:
    return _signed({"id": uid, "first_name": "Ada", "auth_date": int(time.time()), **extra})


def _make():
    runner = GatewayRunner()
    adapter = webui.WebAdapter(runner)
    adapter._state = AdapterState.RUNNING
    runner.register("web", adapter)
    return runner, adapter


async def _client(adapter) -> TestClient:
    client = TestClient(TestServer(adapter.make_app()))
    await client.start_server()
    return client


async def _signed_in(adapter, uid="42") -> TestClient:
    client = await _client(adapter)
    resp = await client.get("/auth", params=_login(uid), allow_redirects=False)
    assert resp.status == 302
    return client


async def _recv_until(ws, kind, timeout=5):
    """Collect events until one of type *kind* arrives; return them all."""
    events = []
    while True:
        ev = await asyncio.wait_for(ws.receive_json(), timeout)
        events.append(ev)
        if ev["type"] == kind:
            return events


# --- Login & session --------------------------------------------------------

def test_requires_bot_token(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN")
    with pytest.raises(ValueError):
        webui.WebAdapter(GatewayRunner())


def test_verify_login_accepts_genuine_and_rejects_forged_or_stale():
    _, adapter = _make()
    assert adapter._verify_login(_login("42")) == "42"

    tampered = _login("42")
    tampered["id"] = "43"
    assert adapter._verify_login(tampered) is None

    assert adapter._verify_login(_signed({"id": "42", "auth_date": int(time.time())},
                                         token="999:OTHER")) is None
    stale = _signed({"id": "42", "auth_date": int(time.time()) - 2 * 86400})
    assert adapter._verify_login(stale) is None
    assert adapter._verify_login({"id": "42", "auth_date": str(int(time.time()))}) is None


def test_login_sets_session_and_me_reports_user():
    _, adapter = _make()

    async def go():
        client = await _client(adapter)
        try:
            resp = await client.get("/api/me")
            assert resp.status == 401
            assert (await resp.json())["bot"] == "PaulusTestBot"

            resp = await client.get("/auth", params=_login("42"), allow_redirects=False)
            assert resp.status == 302 and resp.headers["Location"] == "./"
            cookie = resp.cookies[webui._COOKIE]
            assert cookie["httponly"]

            me = await (await client.get("/api/me")).json()
            assert me == {"id": "42", "name": "Ada", "can_approve": True}

            await client.post("/api/logout")
            assert (await client.get("/api/me")).status == 401
        finally:
            await client.close()

    asyncio.run(go())


def test_login_enforces_allowlist_now_and_later(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "42")
    _, adapter = _make()

    async def go():
        client = await _client(adapter)
        try:
            resp = await client.get("/auth", params=_login("7"), allow_redirects=False)
            assert resp.headers["Location"] == "./?login=unauthorized"
            assert (await client.get("/api/me")).status == 401

            resp = await client.get("/auth", params=_login("42"), allow_redirects=False)
            assert (await client.get("/api/me")).status == 200
            # Dropped from the allowlist -> the existing session stops working.
            adapter._allowed = {"99"}
            assert (await client.get("/api/me")).status == 401
        finally:
            await client.close()

    asyncio.run(go())


def test_forged_session_cookie_is_rejected():
    _, adapter = _make()
    token = adapter._issue_session("42", "Ada")
    payload, _, sig = token.rpartition(".")
    claims = base64.urlsafe_b64encode(b'{"uid": "1", "name": "x", "exp": 9999999999}')
    forged = claims.decode().rstrip("=") + "." + sig

    class Req:
        def __init__(self, value):
            self.cookies = {webui._COOKIE: value}

    assert adapter._session_user(Req(token))["uid"] == "42"
    assert adapter._session_user(Req(forged)) is None
    assert adapter._session_user(Req("garbage")) is None


def test_trusted_users_default_to_allowlist(monkeypatch):
    monkeypatch.delenv("TELEGRAM_TRUSTED_USERS")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "1,2")
    _, adapter = _make()
    assert adapter.can_approve("1") and not adapter.can_approve("3")
    monkeypatch.setenv("TELEGRAM_TRUSTED_USERS", "")
    _, adapter = _make()
    assert not adapter.can_approve("1")


def test_page_is_served_with_frame_protection():
    _, adapter = _make()

    async def go():
        client = await _client(adapter)
        try:
            resp = await client.get("/")
            assert resp.status == 200 and "PaulusAI" in await resp.text()
            assert "frame-ancestors 'none'" in resp.headers["Content-Security-Policy"]
            assert (await client.get("/app.js")).headers["Content-Type"].startswith(
                "text/javascript")
        finally:
            await client.close()

    asyncio.run(go())


# --- WebSocket ----------------------------------------------------------------

def test_socket_refuses_anonymous_and_cross_site():
    _, adapter = _make()

    async def go():
        client = await _client(adapter)
        try:
            assert (await client.get("/ws")).status == 403
            await client.get("/auth", params=_login("42"), allow_redirects=False)
            resp = await client.get("/ws", headers={"Origin": "https://evil.example"})
            assert resp.status == 403
        finally:
            await client.close()

    asyncio.run(go())


def test_turn_streams_reply_over_socket(monkeypatch):
    import paulus.agent as agent
    runner, adapter = _make()
    seen = {}

    def fake_respond(text, user_id=None, on_delta=None, images=None, documents=None,
                     on_break=None):
        seen.update(text=text, user_id=user_id)
        for part in ("Hel", "lo ", "**there**"):
            on_delta(part)
        return "Hello **there**"

    monkeypatch.setattr(agent, "respond", fake_respond)

    async def go():
        client = await _signed_in(adapter)
        try:
            ws = await client.ws_connect("/ws")
            await ws.send_json({"type": "message", "text": "hi"})
            events = await _recv_until(ws, "idle")
            await ws.close()
            return events
        finally:
            await client.close()

    events = asyncio.run(go())
    kinds = [e["type"] for e in events]
    assert kinds[:2] == ["user", "typing"]
    assert "".join(e["text"] for e in events if e["type"] == "delta") == "Hello **there**"
    assert {"type": "final", "text": "Hello **there**"} in events
    assert kinds[-1] == "idle"
    assert seen == {"text": "[via web] hi", "user_id": "42"}


def test_silent_reply_is_never_shown(monkeypatch):
    import paulus.agent as agent
    _, adapter = _make()

    def fake_respond(text, user_id=None, on_delta=None, **kw):
        on_delta("[SIL")
        on_delta("ENT]")
        return "[SILENT]"

    monkeypatch.setattr(agent, "respond", fake_respond)

    async def go():
        client = await _signed_in(adapter)
        try:
            ws = await client.ws_connect("/ws")
            await ws.send_json({"type": "message", "text": "hi"})
            return await _recv_until(ws, "idle")
        finally:
            await client.close()

    kinds = [e["type"] for e in asyncio.run(go())]
    assert "delta" not in kinds and "final" not in kinds
    assert "discard" in kinds


def test_commands_route_to_runner(monkeypatch):
    import paulus.affect as affect
    _, adapter = _make()
    monkeypatch.setattr(affect, "describe", lambda: "calm")

    async def go():
        client = await _signed_in(adapter)
        try:
            ws = await client.ws_connect("/ws")
            await ws.send_json({"type": "message", "text": "/mood"})
            events = await _recv_until(ws, "idle")
            await ws.send_json({"type": "message", "text": "/nope"})
            events += await _recv_until(ws, "idle")
            return events
        finally:
            await client.close()

    replies = [e["text"] for e in asyncio.run(go()) if e["type"] == "message"]
    assert replies[0] == "mood: calm"
    assert replies[1].startswith("Unknown command: /nope")


def test_text_document_upload_reaches_agent(monkeypatch):
    import paulus.agent as agent
    _, adapter = _make()
    seen = {}

    def fake_respond(text, user_id=None, on_delta=None, images=None, documents=None,
                     on_break=None):
        seen["documents"] = documents
        return "got it"

    monkeypatch.setattr(agent, "respond", fake_respond)
    data = base64.b64encode(b"a,b\n1,2\n").decode()
    binary = base64.b64encode(b"\xff\xfe\x00").decode()

    async def go():
        client = await _signed_in(adapter)
        try:
            ws = await client.ws_connect("/ws")
            await ws.send_json({"type": "message", "text": "", "attachments": [
                {"name": "../t.csv", "type": "text/csv", "data": data}]})
            await _recv_until(ws, "idle")
            await ws.send_json({"type": "message", "text": "", "attachments": [
                {"name": "x.bin", "type": "application/octet-stream", "data": binary}]})
            return await _recv_until(ws, "idle")
        finally:
            await client.close()

    events = asyncio.run(go())
    assert seen["documents"] == [{"filename": "t.csv", "content": "a,b\n1,2\n"}]
    assert {"type": "message", "text": "I can only read text documents."} in events


def test_approval_roundtrip_and_only_owner_may_answer():
    runner, adapter = _make()
    runner._presence.touch(SessionSource("web", "42", "42"))

    async def go():
        runner._loop = asyncio.get_running_loop()
        owner = await _signed_in(adapter, "42")
        other = await _signed_in(adapter, "7")
        try:
            ws = await owner.ws_connect("/ws")
            ws_other = await other.ws_connect("/ws")
            asked = runner._loop.run_in_executor(
                None, lambda: runner.request_approval("42", "run_command", {"command": "ls"}))
            prompt = (await _recv_until(ws, "approval"))[-1]
            assert "run_command" in prompt["prompt"]

            # Someone else answering the (guessable) id is ignored.
            await ws_other.send_json({"type": "approve", "id": prompt["id"], "answer": "ok"})
            await asyncio.sleep(0.1)
            assert not asked.done()

            await ws.send_json({"type": "approve", "id": prompt["id"], "answer": "no"})
            settled = (await _recv_until(ws, "approval_settled"))[-1]
            assert settled["verdict"] == "🚫 Denied."
            return await asked
        finally:
            await owner.close()
            await other.close()

    assert asyncio.run(go()) is False


def test_pending_approval_is_replayed_to_a_new_tab():
    runner, adapter = _make()

    async def go():
        client = await _signed_in(adapter, "42")
        try:
            await adapter.request_approval(SessionSource("web", "42", "42"), "a-1", "ok?")
            ws = await client.ws_connect("/ws")
            return await asyncio.wait_for(ws.receive_json(), 5)
        finally:
            await client.close()

    ev = asyncio.run(go())
    assert ev["type"] == "approval" and ev["id"] == "a-1"


def test_offline_messages_wait_and_history_supersedes_them():
    import paulus.memory as memory
    runner, adapter = _make()
    src = SessionSource("web", "42", "42")
    memory.log_episode("owner", "[via telegram] hello from the phone", user_id="42")
    memory.log_episode("agent", "hi!", user_id="42")

    async def go():
        client = await _signed_in(adapter, "42")
        try:
            await adapter.send(src, "a nudge")
            await adapter.send_document(src, "notes.txt", "body")
            await adapter.send_typing(src)              # transient: not kept
            # A reconnecting tab (no history fetch) gets everything it missed.
            ws = await client.ws_connect("/ws")
            first = [await asyncio.wait_for(ws.receive_json(), 5) for _ in range(2)]
            await ws.close()
            await asyncio.sleep(0.05)

            # A fresh page load reads history, which already holds the text.
            await adapter.send(src, "another nudge")
            await adapter.send_document(src, "more.txt", "body")
            history = await (await client.get("/api/history")).json()
            ws = await client.ws_connect("/ws")
            second = await asyncio.wait_for(ws.receive_json(), 5)
            return first, history, second
        finally:
            await client.close()

    first, history, second = asyncio.run(go())
    assert [e["type"] for e in first] == ["message", "document"]
    assert second == {"type": "document", "filename": "more.txt", "content": "body"}
    assert history[0] == {"role": "owner", "ts": history[0]["ts"],
                          "via": "telegram", "text": "hello from the phone"}
    assert history[1]["text"] == "hi!"


def test_history_item_reduces_documents_to_names():
    item = webui._history_item({
        "role": "owner", "ts": "t",
        "text": "[via web] see this\n\n[Owner sent a document: a.md (12 chars), "
                "saved to inbox/a.md]\n\n<untrusted_data>...</untrusted_data>",
    })
    assert item == {"role": "owner", "ts": "t", "via": "web", "text": "see this",
                    "attachments": ["a.md"]}


def test_ui_under_a_path_keeps_its_cookie_there(monkeypatch):
    """Mounted at /chat/ on a domain shared with another site (proxy strips the
    prefix): the session cookie must not reach the rest of the domain."""
    monkeypatch.setenv("WEB_PUBLIC_URL", "https://paulus.example.com/chat/")
    _, adapter = _make()

    async def go():
        client = await _client(adapter)
        try:
            resp = await client.get("/auth", params=_login("42"), allow_redirects=False)
            assert resp.headers["Location"] == "./"          # -> /chat/ in the browser
            assert resp.cookies[webui._COOKIE]["path"] == "/chat/"
            assert resp.cookies[webui._COOKIE]["secure"]
            resp = await client.post("/api/logout",
                                     headers={"Origin": "https://paulus.example.com"})
            assert resp.status == 200
            assert "Path=/chat/" in resp.headers["Set-Cookie"]
            resp = await client.post("/api/logout", headers={"Origin": "https://evil.example"})
            assert resp.status == 403
        finally:
            await client.close()

    asyncio.run(go())
    monkeypatch.delenv("WEB_PUBLIC_URL")
    assert webui.WebAdapter(GatewayRunner())._cookie_path == "/"
