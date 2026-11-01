"""Web chat adapter: a browser UI served by the gateway itself (aiohttp).

Users sign in with the Telegram Login Widget, so a web user *is* their
Telegram user — same id, same memory, tasks and billing — and
TELEGRAM_ALLOWED_USERS / TELEGRAM_TRUSTED_USERS apply unchanged. Each browser
tab holds one WebSocket that carries the user's turns up and streamed replies,
approval prompts and notifications down.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import hmac
import json
import os
import re
import time
from collections import deque
from pathlib import Path
from urllib.parse import urlsplit

from aiohttp import ClientSession, ClientTimeout, WSMsgType, web

from ... import security
from ..base import AdapterState, BasePlatformAdapter, SessionSource, parse_ids, revealable

_STATIC_DIR = Path(__file__).with_name("webui_static")
_STATIC_TYPES = {".html": "text/html", ".js": "text/javascript", ".css": "text/css"}
_COOKIE = "paulus_session"
# A Telegram login payload older than this is refused (replayed or stale).
_LOGIN_MAX_AGE = 86400
# Undelivered messages kept per user for a tab that reconnects (in memory only).
_BACKLOG_LIMIT = 50
_HISTORY_LIMIT = 100
# Base64 inflates by a third; leave room for an image near the size cap.
_WS_MAX_MSG = 16 * 1024 * 1024
_IMAGE_MAX_BYTES = 10 * 1024 * 1024
_COMMANDS = ("sleep", "mood", "memory", "skills", "bg", "tasks", "cancel")
# Event types that are worth keeping for a tab that wasn't connected. Deltas and
# typing are transient; a committed segment or final carries the whole text.
_DURABLE = frozenset({"message", "segment", "final", "document"})
# Conversation text that /api/history also returns, so a fresh page load drops
# it from the backlog instead of showing it twice.
_IN_HISTORY = frozenset({"message", "segment", "final"})
_VIA = re.compile(r"^\[via (\w+)\] ")
_DOC_NOTE = re.compile(r"\[Owner sent a document: (.+?) \(\d+ chars\)")


class WebAdapter(BasePlatformAdapter):
    """
    Browser chat served on WEB_HOST:WEB_PORT, meant to sit behind a TLS
    reverse proxy. Supports streaming, approvals (with task grants), image and
    text-document uploads, and outbound documents.
    """
    supports_typing_indicator = True
    supports_approvals = True
    supports_grants = True
    supports_images = True
    supports_documents = True
    supports_streaming = True

    def __init__(self, runner) -> None:
        super().__init__(runner)
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        if not token:
            raise ValueError("The web UI signs users in with Telegram; set TELEGRAM_BOT_TOKEN.")
        self._token = token
        self._bot_username = os.environ.get("TELEGRAM_BOT_USERNAME", "").strip().lstrip("@")
        self._host = os.environ.get("WEB_HOST", "127.0.0.1")
        self._port = int(os.environ.get("WEB_PORT", "8080"))
        # The public address (e.g. https://paulus.example.com, or .../chat when
        # a proxy mounts the UI under a path). Used to check the Origin of
        # logins and sockets; unset -> the request's own Host.
        self._public_url = os.environ.get("WEB_PUBLIC_URL", "").strip().rstrip("/")
        # Scope the session cookie to that path, so a UI sharing its domain
        # with another site never sends that site our session.
        self._cookie_path = urlsplit(self._public_url).path.rstrip("/") + "/"
        self._session_ttl = float(os.environ.get("WEB_SESSION_DAYS", "30")) * 86400
        self._doc_max_bytes = int(os.environ.get("WEB_DOC_MAX_BYTES", "262144"))
        # Same allow/trust semantics as the Telegram adapter (see there).
        self._allowed: set[str] = parse_ids(os.environ.get("TELEGRAM_ALLOWED_USERS", ""))
        trusted_env = os.environ.get("TELEGRAM_TRUSTED_USERS")
        self._trusted: set[str] = (
            parse_ids(trusted_env) if trusted_env is not None else set(self._allowed)
        )
        # Session cookies are signed with this. Derived from the bot token by
        # default, so rotating the token also signs everyone out.
        secret = os.environ.get("WEB_SESSION_SECRET", "")
        self._secret = (secret.encode() if secret else
                        hmac.new(token.encode(), b"paulus-web-session", hashlib.sha256).digest())
        self._site_runner: web.AppRunner | None = None
        # Live sockets per user id; a user may have several tabs open.
        self._sockets: dict[str, set[_Connection]] = {}
        self._backlog: dict[str, deque] = {}
        # Unsettled approval prompts: id -> (owner user id, event). Replayed to
        # every socket the owner opens, so a reload doesn't strand a prompt.
        self._open_approvals: dict[str, tuple[str, dict]] = {}
        # Running turns, held so they aren't garbage-collected mid-flight.
        self._turns: set[asyncio.Task] = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if not self._bot_username:
            self._bot_username = await self._fetch_bot_username()
        self._site_runner = web.AppRunner(self.make_app(), access_log=None)
        await self._site_runner.setup()
        await web.TCPSite(self._site_runner, self._host, self._port).start()
        self._state = AdapterState.RUNNING

    def make_app(self) -> web.Application:
        app = web.Application()
        app.add_routes([
            web.get("/", self._on_index),
            web.get("/app.js", self._on_static),
            web.get("/app.css", self._on_static),
            web.get("/auth", self._on_auth),
            web.get("/api/me", self._on_me),
            web.post("/api/logout", self._on_logout),
            web.get("/api/history", self._on_history),
            web.get("/ws", self._on_ws),
        ])
        return app

    async def stop(self) -> None:
        self._state = AdapterState.STOPPED
        for conns in list(self._sockets.values()):
            for conn in list(conns):
                await conn.ws.close()
        if self._site_runner is not None:
            await self._site_runner.cleanup()
            self._site_runner = None

    async def _fetch_bot_username(self) -> str:
        """The login widget needs the bot's @username; ask Telegram for it."""
        url = f"https://api.telegram.org/bot{self._token}/getMe"
        try:
            async with ClientSession(timeout=ClientTimeout(total=10)) as http:
                async with http.get(url) as resp:
                    data = await resp.json()
            return str(data["result"]["username"])
        except Exception as exc:
            # Never log the URL: it carries the bot token.
            security.audit("web_bot_username_error", type(exc).__name__)
            return ""

    # ------------------------------------------------------------------
    # HTTP: page, login, session
    # ------------------------------------------------------------------

    async def _on_index(self, request: web.Request) -> web.Response:
        return self._static_response("index.html")

    async def _on_static(self, request: web.Request) -> web.Response:
        return self._static_response(request.path.lstrip("/"))

    def _static_response(self, name: str) -> web.Response:
        path = _STATIC_DIR / name
        resp = web.Response(body=path.read_bytes(), content_type=_STATIC_TYPES[path.suffix],
                            charset="utf-8")
        resp.headers["Cache-Control"] = "no-cache"
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Referrer-Policy"] = "same-origin"
        # The model's Markdown is sanitised client-side; the CSP is the second
        # line. frame-ancestors keeps the Approve buttons from being clickjacked.
        resp.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' https://telegram.org https://cdn.jsdelivr.net; "
            "frame-src https://oauth.telegram.org; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob: https:; "
            "connect-src 'self' ws: wss:; "
            "frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        )
        return resp

    async def _on_auth(self, request: web.Request) -> web.Response:
        """The login widget redirects here with the signed Telegram user."""
        fields = dict(request.query)
        user_id = self._verify_login(fields)
        if user_id is None:
            security.audit("web_login_rejected", fields.get("id", "?"))
            raise web.HTTPFound("./?login=invalid")
        if self._allowed and user_id not in self._allowed:
            security.audit("web_login_unauthorized", user_id)
            raise web.HTTPFound("./?login=unauthorized")
        name = fields.get("first_name") or fields.get("username") or user_id
        resp = web.HTTPFound("./")
        resp.set_cookie(_COOKIE, self._issue_session(user_id, name),
                        max_age=int(self._session_ttl), path=self._cookie_path, httponly=True,
                        samesite="Lax", secure=self._is_https(request))
        security.audit("web_login", user_id)
        raise resp

    async def _on_me(self, request: web.Request) -> web.Response:
        user = self._session_user(request)
        if user is None:
            return web.json_response({"bot": self._bot_username}, status=401)
        return web.json_response({
            "id": user["uid"], "name": user["name"],
            "can_approve": self.can_approve(user["uid"]),
        })

    async def _on_logout(self, request: web.Request) -> web.Response:
        if not self._same_origin(request):
            raise web.HTTPForbidden()
        resp = web.json_response({"ok": True})
        resp.del_cookie(_COOKIE, path=self._cookie_path)
        return resp

    async def _on_history(self, request: web.Request) -> web.Response:
        """The user's recent conversation (shared with Telegram), oldest first."""
        user = self._session_user(request)
        if user is None:
            raise web.HTTPUnauthorized()
        from ... import memory
        uid = user["uid"]
        # The page renders history in place of anything already logged, so a
        # backlog of the same messages would show them twice.
        backlog = self._backlog.get(uid)
        if backlog:
            self._backlog[uid] = deque((e for e in backlog if e["type"] not in _IN_HISTORY),
                                       maxlen=_BACKLOG_LIMIT)
        episodes = memory.recent_episodes(_HISTORY_LIMIT, user_id=uid)
        return web.json_response([_history_item(e) for e in episodes
                                  if e.get("role") in ("owner", "agent")])

    def _verify_login(self, fields: dict) -> str | None:
        """Check a Telegram Login Widget payload; return the user id if genuine.
        See https://core.telegram.org/widgets/login#checking-authorization."""
        received = fields.get("hash", "")
        check = "\n".join(f"{k}={fields[k]}" for k in sorted(fields) if k != "hash")
        secret = hashlib.sha256(self._token.encode()).digest()
        expected = hmac.new(secret, check.encode(), hashlib.sha256).hexdigest()
        if not received or not hmac.compare_digest(expected, received):
            return None
        try:
            auth_date = int(fields.get("auth_date", ""))
        except ValueError:
            return None
        if time.time() - auth_date > _LOGIN_MAX_AGE:
            return None
        return fields.get("id") or None

    def _issue_session(self, user_id: str, name: str) -> str:
        claims = {"uid": user_id, "name": name, "exp": int(time.time() + self._session_ttl)}
        payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
        return f"{payload}.{self._sign(payload)}"

    def _session_user(self, request: web.Request) -> dict | None:
        """The signed-in user's claims, or None. The allowlist is re-checked on
        every request, so removing someone from it locks them out at once."""
        payload, _, sig = request.cookies.get(_COOKIE, "").rpartition(".")
        if not payload or not hmac.compare_digest(self._sign(payload), sig):
            return None
        try:
            claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        except (ValueError, binascii.Error):
            return None
        if claims.get("exp", 0) < time.time():
            return None
        if self._allowed and claims.get("uid") not in self._allowed:
            return None
        return claims

    def _sign(self, payload: str) -> str:
        return hmac.new(self._secret, payload.encode(), hashlib.sha256).hexdigest()

    def _same_origin(self, request: web.Request) -> bool:
        """Refuse cross-site requests riding on the session cookie. Browsers
        always send Origin on WebSocket handshakes and POSTs."""
        origin = request.headers.get("Origin")
        if origin is None:
            return True
        if self._public_url:
            want = urlsplit(self._public_url)
            return origin == f"{want.scheme}://{want.netloc}"
        return urlsplit(origin).netloc == request.host

    def _is_https(self, request: web.Request) -> bool:
        return (request.secure or request.headers.get("X-Forwarded-Proto") == "https"
                or self._public_url.startswith("https://"))

    # ------------------------------------------------------------------
    # WebSocket
    # ------------------------------------------------------------------

    async def _on_ws(self, request: web.Request) -> web.WebSocketResponse:
        user = self._session_user(request)
        if user is None or not self._same_origin(request):
            raise web.HTTPForbidden()
        ws = web.WebSocketResponse(heartbeat=30, max_msg_size=_WS_MAX_MSG)
        await ws.prepare(request)
        uid = user["uid"]
        conn = _Connection(ws)
        pump = asyncio.create_task(conn.pump())
        self._sockets.setdefault(uid, set()).add(conn)
        for event in self._backlog.pop(uid, ()):
            conn.queue.put_nowait(event)
        for owner, event in self._open_approvals.values():
            if owner == uid:
                conn.queue.put_nowait(event)
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    event = json.loads(msg.data)
                except ValueError:
                    continue
                if isinstance(event, dict):
                    self._on_event(uid, event)
        finally:
            conns = self._sockets.get(uid, set())
            conns.discard(conn)
            if not conns:
                self._sockets.pop(uid, None)
            pump.cancel()
        return ws

    def _on_event(self, uid: str, event: dict) -> None:
        kind = event.get("type")
        if kind == "message":
            # A turn runs as its own task: it can block for minutes on an
            # approval, which has to arrive over this same socket meanwhile.
            task = asyncio.create_task(self._on_message(uid, event))
            self._turns.add(task)
            task.add_done_callback(self._turns.discard)
        elif kind == "approve":
            self._on_approve(uid, event)

    async def _on_message(self, uid: str, event: dict) -> None:
        source = SessionSource(platform="web", chat_id=uid, user_id=uid)
        try:
            await self._handle_message(source, event)
        except Exception as exc:
            security.audit("web_turn_error", f"{uid}: {exc}")
            self._emit(uid, {"type": "message",
                             "text": "Sorry — something went wrong handling that."})
        finally:
            self._emit(uid, {"type": "idle"})

    async def _handle_message(self, source: SessionSource, event: dict) -> None:
        uid = source.user_id
        text = str(event.get("text") or "").strip()
        attachments = event.get("attachments") or []
        if not isinstance(attachments, list):
            attachments = []

        if text.startswith("/") and not attachments:
            await self._handle_command(source, text)
            return

        images, documents = [], []
        for att in attachments:
            if not isinstance(att, dict):
                continue
            item, error = self._decode_attachment(att)
            if error:
                self._emit(uid, {"type": "message", "text": error})
                return
            (images if "media_type" in item else documents).append(item)
        if not text and not images and not documents:
            return

        if images:
            from ... import config, llm
            if not llm.supports_vision(config.chat_model()):
                self._emit(uid, {"type": "message", "text": (
                    "I can't analyse images — the configured model isn't "
                    "vision-capable. Set DP_CORE_MODEL (or DP_ROUTER_MODEL) to a "
                    "model that supports images.")})
                return
            text = text or "(image, no caption)"

        names = [str(a.get("name") or "file") for a in attachments if isinstance(a, dict)]
        self._emit(uid, {"type": "user", "text": text, "attachments": names})
        await self.send_typing(source)
        await self._runner.handle_inbound(source, text, images=images or None,
                                          documents=documents or None)

    async def _handle_command(self, source: SessionSource, text: str) -> None:
        """The same in-chat commands as Telegram (/sleep, /mood, ..., /reset)."""
        uid = source.user_id
        parts = text[1:].split(maxsplit=1)
        cmd = parts[0].lower() if parts else ""
        args = parts[1].strip() if len(parts) > 1 else ""
        self._emit(uid, {"type": "user", "text": text, "attachments": []})
        if cmd == "reset":
            self._runner._sessions.reset(source.key())
            security.audit("gateway_session_reset", source.key())
            reply = "Session reset."
        elif cmd in _COMMANDS:
            await self.send_typing(source)
            reply = await self._runner.handle_command(source, cmd, args)
        else:
            reply = (f"Unknown command: /{cmd}. Try /" + ", /".join(_COMMANDS + ("reset",)))
        await self.send(source, reply)

    def _decode_attachment(self, att: dict) -> tuple[dict, str | None]:
        """An uploaded file as an image or text document for the agent, or an
        error message for the user."""
        name = os.path.basename(str(att.get("name") or "document.txt")) or "document.txt"
        media_type = str(att.get("type") or "")
        try:
            raw = base64.b64decode(str(att.get("data") or ""), validate=True)
        except (ValueError, binascii.Error):
            return {}, f"Sorry, I couldn't read {name}."
        if media_type.startswith("image/"):
            if len(raw) > _IMAGE_MAX_BYTES:
                return {}, f"{name} is too large; images can be up to {_IMAGE_MAX_BYTES} bytes."
            return {"media_type": media_type, "data": base64.b64encode(raw).decode("ascii")}, None
        if len(raw) > self._doc_max_bytes:
            return {}, (f"That document is too large ({len(raw)} bytes). "
                        f"I can read text files up to {self._doc_max_bytes} bytes.")
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            return {}, "I can only read text documents."
        return {"filename": name, "content": content}, None

    # ------------------------------------------------------------------
    # Approvals (high-impact action gate)
    # ------------------------------------------------------------------

    def can_approve(self, user_id) -> bool:
        return str(user_id) in self._trusted

    async def request_approval(self, source: SessionSource, approval_id: str,
                               prompt: str, grantable: bool = False) -> None:
        event = {"type": "approval", "id": approval_id, "prompt": prompt,
                 "grantable": grantable}
        self._open_approvals[approval_id] = (source.chat_id, event)
        self._emit(source.chat_id, event)

    async def expire_approval(self, source: SessionSource, approval_id: str) -> None:
        if self._open_approvals.pop(approval_id, None) is not None:
            self._emit(source.chat_id, {
                "type": "approval_settled", "id": approval_id,
                "verdict": "⏲️ Approval request timed out — action denied."})

    def _on_approve(self, uid: str, event: dict) -> None:
        approval_id = str(event.get("id", ""))
        answer = event.get("answer")
        if answer not in ("ok", "no", "all"):
            return
        owner, _ = self._open_approvals.get(approval_id, (None, None))
        # Approval ids are guessable; only the user who was asked may answer.
        if owner != uid or not self.can_approve(uid):
            security.audit("approval_unauthorized", f"web {uid} {approval_id}")
            return
        approved = {"ok": True, "no": False, "all": security.GRANT}[answer]
        settled = self._runner.resolve_approval(approval_id, approved)
        self._open_approvals.pop(approval_id, None)
        if not settled:
            verdict = "⏲️ This request already expired."
        elif approved == security.GRANT:
            verdict = "✅ Approved, and allowed for the rest of this task."
        else:
            verdict = "✅ Approved." if approved else "🚫 Denied."
        self._emit(uid, {"type": "approval_settled", "id": approval_id, "verdict": verdict})

    # ------------------------------------------------------------------
    # Outbound
    # ------------------------------------------------------------------

    def _emit(self, uid: str, event: dict) -> None:
        """Queue ``event`` for every open tab of ``uid``, or keep it for the
        next one if a durable event finds none. Runs on the gateway loop."""
        conns = self._sockets.get(uid)
        if conns:
            for conn in conns:
                conn.queue.put_nowait(event)
        elif event["type"] in _DURABLE:
            self._backlog.setdefault(uid, deque(maxlen=_BACKLOG_LIMIT)).append(event)

    async def send(self, source: SessionSource, text: str) -> None:
        if text.strip():
            self._emit(source.chat_id, {"type": "message", "text": text})

    async def send_with_link(self, source: SessionSource, text: str, label: str,
                             url: str) -> None:
        self._emit(source.chat_id, {"type": "message", "text": text.strip(),
                                    "link": {"label": label, "url": url}})

    async def send_document(self, source: SessionSource, filename: str,
                            content: str) -> None:
        self._emit(source.chat_id, {"type": "document", "filename": filename,
                                    "content": content})

    async def send_typing(self, source: SessionSource) -> None:
        self._emit(source.chat_id, {"type": "typing"})

    def stream_sink(self, source: SessionSource, loop) -> _StreamSink:
        return _StreamSink(self, source, loop)


class _Connection:
    """One browser tab. Events go through a queue drained by a single writer,
    so they reach the socket in the order they were emitted."""

    def __init__(self, ws: web.WebSocketResponse) -> None:
        self.ws = ws
        self.queue: asyncio.Queue = asyncio.Queue()

    async def pump(self) -> None:
        while True:
            event = await self.queue.get()
            try:
                await self.ws.send_json(event)
            except Exception:
                return                    # socket gone; the reader cleans up


class _StreamSink:
    """Streams one reply to the user's tabs as ``delta`` events. ``feed`` and
    ``end_segment`` are called from the agent's worker thread and hop onto the
    gateway loop; each step ends in a ``segment`` (or ``discard``) carrying its
    full text, and the reply in a ``final``. Text is held back until it is
    clearly not a silence sentinel."""

    def __init__(self, adapter: WebAdapter, source: SessionSource, loop) -> None:
        self._adapter = adapter
        self._uid = source.chat_id
        self._loop = loop
        self._buf = ""
        self._shown = 0               # chars of _buf already sent as deltas

    def _post(self, event: dict) -> None:
        self._loop.call_soon_threadsafe(self._adapter._emit, self._uid, event)

    # -- called from the worker thread -------------------------------------
    def feed(self, delta: str) -> None:
        if not delta:
            return
        self._buf += delta
        if not revealable(self._buf):
            return
        self._post({"type": "delta", "text": self._buf[self._shown:]})
        self._shown = len(self._buf)

    def end_segment(self) -> None:
        text, self._buf, self._shown = self._buf, "", 0
        self._post({"type": "segment", "text": text} if revealable(text)
                   else {"type": "discard"})

    # -- awaited on the gateway loop ---------------------------------------
    async def finalize(self, reply: str) -> None:
        text = self._buf if self._buf.strip() else reply
        self._adapter._emit(self._uid, {"type": "final", "text": text} if text.strip()
                            else {"type": "discard"})

    async def discard(self) -> None:
        self._adapter._emit(self._uid, {"type": "discard"})


def _history_item(episode: dict) -> dict:
    """An episodic-log entry as the page shows it: the platform tag lifted
    out, and an ingested document reduced to its name."""
    text = episode.get("text", "")
    item = {"role": episode["role"], "ts": episode.get("ts", "")}
    if episode["role"] == "owner":
        m = _VIA.match(text)
        if m:
            item["via"] = m.group(1)
            text = text[m.end():]
        names = _DOC_NOTE.findall(text)
        if names:
            text = text.split("[Owner sent a document:", 1)[0].rstrip()
            item["attachments"] = names
    item["text"] = text
    return item
