"""Telegram platform adapter using python-telegram-bot."""
from __future__ import annotations

import asyncio
import base64
import os
import time

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, InputFile, Update
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

try:
    # Optional: converts the model's CommonMark output into Telegram's strict
    # MarkdownV2 (with the escaping it requires). Absent -> plain-text delivery.
    import telegramify_markdown
except ImportError:  # pragma: no cover - exercised only without the extra
    telegramify_markdown = None

from ... import security
from ..base import SILENCE_TOKENS, AdapterState, BasePlatformAdapter, SessionSource

_TELEGRAM_MSG_LIMIT = 4096
# Split raw text below the hard limit: MarkdownV2 escaping adds backslashes, so
# a converted chunk can grow past its pre-escaped length.
_SPLIT_LIMIT = 3500
# Minimum seconds between intermediate edits while streaming, to stay under
# Telegram's per-message edit-rate limit.
_STREAM_EDIT_INTERVAL = 1.2


class TelegramAdapter(BasePlatformAdapter):
    """
    Telegram integration via python-telegram-bot.
    Supports DMs, group chats, and forum topics.
    Rapid messages from the same chat are batched with a 300 ms debounce.
    """
    supports_typing_indicator = True
    supports_approvals = True   # high-impact actions can be approved via buttons
    supports_images = True      # photos / image documents are routed to the agent
    supports_documents = True   # text documents are read in and can be sent back

    def __init__(self, runner) -> None:
        super().__init__(runner)
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        if not token:
            raise ValueError("TELEGRAM_BOT_TOKEN environment variable not set.")
        self._token = token
        self._allowed: set[str] = _parse_ids(os.environ.get("TELEGRAM_ALLOWED_USERS", ""))
        # Who may approve high-impact actions. Distinct from _allowed so chat can
        # be open to everyone (empty allowlist) while approvals stay restricted.
        # Unset -> defaults to the chat allowlist (the common single-owner case);
        # set but empty -> nobody can approve (high-impact falls back to policy).
        trusted_env = os.environ.get("TELEGRAM_TRUSTED_USERS")
        self._trusted: set[str] = (
            _parse_ids(trusted_env) if trusted_env is not None else set(self._allowed)
        )
        self._app: Application | None = None
        # Live approval prompts keyed by approval_id -> (chat_id, message_id),
        # so the inline buttons can be cleared once the action is settled.
        self._approval_msgs: dict[str, tuple[str, int]] = {}
        # Pending batched messages keyed by SessionSource.key()
        self._pending: dict[str, list[str]] = {}
        self._batch_delay = 0.3  # seconds
        # Largest inbound text document we'll ingest. Bigger files are rejected
        # so a huge upload can't blow the model context / episodic memory.
        self._doc_max_bytes = int(os.environ.get("TELEGRAM_DOC_MAX_BYTES", "262144"))
        # Outbound formatting. Default renders Markdown; set TELEGRAM_PARSE_MODE
        # to plain/none/off (or leave the library uninstalled) for raw text.
        mode = os.environ.get("TELEGRAM_PARSE_MODE", "MarkdownV2").strip().lower()
        self._parse_mode = None if mode in ("", "plain", "none", "off") else "MarkdownV2"
        # Stream replies by live-editing one message as tokens arrive. On by
        # default; TELEGRAM_STREAMING=0/off/false delivers the reply in one shot.
        stream_env = os.environ.get("TELEGRAM_STREAMING", "1").strip().lower()
        self.supports_streaming = stream_env not in ("", "0", "off", "false", "no")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._app = Application.builder().token(self._token).build()
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_message)
        )
        # Photos and image documents -> vision turn (a caption rides along as text).
        self._app.add_handler(
            MessageHandler(filters.PHOTO | filters.Document.IMAGE, self._on_photo)
        )
        # Non-image documents (.txt/.md/.csv/...) -> read in as text. Registered
        # after the photo handler (and excluding image docs) so images still
        # route to the vision turn above.
        self._app.add_handler(
            MessageHandler(
                filters.Document.ALL & ~filters.Document.IMAGE, self._on_document
            )
        )
        self._app.add_handler(CommandHandler("reset", self._on_reset))
        # The terminal CLI's in-chat commands, mirrored over Telegram. Routed
        # to the runner so the behaviour stays identical across surfaces.
        for cmd in ("sleep", "mood", "memory", "skills"):
            self._app.add_handler(CommandHandler(cmd, self._on_command))
        self._app.add_handler(CallbackQueryHandler(self._on_callback))
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        self._state = AdapterState.RUNNING

    async def stop(self) -> None:
        self._state = AdapterState.STOPPED
        if self._app:
            if self._app.updater.running:
                await self._app.updater.stop()
            if self._app.running:
                await self._app.stop()
                await self._app.shutdown()

    # ------------------------------------------------------------------
    # Inbound
    # ------------------------------------------------------------------

    async def _on_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.effective_user or not update.effective_chat:
            return

        source = self._source_from(update)
        if not self._runner._is_user_authorized(source, self._allowed):
            await update.message.reply_text("Unauthorized.")
            return

        text = (update.message.text or "").strip()
        if not text:
            return

        key = source.key()
        if key not in self._pending:
            self._pending[key] = []
            asyncio.create_task(self._flush_batch(source))
        self._pending[key].append(text)

    async def _flush_batch(self, source: SessionSource) -> None:
        """Wait for the debounce window, then send the combined text to the agent."""
        await asyncio.sleep(self._batch_delay)
        key = source.key()
        texts = self._pending.pop(key, [])
        if not texts:
            return

        await self.send_typing(source)
        await self._runner.handle_inbound(source, " ".join(texts))

    async def _on_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle an inbound photo (or image document): download the image,
        base64-encode it, and pass it to the agent as a vision turn. Any caption
        rides along as the message text. Photos bypass the text debounce batch —
        each is its own turn — but still serialise on the agent lock upstream."""
        msg = update.message
        if not msg or not update.effective_user or not update.effective_chat:
            return

        source = self._source_from(update)
        if not self._runner._is_user_authorized(source, self._allowed):
            await msg.reply_text("Unauthorized.")
            return

        # Don't let the API reject the request — tell the owner the model is blind.
        from ... import llm
        if not llm.supports_vision():
            await msg.reply_text(
                "I can't analyse images with the current model. "
                "Switch DP_CORE_MODEL to a vision-capable one to enable this."
            )
            return

        image = await self._download_image(msg)
        if image is None:
            await msg.reply_text("Sorry, I couldn't download that image.")
            return

        text = (msg.caption or "").strip() or "(image, no caption)"
        await self.send_typing(source)
        await self._runner.handle_inbound(source, text, images=[image])

    async def _download_image(self, msg) -> dict | None:
        """Fetch the best version of the image on ``msg`` and return it as a
        ``{"media_type", "data"}`` dict (base64), or None if nothing usable is
        attached. Telegram re-encodes ``photo`` uploads as JPEG; image documents
        keep their original mime type."""
        if msg.photo:
            tg_file = await msg.photo[-1].get_file()   # last = highest resolution
            media_type = "image/jpeg"
        elif msg.document and (msg.document.mime_type or "").startswith("image/"):
            tg_file = await msg.document.get_file()
            media_type = msg.document.mime_type
        else:
            return None
        try:
            raw = await tg_file.download_as_bytearray()
        except Exception as exc:
            security.audit("telegram_image_download_error", str(exc))
            return None
        return {"media_type": media_type, "data": base64.b64encode(raw).decode("ascii")}

    async def _on_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle an inbound text document: download it, decode as UTF-8, and pass
        the contents to the agent as a turn (folded in as untrusted data and saved
        to the user's workspace upstream). Any caption rides along as the message
        text. Image documents are handled by ``_on_photo``; binary documents are
        rejected. Like photos, documents bypass the text debounce batch."""
        msg = update.message
        if not msg or not msg.document or not update.effective_user or not update.effective_chat:
            return

        source = self._source_from(update)
        if not self._runner._is_user_authorized(source, self._allowed):
            await msg.reply_text("Unauthorized.")
            return

        doc = msg.document
        if doc.file_size and doc.file_size > self._doc_max_bytes:
            await msg.reply_text(
                f"That document is too large ({doc.file_size} bytes). "
                f"I can read text files up to {self._doc_max_bytes} bytes."
            )
            return

        try:
            tg_file = await doc.get_file()
            raw = await tg_file.download_as_bytearray()
        except Exception as exc:
            security.audit("telegram_document_download_error", str(exc))
            await msg.reply_text("Sorry, I couldn't download that document.")
            return

        try:
            text = bytes(raw).decode("utf-8")
        except UnicodeDecodeError:
            await msg.reply_text("I can only read text documents.")
            return

        caption = (msg.caption or "").strip()
        await self.send_typing(source)
        await self._runner.handle_inbound(
            source, caption,
            documents=[{"filename": doc.file_name or "document.txt", "content": text}],
        )

    async def _on_reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_user:
            return
        source = self._source_from(update)
        if not self._runner._is_user_authorized(source, self._allowed):
            return
        self._runner._sessions.reset(source.key())
        await update.message.reply_text("Session reset.")
        security.audit("gateway_session_reset", source.key())

    async def _on_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the CLI-parity commands (/sleep, /mood, /memory, /skills)."""
        if not update.message or not update.effective_user or not update.effective_chat:
            return
        source = self._source_from(update)
        if not self._runner._is_user_authorized(source, self._allowed):
            await update.message.reply_text("Unauthorized.")
            return
        # Strip the leading slash and any "@botname" suffix Telegram appends in
        # group chats, leaving the bare command name.
        cmd = (update.message.text or "").lstrip("/").split()[0].split("@", 1)[0].lower()
        await self.send_typing(source)
        reply = await self._runner.handle_command(source, cmd)
        await self.send(source, reply)   # send() splits long output (e.g. /memory)

    # ------------------------------------------------------------------
    # Approvals (high-impact action gate)
    # ------------------------------------------------------------------

    def can_approve(self, user_id) -> bool:
        """Whether this user is trusted to approve high-impact actions."""
        return str(user_id) in self._trusted

    async def request_approval(self, source: SessionSource, approval_id: str,
                               prompt: str) -> None:
        """Send an Approve/Deny prompt for a high-impact action. Runs on the
        gateway loop; the agent thread is blocked waiting for the answer."""
        if not self._app:
            return
        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton("✅ Approve", callback_data=f"dpok:{approval_id}"),
            InlineKeyboardButton("🚫 Deny", callback_data=f"dpno:{approval_id}"),
        ]])
        kwargs: dict = {}
        if source.thread_id:
            kwargs["message_thread_id"] = int(source.thread_id)
        msg = await self._app.bot.send_message(
            chat_id=source.chat_id, text=prompt, reply_markup=keyboard, **kwargs
        )
        self._approval_msgs[approval_id] = (source.chat_id, msg.message_id)

    async def expire_approval(self, source: SessionSource, approval_id: str) -> None:
        """Replace a timed-out prompt's buttons with a terminal notice."""
        entry = self._approval_msgs.pop(approval_id, None)
        if not entry or not self._app:
            return
        chat_id, message_id = entry
        try:
            await self._app.bot.edit_message_text(
                chat_id=chat_id, message_id=message_id,
                text="⏲️ Approval request timed out — action denied.",
            )
        except BadRequest:
            pass

    async def _on_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if query is None or not query.data or ":" not in query.data:
            return
        action, approval_id = query.data.split(":", 1)
        if action not in ("dpok", "dpno"):
            return

        # Only a trusted user may approve, even if they can press the button.
        user = update.effective_user
        if user is None or not self.can_approve(user.id):
            await query.answer("Unauthorized.", show_alert=True)
            security.audit("approval_unauthorized",
                           f"{user.id if user else '?'} {approval_id}")
            return
        await query.answer()

        approved = action == "dpok"
        settled = self._runner.resolve_approval(approval_id, approved)
        self._approval_msgs.pop(approval_id, None)

        if not settled:
            verdict = "⏲️ This request already expired."
        else:
            verdict = "✅ Approved." if approved else "🚫 Denied."
        base = query.message.text if query.message else ""
        try:
            await query.edit_message_text(f"{base}\n\n— {verdict}".strip())
        except BadRequest:
            pass

    # ------------------------------------------------------------------
    # Outbound
    # ------------------------------------------------------------------

    async def send(self, source: SessionSource, text: str) -> None:
        if not self._app or not text.strip():
            return                          # nothing to say; Telegram rejects empty
        kwargs: dict = {}
        if source.thread_id:
            kwargs["message_thread_id"] = int(source.thread_id)
        for chunk in _split(text):
            await self._send_chunk(source.chat_id, chunk, kwargs)

    async def send_link_button(self, source: SessionSource, text: str,
                               label: str, url: str) -> None:
        """Send *text* with an inline URL button (used for the pay prompt). A
        t.me/<bot>/<app> link opens the Mini App directly."""
        if not self._app or not text.strip():
            return
        if not url:
            await self.send(source, text)
            return
        kwargs: dict = {"reply_markup": InlineKeyboardMarkup(
            [[InlineKeyboardButton(label, url=url)]])}
        if source.thread_id:
            kwargs["message_thread_id"] = int(source.thread_id)
        # Keep the button attached: send as one chunk (pay prompts are short).
        await self._send_chunk(source.chat_id, text, kwargs)

    async def send_document(self, source: SessionSource, filename: str,
                            content: str) -> None:
        """Deliver text ``content`` as a file attachment named ``filename``."""
        if not self._app:
            return
        kwargs: dict = {}
        if source.thread_id:
            kwargs["message_thread_id"] = int(source.thread_id)
        await self._app.bot.send_document(
            chat_id=source.chat_id,
            document=InputFile(content.encode("utf-8"), filename=filename),
            **kwargs,
        )

    async def _send_chunk(self, chat_id: str, chunk: str, kwargs: dict) -> None:
        """Deliver one <=4096-char chunk, rendering Markdown when enabled.

        The model emits CommonMark; we convert it to the MarkdownV2 dialect
        Telegram requires. If Telegram still rejects the formatted version we
        resend the original chunk as plain text, so a formatting glitch never
        costs the user the message.
        """
        if self._parse_mode and telegramify_markdown is not None:
            try:
                body = telegramify_markdown.markdownify(chunk)
                await self._app.bot.send_message(
                    chat_id=chat_id, text=body, parse_mode=self._parse_mode, **kwargs
                )
                return
            except BadRequest as exc:
                security.audit("telegram_markdown_fallback", str(exc))
        await self._app.bot.send_message(chat_id=chat_id, text=chunk, **kwargs)

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def stream_sink(self, source: SessionSource, loop) -> _StreamSink:
        """A sink that live-edits one message as the agent streams its reply.
        ``loop`` is the gateway's event loop; the agent feeds deltas from a
        worker thread, so edits are marshalled back onto it."""
        return _StreamSink(self, source, loop)

    async def _stream_render(self, source: SessionSource, message, text: str) -> object:
        """Show ``text`` as plain text: create the message if needed, else edit
        it. Returns the message object (created or existing). Used for the
        intermediate, still-growing frames."""
        text = _clip(text.strip())
        if not text:
            return message
        if message is None:
            kwargs: dict = {}
            if source.thread_id:
                kwargs["message_thread_id"] = int(source.thread_id)
            return await self._app.bot.send_message(
                chat_id=source.chat_id, text=text, **kwargs
            )
        try:
            await self._app.bot.edit_message_text(
                chat_id=source.chat_id, message_id=message.message_id, text=text
            )
        except BadRequest as exc:
            if "not modified" not in str(exc).lower():
                raise
        return message

    async def _stream_finalize(self, source: SessionSource, message, text: str) -> None:
        """Replace the streamed placeholder with the final, Markdown-rendered
        reply. Long replies are split: the first chunk edits the placeholder,
        the rest are sent as fresh messages (a message can't exceed 4096)."""
        if not text.strip():
            return                          # nothing to commit; never send empty
        chunks = _split(text)
        if not chunks:
            return
        if message is None:
            await self.send(source, text)
            return

        first, rest = chunks[0], chunks[1:]
        body = first
        parse_mode = None
        if self._parse_mode and telegramify_markdown is not None:
            body, parse_mode = telegramify_markdown.markdownify(first), self._parse_mode
        try:
            await self._app.bot.edit_message_text(
                chat_id=source.chat_id, message_id=message.message_id,
                text=body, parse_mode=parse_mode,
            )
        except BadRequest as exc:
            low = str(exc).lower()
            if "not modified" not in low:
                if parse_mode:                      # formatting rejected -> plain
                    security.audit("telegram_markdown_fallback", str(exc))
                    await self._app.bot.edit_message_text(
                        chat_id=source.chat_id, message_id=message.message_id, text=first
                    )
                else:
                    raise

        kwargs: dict = {}
        if source.thread_id:
            kwargs["message_thread_id"] = int(source.thread_id)
        for chunk in rest:
            await self._send_chunk(source.chat_id, chunk, kwargs)

    async def send_typing(self, source: SessionSource) -> None:
        if not self._app:
            return
        try:
            await self._app.bot.send_chat_action(chat_id=source.chat_id, action="typing")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _source_from(self, update: Update) -> SessionSource:
        chat = update.effective_chat
        user = update.effective_user
        thread_id = (
            str(update.message.message_thread_id)
            if update.message and update.message.is_topic_message
            else None
        )
        return SessionSource(
            platform="telegram",
            chat_id=str(chat.id),
            user_id=str(user.id),
            thread_id=thread_id,
        )


class _StreamSink:
    """Renders one streamed reply as a single live-edited Telegram message.

    ``feed`` is called from the agent's worker thread for every text fragment;
    it buffers the text and, on a throttle, schedules a plain-text edit on the
    gateway loop. ``finalize`` (awaited on the loop once the reply is complete)
    applies Markdown. A short silence sentinel is never shown: nothing is sent
    until the buffer is clearly not a sentinel.
    """

    def __init__(self, adapter: TelegramAdapter, source: SessionSource, loop) -> None:
        self._adapter = adapter
        self._source = source
        self._loop = loop
        self._buf = ""
        self._message = None          # telegram Message, once created
        self._last_edit = 0.0
        self._lock = asyncio.Lock()   # serialises edits; orders interim vs final
        self._done = False

    # -- called from the worker thread -------------------------------------
    def feed(self, delta: str) -> None:
        if not delta:
            return
        self._buf += delta
        if not _revealable(self._buf):
            return
        now = time.monotonic()
        if self._message is None or now - self._last_edit >= _STREAM_EDIT_INTERVAL:
            self._last_edit = now
            asyncio.run_coroutine_threadsafe(self._render(self._buf), self._loop)

    # -- run on the gateway loop -------------------------------------------
    async def _render(self, snapshot: str) -> None:
        async with self._lock:
            if self._done or not self._adapter._app:
                return
            self._message = await self._adapter._stream_render(
                self._source, self._message, snapshot
            )

    async def finalize(self, reply: str) -> None:
        """Commit the final reply. ``reply`` is the agent's return value; the
        streamed buffer (everything shown) is preferred when present so the
        bubble isn't truncated to just the last turn."""
        async with self._lock:
            self._done = True
            text = self._buf if self._buf.strip() else reply
            if not text.strip():
                # The model produced no visible text at all. Drop any placeholder
                # rather than committing an empty message Telegram would reject.
                await self._delete_message()
                return
            await self._adapter._stream_finalize(self._source, self._message, text)

    async def discard(self) -> None:
        """Drop the in-progress message (e.g. the reply turned out silent)."""
        async with self._lock:
            self._done = True
            await self._delete_message()

    async def _delete_message(self) -> None:
        """Best-effort removal of the placeholder, if one was created."""
        if self._message is not None and self._adapter._app:
            try:
                await self._adapter._app.bot.delete_message(
                    chat_id=self._source.chat_id,
                    message_id=self._message.message_id,
                )
            except Exception:
                pass


def _parse_ids(spec: str) -> set[str]:
    """Parse a comma-separated list of user ids into a set, dropping blanks."""
    return {u.strip() for u in spec.split(",") if u.strip()}


def _clip(text: str, limit: int = _TELEGRAM_MSG_LIMIT) -> str:
    """A single Telegram message can't exceed the hard limit; clip interim
    streaming frames (the final reply is split across messages instead)."""
    return text if len(text) <= limit else text[:limit]


def _revealable(text: str) -> bool:
    """True once buffered text is clearly not a silence sentinel, so a streamed
    ``[SILENT]`` / ``NO_REPLY`` is never momentarily shown before it's swallowed."""
    s = text.strip()
    if not s:
        return False
    return not any(tok.startswith(s) or tok in s for tok in SILENCE_TOKENS)


def _split(text: str, limit: int = _SPLIT_LIMIT) -> list[str]:
    """Split text into chunks under ``limit``, breaking on line boundaries and
    keeping fenced code blocks (```) intact so per-chunk Markdown conversion
    can't be handed a half-open fence. Over-long single lines are hard-split,
    and every chunk is finally capped at the hard Telegram limit."""
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    cur_len = 0
    in_fence = False

    def flush() -> None:
        nonlocal current, cur_len
        if current:
            chunks.append("\n".join(current))
            current = []
            cur_len = 0

    for line in text.split("\n"):
        is_fence = line.lstrip().startswith("```")
        add = len(line) + 1  # + the newline that rejoins it
        # Decide on the break using the fence state *before* this line: we may
        # break before an opening ``` (still outside the fence) but not before a
        # closing ``` (still inside it), so a fence is never severed.
        if current and cur_len + add > limit and not in_fence:
            flush()
        if is_fence:
            in_fence = not in_fence
        if add > limit:
            flush()
            chunks.extend(line[i : i + limit] for i in range(0, len(line), limit))
            continue
        current.append(line)
        cur_len += add
    flush()

    # Safety net: a single fenced block larger than the limit can still produce
    # an oversized chunk above; enforce the hard Telegram ceiling.
    out: list[str] = []
    for c in chunks:
        if len(c) <= _TELEGRAM_MSG_LIMIT:
            out.append(c)
        else:
            out.extend(
                c[i : i + _TELEGRAM_MSG_LIMIT]
                for i in range(0, len(c), _TELEGRAM_MSG_LIMIT)
            )
    return out
