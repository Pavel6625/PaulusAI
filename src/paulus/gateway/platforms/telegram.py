"""Telegram platform adapter using python-telegram-bot."""
from __future__ import annotations

import asyncio
import os

from telegram import Update
from telegram.error import BadRequest
from telegram.ext import (
    Application,
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
from ..base import AdapterState, BasePlatformAdapter, SessionSource

_TELEGRAM_MSG_LIMIT = 4096
# Split raw text below the hard limit: MarkdownV2 escaping adds backslashes, so
# a converted chunk can grow past its pre-escaped length.
_SPLIT_LIMIT = 3500


class TelegramAdapter(BasePlatformAdapter):
    """
    Telegram integration via python-telegram-bot.
    Supports DMs, group chats, and forum topics.
    Rapid messages from the same chat are batched with a 300 ms debounce.
    """
    supports_typing_indicator = True

    def __init__(self, runner) -> None:
        super().__init__(runner)
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        if not token:
            raise ValueError("TELEGRAM_BOT_TOKEN environment variable not set.")
        self._token = token
        self._allowed: set[str] = {
            u.strip()
            for u in os.environ.get("TELEGRAM_ALLOWED_USERS", "").split(",")
            if u.strip()
        }
        self._app: Application | None = None
        # Pending batched messages keyed by SessionSource.key()
        self._pending: dict[str, list[str]] = {}
        self._batch_delay = 0.3  # seconds
        # Outbound formatting. Default renders Markdown; set TELEGRAM_PARSE_MODE
        # to plain/none/off (or leave the library uninstalled) for raw text.
        mode = os.environ.get("TELEGRAM_PARSE_MODE", "MarkdownV2").strip().lower()
        self._parse_mode = None if mode in ("", "plain", "none", "off") else "MarkdownV2"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._app = Application.builder().token(self._token).build()
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_message)
        )
        self._app.add_handler(CommandHandler("reset", self._on_reset))
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

    async def _on_reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_user:
            return
        source = self._source_from(update)
        if not self._runner._is_user_authorized(source, self._allowed):
            return
        self._runner._sessions.reset(source.key())
        await update.message.reply_text("Session reset.")
        security.audit("gateway_session_reset", source.key())

    # ------------------------------------------------------------------
    # Outbound
    # ------------------------------------------------------------------

    async def send(self, source: SessionSource, text: str) -> None:
        if not self._app:
            return
        kwargs: dict = {}
        if source.thread_id:
            kwargs["message_thread_id"] = int(source.thread_id)
        for chunk in _split(text):
            await self._send_chunk(source.chat_id, chunk, kwargs)

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
