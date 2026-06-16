"""Telegram platform adapter using python-telegram-bot."""
from __future__ import annotations

import asyncio
import os

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from ... import security
from ..base import AdapterState, BasePlatformAdapter, SessionSource

_TELEGRAM_MSG_LIMIT = 4096


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
            await self._app.bot.send_message(chat_id=source.chat_id, text=chunk, **kwargs)

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


def _split(text: str, limit: int = _TELEGRAM_MSG_LIMIT) -> list[str]:
    if len(text) <= limit:
        return [text]
    return [text[i : i + limit] for i in range(0, len(text), limit)]
