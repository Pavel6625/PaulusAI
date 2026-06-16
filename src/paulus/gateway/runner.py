"""GatewayRunner: routes messages between platform adapters and the agent."""
from __future__ import annotations

import asyncio

from .. import security
from .base import SILENCE_TOKENS, AdapterState, BasePlatformAdapter, SessionSource
from .session_store import SessionStore

_instance: GatewayRunner | None = None


def get_runner() -> GatewayRunner | None:
    return _instance


class GatewayRunner:
    """
    Manages platform adapters, routes inbound messages to the agent, and
    dispatches outbound messages from the send_message tool.
    """
    def __init__(self) -> None:
        global _instance
        self._adapters: dict[str, BasePlatformAdapter] = {}
        self._sessions = SessionStore()
        # Serialize agent calls: PaulusAI's memory modules are not thread-safe.
        self._agent_lock = asyncio.Lock()
        _instance = self

    def register(self, name: str, adapter: BasePlatformAdapter) -> None:
        self._adapters[name] = adapter

    def has_adapters(self) -> bool:
        return bool(self._adapters)

    def _is_user_authorized(self, source: SessionSource, allowed: set[str]) -> bool:
        return not allowed or str(source.user_id) in allowed

    async def handle_inbound(self, source: SessionSource, text: str) -> None:
        """Called by adapters when a message arrives from a platform user."""
        session = self._sessions.get_or_create(source.key())
        session.touch()

        tagged = f"[via {source.platform}] {text}"

        async with self._agent_lock:
            from .. import agent as _agent
            loop = asyncio.get_running_loop()
            reply = await loop.run_in_executor(None, _agent.respond, tagged)

        if any(token in reply for token in SILENCE_TOKENS):
            return

        adapter = self._adapters.get(source.platform)
        if adapter and adapter.state == AdapterState.RUNNING:
            try:
                await adapter.send(source, reply)
                adapter._on_success()
            except Exception as exc:
                security.audit("gateway_send_error", str(exc))
                adapter._on_failure()

    def dispatch_outbound(self, to: str, body: str) -> str:
        """
        Deliver an outbound message; called synchronously from tools.execute().
        'to' format: 'platform:chat_id' or bare 'chat_id' for the default adapter.
        """
        parts = to.split(":", 1)
        if len(parts) == 2:
            platform_name, chat_id = parts
        else:
            platform_name = self._default_platform()
            chat_id = to
            if platform_name is None:
                return f"[gateway] no active adapter to deliver message to {to}."

        adapter = self._adapters.get(platform_name)
        if not adapter or adapter.state != AdapterState.RUNNING:
            return f"[{platform_name}] adapter unavailable; message to {to} not sent."

        source = SessionSource(platform=platform_name, chat_id=chat_id, user_id="agent")
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(adapter.send(source, body))
        except RuntimeError:
            # No running event loop (terminal REPL mode) — run synchronously.
            asyncio.run(adapter.send(source, body))

        return f"Message dispatched to {to} via {platform_name}."

    def _default_platform(self) -> str | None:
        for name, adapter in self._adapters.items():
            if adapter.state == AdapterState.RUNNING:
                return name
        return None

    async def start_all(self) -> None:
        for name, adapter in self._adapters.items():
            try:
                await adapter.start()
                security.audit("gateway_start", name)
            except Exception as exc:
                security.audit("gateway_start_error", f"{name}: {exc}")

    async def stop_all(self) -> None:
        for adapter in self._adapters.values():
            try:
                await adapter.stop()
            except Exception:
                pass

    def status(self) -> str:
        if not self._adapters:
            return "  (no adapters registered)"
        return "\n".join(f"  {n}: {a.state.value}" for n, a in self._adapters.items())
