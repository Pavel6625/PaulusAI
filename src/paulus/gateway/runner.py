"""GatewayRunner: routes messages between platform adapters and the agent."""
from __future__ import annotations

import asyncio
import time

from .. import config, security
from .base import SILENCE_TOKENS, AdapterState, BasePlatformAdapter, SessionSource
from .presence import PresenceStore
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
        self._presence = PresenceStore(config.PRESENCE_FILE)
        # Serialize agent calls: PaulusAI's memory modules are not thread-safe.
        self._agent_lock = asyncio.Lock()
        self._idle_task: asyncio.Task | None = None
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
        self._presence.touch(source)   # per-user idle clock + reachability

        tagged = f"[via {source.platform}] {text}"

        async with self._agent_lock:
            from .. import agent as _agent
            loop = asyncio.get_running_loop()
            user_id = str(source.user_id)
            reply = await loop.run_in_executor(
                None, lambda: _agent.respond(tagged, user_id=user_id)
            )

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
        security.audit(
            "gateway_boot",
            f"data_dir={config.DATA_DIR} idle_check={config.IDLE_CHECK_INTERVAL}s "
            f"min_idle={config.MIN_IDLE_MINUTES}m max_msg={config.MAX_IDLE_MSG_SESSION} "
            f"quiet={config.QUIET_START}-{config.QUIET_END}",
        )
        for name, adapter in self._adapters.items():
            try:
                await adapter.start()
                security.audit("gateway_start", name)
            except Exception as exc:
                security.audit("gateway_start_error", f"{name}: {exc}")

        if config.IDLE_CHECK_INTERVAL > 0 and self._idle_task is None:
            self._idle_task = asyncio.create_task(self._idle_loop())
            security.audit("idle_loop_start", f"every {config.IDLE_CHECK_INTERVAL}s")
        elif config.IDLE_CHECK_INTERVAL <= 0:
            security.audit("idle_loop_disabled", "DP_IDLE_CHECK<=0")

    async def stop_all(self) -> None:
        if self._idle_task is not None:
            self._idle_task.cancel()
            try:
                await self._idle_task
            except asyncio.CancelledError:
                pass
            self._idle_task = None

        for adapter in self._adapters.values():
            try:
                await adapter.stop()
            except Exception:
                pass

    async def _idle_loop(self) -> None:
        """Periodically reach out to users who have gone quiet. Idleness is
        tracked per user; each gets at most MAX_IDLE_MSG_SESSION unprompted
        messages until they speak again, and the model may decline (stay
        silent) on any given check."""
        min_idle_s = config.MIN_IDLE_MINUTES * 60
        while True:
            try:
                await asyncio.sleep(config.IDLE_CHECK_INTERVAL)
                await self._run_idle_pass(min_idle_s)
            except asyncio.CancelledError:
                raise
            except Exception as exc:               # never let the loop die
                security.audit("idle_loop_error", str(exc))

    async def _run_idle_pass(self, min_idle_s: float) -> None:
        if config.in_quiet_hours():
            security.audit("idle_skip", "quiet_hours")
            return                             # don't ping during quiet hours
        candidates = self._presence.idle_users(min_idle_s, config.MAX_IDLE_MSG_SESSION)
        security.audit(
            "idle_pass",
            f"tracked={len(self._presence._users)} candidates={len(candidates)} "
            f"min_idle={min_idle_s / 60:.0f}m cap={config.MAX_IDLE_MSG_SESSION}",
        )
        for p in candidates:
            idle_min = (time.time() - p.last_active) / 60
            adapter = self._adapters.get(p.last_source.platform)
            if not adapter or adapter.state != AdapterState.RUNNING:
                state = adapter.state.value if adapter else "missing"
                security.audit(
                    "idle_skip",
                    f"{p.user_id} adapter={p.last_source.platform}:{state}",
                )
                continue

            security.audit(
                "idle_check",
                f"{p.user_id} idle={idle_min:.1f}m nudges={p.nudges_sent} "
                f"reach={p.last_source.platform}:{p.last_source.chat_id}",
            )
            async with self._agent_lock:
                from .. import agent as _agent
                loop = asyncio.get_running_loop()
                user_id = p.user_id
                reply = await loop.run_in_executor(
                    None, lambda u=user_id: _agent.proactive_check(u)
                )

            if any(token in reply for token in SILENCE_TOKENS):
                security.audit("idle_silent", p.user_id)
                continue                           # model chose not to intrude

            try:
                await adapter.send(p.last_source, reply)
                adapter._on_success()
                self._presence.mark_nudged(p.user_id)
                security.audit("idle_nudge", p.user_id)
            except Exception as exc:
                security.audit("idle_send_error", str(exc))
                adapter._on_failure()

    def status(self) -> str:
        if not self._adapters:
            return "  (no adapters registered)"
        return "\n".join(f"  {n}: {a.state.value}" for n, a in self._adapters.items())
