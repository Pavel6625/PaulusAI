"""GatewayRunner: routes messages between platform adapters and the agent."""
from __future__ import annotations

import asyncio
import concurrent.futures
import time

from .. import config, security
from .base import SILENCE_TOKENS, AdapterState, BasePlatformAdapter, SessionSource
from .presence import PresenceStore
from .session_store import SessionStore

_instance: GatewayRunner | None = None


def get_runner() -> GatewayRunner | None:
    return _instance


def _approval_prompt(tool_name: str, tool_input) -> str:
    """A concise, human-readable description of the action awaiting approval."""
    if isinstance(tool_input, dict):
        if tool_name == "run_command":
            detail = tool_input.get("command", "")
        elif tool_name == "write_local_file":
            detail = f"write to {tool_input.get('path', '?')}"
        elif tool_name == "send_message":
            body = str(tool_input.get("body", ""))
            preview = body if len(body) <= 200 else body[:200] + "…"
            detail = f"to {tool_input.get('to', '?')}: {preview}"
        else:
            detail = str(tool_input)
    else:
        detail = str(tool_input)
    return (
        "⚠️ Approval needed for a high-impact action.\n"
        f"Action: {tool_name}\n"
        f"Details: {detail}\n\n"
        "Approve this single action?"
    )


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
        # The gateway event loop, captured once it is running, so the agent's
        # worker thread can marshal interactive-approval prompts back onto it.
        self._loop: asyncio.AbstractEventLoop | None = None
        # Pending high-impact approvals awaiting a user's Approve/Deny, keyed by
        # a short id carried in the prompt's callback.
        self._pending_approvals: dict[str, concurrent.futures.Future] = {}
        self._approval_seq = 0
        _instance = self

    def register(self, name: str, adapter: BasePlatformAdapter) -> None:
        self._adapters[name] = adapter

    def has_adapters(self) -> bool:
        return bool(self._adapters)

    def _is_user_authorized(self, source: SessionSource, allowed: set[str]) -> bool:
        return not allowed or str(source.user_id) in allowed

    async def handle_inbound(self, source: SessionSource, text: str,
                             images: list | None = None) -> None:
        """Called by adapters when a message arrives from a platform user.

        ``images`` (optional) is a list of ``{"media_type", "data"}`` dicts
        (base64) attached to this turn — used by adapters that accept photos."""
        session = self._sessions.get_or_create(source.key())
        session.touch()
        self._presence.touch(source)   # per-user idle clock + reachability

        tagged = f"[via {source.platform}] {text}"

        adapter = self._adapters.get(source.platform)
        # Stream (live-edit the reply as it's generated) when the adapter opts
        # in; otherwise deliver the reply in one shot once it's complete.
        sink = None
        if (adapter and adapter.state == AdapterState.RUNNING
                and getattr(adapter, "supports_streaming", False)):
            sink = adapter.stream_sink(source, asyncio.get_running_loop())

        async with self._agent_lock:
            from .. import agent as _agent
            loop = asyncio.get_running_loop()
            user_id = str(source.user_id)
            on_delta = sink.feed if sink else None
            reply = await loop.run_in_executor(
                None,
                lambda: _agent.respond(tagged, user_id=user_id, on_delta=on_delta,
                                       images=images),
            )

        silent = any(token in reply for token in SILENCE_TOKENS)

        if sink is not None:
            try:
                await (sink.discard() if silent else sink.finalize(reply))
                adapter._on_success()
            except Exception as exc:
                security.audit("gateway_send_error", str(exc))
                adapter._on_failure()
            return

        if silent:
            return
        if adapter and adapter.state == AdapterState.RUNNING:
            try:
                await adapter.send(source, reply)
                adapter._on_success()
            except Exception as exc:
                security.audit("gateway_send_error", str(exc))
                adapter._on_failure()

    async def handle_command(self, source: SessionSource, command: str) -> str:
        """Run an in-chat slash command and return the reply text. Mirrors the
        terminal CLI's /sleep, /mood, /memory and /skills, but scoped to the
        calling user (per-user episodic/semantic memory; skills and mood are
        global). Called by adapters that register command handlers."""
        from .. import affect, memory, skills
        user_id = str(source.user_id)

        if command == "sleep":
            # Consolidation hits the model and writes memory, so it runs under
            # the agent lock (PaulusAI's memory modules aren't thread-safe) and
            # off the event loop, exactly like a normal turn.
            self._presence.touch(source)
            async with self._agent_lock:
                from .. import agent as _agent
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(
                    None, lambda: _agent.sleep(user_id=user_id)
                )
        if command == "mood":
            return f"mood: {affect.describe()}"
        if command == "memory":
            return memory.semantic_text(user_id)
        if command == "skills":
            return skills.describe()
        return f"Unknown command: /{command}"

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

    # ------------------------------------------------------------------
    # Interactive approval of high-impact actions
    # ------------------------------------------------------------------

    def request_approval(self, user_id: str, tool_name: str, tool_input) -> bool | None:
        """Ask a reachable user to approve a single high-impact action.

        Called synchronously from the agent's worker thread (the gate blocks
        there). Returns True (approved), False (denied or timed out), or None
        when the user has no interactive channel — so the caller falls back to
        the unattended policy. The Approve/Deny answer arrives out-of-band (e.g.
        a Telegram button), bypassing the agent lock, and resolves the future.
        """
        reason = self._approval_unavailable_reason(user_id)
        if reason is not None:
            # Audit WHY no buttons were shown so a misconfiguration (e.g. the user
            # not being in TELEGRAM_TRUSTED_USERS) is diagnosable, then fall back.
            security.audit("approval_skip", f"{user_id} {tool_name}: {reason}")
            return None

        source = self._presence._users[str(user_id)].last_source
        adapter = self._adapters[source.platform]
        self._approval_seq += 1
        approval_id = f"{int(time.time())}-{self._approval_seq}"
        fut: concurrent.futures.Future = concurrent.futures.Future()
        self._pending_approvals[approval_id] = fut

        prompt = _approval_prompt(tool_name, tool_input)
        security.audit("approval_request", f"{user_id} {tool_name} {tool_input}")
        cf = asyncio.run_coroutine_threadsafe(
            adapter.request_approval(source, approval_id, prompt), self._loop
        )
        # If sending the prompt itself fails, surface it and deny immediately
        # rather than leaving the agent blocked until the timeout.
        cf.add_done_callback(lambda f: self._on_prompt_sent(approval_id, f))
        try:
            return fut.result(timeout=config.APPROVAL_TIMEOUT)
        except concurrent.futures.TimeoutError:
            security.audit("approval_timeout", f"{user_id} {tool_name}")
            asyncio.run_coroutine_threadsafe(
                adapter.expire_approval(source, approval_id), self._loop
            )
            return False
        finally:
            self._pending_approvals.pop(approval_id, None)

    def _approval_unavailable_reason(self, user_id: str) -> str | None:
        """Why an interactive approval can't be sought, or None if it can."""
        if self._loop is None:
            return "gateway loop not running"
        presence = self._presence._users.get(str(user_id))
        if presence is None:
            return "user not reachable (no presence record)"
        adapter = self._adapters.get(presence.last_source.platform)
        if adapter is None or adapter.state != AdapterState.RUNNING:
            return f"adapter {presence.last_source.platform} unavailable"
        if not getattr(adapter, "supports_approvals", False):
            return f"adapter {presence.last_source.platform} has no approval UI"
        if not adapter.can_approve(user_id):
            return "user not trusted to approve (TELEGRAM_TRUSTED_USERS)"
        return None

    def _on_prompt_sent(self, approval_id: str, cf: concurrent.futures.Future) -> None:
        exc = cf.exception()
        if exc is not None:
            security.audit("approval_send_error", f"{approval_id}: {exc}")
            self.resolve_approval(approval_id, False)

    def resolve_approval(self, approval_id: str, approved: bool) -> bool:
        """Settle a pending approval. Called on the gateway loop by an adapter
        when the user answers. Returns False if the id is unknown or already
        settled (e.g. it had timed out)."""
        fut = self._pending_approvals.get(approval_id)
        if fut is None or fut.done():
            return False
        fut.set_result(approved)
        return True

    async def start_all(self) -> None:
        self._loop = asyncio.get_running_loop()
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
