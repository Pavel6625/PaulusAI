"""GatewayRunner: routes messages between platform adapters and the agent."""
from __future__ import annotations

import asyncio
import concurrent.futures
import itertools
import time
import uuid
from dataclasses import asdict

from .. import billing, config, grants, interactions, security
from .base import SILENCE_TOKENS, AdapterState, BasePlatformAdapter, SessionSource
from .presence import PresenceStore
from .session_store import SessionStore

_instance: GatewayRunner | None = None

# Leaves room for the prompt's framing and an auto-review concern within
# Telegram's 4096-char message limit.
_APPROVAL_DETAIL_LIMIT = 3000


def get_runner() -> GatewayRunner | None:
    return _instance


def _approval_prompt(tool_name: str, tool_input, concern: str | None = None,
                     origin: str | None = None, grantable: bool = False) -> str:
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
        elif tool_name == "send_email_agentmail":
            body = str(tool_input.get("body", ""))
            preview = body if len(body) <= 200 else body[:200] + "…"
            detail = (f"email to {tool_input.get('to', '?')}, subject "
                      f"'{tool_input.get('subject', '')}': {preview}")
        elif tool_name == "send_document":
            dest = tool_input.get("to") or "the current chat"
            detail = f"file '{tool_input.get('filename', '?')}' to {dest}"
        else:
            detail = str(tool_input)
    else:
        detail = str(tool_input)
    # Platforms cap message length (Telegram: 4096). A prompt over the cap fails
    # to send, which denies the action outright — e.g. a heredoc that writes a
    # whole file. Show the head and say how much is hidden instead.
    if len(detail) > _APPROVAL_DETAIL_LIMIT:
        hidden = len(detail) - _APPROVAL_DETAIL_LIMIT
        detail = f"{detail[:_APPROVAL_DETAIL_LIMIT]}… [+{hidden} more chars not shown]"
    source = f"From: {origin}\n" if origin else ""
    flagged = f"Auto-review flagged: {concern}\n" if concern else ""
    scope = ""
    if grantable:
        scope = ("\n“Allow for this task” also allows "
                 f"{grants.describe(grants.keys(tool_name, tool_input))} "
                 "until this task ends.")
    return (
        "⚠️ Approval needed for a high-impact action.\n"
        f"{source}"
        f"Action: {tool_name}\n"
        f"Details: {detail}\n"
        f"{flagged}\n"
        f"Approve this single action?{scope}"
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
        # Serialize conversation turns, so each one sees the last one's reply.
        # Background tasks (tasks.py) run beside them, outside this lock; the
        # memory stores lock themselves.
        self._agent_lock = asyncio.Lock()
        self._idle_task: asyncio.Task | None = None
        self._timer_task: asyncio.Task | None = None
        self._timer_stop = asyncio.Event()
        # The gateway event loop, captured once it is running, so the agent's
        # worker thread can marshal interactive-approval prompts back onto it.
        self._loop: asyncio.AbstractEventLoop | None = None
        # Pending high-impact approvals awaiting a user's Approve/Deny, keyed by
        # a short id carried in the prompt's callback.
        self._pending_approvals: dict[str, concurrent.futures.Future] = {}
        # Prompts are requested from several threads at once (subagents,
        # background tasks); next() on a count is atomic, += is not.
        self._approval_seq = itertools.count(1)
        _instance = self
        from .. import tasks
        tasks.set_notifier(self.notify_user)

    def register(self, name: str, adapter: BasePlatformAdapter) -> None:
        self._adapters[name] = adapter

    def has_adapters(self) -> bool:
        return bool(self._adapters)

    def _is_user_authorized(self, source: SessionSource, allowed: set[str]) -> bool:
        return not allowed or str(source.user_id) in allowed

    async def handle_inbound(self, source: SessionSource, text: str,
                             images: list | None = None,
                             documents: list | None = None) -> None:
        """Called by adapters when a message arrives from a platform user.

        ``images`` (optional) is a list of ``{"media_type", "data"}`` dicts
        (base64) attached to this turn — used by adapters that accept photos.
        ``documents`` (optional) is a list of ``{"filename", "content"}`` text
        documents — folded into the turn and saved to the user's workspace."""
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

        reply = None
        error: Exception | None = None
        async with self._agent_lock:
            from .. import agent as _agent
            loop = asyncio.get_running_loop()
            user_id = str(source.user_id)
            on_delta = sink.feed if sink else None
            on_break = sink.end_segment if sink else None
            def respond():
                from .. import tasks
                token = tasks.INTERACTION_SOURCE.set(asdict(source))
                try:
                    options = {}
                    if self._loop is not None:
                        options["on_interaction"] = self._interaction_callback(
                            source, f"turn:{uuid.uuid4().hex}", tagged)
                    return _agent.respond(tagged, user_id=user_id, on_delta=on_delta,
                                          images=images, documents=documents,
                                          on_break=on_break, **options)
                finally:
                    tasks.INTERACTION_SOURCE.reset(token)
            try:
                reply = await loop.run_in_executor(None, respond)
            except security.DeferredInteraction as exc:
                reply = f"Waiting for your response to {exc.request_id}."
            except Exception as exc:        # model/tool failure mid-turn
                error = exc

        # The agent raised (e.g. a non-vision model rejecting an image). Drop any
        # streamed placeholder and tell the user, instead of dying silently.
        if error is not None:
            security.audit("gateway_agent_error", f"{source.key()}: {error}")
            if sink is not None:
                try:
                    await sink.discard()
                except Exception:
                    pass
            note = (
                "Sorry — I couldn't process that image. The model it was sent "
                "to may not be vision-capable; set DP_CORE_MODEL (or "
                "DP_ROUTER_MODEL) to one that supports images."
                if images else
                "Sorry — something went wrong handling that. Please try again."
            )
            if adapter and adapter.state == AdapterState.RUNNING:
                try:
                    await adapter.send(source, note)
                except Exception as exc:
                    security.audit("gateway_send_error", str(exc))
                    adapter._on_failure()
            return

        # A pay-gate block carries its top-up link inline (see billing.gate());
        # pull it back out so it can render as a real button rather than a
        # pasted URL. text == reply, unchanged, when no link is present.
        text, pay_url = billing.split_pay_link(reply)
        silent = any(token in text for token in SILENCE_TOKENS)

        if sink is not None:
            try:
                if silent:
                    await sink.discard()
                elif pay_url:
                    # Nothing was streamed for a pay-gate block (it short-circuits
                    # before the model runs), so there's no placeholder to finalize.
                    await sink.discard()
                    await adapter.send_with_link(source, text, billing.TOPUP_BUTTON_LABEL, pay_url)
                else:
                    await sink.finalize(reply)
                adapter._on_success()
            except Exception as exc:
                security.audit("gateway_send_error", str(exc))
                adapter._on_failure()
            return

        if silent:
            return
        if adapter and adapter.state == AdapterState.RUNNING:
            try:
                if pay_url:
                    await adapter.send_with_link(source, text, billing.TOPUP_BUTTON_LABEL, pay_url)
                else:
                    await adapter.send(source, text)
                adapter._on_success()
            except Exception as exc:
                security.audit("gateway_send_error", str(exc))
                adapter._on_failure()

    async def handle_command(self, source: SessionSource, command: str,
                             args: str = "") -> str:
        """Run an in-chat slash command and return the reply text. Mirrors the
        terminal CLI's /sleep, /mood, /memory, /skills, /bg, /tasks and
        /cancel, but scoped to the calling user (per-user episodic/semantic
        memory and tasks; skills and mood are global). Called by adapters that
        register command handlers."""
        from .. import affect, memory, skills, tasks
        user_id = str(source.user_id)

        if command == "answer":
            parts = args.strip().split(None, 1)
            if len(parts) != 2:
                return "Usage: /answer <id> <text>"
            request_id, answer = parts
            record = interactions.get(request_id)
            if record is None or record["kind"] != "input" or not interactions.answer(
                    request_id, user_id, asdict(source), answer):
                return "That question is unavailable or has already been answered."
            asyncio.create_task(self._resume_interaction(record["id"], source))
            return f"Answer received for {request_id}; resuming."

        if command == "bg":
            # Starting a task is quick (routing + a thread), but it reads and
            # writes memory, so it takes the same path as a turn.
            self._presence.touch(source)
            async with self._agent_lock:
                loop = asyncio.get_running_loop()
                def start_task():
                    token = tasks.INTERACTION_SOURCE.set(asdict(source))
                    try:
                        return tasks.start_from_owner(args, user_id=user_id)
                    finally:
                        tasks.INTERACTION_SOURCE.reset(token)
                return await loop.run_in_executor(None, start_task)
        if command == "resume":
            self._presence.touch(source)
            request_id = args.strip()
            interaction_id = request_id[2:] if request_id.startswith("d-") else request_id
            record = interactions.get(interaction_id)
            if record is not None:
                if record["user_id"] != user_id or any(
                    record["source"].get(k) != asdict(source).get(k)
                    for k in ("platform", "chat_id", "user_id", "thread_id")
                ):
                    return "That interaction is unavailable."
                if record["status"] in {"answered", "expired"}:
                    if record["kind"] == "approval" and record["decision"] == "approve":
                        adapter = self._adapters.get(source.platform)
                        if adapter is None or not adapter.can_approve(user_id):
                            return "That interaction is unavailable."
                    asyncio.create_task(self._resume_interaction(interaction_id, source))
                    return f"Resuming interaction {interaction_id}."
                if interactions.continuation_ready(interaction_id):
                    asyncio.create_task(self._resume_interaction(interaction_id, source, continuation=True))
                    return f"Resuming interaction {interaction_id}."
                return "That interaction is not ready to resume."
            return await asyncio.get_running_loop().run_in_executor(
                None, lambda: tasks.resume(user_id, request_id))
        if command == "tasks":
            return tasks.status(user_id)
        if command == "cancel":
            return tasks.cancel(user_id, args.strip())

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

    def notify_user(self, user_id, text: str) -> bool:
        """Deliver *text* to *user_id*'s most recent chat. Called from a
        background task's thread when it finishes; blocks until sent. Returns
        whether delivery succeeded (also used for startup recovery notices)."""
        source, error = self._resolve_destination("", user_id)
        if error is not None or self._loop is None:
            security.audit("notify_skip", f"{user_id}: {error or 'loop not running'}")
            return False
        adapter = self._adapters[source.platform]
        # A task stopped for balance carries its top-up link (billing.metered).
        text, pay_url = billing.split_pay_link(text)
        send = (adapter.send_with_link(source, text, billing.TOPUP_BUTTON_LABEL, pay_url)
                if pay_url else adapter.send(source, text))
        try:
            asyncio.run_coroutine_threadsafe(send, self._loop).result(timeout=60)
            adapter._on_success()
            return True
        except Exception as exc:
            security.audit("notify_error", f"{user_id}: {exc}")
            adapter._on_failure()
            return False

    def _resolve_target(self, to: str) -> tuple[str | None, str, str | None]:
        """Resolve a 'to' spec into (platform_name, chat_id, error).
        'to' format: 'platform:chat_id' or bare 'chat_id' for the default adapter.
        ``error`` is a user-facing string when no live adapter can serve it."""
        parts = to.split(":", 1)
        if len(parts) == 2:
            platform_name, chat_id = parts
        else:
            platform_name = self._default_platform()
            chat_id = to
            if platform_name is None:
                return None, chat_id, f"[gateway] no active adapter to deliver to {to}."

        adapter = self._adapters.get(platform_name)
        if not adapter or adapter.state != AdapterState.RUNNING:
            return None, chat_id, f"[{platform_name}] adapter unavailable; nothing sent to {to}."
        return platform_name, chat_id, None

    def _schedule_send(self, coro) -> None:
        """Run an adapter send coroutine on the gateway loop, or synchronously
        when there is no running loop (terminal REPL mode)."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(coro)
        except RuntimeError:
            asyncio.run(coro)

    def dispatch_outbound(self, to: str, body: str) -> str:
        """
        Deliver an outbound message; called synchronously from tools.execute().
        'to' format: 'platform:chat_id' or bare 'chat_id' for the default adapter.
        """
        platform_name, chat_id, error = self._resolve_target(to)
        if error is not None:
            return error

        adapter = self._adapters[platform_name]
        source = SessionSource(platform=platform_name, chat_id=chat_id, user_id="agent")
        self._schedule_send(adapter.send(source, body))
        return f"Message dispatched to {to} via {platform_name}."

    def _resolve_destination(self, to: str, user_id) -> tuple[SessionSource | None, str | None]:
        """Resolve where an agent-initiated payload should go, returning
        (source, error). An explicit ``to`` ('platform:chat_id' or bare chat_id)
        wins; otherwise it falls back to the caller's most recent reachable
        conversation — preserving its thread — so the agent can answer in the
        current chat without knowing the chat id."""
        if to:
            platform_name, chat_id, error = self._resolve_target(to)
            if error is not None:
                return None, error
            return SessionSource(platform=platform_name, chat_id=chat_id, user_id="agent"), None

        presence = self._presence._users.get(str(user_id)) if user_id is not None else None
        if presence is None:
            return None, "[gateway] no destination given and no known chat for this user."
        src = presence.last_source
        adapter = self._adapters.get(src.platform)
        if not adapter or adapter.state != AdapterState.RUNNING:
            return None, f"[{src.platform}] adapter unavailable; nothing sent."
        return SessionSource(platform=src.platform, chat_id=src.chat_id,
                             user_id="agent", thread_id=src.thread_id), None

    def dispatch_document(self, to: str, filename: str, body: str, user_id=None) -> str:
        """Deliver an outbound text document as a file attachment; called
        synchronously from tools.execute(). ``to`` is optional: when empty, the
        document goes to the calling user's current chat (see
        ``_resolve_destination``). Falls back with a note if the target adapter
        can't send documents."""
        source, error = self._resolve_destination(to, user_id)
        if error is not None:
            return error

        adapter = self._adapters[source.platform]
        if not getattr(adapter, "supports_documents", False):
            return f"[{source.platform}] adapter can't send documents; {filename} not sent."

        self._schedule_send(adapter.send_document(source, filename, body))
        return f"Document '{filename}' dispatched to {source.platform}:{source.chat_id}."

    def _default_platform(self) -> str | None:
        for name, adapter in self._adapters.items():
            if adapter.state == AdapterState.RUNNING:
                return name
        return None

    # ------------------------------------------------------------------
    # Interactive approval of high-impact actions
    # ------------------------------------------------------------------

    def _interaction_callback(self, source, run_id, owner_request, task_id=None, model=None,
                              continuation_id=None):
        def suspend(payload):
            checkpoint = dict(payload["checkpoint"])
            checkpoint["owner_request"] = owner_request
            checkpoint["model"] = model or config.chat_model()
            checkpoint["concern"] = payload.get("concern")
            checkpoint["origin"] = payload.get("origin")
            request_id = interactions.create(
                run_id=run_id, user_id=str(source.user_id), source=asdict(source),
                task_id=task_id, suspend_task=task_id is not None,
                kind=payload["kind"], tool_call_id=payload["tool_call_id"],
                tool_name=payload["tool_name"], tool_input=payload["tool_input"],
                checkpoint=checkpoint, continuation_id=continuation_id)
            if task_id is None:
                self._present_interaction(request_id, source, payload)
            raise security.DeferredInteraction(request_id)
        return suspend

    def _present_interaction(self, request_id, source, payload):
        adapter = self._adapters.get(source.platform)
        if self._loop is None or adapter is None or adapter.state != AdapterState.RUNNING:
            return
        if payload["kind"] == "approval":
            checkpoint = payload.get("checkpoint", {})
            prompt = _approval_prompt(payload["tool_name"], payload["tool_input"],
                                      payload.get("concern") or checkpoint.get("concern"),
                                      payload.get("origin") or checkpoint.get("origin"))
            if not getattr(adapter, "supports_approvals", False) or not adapter.can_approve(source.user_id):
                send = adapter.send(source, f"{prompt}\nInteraction: {request_id}. "
                                    "This channel cannot approve; the action remains unexecuted.")
            else:
                send = adapter.request_approval(source, "d-" + request_id, prompt)
        else:
            args = payload["tool_input"]
            choices = args.get("choices", [])
            detail = "\nChoices: " + ", ".join(choices) if choices else ""
            send = adapter.send(source, f"Question: {args['question']}{detail}\n"
                                        f"Reply with /answer {request_id} <text>")
        future = asyncio.run_coroutine_threadsafe(send, self._loop)
        def report(f):
            try:
                f.result()
            except Exception as exc:
                security.audit("interaction_send_error", f"{request_id}: {exc}")
        future.add_done_callback(report)

    def resolve_deferred_approval(self, approval_id, approved, source):
        if not isinstance(approved, bool) or not approval_id.startswith("d-"):
            return False
        adapter = self._adapters.get(source.platform)
        if (self._loop is None or adapter is None or adapter.state != AdapterState.RUNNING
                or not adapter.can_approve(source.user_id)):
            return False
        request_id = approval_id[2:]
        record = interactions.get(request_id)
        if record is None or record["kind"] != "approval":
            return False
        if not interactions.answer(request_id, str(source.user_id), asdict(source),
                                   "approve" if approved else "deny"):
            return False
        self._loop.create_task(self._resume_interaction(request_id, source))
        return True

    async def _resume_interaction(self, request_id, source, *, continuation=False):
        from .. import tasks
        record = None
        owned = {"continuation": False}
        if not continuation:
            record = await asyncio.get_running_loop().run_in_executor(None, interactions.claim, request_id)
            if record is None:
                return
            if record["task_id"] is not None:
                completed = await asyncio.get_running_loop().run_in_executor(
                    None, lambda: tasks.resume_interaction(record, self))
                if not completed:
                    released = await asyncio.get_running_loop().run_in_executor(
                        None, interactions.release_claim, request_id)
                    adapter = self._adapters.get(source.platform)
                    if released and adapter and adapter.state == AdapterState.RUNNING:
                        await adapter.send(source, f"When ready, use /resume {request_id} "
                                                   "to retry this interaction.")
                return
        completed = False
        async with self._agent_lock:
            try:
                def resume():
                    if record is not None:
                        return self._continue_interaction(record, source, owned=owned)
                    with billing.metered(str(source.user_id)):
                        return self._run_foreground_continuation(request_id, source, owned)
                reply = await asyncio.get_running_loop().run_in_executor(None, resume)
                completed = reply is not None
            except security.DeferredInteraction as exc:
                reply = f"Waiting for your response to {exc.request_id}."
            except billing.BalanceExhausted as exc:
                # A pre-execution billing block is safe to retry; an executing
                # action remains unknown and cannot be claimed again.
                released = False
                if record is not None:
                    released = await asyncio.get_running_loop().run_in_executor(
                        None, interactions.release_claim, request_id)
                released_continuation = False
                if owned["continuation"]:
                    released_continuation = await asyncio.get_running_loop().run_in_executor(
                        None, interactions.release_continuation, request_id)
                if not released_continuation:
                    released_continuation = interactions.continuation_ready(request_id)
                visible, pay_url = billing.split_pay_link(exc.block_message)
                if released or released_continuation:
                    visible += f"\n\nAfter resolving billing, use /resume {request_id}."
                reply = visible + (billing._PAY_LINK_MARKER + pay_url if pay_url else "")
            except Exception as exc:
                security.audit("interaction_resume_error", f"{request_id}: {exc}")
                released = False
                if record is not None:
                    released = await asyncio.get_running_loop().run_in_executor(
                        None, interactions.release_claim, request_id)
                if owned["continuation"]:
                    released = await asyncio.get_running_loop().run_in_executor(
                        None, interactions.release_continuation, request_id) or released
                saved = interactions.get(request_id)
                if saved and saved["status"] == "executing":
                    reply = ("The action outcome is UNKNOWN; it may have completed and will not "
                             "be replayed. Reconcile its effects before retrying.")
                else:
                    reply = "Could not resume this interaction; its saved state requires attention."
                    if released:
                        reply += f" After resolving the problem, use /resume {request_id}."
        if reply is None:
            return
        adapter = self._adapters.get(source.platform)
        delivered = False
        if adapter and adapter.state == AdapterState.RUNNING:
            text, pay_url = billing.split_pay_link(reply)
            try:
                if pay_url:
                    await adapter.send_with_link(source, text, billing.TOPUP_BUTTON_LABEL, pay_url)
                else:
                    await adapter.send(source, text)
                delivered = True
            except Exception as exc:
                security.audit("interaction_send_error", f"{request_id}: {exc}")
        if completed and owned["continuation"]:
            transition = interactions.complete_continuation if delivered else interactions.release_continuation
            await asyncio.get_running_loop().run_in_executor(None, transition, request_id)

    def _continue_interaction(self, record, source, task=None, owned=None):
        if task is not None:
            try:
                return self._apply_interaction(record, source, task)
            except billing.BalanceExhausted as exc:
                # The task manager reports the interruption separately; pass
                # the embedded payment link through the gateway unchanged.
                if self._loop is not None:
                    adapter = self._adapters.get(source.platform)
                    if adapter and adapter.state == AdapterState.RUNNING:
                        visible, pay_url = billing.split_pay_link(exc.block_message)
                        send = (adapter.send_with_link(source, visible, billing.TOPUP_BUTTON_LABEL, pay_url)
                                if pay_url else adapter.send(source, visible))
                        asyncio.run_coroutine_threadsafe(send, self._loop)
                raise
        with billing.metered(record["user_id"]):
            return self._apply_interaction(record, source, task, owned=owned)

    def _apply_interaction(self, record, source, task=None, owned=None):
        """Apply one claimed result; unrun sibling calls are reported, not replayed.

        Resume through the normal loop with fresh tools/review. An interrupted
        executing action remains unknown; no retry of it is attempted here.
        """
        from .. import agent, tasks, tools
        state = record["checkpoint"]
        tasks._validate_progress(state)
        blocks = state["messages"][-1]["content"]
        matches = [b for b in blocks if b.get("type") == "tool_use"
                   and b.get("id") == record["tool_call_id"]]
        if (len(matches) != 1 or matches[0].get("name") != record["tool_name"]
                or matches[0].get("input") != record["tool_input"]
                or any(r.get("tool_use_id") == record["tool_call_id"]
                       for r in state.get("results", []))):
            raise ValueError("Invalid interaction checkpoint")
        user_id = record["user_id"]
        allowed, block_message = billing.gate(user_id)
        if not allowed:
            raise billing.BalanceExhausted(block_message)
        agent._check(task.cancel if task else None)
        review = agent._review_context(state.get("owner_request", ""), user_id)
        if record["kind"] == "input":
            result = agent._tool_result(record["tool_call_id"], record["decision"] or
                                        "No answer received: the question expired.",
                                        is_error=record["decision"] is None)
        elif record["decision"] != "approve":
            result = agent._tool_result(record["tool_call_id"], agent._DECLINED_NOTE)
        else:
            specs = tools.agent_specs(tasks=task is None, user_id=user_id)
            from .. import mcp_client
            if state.get("enabled"):
                specs += mcp_client.specs(user_id, servers=set(state["enabled"]))
            offered = {s["name"] for s in specs}
            adapter = self._adapters.get(source.platform)
            if (record["tool_name"] not in offered
                    or not security.is_high_impact(record["tool_name"])
                    or adapter is None or not adapter.can_approve(user_id)):
                raise ValueError("Approved tool or owner approval policy is no longer available")
            if security.clearance(record["tool_name"], record["tool_input"], user_id=user_id,
                                  context=review, grant="owner") != "owner":
                raise ValueError("Approval gate rejected action")
            boundary = dict(state, phase="executing", current=record["tool_call_id"])
            if not interactions.begin_execution(record["id"],
                                                 task_checkpoint=boundary if task else None):
                raise RuntimeError("Could not claim execution")
            try:
                value, is_error = agent._execute(
                    record["tool_name"], record["tool_input"], user_id,
                    state.get("model") or config.chat_model(), review, task.cancel if task else None)
                # Tool dispatchers can catch BalanceExhausted and return an
                # error string after partial effects. Treat that as unknown.
                if is_error:
                    billing.ensure_can_spend()
            except billing.BalanceExhausted as exc:
                # The tool was entered: even a billing failure cannot prove it
                # had no external effects. Keep the executing boundary unknown.
                warning = ("The action outcome is UNKNOWN; it may have completed and will not "
                           "be replayed. Reconcile its effects before retrying.")
                visible, pay_url = billing.split_pay_link(exc.block_message)
                raise billing.BalanceExhausted(
                    f"{warning}\n\n{visible}" +
                    (billing._PAY_LINK_MARKER + pay_url if pay_url else "")) from exc
            review.actions.append((record["tool_name"], record["tool_input"], "owner"))
            result = agent._tool_result(record["tool_call_id"], value, is_error)
        state["results"].append(result)
        messages = tasks._resume_messages(state)
        next_state = {"version": 1, "messages": messages, "phase": "model",
                      "results": [], "current": None, "enabled": [],
                      "owner_request": state.get("owner_request", ""),
                      "model": state.get("model")}
        if not interactions.finish(record["id"], result, continuation=next_state,
                                   task_checkpoint=next_state if task is not None else None):
            raise RuntimeError("Could not persist interaction result")
        # Some orchestration tools report billing exhaustion as a tool error;
        # retain the result before stopping this same meter for the next step.
        billing.ensure_can_spend()
        if task is not None:
            return messages
        return self._run_foreground_continuation(record["id"], source, owned)

    def _run_foreground_continuation(self, request_id, source, owned=None):
        from .. import agent, memory, tasks, tools
        record = interactions.claim_continuation(request_id)
        if record is None:
            return None
        if owned is not None:
            owned["continuation"] = True
        state = record["checkpoint"]
        tasks._validate_progress(state)
        user_id = record["user_id"]
        allowed, block_message = billing.gate(user_id)
        if not allowed:
            raise billing.BalanceExhausted(block_message)
        if state["phase"] == "final":
            return "".join(b["text"] for b in state["messages"][-1]["content"]
                           if b.get("type") == "text")
        if state["phase"] not in {"model", "tools", "executing"}:
            raise ValueError("Invalid foreground continuation phase")
        messages = (tasks._resume_messages(state) if state["phase"] in {"tools", "executing"}
                    else state["messages"])
        review = agent._review_context(state.get("owner_request", ""), user_id)
        specs = tools.agent_specs(user_id=user_id)
        system = agent._build_system(user_id, specs)
        token = tasks.INTERACTION_SOURCE.set(asdict(source))

        def checkpoint(snapshot):
            snapshot["owner_request"] = state.get("owner_request", "")
            snapshot["model"] = state.get("model")
            if not interactions.save_continuation(request_id, snapshot):
                raise RuntimeError("Foreground continuation is no longer claimed")

        try:
            reply = agent._run_tool_loop(
                system, messages, user_id=user_id,
                model=state.get("model") or config.chat_model(), review=review, specs=specs,
                checkpoint=checkpoint,
                on_interaction=self._interaction_callback(
                    source, record["run_id"], state.get("owner_request", ""),
                    model=state.get("model"), continuation_id=request_id))
            memory.log_episode("agent", reply, trust="trusted", user_id=user_id)
            return reply
        finally:
            tasks.INTERACTION_SOURCE.reset(token)

    def request_approval(self, user_id: str, tool_name: str, tool_input,
                         concern: str | None = None, origin: str | None = None,
                         grantable: bool = False) -> bool | str | None:
        """Ask a reachable user to approve a single high-impact action.

        Called synchronously from the agent's worker thread (the gate blocks
        there). Returns True (approved), security.GRANT (approved, and similar
        actions allowed for the rest of the task), False (denied or timed out),
        or None when the user has no interactive channel — so the caller falls
        back to the unattended policy. The answer arrives out-of-band (e.g. a
        Telegram button), bypassing the agent lock, and resolves the future.

        *grantable* offers the "allow for this task" answer, on adapters that
        support it; *origin* names who is asking (a subagent, a background task).
        """
        reason = self._approval_unavailable_reason(user_id)
        if reason is not None:
            # Audit WHY no buttons were shown so a misconfiguration (e.g. the user
            # not being in TELEGRAM_TRUSTED_USERS) is diagnosable, then fall back.
            security.audit("approval_skip", f"{user_id} {tool_name}: {reason}")
            return None

        source = self._presence._users[str(user_id)].last_source
        adapter = self._adapters[source.platform]
        approval_id = f"{int(time.time())}-{next(self._approval_seq)}"
        fut: concurrent.futures.Future = concurrent.futures.Future()
        self._pending_approvals[approval_id] = fut

        grantable = grantable and getattr(adapter, "supports_grants", False)
        prompt = _approval_prompt(tool_name, tool_input, concern, origin, grantable)
        security.audit("approval_request", f"{user_id} {tool_name} {tool_input}")
        extra = {"grantable": True} if grantable else {}
        cf = asyncio.run_coroutine_threadsafe(
            adapter.request_approval(source, approval_id, prompt, **extra), self._loop
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

    def can_request_approval(self, user_id: str) -> bool:
        """Whether *user_id* could be shown an approval prompt right now. Auto
        mode only answers on behalf of a user who could have answered."""
        return self._approval_unavailable_reason(user_id) is None

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

    def resolve_approval(self, approval_id: str, approved: bool | str) -> bool:
        """Settle a pending approval with True, False or security.GRANT. Called
        on the gateway loop by an adapter when the user answers. Returns False
        if the id is unknown or already settled (e.g. it had timed out)."""
        fut = self._pending_approvals.get(approval_id)
        if fut is None or fut.done():
            return False
        fut.set_result(approved)
        return True

    async def start_all(self) -> None:
        self._loop = asyncio.get_running_loop()
        # Only startup may recover execution boundaries; never replay an action.
        unknown = await self._loop.run_in_executor(None, interactions.recover)
        if unknown:
            security.audit("interaction_unknown", ", ".join(unknown))
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

        from .. import tasks
        await self._loop.run_in_executor(
            None, lambda: tasks.report_interrupted(all_users=True))
        await self._restore_interactions()

        if self._timer_task is None:
            from .. import timers
            await self._loop.run_in_executor(None, timers.recover)
            self._timer_stop.clear()
            self._timer_task = asyncio.create_task(self._timer_loop())

        if config.IDLE_CHECK_INTERVAL > 0 and self._idle_task is None:
            self._idle_task = asyncio.create_task(self._idle_loop())
            security.audit("idle_loop_start", f"every {config.IDLE_CHECK_INTERVAL}s")
        elif config.IDLE_CHECK_INTERVAL <= 0:
            security.audit("idle_loop_disabled", "DP_IDLE_CHECK<=0")

    async def stop_all(self) -> None:
        from .. import tasks
        self._timer_stop.set()
        if self._timer_task is not None:
            await self._timer_task
            self._timer_task = None
        tasks.cancel_all()
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

        # Usage still being priced (OpenRouter's generation data lags the
        # reply) must reach the billing service before the process exits.
        await asyncio.get_running_loop().run_in_executor(None, billing.flush, 30)

    async def _restore_interactions(self):
        await asyncio.get_running_loop().run_in_executor(None, interactions.expire)
        continuations = await asyncio.get_running_loop().run_in_executor(
            None, interactions.ready_continuations)
        for record in continuations:
            self._loop.create_task(self._resume_interaction(
                record["id"], SessionSource(**record["source"]), continuation=True))
        records = await asyncio.get_running_loop().run_in_executor(None, interactions.unsettled)
        for record in records:
            source = SessionSource(**record["source"])
            if record["status"] == "pending":
                self._present_interaction(record["id"], source, record)
            elif record["status"] in {"answered", "expired"}:
                # A durable owner decision is safe to claim again only before
                # the executing boundary; recover() never rewinds that boundary.
                self._loop.create_task(self._resume_interaction(record["id"], source))
            else:
                adapter = self._adapters.get(source.platform)
                if adapter and adapter.state == AdapterState.RUNNING:
                    await adapter.send(source, f"Interaction {record['id']}: outcome UNKNOWN. "
                                       "The action may have completed; it will not be replayed. "
                                       "Reconcile its effects before starting another action.")

    async def _expire_interactions(self):
        ids = await asyncio.get_running_loop().run_in_executor(None, interactions.expire)
        for request_id in ids:
            record = interactions.get(request_id)
            source = SessionSource(**record["source"])
            adapter = self._adapters.get(source.platform)
            if adapter and record["kind"] == "approval":
                try:
                    await adapter.expire_approval(source, "d-" + request_id)
                except Exception as exc:
                    security.audit("interaction_expire_error", str(exc))
            self._loop.create_task(self._resume_interaction(request_id, source))

    async def _timer_loop(self) -> None:
        from .. import timers
        while not self._timer_stop.is_set():
            try:
                await self._expire_interactions()
                await asyncio.get_running_loop().run_in_executor(
                    None, timers.fire_due, self.notify_user)
            except Exception as exc:
                security.audit("timer_loop_error", str(exc))
            try:
                await asyncio.wait_for(self._timer_stop.wait(), config.TIMER_CHECK_INTERVAL)
            except asyncio.TimeoutError:
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
