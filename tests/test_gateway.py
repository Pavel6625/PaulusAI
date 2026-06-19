import asyncio

import pytest

from paulus.gateway.base import AdapterState, BasePlatformAdapter, SessionSource
from paulus.gateway.presence import PresenceStore
from paulus.gateway.runner import GatewayRunner
from paulus.gateway.session_store import SessionStore


def test_session_source_key():
    assert SessionSource("telegram", "42", "7").key() == "telegram:42"
    assert SessionSource("telegram", "42", "7", thread_id="9").key() == "telegram:42:9"


def test_session_store_recreates_after_idle():
    store = SessionStore(idle_timeout=-1)  # always considered idle
    a = store.get_or_create("k")
    b = store.get_or_create("k")
    assert a is not b


def test_authorization_allowlist():
    runner = GatewayRunner()
    src = SessionSource("t", "1", "123")
    assert runner._is_user_authorized(src, set())        # empty allowlist = allow all
    assert runner._is_user_authorized(src, {"123"})
    assert not runner._is_user_authorized(src, {"999"})


def test_dispatch_outbound_without_adapter():
    runner = GatewayRunner()
    msg = runner.dispatch_outbound("telegram:1", "hi")
    assert "unavailable" in msg or "no active adapter" in msg


def test_presence_is_keyed_by_user_not_chat():
    store = PresenceStore()
    # Two users in the SAME chat must be tracked separately.
    store.touch(SessionSource("telegram", "group1", "alice"))
    store.touch(SessionSource("telegram", "group1", "bob"))
    assert {p.user_id for p in store.idle_users(-1, max_nudges=3)} == {"alice", "bob"}


def test_presence_activity_resets_nudge_budget():
    store = PresenceStore()
    src = SessionSource("telegram", "c", "u")
    store.touch(src)
    store.mark_nudged("u")
    store.mark_nudged("u")
    # Cap reached -> not idle-eligible.
    assert store.idle_users(-1, max_nudges=2) == []
    # The user speaks again -> budget clears.
    store.touch(src)
    assert [p.user_id for p in store.idle_users(-1, max_nudges=2)] == ["u"]


def test_presence_tracks_latest_reachable_source():
    store = PresenceStore()
    store.touch(SessionSource("telegram", "old", "u"))
    store.touch(SessionSource("telegram", "new", "u"))
    (p,) = store.idle_users(-1, max_nudges=3)
    assert p.last_source.chat_id == "new"


def test_idle_pass_nudges_then_respects_cap(monkeypatch):
    import paulus.agent as agent
    import paulus.config as config

    runner = GatewayRunner()
    sent: list[tuple[str, str]] = []

    class Dummy(BasePlatformAdapter):
        async def start(self): ...
        async def stop(self): ...
        async def send(self, source, text):
            sent.append((source.user_id, text))

    adapter = Dummy(runner)
    adapter._state = AdapterState.RUNNING
    runner.register("telegram", adapter)

    monkeypatch.setattr(agent, "proactive_check", lambda user_id=None: "checking in!")
    monkeypatch.setattr(config, "MAX_IDLE_MSG_SESSION", 1)

    runner._presence.touch(SessionSource("telegram", "c", "u"))

    # min_idle_s = -1 makes everyone idle; cap of 1 allows exactly one nudge.
    asyncio.run(runner._run_idle_pass(min_idle_s=-1))
    asyncio.run(runner._run_idle_pass(min_idle_s=-1))

    assert sent == [("u", "checking in!")]  # second pass blocked by per-user cap


def test_presence_persists_across_restart(tmp_path):
    path = tmp_path / "presence.json"
    store = PresenceStore(path)
    store.touch(SessionSource("telegram", "c", "u"))
    store.mark_nudged("u")

    restored = PresenceStore(path)            # simulate a gateway restart
    (p,) = restored.idle_users(-1, max_nudges=5)
    assert p.user_id == "u"
    assert p.nudges_sent == 1                 # budget survived the restart
    assert p.last_source.chat_id == "c"       # reachability survived too


def test_presence_restart_resets_idle_clock_but_keeps_budget(tmp_path):
    path = tmp_path / "presence.json"
    store = PresenceStore(path)
    store.touch(SessionSource("telegram", "c", "u"))
    store.mark_nudged("u")

    restored = PresenceStore(path)
    # Idle clock reset: not eligible until genuinely idle again (no startup burst).
    assert restored.idle_users(min_idle_s=60, max_nudges=5) == []
    # ...but the nudge budget carried over.
    (p,) = restored.idle_users(min_idle_s=-1, max_nudges=5)
    assert p.nudges_sent == 1


def test_presence_in_memory_writes_nothing(tmp_path):
    store = PresenceStore()                   # no path -> never touches disk
    store.touch(SessionSource("telegram", "c", "u"))
    assert list(tmp_path.iterdir()) == []


def test_parse_quiet_hours():
    from paulus.config import _parse_quiet
    assert _parse_quiet("23-7") == (23, 7)
    assert _parse_quiet("9-17") == (9, 17)
    assert _parse_quiet("25-30") == (1, 6)    # hours taken mod 24
    assert _parse_quiet("") == (None, None)
    assert _parse_quiet("nonsense") == (None, None)


def test_in_quiet_hours_wraps_midnight(monkeypatch):
    from datetime import datetime

    import paulus.config as config
    monkeypatch.setattr(config, "QUIET_START", 23)
    monkeypatch.setattr(config, "QUIET_END", 7)
    assert config.in_quiet_hours(datetime(2026, 1, 1, 2, 0))     # 02:00 -> quiet
    assert config.in_quiet_hours(datetime(2026, 1, 1, 23, 30))   # 23:30 -> quiet
    assert not config.in_quiet_hours(datetime(2026, 1, 1, 12, 0))  # noon -> awake


def test_in_quiet_hours_same_day_and_disabled(monkeypatch):
    from datetime import datetime

    import paulus.config as config
    monkeypatch.setattr(config, "QUIET_START", 9)
    monkeypatch.setattr(config, "QUIET_END", 17)
    assert config.in_quiet_hours(datetime(2026, 1, 1, 10, 0))
    assert not config.in_quiet_hours(datetime(2026, 1, 1, 20, 0))

    monkeypatch.setattr(config, "QUIET_START", None)   # unset -> never quiet
    monkeypatch.setattr(config, "QUIET_END", None)
    assert not config.in_quiet_hours(datetime(2026, 1, 1, 3, 0))


def test_idle_pass_skipped_during_quiet_hours(monkeypatch):
    import paulus.agent as agent
    import paulus.config as config

    runner = GatewayRunner()
    sent = []

    class Dummy(BasePlatformAdapter):
        async def start(self): ...
        async def stop(self): ...
        async def send(self, source, text):
            sent.append(text)

    adapter = Dummy(runner)
    adapter._state = AdapterState.RUNNING
    runner.register("telegram", adapter)

    monkeypatch.setattr(agent, "proactive_check", lambda user_id=None: "hi")
    monkeypatch.setattr(config, "in_quiet_hours", lambda now=None: True)
    runner._presence.touch(SessionSource("telegram", "c", "u"))
    asyncio.run(runner._run_idle_pass(min_idle_s=-1))

    assert sent == []  # quiet hours suppress all outreach


def test_idle_pass_silence_is_not_delivered(monkeypatch):
    import paulus.agent as agent

    runner = GatewayRunner()
    sent = []

    class Dummy(BasePlatformAdapter):
        async def start(self): ...
        async def stop(self): ...
        async def send(self, source, text):
            sent.append(text)

    adapter = Dummy(runner)
    adapter._state = AdapterState.RUNNING
    runner.register("telegram", adapter)

    monkeypatch.setattr(agent, "proactive_check", lambda user_id=None: "[SILENT]")
    runner._presence.touch(SessionSource("telegram", "c", "u"))
    asyncio.run(runner._run_idle_pass(min_idle_s=-1))

    assert sent == []  # silence sentinel swallowed, nothing delivered


def test_request_approval_returns_none_without_interactive_channel():
    runner = GatewayRunner()
    runner._loop = object()  # pretend the loop is up; the guards below should still fire

    # No presence record for this user -> not reachable.
    assert runner.request_approval("u", "run_command", {"command": "ls"}) is None

    # Reachable, but the adapter can't host an approval prompt.
    class NoApprovals(BasePlatformAdapter):
        supports_approvals = False
        async def start(self): ...
        async def stop(self): ...
        async def send(self, source, text): ...

    adapter = NoApprovals(runner)
    adapter._state = AdapterState.RUNNING
    runner.register("telegram", adapter)
    runner._presence.touch(SessionSource("telegram", "c", "u"))
    assert runner.request_approval("u", "run_command", {"command": "ls"}) is None


def test_request_approval_denied_for_untrusted_user():
    runner = GatewayRunner()
    runner._loop = object()

    class TrustedOnly(BasePlatformAdapter):
        supports_approvals = True
        def can_approve(self, user_id): return str(user_id) == "owner"
        async def start(self): ...
        async def stop(self): ...
        async def send(self, source, text): ...

    adapter = TrustedOnly(runner)
    adapter._state = AdapterState.RUNNING
    runner.register("telegram", adapter)
    # An untrusted user can chat, but their high-impact action gets no approval
    # channel -> falls back to the unattended policy (caller treats None as such).
    runner._presence.touch(SessionSource("telegram", "c", "guest"))
    assert runner.request_approval("guest", "run_command", {"command": "ls"}) is None


def test_request_approval_roundtrip_approve_and_deny():
    runner = GatewayRunner()
    answer = {"value": True}

    class Approver(BasePlatformAdapter):
        supports_approvals = True
        def can_approve(self, user_id): return True
        async def start(self): ...
        async def stop(self): ...
        async def send(self, source, text): ...
        async def request_approval(self, source, approval_id, prompt):
            # Simulate the user tapping a button as soon as the prompt is sent.
            runner.resolve_approval(approval_id, answer["value"])
        async def expire_approval(self, source, approval_id): ...

    adapter = Approver(runner)
    adapter._state = AdapterState.RUNNING
    runner.register("telegram", adapter)
    runner._presence.touch(SessionSource("telegram", "c", "u"))

    async def ask():
        runner._loop = asyncio.get_running_loop()
        # request_approval blocks on a future, so it must run off the loop thread.
        return await runner._loop.run_in_executor(
            None, lambda: runner.request_approval("u", "run_command", {"command": "ls"})
        )

    assert asyncio.run(ask()) is True
    answer["value"] = False
    assert asyncio.run(ask()) is False


def test_handle_command_mood_memory_skills(monkeypatch):
    import paulus.affect as affect
    import paulus.memory as memory
    import paulus.skills as skills

    runner = GatewayRunner()
    src = SessionSource("telegram", "c", "u")

    monkeypatch.setattr(affect, "describe", lambda: "calm, alert")
    monkeypatch.setattr(memory, "semantic_text", lambda user_id=None: f"facts<{user_id}>")
    monkeypatch.setattr(skills, "describe", lambda: "(no skills yet)")

    assert asyncio.run(runner.handle_command(src, "mood")) == "mood: calm, alert"
    assert asyncio.run(runner.handle_command(src, "memory")) == "facts<u>"  # per-user
    assert asyncio.run(runner.handle_command(src, "skills")) == "(no skills yet)"
    assert "Unknown" in asyncio.run(runner.handle_command(src, "bogus"))


def test_handle_command_sleep_consolidates_for_user(monkeypatch):
    import paulus.agent as agent

    runner = GatewayRunner()
    seen: dict = {}

    def fake_sleep(user_id=None):
        seen["uid"] = user_id
        return "Consolidated 2 fact(s)."

    monkeypatch.setattr(agent, "sleep", fake_sleep)
    out = asyncio.run(runner.handle_command(SessionSource("telegram", "c", "u"), "sleep"))

    assert out == "Consolidated 2 fact(s)."
    assert seen["uid"] == "u"               # scoped to the caller's memory


def test_resolve_unknown_approval_is_noop():
    runner = GatewayRunner()
    assert runner.resolve_approval("nope", True) is False


def test_circuit_breaker_opens_then_resumes():
    runner = GatewayRunner()

    class Dummy(BasePlatformAdapter):
        async def start(self): ...
        async def stop(self): ...
        async def send(self, source, text): ...

    a = Dummy(runner)
    a._state = AdapterState.RUNNING
    for _ in range(5):
        a._on_failure()
    assert a.state == AdapterState.CIRCUIT_OPEN
    a.resume()
    assert a.state == AdapterState.RUNNING


# --- Telegram Markdown rendering -------------------------------------------
# These need python-telegram-bot (the adapter imports it at module load);
# importorskip keeps the suite green in a core-only install.

def _telegram_module():
    return pytest.importorskip("paulus.gateway.platforms.telegram")


def test_split_under_hard_limit_and_hard_splits_long_lines():
    tg = _telegram_module()
    chunks = tg._split("x" * 9000, limit=3500)
    assert len(chunks) > 1
    assert all(len(c) <= tg._TELEGRAM_MSG_LIMIT for c in chunks)
    assert "".join(chunks) == "x" * 9000   # nothing lost in the hard split


def test_split_keeps_code_fence_intact():
    tg = _telegram_module()
    body = "\n".join(f"code line {i}" for i in range(10))
    text = f"intro paragraph\n```python\n{body}\n```\noutro paragraph"
    chunks = tg._split(text, limit=40)
    assert len(chunks) > 1                       # genuinely split
    # No chunk may contain an unbalanced fence -> the block stayed whole.
    assert all(c.count("```") % 2 == 0 for c in chunks)
    assert all(len(c) <= tg._TELEGRAM_MSG_LIMIT for c in chunks)


def test_markdownify_escapes_specials_outside_code():
    t = pytest.importorskip("telegramify_markdown")
    out = t.markdownify("**bold** then a-b.c\n```\nkeep-as.is\n```")
    assert "*bold*" in out                       # CommonMark bold -> MarkdownV2
    assert "\\." in out and "\\-" in out          # specials escaped in prose


def _markdown_adapter(monkeypatch, parse_mode="MarkdownV2"):
    tg = _telegram_module()
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "x")
    monkeypatch.setenv("TELEGRAM_PARSE_MODE", parse_mode)
    return tg, tg.TelegramAdapter(runner=None)


def _wire_fake_bot(tg, adapter, fail_on_markdown):
    calls: list[tuple[str, str | None]] = []

    class FakeBot:
        async def send_message(self, chat_id, text, parse_mode=None, **kw):
            calls.append((text, parse_mode))
            if fail_on_markdown and parse_mode == "MarkdownV2":
                from telegram.error import BadRequest
                raise BadRequest("can't parse entities")

    adapter._app = type("App", (), {"bot": FakeBot()})()
    return calls


def test_send_falls_back_to_plain_on_bad_request(monkeypatch):
    pytest.importorskip("telegramify_markdown")
    tg, adapter = _markdown_adapter(monkeypatch)
    calls = _wire_fake_bot(tg, adapter, fail_on_markdown=True)

    audits: list[tuple] = []
    monkeypatch.setattr(tg.security, "audit", lambda *a, **k: audits.append(a))

    asyncio.run(adapter.send(SessionSource("telegram", "c", "u"), "hello **world**"))

    # MarkdownV2 attempted first (rejected), then the ORIGINAL text resent plain.
    assert [pm for _, pm in calls] == ["MarkdownV2", None]
    assert calls[1][0] == "hello **world**"
    assert any(a[0] == "telegram_markdown_fallback" for a in audits)


def test_send_plain_mode_skips_markdown(monkeypatch):
    tg, adapter = _markdown_adapter(monkeypatch, parse_mode="plain")
    assert adapter._parse_mode is None
    calls = _wire_fake_bot(tg, adapter, fail_on_markdown=False)

    asyncio.run(adapter.send(SessionSource("telegram", "c", "u"), "raw **text**"))

    assert calls == [("raw **text**", None)]      # one plain send, untouched


# --- Streaming --------------------------------------------------------------

def test_llm_stream_accumulates_text_and_fires_deltas(monkeypatch):
    from types import SimpleNamespace as NS

    import paulus.llm as llm

    def chunk(content=None, tool=None, finish=None):
        return NS(choices=[NS(delta=NS(content=content, tool_calls=tool), finish_reason=finish)])

    fake = [chunk(content="Hel"), chunk(content="lo"), chunk(finish="stop")]
    monkeypatch.setattr(llm.litellm, "completion", lambda **kw: iter(fake))

    seen: list[str] = []
    resp = llm.stream("sys", [{"role": "user", "content": "hi"}], on_delta=seen.append)

    assert "".join(seen) == "Hello"
    assert resp.stop_reason == "end_turn"
    assert resp.content[0].text == "Hello"


def test_llm_stream_reassembles_tool_calls(monkeypatch):
    from types import SimpleNamespace as NS

    import paulus.llm as llm

    def chunk(tool=None, finish=None):
        return NS(choices=[NS(delta=NS(content=None, tool_calls=tool), finish_reason=finish)])

    # A tool call whose name + arguments are split across two chunks.
    part1 = NS(index=0, id="call_1", function=NS(name="recall", arguments='{"q":'))
    part2 = NS(index=0, id=None, function=NS(name=None, arguments='"x"}'))
    fake = [chunk(tool=[part1]), chunk(tool=[part2]), chunk(finish="tool_calls")]
    monkeypatch.setattr(llm.litellm, "completion", lambda **kw: iter(fake))

    resp = llm.stream("sys", [{"role": "user", "content": "hi"}])

    assert resp.stop_reason == "tool_use"
    (block,) = resp.content
    assert block.type == "tool_use"
    assert block.name == "recall" and block.input == {"q": "x"}


def _approval_adapter(monkeypatch, allowed="", trusted="7"):
    tg = _telegram_module()
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "x")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", allowed)
    monkeypatch.setenv("TELEGRAM_TRUSTED_USERS", trusted)
    runner = GatewayRunner()
    adapter = tg.TelegramAdapter(runner)
    adapter._state = AdapterState.RUNNING
    runner.register("telegram", adapter)

    sent: list[dict] = []

    class FakeBot:
        async def send_message(self, chat_id, text, **kw):
            sent.append({"text": text, **kw})
            return type("Msg", (), {"message_id": 555})()

    adapter._app = type("App", (), {"bot": FakeBot()})()
    return tg, runner, adapter, sent


class _FakeQuery:
    def __init__(self, data, text="prompt"):
        self.data = data
        self.message = type("M", (), {"text": text})()
        self.answered: list = []
        self.edited: list[str] = []

    async def answer(self, text=None, show_alert=False):
        self.answered.append((text, show_alert))

    async def edit_message_text(self, text):
        self.edited.append(text)


def _callback_update(query, user_id="7"):
    return type("U", (), {
        "callback_query": query,
        "effective_user": type("Usr", (), {"id": user_id})(),
    })()


def test_request_approval_sends_buttons_and_callback_settles(monkeypatch):
    tg, runner, adapter, sent = _approval_adapter(monkeypatch)
    src = SessionSource("telegram", "c", "7")

    asyncio.run(adapter.request_approval(src, "abc", "do the thing?"))
    assert sent and sent[0]["text"] == "do the thing?"
    assert sent[0]["reply_markup"] is not None
    assert adapter._approval_msgs["abc"] == ("c", 555)

    resolved: list = []
    monkeypatch.setattr(runner, "resolve_approval",
                        lambda aid, ok: resolved.append((aid, ok)) or True)

    query = _FakeQuery("dpok:abc")
    asyncio.run(adapter._on_callback(_callback_update(query), None))

    assert resolved == [("abc", True)]
    assert "abc" not in adapter._approval_msgs        # prompt cleared
    assert any("Approved" in e for e in query.edited)  # verdict shown


def test_trusted_list_independent_of_chat_allowlist(monkeypatch):
    tg = _telegram_module()
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "x")

    # Open chat (no allowlist) + an explicit trusted approver.
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "")
    monkeypatch.setenv("TELEGRAM_TRUSTED_USERS", "owner")
    a = tg.TelegramAdapter(runner=None)
    assert a._allowed == set()              # everyone may chat
    assert a.can_approve("owner") and not a.can_approve("guest")

    # Trusted unset -> defaults to the chat allowlist (single-owner shorthand).
    monkeypatch.delenv("TELEGRAM_TRUSTED_USERS", raising=False)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "owner")
    b = tg.TelegramAdapter(runner=None)
    assert b.can_approve("owner") and not b.can_approve("guest")

    # Trusted set but empty -> nobody can approve (fail safe).
    monkeypatch.setenv("TELEGRAM_TRUSTED_USERS", "")
    c = tg.TelegramAdapter(runner=None)
    assert not c.can_approve("owner")


def test_callback_from_unlisted_user_is_rejected(monkeypatch):
    tg, runner, adapter, sent = _approval_adapter(monkeypatch, allowed="7")
    calls: list = []
    monkeypatch.setattr(runner, "resolve_approval",
                        lambda aid, ok: calls.append((aid, ok)) or True)

    query = _FakeQuery("dpok:abc")
    asyncio.run(adapter._on_callback(_callback_update(query, user_id="999"), None))

    assert calls == []                                 # never settled
    assert query.answered and query.answered[0][1]     # show_alert "Unauthorized"


def _command_update(text, user_id="7", chat_id="c"):
    """A minimal Update for a /command message, with a recording reply_text."""
    replies: list[str] = []

    async def reply_text(t):
        replies.append(t)

    msg = type("Msg", (), {
        "text": text,
        "is_topic_message": False,
        "message_thread_id": None,
        "reply_text": staticmethod(reply_text),
    })()
    update = type("U", (), {
        "message": msg,
        "effective_user": type("Usr", (), {"id": user_id})(),
        "effective_chat": type("Chat", (), {"id": chat_id})(),
    })()
    return update, replies


def test_telegram_command_routes_to_runner(monkeypatch):
    tg, runner, adapter, events = _streaming_runner(monkeypatch)
    captured: dict = {}

    async def fake_handle(source, command):
        captured["cmd"] = command
        captured["uid"] = source.user_id
        return "result text"

    monkeypatch.setattr(runner, "handle_command", fake_handle)

    # "@botname" suffix (added by Telegram in groups) is stripped to the bare name.
    update, _ = _command_update("/skills@PaulusBot", user_id="7")
    asyncio.run(adapter._on_command(update, None))

    assert captured == {"cmd": "skills", "uid": "7"}
    assert events[-1] == ("send", "result text", {})   # delivered via send()


def test_telegram_command_rejects_unauthorized(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "7")
    tg, runner, adapter, events = _streaming_runner(monkeypatch)

    called = {"n": 0}

    async def fake_handle(source, command):
        called["n"] += 1
        return "should not happen"

    monkeypatch.setattr(runner, "handle_command", fake_handle)

    update, replies = _command_update("/mood", user_id="999")
    asyncio.run(adapter._on_command(update, None))

    assert called["n"] == 0                 # command never dispatched
    assert replies == ["Unauthorized."]


def test_revealable_gates_silence_sentinels():
    tg = _telegram_module()
    for hidden in ("", "[", "[SIL", "[SILENT]", "NO_REPLY"):
        assert not tg._revealable(hidden)
    for shown in ("Hello", "[note] hi"):
        assert tg._revealable(shown)


class _FakeMsg:
    message_id = 99


def _streaming_runner(monkeypatch, parse_mode="plain"):
    tg = _telegram_module()
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "x")
    monkeypatch.setenv("TELEGRAM_PARSE_MODE", parse_mode)
    monkeypatch.setenv("TELEGRAM_STREAMING", "1")

    runner = GatewayRunner()
    adapter = tg.TelegramAdapter(runner)
    adapter._state = AdapterState.RUNNING
    runner.register("telegram", adapter)

    events: list[tuple[str, str, dict]] = []

    class FakeBot:
        async def send_message(self, chat_id, text, **kw):
            events.append(("send", text, kw))
            return _FakeMsg()

        async def edit_message_text(self, chat_id, message_id, text, **kw):
            events.append(("edit", text, kw))

        async def delete_message(self, chat_id, message_id):
            events.append(("delete", "", {}))

        async def send_chat_action(self, **kw):
            pass

    adapter._app = type("App", (), {"bot": FakeBot()})()
    return tg, runner, adapter, events


def test_streaming_inbound_creates_and_finalizes(monkeypatch):
    import paulus.agent as agent
    tg, runner, adapter, events = _streaming_runner(monkeypatch)

    def fake_respond(text, user_id=None, on_delta=None):
        for piece in ("Hello ", "world"):
            on_delta(piece)
        return "Hello world"

    monkeypatch.setattr(agent, "respond", fake_respond)
    asyncio.run(runner.handle_inbound(SessionSource("telegram", "c", "u"), "hi"))

    # A message is created and the final visible text is the full reply.
    assert any(kind == "send" for kind, _, _ in events)
    assert events[-1][1] == "Hello world"


def test_streaming_silent_reply_shows_nothing(monkeypatch):
    import paulus.agent as agent
    tg, runner, adapter, events = _streaming_runner(monkeypatch)

    def fake_respond(text, user_id=None, on_delta=None):
        on_delta("[SILENT]")          # sentinel never gets revealed
        return "[SILENT]"

    monkeypatch.setattr(agent, "respond", fake_respond)
    asyncio.run(runner.handle_inbound(SessionSource("telegram", "c", "u"), "hi"))

    assert events == []               # no placeholder, nothing to delete


def test_streaming_empty_reply_sends_nothing(monkeypatch):
    import paulus.agent as agent
    tg, runner, adapter, events = _streaming_runner(monkeypatch)

    # Model streams nothing and returns empty text (e.g. a denied high-impact
    # action with no follow-up). Must not try to send an empty Telegram message.
    monkeypatch.setattr(agent, "respond", lambda text, user_id=None, on_delta=None: "")
    asyncio.run(runner.handle_inbound(SessionSource("telegram", "c", "u"), "hi"))

    assert events == []        # nothing sent, no "Message text is empty" crash


def test_streaming_keeps_preamble_when_final_text_empty(monkeypatch):
    import paulus.agent as agent
    tg, runner, adapter, events = _streaming_runner(monkeypatch)

    def fake_respond(text, user_id=None, on_delta=None):
        on_delta("Working on it…")   # visible preamble streamed
        return ""                    # but the final turn returns no text
    monkeypatch.setattr(agent, "respond", fake_respond)
    asyncio.run(runner.handle_inbound(SessionSource("telegram", "c", "u"), "hi"))

    # The streamed text is preserved (not discarded as if silent), never deleted.
    assert any(kind == "send" for kind, _, _ in events)
    assert all(kind != "delete" for kind, _, _ in events)
    assert events[-1][1] == "Working on it…"


def test_streaming_finalizes_with_markdown(monkeypatch):
    pytest.importorskip("telegramify_markdown")
    import paulus.agent as agent
    tg, runner, adapter, events = _streaming_runner(monkeypatch, parse_mode="MarkdownV2")

    def fake_respond(text, user_id=None, on_delta=None):
        on_delta("**done**")
        return "**done**"

    monkeypatch.setattr(agent, "respond", fake_respond)
    asyncio.run(runner.handle_inbound(SessionSource("telegram", "c", "u"), "hi"))

    final_kind, final_text, final_kw = events[-1]
    assert final_kw.get("parse_mode") == "MarkdownV2"
    assert "*done*" in final_text      # CommonMark bold -> MarkdownV2
