import asyncio

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
