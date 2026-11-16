"""Persistent schedules, crash policy, and owner-scoped gateway execution."""
import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from paulus import agent, automode, billing, config, llm, security, tasks, timers, tools
from paulus.gateway.base import AdapterState, BasePlatformAdapter, SessionSource
from paulus.gateway.runner import GatewayRunner

NOW = datetime(2030, 1, 1, tzinfo=timezone.utc).timestamp()


class Adapter(BasePlatformAdapter):
    async def start(self):
        self._state = AdapterState.RUNNING

    async def stop(self):
        self._state = AdapterState.STOPPED

    async def send(self, source, text):
        self.sent.append((source.user_id, text))


@pytest.fixture
def gateway(monkeypatch):
    monkeypatch.setattr(timers.time, "time", lambda: NOW)
    runner = GatewayRunner()
    runner._loop = object()
    adapter = Adapter(runner)
    adapter.sent = []
    adapter._state = AdapterState.RUNNING
    runner.register("test", adapter)
    runner._presence.touch(SessionSource("test", "c1", "u1"))
    return runner, adapter


def create(**kwargs):
    payload = {"title": "check", "payload": "hello", "kind": "reminder",
               "at": "2030-01-01T01:00:00Z", **kwargs}
    return timers.create(payload, "u1", "Remind me later")


def due(monkeypatch):
    monkeypatch.setattr(timers.time, "time", lambda: NOW + 3 * 86400)


def test_persisted_reminder_fires_once_after_downtime(gateway, monkeypatch):
    assert create()[1] is False
    timers.recover()
    due(monkeypatch)
    delivered = []
    timers.fire_due(lambda uid, text: delivered.append((uid, text)))
    timers.recover()
    timers.fire_due(lambda *a: pytest.fail("duplicate"))
    assert delivered == [("u1", "⏰ check\n\nhello")]
    assert "[done]" in timers.listing("u1")
    assert timers.listing("u2") == "No timers."


@pytest.mark.parametrize("overrides", [
    {"at": "2030-01-01T01:00:00"}, {"at": "2020-01-01T00:00:00Z"},
    {"at": "nonsense"}, {"at": "", "cron": "* * * * * *"},
    {"at": "", "cron": "0 0 30 2 *"}, {"cron": "0 9 * * *"},
    {"timezone": "not/a-zone"}, {"kind": "shell"}, {"payload": ""},
])
def test_invalid_schedules_are_not_saved(gateway, overrides):
    assert create(**overrides)[1] is True
    assert timers.listing("u1") == "No timers."


def test_cron_coalesces_missed_runs_and_uses_timezone(gateway, monkeypatch):
    assert create(at="", cron="0 9 * * *", timezone="Europe/Moscow")[1] is False
    assert "2030-01-01T06:00:00+00:00" in timers.listing("u1")
    due(monkeypatch)
    got = []
    timers.fire_due(lambda *a: got.append(a))
    assert len(got) == 1
    assert "2030-01-04T06:00:00+00:00" in timers.listing("u1")
    assert "[scheduled]" in timers.listing("u1")


@pytest.mark.parametrize("cron", ["", "0 9 * * *"])
def test_claimed_occurrence_is_not_replayed_after_restart(gateway, monkeypatch, cron):
    create(at="" if cron else "2030-01-01T01:00:00Z", cron=cron)
    due(monkeypatch)
    claimed = timers._claim(timers.time.time())
    assert claimed is not None
    timers.recover()
    timers.fire_due(lambda *a: pytest.fail("replayed uncertain occurrence"))
    assert "outcome unknown" in timers.listing("u1")
    assert ("[scheduled]" if cron else "[interrupted]") in timers.listing("u1")


def test_cancel_is_owner_scoped_and_persisted(gateway, monkeypatch):
    create()
    assert timers.cancel(1, "u2")[1] is True
    assert timers.cancel(1, "u1")[1] is False
    timers.recover()
    due(monkeypatch)
    timers.fire_due(lambda *a: pytest.fail("cancelled timer fired"))
    assert "[cancelled]" in timers.listing("u1")


def test_occurrence_can_only_be_claimed_once(gateway, monkeypatch):
    create()
    due(monkeypatch)
    assert timers._claim(timers.time.time()) is not None
    assert timers._claim(timers.time.time()) is None


@pytest.mark.parametrize("after, expected", [
    ("2030-03-30T12:00:00+00:00", "2030-03-31T07:00:00+00:00"),
    ("2030-10-26T12:00:00+00:00", "2030-10-27T08:00:00+00:00"),
])
def test_cron_preserves_local_hour_across_dst(after, expected):
    stamp = timers._next("0 9 * * *", "Europe/Berlin",
                         datetime.fromisoformat(after).timestamp())
    assert datetime.fromtimestamp(stamp, timezone.utc).isoformat() == expected


def test_recurring_timer_can_be_cancelled_after_an_occurrence(gateway, monkeypatch):
    create(at="", cron="0 9 * * *")
    due(monkeypatch)
    timers.fire_due(lambda *a: True)
    assert timers.cancel(1, "u1")[1] is False
    monkeypatch.setattr(timers.time, "time", lambda: NOW + 10 * 86400)
    timers.fire_due(lambda *a: pytest.fail("cancelled recurrence fired"))


def test_active_timer_limit(gateway, monkeypatch):
    monkeypatch.setattr(config, "MAX_TIMERS_PER_USER", 1)
    assert create()[1] is False
    assert create()[1] is True
    timers.cancel(1, "u1")
    assert create()[1] is False


def test_delivery_failure_is_recorded_not_replayed(gateway, monkeypatch):
    create()
    due(monkeypatch)
    timers.fire_due(lambda *a: False)
    assert "[failed]" in timers.listing("u1")
    timers.fire_due(lambda *a: pytest.fail("retried uncertain send"))


def test_agent_run_uses_billing_and_fresh_owner_review(gateway, monkeypatch):
    create(kind="agent", payload="research flights")
    due(monkeypatch)
    monkeypatch.setattr(billing, "gate", lambda uid: (True, ""))
    seen = []

    def start(uid, title, payload, **kwargs):
        seen.append((uid, payload, kwargs["review"]))
        return SimpleNamespace(id=12), None

    monkeypatch.setattr(tasks, "start", start)
    timers.fire_due(lambda *a: True)
    assert seen[0][0:2] == ("u1", "research flights")
    assert seen[0][2].request == "Remind me later"
    assert not seen[0][2].grants and not seen[0][2].actions
    assert "task #12" in timers.listing("u1")


def test_agent_run_cannot_bypass_balance_gate(gateway, monkeypatch):
    create(kind="agent")
    due(monkeypatch)
    monkeypatch.setattr(billing, "gate", lambda uid: (False, "top up"))
    monkeypatch.setattr(tasks, "start", lambda *a, **k: pytest.fail("unpaid task"))
    got = []
    timers.fire_due(lambda uid, text: got.append(text))
    assert got == ["top up"]
    assert "insufficient balance" in timers.listing("u1")


def test_timer_creation_goes_through_safety_gate(gateway, monkeypatch):
    responses = iter([
        llm._Response(stop_reason="tool_use", content=[llm._ToolUseBlock(
            id="t", name="create_timer", input={"title": "x", "payload": "y",
                                               "at": "2030-01-01T01:00:00Z"})]),
        llm._Response(content=[llm._TextBlock(text="not scheduled")]),
    ])
    monkeypatch.setattr(llm, "complete", lambda *a, **k: next(responses))
    asked = []
    monkeypatch.setattr(security, "clearance", lambda name, *a, **k: asked.append(name))
    agent._run_tool_loop("sys", [], user_id="u1", specs=tools.agent_specs(user_id="u1"),
                         review=automode.ReviewContext(request="remind me"))
    assert asked == ["create_timer"]
    assert timers.listing("u1") == "No timers."


def test_background_and_unrequested_loops_cannot_create_timers(gateway):
    assert not {s['name'] for s in tools.agent_specs(tasks=False, user_id="u1")} & tools.TIMER_TOOLS
    assert agent._execute("create_timer", {}, "u1", None, None, None)[1] is True
    assert timers.create({}, None, "request")[1] is True


def test_gateway_services_timers_without_idle_loop_and_stops_cleanly(gateway, monkeypatch):
    runner, adapter = gateway
    create()
    due(monkeypatch)
    monkeypatch.setattr(config, "IDLE_CHECK_INTERVAL", 0)
    monkeypatch.setattr(billing, "flush", lambda *a: None)

    async def run():
        await runner.start_all()
        for _ in range(100):
            if adapter.sent:
                break
            await asyncio.sleep(0.01)
        await runner.stop_all()
        assert runner._timer_task is None

    asyncio.run(run())
    assert len(adapter.sent) == 1 and "hello" in adapter.sent[0][1]
