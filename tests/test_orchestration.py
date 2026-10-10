"""Subagents (subagents.py), background tasks (tasks.py), and the tool-loop
knobs they're built on (agent._run_tool_loop)."""
import asyncio
import threading

import pytest

from paulus import agent, automode, config, llm, memory, security, subagents, tasks, tools
from paulus.gateway.base import AdapterState, BasePlatformAdapter, SessionSource
from paulus.gateway.runner import GatewayRunner, _approval_prompt


def _text(text):
    return llm._Response(content=[llm._TextBlock(text=text)])


def _call(name, tool_input, call_id="t1"):
    return llm._Response(stop_reason="tool_use", content=[
        llm._ToolUseBlock(id=call_id, name=name, input=tool_input)])


def _results(messages):
    """Every tool_result the loop sent back, in order."""
    return [b for m in messages if m["role"] == "user" and isinstance(m["content"], list)
            for b in m["content"] if b.get("type") == "tool_result"]


def _ctx():
    return automode.ReviewContext(request="look into it")


# --- the loop ----------------------------------------------------------------

def test_loop_refuses_a_tool_it_did_not_offer(monkeypatch):
    replies = iter([_call("send_message", {"to": "x", "body": "hi"}), _text("ok")])
    monkeypatch.setattr(llm, "complete", lambda *a, **k: next(replies))
    monkeypatch.setattr(security, "clearance", lambda *a, **k: pytest.fail("gated"))
    monkeypatch.setattr(tools, "execute", lambda *a, **k: pytest.fail("ran"))
    messages = [{"role": "user", "content": "go"}]

    agent._run_tool_loop("sys", messages, specs=tools.pick({"web_search"}))

    assert "isn't available" in _results(messages)[0]["content"]


def test_step_budget_asks_for_the_report(monkeypatch):
    def model(system, messages, tools=None, model=None):
        done = any(r["content"] == agent._BUDGET_NOTE for r in _results(messages))
        return _text("report") if done else _call("recall", {"query": "q"})

    monkeypatch.setattr(llm, "complete", model)
    ran = []
    monkeypatch.setattr(tools, "execute", lambda name, *a, **k: ran.append(name) or ("x", False))

    out = agent._run_tool_loop("sys", [{"role": "user", "content": "go"}], max_steps=2)

    assert out == "report" and ran == ["recall", "recall"]


def test_step_budget_stops_a_model_that_ignores_it(monkeypatch):
    monkeypatch.setattr(llm, "complete", lambda *a, **k: _call("recall", {"query": "q"}))
    monkeypatch.setattr(tools, "execute", lambda *a, **k: ("x", False))
    out = agent._run_tool_loop("sys", [{"role": "user", "content": "go"}], max_steps=1)
    assert "step budget" in out


def test_cancel_stops_the_loop(monkeypatch):
    monkeypatch.setattr(llm, "complete", lambda *a, **k: pytest.fail("called the model"))
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(agent.Cancelled):
        agent._run_tool_loop("sys", [{"role": "user", "content": "go"}], cancel=cancel)


def test_the_conversation_offers_orchestration_but_idle_nudges_do_not(monkeypatch):
    offered = []

    def model(system, messages, tools=None, model=None):
        offered.append({s["name"] for s in tools})
        return _text("hi")

    monkeypatch.setattr(llm, "complete", model)
    agent.respond("hello", user_id="u1")
    agent.proactive_check(user_id="u1")

    assert {"delegate", "start_task", "task_status"} <= offered[0]
    assert not offered[1] & tools.ORCHESTRATION_TOOLS


def test_orchestration_can_be_turned_off(monkeypatch):
    monkeypatch.setattr(config, "SUBAGENTS", False)
    monkeypatch.setattr(config, "TASKS", False)
    assert not {s["name"] for s in tools.agent_specs()} & tools.ORCHESTRATION_TOOLS


# --- subagents ----------------------------------------------------------------

def test_delegate_runs_subagents_in_parallel(monkeypatch):
    # Each subagent waits at the barrier on its first call: run one after the
    # other, the first would time out waiting for the second.
    barrier = threading.Barrier(2, timeout=5)

    def model(system, messages, tools=None, model=None):
        assert "researcher subagent" in system
        task = messages[0]["content"]
        if len(messages) == 1:
            barrier.wait()
            return _call("web_search", {"query": task})
        return _text(f"answer to {task}")

    monkeypatch.setattr(llm, "complete", model)
    monkeypatch.setattr(tools, "execute", lambda *a, **k: ("results", False))

    out, is_error = subagents.delegate(
        {"tasks": [{"agent": "researcher", "task": "A"},
                   {"agent": "researcher", "task": "B"}]},
        user_id="u1", review=_ctx())

    assert not is_error
    assert "answer to A" in out and "answer to B" in out
    # Reports are built from the web: they come back as untrusted data.
    assert out.count("<untrusted_data") == 2
    assert 'source="subagent 1 (researcher)"' in out


@pytest.mark.parametrize("tool_input", [
    {}, {"tasks": []},
    {"tasks": [{"agent": "hacker", "task": "x"}]},
    {"tasks": [{"agent": "researcher", "task": ""}]},
    {"tasks": [{"agent": "researcher", "task": "x"}] * 9},
])
def test_delegate_rejects_bad_input(monkeypatch, tool_input):
    monkeypatch.setattr(llm, "complete", lambda *a, **k: pytest.fail("ran"))
    _out, is_error = subagents.delegate(tool_input, user_id="u1", review=_ctx())
    assert is_error


def test_subagent_actions_go_through_the_shared_gate(monkeypatch):
    replies = iter([_call("run_command", {"command": "pytest"}), _text("all green")])
    monkeypatch.setattr(llm, "complete", lambda *a, **k: next(replies))
    monkeypatch.setattr(tools, "execute", lambda *a, **k: ("1 passed", False))
    asked = []
    monkeypatch.setattr(security, "clearance",
                        lambda name, inp, user_id=None, context=None, origin=None:
                        asked.append((context, origin)) or "owner")
    review = _ctx()

    out, _ = subagents.delegate({"tasks": [{"agent": "worker", "task": "run tests"}]},
                                user_id="u1", review=review)

    assert "all green" in out
    assert asked == [(review, "subagent 1 (worker)")]
    assert review.actions == [("run_command", {"command": "pytest"}, "owner")]


def test_researchers_cannot_write_or_send():
    names = {s["name"] for s in tools.pick(subagents.PROFILES["researcher"]["tools"])}
    assert not names & security.HIGH_IMPACT_TOOLS
    for profile in subagents.PROFILES.values():
        assert not profile["tools"] & {"remember", "save_skill", "send_message",
                                       "send_document", "send_email_agentmail",
                                       *tools.ORCHESTRATION_TOOLS}


# --- background tasks -----------------------------------------------------------

@pytest.fixture
def inbox(monkeypatch):
    """Capture what finished tasks announce; wait() blocks for the next one."""
    got, arrived = [], threading.Event()

    def notify(user_id, text):
        got.append((user_id, text))
        arrived.set()

    def wait():
        assert arrived.wait(5), "no task finished"
        arrived.clear()
        return got[-1]

    monkeypatch.setattr(tasks, "_notify", notify)
    monkeypatch.setattr(tasks, "_tasks", {})
    return wait


def test_task_runs_in_the_background_and_reports(monkeypatch, inbox):
    seen = {}

    def model(system, messages, tools=None, model=None):
        seen["background"] = security.BACKGROUND.get()
        seen["tools"] = {s["name"] for s in tools}
        seen["system"] = system
        return _text("found 3 flights")

    monkeypatch.setattr(llm, "complete", model)
    task, error = tasks.start("u1", "flights", "find flights", review=_ctx())

    assert error is None
    user_id, message = inbox()
    assert user_id == "u1"
    assert f"#{task.id} done" in message and "found 3 flights" in message
    assert seen["background"] is True                  # no console prompts
    assert "start_task" not in seen["tools"] and "delegate" in seen["tools"]
    assert "BACKGROUND TASK" in seen["system"]
    # The next turn knows how it went.
    assert "found 3 flights" in memory.recent_episodes(user_id="u1")[-1]["text"]
    assert "[done] flights" in tasks.status("u1")


def test_start_task_tool_hands_off_from_a_turn(monkeypatch, inbox):
    def model(system, messages, tools=None, model=None):
        if "BACKGROUND TASK" in system:
            return _text("report")
        if len(messages) and messages[-1]["role"] == "user" \
                and isinstance(messages[-1]["content"], list):
            return _text("On it — I'll message you.")
        return _call("start_task", {"title": "research", "instructions": "dig"})

    monkeypatch.setattr(llm, "complete", model)
    reply = agent.respond("research this for me", user_id="u1")

    assert reply == "On it — I'll message you."
    assert "report" in inbox()[1]


def test_task_failure_is_reported(monkeypatch, inbox):
    def model(*a, **k):
        raise RuntimeError("provider down")

    monkeypatch.setattr(llm, "complete", model)
    tasks.start("u1", "x", "y", review=_ctx())
    message = inbox()[1]
    assert "failed" in message and "provider down" in message


def test_cancel_stops_a_task_at_its_next_step(monkeypatch, inbox):
    entered, release = threading.Event(), threading.Event()

    def model(*a, **k):
        entered.set()
        release.wait(5)
        return _call("recall", {"query": "q"})

    monkeypatch.setattr(llm, "complete", model)
    monkeypatch.setattr(tools, "execute", lambda *a, **k: pytest.fail("ran after cancel"))
    task, _ = tasks.start("u1", "slow", "y", review=_ctx())
    assert entered.wait(5)

    assert "No background task" in tasks.cancel("someone-else", task.id)
    assert "Cancelling" in tasks.cancel("u1", task.id)
    release.set()

    assert "cancelled" in inbox()[1]
    assert task.status == "cancelled"


def test_tasks_per_user_are_capped(monkeypatch, inbox):
    release = threading.Event()
    monkeypatch.setattr(config, "MAX_TASKS_PER_USER", 1)
    monkeypatch.setattr(llm, "complete", lambda *a, **k: release.wait(5) and _text("ok"))

    first, _ = tasks.start("u1", "a", "a", review=_ctx())
    second, error = tasks.start("u1", "b", "b", review=_ctx())
    other, other_error = tasks.start("u2", "c", "c", review=_ctx())

    assert first and second is None and "limit" in error
    assert other is not None and other_error is None
    release.set()
    inbox()
    inbox()


def test_a_task_is_never_started_for_an_idle_nudge():
    out, is_error = tasks.start_from_tool({"title": "t", "instructions": "i"},
                                          user_id="u1", review=None)
    assert is_error


def test_bg_starts_a_task_from_the_owners_words(monkeypatch, inbox):
    reviewed = []

    def model(system, messages, tools=None, model=None):
        return _text("done it")

    monkeypatch.setattr(llm, "complete", model)
    real_start = tasks.start
    monkeypatch.setattr(tasks, "start",
                        lambda *a, **k: reviewed.append(k["review"]) or real_start(*a, **k))

    reply = tasks.start_from_owner("  summarise my inbox  ", user_id="u1")

    assert reply.startswith("Started background task #")
    assert reviewed[0].request == "summarise my inbox"   # the owner's own words
    assert "done it" in inbox()[1]


# --- gateway ------------------------------------------------------------------

def test_approval_prompt_names_the_actor_and_the_grant():
    prompt = _approval_prompt("run_command", {"command": "pytest -q"},
                              origin="background task #2 (tests)", grantable=True)
    assert "From: background task #2 (tests)" in prompt
    assert "also allows running `pytest`" in prompt


class _Adapter(BasePlatformAdapter):
    supports_approvals = True

    def __init__(self, runner):
        super().__init__(runner)
        self.sent, self.prompts = [], []

    def can_approve(self, user_id):
        return True

    async def start(self): ...
    async def stop(self): ...

    async def send(self, source, text):
        self.sent.append((source.chat_id, text))

    async def request_approval(self, source, approval_id, prompt):
        # No grantable parameter: this adapter has no "allow for this task" UI.
        self.prompts.append(prompt)
        self._runner.resolve_approval(approval_id, True)

    async def expire_approval(self, source, approval_id): ...


def _gateway():
    runner = GatewayRunner()
    adapter = _Adapter(runner)
    adapter._state = AdapterState.RUNNING
    runner.register("telegram", adapter)
    runner._presence.touch(SessionSource("telegram", "chat9", "u1"))
    return runner, adapter


def _off_loop(runner, fn):
    async def go():
        runner._loop = asyncio.get_running_loop()
        return await runner._loop.run_in_executor(None, fn)
    return asyncio.run(go())


def test_grant_is_not_offered_by_adapters_without_it():
    runner, adapter = _gateway()
    ok = _off_loop(runner, lambda: runner.request_approval(
        "u1", "run_command", {"command": "pytest"}, grantable=True))
    assert ok is True and "Allow for this task" not in adapter.prompts[0]


def test_notify_user_sends_to_their_last_chat():
    runner, adapter = _gateway()
    _off_loop(runner, lambda: runner.notify_user("u1", "task done"))
    assert adapter.sent == [("chat9", "task done")]


def test_task_commands_in_chat(monkeypatch):
    monkeypatch.setattr(tasks, "_tasks", {})
    runner, _ = _gateway()
    src = SessionSource("telegram", "chat9", "u1")
    assert asyncio.run(runner.handle_command(src, "tasks")) == "No background tasks."
    assert asyncio.run(runner.handle_command(src, "cancel", "#9")) == "No background task #9."


def test_telegram_allow_for_task_button(monkeypatch):
    tg = pytest.importorskip("paulus.gateway.platforms.telegram")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "x")
    monkeypatch.setenv("TELEGRAM_TRUSTED_USERS", "7")
    runner = GatewayRunner()
    adapter = tg.TelegramAdapter(runner)
    sent = []

    class Bot:
        async def send_message(self, chat_id, text, **kw):
            sent.append(kw["reply_markup"])
            return type("Msg", (), {"message_id": 1})()

    adapter._app = type("App", (), {"bot": Bot()})()
    asyncio.run(adapter.request_approval(SessionSource("telegram", "c", "7"), "a1", "p",
                                         grantable=True))
    labels = [b.text for row in sent[0].inline_keyboard for b in row]
    assert "✅ Allow for this task" in labels

    resolved = []
    monkeypatch.setattr(runner, "resolve_approval",
                        lambda aid, answer: resolved.append(answer) or True)
    edited = []

    class Query:
        data = "dpall:a1"
        message = type("M", (), {"text": "p"})()

        async def answer(self, *a, **k): ...

        async def edit_message_text(self, text):
            edited.append(text)

    update = type("U", (), {"callback_query": Query(),
                            "effective_user": type("Usr", (), {"id": "7"})()})()
    asyncio.run(adapter._on_callback(update, None))
    assert resolved == [security.GRANT]
    assert "rest of this task" in edited[0]
