"""Gateway suspension, authenticated decisions, and original-task continuation."""
import asyncio
import json
import threading

import pytest

from paulus import agent, billing, config, interactions, llm, security, tasks, tools
from paulus.gateway.base import AdapterState, BasePlatformAdapter, SessionSource
from paulus.gateway.runner import GatewayRunner

SOURCE = SessionSource("test", "chat", "u1", "topic")


class Adapter(BasePlatformAdapter):
    supports_approvals = True

    def __init__(self, runner):
        super().__init__(runner)
        self._state = AdapterState.RUNNING
        self.sent = []
        self.prompts = []
        self.trusted = True

    async def start(self):
        pass

    async def stop(self):
        pass

    async def send(self, source, text):
        self.sent.append((source, text))

    async def request_approval(self, source, request_id, prompt, grantable=False):
        assert interactions.get(request_id[2:])["status"] == "pending"
        assert not grantable
        self.prompts.append((source, request_id, prompt))

    def can_approve(self, user_id):
        return self.trusted and user_id == "u1"


def setup(monkeypatch):
    runner = GatewayRunner()
    adapter = Adapter(runner)
    runner.register("test", adapter)
    monkeypatch.setattr(config, "PERMISSION_MODE", "manual")
    monkeypatch.setattr(config, "MCP_ON_DEMAND", True)
    monkeypatch.setattr(agent, "_build_system", lambda *a: "system")
    monkeypatch.setattr(agent, "_history_to_messages", lambda *a: [{"role": "user", "content": "go"}])
    return runner, adapter


def call(name="request_input", args=None, call_id="q"):
    return llm._Response(stop_reason="tool_use", content=[llm._ToolUseBlock(
        id=call_id, name=name, input=args or {"question": "Which color?"})])


def text(value="done"):
    return llm._Response(content=[llm._TextBlock(text=value)])


async def until(predicate):
    for _ in range(300):
        if predicate():
            return
        await asyncio.sleep(.01)
    pytest.fail("continuation did not complete")


def test_question_releases_lock_and_correlates_answer(monkeypatch):
    runner, adapter = setup(monkeypatch)
    replies = iter([call(), text("other turn"), text("blue selected")])
    seen = []

    def model(system, messages, **kw):
        seen.append(json.loads(json.dumps(messages)))
        return next(replies)

    monkeypatch.setattr(llm, "complete", model)

    async def scenario():
        runner._loop = asyncio.get_running_loop()
        await runner.handle_inbound(SOURCE, "go")
        assert not runner._agent_lock.locked()
        record, = interactions.pending("u1", vars(SOURCE))
        await asyncio.sleep(0)
        assert any("/answer " + record["id"] in msg for _, msg in adapter.sent)
        await runner.handle_inbound(SOURCE, "unrelated turn")
        assert any("other turn" == msg for _, msg in adapter.sent)
        for bad in (SessionSource("test", "other", "u1", "topic"),
                    SessionSource("test", "chat", "u1", "else"),
                    SessionSource("test", "chat", "u2", "topic")):
            assert "unavailable" in await runner.handle_command(bad, "answer", record["id"] + " blue")
        assert "resuming" in await runner.handle_command(SOURCE, "answer", record["id"] + " blue")
        assert "already" in await runner.handle_command(SOURCE, "answer", record["id"] + " red")
        await until(lambda: any(msg == "blue selected" for _, msg in adapter.sent))
        saved = interactions.get(record["id"])
        assert saved["status"] == "applied"
        assert saved["tool_result"]["tool_use_id"] == "q"
        assert saved["tool_result"]["content"] == "blue"
        assert "blue" in json.dumps(seen[-1])

    asyncio.run(scenario())


def test_approval_exact_action_execution_boundary_and_policy(monkeypatch):
    runner, adapter = setup(monkeypatch)
    command = {"command": "echo approved"}
    replies = iter([call("run_command", command), text()])
    monkeypatch.setattr(llm, "complete", lambda *a, **kw: next(replies))
    executions = []

    def execute(name, args, user_id=None):
        record, = [interactions.get(r) for r in ids]
        assert record["status"] == "executing"
        executions.append((name, args, user_id))
        return "exact output", False

    monkeypatch.setattr(tools, "execute", execute)
    ids = []

    async def scenario():
        runner._loop = asyncio.get_running_loop()
        await runner.handle_inbound(SOURCE, "go")
        record, = interactions.pending("u1", vars(SOURCE))
        ids.append(record["id"])
        aid = "d-" + record["id"]
        assert not runner.resolve_deferred_approval(aid, security.GRANT, SOURCE)
        for bad in (SessionSource("test", "else", "u1", "topic"),
                    SessionSource("test", "chat", "u1"),
                    SessionSource("test", "chat", "u2", "topic")):
            assert not runner.resolve_deferred_approval(aid, True, bad)
        adapter.trusted = False
        assert not runner.resolve_deferred_approval(aid, True, SOURCE)
        adapter.trusted = True
        assert runner.resolve_deferred_approval(aid, True, SOURCE)
        assert not runner.resolve_deferred_approval(aid, True, SOURCE)
        await until(lambda: any(msg == "done" for _, msg in adapter.sent))
        assert executions == [("run_command", command, "u1")]
        saved = interactions.get(record["id"])
        assert saved["tool_result"]["content"] == "exact output"
        assert interactions.recover() == []
        assert not runner.resolve_deferred_approval(aid, True, SOURCE)

    asyncio.run(scenario())


def test_preexecution_failure_can_retry_without_restart(monkeypatch):
    runner, adapter = setup(monkeypatch)
    replies = iter([call("run_command", {"command": "once"}), text("recovered")])
    monkeypatch.setattr(llm, "complete", lambda *a, **kw: next(replies))
    ran = []
    monkeypatch.setattr(tools, "execute", lambda *a, **kw: (ran.append(a) or "ok", False))

    async def scenario():
        runner._loop = asyncio.get_running_loop()
        await runner.handle_inbound(SOURCE, "go")
        record, = interactions.pending("u1", vars(SOURCE))
        assert interactions.answer(record["id"], "u1", vars(SOURCE), "approve")
        adapter.trusted = False
        await runner._resume_interaction(record["id"], SOURCE)
        assert interactions.get(record["id"])["status"] == "answered"
        assert not ran
        assert any("/resume " + record["id"] in msg for _, msg in adapter.sent)
        adapter.trusted = True
        assert "Resuming" in await runner.handle_command(SOURCE, "resume", record["id"])
        await until(lambda: any(msg == "recovered" for _, msg in adapter.sent))
        assert len(ran) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("status", ["pending", "answered", "processing"])
def test_cancel_suspended_task_invalidates_unexecuted_decision(monkeypatch, status):
    runner, adapter = setup(monkeypatch)
    monkeypatch.setattr(llm, "complete", lambda *a, **kw: call("run_command", {"command": "never"}))
    monkeypatch.setattr(tools, "execute", lambda *a, **kw: pytest.fail("cancelled action ran"))

    async def scenario():
        runner._loop = asyncio.get_running_loop()
        assert "Started" in await runner.handle_command(SOURCE, "bg", "job")
        await until(lambda: bool(interactions.pending("u1", vars(SOURCE))))
        record, = interactions.pending("u1", vars(SOURCE))
        task = tasks._tasks[record["task_id"]]
        await until(lambda: task.status == "suspended")
        await until(lambda: not any(t.name == f"paulus-task-{task.id}" for t in threading.enumerate()))
        if status != "pending":
            assert interactions.answer(record["id"], "u1", vars(SOURCE), "approve")
        claimed = interactions.claim(record["id"]) if status == "processing" else None
        assert "Cancelled" in tasks.cancel("u1", task.id)
        assert task.cancel.is_set() and task.status == "cancelled"
        assert interactions.get(record["id"])["status"] == "cancelled"
        if claimed:
            assert not tasks.resume_interaction(claimed, runner)
        else:
            await runner._resume_interaction(record["id"], SOURCE)
        assert task.status == "cancelled"
        assert not runner.resolve_deferred_approval("d-" + record["id"], True, SOURCE)

    asyncio.run(scenario())


def test_denial_never_invokes_tool(monkeypatch):
    runner, adapter = setup(monkeypatch)
    replies = iter([call("run_command", {"command": "no"}), text("declined")])
    monkeypatch.setattr(llm, "complete", lambda *a, **kw: next(replies))
    monkeypatch.setattr(tools, "execute", lambda *a, **kw: pytest.fail("denied action ran"))

    async def scenario():
        runner._loop = asyncio.get_running_loop()
        await runner.handle_inbound(SOURCE, "go")
        record, = interactions.pending("u1", vars(SOURCE))
        assert runner.resolve_deferred_approval("d-" + record["id"], False, SOURCE)
        await until(lambda: any(msg == "declined" for _, msg in adapter.sent))
        assert "NOT approved" in interactions.get(record["id"])["tool_result"]["content"]

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["input", "approval"])
def test_background_suspends_and_resumes_original_task(monkeypatch, kind):
    runner, adapter = setup(monkeypatch)
    name, args = (("request_input", {"question": "Color?"}) if kind == "input" else
                  ("run_command", {"command": "echo task"}))
    replies = iter([call(name, args), text("task complete")])
    monkeypatch.setattr(llm, "complete", lambda *a, **kw: next(replies))
    ran = []
    monkeypatch.setattr(tools, "execute", lambda *a, **kw: (ran.append(a) or "output", False))

    async def scenario():
        runner._loop = asyncio.get_running_loop()
        assert "Started" in await runner.handle_command(SOURCE, "bg", "choose a color")
        await until(lambda: bool(interactions.pending("u1", vars(SOURCE))))
        record, = interactions.pending("u1", vars(SOURCE))
        await until(lambda: tasks._tasks[record["task_id"]].status == "suspended")
        task = tasks._tasks[record["task_id"]]
        await until(lambda: not any(t.name == f"paulus-task-{task.id}" for t in threading.enumerate()))
        assert record["run_id"] == f"task:{task.id}"
        assert f"#{task.id} [suspended]" in tasks.status("u1")
        # A new process hydrates suspended tasks, rather than marking them done.
        tasks._initialized = False
        tasks._tasks = {}
        assert "suspended" in tasks.status("u1")
        if kind == "input":
            await runner.handle_command(SOURCE, "answer", record["id"] + " blue")
        else:
            assert runner.resolve_deferred_approval("d-" + record["id"], True, SOURCE)
        await until(lambda: tasks._tasks[task.id].status == "done")
        assert len(tasks._tasks) == 1
        assert "task complete" in tasks._tasks[task.id].result
        assert bool(ran) == (kind == "approval")
        saved = interactions.get(record["id"])
        assert saved["status"] == "applied"
        with tasks._store() as db:
            state = json.loads(db.execute("SELECT state FROM task_progress WHERE task_id = ?", (task.id,)).fetchone()[0])
        assert "tool_result" in json.dumps(state)
        await until(lambda: any("task complete" in msg for _, msg in adapter.sent))

    asyncio.run(scenario())


def test_executing_restart_is_unknown_and_never_replayed(monkeypatch):
    runner, adapter = setup(monkeypatch)
    monkeypatch.setattr(llm, "complete", lambda *a, **kw: call("run_command", {"command": "once"}))

    async def scenario():
        runner._loop = asyncio.get_running_loop()
        await runner.handle_inbound(SOURCE, "go")
        record, = interactions.pending("u1", vars(SOURCE))
        assert interactions.answer(record["id"], "u1", vars(SOURCE), "approve")
        claimed = interactions.claim(record["id"])
        assert interactions.begin_execution(claimed["id"])
        assert interactions.recover() == [record["id"]]
        monkeypatch.setattr(tools, "execute", lambda *a, **kw: pytest.fail("replayed"))
        assert not runner.resolve_deferred_approval("d-" + record["id"], True, SOURCE)
        await runner._resume_interaction(record["id"], SOURCE)
        assert interactions.get(record["id"])["status"] == "unknown"

    asyncio.run(scenario())


def test_restart_after_approved_result_resumes_once_without_reexecuting(monkeypatch):
    runner, adapter = setup(monkeypatch)
    monkeypatch.setattr(llm, "complete", lambda *a, **kw: call("run_command", {"command": "once"}))
    executions = []
    monkeypatch.setattr(tools, "execute", lambda *a, **kw: (executions.append(a) or "done once", False))
    original_claim = interactions.claim_continuation
    monkeypatch.setattr(interactions, "claim_continuation", lambda request_id: None)

    async def scenario():
        runner._loop = asyncio.get_running_loop()
        await runner.handle_inbound(SOURCE, "go")
        record, = interactions.pending("u1", vars(SOURCE))
        assert runner.resolve_deferred_approval("d-" + record["id"], True, SOURCE)
        await until(lambda: interactions.get(record["id"])["status"] == "applied")
        assert len(executions) == 1
        assert interactions.get(record["id"])["tool_result"]["content"] == "done once"
        assert [r["id"] for r in interactions.ready_continuations()] == [record["id"]]
        monkeypatch.setattr(interactions, "claim_continuation", original_claim)
        observed = []

        def model(system, messages, **kw):
            observed.append(json.loads(json.dumps(messages)))
            return text("recovered reply")

        monkeypatch.setattr(llm, "complete", model)
        interactions.recover()
        await runner._restore_interactions()
        # Two restoration passes must not start duplicate continuations.
        await runner._restore_interactions()
        await until(lambda: any(msg == "recovered reply" for _, msg in adapter.sent))
        await until(lambda: not interactions.ready_continuations())
        assert len(executions) == 1 and len(observed) == 1
        assert "done once" in json.dumps(observed[0])
        assert not interactions.claim_continuation(record["id"])
        # The final send and durable completion also settle before another restart.
        await asyncio.sleep(.05)
        interactions.recover()
        await runner._restore_interactions()
        await asyncio.sleep(.05)
        assert len(observed) == 1 and len(executions) == 1

    asyncio.run(scenario())


def test_restart_after_followup_execution_boundary_does_not_replay(monkeypatch):
    runner, adapter = setup(monkeypatch)
    monkeypatch.setattr(llm, "complete", lambda *a, **kw: call())
    original_claim = interactions.claim_continuation
    monkeypatch.setattr(interactions, "claim_continuation", lambda request_id: None)

    async def scenario():
        runner._loop = asyncio.get_running_loop()
        await runner.handle_inbound(SOURCE, "go")
        record, = interactions.pending("u1", vars(SOURCE))
        await runner.handle_command(SOURCE, "answer", record["id"] + " blue")
        await until(lambda: interactions.get(record["id"])["status"] == "applied")
        monkeypatch.setattr(interactions, "claim_continuation", original_claim)
        claimed = interactions.claim_continuation(record["id"])
        state = claimed["checkpoint"]
        state["messages"].append({"role": "assistant", "content": [
            {"type": "tool_use", "id": "later", "name": "run_command",
             "input": {"command": "possibly ran"}}]})
        state.update(phase="executing", results=[], current="later")
        assert interactions.save_continuation(record["id"], state)
        assert not interactions.claim_continuation(record["id"])
        interactions.recover()
        seen = []
        monkeypatch.setattr(tools, "execute", lambda *a, **kw: pytest.fail("replayed"))
        monkeypatch.setattr(llm, "complete", lambda sys, messages, **kw:
                            (seen.append(json.dumps(messages)) or text("reconciled")))
        await runner._restore_interactions()
        await until(lambda: any(msg == "reconciled" for _, msg in adapter.sent))
        assert "Outcome UNKNOWN" in seen[0]
        assert "blue" in seen[0]

    asyncio.run(scenario())


def test_continuation_claim_is_atomic_and_new_question_handoff(monkeypatch):
    runner, adapter = setup(monkeypatch)
    responses = iter([call(), call(call_id="q2"), text("done")])
    monkeypatch.setattr(llm, "complete", lambda *a, **kw: next(responses))

    async def scenario():
        runner._loop = asyncio.get_running_loop()
        await runner.handle_inbound(SOURCE, "go")
        first, = interactions.pending("u1", vars(SOURCE))
        await runner.handle_command(SOURCE, "answer", first["id"] + " blue")
        await until(lambda: any(r["id"] != first["id"] for r in interactions.pending("u1", vars(SOURCE))))
        second, = interactions.pending("u1", vars(SOURCE))
        assert second["run_id"] == first["run_id"]
        interactions.recover()
        assert not interactions.claim_continuation(first["id"])
        await runner.handle_command(SOURCE, "answer", second["id"] + " green")
        await until(lambda: any(msg == "done" for _, msg in adapter.sent))
        assert not interactions.claim_continuation(second["id"])

    asyncio.run(scenario())


def test_approved_execution_is_metered_and_balance_pay_link_preserved(monkeypatch):
    runner, adapter = setup(monkeypatch)
    monkeypatch.setattr(config, "BILLING_API_BASE", "https://billing.invalid")
    monkeypatch.setattr(billing, "gate", lambda user_id: (True, None))
    replies = iter([call("run_command", {"command": "once"}), text("should not execute")])
    monkeypatch.setattr(llm, "complete", lambda *a, **kw: next(replies))
    pay_message = "Top up" + billing._PAY_LINK_MARKER + "https://example.test/pay"
    ran = []
    links = []

    def execute(*args, **kw):
        assert billing.metering()
        ran.append(args)
        raise billing.BalanceExhausted(pay_message)

    async def send_with_link(source, text, label, url):
        links.append((text, url))

    monkeypatch.setattr(tools, "execute", execute)
    monkeypatch.setattr(adapter, "send_with_link", send_with_link)

    async def scenario():
        runner._loop = asyncio.get_running_loop()
        await runner.handle_inbound(SOURCE, "go")
        record, = interactions.pending("u1", vars(SOURCE))
        assert runner.resolve_deferred_approval("d-" + record["id"], True, SOURCE)
        await until(lambda: bool(links))
        assert links[0][1] == "https://example.test/pay"
        assert "UNKNOWN" in links[0][0] and "Top up" in links[0][0]
        assert len(ran) == 1
        assert interactions.get(record["id"])["status"] == "executing"
        assert interactions.recover() == [record["id"]]
        await runner._restore_interactions()
        await until(lambda: any("UNKNOWN" in msg for _, msg in adapter.sent))
        assert len(ran) == 1

    asyncio.run(scenario())


def test_caught_billing_error_during_action_remains_unknown(monkeypatch):
    runner, adapter = setup(monkeypatch)
    monkeypatch.setattr(config, "BILLING_API_BASE", "https://billing.invalid")
    monkeypatch.setattr(billing, "gate", lambda user_id: (True, None))
    monkeypatch.setattr(llm, "complete", lambda *a, **kw:
                        call("run_command", {"command": "once"}))
    pay_message = "Top up" + billing._PAY_LINK_MARKER + "https://example.test/pay"
    links = []

    def execute(*args, **kw):
        billing._stop(billing._meter.get(), pay_message)
        return "Tool error: Top up", True

    async def send_with_link(source, text, label, url):
        links.append((text, url))

    monkeypatch.setattr(tools, "execute", execute)
    monkeypatch.setattr(adapter, "send_with_link", send_with_link)

    async def scenario():
        runner._loop = asyncio.get_running_loop()
        await runner.handle_inbound(SOURCE, "go")
        record, = interactions.pending("u1", vars(SOURCE))
        assert runner.resolve_deferred_approval("d-" + record["id"], True, SOURCE)
        await until(lambda: bool(links))
        assert "UNKNOWN" in links[0][0] and links[0][1] == "https://example.test/pay"
        assert interactions.get(record["id"])["status"] == "executing"
        assert not interactions.ready_continuations()

    asyncio.run(scenario())


def test_balance_stop_after_success_keeps_result_and_recovery(monkeypatch):
    runner, adapter = setup(monkeypatch)
    monkeypatch.setattr(config, "BILLING_API_BASE", "https://billing.invalid")
    monkeypatch.setattr(billing, "gate", lambda user_id: (True, None))
    replies = iter([call("run_command", {"command": "once"}), text("after topup")])
    monkeypatch.setattr(llm, "complete", lambda *a, **kw: next(replies))
    pay_message = "Top up" + billing._PAY_LINK_MARKER + "https://example.test/pay"
    ran = []
    links = []

    def execute(*args, **kw):
        assert billing.metering()
        ran.append(args)
        billing._stop(billing._meter.get(), pay_message)
        return "completed", False

    async def send_with_link(source, text, label, url):
        links.append((text, url))

    monkeypatch.setattr(tools, "execute", execute)
    monkeypatch.setattr(adapter, "send_with_link", send_with_link)

    async def scenario():
        runner._loop = asyncio.get_running_loop()
        await runner.handle_inbound(SOURCE, "go")
        record, = interactions.pending("u1", vars(SOURCE))
        assert runner.resolve_deferred_approval("d-" + record["id"], True, SOURCE)
        await until(lambda: bool(links))
        assert "Top up" in links[0][0] and "/resume " + record["id"] in links[0][0]
        assert links[0][1] == "https://example.test/pay"
        assert interactions.get(record["id"])["tool_result"]["content"] == "completed"
        assert len(interactions.ready_continuations()) == 1
        wrong = SessionSource("test", "other", "u1", "topic")
        assert "unavailable" in await runner.handle_command(wrong, "resume", record["id"])
        assert "Resuming" in await runner.handle_command(SOURCE, "resume", record["id"])
        await until(lambda: any(msg == "after topup" for _, msg in adapter.sent))
        assert len(ran) == 1

    asyncio.run(scenario())


def test_background_approval_exhaustion_keeps_task_meter_and_payment_link(monkeypatch):
    runner, adapter = setup(monkeypatch)
    monkeypatch.setattr(config, "BILLING_API_BASE", "https://billing.invalid")
    monkeypatch.setattr(billing, "gate", lambda user_id: (True, None))
    monkeypatch.setattr(llm, "complete", lambda *a, **kw:
                        call("run_command", {"command": "task action"}))
    pay_message = "Top up task" + billing._PAY_LINK_MARKER + "https://example.test/task"
    links = []
    invoked = []

    def execute(*args, **kw):
        meter = billing._meter.get()
        assert meter and meter.user_id == "u1" and meter.kind == "task"
        invoked.append(args)
        raise billing.BalanceExhausted(pay_message)

    async def send_with_link(source, text, label, url):
        links.append((source, text, url))

    monkeypatch.setattr(tools, "execute", execute)
    monkeypatch.setattr(adapter, "send_with_link", send_with_link)

    async def scenario():
        runner._loop = asyncio.get_running_loop()
        await runner.handle_command(SOURCE, "bg", "task action")
        await until(lambda: bool(interactions.pending("u1", vars(SOURCE))))
        record, = interactions.pending("u1", vars(SOURCE))
        await until(lambda: tasks._tasks[record["task_id"]].status == "suspended")
        assert runner.resolve_deferred_approval("d-" + record["id"], True, SOURCE)
        await until(lambda: bool(links))
        await until(lambda: tasks._tasks[record["task_id"]].status == "interrupted")
        assert links[0][0] == SOURCE and links[0][2] == "https://example.test/task"
        assert "UNKNOWN" in links[0][1] and "Top up task" in links[0][1]
        assert interactions.get(record["id"])["status"] == "executing"
        assert len(invoked) == 1

    asyncio.run(scenario())


def test_unsent_final_continuation_recovers_without_calling_model(monkeypatch):
    runner, adapter = setup(monkeypatch)
    replies = iter([call(), text("final answer")])
    model_calls = []

    def model(*args, **kw):
        model_calls.append(1)
        return next(replies)

    monkeypatch.setattr(llm, "complete", model)
    original_send = adapter.send

    async def fail_final(source, message):
        if message == "final answer":
            raise OSError("offline")
        await original_send(source, message)

    monkeypatch.setattr(adapter, "send", fail_final)

    async def scenario():
        runner._loop = asyncio.get_running_loop()
        await runner.handle_inbound(SOURCE, "go")
        record, = interactions.pending("u1", vars(SOURCE))
        await runner.handle_command(SOURCE, "answer", record["id"] + " blue")
        await until(lambda: len(model_calls) == 2)
        await until(lambda: bool(interactions.ready_continuations()))
        monkeypatch.setattr(adapter, "send", original_send)
        interactions.recover()
        await runner._restore_interactions()
        await until(lambda: any(msg == "final answer" for _, msg in adapter.sent))
        assert len(model_calls) == 2

    asyncio.run(scenario())


@pytest.mark.parametrize("background", [False, True])
def test_billing_block_before_execution_can_resume_after_topup(monkeypatch, background):
    runner, adapter = setup(monkeypatch)
    monkeypatch.setattr(config, "BILLING_API_BASE", "https://billing.invalid")
    allowed = [True]
    pay_message = "Top up" + billing._PAY_LINK_MARKER + "https://example.test/pay"
    monkeypatch.setattr(billing, "gate", lambda user_id:
                        (True, None) if allowed[0] else (False, pay_message))
    replies = iter([call("run_command", {"command": "once"}), text("resumed")])
    monkeypatch.setattr(llm, "complete", lambda *a, **kw: next(replies))
    invoked = []
    links = []
    monkeypatch.setattr(tools, "execute", lambda *a, **kw: (invoked.append(a) or "ok", False))

    async def send_with_link(source, text, label, url):
        links.append((source, text, url))

    monkeypatch.setattr(adapter, "send_with_link", send_with_link)

    async def scenario():
        runner._loop = asyncio.get_running_loop()
        if background:
            await runner.handle_command(SOURCE, "bg", "run command")
        else:
            await runner.handle_inbound(SOURCE, "run command")
        await until(lambda: bool(interactions.pending("u1", vars(SOURCE))))
        record, = interactions.pending("u1", vars(SOURCE))
        if background:
            await until(lambda: tasks._tasks[record["task_id"]].status == "suspended")
        allowed[0] = False
        assert runner.resolve_deferred_approval("d-" + record["id"], True, SOURCE)
        await until(lambda: bool(links))
        await until(lambda: interactions.get(record["id"])["status"] == "answered")
        if background:
            await until(lambda: tasks._tasks[record["task_id"]].status == "interrupted")
        assert not invoked and links[0][2] == "https://example.test/pay"
        allowed[0] = True
        wrong = SessionSource("test", "chat", "u1", "wrong")
        assert "unavailable" in await runner.handle_command(wrong, "resume", record["id"])
        assert "Resuming" in await runner.handle_command(SOURCE, "resume", record["id"])
        if background:
            await until(lambda: tasks._tasks[record["task_id"]].status == "done")
        else:
            await until(lambda: any(message == "resumed" for _, message in adapter.sent))
        assert len(invoked) == 1

    asyncio.run(scenario())
