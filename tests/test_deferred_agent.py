"""The tool loop yields before owner interactions, never executing while waiting."""
import json

import pytest

from paulus import agent, automode, config, interactions, llm, security, subagents, tools


def _call(name, args, call_id="t1"):
    return llm._ToolUseBlock(id=call_id, name=name, input=args)


def _response(*blocks):
    return llm._Response(stop_reason="tool_use", content=list(blocks))


def _review():
    return automode.ReviewContext(request="Please do this job")


def _model(monkeypatch, *responses):
    replies = iter(responses)
    monkeypatch.setattr(llm, "complete", lambda *a, **kw: next(replies))


@pytest.mark.parametrize("background", [False, True])
def test_question_spec_only_in_owner_gateway_toolset(background):
    names = lambda specs: {s["name"] for s in specs}  # noqa: E731
    assert "request_input" in names(tools.agent_specs(tasks=not background, user_id="u1"))
    assert "request_input" not in names(tools.agent_specs(tasks=not background))
    assert "request_input" not in names(tools.TOOL_SPECS)
    for profile in subagents.PROFILES.values():
        assert "request_input" not in names(tools.pick(profile["tools"]))


@pytest.mark.parametrize("user_id,owner,hook,expected", [
    ("u1", True, True, True),
    ("u1", False, True, False),  # idle
    (None, True, True, False),  # CLI
    ("u1", True, False, False),  # legacy callers cannot suspend
])
def test_loop_filters_question_without_owner_and_hook(monkeypatch, user_id, owner, hook, expected):
    def model(*a, tools, **kw):
        assert ("request_input" in {s["name"] for s in tools}) is expected
        return llm._Response(content=[llm._TextBlock(text="done")])

    monkeypatch.setattr(llm, "complete", model)
    agent._run_tool_loop("system", [], user_id=user_id,
                         review=_review() if owner else None,
                         specs=[tools.REQUEST_INPUT_SPEC],
                         on_interaction=(lambda payload: None) if hook else None)


def test_respond_forwards_hook_and_propagates_suspension(monkeypatch):
    _model(monkeypatch, _response(_call("request_input", {"question": "Which?"})))
    received = []

    def defer(payload):
        received.append(payload)
        raise security.DeferredInteraction("question-id")

    with pytest.raises(security.DeferredInteraction, match="question-id"):
        agent.respond("Please start this job", user_id="u1", on_interaction=defer)
    assert received[0]["kind"] == "input"
    assert received[0]["tool_input"] == {"question": "Which?"}


def test_question_durably_suspends_with_prior_results(monkeypatch):
    _model(monkeypatch, _response(
        _call("recall", {"query": "old"}, "r"),
        _call("request_input", {"question": "Which one?", "choices": ["A", "B"]}, "q"),
        _call("run_command", {"command": "must not run"}, "c")))
    ran = []
    monkeypatch.setattr(tools, "execute", lambda name, *a, **kw:
                        (ran.append(name) or "remembered", False))
    captured = []
    source = {"platform": "test", "chat_id": "chat", "user_id": "u1"}

    def defer(payload):
        captured.append(payload)
        request_id = interactions.create(run_id="run", user_id="u1", source=source,
                                         **{k: payload[k] for k in (
                                             "kind", "tool_call_id", "tool_name", "tool_input",
                                             "checkpoint")})
        raise security.DeferredInteraction(request_id)

    review = _review()
    review.grants.add("not-for-resume")
    messages = [{"role": "user", "content": "go"}]
    with pytest.raises(security.DeferredInteraction) as caught:
        agent._run_tool_loop("system", messages, user_id="u1", review=review,
                             specs=tools.agent_specs(user_id="u1"), on_interaction=defer)

    saved = interactions.get(caught.value.request_id)
    assert saved["status"] == "pending"
    assert saved["kind"] == "input"
    state = saved["checkpoint"]
    assert state["phase"] == "tools" and state["current"] is None
    assert [r["tool_use_id"] for r in state["results"]] == ["r"]
    assert state["messages"][-1]["role"] == "assistant"
    assert "not-for-resume" not in json.dumps(state)
    assert ran == ["recall"]
    messages.clear()
    assert captured[0]["checkpoint"]["messages"]


@pytest.mark.parametrize("kind", ["input", "approval"])
def test_interaction_cannot_return_permission(monkeypatch, kind):
    monkeypatch.setattr(config, "PERMISSION_MODE", "manual")
    name, args = (("request_input", {"question": "Which?"}) if kind == "input" else
                  ("run_command", {"command": "do something"}))
    _model(monkeypatch, _response(_call(name, args)))
    monkeypatch.setattr(tools, "execute", lambda *a, **kw: pytest.fail("executed"))
    monkeypatch.setattr(security, "_gateway_confirm", lambda *a, **kw: pytest.fail("blocking ask"))
    with pytest.raises(RuntimeError, match="must persist and raise"):
        agent._run_tool_loop("system", [], user_id="u1", review=_review(),
                             specs=tools.agent_specs(user_id="u1"),
                             on_interaction=lambda payload: True)


def test_approval_hook_suspends_before_prompt_or_any_dispatch(monkeypatch):
    monkeypatch.setattr(config, "PERMISSION_MODE", "manual")
    _model(monkeypatch, _response(_call("run_command", {"command": "do something"}),
                                  _call("recall", {"query": "later"}, "t2")))
    monkeypatch.setattr(security, "_gateway_confirm", lambda *a, **kw: pytest.fail("blocking ask"))
    monkeypatch.setattr(tools, "execute", lambda *a, **kw: pytest.fail("executed"))
    events = []

    def defer(payload):
        events.append(("defer", payload))
        raise security.DeferredInteraction("approval-id")

    with pytest.raises(security.DeferredInteraction, match="approval-id"):
        agent._run_tool_loop("system", [], user_id="u1", review=_review(),
                             specs=tools.agent_specs(user_id="u1"),
                             checkpoint=lambda state: events.append(("save", state)),
                             on_interaction=defer)
    payload = events[-1][1]
    assert events[-2][0] == "save"
    assert payload["kind"] == "approval" and payload["tool_call_id"] == "t1"
    assert payload["checkpoint"]["results"] == []
    assert payload["checkpoint"]["current"] is None


def test_auto_clearance_still_passes_safety_gate_without_suspending(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(config, "PERMISSION_MODE", "auto")
    monkeypatch.setattr(security, "_can_ask", lambda uid: True)
    monkeypatch.setattr(security, "_auto_review", lambda *a: SimpleNamespace(allow=True))
    _model(monkeypatch, _response(_call("run_command", {"command": "ok"})),
           llm._Response(content=[llm._TextBlock(text="done")]))
    ran = []
    monkeypatch.setattr(tools, "execute", lambda name, *a, **kw:
                        (ran.append(name) or "ok", False))
    review = _review()
    assert agent._run_tool_loop(
        "system", [], user_id="u1", review=review, specs=tools.agent_specs(user_id="u1"),
        on_interaction=lambda p: pytest.fail("already cleared")) == "done"
    assert ran == ["run_command"]
    assert review.actions[-1][2] == "auto"


def test_unoffered_question_is_never_dispatched(monkeypatch):
    _model(monkeypatch, _response(_call("request_input", {"question": "Which?"})),
           llm._Response(content=[llm._TextBlock(text="done")]))
    monkeypatch.setattr(tools, "execute", lambda *a, **kw: pytest.fail("executed"))
    messages = []
    agent._run_tool_loop("system", messages, user_id="u1", review=_review(),
                         specs=tools.pick({"recall"}),
                         on_interaction=lambda p: pytest.fail("subagent asked owner"))
    assert "isn't available" in messages[-2]["content"][0]["content"]


def test_checkpoint_failure_cannot_reach_interaction_or_execution(monkeypatch):
    _model(monkeypatch, _response(_call("request_input", {"question": "Which?"})))
    monkeypatch.setattr(tools, "execute", lambda *a, **kw: pytest.fail("executed"))

    def checkpoint(state):
        if state["messages"] and state["phase"] == "tools":
            raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        agent._run_tool_loop("system", [], user_id="u1", review=_review(),
                             specs=tools.agent_specs(user_id="u1"), checkpoint=checkpoint,
                             on_interaction=lambda p: pytest.fail("presented"))


def test_legacy_approval_path_remains_synchronous(monkeypatch):
    _model(monkeypatch, _response(_call("run_command", {"command": "ok"})),
           llm._Response(content=[llm._TextBlock(text="done")]))
    called = []

    def clearance(name, args, user_id=None, context=None, origin=None):
        called.append("approve")
        return "owner"

    monkeypatch.setattr(security, "clearance", clearance)
    monkeypatch.setattr(tools, "execute", lambda *a, **kw:
                        (called.append("execute") or "ok", False))
    assert agent._run_tool_loop("system", [], user_id="u1", review=_review()) == "done"
    assert called == ["approve", "execute"]


@pytest.mark.parametrize("args", [{}, {"question": " "}, {"question": "Q", "choices": "A"},
                                  {"question": "Q", "choices": [1]}])
def test_invalid_question_is_an_error_not_a_suspension(monkeypatch, args):
    _model(monkeypatch, _response(_call("request_input", args)),
           llm._Response(content=[llm._TextBlock(text="done")]))
    messages = []
    agent._run_tool_loop("system", messages, user_id="u1", review=_review(),
                         specs=tools.agent_specs(user_id="u1"),
                         on_interaction=lambda p: pytest.fail("invalid question"))
    assert messages[-2]["content"][0]["is_error"]
