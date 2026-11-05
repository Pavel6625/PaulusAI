"""Per-token billing: billing.metered / record / ensure_can_spend, llm.py's
usage and cost extraction (OpenRouter's usage.cost), and the wiring that
meters a turn (agent.respond) or a background task (tasks._run) but never an
idle check-in."""
import concurrent.futures
import contextvars
import json
import threading
from types import SimpleNamespace

import pytest

from paulus import agent, billing, config, llm, memory, tasks

BASE = "https://billing.example.com"


@pytest.fixture
def service(monkeypatch):
    """Billing enabled; captures every record POST. Set ``service.reply`` to
    the dict the record call returns, or to an exception to raise."""
    monkeypatch.setattr(config, "BILLING_API_BASE", BASE)
    monkeypatch.setattr(config, "BILLING_TOPUP_URL", "")
    state = SimpleNamespace(records=[], reply={})

    def fake_post(path, payload, what):
        assert path == config.BILLING_RECORD_PATH
        state.records.append(payload)
        if isinstance(state.reply, Exception):
            raise state.reply
        return state.reply

    monkeypatch.setattr(billing, "_post", fake_post)
    return state


def _record(**overrides):
    kw = dict(cost_usd=0.01, cost_source="provider", prompt_tokens=100,
              completion_tokens=20, model="openrouter/openai/gpt-4o",
              request_id="gen-1")
    kw.update(overrides)
    billing.record(**kw)


# --- billing.metered / record -------------------------------------------------

def test_metered_is_a_noop_when_billing_is_off(monkeypatch):
    monkeypatch.setattr(config, "BILLING_API_BASE", "")
    monkeypatch.setattr(billing, "_post", lambda *a: pytest.fail("posted"))
    with billing.metered("u1") as meter:
        assert meter is None and not billing.metering()
        _record()


def test_metered_is_a_noop_for_the_cli_user(service):
    with billing.metered(None) as meter:
        assert meter is None
        _record()
    assert service.records == []


def test_record_outside_a_metered_turn_posts_nothing(service):
    _record()
    assert service.records == []


def test_record_posts_tokens_and_raw_cost(service):
    with billing.metered("u1", kind="task") as meter:
        _record()
        _record(cost_usd=0.02, prompt_tokens=50, completion_tokens=5, request_id="gen-2")

    assert service.records[0] == {
        "user_id": "u1", "kind": "task", "model": "openrouter/openai/gpt-4o",
        "request_id": "gen-1", "prompt_tokens": 100, "completion_tokens": 20,
        "cost_usd": 0.01, "cost_source": "provider",
    }
    assert meter.calls == 2
    assert meter.cost_usd == pytest.approx(0.03)
    assert (meter.prompt_tokens, meter.completion_tokens) == (150, 25)
    assert "billing_usage" in config.AUDIT_LOG.read_text(encoding="utf-8")
    assert not billing.metering()          # the meter is gone after the block


def test_exhausted_reply_stops_the_next_call_with_a_pay_link(service, monkeypatch):
    monkeypatch.setattr(config, "BILLING_TOPUP_URL", "https://pay.example.com/{user_id}")
    service.reply = {"allowed": False, "balance": 0}
    with billing.metered("u1"):
        billing.ensure_can_spend()         # nothing recorded yet: may spend
        _record()
        with pytest.raises(billing.BalanceExhausted) as caught:
            billing.ensure_can_spend()

    assert str(caught.value) == billing.INSUFFICIENT_BALANCE_MESSAGE
    assert billing.split_pay_link(caught.value.block_message) == (
        billing.INSUFFICIENT_BALANCE_MESSAGE, "https://pay.example.com/u1")


def test_allowed_or_empty_reply_keeps_spending(service):
    for reply in ({}, {"allowed": True, "balance": 3}):
        service.reply = reply
        with billing.metered("u1"):
            _record()
            billing.ensure_can_spend()


def test_record_failure_fails_closed_and_keeps_the_payload(service):
    service.reply = billing.BillingError("usage record HTTP 500")
    with billing.metered("u1"):
        _record()
        with pytest.raises(billing.BalanceExhausted) as caught:
            billing.ensure_can_spend()

    assert caught.value.block_message == billing.SERVICE_UNAVAILABLE_MESSAGE
    log = config.AUDIT_LOG.read_text(encoding="utf-8")
    assert "billing_record_error" in log and '"request_id": "gen-1"' in log


def test_unpriced_call_is_audited(service):
    with billing.metered("u1"):
        _record(cost_usd=0.0, cost_source="none")
    assert "billing_unpriced" in config.AUDIT_LOG.read_text(encoding="utf-8")


def test_subagent_threads_share_the_turns_meter(service):
    """subagents.delegate runs workers in a copy of the turn's context."""
    with billing.metered("u1") as meter:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            for f in [pool.submit(contextvars.copy_context().run, _record)
                      for _ in range(3)]:
                f.result()
    assert meter.calls == 3 and len(service.records) == 3


def test_post_accepts_an_empty_2xx_body(monkeypatch):
    monkeypatch.setattr(config, "BILLING_API_BASE", BASE)

    class Resp:
        def read(self):
            return b""

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["body"] = json.loads(req.data.decode())
        return Resp()

    monkeypatch.setattr("paulus.billing.urllib.request.urlopen", fake_urlopen)
    assert billing._post("/usage/record", {"user_id": "u1"}, "usage record") == {}
    assert seen == {"url": f"{BASE}/usage/record", "body": {"user_id": "u1"}}


# --- llm.py: usage & cost extraction ----------------------------------------

def _reply(usage=None, hidden=None, text="ok"):
    resp = SimpleNamespace(id="gen-abc", usage=usage, choices=[SimpleNamespace(
        finish_reason="stop", message=SimpleNamespace(content=text, tool_calls=None))])
    if hidden is not None:
        resp._hidden_params = hidden
    return resp


def _litellm(monkeypatch, *responses):
    calls = []
    replies = iter(responses)

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return next(replies)

    monkeypatch.setattr(llm.litellm, "completion", fake_completion)
    return calls


def _ask():
    return llm.complete("sys", [{"role": "user", "content": "hi"}],
                        model="openrouter/openai/gpt-4o")


def test_complete_records_openrouter_usage_cost(service, monkeypatch):
    usage = SimpleNamespace(prompt_tokens=1200, completion_tokens=80, cost=0.0042)
    _litellm(monkeypatch, _reply(usage))
    with billing.metered("u1"):
        assert _ask().content[0].text == "ok"

    rec = service.records[0]
    assert (rec["cost_usd"], rec["cost_source"]) == (0.0042, "provider")
    assert (rec["prompt_tokens"], rec["completion_tokens"]) == (1200, 80)
    assert rec["model"] == "openrouter/openai/gpt-4o"
    assert rec["request_id"] == "gen-abc"


def test_cost_falls_back_to_litellms_hidden_openrouter_cost(service, monkeypatch):
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=2)
    hidden = {"additional_headers": {"llm_provider-x-litellm-response-cost": 0.5}}
    _litellm(monkeypatch, _reply(usage, hidden))
    with billing.metered("u1"):
        _ask()
    assert service.records[0]["cost_usd"] == 0.5


def test_byok_adds_the_upstream_provider_cost(service, monkeypatch):
    usage = {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.001,
             "is_byok": True, "cost_details": {"upstream_inference_cost": 0.02}}
    _litellm(monkeypatch, _reply(usage))
    with billing.metered("u1"):
        _ask()
    assert service.records[0]["cost_usd"] == pytest.approx(0.021)


def test_cost_is_estimated_from_tokens_without_a_provider_cost(service, monkeypatch):
    _litellm(monkeypatch, _reply(SimpleNamespace(prompt_tokens=1000, completion_tokens=100)))
    monkeypatch.setattr(llm.litellm, "cost_per_token",
                        lambda model, prompt_tokens, completion_tokens:
                        (prompt_tokens * 1e-6, completion_tokens * 2e-6))
    with billing.metered("u1"):
        _ask()
    rec = service.records[0]
    assert rec["cost_source"] == "estimate"
    assert rec["cost_usd"] == pytest.approx(0.0012)


def test_unpriceable_call_still_reports_its_tokens(service, monkeypatch):
    _litellm(monkeypatch, _reply(SimpleNamespace(prompt_tokens=7, completion_tokens=3)))

    def unknown(**kw):
        raise Exception("model not mapped")

    monkeypatch.setattr(llm.litellm, "cost_per_token", unknown)
    with billing.metered("u1"):
        _ask()
    rec = service.records[0]
    assert (rec["cost_usd"], rec["cost_source"]) == (0.0, "none")
    assert (rec["prompt_tokens"], rec["completion_tokens"]) == (7, 3)


def test_router_fallback_is_billed_as_the_model_that_answered(service, monkeypatch):
    monkeypatch.setattr(config, "ROUTER_MODEL", "openrouter/typesafe/jev-router")
    monkeypatch.setattr(config, "CORE_MODEL", "anthropic/claude-sonnet-4-6")
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs["model"])
        if kwargs["model"] == config.ROUTER_MODEL:
            raise RuntimeError("503")
        return _reply(SimpleNamespace(prompt_tokens=1, completion_tokens=1, cost=0.1))

    monkeypatch.setattr(llm.litellm, "completion", fake_completion)
    with billing.metered("u1"):
        llm.complete("sys", [{"role": "user", "content": "hi"}], model=config.ROUTER_MODEL)
    assert service.records[0]["model"] == "anthropic/claude-sonnet-4-6"


def test_stopped_meter_refuses_the_call_before_sending_it(service, monkeypatch):
    monkeypatch.setattr(llm.litellm, "completion", lambda **k: pytest.fail("sent"))
    with billing.metered("u1") as meter:
        meter.block_message = "out"
        with pytest.raises(billing.BalanceExhausted):
            _ask()


def test_unmetered_call_does_no_billing_work(monkeypatch):
    monkeypatch.setattr(config, "BILLING_API_BASE", BASE)
    monkeypatch.setattr(billing, "_post", lambda *a: pytest.fail("posted"))
    _litellm(monkeypatch, _reply(SimpleNamespace(prompt_tokens=1, completion_tokens=1, cost=1)))
    _ask()                                 # e.g. memory consolidation


def _chunk(text=None, finish=None, usage=None, choices=True):
    return SimpleNamespace(
        id="gen-stream", usage=usage,
        choices=[SimpleNamespace(finish_reason=finish,
                                 delta=SimpleNamespace(content=text, tool_calls=None))]
        if choices else [])


def test_stream_asks_for_usage_and_bills_the_final_chunk(service, monkeypatch):
    usage = SimpleNamespace(prompt_tokens=900, completion_tokens=40, cost=0.003)
    calls = _litellm(monkeypatch, [_chunk("hel"), _chunk("lo", finish="stop"),
                                   _chunk(usage=usage, choices=False)])
    pieces = []
    with billing.metered("u1"):
        resp = llm.stream("sys", [{"role": "user", "content": "hi"}],
                          on_delta=pieces.append, model="openrouter/openai/gpt-4o")

    assert calls[0]["stream_options"] == {"include_usage": True}
    assert resp.content[0].text == "hello" and pieces == ["hel", "lo"]
    rec = service.records[0]
    assert (rec["cost_usd"], rec["prompt_tokens"], rec["request_id"]) == (
        0.003, 900, "gen-stream")


def test_unmetered_stream_sends_no_stream_options(monkeypatch):
    calls = _litellm(monkeypatch, [_chunk("hi", finish="stop")])
    llm.stream("sys", [{"role": "user", "content": "hi"}])
    assert "stream_options" not in calls[0]


# --- wiring: which work is billed --------------------------------------------

def _tool_call():
    return llm._Response(stop_reason="tool_use", content=[
        llm._ToolUseBlock(id="t1", name="recall", input={"query": "q"})])


def test_respond_meters_the_turn_and_stops_when_balance_runs_out(service, monkeypatch):
    monkeypatch.setattr(billing, "_check_usage", lambda uid: {"allowed": True})
    service.reply = {"allowed": False, "balance": 0,
                     "payment_url": "https://pay.example.com/s/1"}
    calls = []

    def model(*a, **k):                    # what llm.complete does around a call
        billing.ensure_can_spend()
        calls.append(1)
        billing.record(cost_usd=0.5, cost_source="provider", prompt_tokens=1,
                       completion_tokens=1, model="m")
        return _tool_call()

    monkeypatch.setattr(llm, "complete", model)
    monkeypatch.setattr(agent, "_execute", lambda *a, **k: ("nothing", False))

    reply = agent.respond("look it up", user_id="u1")

    assert len(calls) == 1                 # the second step never reached the model
    text, url = billing.split_pay_link(reply)
    assert text == billing.INSUFFICIENT_BALANCE_MESSAGE
    assert url == "https://pay.example.com/s/1"
    assert memory.recent_episodes(user_id="u1")[-1]["text"] == text


def test_proactive_check_is_never_metered(service, monkeypatch):
    _litellm(monkeypatch, _reply(SimpleNamespace(prompt_tokens=5, completion_tokens=5,
                                                 cost=0.01), text="hey"))
    monkeypatch.setattr(config, "UTILITY_MODEL", "openrouter/openai/gpt-4o-mini")
    assert agent.proactive_check(user_id="u1") == "hey"
    assert service.records == []


def test_background_task_meters_itself_and_stops_out_of_balance(service, monkeypatch):
    service.reply = {"allowed": False, "balance": 0,
                     "payment_url": "https://pay.example.com/s/2"}
    got, done = [], threading.Event()
    monkeypatch.setattr(tasks, "_notify", lambda uid, text: (got.append(text), done.set()))
    monkeypatch.setattr(tasks, "_tasks", {})
    replies = iter([_tool_call()])

    def model(*a, **k):
        billing.ensure_can_spend()
        billing.record(cost_usd=1.0, cost_source="provider", prompt_tokens=1,
                       completion_tokens=1, model="m")
        return next(replies)

    monkeypatch.setattr(llm, "complete", model)
    monkeypatch.setattr(agent, "_execute", lambda *a, **k: ("nothing", False))

    from paulus import automode
    task, _ = tasks.start("u1", "dig", "dig deep", review=automode.ReviewContext(request="x"))
    assert done.wait(5)

    assert service.records[0]["kind"] == "task"
    assert task.status == "stopped"
    text, url = billing.split_pay_link(got[0])
    assert f"#{task.id} stopped" in text and billing.INSUFFICIENT_BALANCE_MESSAGE in text
    assert url == "https://pay.example.com/s/2"
    # Memory keeps the clean text, never the marker.
    assert "PAULUS_PAY_LINK" not in memory.recent_episodes(user_id="u1")[-1]["text"]
