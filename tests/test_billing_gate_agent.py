"""Confirms the pay gate short-circuits agent.respond()/proactive_check()
before the LLM (or memory) is ever touched. Kept separate from test_billing.py
since it exercises agent.py's wiring, not billing.py's HTTP logic."""
from paulus import agent, billing, llm, memory


def _forbid_llm(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("LLM must not be called when the pay gate blocks")
    monkeypatch.setattr(llm, "complete", _boom)
    monkeypatch.setattr(llm, "stream", _boom)


def test_respond_returns_block_message_without_calling_llm(monkeypatch):
    _forbid_llm(monkeypatch)
    monkeypatch.setattr(billing, "gate", lambda uid: (False, "You're out of balance."))

    reply = agent.respond("hello", user_id="u1")

    assert reply == "You're out of balance."
    assert memory.recent_episodes(user_id="u1") == []


def test_proactive_check_stays_silent_without_calling_llm(monkeypatch):
    _forbid_llm(monkeypatch)
    monkeypatch.setattr(billing, "gate", lambda uid: (False, "You're out of balance."))

    reply = agent.proactive_check(user_id="u1")

    assert reply in agent._SILENCE_TOKENS
