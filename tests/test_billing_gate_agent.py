"""agent.py's billing wiring: respond() is pay-gated (the user asked for that
turn); proactive_check() is NOT (a nudge is bot-initiated and must never debit
the user). Kept separate from test_billing.py since it exercises agent.py's
wiring, not billing.py's HTTP logic."""
from paulus import agent, billing, config, llm, memory


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


def test_proactive_check_never_consults_the_pay_gate(monkeypatch):
    """A nudge is unsolicited, so it must not debit the user. proactive_check
    must never call billing.gate() — not even to read balance — and must run
    even when the gate would block a paid reply."""
    def _no_gate(uid):
        raise AssertionError("proactive_check must not consult the pay gate")
    monkeypatch.setattr(billing, "gate", _no_gate)
    monkeypatch.setattr(llm, "complete",
                        lambda *a, **k: llm._Response(
                            content=[llm._TextBlock(text="hey, thinking of you")]))

    reply = agent.proactive_check(user_id="u1")

    assert reply == "hey, thinking of you"


def test_proactive_check_uses_the_utility_model(monkeypatch):
    """The idle nudge is an unprompted internal job: it must run on the pinned
    utility model, not default to the flagship CORE_MODEL, so an operator can
    make the idle loop free by pointing DP_UTILITY_MODEL at a cheap model."""
    monkeypatch.setattr(config, "UTILITY_MODEL", "cheap/model")

    seen = {}

    def _capture(system, messages, tools=None, model=None):
        seen["model"] = model
        return llm._Response(content=[llm._TextBlock(text="[SILENT]")])

    monkeypatch.setattr(llm, "complete", _capture)

    agent.proactive_check(user_id="u1")

    assert seen["model"] == "cheap/model"
