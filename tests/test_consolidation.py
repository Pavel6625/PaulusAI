"""agent.sleep()'s handling of what the model actually returns.

Kept separate from test_memory.py because it exercises agent.py's wiring, not
memory.py's storage. Every case here is a real thing a small model does when
told to reply with strict JSON — the production model
(ollama_chat/gemma4:31b-cloud) is documented to append stray content after a
valid object, which is the `trailing` case below.
"""
from types import SimpleNamespace

import pytest

from paulus import agent, llm, memory, skills


@pytest.fixture
def replies(monkeypatch):
    """Make the model return a given raw string, and give sleep() something to
    consolidate."""
    memory.log_episode("owner", "my sister is called Ana", user_id="u1")

    def _set(raw):
        monkeypatch.setattr(llm.litellm, "completion", lambda **kw: SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="stop",
                                     message=SimpleNamespace(content=raw,
                                                             tool_calls=None))]))
    return _set


def _facts(user_id="u1"):
    return [f["fact"] for f in memory._load_facts(user_id)]


GOOD = '{"facts": ["owner has a sister named Ana"], "skills": []}'


@pytest.mark.parametrize("raw", [
    GOOD,
    f"```json\n{GOOD}\n```",
    f"Here is the JSON you asked for:\n{GOOD}",
    f"{GOOD}\nHope that helps!",
])
def test_consolidation_recovers_from_ordinary_model_sloppiness(replies, raw):
    replies(raw)
    said = agent.sleep(user_id="u1")
    assert _facts() == ["owner has a sister named Ana"]
    assert "1 fact(s)" in said and "failed" not in said


def test_one_bad_entry_does_not_drop_the_good_ones(replies):
    # The loop used to abort mid-flight: 'fact four' was silently discarded and
    # the result still read like a success.
    replies('{"facts": ["fact one", "fact two", {"oops": 1}, "fact four"],'
            ' "skills": []}')
    said = agent.sleep(user_id="u1")
    assert sorted(_facts()) == ["fact four", "fact one", "fact two"]
    assert "3 fact(s)" in said and "dropped 1 malformed" in said


def test_a_non_string_fact_never_reaches_storage(replies):
    replies('{"facts": [{"fact": "nested"}], "skills": []}')
    said = agent.sleep(user_id="u1")
    assert _facts() == []
    assert "dropped 1 malformed" in said


def test_the_agent_still_replies_after_a_malformed_consolidation(replies):
    # The real damage: a stored non-string used to make every later respond()
    # raise AttributeError out of _keyword_search, so the owner got a generic
    # error forever.
    replies('{"facts": [{"fact": "nested"}], "skills": []}')
    agent.sleep(user_id="u1")
    replies("hello again!")
    assert agent.respond("hey, how are you?", user_id="u1") == "hello again!"


def test_unparseable_reply_is_reported_not_swallowed(replies):
    replies("I'm afraid I can't do that, Dave.")
    said = agent.sleep(user_id="u1")
    # Must not read like "there was nothing worth consolidating".
    assert "failed" in said
    assert "0 fact(s)" not in said


def test_top_level_list_is_reported_as_a_failure(replies):
    replies('[{"facts": ["a"]}]')
    assert "failed" in agent.sleep(user_id="u1")


def test_decay_still_runs_when_the_reply_is_unusable(replies):
    replies("garbage")
    assert "memory decayed" in agent.sleep(user_id="u1")


def test_skill_missing_a_key_is_dropped_without_losing_facts(replies):
    replies('{"facts": ["a good fact"],'
            ' "skills": [{"name": "n", "when_to_use": "w"}]}')
    said = agent.sleep(user_id="u1")
    assert _facts() == ["a good fact"]
    assert "1 fact(s)" in said and "dropped 1 malformed" in said


def test_a_well_formed_skill_is_still_proposed(replies):
    replies('{"facts": [], "skills": [{"name": "deploy", "when_to_use": "w",'
            ' "steps": "s"}]}')
    said = agent.sleep(user_id="u1")
    assert "1 skill(s)" in said
    assert any(s["name"] == "deploy" and s["status"] == "unverified"
               for s in skills._load())


def test_nothing_to_consolidate_is_distinct_from_a_failure():
    assert agent.sleep(user_id="u_silent") == "Nothing to consolidate yet."
