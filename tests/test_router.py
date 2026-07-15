"""Router tests.

The semantic path needs the optional `vectors` extra, so those tests skip when
chromadb is absent. Everything that must hold WITHOUT embeddings — the
fail-safe to the core model, tier resolution, the structural pass — is tested
unconditionally, because that is the path a minimal install actually runs.
"""
import importlib.util

import pytest

from paulus import config, router

# Marks the semantic tests only. A module-level importorskip would skip the
# whole file, taking the fail-safe tests with it — and those exist precisely to
# cover the install that has no embeddings.
needs_vectors = pytest.mark.skipif(
    importlib.util.find_spec("chromadb") is None,
    reason="needs the `vectors` extra",
)


@pytest.fixture
def tiers(monkeypatch):
    """A fully configured three-tier setup with routing enabled."""
    monkeypatch.setattr(config, "ROUTING", "semantic")
    monkeypatch.setattr(config, "CORE_MODEL", "ollama_chat/gemma4:31b-cloud")
    monkeypatch.setattr(config, "MID_MODEL", "openrouter/openai/gpt-4o-mini")
    monkeypatch.setattr(config, "TOP_MODEL", "openrouter/anthropic/claude-opus-4-8")


# --- tier resolution --------------------------------------------------------

def test_tiers_collapse_downward_when_unset(monkeypatch):
    monkeypatch.setattr(config, "CORE_MODEL", "core/m")
    monkeypatch.setattr(config, "MID_MODEL", "")
    monkeypatch.setattr(config, "TOP_MODEL", "")
    assert config.tier_model("low") == "core/m"
    assert config.tier_model("mid") == "core/m"
    assert config.tier_model("top") == "core/m"

    # Only a top tier configured: mid must collapse to core, top must be used.
    monkeypatch.setattr(config, "TOP_MODEL", "top/m")
    assert config.tier_model("mid") == "core/m"
    assert config.tier_model("top") == "top/m"


def test_utility_model_defaults_to_core(monkeypatch):
    monkeypatch.setattr(config, "CORE_MODEL", "core/m")
    monkeypatch.setattr(config, "UTILITY_MODEL", "")
    assert config.utility_model() == "core/m"
    monkeypatch.setattr(config, "UTILITY_MODEL", "util/m")
    assert config.utility_model() == "util/m"


# --- fail-safe --------------------------------------------------------------

def test_routing_off_pins_everything_to_core(monkeypatch):
    monkeypatch.setattr(config, "ROUTING", "off")
    monkeypatch.setattr(config, "CORE_MODEL", "core/m")
    monkeypatch.setattr(config, "TOP_MODEL", "top/m")
    # Even an obvious escalation must stay on core while routing is off.
    model, tier, _ = router.route("Traceback (most recent call last):")
    assert (model, tier) == ("core/m", "low")


def test_unavailable_embeddings_fall_back_to_low(monkeypatch, tiers):
    monkeypatch.setattr(router, "AVAILABLE", False)
    model, tier, reason = router.route("what do you think about that")
    assert tier == "low"
    assert model == "ollama_chat/gemma4:31b-cloud"
    assert "unavailable" in reason


def test_embedding_error_falls_back_to_low(monkeypatch, tiers):
    monkeypatch.setattr(router, "AVAILABLE", True)
    def _boom(text, user_id=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(router, "_similarities", _boom)
    _model, tier, reason = router.route("hello")
    assert tier == "low"
    assert "boom" in reason


# --- structural pass (no embeddings required) -------------------------------

@pytest.mark.parametrize("text", [
    "Traceback (most recent call last):\n  File \"a.py\", line 3, in <module>",
    "here's my code:\n```python\ndef f(): pass\n```",
    "it raises KeyError: 'user_id' every time",
    "why do I get a TypeError here",
])
def test_structural_signals_escalate_to_top_without_embeddings(monkeypatch, tiers, text):
    # AVAILABLE=False proves these fire before (and independently of) the
    # semantic pass — which is the point: embeddings are worst at exactly these.
    monkeypatch.setattr(router, "AVAILABLE", False)
    _model, tier, reason = router.route(text)
    assert tier == "top"
    assert "structural" in reason


def test_documents_escalate_to_mid_on_presence_not_content(monkeypatch, tiers):
    monkeypatch.setattr(router, "AVAILABLE", False)
    _model, tier, reason = router.route("have a look", has_documents=True)
    assert tier == "mid"
    assert "document" in reason


def test_long_input_escalates_to_mid(monkeypatch, tiers):
    monkeypatch.setattr(router, "AVAILABLE", False)
    _model, tier, _ = router.route("word " * 200)
    assert tier == "mid"


def test_chitchat_stays_low_without_embeddings(monkeypatch, tiers):
    monkeypatch.setattr(router, "AVAILABLE", False)
    _model, tier, _ = router.route("hey, how was your day?")
    assert tier == "low"


# --- vision -----------------------------------------------------------------

def test_image_escalates_to_weakest_sighted_tier(monkeypatch, tiers):
    sighted = {"openrouter/openai/gpt-4o-mini", "openrouter/anthropic/claude-opus-4-8"}
    monkeypatch.setattr("paulus.llm.supports_vision", lambda m: m in sighted)
    model, tier, reason = router.route("what is this?", has_images=True)
    # The blind core model must be skipped, but the flagship must not be spent
    # when the cheaper sighted tier can do it.
    assert tier == "mid"
    assert model == "openrouter/openai/gpt-4o-mini"
    assert "image" in reason


def test_vision_model_matches_what_an_image_turn_would_route_to(monkeypatch, tiers):
    sighted = {"openrouter/anthropic/claude-opus-4-8"}
    monkeypatch.setattr("paulus.llm.supports_vision", lambda m: m in sighted)
    routed, _tier, _ = router.route("look", has_images=True)
    # The gateway's up-front check must agree with the real decision, or it
    # refuses images the agent could actually handle.
    assert router.vision_model() == routed


def test_image_beats_structural_signals(monkeypatch, tiers):
    monkeypatch.setattr("paulus.llm.supports_vision", lambda m: True)
    _model, tier, reason = router.route("Traceback (most recent call last):",
                                        has_images=True)
    # Vision is a hard constraint; a blind top tier would fail the call outright.
    assert "image" in reason and tier == "low"


# --- semantic pass ----------------------------------------------------------

@needs_vectors
@pytest.mark.parametrize("text,expected", [
    ("hey", "low"),
    ("goodnight, talk tomorrow", "low"),
    ("i had a rough day at work", "low"),
    ("what's the difference between rust and go", "mid"),
    ("write a formal complaint letter to my bank", "mid"),
    ("can you review this code for bugs", "top"),
    ("why is this function returning None instead of a list", "top"),
])
def test_semantic_classification(monkeypatch, tiers, text, expected):
    monkeypatch.setattr(config, "ROUTE_MARGIN", 0.08)
    router.init()
    tier, reason = router.classify(text)
    assert tier == expected, f"{text!r} -> {tier} ({reason})"


@needs_vectors
def test_thin_margin_falls_back_to_the_weaker_tier(monkeypatch, tiers):
    router.init()
    monkeypatch.setattr(router, "_similarities",
                        lambda text, user_id=None: {"low": 0.40, "mid": 0.42,
                                                    "top": 0.10})
    monkeypatch.setattr(config, "ROUTE_MARGIN", 0.08)
    tier, reason = router.classify("ambiguous")
    # mid wins on score but only by 0.02 — an unconfident escalation must not
    # spend a stronger model, so the weaker of the two candidates is taken.
    assert tier == "low"
    assert "low margin" in reason


@needs_vectors
def test_wide_margin_is_trusted(monkeypatch, tiers):
    router.init()
    monkeypatch.setattr(router, "_similarities",
                        lambda text, user_id=None: {"low": 0.10, "mid": 0.20,
                                                    "top": 0.60})
    monkeypatch.setattr(config, "ROUTE_MARGIN", 0.08)
    tier, _reason = router.classify("clearly hard")
    assert tier == "top"


# --- decision log -----------------------------------------------------------

@pytest.fixture
def learning(monkeypatch, tiers):
    monkeypatch.setattr(config, "ROUTE_LEARNING", True)
    monkeypatch.setattr(config, "MAX_LEARNED_EXEMPLARS", 40)
    router._last_turn.clear()
    router._learned_cache.clear()
    return None


def test_decision_log_records_tier_and_reason(learning):
    turn = router.log_decision("hey there", "low", "semantic low (0.9)",
                               user_id="u1")
    log = router.read_log("u1")
    assert log[0]["type"] == "decision"
    assert log[0]["id"] == turn
    assert (log[0]["tier"], log[0]["text"]) == ("low", "hey there")


def test_log_is_isolated_per_user(learning):
    router.log_decision("user one's private words", "low", "semantic low",
                        user_id="u1")
    assert router.read_log("u2") == []


def test_outcome_ignores_introspective_tools(learning):
    turn = router.log_decision("what did i tell you about my sister", "low",
                               "semantic low", user_id="u1")
    # recall/find_skill fire on ordinary small talk (the system prompt tells the
    # model to), so they must not count as evidence the turn was hard.
    router.log_outcome(turn, ["recall", "find_skill"], user_id="u1")
    assert [r for r in router.read_log("u1") if r["type"] == "outcome"] == []


def test_outcome_records_tools_that_reach_outside(learning):
    turn = router.log_decision("look this up", "low", "semantic low", user_id="u1")
    router.log_outcome(turn, ["recall", "web_search"], user_id="u1")
    out = [r for r in router.read_log("u1") if r["type"] == "outcome"]
    assert out[0]["effort_tools"] == ["web_search"]


def test_complaint_attaches_to_the_previous_turn(learning):
    first = router.log_decision("explain the thing", "low", "semantic low",
                                user_id="u1")
    # The owner reads the reply, then complains on the NEXT message.
    router.log_complaint(user_id="u1")
    complaints = [r for r in router.read_log("u1") if r["type"] == "complaint"]
    assert complaints[0]["ref"] == first


def test_complaint_with_no_prior_turn_is_a_noop(learning):
    router.log_complaint(user_id="u_fresh")
    assert router.read_log("u_fresh") == []


def test_log_is_trimmed_to_the_cap(monkeypatch, learning):
    monkeypatch.setattr(config, "MAX_ROUTE_LOG", 5)
    for i in range(12):
        router.log_decision(f"message {i}", "low", "semantic low", user_id="u1")
    assert len(router.read_log("u1")) == 5


# --- learning ---------------------------------------------------------------

def test_learning_off_promotes_nothing(monkeypatch, learning):
    monkeypatch.setattr(config, "ROUTE_LEARNING", False)
    turn = router.log_decision("check the build", "low", "semantic low",
                               user_id="u1")
    router.log_outcome(turn, ["run_command"], user_id="u1")
    assert router.learn(user_id="u1") == 0
    assert router._load_learned("u1") == {}


def test_effort_promotes_a_low_route_one_step_to_mid(learning):
    turn = router.log_decision("check the build", "low", "semantic low (0.4)",
                               user_id="u1")
    router.log_outcome(turn, ["run_command"], user_id="u1")
    assert router.learn(user_id="u1") == 1
    learned = router._load_learned("u1")
    # One step, never straight to top: the evidence says "more than it got",
    # not "the flagship".
    assert learned["mid"] == ["check the build"]
    assert "top" not in learned


def test_complaint_promotes_mid_to_top(learning):
    router.log_decision("why does this fail", "mid", "semantic mid (0.4)",
                        user_id="u1")
    router.log_complaint(user_id="u1")   # attaches to the turn just logged
    router.learn(user_id="u1")
    assert router._load_learned("u1")["top"] == ["why does this fail"]


def test_structural_decisions_are_never_learned_from(learning):
    # A structural or vision decision was already made on hard evidence; adding
    # its text as an exemplar teaches the semantic pass nothing.
    turn = router.log_decision("Traceback (most recent call last):", "top",
                               "structural: code/traceback", user_id="u1")
    router.log_outcome(turn, ["run_command"], user_id="u1")
    assert router.learn(user_id="u1") == 0


def test_top_routes_are_not_promoted(learning):
    turn = router.log_decision("hard thing", "top", "semantic top (0.9)",
                               user_id="u1")
    router.log_outcome(turn, ["run_command"], user_id="u1")
    assert router.learn(user_id="u1") == 0


def test_learning_is_idempotent(learning):
    turn = router.log_decision("check the build", "low", "semantic low",
                               user_id="u1")
    router.log_outcome(turn, ["run_command"], user_id="u1")
    assert router.learn(user_id="u1") == 1
    # Running consolidation again must not re-add the same phrase.
    assert router.learn(user_id="u1") == 0
    assert router._load_learned("u1")["mid"] == ["check the build"]


def test_learned_exemplars_are_capped_evicting_oldest(monkeypatch, learning):
    monkeypatch.setattr(config, "MAX_LEARNED_EXEMPLARS", 3)
    for i in range(6):
        turn = router.log_decision(f"phrase {i}", "low", "semantic low",
                                   user_id="u1")
        router.log_outcome(turn, ["web_search"], user_id="u1")
        router.learn(user_id="u1")
    learned = router._load_learned("u1")["mid"]
    assert len(learned) == 3
    assert learned == ["phrase 3", "phrase 4", "phrase 5"]


def test_learned_exemplars_are_isolated_per_user(learning):
    turn = router.log_decision("u1's private phrasing", "low", "semantic low",
                               user_id="u1")
    router.log_outcome(turn, ["web_search"], user_id="u1")
    router.learn(user_id="u1")
    assert router._load_learned("u2") == {}


@needs_vectors
def test_a_learned_phrase_changes_the_next_routing_decision(learning):
    router.init()
    phrase = "sort that out for me"
    # Cold: nothing about this phrasing looks like work, so it sits on the floor.
    assert router.classify(phrase, user_id="u1")[0] == "low"

    turn = router.log_decision(phrase, "low", "semantic low", user_id="u1")
    router.log_outcome(turn, ["run_command"], user_id="u1")
    router.learn(user_id="u1")

    # Learned: the same phrasing now escalates, and only for this user.
    assert router.classify(phrase, user_id="u1")[0] == "mid"
    assert router.classify(phrase, user_id="u2")[0] == "low"
