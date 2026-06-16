from paulus import memory, vectorstore


def test_keyword_fallback_active_in_tests():
    # Tests never call vectorstore.init(), so retrieval uses keyword fallback.
    assert vectorstore.AVAILABLE is False


def test_add_and_search_fact():
    memory.add_fact("The owner's name is Pavel.")
    hits = memory.search_facts("what is the owner name")
    assert any("Pavel" in h["fact"] for h in hits)


def test_duplicate_fact_reinforces():
    assert memory.add_fact("Likes coffee") == "stored new fact"
    assert memory.add_fact("likes coffee") == "reinforced existing fact"
    coffee = [f for f in memory._load_facts() if f["fact"].lower() == "likes coffee"]
    assert len(coffee) == 1 and coffee[0]["salience"] > 1.0


def test_episode_logging_roundtrip():
    memory.log_episode("owner", "hello")
    memory.log_episode("agent", "hi there")
    eps = memory.recent_episodes()
    assert eps[-1]["text"] == "hi there" and eps[-1]["role"] == "agent"


def test_decay_reduces_salience():
    memory.add_fact("Some fact")
    before = memory._load_facts()[0]["salience"]
    memory.decay()
    assert memory._load_facts()[0]["salience"] < before
