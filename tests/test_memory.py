import json
from types import SimpleNamespace

import pytest

from paulus import config, memory, vectorstore


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


def test_decay_evicts_low_salience(monkeypatch):
    monkeypatch.setattr(config, "SALIENCE_FLOOR", 0.95)
    memory.add_fact("Trivial passing remark")
    assert memory.decay() == 1
    assert memory._load_facts() == []
    assert memory.search_facts("trivial remark") == []


def test_decay_keeps_reinforced_fact(monkeypatch):
    monkeypatch.setattr(config, "SALIENCE_FLOOR", 0.95)
    memory.add_fact("Owner likes tea")
    memory.add_fact("owner likes tea")  # reinforce: salience 1.0 -> 1.3
    assert memory.decay() == 0
    assert any("tea" in f["fact"].lower() for f in memory._load_facts())


def test_decay_deindexes_evicted(monkeypatch):
    monkeypatch.setattr(config, "SALIENCE_FLOOR", 0.95)
    memory.add_fact("Ephemeral detail")          # indexed via keyword fallback (no-op)
    fid = memory._load_facts()[0]["id"]
    deleted = []
    monkeypatch.setattr(vectorstore, "AVAILABLE", True)
    monkeypatch.setattr(vectorstore, "delete",
                        lambda ids, user_id=None: deleted.extend(ids))
    assert memory.decay() == 1
    assert deleted == [fid]


def test_eviction_disabled_when_floor_zero(monkeypatch):
    monkeypatch.setattr(config, "SALIENCE_FLOOR", 0.0)
    memory.add_fact("Keep me forever")
    for _ in range(200):
        memory.decay()
    assert len(memory._load_facts()) == 1


def _stub_semantic(monkeypatch, verdict):
    """Enable the semantic-reconcile path without touching Chroma or the LLM:
    candidates = all current facts, index writes are no-ops, LLM returns verdict."""
    monkeypatch.setattr(vectorstore, "AVAILABLE", True)
    monkeypatch.setattr(memory, "search_facts",
                        lambda q, k=None, user_id=None: memory._load_facts(user_id))
    monkeypatch.setattr(memory, "_index", lambda *a, **k: None)
    monkeypatch.setattr(memory, "_llm_reconcile", lambda new, cands: verdict)


def test_semantic_dedup_reinforces_paraphrase(monkeypatch):
    memory.add_fact("The owner lives in Boston.")
    fid = memory._load_facts()[0]["id"]
    _stub_semantic(monkeypatch, {"action": "duplicate", "id": fid})
    assert memory.add_fact("The owner is based in Boston.") == "reinforced existing fact"
    facts = memory._load_facts()
    assert len(facts) == 1 and facts[0]["salience"] > 1.0


def test_contradiction_supersedes(monkeypatch):
    memory.add_fact("The owner lives in Boston.")
    fid = memory._load_facts()[0]["id"]
    _stub_semantic(monkeypatch, {"action": "contradiction", "id": fid})
    assert "supersede" in memory.add_fact("The owner lives in Seattle.")
    facts = memory._load_facts()
    assert len(facts) == 1
    assert facts[0]["id"] == fid                       # same record, updated in place
    assert "Seattle" in facts[0]["fact"] and "Boston" not in facts[0]["fact"]


def test_novel_fact_stored(monkeypatch):
    memory.add_fact("The owner lives in Boston.")
    _stub_semantic(monkeypatch, {"action": "novel", "id": None})
    assert memory.add_fact("The owner has a dog named Rex.") == "stored new fact"
    assert len(memory._load_facts()) == 2


def test_reconcile_failure_falls_back_to_store(monkeypatch):
    memory.add_fact("Fact one.")
    _stub_semantic(monkeypatch, {})  # verdict unused; we override with a raiser
    def boom(new, cands):
        raise RuntimeError("llm unavailable")
    monkeypatch.setattr(memory, "_llm_reconcile", boom)
    assert memory.add_fact("A wholly different second fact.") == "stored new fact"
    assert len(memory._load_facts()) == 2


def test_no_semantic_reconcile_without_vectorstore():
    # vectorstore.AVAILABLE is False in tests: paraphrases are stored separately.
    memory.add_fact("The owner enjoys hiking.")
    memory.add_fact("The owner likes to hike.")
    assert len(memory._load_facts()) == 2


def test_episodic_log_stays_bounded(monkeypatch):
    monkeypatch.setattr(config, "MAX_EPISODES", 50)
    for i in range(500):
        memory.log_episode("owner", f"msg {i}")
    lines = config.EPISODIC_LOG.read_text(encoding="utf-8").splitlines()
    # Bounded by the cap plus the slack margin that delays each rewrite.
    assert 50 <= len(lines) <= 50 + max(64, 50 // 5)
    # Trimming keeps the *most recent* entries and they remain parseable.
    eps = memory.recent_episodes(5)
    assert [e["text"] for e in eps] == [f"msg {i}" for i in range(495, 500)]


def test_recent_episodes_tail_read_order():
    for i in range(20):
        memory.log_episode("owner", f"e{i}")
    eps = memory.recent_episodes(3)
    assert [e["text"] for e in eps] == ["e17", "e18", "e19"]


def test_tail_lines_handles_fewer_than_requested():
    memory.log_episode("owner", "only one")
    assert len(memory.recent_episodes(10)) == 1


def test_trim_disabled_when_cap_zero(monkeypatch):
    monkeypatch.setattr(config, "MAX_EPISODES", 0)
    for i in range(120):
        memory.log_episode("owner", f"x{i}")
    lines = config.EPISODIC_LOG.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 120


# --- the fact-is-a-string invariant ----------------------------------------
# Every reader assumes it and none check, so a violation used to raise
# AttributeError out of _keyword_search — which runs on every turn.

@pytest.mark.parametrize("bad", [
    {"fact": "nested"},          # what a model emits when told facts are objects
    ["a list"],
    42,
    None,
    "",
    "   ",
])
def test_add_fact_rejects_anything_that_is_not_a_real_string(bad):
    with pytest.raises(ValueError):
        memory.add_fact(bad)


def test_add_fact_rejection_stores_nothing():
    with pytest.raises(ValueError):
        memory.add_fact({"fact": "nested"})
    assert memory._load_facts() == []


def _write_facts(records, user_id=None):
    path = memory._facts_file(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records), encoding="utf-8")


def test_poisoned_records_are_quarantined_on_read():
    # Data written before add_fact validated; the good record must survive.
    _write_facts([
        {"id": "1", "fact": {"fact": "nested"}, "salience": 1.0},
        {"id": "2", "fact": "a real fact", "salience": 1.0},
        {"id": "3", "fact": "", "salience": 1.0},
    ])
    assert [f["fact"] for f in memory._load_facts()] == ["a real fact"]


def test_search_survives_a_poisoned_facts_file():
    # The regression: one bad record used to make every search_facts call raise,
    # and _build_system calls it on every turn.
    _write_facts([
        {"id": "1", "fact": {"fact": "nested"}, "salience": 1.0},
        {"id": "2", "fact": "the owner likes coffee", "salience": 1.0},
    ])
    hits = memory.search_facts("coffee")
    assert [h["fact"] for h in hits] == ["the owner likes coffee"]


def test_quarantine_survives_a_facts_file_that_is_not_a_list():
    _write_facts({"not": "a list"})
    assert memory._load_facts() == []


def test_reconcile_recovers_a_wrapped_verdict(monkeypatch):
    # Production reality: every reconcile call failed on strict json.loads with
    # "Expecting value: line 1 column 1" -- the model puts something before the
    # JSON. _reconcile swallows that by storing the fact as new, so dedup never
    # ran and paraphrases accumulated. Recovery has to happen here.
    verdict = '```json\n{"action": "duplicate", "id": "abc"}\n```'
    monkeypatch.setattr(memory.llm, "complete", lambda *a, **k: SimpleNamespace(
        content=[SimpleNamespace(type="text", text=verdict)]))
    assert memory._llm_reconcile("x", [{"id": "abc", "fact": "y"}]) == {
        "action": "duplicate", "id": "abc",
    }


def test_reconcile_error_names_what_the_model_said(monkeypatch):
    # The old failure logged only "Expecting value: line 1 column 1", which says
    # nothing about the cause. The excerpt is what makes it diagnosable.
    monkeypatch.setattr(memory.llm, "complete", lambda *a, **k: SimpleNamespace(
        content=[SimpleNamespace(type="text", text="I cannot help with that.")]))
    with pytest.raises(ValueError, match="I cannot help"):
        memory._llm_reconcile("x", [{"id": "abc", "fact": "y"}])


def test_next_save_persists_the_cleaned_list():
    _write_facts([
        {"id": "1", "fact": {"fact": "nested"}, "salience": 1.0},
        {"id": "2", "fact": "keep me", "salience": 1.0},
    ])
    memory.add_fact("a new fact")     # any mutation rewrites the file
    on_disk = json.loads(memory._facts_file().read_text(encoding="utf-8"))
    assert sorted(f["fact"] for f in on_disk) == ["a new fact", "keep me"]
