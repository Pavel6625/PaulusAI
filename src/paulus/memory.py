"""Multi-store memory.

Stores:
  - episodic: append-only log of what happened (episodic.jsonl)
  - semantic: durable facts (canonical in facts.json, rendered to a readable,
    editable semantic.md). Retrieval is SEMANTIC via embeddings (vectorstore.py)
    with an automatic keyword fallback if Chroma isn't installed.

The vector index is derived; facts.json stays the source of truth, so memory
remains inspectable and the index can always be rebuilt from it.
"""
import datetime
import json
import re
import uuid

from . import config, vectorstore

# --------------------------------------------------------------------------- #
# Episodic                                                                     #
# --------------------------------------------------------------------------- #

def log_episode(role, text, trust="trusted"):
    config.ensure_dirs()
    rec = {
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "role": role, "trust": trust, "text": text,
    }
    with open(config.EPISODIC_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def recent_episodes(n=None):
    n = n or config.RECENT_EPISODES
    if not config.EPISODIC_LOG.exists():
        return []
    lines = config.EPISODIC_LOG.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines[-n:]]


# --------------------------------------------------------------------------- #
# Semantic (facts)                                                             #
# --------------------------------------------------------------------------- #

def _load_facts():
    if config.FACTS_FILE.exists():
        return json.loads(config.FACTS_FILE.read_text(encoding="utf-8"))
    return []


def _save_facts(facts):
    config.ensure_dirs()
    config.FACTS_FILE.write_text(json.dumps(facts, indent=2), encoding="utf-8")
    _render_semantic_md(facts)


def _render_semantic_md(facts):
    lines = ["# Semantic memory", "",
             "_Auto-rendered from facts.json. What the agent currently believes._", ""]
    for f in sorted(facts, key=lambda x: -x.get("salience", 0)):
        prov = ", ".join(f.get("provenance", [])) or "—"
        lines.append(
            f"- **{f['fact']}**  "
            f"_(confidence {f.get('confidence', 0.5):.2f}, "
            f"salience {f.get('salience', 1.0):.2f}, source: {prov})_"
        )
    config.SEMANTIC_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _index(fact):
    if vectorstore.AVAILABLE:
        try:
            vectorstore.upsert(fact["id"], fact["fact"])
        except Exception as e:
            print(f"[memory] index upsert failed: {e}")


def add_fact(fact, confidence=0.7, provenance=None):
    facts = _load_facts()
    for f in facts:
        if f["fact"].strip().lower() == fact.strip().lower():
            f["salience"] = min(2.0, f.get("salience", 1.0) + 0.3)
            f["last_reinforced"] = datetime.datetime.now().isoformat(timespec="seconds")
            f["confidence"] = max(f.get("confidence", 0.5), confidence)
            _save_facts(facts)
            _index(f)
            return "reinforced existing fact"
    rec = {
        "id": uuid.uuid4().hex[:12],
        "fact": fact,
        "confidence": confidence,
        "salience": 1.0,
        "provenance": provenance or ["conversation"],
        "last_reinforced": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    facts.append(rec)
    _save_facts(facts)
    _index(rec)
    return "stored new fact"


# ---- Retrieval: semantic first, keyword fallback -------------------------- #

_WORD = re.compile(r"[a-z0-9]+")


def _keywords(text):
    return set(_WORD.findall(text.lower()))


def _keyword_search(query, facts, k):
    q = _keywords(query)
    scored = []
    for f in facts:
        overlap = len(q & _keywords(f["fact"]))
        if overlap:
            scored.append((overlap + 0.5 * f.get("salience", 1.0), f))
    scored.sort(key=lambda x: -x[0])
    return [f for _, f in scored[:k]]


def _ensure_index(facts):
    """Lazily (re)build the vector index from canonical facts if out of sync."""
    if not vectorstore.AVAILABLE:
        return
    try:
        if vectorstore.count() < len(facts):
            for f in facts:
                vectorstore.upsert(f["id"], f["fact"])
    except Exception as e:
        print(f"[memory] reindex failed: {e}")


def search_facts(query, k=None):
    k = k or config.TOP_FACTS
    facts = _load_facts()
    if not facts:
        return []
    by_id = {f["id"]: f for f in facts if "id" in f}
    if vectorstore.AVAILABLE:
        _ensure_index(facts)
        try:
            ids = vectorstore.query(query, k)
            hits = [by_id[i] for i in ids if i in by_id]
            if hits:
                return hits
        except Exception as e:
            print(f"[memory] semantic query failed, falling back: {e}")
    return _keyword_search(query, facts, k)


# --------------------------------------------------------------------------- #
# Consolidation                                                                #
# --------------------------------------------------------------------------- #

def decay():
    facts = _load_facts()
    for f in facts:
        f["salience"] = round(f.get("salience", 1.0) * config.DECAY_PER_SLEEP, 3)
    _save_facts(facts)
