"""Multi-store memory.

Stores:
  - episodic: append-only log of what happened (episodic.jsonl)
  - semantic: durable facts (canonical in facts.json, rendered to a readable,
    editable semantic.md). Retrieval is SEMANTIC via embeddings (vectorstore.py)
    with an automatic keyword fallback if Chroma isn't installed.

The vector index is derived; facts.json stays the source of truth, so memory
remains inspectable and the index can always be rebuilt from it.

User isolation: all public functions accept an optional *user_id* argument.
When provided, data is stored under ``MEMORY_DIR/users/<safe_uid>/``, keeping
each user's episodic log and facts file separate. CLI (single-user) mode
passes ``user_id=None``, which falls back to the original global paths so
existing installations are unaffected.
"""
import datetime
import json
import re
import uuid
from pathlib import Path

from . import config, vectorstore

_SAFE_UID_RE = re.compile(r"[^a-zA-Z0-9_-]")


def _safe_uid(user_id: str) -> str:
    return _SAFE_UID_RE.sub("_", str(user_id))


def _user_dir(user_id) -> Path | None:
    if user_id is None:
        return None
    return config.MEMORY_DIR / "users" / _safe_uid(user_id)


def _episodic_log(user_id=None) -> Path:
    d = _user_dir(user_id)
    return d / "episodic.jsonl" if d else config.EPISODIC_LOG


def _facts_file(user_id=None) -> Path:
    d = _user_dir(user_id)
    return d / "facts.json" if d else config.FACTS_FILE


def _semantic_md(user_id=None) -> Path:
    d = _user_dir(user_id)
    return d / "semantic.md" if d else config.SEMANTIC_MD


def _ensure_user_dir(user_id):
    d = _user_dir(user_id)
    if d:
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Episodic                                                                     #
# --------------------------------------------------------------------------- #

def log_episode(role, text, trust="trusted", user_id=None):
    config.ensure_dirs()
    _ensure_user_dir(user_id)
    rec = {
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "role": role, "trust": trust, "text": text,
    }
    with open(_episodic_log(user_id), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def recent_episodes(n=None, user_id=None):
    n = n or config.RECENT_EPISODES
    path = _episodic_log(user_id)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines[-n:]]


# --------------------------------------------------------------------------- #
# Semantic (facts)                                                             #
# --------------------------------------------------------------------------- #

def _load_facts(user_id=None):
    path = _facts_file(user_id)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return []


def _save_facts(facts, user_id=None):
    config.ensure_dirs()
    _ensure_user_dir(user_id)
    path = _facts_file(user_id)
    path.write_text(json.dumps(facts, indent=2), encoding="utf-8")
    _render_semantic_md(facts, _semantic_md(user_id))


def _render_semantic_md(facts, path):
    lines = ["# Semantic memory", "",
             "_Auto-rendered from facts.json. What the agent currently believes._", ""]
    for f in sorted(facts, key=lambda x: -x.get("salience", 0)):
        prov = ", ".join(f.get("provenance", [])) or "—"
        lines.append(
            f"- **{f['fact']}**  "
            f"_(confidence {f.get('confidence', 0.5):.2f}, "
            f"salience {f.get('salience', 1.0):.2f}, source: {prov})_"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _index(fact, user_id=None):
    if vectorstore.AVAILABLE:
        try:
            vectorstore.upsert(fact["id"], fact["fact"], user_id=user_id)
        except Exception as e:
            print(f"[memory] index upsert failed: {e}")


def add_fact(fact, confidence=0.7, provenance=None, user_id=None):
    facts = _load_facts(user_id)
    for f in facts:
        if f["fact"].strip().lower() == fact.strip().lower():
            f["salience"] = min(2.0, f.get("salience", 1.0) + 0.3)
            f["last_reinforced"] = datetime.datetime.now().isoformat(timespec="seconds")
            f["confidence"] = max(f.get("confidence", 0.5), confidence)
            _save_facts(facts, user_id)
            _index(f, user_id)
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
    _save_facts(facts, user_id)
    _index(rec, user_id)
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


def _ensure_index(facts, user_id=None):
    """Lazily (re)build the vector index from canonical facts if out of sync."""
    if not vectorstore.AVAILABLE:
        return
    try:
        if vectorstore.count(user_id) < len(facts):
            for f in facts:
                vectorstore.upsert(f["id"], f["fact"], user_id=user_id)
    except Exception as e:
        print(f"[memory] reindex failed: {e}")


def search_facts(query, k=None, user_id=None):
    k = k or config.TOP_FACTS
    facts = _load_facts(user_id)
    if not facts:
        return []
    by_id = {f["id"]: f for f in facts if "id" in f}
    if vectorstore.AVAILABLE:
        _ensure_index(facts, user_id)
        try:
            ids = vectorstore.query(query, k, user_id=user_id)
            hits = [by_id[i] for i in ids if i in by_id]
            if hits:
                return hits
        except Exception as e:
            print(f"[memory] semantic query failed, falling back: {e}")
    return _keyword_search(query, facts, k)


# --------------------------------------------------------------------------- #
# Consolidation                                                                #
# --------------------------------------------------------------------------- #

def decay(user_id=None):
    facts = _load_facts(user_id)
    for f in facts:
        f["salience"] = round(f.get("salience", 1.0) * config.DECAY_PER_SLEEP, 3)
    _save_facts(facts, user_id)
