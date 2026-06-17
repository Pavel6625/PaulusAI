"""Semantic vector index over long-term facts.

Wraps Chroma. Chroma's default embedding function runs a small model locally
(no API key, nothing leaves the machine), which fits the privacy goal; swap in
an API-based embedder by passing embedding_function to get_or_create_collection.

facts.json remains the canonical, inspectable store — this is only the index.
If Chroma isn't installed, AVAILABLE stays False and memory.py falls back to
keyword retrieval so the agent still runs.

User isolation: every document is tagged with a ``user_id`` metadata field.
All reads are filtered to the requesting user so one user can never retrieve
another user's facts. CLI (single-user) mode uses the sentinel "__local__".
"""
from . import config

AVAILABLE = False
_collection = None

_LOCAL_USER = "__local__"


def _uid(user_id):
    return user_id if user_id is not None else _LOCAL_USER


def _get_collection():
    global _collection, AVAILABLE
    if _collection is not None:
        return _collection
    import chromadb
    client = chromadb.PersistentClient(path=str(config.VECTOR_DIR))
    try:
        _collection = client.get_or_create_collection(
            "semantic", configuration={"hnsw": {"space": "cosine"}}
        )
    except TypeError:
        # Older Chroma versions use a different config kwarg.
        _collection = client.get_or_create_collection("semantic")
    AVAILABLE = True
    return _collection


def init():
    """Try to bring the index up; on any failure, stay in fallback mode."""
    global AVAILABLE
    try:
        config.ensure_dirs()
        _get_collection()
    except Exception as e:
        AVAILABLE = False
        print(f"[vectorstore] embeddings unavailable, using keyword fallback: {e}")
    return AVAILABLE


def upsert(fact_id, text, user_id=None):
    """Index a fact under the given user. IDs are namespaced to prevent collisions."""
    col = _get_collection()
    uid = _uid(user_id)
    col.upsert(
        ids=[f"{uid}:{fact_id}"],
        documents=[text],
        metadatas=[{"user_id": uid}],
    )


def query(text, k, user_id=None):
    """Return fact IDs semantically closest to *text*, scoped to *user_id*."""
    col = _get_collection()
    uid = _uid(user_id)
    where = {"user_id": uid}
    user_doc_ids = col.get(where=where, include=[]).get("ids", [])
    if not user_doc_ids:
        return []
    n = min(k, len(user_doc_ids))
    res = col.query(query_texts=[text], n_results=n, where=where)
    raw_ids = res.get("ids", [[]])[0]
    prefix = f"{uid}:"
    return [rid[len(prefix):] if rid.startswith(prefix) else rid for rid in raw_ids]


def count(user_id=None):
    uid = _uid(user_id)
    col = _get_collection()
    return len(col.get(where={"user_id": uid}, include=[]).get("ids", []))


def delete(ids, user_id=None):
    """Remove specific facts from the index for *user_id*. IDs are the bare
    fact ids; they are namespaced here the same way upsert stores them."""
    if not ids:
        return
    uid = _uid(user_id)
    col = _get_collection()
    col.delete(ids=[f"{uid}:{fact_id}" for fact_id in ids])


def reset(user_id=None):
    uid = _uid(user_id)
    col = _get_collection()
    existing = col.get(where={"user_id": uid}, include=[]).get("ids", [])
    if existing:
        col.delete(ids=existing)
