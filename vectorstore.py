"""Semantic vector index over long-term facts.

Wraps Chroma. Chroma's default embedding function runs a small model locally
(no API key, nothing leaves the machine), which fits the privacy goal; swap in
an API-based embedder by passing embedding_function to get_or_create_collection.

facts.json remains the canonical, inspectable store — this is only the index.
If Chroma isn't installed, AVAILABLE stays False and memory.py falls back to
keyword retrieval so the agent still runs.
"""
import config

AVAILABLE = False
_collection = None


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
        _get_collection()
    except Exception as e:
        AVAILABLE = False
        print(f"[vectorstore] embeddings unavailable, using keyword fallback: {e}")
    return AVAILABLE


def upsert(fact_id, text):
    col = _get_collection()
    col.upsert(ids=[fact_id], documents=[text])


def query(text, k):
    col = _get_collection()
    if col.count() == 0:
        return []
    res = col.query(query_texts=[text], n_results=min(k, col.count()))
    return res.get("ids", [[]])[0]


def count():
    return _get_collection().count()


def reset():
    col = _get_collection()
    existing = col.get(include=[]).get("ids", [])
    if existing:
        col.delete(ids=existing)
