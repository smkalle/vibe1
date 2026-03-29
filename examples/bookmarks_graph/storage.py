"""ChromaDB wrapper — stores tweets, embeddings, and enables semantic queries."""

from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb.utils import embedding_functions

CHROMA_PATH = str(Path.home() / ".bookmarks_graph_chroma")
COLLECTION_NAME = "twitter_bookmarks"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_tweets(tweets: List[dict]):
    """Store or update a list of tweet dicts. Embeddings are auto-generated."""
    col = _collection()
    col.upsert(
        ids=[t["id"] for t in tweets],
        documents=[t["text"] for t in tweets],
        # ChromaDB metadata values must be str/int/float/bool
        metadatas=[{k: str(v) for k, v in t.items() if k != "text"} for t in tweets],
    )


def count() -> int:
    return _collection().count()


def get_all_ids() -> List[str]:
    return _collection().get(include=[])["ids"]


def get_tweet(tweet_id: str) -> Optional[dict]:
    result = _collection().get(ids=[tweet_id], include=["documents", "metadatas"])
    if not result["ids"]:
        return None
    return {"id": tweet_id, "text": result["documents"][0], **result["metadatas"][0]}


def get_all_tweets() -> List[dict]:
    result = _collection().get(include=["documents", "metadatas"])
    return [
        {"id": tid, "text": result["documents"][i], **result["metadatas"][i]}
        for i, tid in enumerate(result["ids"])
    ]


def search(query: str, n_results: int = 10) -> List[dict]:
    """Semantic search — returns hits sorted by similarity (highest first)."""
    col = _collection()
    n = min(n_results, col.count())
    if n == 0:
        return []
    result = col.query(
        query_texts=[query],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for i, tid in enumerate(result["ids"][0]):
        # ChromaDB cosine distance: 0 = identical, 2 = opposite → convert to similarity
        score = 1.0 - result["distances"][0][i]
        hits.append(
            {
                "id": tid,
                "score": round(score, 4),
                "text": result["documents"][0][i],
                **result["metadatas"][0][i],
            }
        )
    return hits


def get_neighbors(tweet_id: str, n: int = 15, threshold: float = 0.45) -> List[dict]:
    """Return semantically similar tweets above the threshold for graph edge building."""
    tweet = get_tweet(tweet_id)
    if not tweet:
        return []
    col = _collection()
    # +1 because the tweet itself will appear in results
    n_query = min(n + 1, col.count())
    result = col.query(
        query_texts=[tweet["text"]],
        n_results=n_query,
        include=["metadatas", "distances"],
    )
    neighbors = []
    for i, tid in enumerate(result["ids"][0]):
        if tid == tweet_id:
            continue
        similarity = 1.0 - result["distances"][0][i]
        if similarity >= threshold:
            neighbors.append({"id": tid, "similarity": round(similarity, 4)})
    return neighbors
