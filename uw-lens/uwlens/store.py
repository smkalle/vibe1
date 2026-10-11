"""Vector stores. NumpyStore is the default (exact cosine, zero extra deps, fine to ~1M chunks
on a laptop); ChromaStore is optional for users who already run Chroma."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np

MANIFEST = "manifest.json"


class EmbedderMismatch(RuntimeError):
    """The index was built with a different embedder/dimension; vectors are not comparable."""


class VectorStore(Protocol):
    def add(self, records: list[dict], vectors: np.ndarray) -> None: ...
    def delete_paths(self, paths: Iterable[str]) -> int: ...
    def query(self, vector: np.ndarray, k: int, allowed: set[str] | None = None) -> list[tuple[str, float]]: ...
    def records(self) -> list[dict]: ...
    def files(self) -> dict[str, str]: ...
    def set_file(self, path: str, fp: str | None) -> None: ...
    def save(self) -> None: ...
    def clear(self) -> None: ...
    def count(self) -> int: ...


def _atomic_write(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)


class _Manifest:
    """embedder id + dim + file fingerprints, shared by both backends."""

    def __init__(self, directory: Path, embedder_id: str, dim: int):
        self.path = directory / MANIFEST
        self.embedder_id, self.dim = embedder_id, dim
        self.file_fps: dict[str, str] = {}
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("embedder_id") != embedder_id or data.get("dim") != dim:
                raise EmbedderMismatch(
                    f"Index at {directory} was built with {data.get('embedder_id')} ({data.get('dim')}-d); "
                    f"current embedder is {embedder_id} ({dim}-d). Rebuild the index."
                )
            self.file_fps = data.get("files", {})

    def save(self) -> None:
        payload = {"embedder_id": self.embedder_id, "dim": self.dim, "files": self.file_fps}
        _atomic_write(self.path, json.dumps(payload, ensure_ascii=False, indent=1).encode("utf-8"))


class NumpyStore:
    def __init__(self, directory: Path, embedder_id: str, dim: int):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self.manifest = _Manifest(self.dir, embedder_id, dim)
        self._recs: list[dict] = []
        self._vecs = np.zeros((0, dim), dtype=np.float32)
        rec_path, vec_path = self.dir / "records.jsonl", self.dir / "vectors.npy"
        if rec_path.exists() and vec_path.exists():
            self._recs = [json.loads(line) for line in rec_path.read_text(encoding="utf-8").splitlines() if line]
            self._vecs = np.load(vec_path)
            if len(self._recs) != len(self._vecs):  # torn write from an older crash: start clean
                self._recs, self._vecs = [], np.zeros((0, dim), dtype=np.float32)
                self.manifest.file_fps = {}

    def add(self, records, vectors):
        if not records:
            return
        vectors = np.asarray(vectors, dtype=np.float32).reshape(len(records), self.dim)
        ids = {r["id"] for r in records}
        keep = [i for i, r in enumerate(self._recs) if r["id"] not in ids]
        self._recs = [self._recs[i] for i in keep] + list(records)
        self._vecs = np.vstack([self._vecs[keep], vectors])

    def delete_paths(self, paths):
        paths = set(paths)
        keep = [i for i, r in enumerate(self._recs) if r["path"] not in paths]
        removed = len(self._recs) - len(keep)
        self._recs = [self._recs[i] for i in keep]
        self._vecs = self._vecs[keep]
        return removed

    def query(self, vector, k, allowed=None):
        if not self._recs:
            return []
        scores = self._vecs @ np.asarray(vector, dtype=np.float32)
        if allowed is not None:
            mask = np.array([r["id"] in allowed for r in self._recs])
            scores = np.where(mask, scores, -np.inf)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self._recs[i]["id"], float(scores[i])) for i in top if np.isfinite(scores[i])]

    def records(self):
        return list(self._recs)

    def files(self):
        return dict(self.manifest.file_fps)

    def set_file(self, path, fp):
        if fp is None:
            self.manifest.file_fps.pop(path, None)
        else:
            self.manifest.file_fps[path] = fp

    def save(self):
        body = "\n".join(json.dumps(r, ensure_ascii=False) for r in self._recs)
        _atomic_write(self.dir / "records.jsonl", body.encode("utf-8"))
        tmp = self.dir / "vectors.tmp.npy"
        np.save(tmp, self._vecs)
        os.replace(tmp, self.dir / "vectors.npy")
        self.manifest.save()

    def clear(self):
        self._recs, self._vecs = [], np.zeros((0, self.dim), dtype=np.float32)
        self.manifest.file_fps = {}

    def count(self):
        return len(self._recs)


class ChromaStore:
    """Chroma persistent collection (cosine). Metadata keeps the full record as JSON."""

    def __init__(self, directory: Path, embedder_id: str, dim: int):
        import chromadb

        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self.manifest = _Manifest(self.dir, embedder_id, dim)
        self.client = chromadb.PersistentClient(path=str(self.dir / "chroma"))
        self.col = self.client.get_or_create_collection("uwlens", metadata={"hnsw:space": "cosine"})

    def add(self, records, vectors):
        if not records:
            return
        self.col.upsert(
            ids=[r["id"] for r in records],
            embeddings=np.asarray(vectors, dtype=np.float32).tolist(),
            documents=[r["text"] for r in records],
            metadatas=[{"path": r["path"], "json": json.dumps(r, ensure_ascii=False)} for r in records],
        )

    def delete_paths(self, paths):
        removed = 0
        for p in set(paths):
            got = self.col.get(where={"path": p})
            if got["ids"]:
                self.col.delete(ids=got["ids"])
                removed += len(got["ids"])
        return removed

    def query(self, vector, k, allowed=None):
        n = self.count()
        if n == 0:
            return []
        # Chroma can't filter by an arbitrary id set efficiently; over-fetch then filter.
        fetch = n if allowed is not None else min(k, n)
        res = self.col.query(query_embeddings=[np.asarray(vector).tolist()], n_results=fetch, include=["distances"])
        hits = [(i, 1.0 - d) for i, d in zip(res["ids"][0], res["distances"][0])]
        if allowed is not None:
            hits = [h for h in hits if h[0] in allowed]
        return hits[:k]

    def records(self):
        got = self.col.get(include=["metadatas"])
        return [json.loads(m["json"]) for m in got["metadatas"]]

    def files(self):
        return dict(self.manifest.file_fps)

    def set_file(self, path, fp):
        NumpyStore.set_file(self, path, fp)

    def save(self):
        self.manifest.save()

    def clear(self):
        self.client.delete_collection("uwlens")
        self.col = self.client.get_or_create_collection("uwlens", metadata={"hnsw:space": "cosine"})
        self.manifest.file_fps = {}

    def count(self):
        return self.col.count()


def open_store(kind: str, directory: Path, embedder_id: str, dim: int) -> VectorStore:
    sub = Path(directory) / kind  # numpy and chroma indexes never share files
    return ChromaStore(sub, embedder_id, dim) if kind == "chroma" else NumpyStore(sub, embedder_id, dim)


def reset_index(kind: str, directory: Path) -> None:
    """Delete an index (used on embedder mismatch / 'Rebuild')."""
    import shutil

    sub = Path(directory) / kind
    if sub.exists():
        shutil.rmtree(sub)
