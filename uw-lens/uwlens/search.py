"""Hybrid retrieval: semantic (vector store) + BM25, fused, filtered, grouped by file."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .bm25 import BM25
from .embedders import Embedder
from .store import VectorStore

Fusion = Literal["rrf", "weighted", "semantic", "keyword"]
RRF_K = 60


@dataclass
class Filters:
    modalities: set[str] | None = None
    sources: set[str] | None = None
    languages: set[str] | None = None

    def allows(self, r: dict) -> bool:
        return (
            (not self.modalities or r["modality"] in self.modalities)
            and (not self.sources or r["source"] in self.sources)
            and (not self.languages or r["language"] in self.languages)
        )


@dataclass
class Hit:
    record: dict
    score: float
    semantic: float | None = None  # raw cosine
    keyword: float | None = None  # raw BM25
    sem_rank: int | None = None
    kw_rank: int | None = None
    also: list[dict] = field(default_factory=list)  # other matching chunks of the same file

    @property
    def path(self) -> str:
        return self.record["path"]


class SearchEngine:
    """Holds the BM25 index in memory; rebuild with `refresh()` after (re)indexing."""

    def __init__(self, embedder: Embedder, store: VectorStore):
        self.embedder, self.store = embedder, store
        self.refresh()

    def refresh(self) -> None:
        self.records = {r["id"]: r for r in self.store.records()}
        self.bm25 = BM25(list(self.records.values()))

    def facets(self) -> dict[str, list[str]]:
        rs = self.records.values()
        return {
            "modality": sorted({r["modality"] for r in rs}),
            "source": sorted({r["source"] for r in rs}),
            "language": sorted({r["language"] for r in rs}),
        }

    def search(self, query: str = "", *, k: int = 8, fusion: Fusion = "rrf", alpha: float = 0.6,
               filters: Filters | None = None, group_by_file: bool = True, modality_norm: bool = False,
               query_vector: np.ndarray | None = None) -> list[Hit]:
        """`query_vector` enables query-by-example (e.g. an uploaded image); then BM25 uses `query` only if given."""
        filters = filters or Filters()
        allowed = {i for i, r in self.records.items() if filters.allows(r)}
        if not allowed or (not query.strip() and query_vector is None):
            return []
        pool = max(50, 5 * k)

        sem: list[tuple[str, float]] = []
        if fusion != "keyword":
            qv = query_vector if query_vector is not None else self.embedder.embed_queries([query])[0]
            sem = self.store.query(qv, pool, allowed)
            if modality_norm:
                sem = self._modality_normalize(sem, qv, allowed)
        kw: list[tuple[str, float]] = []
        if fusion != "semantic" and query.strip():
            kw = self.bm25.query(query, pool, allowed)

        sem_map = {i: (rank, s) for rank, (i, s) in enumerate(sem, 1)}
        kw_map = {i: (rank, s) for rank, (i, s) in enumerate(kw, 1)}
        fused = self._fuse(sem, kw, sem_map, kw_map, fusion, alpha)

        hits = [
            Hit(self.records[i], score,
                semantic=sem_map[i][1] if i in sem_map else None, keyword=kw_map[i][1] if i in kw_map else None,
                sem_rank=sem_map[i][0] if i in sem_map else None, kw_rank=kw_map[i][0] if i in kw_map else None)
            for i, score in fused if i in self.records
        ]
        if group_by_file:
            grouped: dict[str, Hit] = {}
            for h in hits:  # already sorted: first hit per file is its best chunk
                if h.path in grouped:
                    grouped[h.path].also.append(h.record)
                else:
                    grouped[h.path] = h
            hits = list(grouped.values())
        return hits[:k]

    @staticmethod
    def _fuse(sem, kw, sem_map, kw_map, fusion, alpha) -> list[tuple[str, float]]:
        if fusion == "semantic":
            return sem
        if fusion == "keyword":
            return kw
        ids = list(dict.fromkeys([i for i, _ in sem] + [i for i, _ in kw]))
        if fusion == "rrf":
            scores = {
                i: alpha * (1 / (RRF_K + sem_map[i][0]) if i in sem_map else 0.0)
                + (1 - alpha) * (1 / (RRF_K + kw_map[i][0]) if i in kw_map else 0.0)
                for i in ids
            }
            # rescale so the best possible score (rank 1 in both lists) is 1.0
            top = 1 / (RRF_K + 1)
            scores = {i: s / top for i, s in scores.items()}
        else:  # weighted on min-max normalized scores
            sn, kn = _minmax(dict(sem)), _minmax(dict(kw))
            scores = {i: alpha * sn.get(i, 0.0) + (1 - alpha) * kn.get(i, 0.0) for i in ids}
        return sorted(scores.items(), key=lambda x: -x[1])

    def _modality_normalize(self, sem, qv, allowed):
        """z-score semantic scores within each modality.

        Cross-modal cosines (text query vs image) sit lower than text-text cosines, so raw
        scores bury relevant media. Stats come from the full allowed set per modality.
        """
        stats = {}
        by_mod: dict[str, list[str]] = {}
        for i in allowed:
            by_mod.setdefault(self.records[i]["modality"], []).append(i)
        for mod, ids in by_mod.items():
            scores = np.array([s for _, s in self.store.query(qv, len(ids), set(ids))])
            stats[mod] = (float(scores.mean()), float(scores.std()) or 1.0) if len(scores) > 1 else (0.0, 1.0)
        out = [(i, (s - stats[self.records[i]["modality"]][0]) / stats[self.records[i]["modality"]][1]) for i, s in sem]
        return sorted(out, key=lambda x: -x[1])


def _minmax(d: dict[str, float]) -> dict[str, float]:
    if not d:
        return {}
    lo, hi = min(d.values()), max(d.values())
    if hi - lo < 1e-12:
        return {i: 1.0 for i in d}
    return {i: (v - lo) / (hi - lo) for i, v in d.items()}
