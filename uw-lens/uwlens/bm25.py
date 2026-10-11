"""Okapi BM25 over chunk text (+ title, tags), bilingual via textproc.tokenize."""
from __future__ import annotations

import math
from collections import Counter

from .textproc import tokenize


class BM25:
    def __init__(self, records: list[dict], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.ids = [r["id"] for r in records]
        self.tfs: list[Counter] = []
        df: Counter = Counter()
        for r in records:
            # title and filename carry strong signal for media items (DigUp-style filename matching)
            text = " ".join([r.get("title", ""), r.get("title", ""), " ".join(r.get("tags", [])), r.get("text", "")])
            tf = Counter(tokenize(text))
            self.tfs.append(tf)
            df.update(tf.keys())
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avgdl = (sum(self.lens) / len(self.lens)) if self.lens else 0.0
        n = len(records)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def query(self, text: str, k: int, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        terms = [t for t in set(tokenize(text)) if t in self.idf]
        if not terms or not self.ids:
            return []
        scored = []
        for i, tf in enumerate(self.tfs):
            if allowed is not None and self.ids[i] not in allowed:
                continue
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * self.lens[i] / (self.avgdl or 1))
            for t in terms:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + norm)
            if s > 0:
                scored.append((self.ids[i], s))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]
