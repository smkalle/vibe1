"""Offline retrieval eval: build the sample corpus, index it, score each fusion mode -> evals/REPORT.md.

Uses whatever embedder UWLENS_EMBEDDER selects (default here: hashing, so it runs anywhere).
Run with UWLENS_EMBEDDER=gemma on a machine that can download EmbeddingGemma 2 to compare.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("UWLENS_EMBEDDER", "hashing")

from uwlens.cli import run_eval  # noqa: E402
from uwlens.config import Settings  # noqa: E402
from uwlens.embedders import load_embedder  # noqa: E402
from uwlens.indexer import index_folder  # noqa: E402
from uwlens.sample_corpus import build  # noqa: E402
from uwlens.search import SearchEngine  # noqa: E402
from uwlens.store import open_store  # noqa: E402


def main(k: int = 5) -> dict:
    queries = json.loads((ROOT / "evals" / "queries.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp:
        corpus, idx = Path(tmp) / "corpus", Path(tmp) / "idx"
        build(corpus)
        settings = Settings(index_dir=idx)
        embedder, status = load_embedder(settings)
        store = open_store("numpy", idx, embedder.id, embedder.dim)
        rep = index_folder(corpus, embedder, store, chunk_chars=settings.chunk_chars)
        res = run_eval(SearchEngine(embedder, store), queries, k=k)

    lines = [
        "# UW Lens retrieval eval",
        "",
        f"- Run: {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}",
        f"- Embedder: {embedder.label} (`{embedder.id}`)",
        f"- Corpus: synthetic sample, {len(rep.indexed)} files / {rep.chunks} chunks, skipped {len(rep.skipped)}",
        f"- Queries: {len(queries)} (`evals/queries.json`); a hit = any expected file in top-{k} (grouped by file)",
        "",
        "Cross-lingual queries (EN query → JA doc and vice versa) need a semantic multilingual model; the",
        "lexical hashing fallback is expected to miss them. They are reported separately and excluded from",
        "the same-language gate.",
        "",
        f"| Fusion | Group | n | Recall@{k} | MRR |",
        "|---|---|---|---|---|",
    ]
    for mode, data in res.items():
        for g, m in data["groups"].items():
            lines.append(f"| {mode} | {g} | {m['n']} | {m[f'recall@{k}']:.3f} | {m['mrr']:.3f} |")
    lines += ["", "## Per-query ranks (rrf)", "", "| Query | Group | Rank | Top-3 |", "|---|---|---|---|"]
    for r in res["rrf"]["rows"]:
        lines.append(f"| {r['q']} | {r['group']} | {r['rank'] or '—'} | {', '.join(r['top'])} |")
    report = ROOT / "evals" / "REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{status}\nwrote {report}")
    for mode, data in res.items():
        print(mode, json.dumps(data["groups"]["all_same_language"]))
    return res


if __name__ == "__main__":
    main()
