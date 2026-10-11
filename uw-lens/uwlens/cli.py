"""Command line: python -m uwlens {sample,index,search,eval}."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ROOT, Settings
from .embedders import load_embedder
from .extractors import SOURCES
from .indexer import index_folder
from .search import Filters, SearchEngine
from .store import EmbedderMismatch, open_store, reset_index


def build_engine(settings: Settings, rebuild_on_mismatch: bool = False):
    embedder, status = load_embedder(settings)
    try:
        store = open_store(settings.store, settings.index_dir, embedder.id, embedder.dim)
    except EmbedderMismatch:
        if not rebuild_on_mismatch:
            raise
        reset_index(settings.store, settings.index_dir)
        store = open_store(settings.store, settings.index_dir, embedder.id, embedder.dim)
    return embedder, store, status


def run_eval(engine: SearchEngine, queries: list[dict], k: int = 5) -> dict:
    """Recall@k (any expected file in top-k) and MRR per fusion mode, split by query group."""
    out: dict = {}
    for mode in ("keyword", "semantic", "rrf", "weighted"):
        rows = []
        for item in queries:
            hits = engine.search(item["q"], k=k, fusion=mode)
            names = [Path(h.path).name for h in hits]
            rank = next((i + 1 for i, n in enumerate(names) if n in item["expect"]), None)
            group = "cross_lingual" if item.get("cross_lingual") else item.get("lang", "en")
            rows.append({"q": item["q"], "group": group, "rank": rank, "top": names[:3]})
        groups = {}
        for g in sorted({r["group"] for r in rows}) + ["all_same_language"]:
            sel = [r for r in rows if (r["group"] == g if g != "all_same_language" else r["group"] != "cross_lingual")]
            if sel:
                groups[g] = {
                    "n": len(sel),
                    f"recall@{k}": round(sum(r["rank"] is not None for r in sel) / len(sel), 3),
                    "mrr": round(sum(1 / r["rank"] for r in sel if r["rank"]) / len(sel), 3),
                }
        out[mode] = {"groups": groups, "rows": rows}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="uwlens", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="generate the synthetic demo corpus")
    s.add_argument("--out", type=Path, default=None)
    i = sub.add_parser("index", help="index (incrementally) a folder")
    i.add_argument("folder", type=Path, nargs="?")
    i.add_argument("--rebuild", action="store_true")
    i.add_argument("--no-redact", action="store_true")
    q = sub.add_parser("search", help="search the index")
    q.add_argument("query")
    q.add_argument("-k", type=int, default=5)
    q.add_argument("--fusion", choices=["rrf", "weighted", "semantic", "keyword"], default="rrf")
    q.add_argument("--source", choices=SOURCES, action="append")
    q.add_argument("--json", action="store_true")
    e = sub.add_parser("eval", help="retrieval eval on evals/queries.json")
    e.add_argument("--queries", type=Path, default=ROOT / "evals" / "queries.json")
    e.add_argument("-k", type=int, default=5)
    args = ap.parse_args(argv)
    settings = Settings()

    if args.cmd == "sample":
        from .sample_corpus import build

        out = args.out or settings.corpus_dir
        print(json.dumps(build(out)), "->", out)
        return 0

    embedder, store, status = build_engine(settings, rebuild_on_mismatch=getattr(args, "rebuild", False))
    print(status, file=sys.stderr)
    if args.cmd == "index":
        if args.rebuild:
            store.clear()
        folder = args.folder or settings.corpus_dir
        rep = index_folder(folder, embedder, store, redact=settings.redact_phi and not args.no_redact,
                           chunk_chars=settings.chunk_chars, audio_window_s=settings.audio_window_s,
                           progress=lambda n, t, name: print(f"[{n}/{t}] {name}", file=sys.stderr))
        print(f"indexed {len(rep.indexed)} files ({rep.chunks} chunks), unchanged {len(rep.unchanged)}, "
              f"removed {len(rep.removed)}, skipped {len(rep.skipped)}")
        for path, why in rep.skipped:
            print(f"  skipped {path}: {why}")
        for path, kinds in rep.phi_warnings:
            print(f"  PHI patterns remain in {path}: {kinds}")
        return 0

    engine = SearchEngine(embedder, store)
    if args.cmd == "search":
        hits = engine.search(args.query, k=args.k, fusion=args.fusion,
                             filters=Filters(sources=set(args.source) if args.source else None))
        if args.json:
            print(json.dumps([{"score": h.score, **h.record} for h in hits], ensure_ascii=False, indent=1))
        for n, h in enumerate(hits, 1):
            r = h.record
            print(f"{n}. {h.score:.3f} [{r['modality']}/{r['source']}/{r['language']}] {r['path']} {r['locator']}")
            print("   " + r["text"][:160].replace("\n", " "))
        return 0

    if args.cmd == "eval":
        res = run_eval(engine, json.loads(args.queries.read_text(encoding="utf-8")), k=args.k)
        print(json.dumps({m: v["groups"] for m, v in res.items()}, indent=1))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
