"""UW Lens — local multimodal health-document search for insurance underwriting research.

Run:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import html
import re
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uwlens import __version__  # noqa: E402
from uwlens.config import MATRYOSHKA_DIMS, MODALITY_PROFILES, Settings  # noqa: E402
from uwlens.embedders import HashingEmbedder, describe_load_error, load_embedder  # noqa: E402
from uwlens.evidence import DISCLAIMER, EvidencePack, PinnedItem  # noqa: E402
from uwlens.indexer import index_folder  # noqa: E402
from uwlens.questions import CATEGORIES  # noqa: E402
from uwlens.rag import answer as rag_answer  # noqa: E402
from uwlens.search import Filters, SearchEngine  # noqa: E402
from uwlens.store import EmbedderMismatch, open_store, reset_index  # noqa: E402
from uwlens.textproc import tokenize  # noqa: E402

st.set_page_config(page_title="UW Lens", page_icon="🔎", layout="wide")

ICON = {"text": "📄", "code": "🧩", "pdf": "📕", "image": "🖼️", "audio": "🎧", "video": "🎬"}
SOURCE_LABEL = {"nih": "NIH-style clinical ref", "japan": "Japan public insurance", "policy": "Underwriting manual",
                "case": "Case evidence", "other": "Other"}
CATEGORY_NAMES = list(CATEGORIES)

# ---------------------------------------------------------------------------------- state
ss = st.session_state
base = Settings()
ss.setdefault("pack", EvidencePack())
ss.setdefault("index_version", 0)
ss.setdefault("query", "")
ss.setdefault("rag_result", None)
ss.setdefault("embedder_choice", base.embedder)
ss.setdefault("profile", base.modalities)
ss.setdefault("dim", base.dim)
ss.setdefault("index_dir", str(base.index_dir))
ss.setdefault("corpus_dir", str(base.corpus_dir))
ss.setdefault("hf_token", "")  # session memory only: never written to disk, logs or the index


def md(text: str) -> str:
    """Escape user/corpus text for st.markdown (incl. `$`, which Streamlit treats as LaTeX)."""
    text = html.escape(text, quote=False)
    text = re.sub(r"([\\`*_{}\[\]#+!|$~>])", r"\\\1", text)
    # neutralize list markers at line starts ("- item", "5. item") so snippets stay compact
    text = re.sub(r"(?m)^(\s*)([-=])(?=\s|$)", r"\1\\\2", text)
    return re.sub(r"(?m)^(\s*\d+)\.(?=\s)", r"\1\\.", text)


def plain_snippet(text: str) -> str:
    """Drop markdown decoration (headings, emphasis) from stored chunk text for display."""
    text = re.sub(r"(?m)^\s*#{1,6}\s*", "", text)
    return re.sub(r"(?<!\w)\*{1,2}([^*\n]+?)\*{1,2}(?!\w)", r"\1", text)


def highlight(snippet: str, query: str) -> str:
    snippet = plain_snippet(snippet)
    out = md(snippet).replace("\n", "  \n")
    latin = {t for t in re.findall(r"[A-Za-z0-9.%]{3,}", query)}
    cjk = set()
    for run in re.findall(r"[぀-ヿ㐀-鿿]{2,}", query):
        # split on common particles so "メタボリックシンドロームの判定基準" highlights both halves
        cjk |= {p for p in re.split(r"[のはをがにとでや]", run) if len(p) >= 2 and p in snippet}
    for term in sorted(latin, key=len, reverse=True):
        out = re.sub(f"(?i)(?<![\\w*])({re.escape(md(term))})(?![\\w*])", r"**\1**", out)
    for term in sorted(cjk, key=len, reverse=True):  # no word boundaries in Japanese
        out = re.sub(f"(?<!\\*\\*)({re.escape(term)})", r"**\1**", out)
    return out


def token_fingerprint(token: str) -> str:
    """Cache key for a token: a short hash, so the secret itself is never a cache key."""
    return hashlib.sha256(token.encode()).hexdigest()[:16] if token else ""


def mask_token(token: str) -> str:
    return f"{token[:3]}…{token[-4:]}" if len(token) > 10 else "•" * len(token)


@st.cache_resource(show_spinner="Loading embedding model…")
def _load_embedder_cached(choice: str, profile: str, dim: int, token_fp: str, _token: str):
    # `_token` is excluded from Streamlit's cache hashing (leading underscore); `token_fp` keys the cache
    settings = base.with_(embedder=choice, modalities=profile, dim=dim, hf_token=_token or None)
    try:
        return load_embedder(settings)
    except Exception as exc:  # explicit "gemma" mode: show the error instead of crashing the app
        return HashingEmbedder(dim), (f"EmbeddingGemma 2 failed to load ({describe_load_error(exc, _token)}). "
                                      "Using the offline hashing fallback.")


def get_embedder():
    return _load_embedder_cached(ss.embedder_choice, ss.profile, ss.dim, token_fingerprint(ss.hf_token), ss.hf_token)


def set_hf_token():
    ss.hf_token = (ss.get("hf_token_input") or "").strip()
    ss.hf_token_input = ""  # don't keep the secret in the widget either


def forget_hf_token():
    ss.hf_token = ""


def get_engine():
    """(engine | None, error | None). The engine is cached per index version in session state."""
    embedder, _ = get_embedder()
    key = (embedder.id, ss.index_dir, base.store, ss.index_version)
    if ss.get("_engine_key") != key:
        try:
            store = open_store(base.store, Path(ss.index_dir), embedder.id, embedder.dim)
        except EmbedderMismatch as exc:
            return None, str(exc)
        ss["_engine"] = SearchEngine(embedder, store)
        ss["_engine_key"] = key
    return ss["_engine"], None


@st.cache_data(show_spinner=False, max_entries=64)
def pdf_page_png(path: str, locator: str, mtime: float) -> bytes | None:
    import pymupdf

    m = re.match(r"p(\d+)", locator or "")
    page_no = int(m.group(1)) - 1 if m else 0
    with pymupdf.open(path) as doc:
        if page_no >= len(doc):
            return None
        return doc[page_no].get_pixmap(dpi=80).tobytes("png")


def sample_key(category: str) -> str:
    return f"sample_q_{CATEGORY_NAMES.index(category)}"


def use_sample_question(key: str):
    """Copy the picked sample question into the query box (runs before the script reruns)."""
    if ss.get(key):
        ss.query = ss[key]
        ss.rag_result = None


def on_query_typed():
    """A typed question replaces any picked sample, so the drop-down never shows a stale choice."""
    ss.rag_result = None
    for cat in CATEGORY_NAMES:
        if ss.get(sample_key(cat)) not in (None, ss.query):
            ss[sample_key(cat)] = None


# ---------------------------------------------------------------------------------- sidebar
embedder, embed_status = get_embedder()
engine, engine_error = get_engine()

with st.sidebar:
    st.title("🔎 UW Lens")
    st.caption(f"Local multimodal health-doc search for underwriting research · v{__version__}")
    with st.container(border=True):
        st.markdown(f"**Embedder:** {md(embedder.label)}")
        if "fallback" in embed_status.lower() or "unavailable" in embed_status.lower():
            st.warning(embed_status, icon="⚠️")
        if engine:
            files = len({r["path"] for r in engine.records.values()})
            c1, c2 = st.columns(2)
            c1.metric("Files", files)
            c2.metric("Chunks", len(engine.records))
    gemma_active = not isinstance(embedder, HashingEmbedder)
    with st.expander("🤗 Hugging Face token", expanded=not gemma_active):
        st.caption("EmbeddingGemma 2 is a gated model: accept its license on "
                   "[huggingface.co/google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2), then "
                   "paste a **read** token from [your settings](https://huggingface.co/settings/tokens).")
        st.text_input("Access token", key="hf_token_input", type="password", placeholder="hf_…",
                      on_change=set_hf_token, help="Kept in this browser session's memory only. Never saved to disk, "
                      "logs or the index. HF_TOKEN / `huggingface-cli login` also work.")
        if ss.hf_token:
            if not ss.hf_token.startswith("hf_"):
                st.warning("Hugging Face tokens usually start with `hf_`.", icon="⚠️")
            st.success(f"Token active for this session: `{mask_token(ss.hf_token)}`", icon="🔑")
            st.button("Forget token", key="hf_forget", on_click=forget_hf_token)
            if ss.embedder_choice == "hashing":
                st.info("The embedder is set to `hashing`, so the token isn't used. Choose `auto` or `gemma` "
                        "under Engine settings.")
    with st.expander("Engine settings"):
        st.selectbox("Embedder", ["auto", "gemma", "hashing"], key="embedder_choice",
                     help="auto = EmbeddingGemma 2 if it loads, else the offline hashing fallback.")
        st.selectbox("Modalities to load", MODALITY_PROFILES, key="profile",
                     help="text ≈270M params · vision ≈440M · full ≈740M. All share one vector space.")
        st.selectbox("Matryoshka dimensions", MATRYOSHKA_DIMS, key="dim",
                     help="Smaller = faster/lighter index, modest quality loss. Changing it needs a re-index.")
        st.text_input("Index folder", key="index_dir")

    st.subheader("Ranking")
    k = st.slider("Results", 3, 20, 8, key="k")
    fusion = st.radio("Mode", ["rrf", "weighted", "semantic", "keyword"], horizontal=True, key="fusion",
                      format_func={"rrf": "Hybrid (RRF)", "weighted": "Hybrid (weighted)",
                                   "semantic": "Semantic", "keyword": "Keyword (BM25)"}.get)
    alpha = st.slider("Semantic weight α", 0.0, 1.0, 0.6, 0.05, key="alpha", disabled=fusion not in ("rrf", "weighted"))
    modality_norm = st.checkbox("Normalize scores per modality", key="modality_norm",
                                help="z-score semantic scores within each modality so images/audio aren't buried.")
    group = st.checkbox("One result per file", value=True, key="group")

    st.subheader("Filters")
    facets = engine.facets() if engine else {"modality": [], "source": [], "language": []}
    f_mod = st.multiselect("Modality", facets["modality"], key="f_mod", format_func=lambda m: f"{ICON.get(m, '')} {m}")
    f_src = st.multiselect("Source", facets["source"], key="f_src", format_func=lambda s: SOURCE_LABEL.get(s, s))
    f_lang = st.multiselect("Language", facets["language"], key="f_lang")

    st.subheader("Local LLM summary")
    use_rag = st.toggle("Enable (Ollama)", value=False, key="use_rag",
                        help="Off by default. Sends retrieved snippets only to a local Ollama server.")
    if use_rag:
        st.text_input("Ollama URL", base.ollama_url, key="ollama_url")
        st.text_input("Model", base.ollama_model, key="ollama_model")

tab_search, tab_index, tab_pack, tab_about = st.tabs(
    ["🔎 Search", "📂 Index", f"📌 Evidence pack ({len(ss.pack.items)})", "ℹ️ About"])

# ---------------------------------------------------------------------------------- search
with tab_search:
    st.caption(DISCLAIMER)
    if engine_error:
        st.error(engine_error)
        if st.button("Rebuild index for the current embedder", key="rebuild_mismatch"):
            reset_index(base.store, Path(ss.index_dir))
            ss.index_version += 1
            st.rerun()
    elif engine and not engine.records:
        st.info("The index is empty. Open the **📂 Index** tab to generate the sample corpus and index it.")

    st.text_input("Ask your own question about the corpus (English or Japanese)", key="query",
                  placeholder="e.g. CPAP compliance for moderate sleep apnea", on_change=on_query_typed)
    c_cat, c_q = st.columns([1, 2])
    category = c_cat.selectbox("…or browse sample questions by category", CATEGORY_NAMES, key="q_category")
    q_key = sample_key(category)
    c_q.selectbox("Sample question", [q.text for q in CATEGORIES[category]], index=None, key=q_key,
                  placeholder="Pick one of 3 sample questions…", on_change=use_sample_question, args=(q_key,))

    query_vector = None
    with st.expander("Search by example image"):
        if "image" in embedder.media_kinds:
            up = st.file_uploader("Image", type=["png", "jpg", "jpeg", "webp"], key="img_query")
            if up is not None:
                from PIL import Image

                img = Image.open(up).convert("RGB")
                st.image(img, width=160)
                query_vector = embedder.embed_media("image", img)
        else:
            st.info("Image-to-document search needs EmbeddingGemma 2 with the vision encoder "
                    "(embedder `auto`/`gemma`, modalities `vision` or `full`).")

    hits = []
    if engine and engine.records and (ss.query.strip() or query_vector is not None):
        filters = Filters(set(f_mod) or None, set(f_src) or None, set(f_lang) or None)
        hits = engine.search(ss.query, k=k, fusion=fusion, alpha=alpha, filters=filters, group_by_file=group,
                             modality_norm=modality_norm, query_vector=query_vector)
        st.markdown(f"**{len(hits)} result(s)**")
        if not hits:
            st.warning("No matches. Try fewer filters, another mode, or different wording.")

    if hits and use_rag:
        if st.button("✨ Generate research summary from these results", key="rag_go"):
            with st.spinner("Asking the local LLM…"):
                ss.rag_result = rag_answer(ss.query or "Summarize the evidence.", hits, url=ss.ollama_url,
                                           model=ss.ollama_model)
        res = ss.rag_result
        if res is not None:
            with st.container(border=True):
                if res.error:
                    st.error(res.error)
                else:
                    st.markdown(f"**Research summary** · `{md(res.model)}` · cites {res.citations or 'nothing'}")
                    st.markdown(md(res.text).replace("\n", "  \n"))
                    st.caption("Generated text can be wrong. Verify every claim against the cited evidence below.")

    for n, h in enumerate(hits, 1):
        r = h.record
        path = Path(r["path"])
        with st.container(border=True):
            left, right = st.columns([3, 2])
            with left:
                loc = f" · {r['locator']}" if r.get("locator") else ""
                st.markdown(f"**[{n}] {ICON.get(r['modality'], '')} {md(r['title'])}**")
                st.caption(f"{SOURCE_LABEL.get(r['source'], r['source'])} · {r['modality']} · {r['language']}{loc} · "
                           f"`{path.name}`")
                parts = [f"fused {h.score:.3f}"]
                if h.semantic is not None:
                    parts.append(f"semantic {h.semantic:.3f} (#{h.sem_rank})")
                if h.keyword is not None:
                    parts.append(f"BM25 {h.keyword:.2f} (#{h.kw_rank})")
                st.caption(" · ".join(parts))
                st.markdown(highlight(r["text"][:600], ss.query))
                if h.also:
                    st.caption(f"+{len(h.also)} more matching chunk(s) in this file: "
                               + ", ".join(a.get("locator", "") for a in h.also[:6]))
                pinned = any(i.id == r["id"] for i in ss.pack.items)
                if st.button("📌 Pinned" if pinned else "📌 Pin to evidence pack", key=f"pin_{r['id']}",
                             disabled=pinned):
                    ss.pack.pin(PinnedItem.from_hit(h, ss.query or "(image query)"))
                    st.rerun()
            with right:
                try:
                    if not path.exists():
                        st.caption("File moved or deleted — re-index.")
                    elif r["modality"] == "image":
                        st.image(str(path), width="stretch")
                    elif r["modality"] == "audio":
                        st.audio(str(path))
                    elif r["modality"] == "video":
                        st.video(str(path))
                    elif r["modality"] == "pdf":
                        png = pdf_page_png(str(path), r.get("locator", ""), path.stat().st_mtime)
                        if png:
                            st.image(png, width="stretch")
                except Exception as exc:  # preview must never break the results list
                    st.caption(f"Preview unavailable: {exc}")

# ---------------------------------------------------------------------------------- index
with tab_index:
    st.subheader("Index a folder")
    st.markdown("Supported: `.txt .md` · code `.py .js .ts .sql .json .yaml` · `.pdf` · images `.png .jpg .webp` · "
                "audio `.wav .mp3 .m4a .flac .ogg` · video `.mp4 .mov .mkv .webm`. Optional sidecar "
                "`<file>.meta.json` adds title/source/caption/transcript (stand-in for OCR/ASR).")
    st.text_input("Folder to index", key="corpus_dir")
    c1, c2, c3 = st.columns(3)
    redact = c1.toggle("Redact PHI before embedding", value=base.redact_phi, key="redact")
    chunk_chars = c2.slider("Chunk size (chars)", 400, 4000, base.chunk_chars, 100, key="chunk_chars")
    mods = c3.multiselect("Modalities", list(ICON), default=list(ICON), key="index_mods")
    if not redact:
        st.warning("PHI redaction is off: identifiers will be stored in the local index and shown in results.")

    b1, b2, b3 = st.columns(3)
    if b1.button("🧪 Generate sample corpus", key="make_sample", help="Synthetic EN/JA corpus, no real patient data"):
        from uwlens.sample_corpus import build

        with st.spinner("Rendering PDFs, images, audio, video…"):
            counts = build(Path(ss.corpus_dir))
        st.success(f"Sample corpus written to {ss.corpus_dir}: {counts}")
    go = b2.button("▶️ Index / update", type="primary", key="index_go", disabled=engine is None)
    rebuild = b3.button("♻️ Rebuild from scratch", key="index_rebuild")

    if go or rebuild:
        folder = Path(ss.corpus_dir)
        if not folder.is_dir():
            st.error(f"Folder not found: {folder}. Generate the sample corpus or enter another path.")
        else:
            if rebuild:
                reset_index(base.store, Path(ss.index_dir))
                ss.index_version += 1
                engine, _ = get_engine()
            bar = st.progress(0.0, text="Starting…")

            def progress(i, total, name):
                bar.progress(i / max(total, 1), text=f"{i}/{total} · {name}")

            rep = index_folder(folder, engine.embedder, engine.store, redact=redact, chunk_chars=chunk_chars,
                               audio_window_s=base.audio_window_s, modalities=set(mods), progress=progress)
            ss.index_version += 1
            ss.last_report = rep
            st.rerun()  # refresh sidebar counts and facets

    rep = ss.get("last_report")
    if rep is not None:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Indexed files", len(rep.indexed))
        m2.metric("Chunks", rep.chunks)
        m3.metric("Unchanged", len(rep.unchanged))
        m4.metric("Skipped", len(rep.skipped))
        if rep.removed:
            st.info(f"Removed {len(rep.removed)} deleted file(s) from the index.")
        for path, kinds in rep.phi_warnings:
            st.warning(f"PHI-like patterns remain in {Path(path).name}: {', '.join(kinds)}")
        if rep.skipped:
            st.dataframe([{"file": Path(p).name, "reason": why} for p, why in rep.skipped], width="stretch")
        st.caption("Note: text inside images/video frames is not redacted (pixels are embedded as-is).")

    if engine and engine.records:
        st.subheader("Index contents")
        rows = {}
        for r in engine.records.values():
            key = (r["source"], r["modality"], r["language"])
            rows.setdefault(key, {"files": set(), "chunks": 0})
            rows[key]["files"].add(r["path"])
            rows[key]["chunks"] += 1
        st.dataframe([{"source": s, "modality": m, "language": lang, "files": len(v["files"]), "chunks": v["chunks"]}
                      for (s, m, lang), v in sorted(rows.items())], width="stretch", hide_index=True)

# ---------------------------------------------------------------------------------- evidence pack
with tab_pack:
    pack: EvidencePack = ss.pack
    st.subheader("Evidence pack")
    st.caption(DISCLAIMER)
    c1, c2 = st.columns(2)
    pack.case_ref = c1.text_input("Case reference", pack.case_ref, key="case_ref",
                                  help="Use an internal pseudonymous reference, not the applicant's name.")
    pack.analyst = c2.text_input("Analyst", pack.analyst, key="analyst")
    pack.summary = st.text_area("Analyst summary", pack.summary, key="pack_summary", height=100)
    if not pack.items:
        st.info("Nothing pinned yet. Use **📌 Pin** on search results.")
    for n, it in enumerate(list(pack.items), 1):
        with st.container(border=True):
            a, b = st.columns([5, 1])
            loc = f" · {it.locator}" if it.locator else ""
            a.markdown(f"**{n}. {ICON.get(it.modality, '')} {md(it.title)}**")
            a.caption(f"{SOURCE_LABEL.get(it.source, it.source)} · {it.modality}{loc} · query “{it.query}”")
            if b.button("Remove", key=f"unpin_{it.id}"):
                pack.unpin(it.id)
                st.rerun()
            it.note = st.text_input("Note", it.note, key=f"note_{it.id}", label_visibility="collapsed",
                                    placeholder="Why this matters for the case…")
            st.markdown(md(plain_snippet(it.snippet[:300])).replace("\n", "  \n"))
    if pack.items:
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", pack.case_ref or "case").strip("_") or "case"
        d1, d2, d3 = st.columns(3)
        d1.download_button("⬇️ Markdown", pack.to_markdown(), file_name=f"evidence_{slug}.md", mime="text/markdown",
                           key="dl_md")
        d2.download_button("⬇️ CSV", pack.to_csv(), file_name=f"evidence_{slug}.csv", mime="text/csv", key="dl_csv")
        if d3.button("Clear pack", key="clear_pack"):
            ss.pack = EvidencePack()
            st.rerun()

# ---------------------------------------------------------------------------------- about
with tab_about:
    st.subheader("About UW Lens")
    st.markdown(
        f"""
UW Lens indexes a folder of mixed health documents — clinical references, Japanese public-insurance
documents, underwriting manuals, scanned reports, interview audio, short videos — and lets an underwriter
search them in English or Japanese. Retrieval fuses **semantic** similarity from
[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) (one 768-d space for text, images,
audio and video) with **BM25** keyword matching (CJK bigrams for Japanese).

**Active embedder:** {md(embedder.label)}
**Status:** {md(embed_status)}

**Privacy.** Everything runs on this machine after the one-time model download. The optional summary
only talks to a *localhost* Ollama server; remote endpoints are refused. Pattern-based PHI redaction
masks emails, phone numbers, SSNs, My Number, member/policy IDs, labelled names and dates of birth in
stored text before embedding. It is a safety net, not certified de-identification.

**Sample data.** The bundled corpus is synthetic: NIH/MedlinePlus-*style* summaries and Japanese
public-insurance explainers written for this project, plus a fictional insurer's manual and a fictional
applicant. Nothing is an official NIH or MHLW publication.

**Limits.** {md(DISCLAIMER)} The offline hashing fallback is lexical only: it cannot see images, hear audio
or match across languages, so media are found through their sidecar captions/transcripts.
"""
    )
