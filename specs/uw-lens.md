# UW Lens — local multimodal health-document search for insurance underwriting

Status: v1 (built 2026-10-11). Code: `uw-lens/`.

## 1. Problem

Life/medical underwriters research applicant conditions across a messy, mixed corpus:
NIH/MedlinePlus-style clinical references, Japanese public-insurance documents
(MHLW / NHI / 高額療養費 / 特定健診), the insurer's own underwriting manual (PDF),
and per-case evidence (attending physician statements, scanned lab reports,
tele-interview audio, short video). Keyword search fails across modalities and
across EN/JA. Health data must not leave the machine.

The EmbeddingGemma 2 tutorial (DigUp-style local multimodal search) gives the core:
one 768-d space for text/code/image/audio/video. UW Lens turns it into an
open-source Streamlit app for underwriting research.

## 2. Requirements (confirmed with the user)

| # | Requirement | Decision |
|---|---|---|
| R1 | Primary user | Underwriting researcher (risk assessment at policy issuance) |
| R2 | Sample corpus | Synthetic + public-domain-style content authored for this repo; no real PHI, no copied copyrighted text |
| R3 | Language | Bilingual EN + JA: JA text indexed, CJK-aware keyword search, JA example queries |
| R4 | Hybrid ranking + filters | BM25 (CJK bigrams) fused with semantic; filters by modality / source / language |
| R5 | PHI redaction | Mask emails, phones (US/JP), SSN, My Number, member/policy IDs, DOB in snippets **and** in text before embedding; on by default |
| R6 | Evidence pack | Pin results, add notes, export Markdown + CSV for the case file |
| R7 | Local LLM RAG | Optional cited summary via Ollama on localhost; off by default; never auto-decides |
| R8 | Local-first | After the model download nothing leaves the machine (Ollama is localhost) |

Non-goals (v1): OCR engine integration (sidecar text is used instead), Excel/PPT extractors,
multi-user auth, making any underwriting decision.

## 3. Constraints and risks

- **Model access.** `google/embeddinggemma-2` is downloaded from Hugging Face via
  `sentence-transformers>=6.1`. CI/sandboxes may not reach HF, so the embedder is
  pluggable with a deterministic offline **hashing embedder** (char n-gram feature
  hashing, 768-d). The fallback is *lexical*, not semantic: it cannot see pixels or
  hear audio and does not do cross-lingual matching; for media it embeds the sidecar
  caption/transcript. The UI always shows which embedder is active.
- **Index consistency.** Vectors from different embedders/dims never mix: the index
  manifest records embedder id + dim and a mismatch forces a rebuild.
- **Memory.** Modular loading (`text` 270M / `vision` 440M / `full` 740M) via
  `config_kwargs`; bf16 only when supported, never fp16.
- **Medical/regulatory.** Research aid only. Every page carries a disclaimer; RAG prompt
  forbids issuing decisions and requires citations.

## 4. Architecture

```
uw-lens/
  app.py                 Streamlit UI (Search | Index | Evidence pack | About)
  uwlens/
    config.py            Settings from env (UWLENS_*), defaults
    textproc.py          tokenize (latin words + CJK bigrams), detect_language, chunk_text, redact_phi
    extractors.py        folder walk -> Chunk records (txt/md/code, pdf pages, image, audio 30 s windows, video), sidecars
    embedders.py         Embedder protocol; GemmaEmbedder (sentence-transformers); HashingEmbedder; load_embedder()
    store.py             NumpyStore (default, .npy + jsonl) and ChromaStore (optional); manifest guard
    bm25.py              Okapi BM25 over chunk text
    indexer.py           incremental indexing (file fingerprint), stale removal, progress callback
    search.py            hybrid search: semantic + BM25, RRF or weighted fusion, per-modality z-norm, filters, group-by-file
    rag.py               Ollama client + cited prompt; injectable transport for tests
    evidence.py          evidence pack -> Markdown / CSV
    sample_corpus.py     generates the synthetic EN/JA corpus (md, pdf, png, wav, mp4) from corpus_src/
    cli.py               python -m uwlens {sample,index,search,eval}
  corpus_src/            authored source texts (synthetic, CC0)
  evals/queries.json     labelled retrieval queries; run_evals.py -> evals/REPORT.md
  tests/                 pytest (unit + Streamlit AppTest), all offline
```

### Data model — `Chunk`
`id` (stable: sha1(path)+locator), `path`, `title`, `modality` (text|pdf|image|audio|video),
`source` (nih|japan|policy|case|other — from sidecar or folder name), `language` (en|ja|mixed),
`text` (redacted; used for BM25, snippet, fallback embedding), `locator` (page / chunk / t_start–t_end),
`media` (embed payload, not persisted).

Sidecar `<file>.meta.json` (optional): `{title, source, language, caption, transcript, tags}`.

### Embedding contract
- Queries: `prompt_name="SearchQuery"`; text docs: `prompt_name="Document"`; code: `"CodeRetrieval"` if the model
  defines it, else `"Document"`.
- Media: `{"image": path}`, `{"audio": array@16k}`, `{"video": path}`, no prompt.
- Always `normalize_embeddings=True`; Matryoshka `truncate_dim` ∈ {128,256,512,768}, re-normalized.

### Ranking
1. Semantic: cosine top-N (N = max(50, 5k)) from the store, after filters.
2. Keyword: BM25 top-N over the same filtered set.
3. Fusion: RRF (k=60, default) or weighted `α·sem + (1-α)·bm25` on min-max normalized scores;
   optional per-modality z-normalization of semantic scores (image/audio cosines run lower than text).
4. Group by file (best chunk wins), return top-k with a score breakdown.

## 5. UI

- Sidebar: engine card (embedder, dims, modalities, chunks indexed), top-k, fusion mode/α, filters, group-by-file.
- **Search**: query box, EN/JA example chips, image query (vision-capable embedder only), result cards with
  modality badge, source/lang, scores, redacted snippet, inline preview (image/audio/video/PDF page), Pin button;
  optional "Research summary (local LLM)" with numbered citations.
- **Index**: folder path, "Generate sample corpus", PHI redaction toggle, chunk size, modalities, Index/Rebuild,
  progress bar, per-file log.
- **Evidence pack**: case ref, analyst notes, pinned items with per-item notes, Markdown/CSV download.
- **About**: disclaimers, data provenance, model/fallback status.

## 6. Verification plan (PDCA)

- **Plan**: this spec. **Do**: build modules + app. **Check**: `pytest` (unit + AppTest) and
  `run_evals.py` (Recall@5, MRR for BM25 / semantic / hybrid). **Act**: fix regressions and re-run until all
  tests pass and hybrid ≥ max(BM25, semantic) on the eval set.
- Manual: `streamlit run app.py`, headless screenshot of Search and Evidence tabs.
- Acceptance: A1 index sample corpus offline; A2 EN and JA queries return the labelled doc in top-5;
  A3 filters work; A4 no PHI pattern in any stored snippet; A5 evidence pack exports; A6 RAG off by default and
  cites sources when on; A7 index rejects embedder mismatch; A8 Gemma adapter calls follow the embedding contract
  (verified with a fake model since HF is unreachable in CI).

## 7. Verification log (PDCA) and sign-off — 2026-10-11

| Cycle | Check finding | Act |
|---|---|---|
| 1 | `fitz` import deprecated; CJK font renders Latin full-width; em-dash glyph broken in PDFs | `import pymupdf`; Helvetica for ASCII pages, built-in `japan` font for JA; dash normalization |
| 2 | Chunker dropped newlines and started chunks mid-word; labelled names (`Patient name:`, `氏名：`) not redacted | whitespace-preserving piece splitter with whole-piece overlap; `NAME` redaction pattern |
| 3 | AppTest: sidebar counts stale after indexing; `use_container_width` deprecated; titles were filenames | `st.rerun()` after indexing; `width="stretch"`; titles from first heading |
| 4 | Browser screenshots: snippets rendered as markdown lists/headings; JA terms not highlighted; redundant media locators | markdown-decoration stripping + list escaping; particle-split CJK highlighting; empty locator for single-item media |
| 5 | Self-review: toggling redaction on left previously indexed text unredacted (fingerprint ignored settings) | fingerprint includes redact/chunk/window settings; regression test |

Final state:
- `pytest`: 47 passed (Python 3.13). In a clean Python 3.11 venv from `requirements.txt` only: 45 passed, 1 skipped (Chroma optional).
- Eval (hashing fallback, 21 same-language queries): Recall@5 1.000 for all modes; MRR RRF 0.952 ≥ BM25 0.905. Cross-lingual
  recall 0.50 (expected: needs EmbeddingGemma 2).
- Live app (Chromium via Playwright): no page errors or Streamlit exceptions through search EN/JA → pin → evidence → index.
- Acceptance A1–A8: met. A2 and A8 were verified offline: A2 with the lexical fallback, A8 with a recording fake model. Nobody has
  measured real EmbeddingGemma 2 retrieval quality here, because the CI sandbox can't reach Hugging Face. First thing to do on a
  machine with model access: `UWLENS_EMBEDDER=gemma python run_evals.py`.
