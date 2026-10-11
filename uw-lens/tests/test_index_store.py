import json
from pathlib import Path

import numpy as np
import pytest

from uwlens.embedders import GemmaEmbedder, HashingEmbedder
from uwlens.config import Settings
from uwlens.extractors import extract
from uwlens.indexer import index_folder
from uwlens.store import EmbedderMismatch, NumpyStore, open_store
from uwlens.textproc import find_phi

from .test_embedders import FakeST


def _rec(i, path="a.md"):
    return {"id": i, "path": path, "title": i, "modality": "text", "source": "nih", "language": "en", "text": i,
            "locator": "", "tags": []}


def test_numpy_store_roundtrip_query_and_delete(tmp_path):
    s = NumpyStore(tmp_path, "emb", 4)
    s.add([_rec("a"), _rec("b", "b.md")], np.eye(4)[:2])
    s.set_file("a.md", "fp1")
    s.save()
    s2 = NumpyStore(tmp_path, "emb", 4)
    assert s2.count() == 2 and s2.files() == {"a.md": "fp1"}
    assert s2.query(np.array([1, 0, 0, 0]), 1)[0][0] == "a"
    assert [i for i, _ in s2.query(np.array([1, 0, 0, 0]), 5, allowed={"b"})] == ["b"]
    s2.add([_rec("a")], np.eye(4)[2:3])  # upsert replaces, never duplicates
    assert s2.count() == 2
    assert s2.delete_paths(["a.md"]) == 1 and s2.count() == 1


def test_store_rejects_other_embedder(tmp_path):
    s = NumpyStore(tmp_path, "hashing-v1-768", 768)
    s.save()
    with pytest.raises(EmbedderMismatch):
        NumpyStore(tmp_path, "google/embeddinggemma-2:768", 768)


def test_chroma_store_matches_numpy(tmp_path):
    pytest.importorskip("chromadb")
    emb = HashingEmbedder(768)
    vecs = emb.embed_documents(["sleep apnea CPAP", "高額療養費", "HbA1c rating"])
    recs = [_rec("x"), _rec("y", "y.md"), _rec("z", "z.md")]
    results = {}
    for kind in ("numpy", "chroma"):
        st = open_store(kind, tmp_path, emb.id, emb.dim)
        st.add(recs, vecs)
        st.save()
        q = emb.embed_queries(["CPAP sleep apnea"])[0]
        results[kind] = [i for i, _ in st.query(q, 3)]
        assert [i for i, _ in st.query(q, 3, allowed={"y"})] == ["y"]
        assert st.delete_paths(["y.md"]) == 1 and st.count() == 2
    assert results["numpy"] == results["chroma"]


def test_index_sample_corpus_all_modalities(engine):
    mods = {r["modality"] for r in engine.records.values()}
    assert {"text", "code", "pdf", "image", "audio", "video"} <= mods
    assert {r["source"] for r in engine.records.values()} == {"nih", "japan", "policy", "case"}
    assert {"en", "ja"} <= {r["language"] for r in engine.records.values()}


def test_no_phi_in_stored_records(engine):  # A4
    for r in engine.records.values():
        assert find_phi(r["text"]) == [], (r["path"], r["text"][:200])
    aps = [r for r in engine.records.values() if r["path"].endswith("aps_case_0417.md")]
    assert aps and all("Taro Example" not in r["text"] and "090-1234-5678" not in r["text"] for r in aps)


def test_incremental_reindex_update_and_delete(corpus_copy, tmp_path):
    emb = HashingEmbedder(768)
    store = open_store("numpy", tmp_path / "idx", emb.id, emb.dim)
    first = index_folder(corpus_copy, emb, store)
    assert len(first.indexed) == 19 and not first.skipped
    second = index_folder(corpus_copy, emb, store)
    assert not second.indexed and len(second.unchanged) == 19

    md = corpus_copy / "nih" / "asthma.md"
    md.write_text(md.read_text() + "\n\nNew paragraph about omalizumab biologic therapy.\n")
    (corpus_copy / "nih" / "hypertension.md").unlink()
    (corpus_copy / "case" / "broken.pdf").write_bytes(b"not a pdf")
    third = index_folder(corpus_copy, emb, store)
    assert [Path(p).name for p, _ in third.indexed] == ["asthma.md"]
    assert [Path(p).name for p in third.removed] == ["hypertension.md"]
    assert [Path(p).name for p, _ in third.skipped] == ["broken.pdf"]  # bad file skipped, run continues
    paths = {r["path"] for r in store.records()}
    assert not any(p.endswith("hypertension.md") for p in paths)
    assert any("omalizumab" in r["text"] for r in store.records())

    # sidecar edits count as a change
    side = corpus_copy / "case" / "bp_trend_0417.png.meta.json"
    meta = json.loads(side.read_text())
    meta["caption"] += " Includes diastolic readings."
    side.write_text(json.dumps(meta))
    fourth = index_folder(corpus_copy, emb, store)
    assert [Path(p).name for p, _ in fourth.indexed] == ["bp_trend_0417.png"]


def test_modality_filter_skips(corpus_copy, tmp_path):
    emb = HashingEmbedder(768)
    store = open_store("numpy", tmp_path / "idx", emb.id, emb.dim)
    rep = index_folder(corpus_copy, emb, store, modalities={"pdf"})
    assert {Path(p).suffix for p, _ in rep.indexed} == {".pdf"}
    assert {r["modality"] for r in store.records()} == {"pdf"}


def test_audio_is_windowed_for_gemma(corpus):
    wav = corpus / "case" / "teleinterview_0417.wav"
    chunks = extract(wav, corpus, load_audio=True, audio_window_s=30)
    assert [c.locator for c in chunks] == ["00:00-00:30", "00:30-01:00", "01:00-01:15"]
    assert all(c.media.dtype == np.float32 and c.media.ndim == 1 for c in chunks)
    assert len(chunks[0].media) == 30 * 16_000
    assert "smoking" in chunks[2].text.lower()  # per-window transcript segment attached


def test_index_with_fake_gemma_routes_media(corpus, tmp_path):
    fake = FakeST()
    g = GemmaEmbedder(Settings(modalities="full"), model=fake)
    store = open_store("numpy", tmp_path / "idx", g.id, g.dim)
    rep = index_folder(corpus, g, store)
    assert not rep.skipped
    media_calls = [c[0][0] for c in fake.calls if isinstance(c[0][0], dict)]
    kinds = sorted({next(iter(m)) for m in media_calls})
    assert kinds == ["audio", "image", "video"]
    assert sum("audio" in m for m in media_calls) == 3  # 75 s clip -> three 30 s windows


def test_turning_redaction_on_reprocesses_files(corpus_copy, tmp_path):
    emb = HashingEmbedder(768)
    store = open_store("numpy", tmp_path / "idx", emb.id, emb.dim)
    index_folder(corpus_copy, emb, store, redact=False)
    assert any("Taro Example" in r["text"] for r in store.records())
    rep = index_folder(corpus_copy, emb, store, redact=True)
    assert len(rep.indexed) == 19 and not rep.unchanged
    assert not any("Taro Example" in r["text"] for r in store.records())
    assert all(find_phi(r["text"]) == [] for r in store.records())
