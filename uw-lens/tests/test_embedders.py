"""Embedding contract (A8). EmbeddingGemma 2 can't be downloaded in CI, so GemmaEmbedder is
exercised against a fake SentenceTransformer that records every encode() call."""
import numpy as np
import pytest

from uwlens import embedders
from uwlens.config import Settings
from uwlens.embedders import GemmaEmbedder, HashingEmbedder, load_embedder, truncate


class FakeST:
    prompts = {"SearchQuery": "task: search result | query: ", "Document": "title: none | text: ",
               "CodeRetrieval": "task: code retrieval | query: "}

    def __init__(self):
        self.calls = []
        self._h = HashingEmbedder(768)

    def encode(self, inputs, **kw):
        self.calls.append((inputs, kw))
        vecs = []
        for x in inputs:
            text = x if isinstance(x, str) else "media:" + ",".join(sorted(x))
            vecs.append(self._h._embed(text) * 3.0)  # not normalized on purpose
        out = np.stack(vecs)
        return out[:, : kw["truncate_dim"]] if "truncate_dim" in kw else out


def test_hashing_embedder_deterministic_and_normalized():
    e = HashingEmbedder(256)
    a, b = e.embed_queries(["sleep apnea CPAP"]), e.embed_documents(["sleep apnea CPAP"])
    assert a.shape == (1, 256)
    assert np.allclose(a, b) and np.isclose(np.linalg.norm(a), 1.0)
    sim = lambda x, y: float(e.embed_queries([x])[0] @ e.embed_queries([y])[0])  # noqa: E731
    assert sim("diabetes HbA1c", "HbA1c of diabetic patients") > sim("diabetes HbA1c", "asthma inhaler")
    assert np.isclose(np.linalg.norm(e.embed_media("image", None, "chart of HbA1c")), 1.0)


def test_gemma_uses_asymmetric_prompts_and_prompt_free_media():
    fake = FakeST()
    g = GemmaEmbedder(Settings(dim=768, modalities="full"), model=fake)
    q = g.embed_queries(["dog on beach"])
    g.embed_documents(["doc text"])
    g.embed_documents(["def f(): pass"], code=True)
    m = g.embed_media("image", "x.png")
    (_, kq), (_, kd), (_, kc), (mi, km) = fake.calls
    assert kq["prompt_name"] == "SearchQuery" and kd["prompt_name"] == "Document"
    assert kc["prompt_name"] == "CodeRetrieval"
    assert "prompt_name" not in km and mi == [{"image": "x.png"}]
    assert all(c[1]["normalize_embeddings"] is True for c in fake.calls)
    assert np.isclose(np.linalg.norm(q[0]), 1.0) and np.isclose(np.linalg.norm(m), 1.0)
    assert g.id == "google/embeddinggemma-2:768"


def test_gemma_matryoshka_truncation_renormalizes():
    fake = FakeST()
    g = GemmaEmbedder(Settings(dim=256), model=fake)
    v = g.embed_queries(["x"])
    assert v.shape == (1, 256) and np.isclose(np.linalg.norm(v), 1.0)
    assert fake.calls[0][1]["truncate_dim"] == 256
    assert g.id.endswith(":256")


def test_gemma_text_profile_falls_back_to_caption_for_media():
    fake = FakeST()
    g = GemmaEmbedder(Settings(modalities="text"), model=fake)
    g.embed_media("image", "x.png", fallback_text="scanned lab report")
    inputs, kw = fake.calls[-1]
    assert inputs == ["scanned lab report"] and kw["prompt_name"] == "Document"


def test_gemma_rejects_nan():
    class NanST(FakeST):
        def encode(self, inputs, **kw):
            return np.full((len(inputs), 768), np.nan)

    with pytest.raises(ValueError, match="NaN"):
        GemmaEmbedder(Settings(), model=NanST()).embed_queries(["x"])


def test_auto_falls_back_to_hashing(monkeypatch):
    def boom(settings):
        raise OSError("huggingface.co unreachable")

    monkeypatch.setattr(embedders.GemmaEmbedder, "_load", staticmethod(boom))
    emb, status = load_embedder(Settings(embedder="auto", dim=512))
    assert isinstance(emb, HashingEmbedder) and emb.dim == 512
    assert "fallback" in status and "unreachable" in status
    with pytest.raises(OSError):
        load_embedder(Settings(embedder="gemma"))


def test_truncate_handles_zero_vectors():
    out = truncate(np.zeros((2, 768)), 128)
    assert out.shape == (2, 128) and np.isfinite(out).all()


def test_settings_validation():
    with pytest.raises(ValueError):
        Settings(dim=300)
    with pytest.raises(ValueError):
        Settings(modalities="audio-only")
