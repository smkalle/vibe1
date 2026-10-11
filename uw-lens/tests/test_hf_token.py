"""Hugging Face token handling: passed to the model download, never shown, logged or cached in clear."""
import sys
import types
from pathlib import Path

import pytest

from uwlens.config import Settings
from uwlens.embedders import GemmaEmbedder, HashingEmbedder, describe_load_error, load_embedder

from .test_embedders import FakeST

TOKEN = "hf_TESTsecretTOKEN1234567890abcd"
APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture()
def fake_hub(monkeypatch):
    """Stand-ins for torch + sentence_transformers that record the SentenceTransformer kwargs."""
    calls = []
    torch = types.SimpleNamespace(bfloat16="bf16", float32="fp32",
                                  cuda=types.SimpleNamespace(is_available=lambda: False))
    st_mod = types.ModuleType("sentence_transformers")

    def factory(model_id, **kw):
        calls.append((model_id, kw))
        if kw.get("token") != TOKEN:
            raise OSError("401 Client Error: Unauthorized. Access to model google/embeddinggemma-2 is restricted.")
        return FakeST()

    st_mod.SentenceTransformer = factory
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "sentence_transformers", st_mod)
    return calls


def test_token_hidden_from_repr_and_equality():
    s = Settings(hf_token=TOKEN)
    assert TOKEN not in repr(s)
    assert s == Settings()  # token doesn't change identity/cache comparisons


def test_token_passed_to_model_download(fake_hub):
    emb, status = load_embedder(Settings(embedder="auto", modalities="text", hf_token=TOKEN))
    assert isinstance(emb, GemmaEmbedder) and status.startswith("Loaded")
    model_id, kw = fake_hub[-1]
    assert model_id == "google/embeddinggemma-2" and kw["token"] == TOKEN
    assert kw["config_kwargs"] == {"vision_config": None, "audio_config": None}


def test_no_token_kwarg_without_token(fake_hub):
    emb, status = load_embedder(Settings(embedder="auto"))
    assert isinstance(emb, HashingEmbedder)
    assert "token" not in fake_hub[-1][1]
    assert "paste a Hugging Face token" in status


def test_auth_error_is_redacted_and_hinted(fake_hub):
    emb, status = load_embedder(Settings(embedder="auto", hf_token="hf_WRONGtoken99999999"))
    assert isinstance(emb, HashingEmbedder)
    assert "hf_WRONGtoken99999999" not in status and "check the Hugging Face token" in status
    leaky = OSError(f"bad token {TOKEN} for url")
    assert TOKEN not in describe_load_error(leaky, TOKEN) and "hf_***" in describe_load_error(leaky, TOKEN)


# ------------------------------------------------------------------------------ UI
def _page_text(at) -> str:
    parts = []
    for kind in ("markdown", "caption", "success", "info", "warning", "error", "text_input"):
        for el in getattr(at, kind):
            parts.append(str(getattr(el, "value", "")))
    return " ".join(parts)


def test_sidebar_token_paste_mask_and_forget():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    box = at.text_input(key="hf_token_input")
    assert box.proto.type == box.proto.PASSWORD  # masked input
    box.set_value(TOKEN).run()
    assert not at.exception
    assert at.session_state["hf_token"] == TOKEN
    assert at.text_input(key="hf_token_input").value == ""  # widget cleared after use
    page = _page_text(at)
    assert TOKEN not in page and "hf_…" in page  # only the masked form is rendered
    assert any("isn't used" in i.value for i in at.info)  # tests run with UWLENS_EMBEDDER=hashing
    at.button(key="hf_forget").click().run()
    assert at.session_state["hf_token"] == ""


def test_sidebar_token_loads_gemma(fake_hub, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("UWLENS_EMBEDDER", "auto")
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert any("fallback" in w.value.lower() for w in at.warning)  # no token yet -> lexical fallback
    at.text_input(key="hf_token_input").set_value(TOKEN).run()
    assert not at.exception
    page = _page_text(at)
    assert "EmbeddingGemma 2" in page and TOKEN not in page
    assert fake_hub[-1][1]["token"] == TOKEN


def test_missing_dependencies_hint():
    assert "requirements-gemma.txt" in describe_load_error(ModuleNotFoundError("No module named 'torch'"), TOKEN)
