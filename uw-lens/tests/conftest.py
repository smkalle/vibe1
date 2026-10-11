import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uwlens.embedders import HashingEmbedder  # noqa: E402
from uwlens.indexer import index_folder  # noqa: E402
from uwlens.sample_corpus import build  # noqa: E402
from uwlens.search import SearchEngine  # noqa: E402
from uwlens.store import open_store  # noqa: E402


@pytest.fixture(autouse=True)
def offline_env(monkeypatch, tmp_path):
    """Tests never download models or touch the user's index."""
    monkeypatch.setenv("UWLENS_EMBEDDER", "hashing")
    monkeypatch.setenv("UWLENS_INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("UWLENS_CORPUS_DIR", str(tmp_path / "corpus"))
    monkeypatch.delenv("UWLENS_DIM", raising=False)
    monkeypatch.delenv("UWLENS_MODALITIES", raising=False)


@pytest.fixture(scope="session")
def corpus(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("corpus")
    build(out)
    return out


@pytest.fixture()
def corpus_copy(corpus, tmp_path) -> Path:
    import shutil

    dst = tmp_path / "corpus_copy"
    shutil.copytree(corpus, dst)
    return dst


@pytest.fixture(scope="session")
def engine(corpus, tmp_path_factory) -> SearchEngine:
    emb = HashingEmbedder(768)
    store = open_store("numpy", tmp_path_factory.mktemp("idx"), emb.id, emb.dim)
    rep = index_folder(corpus, emb, store)
    assert not rep.skipped, rep.skipped
    return SearchEngine(emb, store)
