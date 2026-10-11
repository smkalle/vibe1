import csv
import io
import json
from pathlib import Path

import pytest

from uwlens.cli import run_eval
from uwlens.evidence import DISCLAIMER, EvidencePack, PinnedItem
from uwlens.rag import SYSTEM, answer, build_prompt, is_local
from uwlens.search import Filters

ROOT = Path(__file__).resolve().parents[1]


def names(hits):
    return [Path(h.path).name for h in hits]


@pytest.mark.parametrize("fusion", ["rrf", "weighted", "semantic", "keyword"])
def test_each_fusion_mode_finds_target(engine, fusion):
    hits = engine.search("eGFR stages albuminuria", k=3, fusion=fusion)
    # BM25 alone has no stemming ("stages" vs "Staging") and ranks the diabetes chunk first;
    # the hybrid modes recover the CKD reference at #1, which is why hybrid is the default.
    if fusion == "keyword":
        assert "chronic_kidney_disease.md" in names(hits)
    else:
        assert names(hits)[0] == "chronic_kidney_disease.md"
    assert hits == sorted(hits, key=lambda h: -h.score)


def test_japanese_query(engine):  # A2
    assert names(engine.search("高額療養費 自己負担限度額", k=3))[0] == "kogaku_ryoyohi.md"
    assert "kenshin_result_0417.png" in names(engine.search("健康診断 HbA1c 尿糖", k=5))


def test_media_found_through_sidecar_text(engine):
    assert names(engine.search("tele-interview smoking history", k=3))[0] == "teleinterview_0417.wav"
    assert "cpap_compliance_0417.mp4" in names(engine.search("CPAP machine usage screen recording", k=3))


def test_filters(engine):  # A3
    hits = engine.search("insurance", k=20, filters=Filters(sources={"japan"}))
    assert hits and all(h.record["source"] == "japan" for h in hits)
    hits = engine.search("HbA1c", k=20, filters=Filters(modalities={"image"}))
    assert hits and all(h.record["modality"] == "image" for h in hits)
    hits = engine.search("HbA1c", k=20, filters=Filters(languages={"ja"}))
    assert hits and all(h.record["language"] == "ja" for h in hits)
    assert engine.search("HbA1c", filters=Filters(sources={"nonexistent"})) == []


def test_group_by_file_and_score_breakdown(engine):
    grouped = engine.search("diabetes", k=20)
    assert len(names(grouped)) == len(set(names(grouped)))
    flat = engine.search("diabetes", k=20, group_by_file=False)
    assert len(names(flat)) > len(set(names(flat)))
    h = grouped[0]
    assert h.semantic is not None and h.keyword is not None and h.sem_rank and h.kw_rank


def test_query_vector_and_modality_norm(engine):
    qv = engine.embedder.embed_queries(["sleep apnea"])[0]
    assert engine.search("", k=3, query_vector=qv, fusion="semantic")
    assert engine.search("sleep apnea", k=5, modality_norm=True)
    assert engine.search("   ") == []


def test_retrieval_eval_gate(engine):
    """PDCA gate: same-language recall@5 >= 0.9 and hybrid is no worse than either signal alone."""
    queries = json.loads((ROOT / "evals" / "queries.json").read_text(encoding="utf-8"))
    res = run_eval(engine, queries, k=5)
    g = {m: res[m]["groups"]["all_same_language"] for m in res}
    assert g["rrf"]["recall@5"] >= 0.9
    assert g["rrf"]["mrr"] >= max(g["keyword"]["mrr"], g["semantic"]["mrr"]) - 1e-9


# ------------------------------------------------------------------------------ RAG (A6)
def test_rag_refuses_remote_endpoints(engine):
    hits = engine.search("sleep apnea", k=2)
    res = answer("q", hits, url="https://api.example.com", model="m", transport=lambda *a: pytest.fail("called"))
    assert res.error and "non-local" in res.error
    assert is_local("http://localhost:11434") and is_local("http://127.0.0.1:8080/")
    assert not is_local("http://localhost.evil.com")


def test_rag_cites_and_builds_grounded_prompt(engine):
    hits = engine.search("CPAP compliance", k=3)
    seen = {}

    def transport(url, payload, timeout):
        seen.update(url=url, payload=payload)
        return {"message": {"content": "Compliance is 4h on 70% of nights [1][3]. Also [9]."}}

    res = answer("What is CPAP compliance?", hits, url="http://localhost:11434", model="gemma3:4b",
                 transport=transport)
    assert res.citations == [1, 3] and not res.error  # out-of-range [9] dropped
    assert seen["url"].endswith("/api/chat")
    assert seen["payload"]["messages"][0]["content"] == SYSTEM
    assert "[1]" in seen["payload"]["messages"][1]["content"]
    assert "Do not make or recommend an underwriting decision" in SYSTEM


def test_rag_reports_unreachable_server(engine):
    def down(*a):
        raise OSError("connection refused")

    res = answer("q", engine.search("asthma", k=1), url="http://localhost:1", model="m", transport=down)
    assert res.error and "ollama serve" in res.error
    assert answer("q", [], url="http://localhost:1", model="m").text.startswith("No evidence")


def test_build_prompt_numbers_evidence(engine):
    hits = engine.search("asthma", k=2)
    p = build_prompt("q", hits)
    assert p.count("[1]") == 1 and "[2]" in p and "Question: q" in p


# ------------------------------------------------------------------------------ evidence pack (A5)
def test_evidence_pack_exports(engine):
    hits = engine.search("CPAP compliance", k=2)
    pack = EvidencePack(case_ref="UWR-17", analyst="A. Analyst", summary="OSA treated, compliant.")
    assert pack.pin(PinnedItem.from_hit(hits[0], "CPAP compliance"))
    assert not pack.pin(PinnedItem.from_hit(hits[0], "again"))  # dedupe
    pack.pin(PinnedItem.from_hit(hits[1], "CPAP compliance"))
    pack.items[0].note = "Meets 70% threshold"
    md = pack.to_markdown()
    assert "UWR-17" in md and DISCLAIMER in md and "Meets 70% threshold" in md and "### 2." in md
    rows = list(csv.DictReader(io.StringIO(pack.to_csv())))
    assert len(rows) == 2 and rows[0]["note"] == "Meets 70% threshold" and rows[0]["case_ref"] == "UWR-17"
    pack.unpin(pack.items[0].id)
    assert len(pack.items) == 1


# ------------------------------------------------------------------------------ sample questions
def test_sample_question_categories_shape():
    from uwlens.questions import CATEGORIES

    assert len(CATEGORIES) >= 5
    assert all(len(qs) == 3 for qs in CATEGORIES.values())
    texts = [q.text for qs in CATEGORIES.values() for q in qs]
    assert len(texts) == len(set(texts))
    assert any(not t.isascii() for t in texts)  # Japanese samples included


def test_every_sample_question_finds_its_document(engine):
    from uwlens.questions import CATEGORIES

    for cat, qs in CATEGORIES.items():
        for q in qs:
            top = names(engine.search(q.text, k=5))
            assert set(top) & set(q.expect), (cat, q.text, top)
