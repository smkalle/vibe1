"""Workbench v2 P1 evals: W1-W5, W8, W13 (spec v2 §12), plus runner resume and key hygiene."""
import json
import random
import threading

import pytest

import audit
import benchmark
import jev_client
import mock_jev
import sampling
import stats
import workspace


@pytest.fixture
def synthetic():
    return workspace.load_corpus("synthetic")


@pytest.fixture
def project(tmp_path):
    return workspace.Project("t", base=tmp_path / "ws").ensure()


def _mock_cfg():
    return jev_client.Config(mode="mock")


# W3 sampling -----------------------------------------------------------------------
def test_w3_sample_is_seeded_and_stratified(synthetic):
    a = sampling.freeze("synthetic", synthetic, 40, seed=3)
    b = sampling.freeze("synthetic", synthetic, 40, seed=3)
    c = sampling.freeze("synthetic", synthetic, 40, seed=4)
    assert a == b and a["call_ids"] != c["call_ids"] and a["n"] == 40
    booked_rate = sum(x["label_booked"] for x in synthetic) / len(synthetic)
    assert abs(a["n_booked"] - 40 * booked_rate) <= 1  # proportional within one call


def test_w3_never_single_class_where_first_n_was():
    """R1: the first 30 sgd_test calls are all unbooked; a stratified sample of 10 has both outcomes."""
    sgd = workspace.load_corpus("sgd_test")
    assert len({c["label_booked"] for c in sgd[:30]}) == 1
    for n in (2, 3, 10):
        s = sampling.freeze("sgd_test", sgd, n, seed=1)
        assert 0 < s["n_booked"] < n


def test_w3_tiny_skewed_corpus_still_gets_both_classes():
    calls = [{"call_id": f"c{i}", "label_booked": i == 0, "source": "x", "specialty": "y"} for i in range(50)]
    s = sampling.stratified_ids(calls, 2, seed=0)
    assert "c0" in s and len(s) == 2


def test_auc_halfwidth_shrinks_with_n():
    assert sampling.auc_halfwidth(300, 0.5) < sampling.auc_halfwidth(100, 0.5) < 0.15
    assert sampling.auc_halfwidth(10, 0.0) is None


# W4 bootstrap ------------------------------------------------------------------------
def test_rank_auc_matches_pairwise_definition():
    import evaluate
    rng = random.Random(1)
    p = [round(rng.random(), 1) for _ in range(60)]
    y = [rng.random() < 0.5 for _ in range(60)]
    assert stats.auc(p, y) == pytest.approx(evaluate.auc([{"p": a, "y": b} for a, b in zip(p, y)]))


def test_w4_bootstrap_resamples_calls_and_covers_true_auc():
    """Known AUC 0.75 (normal scores, d' = 0.954); 95% CI should cover it in >= 93% of trials."""
    rng = random.Random(42)
    covered, trials = 0, 120
    for t in range(trials):
        y = [i % 2 == 0 for i in range(120)]
        p = [rng.gauss(0.954 if yi else 0.0, 1) for yi in y]
        out = stats.paired_bootstrap({"m": lambda idx: stats.auc([p[i] for i in idx], [y[i] for i in idx])},
                                     len(p), b=300, seed=t)
        lo, hi = out["ci"]["m"]
        covered += lo <= 0.75 <= hi
    assert covered / trials >= 0.90  # 120 trials: allow sampling noise around the 95% target


def test_paired_diff_is_zero_for_identical_approaches():
    p = [0.1, 0.9, 0.4, 0.6] * 10
    y = [False, True, False, True] * 10
    f = lambda idx: stats.auc([p[i] for i in idx], [y[i] for i in idx])
    out = stats.paired_bootstrap({"a": f, "b": f}, len(p), b=200)
    assert out["diff"][("a", "b")] == (0.0, 0.0, 0.0)


def test_macro_f1_and_escalation():
    assert stats.macro_f1(["a", "b", "a"], ["a", "b", "b"]) == pytest.approx((2 / 3 + 2 / 3) / 2)
    pr = stats.precision_recall_at([0.1, 0.2, 0.9, 0.3], [False, True, True, False], 0.35)
    assert pr["flagged"] == 3 and pr["precision"] == pytest.approx(2 / 3) and pr["recall"] == 1.0


# W1 pairing --------------------------------------------------------------------------
def test_w1_rejects_unpaired_runs(synthetic, project):
    s1 = sampling.freeze("synthetic", synthetic, 30, seed=1)
    s2 = sampling.freeze("synthetic", synthetic, 30, seed=2)
    rid = benchmark.run_approach(project, s2, synthetic, "jev", _mock_cfg(), workers=4)
    entry = {"manifest": project.manifest(rid), "results": project.results(rid)}
    with pytest.raises(benchmark.PairingError, match="sample_id"):
        benchmark.validate_paired(s1, {"jev": entry})
    bad_q = {"manifest": {**entry["manifest"], "sample_id": s1["sample_id"], "corpus_hash": s1["corpus_hash"],
                          "questions_hash": "other"}, "results": []}
    with pytest.raises(benchmark.PairingError, match="questions_hash"):
        benchmark.validate_paired(s1, {"jev": bad_q})
    with pytest.raises(benchmark.PairingError, match="outside sample"):
        benchmark.validate_paired(s1, {"jev": {"manifest": None, "results": entry["results"]}})
    benchmark.validate_paired(s2, {"jev": entry})  # the matching sample is fine


# W2 no stale rule scores ----------------------------------------------------------------
def test_w2_rule_scores_follow_the_sample(synthetic):
    train = workspace.load_train("synthetic")
    a = benchmark.rule_results(sampling.freeze("synthetic", synthetic, 30, seed=1), synthetic, train)
    b = benchmark.rule_results(sampling.freeze("synthetic", synthetic, 30, seed=2), synthetic, train)
    assert {r["call_id"] for r in a["rules_b1"]} != {r["call_id"] for r in b["rules_b1"]}


def test_rules_b2_never_trains_on_the_sample(synthetic, monkeypatch):
    s = sampling.freeze("synthetic", synthetic, 30, seed=1)
    seen = []
    real = benchmark.baselines._prefixes
    monkeypatch.setattr(benchmark.baselines, "_prefixes", lambda calls: (seen.extend(c["call_id"] for c in calls), real(calls))[1])
    benchmark.rule_results(s, synthetic, synthetic)
    assert seen and not set(seen) & set(s["call_ids"])


# W5 verdict rule ------------------------------------------------------------------------
def _appr(point, ci, cost, p95, label=None):
    return {"label": label or "x", "metrics": {"auc_pre_outcome": {"point": point, "ci": ci}}, "cost_per_1k": cost,
            "latency": {"p95": p95}}


UC = {"label": "In-call", "primary": "auc_pre_outcome", "latency_budget_ms": 800, "metric_words": "accuracy"}


@pytest.mark.parametrize("case", [
    # clear win: CI of the difference excludes 0
    dict(apps={"jev": _appr(0.80, (0.76, 0.84), 0.3, 500), "glm": _appr(0.70, (0.65, 0.75), 3.0, 700)},
         diffs={"jev|glm": (0.10, 0.06, 0.14)}, winner="jev", by="accuracy", conf="high"),
    # tie -> cheaper wins even with a lower point estimate
    dict(apps={"glm": _appr(0.81, (0.76, 0.86), 3.0, 700), "jev": _appr(0.79, (0.74, 0.84), 0.3, 500)},
         diffs={"glm|jev": (0.02, -0.03, 0.07)}, winner="jev", by="cost", conf="medium"),
    # tie and equal cost -> faster wins
    dict(apps={"a": _appr(0.80, (0.75, 0.85), 0.0, 300), "b": _appr(0.80, (0.75, 0.85), 0.0, 100)},
         diffs={"a|b": (0.0, -0.04, 0.04), "b|a": (0.0, -0.04, 0.04)}, winner="b", by="latency", conf="medium"),
    # the most accurate approach breaks the latency budget and is excluded
    dict(apps={"glm": _appr(0.90, (0.86, 0.94), 3.0, 3900), "jev": _appr(0.75, (0.70, 0.80), 0.3, 500)},
         diffs={}, winner="jev", by="accuracy", conf="high", excluded="glm"),
    # underpowered: CI half-width > 0.10
    dict(apps={"jev": _appr(0.80, (0.60, 0.95), 0.3, 500)}, diffs={}, winner="jev", by="accuracy", conf="low"),
])
def test_w5_verdict_rule(case):
    v = benchmark.verdict("in_call", UC, case["apps"], {"auc_pre_outcome": case["diffs"]})
    assert (v["winner"], v["decided_by"], v["confidence"]) == (case["winner"], case["by"], case["conf"])
    if "excluded" in case:
        assert "budget" in v["excluded"][case["excluded"]]
    assert v["sentence"].startswith("Use ")


def test_w5_unknown_latency_is_flagged_not_excluded():
    v = benchmark.verdict("in_call", UC, {"jev": _appr(0.8, (0.75, 0.85), 0.3, None)}, {"auc_pre_outcome": {}})
    assert v["winner"] == "jev" and v["latency_unverified"] == ["jev"]


def test_w5_no_data():
    v = benchmark.verdict("in_call", UC, {"jev": _appr(None, (None, None), 0.3, 1)}, {"auc_pre_outcome": {}})
    assert v["status"] == "no_data" and v["winner"] is None


# W8 latency provenance -----------------------------------------------------------------
def test_w8_mock_latency_never_counts(synthetic, project):
    s = sampling.freeze("synthetic", synthetic, 12, seed=1)
    rid = benchmark.run_approach(project, s, synthetic, "jev", _mock_cfg(), workers=2)
    lat = benchmark._latency(project.results(rid), project.manifest(rid), "llm")
    assert lat["p50"] is None and lat["source"] == "none"
    probe = benchmark.latency_probe("jev", _mock_cfg(), s, synthetic, n=5, warmup=1)
    assert probe["p50"] is None and probe["sources"] == ["mock"] and probe["concurrency"] == 1


def test_w8_probe_measures_live_calls_at_concurrency_one(synthetic):
    srv = mock_jev.make_server(0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    cfg = jev_client.Config(mode="live", api_key="k", url=f"http://127.0.0.1:{srv.server_address[1]}/api/alpha/decisions")
    try:
        probe = benchmark.latency_probe("jev", cfg, sampling.freeze("synthetic", synthetic, 6, 1), synthetic, n=4, warmup=1)
    finally:
        srv.shutdown()
    assert probe["sources"] == ["measured"] and probe["p50"] > 0 and probe["n"] == 4


# W13 provenance on import --------------------------------------------------------------------
def test_w13_import_keeps_stated_provenance_and_rejects_foreign_calls(synthetic, project):
    s = sampling.freeze("synthetic", synthetic, 10, seed=1)
    other = sampling.freeze("synthetic", synthetic, 10, seed=9)
    from simulate import run
    by = {c["call_id"]: c for c in synthetic}
    res = run([by[i] for i in s["call_ids"]], workers=2, quiet=True)
    rid = project.import_results(s, res, "jev", "live", "typesafe/jev-1.13-20260917", benchmark.QUESTIONS_HASH)
    m = project.manifest(rid)
    assert (m["mode"], m["model"], m["corpus_id"], m["source"]) == ("live", "typesafe/jev-1.13-20260917", "synthetic", "imported")
    with pytest.raises(ValueError, match="not in sample"):
        project.import_results(other, res, "jev", "live", "m", benchmark.QUESTIONS_HASH)


# runner: resume, cap, scorecard end-to-end, hygiene -----------------------------------------------
def test_run_resumes_without_duplicates(synthetic, project, monkeypatch):
    s = sampling.freeze("synthetic", synthetic, 20, seed=1)
    calls = {"n": 0}
    real = benchmark.forecast_call

    def flaky(c, *a, **k):
        calls["n"] += 1
        if calls["n"] > 8:
            raise RuntimeError("network down")
        return real(c, *a, **k)

    monkeypatch.setattr(benchmark, "forecast_call", flaky)
    rid = benchmark.run_approach(project, s, synthetic, "jev", _mock_cfg(), workers=1)
    assert project.manifest(rid)["status"] == "partial" and len(project.done_ids(rid)) == 8
    monkeypatch.setattr(benchmark, "forecast_call", real)
    benchmark.run_approach(project, s, synthetic, "jev", _mock_cfg(), workers=3, run_id=rid)
    ids = [r["call_id"] for r in project.results(rid)]
    assert sorted(ids) == s["call_ids"] and project.manifest(rid)["status"] == "done"


def test_cost_cap_pauses_the_run(synthetic, project):
    s = sampling.freeze("synthetic", synthetic, 20, seed=1)
    rid = benchmark.run_approach(project, s, synthetic, "jev", _mock_cfg(), workers=1, cap_usd=1e-6)
    m = project.manifest(rid)
    assert m["status"] == "paused_cap" and m["n_done"] < 20


def test_scorecard_end_to_end_and_key_never_on_disk(synthetic, project):
    s = project.save_sample(sampling.freeze("synthetic", synthetic, 40, seed=5))
    srv = mock_jev.make_server(0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    secret = "sk-or-v1-" + "a" * 64
    cfg = jev_client.Config(mode="live", api_key=secret, url=f"http://127.0.0.1:{srv.server_address[1]}/api/alpha/decisions")
    try:
        benchmark.run_approach(project, s, synthetic, "jev", cfg, workers=4)
    finally:
        srv.shutdown()
    sc = benchmark.scorecard(s, benchmark.load_entries(project, s, ["rules_b1", "rules_b2", "jev"]), b=200)
    assert sc["n_calls"] == 40 and set(sc["approaches"]) == {"rules_b1", "rules_b2", "jev"}
    assert sc["approaches"]["jev"]["metrics"]["auc_pre_outcome"]["ci"][0] is not None
    assert sc["approaches"]["rules_b1"]["metrics"]["failure_mode_f1"]["point"] is None
    assert sc["verdicts"]["post_call_qa"]["winner"] == "jev"  # the only approach with failure-mode labels
    files = list(project.dir.rglob("*"))
    blob = "".join(p.read_text() for p in files if p.is_file())
    assert secret not in blob and "api_key" not in blob
    assert not audit.scan_artifact_files([p for p in files if p.is_file()])


def test_rules_b2_unavailable_on_whole_corpus_sample_is_explained(synthetic):
    s = sampling.freeze("synthetic", synthetic, len(synthetic), seed=1)
    ents = {k: {"manifest": None, "results": v} for k, v in benchmark.rule_results(s, synthetic, synthetic).items()}
    assert "rules_b2" not in ents
    assert "outside the sample" in benchmark.unavailable(s, ["rules_b1", "rules_b2"], ents)["rules_b2"]
