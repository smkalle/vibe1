import json

import audit
from simulate import run


def _seed_results(seed_calls):
    return run(list(seed_calls.values()), workers=2, quiet=True)


# hygiene ---------------------------------------------------------------------
def test_clean_run_has_no_secrets(seed_calls):
    assert audit.scan_for_secrets(_seed_results(seed_calls)) == []


def test_planted_fake_key_is_found(seed_calls):
    dirty = _seed_results(seed_calls)
    dirty[0]["turns"][0]["p_book"] = 0.5
    dirty[0]["note"] = "sk-or-v1-TESTFAKEKEY1234567890"
    hits = audit.scan_for_secrets(dirty)
    assert hits and any("note" in h for h in hits)


def test_bearer_header_shape_is_found():
    assert audit.scan_for_secrets({"h": "Bearer abcDEF123._~-"}) != []
    assert "<unreadable" in audit.scan_artifact_files(["/nonexistent/x.json"])[0]


# contract + leak ---------------------------------------------------------------
def test_contract_summary_clean(seed_calls):
    s = audit.contract_summary(_seed_results(seed_calls))
    assert s["ok"] and not s["problems"]


def test_no_leak_summary_clean(seed_calls):
    s = audit.no_leak_summary(list(seed_calls.values()))
    assert s["ok"] and not s["problems"]


# replay-risk --------------------------------------------------------------------
def test_no_shared_keys_in_seed_calls(seed_calls):
    assert audit.duplicate_states(list(seed_calls.values()), "typesafe/jev-1.13")["n_groups"] == 0


def test_known_shared_prefix_in_sgd_test():
    import pathlib
    calls = json.loads((pathlib.Path(__file__).resolve().parents[1] / "data" / "sgd_test.json").read_text())
    d = audit.duplicate_states(calls, "typesafe/jev-1.13")
    assert d["n_groups"] > 0
    assert any(("sgd-test-5_00068", 2) in g["locations"] and ("sgd-test-6_00005", 2) in g["locations"]
               for g in d["groups"])


# summaries -----------------------------------------------------------------------
def test_summarize_runs_row(seed_calls):
    rows = audit.summarize_runs({"r": {"results": _seed_results(seed_calls),
                                       "settings": {"mode": "mock", "model": "m"},
                                       "dataset": "synthetic_calls.json"}})
    assert len(rows) == 1 and rows[0]["calls"] == 3 and rows[0]["contract_errors"] == 0


def test_audit_run_end_to_end(seed_calls):
    calls = list(seed_calls.values())
    rep = audit.audit_run(_seed_results(seed_calls), calls, "typesafe/jev-1.13")
    assert rep["contract"]["ok"] and rep["no_leak"]["ok"]
    assert rep["duplicates"]["n_groups"] == 0 and rep["hygiene"] == []
    assert rep["metrics"]["n_calls"] == 3
