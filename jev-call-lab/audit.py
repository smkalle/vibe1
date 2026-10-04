"""Offline audit helpers for runs: contract, leak, replay-risk, hygiene, summaries.

All functions are pure (no Streamlit, no network) so the workbench UI and the
pytest suite share them. Heavy E1-E7 reporting stays in run_evals; this module
covers per-run checks that are cheap enough to run interactively.
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import evaluate
import run_evals
from jev_client import cache_key
from questions import REVIEW, WILL_BOOK, prefix_state
from simulate import forecast_points

HERE = Path(__file__).parent

# Patterns that must never appear in results, fixtures, reports or logs.
# Pattern-based only: the real key is never read, compared, or printed here.
SECRET_PATTERNS = (
    r"sk-or-v1-[A-Za-z0-9_-]{10,}",
    r"(?i)bearer\s+[A-Za-z0-9._~+/-]+",
    r"(?i)api[_-]?key\s*[:=]\s*\S+",
)
SECRET_KEY_NAMES = ("authorization", "api_key", "apikey", "secret", "openrouter_api_key")


def _strings_with_paths(obj, path="$"):
    """Yield (path, string) for every string in a nested structure."""
    if isinstance(obj, str):
        yield path, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _strings_with_paths(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _strings_with_paths(v, f"{path}[{i}]")


def scan_for_secrets(obj, source="<memory>", max_findings=20):
    """Pattern-scan a JSON-able object for leaked credentials. Returns hit paths."""
    hits = []
    compiled = [re.compile(p) for p in SECRET_PATTERNS]
    for path, s in _strings_with_paths(obj, source):
        low_path = path.lower()
        if any(name in low_path for name in SECRET_KEY_NAMES) and s.strip():
            # A value stored under a secret-looking key (tokens counts are ints, not strings).
            hits.append(path)
        elif any(rx.search(s) for rx in compiled):
            hits.append(path)
        if len(hits) >= max_findings:
            break
    return hits


def scan_artifact_files(paths, max_findings=20):
    """Scan result/fixture/report files for leaked credentials. Returns 'file: path' hits."""
    hits = []
    for p in paths:
        try:
            text = Path(p).read_text()
            # Workspace runs are JSON Lines (one result per call); everything else is one JSON document.
            obj = [json.loads(line) for line in text.splitlines() if line.strip()] if str(p).endswith(".jsonl") \
                else json.loads(text)
        except (OSError, ValueError) as e:  # unreadable files are reported, not fatal
            hits.append(f"{p}: <unreadable: {e}>")
            continue
        for h in scan_for_secrets(obj, source=str(p), max_findings=max_findings - len(hits)):
            hits.append(h)
            if len(hits) >= max_findings:
                return hits
    return hits


def contract_summary(results):
    """E1-style contract check on one run's results."""
    ok, detail, problems = run_evals.e1_contract(results)
    return {"ok": ok, "detail": detail, "problems": problems}


def _utterance_vocab():
    try:
        return set(json.loads((HERE / "data" / "sgd_utterance_vocab.json").read_text()))
    except OSError:
        return None


def no_leak_summary(calls):
    """E2 no-leak check scoped to one run's calls (prefix, call_id, label keys, + words if vocab exists)."""
    ok, detail, problems = run_evals.e2_no_leak(calls, _utterance_vocab() or set())
    return {"ok": ok, "detail": detail, "problems": problems}


def duplicate_states(calls, model, limit=20):
    """Find request cache keys shared by distinct (call, turn) locations.

    Same key + record mode + parallel workers + a stochastic endpoint means the
    fixture file is last-write-wins while each result keeps its own response,
    so replay serves one value to both. Returns {"n_groups": int, "groups": [...]}.
    """
    owners = defaultdict(set)
    for call in calls:
        events = call.get("events", [])
        for i in forecast_points(events):
            owners[cache_key(model, prefix_state(call, events[: i + 1], i), WILL_BOOK)].add((call["call_id"], i))
        if events:
            owners[cache_key(model, prefix_state(call, events, len(events) - 1), REVIEW)].add((call["call_id"], "review"))
    groups = [
        {"key": key[:12], "locations": sorted(locs)}
        for key, locs in owners.items() if len(locs) > 1
    ]
    return {"n_groups": len(groups), "groups": groups[:limit]}


def summarize_runs(named_runs):
    """One analytics row per run (JSON-serializable, no Streamlit)."""
    rows = []
    for name, run in named_runs.items():
        results = run["results"]
        m = evaluate.metrics(results) if results else {}
        bad = sum(bool(t.get("errors")) for r in results for t in r["turns"]) + sum(bool(r.get("review_errors")) for r in results)
        rows.append({
            "run": name,
            "mode": run.get("settings", {}).get("mode"),
            "model": run.get("settings", {}).get("model"),
            "dataset": run.get("dataset"),
            "calls": m.get("n_calls", len(results)),
            "booked": m.get("n_booked"),
            "requests": m.get("n_requests"),
            "auc_mid": (m.get("auc") or {}).get("mid"),
            "auc_pre_outcome": (m.get("auc") or {}).get("pre_outcome"),
            "auc_end": (m.get("auc") or {}).get("end"),
            "brier_pre_outcome": (m.get("brier") or {}).get("pre_outcome"),
            "cost_usd": (m.get("cost") or {}).get("total_usd"),
            "latency_p50_ms": (m.get("latency_ms") or {}).get("p50"),
            "latency_p95_ms": (m.get("latency_ms") or {}).get("p95"),
            "contract_errors": bad,
            "fm_agreement": (m.get("failure_mode") or {}).get("agreement_with_rule_reference"),
        })
    return rows


def audit_run(results, calls, model):
    """Full per-run audit. `calls` are the dataset calls backing `results`."""
    by_id = {c["call_id"]: c for c in calls}
    run_calls = [by_id[r["call_id"]] for r in results if r["call_id"] in by_id]
    return {
        "n_calls": len(results),
        "n_calls_with_data": len(run_calls),
        "contract": contract_summary(results),
        "no_leak": no_leak_summary(run_calls),
        "duplicates": duplicate_states(run_calls, model),
        "hygiene": scan_for_secrets(results, source="<run results>"),
        "metrics": evaluate.metrics(results) if results else {},
    }
