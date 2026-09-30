"""Run evals E1-E7 from specs/jev-call-analysis.md and write evals/REPORT.md.

    python run_evals.py                      # mock: simulates, then gates E1-E7
    python run_evals.py --results results/sgd_test_live.json --live
                                             # live: E4/E5 are reported, not gated
"""
import argparse
import json
import os
import re
import tempfile
import threading
from pathlib import Path

import baselines
import evaluate
import jev_client
import mock_jev
import schema
from questions import REVIEW, WILL_BOOK, prefix_state
from simulate import forecast_points, run

HERE = Path(__file__).parent
DATA = HERE / "data"


def load(name):
    return json.loads((DATA / name).read_text())


# E1 ------------------------------------------------------------------------
def e1_contract(results):
    bad = []
    for r in results:
        for t in r["turns"]:
            bad += [f"{r['call_id']}@{t['event_index']}: {e}" for e in t.get("errors", [])]
        bad += [f"{r['call_id']} review: {e}" for e in r.get("review_errors", [])]
    n = sum(len(r["turns"]) + (1 if r.get("review") else 0) for r in results)
    return not bad, f"{n - len(bad)}/{n} responses valid", bad[:10]


# E2 ------------------------------------------------------------------------
def _strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)


def e2_no_leak(calls, utterance_words):
    """Prefix-only, no id/label in state, and no transcript words beyond the closed vocabulary."""
    allowed = schema.vocabulary() | set(prefix_state_keys())
    problems, checked = [], 0
    for call in calls:
        events = call["events"]
        for i in forecast_points(events):
            state = prefix_state(call, events[: i + 1], i)
            checked += 1
            if state["trace_so_far"] != events[: i + 1]:
                problems.append(f"{call['call_id']}@{i}: prefix mismatch")
            text = json.dumps(state)
            if call["call_id"] in text:
                problems.append(f"{call['call_id']}@{i}: call_id in state")
            if any(k.startswith("label") for k in state):
                problems.append(f"{call['call_id']}@{i}: label key in state")
            for s in _strings(state):
                for tok in re.findall(r"[A-Za-z]+", s):
                    t = tok.lower()
                    if t in utterance_words and t not in allowed:
                        problems.append(f"{call['call_id']}@{i}: transcript word {t!r}")
    return not problems, f"{checked} prefix states checked", problems[:10]


def prefix_state_keys():
    """Every word that may legitimately appear in a state (keys, notes, site/specialty values)."""
    probe = prefix_state({"call_id": "x", "site": "", "specialty": ""}, [{"t_ms": 0, "stage": "", "event": "", "actor": ""}], 0)
    words = set()
    for s in _strings(probe):
        words |= {t.lower() for t in re.findall(r"[A-Za-z]+", s)}
    return words


# E3 ------------------------------------------------------------------------
def e3_reducer(calls):
    problems = []
    for c in calls:
        problems += [f"{c['call_id']}: {e}" for e in schema.validate_call(c)]
    booked = sum(c["label_booked"] for c in calls)
    return not problems, f"{len(calls)} calls, {booked} booked / {len(calls) - booked} not", problems[:10]


# E4 ------------------------------------------------------------------------
def e4_seed(results):
    by = {r["call_id"]: [t["p_book"] for t in r["turns"]] for r in results}
    ev = {r["call_id"]: [t["event"] for t in r["turns"]] for r in results}
    checks = {}
    if "book-001" in by:
        p = by["book-001"]
        checks["book-001 ends > 0.8"] = p[-1] > 0.8
        checks["book-001 rises over the call"] = p[-1] > p[0]
    if "fail-002" in by:
        checks["fail-002 ends < 0.2"] = by["fail-002"][-1] < 0.2
    if "recover-003" in by:
        p, e = by["recover-003"], ev["recover-003"]
        first_fail = e.index("identity_check_failed")
        checks["recover-003 dips after first failure"] = p[first_fail] < max(p[: first_fail] or [0.5])
        checks["recover-003 ends > 0.8"] = p[-1] > 0.8
    ok = bool(checks) and all(checks.values())
    return ok, ", ".join(f"{k}: {'ok' if v else 'FAIL'}" for k, v in checks.items()), []


# E5 ------------------------------------------------------------------------
def e5_signal(results, train_calls, test_calls):
    m = evaluate.metrics(results)
    b = baselines.evaluate_all(train_calls, test_calls)
    ok = (m["auc"]["pre_outcome"] or 0) >= 0.70 and (m["auc"]["end"] or 0) >= 0.95
    rows = {"jev": m["auc"], **{k: v["auc"] for k, v in b.items()}}
    return ok, rows, []


# E6 ------------------------------------------------------------------------
def e6_record_replay(calls):
    server = mock_jev.make_server(0)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    old = {k: os.environ.get(k) for k in ("JEV_MODE", "JEV_URL", "JEV_FIXTURES", "OPENROUTER_API_KEY")}
    try:
        with tempfile.TemporaryDirectory() as fx:
            os.environ.update(
                JEV_URL=f"http://127.0.0.1:{port}/api/alpha/decisions",
                JEV_FIXTURES=fx,
                OPENROUTER_API_KEY="mock-key-not-real",
                JEV_MODE="record",
            )
            recorded = run(calls, workers=4, quiet=True)
            n_fixtures = len(list(Path(fx).glob("*.json")))
            os.environ["JEV_MODE"] = "replay"
            replayed = run(calls, workers=4, quiet=True)
            strip = lambda rs: [
                {**r, "turns": [{k: v for k, v in t.items() if k != "latency_ms"} for t in r["turns"]], "review_latency_ms": 0}
                for r in rs
            ]
            same = strip(recorded) == strip(replayed)
            miss_is_error = False
            try:
                jev_client.decide({"never": "recorded"}, WILL_BOOK)
            except jev_client.JevError:
                miss_is_error = True
    finally:
        server.shutdown()
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    ok = same and miss_is_error and n_fixtures > 0
    return ok, f"{n_fixtures} fixtures over HTTP; replay identical={same}; replay miss raises={miss_is_error}", []


# E7 ------------------------------------------------------------------------
def e7_cost(results, all_sgd_calls):
    m = evaluate.metrics(results)
    proj = evaluate.project_cost(results, all_sgd_calls)
    ok = proj["usd"] < 1.0
    return ok, {"this_run": m["cost"], "projected_full_sgd": proj}, []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", help="SGD test results from simulate.py (default: simulate now)")
    ap.add_argument("--seed-results", help="seed/synthetic results (default: simulate now)")
    ap.add_argument("--live", action="store_true", help="report E4/E5 instead of gating them")
    ap.add_argument("--out", default=str(HERE / "evals" / "REPORT.md"))
    args = ap.parse_args()

    synthetic = load("synthetic_calls.json")
    sgd = {s: load(f"sgd_{s}.json") for s in ("train", "dev", "test")}
    utterance_words = set(json.loads((DATA / "sgd_utterance_vocab.json").read_text()))

    seed_calls = [c for c in synthetic if c["call_id"] in {"book-001", "fail-002", "recover-003"}]
    seed_results = json.loads(Path(args.seed_results).read_text()) if args.seed_results else run(seed_calls, quiet=True)
    test_results = json.loads(Path(args.results).read_text()) if args.results else run(sgd["test"], workers=8, quiet=True)
    if not args.results:
        (HERE / "results").mkdir(exist_ok=True)
        (HERE / "results" / "sgd_test_mock.json").write_text(json.dumps(test_results, indent=1))

    all_calls = synthetic + sgd["train"] + sgd["dev"] + sgd["test"]
    evals = [
        ("E1", "Contract", True, e1_contract(seed_results + test_results)),
        ("E2", "No leak", True, e2_no_leak(all_calls, utterance_words)),
        ("E3", "Reducer fidelity", True, e3_reducer(all_calls)),
        ("E4", "Seed trajectories", not args.live, e4_seed(seed_results)),
        ("E5", "Signal on SGD test", not args.live, e5_signal(test_results, sgd["train"], sgd["test"])),
        ("E6", "Record -> replay", True, e6_record_replay(seed_calls)),
        ("E7", "Cost accounting", True, e7_cost(test_results, sgd["train"] + sgd["dev"] + sgd["test"])),
    ]

    mode = "live" if args.live else "mock"
    lines = [f"# Eval report ({mode})", "", "| ID | Eval | Gate | Result | Detail |", "|---|---|---|---|---|"]
    all_ok = True
    for eid, name, gated, (ok, detail, problems) in evals:
        status = ("PASS" if ok else "FAIL") if gated else ("ok" if ok else "below target")
        all_ok &= ok or not gated
        d = detail if isinstance(detail, str) else "see below"
        lines.append(f"| {eid} | {name} | {'gate' if gated else 'report'} | {status} | {d} |")
    for eid, name, _, (ok, detail, problems) in evals:
        if not isinstance(detail, str) or problems:
            lines += ["", f"## {eid} {name}", ""]
            if not isinstance(detail, str):
                lines += ["```json", json.dumps(detail, indent=2), "```"]
            if problems:
                lines += ["Problems (first 10):", ""] + [f"- {p}" for p in problems]
    lines += ["", "## SGD test trajectories and review (evaluate.py)", "", "```", evaluate.render(test_results, max_calls=4), "```"]
    lines += ["", "## Seed trajectories", "", "```", evaluate.render(seed_results), "```", ""]
    lines.insert(1, f"\n**Overall: {'PASS' if all_ok else 'FAIL'}** (Jev mode: {jev_client.mode()})\n")
    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text("\n".join(lines))
    print("\n".join(lines[:14]))
    print(f"\nwrote {args.out}")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
