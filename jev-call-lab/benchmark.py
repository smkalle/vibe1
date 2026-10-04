"""Paired benchmark engine (spec v2 §5–§7): approaches, runs, scorecard, verdict.

    python benchmark.py --project demo --corpus synthetic --n 60 --approaches rules_b1,rules_b2,jev,glm

Every approach scores ONE frozen sample (sampling.py). Comparisons refuse mismatched samples,
corpora or question sets (W1). CIs are a paired cluster bootstrap over calls (stats.py, W4).
"""
import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

import baselines
import jev_client
import llm_client
import sampling
import stats
import workspace
from evaluate import POINTS
from jev_client import REAL_LATENCY
from questions import REVIEW, WILL_BOOK, prefix_state
from simulate import attempted_booking, decision_index, forecast_call, forecast_points

QUESTIONS_HASH = hashlib.sha256(json.dumps([WILL_BOOK, REVIEW], sort_keys=True).encode()).hexdigest()[:12]

APPROACHES = {
    "rules_b1": {"label": "Rule: furthest stage", "kind": "rule", "about": "Free, offline, no training needed."},
    "rules_b2": {"label": "Rule: logistic regression", "kind": "rule",
                 "about": "Free, offline; trained on labelled calls outside the sample."},
    "jev": {"label": "Jev", "kind": "llm", "about": "TypeSafe Jev via OpenRouter Decisions API; input tokens only."},
    "glm": {"label": "GLM-5.3", "kind": "llm", "about": "Chat model via OpenRouter; reasoning tokens billed."},
}

USE_CASES = {
    "in_call": {"label": "In-call escalation", "primary": "auc_pre_outcome", "latency_budget_ms": 800,
                "question": "Decide during the call whether it is heading for no booking.",
                "metric_words": "ranking accuracy before the outcome is known"},
    "post_call_qa": {"label": "Post-call QA review", "primary": "failure_mode_f1", "latency_budget_ms": None,
                     "question": "Label why a finished call succeeded or failed.",
                     "metric_words": "failure-mode accuracy (macro-F1 vs the rule reference)"},
    "bulk": {"label": "Bulk analytics", "primary": "auc_mid", "latency_budget_ms": None,
             "question": "Score large volumes of calls for reporting and forecasting.",
             "metric_words": "mid-call ranking accuracy"},
}

METRICS = ("auc_mid", "auc_pre_outcome", "auc_attempted", "auc_end", "failure_mode_f1")


class PairingError(ValueError):
    pass


def prompt_hash(approach: str) -> str | None:
    if approach == "glm":
        return hashlib.sha256(json.dumps(llm_client.build_messages({"state": "<state>"}, WILL_BOOK)).encode()).hexdigest()[:12]
    return None


# ---------------------------------------------------------------- rules
def rule_results(sample: dict, corpus: list, train: list) -> dict:
    """B1 and (when trainable) B2 on the sample, as results-shaped dicts. B2 never trains on the sample."""
    ids = set(sample["call_ids"])
    calls = [c for c in corpus if c["call_id"] in ids]
    held_out = [c for c in train if c["call_id"] not in ids]
    out = {"rules_b1": _rule_rows(calls, baselines.stage_reached, "rules_b1")}
    if len(held_out) >= 20 and len({c["label_booked"] for c in held_out}) == 2:
        rows = list(baselines._prefixes(held_out))
        model = baselines.LogReg().fit([baselines.features(p) for _, _, p in rows],
                                       [float(c["label_booked"]) for c, _, _ in rows])
        out["rules_b2"] = _rule_rows(calls, lambda p: model.predict_one(baselines.features(p)), "rules_b2")
    return out


def _rule_rows(calls, score, name):
    rows = []
    for c in calls:
        ev = c["events"]
        rows.append({
            "call_id": c["call_id"], "label_booked": c["label_booked"], "n_events": len(ev),
            "decision_index": decision_index(ev), "attempted_booking": attempted_booking(ev),
            "reference_failure_mode": None, "review": None, "scorer": name,
            "turns": [{"event_index": i, "p_book": score(ev[: i + 1]), "latency_ms": 0.0,
                       "latency_source": "offline", "cost_usd": 0.0, "input_tokens": 0, "errors": [],
                       "t_ms": ev[i]["t_ms"], "stage": ev[i]["stage"], "event": ev[i]["event"], "ok": ev[i]["ok"]}
                      for i in forecast_points(ev)],
        })
    return rows


# ---------------------------------------------------------------- running LLM approaches
def decide_fn_for(approach: str, cfg):
    if approach == "glm":
        return partial(llm_client.decide, cfg=cfg)
    return partial(jev_client.decide, cfg=cfg)


# Jev bills ~2.2x the chars/4 token estimate (measured on the live 500-call run, evals/REPORT_live.md).
JEV_TOKEN_SCALE = 2.2


def estimate(approach: str, calls: list, review: bool = True) -> dict:
    """Pre-run estimate. usd = expected (calibrated on live usage); usd_max = worst case, used for caps."""
    if approach.startswith("rules"):
        return {"requests": 0, "usd": 0.0, "usd_max": 0.0, "usd_per_1k": 0.0}
    if approach == "glm":
        e = llm_client.project_cost(calls, review)
        usd, usd_max = e["usd_expected"], e["usd"]
    else:
        import evaluate
        e = evaluate.project_cost([], calls)
        if not review:
            e["requests"] -= len(calls)
        usd = e["usd"] * JEV_TOKEN_SCALE
        usd_max = usd * 1.25
    n = max(len(calls), 1)
    return {"requests": e["requests"], "usd": usd, "usd_max": usd_max, "usd_per_1k": usd / n * 1000}


def run_approach(project: workspace.Project, sample: dict, corpus: list, approach: str, cfg,
                 workers: int = 8, cap_usd: float | None = None, progress=None, run_id: str | None = None) -> str:
    """Score the sample with one LLM approach, appending per call. Resumes run_id if given.

    Stops (status 'paused_cap') once the actual spend reaches cap_usd. The key stays in cfg (memory).
    """
    if run_id is None:
        run_id = project.new_run(approach=approach, scorer=approach, mode=cfg.mode, model=cfg.model, url=cfg.url,
                                 effort=getattr(cfg, "effort", None), sample_id=sample["sample_id"],
                                 corpus_id=sample["corpus_id"], corpus_hash=sample["corpus_hash"],
                                 questions_hash=QUESTIONS_HASH, prompt_hash=prompt_hash(approach),
                                 concurrency=workers, n_calls=sample["n"], cap_usd=cap_usd)
    else:
        project.update_manifest(run_id, status="running", concurrency=workers, cap_usd=cap_usd)
    done = project.done_ids(run_id)
    by_id = {c["call_id"]: c for c in corpus}
    todo = [by_id[i] for i in sample["call_ids"] if i not in done]
    spent = project.manifest(run_id).get("cost_usd", 0.0)
    n_failed = 0
    decide_fn = decide_fn_for(approach, cfg)
    status = "done"
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {}
        queue = list(todo)

        def submit_next():
            if queue:
                c = queue.pop(0)
                futs[pool.submit(forecast_call, c, True, None, decide_fn, approach)] = c["call_id"]

        for _ in range(min(workers, len(queue))):
            submit_next()
        while futs:
            fut = next(as_completed(list(futs)))
            cid = futs.pop(fut)
            try:
                r = fut.result()
                project.append(run_id, "results", r)
                spent += sum(t["cost_usd"] for t in r["turns"]) + r.get("review_cost_usd", 0)
                done.add(cid)
            except Exception as e:  # noqa: BLE001 - recorded per call, run continues
                n_failed += 1
                project.append(run_id, "errors", {"call_id": cid, "error": f"{type(e).__name__}: {e}"})
            if progress:
                progress(len(done), sample["n"], spent)
            if cap_usd is not None and spent >= cap_usd and queue:
                status = "paused_cap"
                queue.clear()
            submit_next()
    if status == "done" and len(done) < sample["n"]:
        status = "partial"
    project.update_manifest(run_id, status=status, n_done=len(done), n_failed=n_failed, cost_usd=round(spent, 8),
                            finished=workspace.now())
    return run_id


def latency_probe(approach: str, cfg, sample: dict, corpus: list, n: int = 20, warmup: int = 3) -> dict:
    """Per-decision latency at concurrency 1 (spec §7.6). Only measured/recorded times count."""
    by_id = {c["call_id"]: c for c in corpus}
    states = []
    for cid in sample["call_ids"]:
        c = by_id[cid]
        for i in forecast_points(c["events"]):
            states.append(prefix_state(c, c["events"][: i + 1], i))
    decide_fn = decide_fn_for(approach, cfg)
    times, sources = [], set()
    for k, s in enumerate(states[: n + warmup]):
        r = decide_fn(s, WILL_BOOK)
        if k >= warmup:
            times.append(r["_latency_ms"])
            sources.add(r.get("_latency_source", "measured"))
    real = sources <= REAL_LATENCY and bool(times)
    return {"n": len(times), "concurrency": 1, "sources": sorted(sources),
            "p50": stats.percentile(times, 0.5) if real else None,
            "p95": stats.percentile(times, 0.95) if real else None,
            "p99": stats.percentile(times, 0.99) if real else None}


# ---------------------------------------------------------------- corpus profile
def profile(corpus: list) -> dict:
    n = len(corpus)
    labelled = [c for c in corpus if isinstance(c.get("label_booked"), bool)]
    rows = _rule_rows(labelled, baselines.stage_reached, "rules_b1")
    pre = [_point(r, "pre_outcome") for r in rows]
    difficulty = stats.auc(pre, [r["label_booked"] for r in rows])
    if difficulty is None:
        msg = "Not enough labelled calls of both outcomes to judge difficulty."
    elif difficulty >= 0.9:
        msg = ("The outcome is almost fully visible from structure. A free rule already ranks calls well; "
               "LLMs are unlikely to add accuracy here, so compare them on cost and on post-call labels.")
    elif difficulty >= 0.75:
        msg = "The outcome is partly visible from structure. LLMs may add some accuracy; the benchmark will tell."
    else:
        msg = "The outcome is hard to read from structure. This is where an LLM can earn its cost."
    return {
        "calls": n, "labelled": len(labelled),
        "booked_rate": sum(c["label_booked"] for c in labelled) / len(labelled) if labelled else None,
        "avg_events": sum(len(c["events"]) for c in corpus) / n if n else 0,
        "tool_coverage": sum(any(e["actor"] == "tool" for e in c["events"]) for c in corpus) / n if n else 0,
        "attempted_rate": sum(attempted_booking(c["events"]) for c in corpus) / n if n else 0,
        "difficulty_auc": difficulty, "difficulty_message": msg,
    }


# ---------------------------------------------------------------- scorecard
def _point(r, point):
    t = POINTS[point](r)
    return t["p_book"] if t else 0.5


def validate_paired(sample: dict, entries: dict, min_coverage: float = 0.95):
    """entries: {approach: {"manifest": {...} | None, "results": [...]}}. Raises PairingError (W1)."""
    ids = set(sample["call_ids"])
    for name, e in entries.items():
        m = e.get("manifest")
        if m is not None:
            for field, want in (("sample_id", sample["sample_id"]), ("corpus_hash", sample["corpus_hash"]),
                                ("questions_hash", QUESTIONS_HASH)):
                if m.get(field) != want:
                    raise PairingError(f"{name}: {field} {m.get(field)!r} does not match {want!r}")
        got = {r["call_id"] for r in e["results"]}
        if got - ids:
            raise PairingError(f"{name}: {len(got - ids)} calls are outside sample {sample['sample_id']}")
        if len(got) < min_coverage * len(ids):
            raise PairingError(f"{name}: covers {len(got)}/{len(ids)} sample calls (need {min_coverage:.0%})")


def _cost_per_1k(results):
    if not results:
        return None
    usd = sum(t.get("cost_usd", 0) for r in results for t in r["turns"]) + sum(r.get("review_cost_usd", 0) for r in results)
    return usd / len(results) * 1000


def _latency(results, manifest, kind):
    if kind == "rule":
        return {"p50": 0.0, "p95": 0.0, "source": "offline", "concurrency": None}
    probe = (manifest or {}).get("probe")
    if probe and probe.get("p50") is not None:
        return {"p50": probe["p50"], "p95": probe["p95"], "source": "probe", "concurrency": 1}
    times = [t["latency_ms"] for r in results for t in r["turns"] if t.get("latency_source", "measured") in REAL_LATENCY]
    if not times:
        return {"p50": None, "p95": None, "source": "none", "concurrency": None}
    return {"p50": stats.percentile(times, 0.5), "p95": stats.percentile(times, 0.95),
            "source": "under_load", "concurrency": (manifest or {}).get("concurrency")}


def scorecard(sample: dict, entries: dict, use_cases=USE_CASES, b: int = 600, seed: int = 0,
              threshold: float = 0.35) -> dict:
    """Paired metrics with CIs, per-approach cost/latency, and a verdict per use case."""
    validate_paired(sample, entries)
    common = [cid for cid in sample["call_ids"] if all(cid in {r["call_id"] for r in e["results"]} for e in entries.values())]
    n = len(common)
    per = {}
    for name, e in entries.items():
        by = {r["call_id"]: r for r in e["results"]}
        rs = [by[c] for c in common]
        per[name] = {
            "y": [bool(r["label_booked"]) for r in rs],
            "mid": [_point(r, "mid") for r in rs], "pre": [_point(r, "pre_outcome") for r in rs],
            "end": [_point(r, "end") for r in rs], "att": [bool(r["attempted_booking"]) for r in rs],
            "fm": [((r.get("review") or {}).get("failure_mode") or {}).get("choice") for r in rs],
            "ref": [r.get("reference_failure_mode") for r in rs], "results": rs,
        }
    # The rule reference failure mode lives on LLM results; share it so every approach is scored the same way.
    ref = next((p["ref"] for p in per.values() if any(p["ref"])), [None] * n)

    def fns(metric):
        out = {}
        for name, p in per.items():
            if metric == "auc_mid":
                out[name] = lambda idx, p=p: stats.auc([p["mid"][i] for i in idx], [p["y"][i] for i in idx])
            elif metric == "auc_pre_outcome":
                out[name] = lambda idx, p=p: stats.auc([p["pre"][i] for i in idx], [p["y"][i] for i in idx])
            elif metric == "auc_end":
                out[name] = lambda idx, p=p: stats.auc([p["end"][i] for i in idx], [p["y"][i] for i in idx])
            elif metric == "auc_attempted":
                out[name] = lambda idx, p=p: stats.auc([p["pre"][i] for i in idx if p["att"][i]],
                                                       [p["y"][i] for i in idx if p["att"][i]], min_class=5)
            elif metric == "failure_mode_f1" and any(p["fm"]):
                out[name] = lambda idx, p=p: stats.macro_f1([p["fm"][i] for i in idx], [ref[i] for i in idx])
        return out

    boot = {m: stats.paired_bootstrap(fns(m), n, b=b, seed=seed) if n else None for m in METRICS}
    approaches = {}
    for name, e in entries.items():
        p = per[name]
        kind = APPROACHES.get(name, {}).get("kind", "llm")
        reqs = sum(len(r["turns"]) + (1 if r.get("review") else 0) for r in p["results"])
        bad = sum(bool(t.get("errors")) for r in p["results"] for t in r["turns"]) + \
            sum(bool(r.get("review_errors")) for r in p["results"])
        m = e.get("manifest") or {}
        approaches[name] = {
            "label": APPROACHES.get(name, {}).get("label", name), "kind": kind,
            "mode": m.get("mode", "offline" if kind == "rule" else "unknown"), "model": m.get("model"),
            "run_id": m.get("run_id"), "metrics": {
                k: {"point": boot[k]["point"].get(name), "ci": boot[k]["ci"].get(name)} if boot[k] and name in boot[k]["point"]
                else {"point": None, "ci": (None, None)} for k in METRICS},
            "brier_pre_outcome": stats.brier(p["pre"], p["y"]), "ece_pre_outcome": stats.ece(p["pre"], p["y"]),
            "reliability_pre_outcome": stats.reliability(p["pre"], p["y"]),
            "escalation": stats.precision_recall_at(p["pre"], p["y"], threshold),
            "contract_error_rate": bad / reqs if reqs else 0.0,
            "cost_per_1k": _cost_per_1k(p["results"]), "latency": _latency(p["results"], m, kind),
        }
    diffs = {m: {f"{a}|{c}": v for (a, c), v in boot[m]["diff"].items()} if boot[m] else {} for m in METRICS}
    verdicts = {uc: verdict(uc, cfg, approaches, diffs) for uc, cfg in use_cases.items()}
    return {"sample_id": sample["sample_id"], "corpus_id": sample["corpus_id"], "n_calls": n,
            "n_sample": sample["n"], "bootstrap": b, "approaches": approaches, "diffs": diffs, "verdicts": verdicts}


def _fmt_usd(x):
    return "free" if not x else f"${x:,.2f}" if x >= 1 else f"${x:.3f}"


def verdict(uc: str, cfg: dict, approaches: dict, diffs: dict) -> dict:
    """Spec §7.4: drop constraint violators; a winner needs a paired CI excluding 0; else tie -> cost -> latency."""
    metric = cfg["primary"]
    budget = cfg.get("latency_budget_ms")
    excluded, unverified, cands = {}, [], []
    for name, a in approaches.items():
        pt = a["metrics"][metric]["point"]
        if pt is None:
            excluded[name] = "does not produce this metric" if metric == "failure_mode_f1" else "metric undefined on this sample"
            continue
        p95 = a["latency"]["p95"]
        if budget is not None:
            if p95 is None:
                unverified.append(name)
            elif p95 > budget:
                excluded[name] = f"p95 {p95:.0f} ms exceeds the {budget} ms budget"
                continue
        cands.append(name)
    base = {"use_case": uc, "label": cfg["label"], "metric": metric, "excluded": excluded,
            "latency_unverified": unverified}
    if not cands:
        return {**base, "status": "no_data", "winner": None, "contenders": [], "decided_by": None,
                "underpowered": True, "confidence": "none",
                "sentence": f"No approach can be judged for {cfg['label'].lower()} on this sample."}
    best = max(cands, key=lambda n: approaches[n]["metrics"][metric]["point"])
    contenders = [best]
    for n in cands:
        if n == best:
            continue
        mean, lo, hi = diffs[metric].get(f"{best}|{n}", (None, None, None))
        if lo is None or lo <= 0:  # not significantly worse than the best -> still in the running
            contenders.append(n)
    if len(contenders) == 1:
        winner, decided_by = best, "accuracy"
    else:
        def cost_key(n):
            c = approaches[n]["cost_per_1k"]
            return float("inf") if c is None else round(c, 6)

        def lat_key(n):
            v = approaches[n]["latency"]["p95"]
            return float("inf") if v is None else v

        winner = min(contenders, key=lambda n: (cost_key(n), lat_key(n), -approaches[n]["metrics"][metric]["point"]))
        rivals = [n for n in contenders if n != winner]
        if all(cost_key(n) > cost_key(winner) for n in rivals):
            decided_by = "cost"
        elif all(lat_key(n) > lat_key(winner) for n in rivals if cost_key(n) == cost_key(winner)):
            decided_by = "latency"
        else:
            decided_by = "accuracy_tie"  # same cost and speed: the higher point estimate breaks the tie
    lo, hi = approaches[winner]["metrics"][metric]["ci"]
    underpowered = lo is None or (hi - lo) / 2 > 0.10
    confidence = "low" if underpowered else ("high" if decided_by == "accuracy" else "medium")
    return {**base, "status": "ok", "winner": winner, "contenders": contenders, "decided_by": decided_by,
            "underpowered": underpowered, "confidence": confidence,
            "sentence": _sentence(winner, decided_by, contenders, approaches, diffs[metric], cfg, budget)}


def _sentence(winner, decided_by, contenders, approaches, d, cfg, budget):
    a = approaches[winner]
    lab = a["label"]
    words = cfg["metric_words"]
    parts = [f"Use {lab}."]
    others = [n for n in contenders if n != winner]
    if decided_by == "accuracy":
        runner = [n for n in approaches if n != winner and approaches[n]["metrics"][cfg["primary"]]["point"] is not None]
        if runner:
            r = max(runner, key=lambda n: approaches[n]["metrics"][cfg["primary"]]["point"])
            mean, lo, hi = d.get(f"{winner}|{r}", (None, None, None))
            if mean is not None:
                parts.append(f"It beats {approaches[r]['label']} on {words} by {mean:+.2f} (95% CI {lo:+.2f} to {hi:+.2f}).")
    else:
        names = ", ".join(approaches[n]["label"] for n in others)
        why = {"cost": "it is cheaper", "latency": "it is faster",
               "accuracy_tie": "it is among the cheapest and fastest, and has the highest estimate of those"}[decided_by]
        parts.append(f"It ties with {names} on {words} (differences within noise); chosen because {why}.")
    costs = ", ".join(f"{approaches[n]['label']} {_fmt_usd(approaches[n]['cost_per_1k'])}" for n in [winner] + others
                      if approaches[n]["cost_per_1k"] is not None)
    if costs:
        parts.append(f"Cost per 1,000 calls: {costs}.")
    if budget is not None and a["latency"].get("source") == "offline":
        parts.append("It runs offline, with no network latency.")
    elif budget is not None and a["latency"]["p95"] is not None:
        parts.append(f"Latency p95 {a['latency']['p95']:.0f} ms against a {budget} ms budget.")
    return " ".join(parts)


def report_markdown(sc: dict, sample: dict, use_cases: list, decisions: list, manifests: list,
                    volume: int = 10_000) -> str:
    """Exportable benchmark report: verdicts in plain language first, evidence after."""
    L = [f"# Benchmark report: {sample['corpus_id']}", "",
         f"Sample `{sample['sample_id']}`: {sc['n_calls']} calls ({sample['n_booked']} booked), stratified, seed {sample['seed']}. "
         f"95% CIs from a paired bootstrap over calls (B={sc['bootstrap']}).", "", "## Recommendations", ""]
    for uc in use_cases:
        v = sc["verdicts"][uc]
        L += [f"### {v['label']}", "", v["sentence"], "", f"Confidence: **{v['confidence']}**"
              + (" (underpowered: CI half-width > 0.10; use a larger sample)" if v["underpowered"] else ""), ""]
        for name, why in v["excluded"].items():
            L.append(f"- Not eligible: {sc['approaches'][name]['label']}: {why}")
        if v["latency_unverified"]:
            L.append("- Latency not measured (run the latency probe live): "
                     + ", ".join(sc["approaches"][n]["label"] for n in v["latency_unverified"]))
        L.append("")
    if decisions:
        L += ["## Decisions recorded", "", "| Use case | Approach | Note | At |", "|---|---|---|---|"]
        L += [f"| {d['use_case']} | {d['approach']} | {d['note']} | {d['at']} |" for d in decisions]
        L.append("")
    L += ["## Evidence", "", "| Approach | Mode | AUC mid | AUC pre-outcome | AUC attempted | Failure-mode F1 | "
          "$ / 1k calls | $ / month | Latency p50 / p95 (source) | Contract errors |", "|---|---|---|---|---|---|---|---|---|---|"]

    def f(m):
        p, (lo, hi) = m["point"], m["ci"]
        return "–" if p is None else f"{p:.3f}" + (f" [{lo:.3f}, {hi:.3f}]" if lo is not None else "")

    for name, a in sc["approaches"].items():
        mm, lat = a["metrics"], a["latency"]
        latency = "–" if lat["p50"] is None else f"{lat['p50']:.0f} / {lat['p95']:.0f} ms ({lat['source']})"
        cost = "–" if a["cost_per_1k"] is None else f"{a['cost_per_1k']:.3f}"
        month = "–" if a["cost_per_1k"] is None else f"{a['cost_per_1k'] * volume / 1000:,.2f}"
        L.append(f"| {a['label']} | {a['mode']} | {f(mm['auc_mid'])} | {f(mm['auc_pre_outcome'])} | {f(mm['auc_attempted'])} | "
                 f"{f(mm['failure_mode_f1'])} | {cost} | {month} | {latency} | {a['contract_error_rate']:.1%} |")
    L += ["", "## Runs", "", "| Run | Approach | Mode | Model | Calls | Cost (USD) | Status | Git |", "|---|---|---|---|---|---|---|---|"]
    L += [f"| `{m['run_id']}` | {m['approach']} | {m['mode']} | {m.get('model')} | {m.get('n_done')}/{m.get('n_calls')} | "
          f"{m.get('cost_usd', 0):.5f} | {m['status']} | {m.get('git_sha')} |" for m in manifests]
    return "\n".join(L) + "\n"


def unavailable(sample: dict, approaches: list, entries: dict, project: workspace.Project | None = None) -> dict:
    """Why a requested approach has no entry, in words an operator can act on."""
    out = {}
    for a in approaches:
        if a in entries:
            continue
        if a == "rules_b2":
            out[a] = ("needs labelled calls outside the sample to train on; this sample uses the whole corpus "
                      "(freeze a smaller sample, or pick a corpus with a training split)")
        elif a in ("jev", "glm"):
            out[a] = "not run on this sample yet (step ④)"
    return out


# ---------------------------------------------------------------- CLI
def load_entries(project: workspace.Project, sample: dict, approaches: list) -> dict:
    corpus = workspace.load_corpus(sample["corpus_id"])
    rules = rule_results(sample, corpus, workspace.load_train(sample["corpus_id"]))
    entries = {}
    for a in approaches:
        if a in rules:
            entries[a] = {"manifest": None, "results": rules[a]}
        elif not a.startswith("rules"):
            m = project.latest_run(sample["sample_id"], a)
            if m:
                entries[a] = {"manifest": m, "results": project.results(m["run_id"])}
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="default")
    ap.add_argument("--corpus", default="synthetic", choices=list(workspace.BUILTIN_CORPORA))
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--approaches", default="rules_b1,rules_b2,jev")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cap", type=float, default=1.0)
    a = ap.parse_args()
    project = workspace.Project(a.project).ensure()
    corpus = workspace.load_corpus(a.corpus)
    sample = project.save_sample(sampling.freeze(a.corpus, corpus, a.n, a.seed))
    names = a.approaches.split(",")
    for name in names:
        if name in ("jev", "glm"):
            cfg = jev_client.config_from_env() if name == "jev" else llm_client.config_from_env()
            t0 = time.time()
            rid = run_approach(project, sample, corpus, name, cfg, a.workers, a.cap)
            print(f"{name}: run {rid} ({time.time() - t0:.1f}s, mode {cfg.mode})")
    sc = scorecard(sample, load_entries(project, sample, names))
    for v in sc["verdicts"].values():
        print(f"[{v['label']}] {v['sentence']} (confidence: {v['confidence']})")


if __name__ == "__main__":
    main()
