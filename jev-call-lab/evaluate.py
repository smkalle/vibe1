"""Read the flight recorder: trajectories, ranking, calibration, cost and latency.

    python evaluate.py results/results.json [--json] [--max-calls 10]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from jev_client import PRICE_PER_INPUT_TOKEN
from questions import REVIEW, WILL_BOOK, prefix_state
from simulate import forecast_points


def auc(rows, min_class=1):
    """Mann-Whitney AUC of p vs binary y; ties count half. None if a class has < min_class rows."""
    pos = [r["p"] for r in rows if r["y"]]
    neg = [r["p"] for r in rows if not r["y"]]
    if len(pos) < max(1, min_class) or len(neg) < max(1, min_class):
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def brier(rows):
    return sum((r["p"] - r["y"]) ** 2 for r in rows) / len(rows) if rows else None


def midpoint(r):
    return r["turns"][len(r["turns"]) // 2] if r["turns"] else None


def pre_outcome(r):
    """Last forecast strictly before the call enters booking, wrap-up or completion."""
    before = [t for t in r["turns"] if t["event_index"] < r["decision_index"]]
    return before[-1] if before else None


def end(r):
    return r["turns"][-1] if r["turns"] else None


POINTS = {"mid": midpoint, "pre_outcome": pre_outcome, "end": end}


def _rows(results, point):
    return [{"p": (t["p_book"] if (t := POINTS[point](r)) else 0.5), "y": r["label_booked"]} for r in results]


def auc_table(results):
    """AUC at each point, plus pre_outcome on calls that attempted a booking (will the attempt land?)."""
    table = {k: _round(auc(_rows(results, k))) for k in POINTS}
    attempted = [r for r in results if r["attempted_booking"]]
    table["pre_outcome_attempted"] = _round(auc(_rows(attempted, "pre_outcome"), min_class=5))
    return table


def _pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0.0


def calibration(rows, bins=5):
    out = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [r for r in rows if lo <= r["p"] < hi or (b == bins - 1 and r["p"] == 1.0)]
        if sel:
            out.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": len(sel),
                        "mean_p": round(sum(r["p"] for r in sel) / len(sel), 3),
                        "booked_rate": round(sum(r["y"] for r in sel) / len(sel), 3)})
    return out


def metrics(results):
    # Only network latency counts: live calls, or live times saved in fixtures. Results written
    # before latency_source existed came from live runs, hence the "measured" default.
    real = {"measured", "recorded"}
    turns = [t for r in results for t in r["turns"]]
    latencies = [t["latency_ms"] for t in turns if t.get("latency_source", "measured") in real]
    latency_sources = sorted({t.get("latency_source", "measured") for t in turns})
    cost = sum(t["cost_usd"] for r in results for t in r["turns"]) + sum(r.get("review_cost_usd", 0) for r in results)
    tokens = sum(t["input_tokens"] for r in results for t in r["turns"]) + sum(r.get("review_input_tokens", 0) for r in results)
    n_req = sum(len(r["turns"]) + (1 if r.get("review") else 0) for r in results)
    reviewed = [r for r in results if r.get("review")]
    agree = sum(r["review"]["failure_mode"]["choice"] == r["reference_failure_mode"] for r in reviewed)
    return {
        "n_calls": len(results),
        "n_booked": sum(r["label_booked"] for r in results),
        "n_requests": n_req,
        "n_attempted_booking": sum(r["attempted_booking"] for r in results),
        "n_attempted_not_booked": sum(r["attempted_booking"] and not r["label_booked"] for r in results),
        "auc": auc_table(results),
        "brier": {k: _round(brier(_rows(results, k))) for k in POINTS},
        "calibration_pre_outcome": calibration(_rows(results, "pre_outcome")),
        "failure_mode": {
            "agreement_with_rule_reference": _round(agree / len(reviewed)) if reviewed else None,
            "jev_choices": dict(Counter(r["review"]["failure_mode"]["choice"] for r in reviewed)),
            "reference": dict(Counter(r["reference_failure_mode"] for r in results)),
        },
        "cost": {
            "total_usd": round(cost, 8),
            "per_call_usd": round(cost / len(results), 8) if results else 0,
            "input_tokens": tokens,
            "source": sorted({r.get("cost_source", "tokens_x_price") for r in results}),
        },
        "latency_ms": {
            "p50": round(_pct(latencies, 0.5), 2) if latencies else None,
            "p95": round(_pct(latencies, 0.95), 2) if latencies else None,
            "sources": latency_sources,
        },
        "models": sorted({str(r.get("model")) for r in results}),
    }


def _round(x):
    return None if x is None else round(x, 4)


def project_cost(results, calls):
    """Cost of running every call's forecasts + review, from payload size (~4 chars/token).

    When results carry real token counts, the chars/4 estimate is rescaled to match them.
    """
    import json as _j

    def est(state, qs):
        return len(_j.dumps({"model": "typesafe/jev-1.13", "state": state, "questions": qs}, separators=(",", ":"))) // 4

    n_req = tokens = 0
    for c in calls:
        ev = c["events"]
        for i in forecast_points(ev):
            tokens += est(prefix_state(c, ev[: i + 1], i), WILL_BOOK)
            n_req += 1
        tokens += est(prefix_state(c, ev, len(ev) - 1), REVIEW)
        n_req += 1
    actual = sum(t["input_tokens"] for r in results for t in r["turns"])
    by_id = {c["call_id"]: c for c in calls}
    estimated = sum(
        est(prefix_state(by_id[r["call_id"]], by_id[r["call_id"]]["events"][: t["event_index"] + 1], t["event_index"]), WILL_BOOK)
        for r in results if r["call_id"] in by_id for t in r["turns"]
    )
    scale = actual / estimated if actual and estimated else 1.0
    tokens = int(tokens * scale)
    return {"calls": len(calls), "requests": n_req, "input_tokens": tokens,
            "usd": round(tokens * PRICE_PER_INPUT_TOKEN, 6), "token_scale_vs_chars_div_4": round(scale, 3)}


def render(results, max_calls=None):
    lines = [f"{'call':<26} {'y':<6} {'mid_p':<7} {'pre_p':<7} {'end_p':<7} mode (reference)"]
    for r in results[:max_calls]:
        p = {k: (t["p_book"] if (t := f(r)) else 0.5) for k, f in POINTS.items()}
        mode = r["review"]["failure_mode"]["choice"] if r.get("review") else "-"
        lines.append(f"{r['call_id']:<26} {str(r['label_booked']):<6} {p['mid']:<7.3f} {p['pre_outcome']:<7.3f} "
                     f"{p['end']:<7.3f} {mode} ({r['reference_failure_mode']})")
        lines.append("  trajectory:")
        for t in r["turns"]:
            bar = "█" * int(t["p_book"] * 20)
            flag = "" if t["ok"] else "  ✗"
            lines.append(f"    t={t['t_ms'] / 1000:5.1f}s  {t['stage']:<12} {t['p_book']:.2f} {bar:<20} {t['event']}{flag}")
        if r.get("review"):
            rv = r["review"]
            lines.append(f"  review: progress={rv['agent_progress']['score']:.2f} needs_human={rv['needs_human']['noul']:.2f} "
                         f"recoverable_error={rv['visible_error_overweight']['noul']:.2f}")
        lines.append("")
    m = metrics(results)
    lines += [
        f"calls={m['n_calls']} booked={m['n_booked']} requests={m['n_requests']} models={m['models']}",
        f"AUC   mid={m['auc']['mid']} pre_outcome={m['auc']['pre_outcome']} end={m['auc']['end']} "
        f"pre_outcome_attempted={m['auc']['pre_outcome_attempted']} "
        f"(n={m['n_attempted_booking']}, not booked={m['n_attempted_not_booked']})",
        f"Brier mid={m['brier']['mid']} pre_outcome={m['brier']['pre_outcome']} end={m['brier']['end']}",
        f"failure_mode agreement with rule reference: {m['failure_mode']['agreement_with_rule_reference']}",
        (f"latency p50={m['latency_ms']['p50']}ms p95={m['latency_ms']['p95']}ms ({', '.join(m['latency_ms']['sources'])})"
         if m["latency_ms"]["p50"] is not None else
         f"latency n/a: no network timings ({', '.join(m['latency_ms']['sources'])})"),
        f"total USD={m['cost']['total_usd']:.6f} ({', '.join(m['cost']['source'])}), input tokens={m['cost']['input_tokens']}",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default="results/results.json")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--max-calls", type=int, default=10)
    a = ap.parse_args()
    results = json.loads(Path(a.results).read_text())
    print(json.dumps(metrics(results), indent=2) if a.json else render(results, a.max_calls))


if __name__ == "__main__":
    main()
