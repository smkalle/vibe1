"""Turn-level booking forecasts plus one five-question review per call.

    python simulate.py --data data/synthetic_calls.json --limit 3
    python simulate.py --data data/sgd_test.json --workers 16 --out results/sgd_test.json
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from jev_client import decide, validate_response
from mock_jev import failure_mode_rule
from questions import REVIEW, WILL_BOOK, prefix_state


def forecast_points(events: list) -> list[int]:
    """Forecast after every agent or tool event ('after every agent turn')."""
    return [i for i, e in enumerate(events) if e["actor"] in {"agent", "tool"}]


DECIDED_STAGES = {"booking", "wrapup", "complete"}


def decision_index(events: list) -> int:
    """First event where the outcome starts to show: a booking attempt, wrap-up, or the end.

    The 'pre_outcome' forecast is the last one strictly before this index, so booked and
    unbooked calls are both cut before their ending is visible.
    """
    return next(i for i, e in enumerate(events) if e["stage"] in DECIDED_STAGES)


BOOKING_INTENT = {"stated_intent_book", "accepted_booking_offer", "confirmed_new_appointment", "confirmed_follow_up"}


def attempted_booking(events: list) -> bool:
    """The hard subset: the caller or agent committed to booking before the outcome showed."""
    return any(e["event"] in BOOKING_INTENT for e in events[: decision_index(events)])


def forecast_call(call: dict, review: bool = True) -> dict:
    events = call["events"]
    turns = []
    for idx in forecast_points(events):
        resp = decide(prefix_state(call, events[: idx + 1], idx), WILL_BOOK)
        turns.append({
            "event_index": idx,
            "t_ms": events[idx]["t_ms"],
            "stage": events[idx]["stage"],
            "event": events[idx]["event"],
            "ok": events[idx]["ok"],
            "p_book": resp["answers"]["will_book"]["noul"],
            "latency_ms": resp["_latency_ms"],
            "cost_usd": resp["_cost_usd"],
            "input_tokens": resp.get("usage", {}).get("input_tokens", 0),
            "errors": validate_response(WILL_BOOK, resp),
        })

    out = {
        "call_id": call["call_id"],
        "source": call.get("source"),
        "label_booked": call["label_booked"],
        "n_events": len(events),
        "first_booking_index": next((i for i, e in enumerate(events) if e["event"] == "create_appointment"), None),
        "decision_index": decision_index(events),
        "attempted_booking": attempted_booking(events),
        "reference_failure_mode": failure_mode_rule(events),
        "turns": turns,
        "review": None,
    }
    if review:
        # All five questions share one state, so they share one request.
        resp = decide(prefix_state(call, events, len(events) - 1), REVIEW)
        out.update(
            review=resp["answers"],
            review_errors=validate_response(REVIEW, resp),
            review_cost_usd=resp["_cost_usd"],
            review_latency_ms=resp["_latency_ms"],
            review_input_tokens=resp.get("usage", {}).get("input_tokens", 0),
            cost_source=resp["_cost_source"],
            model=resp.get("model"),
        )
    return out


def run(calls: list, workers: int = 8, review: bool = True, quiet: bool = False) -> list:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda c: forecast_call(c, review), calls))
    if not quiet:
        for r in results:
            print("done", r["call_id"])
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic_calls.json")
    ap.add_argument("--out", default="results/results.json")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-review", action="store_true")
    a = ap.parse_args()
    calls = json.loads(Path(a.data).read_text())[: a.limit]
    results = run(calls, a.workers, review=not a.no_review, quiet=len(calls) > 20)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=1))
    print(f"{len(results)} calls, {sum(len(r['turns']) for r in results)} turn forecasts -> {a.out}")


if __name__ == "__main__":
    main()
