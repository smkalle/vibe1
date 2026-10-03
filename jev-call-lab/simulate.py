"""Turn-level booking forecasts plus one five-question review per call.

    python simulate.py --data data/synthetic_calls.json --limit 3
    python simulate.py --data data/sgd_test.json --workers 16 --out results/sgd_test.json
    python simulate.py --scorer glm --data data/sgd_test.json --limit 5 --out results/sgd_glm.json
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from functools import partial
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


def forecast_call(call: dict, review: bool = True, cfg=None, decide_fn=None, scorer: str = "jev") -> dict:
    """Forecast one call. decide_fn(state, questions) defaults to Jev (or cfg); pass a partial for GLM."""
    events = call["events"]
    turns = []
    if decide_fn is None:
        decide_fn = partial(decide, cfg=cfg) if cfg is not None else decide
    for idx in forecast_points(events):
        resp = decide_fn(prefix_state(call, events[: idx + 1], idx), WILL_BOOK)
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
        resp = decide_fn(prefix_state(call, events, len(events) - 1), REVIEW)
        out.update(
            review=resp["answers"],
            review_errors=validate_response(REVIEW, resp),
            review_cost_usd=resp["_cost_usd"],
            review_latency_ms=resp["_latency_ms"],
            review_input_tokens=resp.get("usage", {}).get("input_tokens", 0),
            cost_source=resp["_cost_source"],
            model=resp.get("model"),
        )
    out["scorer"] = scorer
    return out


def run(calls: list, workers: int = 8, review: bool = True, quiet: bool = False, cfg=None,
        decide_fn=None, scorer: str = "jev", strict: bool = True) -> list:
    """Run all calls. strict=False records per-call failures as {"call_id", "error"} instead of raising."""
    if strict:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(lambda c: forecast_call(c, review, cfg, decide_fn, scorer), calls))
    else:
        from concurrent.futures import as_completed
        results = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(forecast_call, c, review, cfg, decide_fn, scorer): c for c in calls}
            for fut in as_completed(futs):
                try:
                    results.append(fut.result())
                except Exception as e:  # noqa: BLE001 - recorded, run continues
                    results.append({"call_id": futs[fut]["call_id"], "error": f"{type(e).__name__}: {e}"})
        order = {c["call_id"]: i for i, c in enumerate(calls)}
        results.sort(key=lambda r: order[r["call_id"]])
    if not quiet:
        for r in results:
            print("done", r["call_id"], r.get("error", ""))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic_calls.json")
    ap.add_argument("--out", default="results/results.json")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-review", action="store_true")
    ap.add_argument("--scorer", choices=["jev", "glm"], default="jev")
    ap.add_argument("--llm-model", default=None, help="GLM model slug (default: LLM_MODEL or z-ai/glm-5.3)")
    ap.add_argument("--keep-going", action="store_true",
                    help="record per-call failures as {call_id, error} instead of aborting the run")
    a = ap.parse_args()
    calls = json.loads(Path(a.data).read_text())[: a.limit]
    decide_fn, scorer = None, a.scorer
    if a.scorer == "glm":
        import llm_client
        llm_cfg = llm_client.config_from_env()
        if a.llm_model:
            llm_cfg = llm_client.Config(mode=llm_cfg.mode, api_key=llm_cfg.api_key, url=llm_cfg.url,
                                        model=a.llm_model, effort=llm_cfg.effort, fixtures=llm_cfg.fixtures)
        decide_fn = partial(llm_client.decide, cfg=llm_cfg)
    results = run(calls, a.workers, review=not a.no_review, quiet=len(calls) > 20,
                  decide_fn=decide_fn, scorer=scorer, strict=not a.keep_going)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=1))
    print(f"{len(results)} calls, {sum(len(r['turns']) for r in results)} turn forecasts -> {a.out}")


if __name__ == "__main__":
    main()
