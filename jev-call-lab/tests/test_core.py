import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import baselines
import evaluate
import generate_synthetic
import jev_client
import mock_jev
import reduce_sgd
import schema
from questions import REVIEW, WILL_BOOK, prefix_state
from simulate import forecast_call, forecast_points, run


# schema ----------------------------------------------------------------------
def test_seed_calls_validate(seed_calls):
    for c in seed_calls.values():
        assert schema.validate_call(c) == []


def test_validate_catches_bad_calls(seed_calls):
    c = json.loads(json.dumps(seed_calls["book-001"]))
    c["events"][3]["t_ms"] = 0
    c["events"][4]["event"] = "made_up_event"
    errors = " ".join(schema.validate_call(c))
    assert "t_ms" in errors and "made_up_event" in errors


def test_validate_catches_label_mismatch(seed_calls):
    c = json.loads(json.dumps(seed_calls["fail-002"]))
    c["label_booked"] = True
    assert any("label" in e for e in schema.validate_call(c))


# questions -------------------------------------------------------------------
def test_prefix_state_is_prefix_and_has_no_id_or_label(seed_calls):
    c = seed_calls["book-001"]
    s = prefix_state(c, c["events"][:5], 4)
    assert s["trace_so_far"] == c["events"][:5]
    assert "book-001" not in json.dumps(s)
    assert not any(k.startswith("label") for k in s)
    assert s["current_event"] == c["events"][4]["event"]
    assert s["tool_calls_so_far"] == 1


def test_review_questions_follow_primitive_rules():
    assert set(REVIEW) == {"will_book", "failure_mode", "agent_progress", "needs_human", "visible_error_overweight"}
    assert "other" in REVIEW["failure_mode"]["criteria"]  # leftover bucket
    assert isinstance(REVIEW["agent_progress"]["criteria"], list)


# mock + contract ---------------------------------------------------------------
def test_mock_answers_validate(seed_calls):
    c = seed_calls["recover-003"]
    body = mock_jev.answer({"model": "m", "state": prefix_state(c, c["events"], 12), "questions": REVIEW})
    assert jev_client.validate_response(REVIEW, body) == []
    assert set(body["answers"]) == set(REVIEW)
    assert isinstance(body["usage"]["input_tokens"], int)


def test_validate_response_catches_bad_shapes():
    bad = {"answers": {
        "will_book": {"type": "noul", "noul": 1.4},
        "failure_mode": {"type": "choice", "choice": "nope", "confidence": 1, "probabilities": {"nope": 1}},
        "agent_progress": {"type": "score", "score": 9, "confidence": 1, "legend": {}, "probabilities": {}},
    }, "usage": {}}
    errs = " ".join(jev_client.validate_response(REVIEW, bad))
    for part in ("will_book", "failure_mode", "agent_progress", "needs_human", "input_tokens"):
        assert part in errs


def test_mock_unknown_question_is_neutral():
    q = {"weather": {"type": "noul", "instructions": "Is it raining?"}}
    body = mock_jev.answer({"model": "m", "state": {"x": 1}, "questions": q})
    assert body["answers"]["weather"]["noul"] == 0.5


def test_mock_orders_success_above_failure(seed_calls):
    ok = seed_calls["book-001"]
    bad = seed_calls["fail-002"]
    p = lambda c: mock_jev.answer({"model": "m", "state": prefix_state(c, c["events"], len(c["events"]) - 1),
                                   "questions": WILL_BOOK})["answers"]["will_book"]["noul"]
    assert p(ok) > 0.8 > 0.2 > p(bad)


# client -------------------------------------------------------------------------
def test_auto_mode_without_key_is_mock(monkeypatch):
    monkeypatch.setenv("JEV_MODE", "auto")
    assert jev_client.mode() == "mock"
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    assert jev_client.mode() == "live"


def test_decide_mock_reports_cost_from_tokens():
    r = jev_client.decide({"current_event": "call_started"}, WILL_BOOK)
    assert r["_cost_source"] == "tokens_x_price"
    assert r["_cost_usd"] == pytest.approx(r["usage"]["input_tokens"] * jev_client.PRICE_PER_INPUT_TOKEN)


def test_cache_key_ignores_dict_order():
    a = jev_client.cache_key("m", {"a": 1, "b": 2}, {"q": {"type": "noul"}})
    b = jev_client.cache_key("m", {"b": 2, "a": 1}, {"q": {"type": "noul"}})
    assert a == b and a != jev_client.cache_key("m2", {"a": 1, "b": 2}, {"q": {"type": "noul"}})


def test_replay_miss_raises(monkeypatch):
    monkeypatch.setenv("JEV_MODE", "replay")
    with pytest.raises(jev_client.JevError, match="replay miss"):
        jev_client.decide({"never": "seen"}, WILL_BOOK)


def _server(handler_cls):
    srv = HTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_live_retries_429_then_uses_usage_cost(monkeypatch):
    calls = {"n": 0}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            calls["n"] += 1
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert self.headers["Authorization"] == "Bearer test-key"
            assert set(body) == {"model", "state", "questions"}
            if calls["n"] == 1:
                self.send_response(429)
                self.end_headers()
                return
            out = json.dumps({"model": "jev-1.13", "answers": {"will_book": {"type": "noul", "noul": 0.3}},
                              "usage": {"input_tokens": 100, "output_tokens": 5, "cost": 0.5}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(out)

    srv = _server(H)
    monkeypatch.setenv("JEV_MODE", "live")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("JEV_URL", f"http://127.0.0.1:{srv.server_address[1]}/api/alpha/decisions")
    monkeypatch.setattr(jev_client, "BACKOFF_S", 0.01)
    try:
        r = jev_client.decide({"s": 1}, WILL_BOOK)
    finally:
        srv.shutdown()
    assert calls["n"] == 2
    assert r["answers"]["will_book"]["noul"] == 0.3
    assert (r["_cost_usd"], r["_cost_source"]) == (0.5, "usage.cost")


def test_live_4xx_is_not_retried(monkeypatch):
    calls = {"n": 0}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            calls["n"] += 1
            self.send_response(422)
            self.end_headers()
            self.wfile.write(b'{"detail": []}')

    srv = _server(H)
    monkeypatch.setenv("JEV_MODE", "live")
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("JEV_URL", f"http://127.0.0.1:{srv.server_address[1]}/x")
    try:
        with pytest.raises(jev_client.JevError, match="422"):
            jev_client.decide({"s": 1}, WILL_BOOK)
    finally:
        srv.shutdown()
    assert calls["n"] == 1


def test_mock_http_server_roundtrip(monkeypatch, seed_calls):
    srv = mock_jev.make_server(0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("JEV_MODE", "live")
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("JEV_URL", f"http://127.0.0.1:{srv.server_address[1]}/api/alpha/decisions")
    c = seed_calls["book-001"]
    try:
        r = jev_client.decide(prefix_state(c, c["events"], 10), REVIEW)
    finally:
        srv.shutdown()
    assert jev_client.validate_response(REVIEW, r) == []


# SGD reducer ----------------------------------------------------------------------
def test_reduce_sgd_recovered_booking(sgd_recovered):
    call = reduce_sgd.reduce_dialogue(sgd_recovered, "train")
    assert schema.validate_call(call) == []
    assert call["label_booked"] is True
    assert call["specialty"] == "dentist"
    names = [e["event"] for e in call["events"]]
    assert names[0] == "call_started" and names[-1] == "call_ended"
    # tool call precedes the agent's reply in the same system turn
    assert names.index("search_providers") < names.index("offered_provider")
    books = [e for e in call["events"] if e["event"] == "create_appointment"]
    assert [b["ok"] for b in books] == [False, True]
    assert "booking_failed" in names and "booking_confirmed" in names
    # NEGATE answering "anything else?" is not a refusal
    assert "nothing_else" in names and "declined" not in names


def test_reduce_sgd_copies_no_text_or_values(sgd_recovered):
    call = reduce_sgd.reduce_dialogue(sgd_recovered, "train")
    blob = json.dumps(call).lower()
    for leaked in ("gilroy", "albert", "march", "11:30", "sorry"):
        assert leaked not in blob


def test_reduce_sgd_unbooked_info_call(sgd_recovered):
    d = sgd_recovered
    d["turns"] = d["turns"][:2] + d["turns"][-2:]
    call = reduce_sgd.reduce_dialogue(d, "dev")
    assert call["label_booked"] is False
    assert schema.validate_call(call) == []


# synthetic generator ----------------------------------------------------------------
def test_generator_is_deterministic_and_consistent():
    a = generate_synthetic.generate(40, seed=3)
    b = generate_synthetic.generate(40, seed=3)
    assert a == b
    assert {"book-001", "fail-002", "recover-003"} <= {c["call_id"] for c in a}
    for c in a:
        assert schema.validate_call(c) == [], c["call_id"]
    labels = {c["label_booked"] for c in a}
    assert labels == {True, False}


# simulate + evaluate -------------------------------------------------------------------
def test_forecast_points_are_agent_and_tool_events(seed_calls):
    ev = seed_calls["book-001"]["events"]
    assert all(ev[i]["actor"] in {"agent", "tool"} for i in forecast_points(ev))
    assert len(forecast_points(ev)) == 7


def test_forecast_call_and_pre_outcome(seed_calls):
    r = forecast_call(seed_calls["book-001"])
    assert len(r["turns"]) == 7 and r["review"]
    assert r["first_booking_index"] == 8
    pre = evaluate.pre_outcome(r)
    assert pre["event_index"] < 8


def test_auc_and_brier():
    rows = [{"p": 0.9, "y": True}, {"p": 0.8, "y": True}, {"p": 0.8, "y": False}, {"p": 0.1, "y": False}]
    assert evaluate.auc(rows) == pytest.approx((2 + 1 + 0.5) / 4)
    assert evaluate.auc([{"p": 1, "y": True}]) is None
    assert evaluate.brier([{"p": 1.0, "y": True}, {"p": 1.0, "y": False}]) == 0.5


def test_run_and_metrics(seed_calls):
    results = run(list(seed_calls.values()), workers=2, quiet=True)
    m = evaluate.metrics(results)
    assert m["n_calls"] == 3 and m["n_requests"] == sum(len(r["turns"]) + 1 for r in results)
    assert m["auc"]["end"] == 1.0
    assert "trajectory" in evaluate.render(results)


def test_logistic_baseline_learns():
    train = generate_synthetic.generate(120, seed=1)
    test = generate_synthetic.generate(60, seed=2)
    out = baselines.evaluate_all(train, test)
    assert set(out) == {"B0_constant", "B1_stage_reached", "B2_logreg_counts"}
    assert out["B0_constant"]["auc"]["end"] == 0.5
    assert out["B2_logreg_counts"]["auc"]["end"] > 0.9
