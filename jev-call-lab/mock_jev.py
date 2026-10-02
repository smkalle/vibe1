"""A deterministic stand-in for Jev. It is NOT Jev and is never fitted to labels.

It returns the exact TypeSafe/OpenRouter wire shape so the pipeline, contract
checks and evaluator can run without a key. Its probabilities come from simple
hand-set weights over the structural trace, with a deliberate penalty on every
visible failure (the bias Sully reported), so recover-style calls dip.

    python mock_jev.py --port 8765    # serves POST /api/alpha/decisions and /api/v1/systemone
"""
import argparse
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MOCK_MODEL = "mock-jev-heuristic"

BOOK_INTENT = {"stated_intent_book", "accepted_booking_offer", "confirmed_new_appointment", "confirmed_follow_up"}
SELECTED = {"selected_offer", "accepted_slot"}
READBACK = {"readback_details", "readback_datetime"}
OFFERS = {"offered_provider", "offered_alternative_time", "offered_two_slots", "offered_one_slot", "offered_to_book"}
WRAPUP = {"asked_anything_else", "thanked", "said_goodbye", "nothing_else", "call_ended"}
DROP = {"hung_up", "caller_went_silent", "refused_to_verify"}
TIME_SLOTS = {"appointment_date", "appointment_time"}
VISIBLE_ERROR_WEIGHT = 0.7


def _sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z))


def _trace(state) -> list | None:
    if isinstance(state, dict) and isinstance(state.get("trace_so_far"), list):
        return state["trace_so_far"]
    return None


def _booked(trace) -> bool:
    return any(e.get("event") in {"create_appointment", "booking_confirmed"} and e.get("ok", True) for e in trace)


def p_book(trace: list) -> float:
    names = [e.get("event") for e in trace]
    s = set(names)
    booked = _booked(trace)
    z = -0.3
    z += 1.0 * bool(s & BOOK_INTENT)
    z += 0.8 * bool(s & SELECTED)
    z += 0.6 * any(set(e.get("slots", [])) & TIME_SLOTS for e in trace if e.get("event") in {"asked_for_info", "provided_info"})
    z += 0.8 * bool(s & READBACK)
    z += 0.8 * ("affirmed" in s and bool(s & READBACK))
    z += 0.3 * bool(s & OFFERS)
    z += 0.4 * any(e.get("event") in {"searched_open_slots", "ehr_patient_lookup", "search_providers"} and e.get("ok") for e in trace)
    z += 5.0 * booked
    z -= VISIBLE_ERROR_WEIGHT * sum(1 for e in trace if not e.get("ok", True))
    z -= 0.8 * names.count("caller_went_silent") + 1.5 * ("refused_to_verify" in s) + 5.0 * ("hung_up" in s)
    z -= 2.0 * ("declined_booking_offer" in s) + 1.0 * ("offered_human_transfer" in s)
    z -= 0.5 * (names.count("repeated_identity_ask") + names.count("agent_repeated_question"))
    if not s & BOOK_INTENT and "stated_intent_find" in s:
        z -= 0.2 * names.count("gave_info")
    if not booked and s & WRAPUP:
        z -= 2.5
    return min(0.99, max(0.01, _sigmoid(z)))


def failure_mode_rule(trace: list) -> str:
    """Rule-derived failure mode (also the reference the evaluator compares Jev's choice against)."""
    names = [e.get("event") for e in trace]
    s = set(names)
    fails = sum(1 for e in trace if not e.get("ok", True))
    if _booked(trace):
        return "recovered_error" if fails else "clean_book"
    if s & DROP:
        return "caller_drop"
    if "identity_check_failed" in s and not any(e.get("event") == "ehr_patient_lookup" and e.get("ok") for e in trace):
        return "identity_block"
    if "booking_failed" in s or any(e.get("actor") == "tool" and not e.get("ok") for e in trace):
        return "no_availability"
    if "declined_booking_offer" in s or "declined_slots" in s or "declined" in s:
        return "declined_offer"
    if not s & BOOK_INTENT:
        return "info_only"
    if len(names) - len(s) > len(names) // 2:
        return "agent_loop"
    return "other"


def progress_level(trace: list) -> int:
    s = {e.get("event") for e in trace}
    if _booked(trace):
        return 3
    if s & (OFFERS | SELECTED):
        return 2
    if s & (BOOK_INTENT | {"stated_intent_find", "search_providers", "searched_open_slots"}):
        return 1
    return 0


def _peaked(labels: list, chosen, mass=0.7) -> dict:
    rest = (1 - mass) / (len(labels) - 1) if len(labels) > 1 else 0
    return {str(l): (mass if l == chosen else rest) if len(labels) > 1 else 1.0 for l in labels}


def _answer_one(name: str, q: dict, trace: list | None) -> dict:
    kind = q["type"]
    if kind == "noul":
        p = 0.5
        if trace is not None:
            fails = sum(1 for e in trace if not e.get("ok", True))
            s = {e.get("event") for e in trace}
            if name == "will_book":
                p = p_book(trace)
            elif name == "needs_human":
                p = _sigmoid(-2 + 1.2 * fails + 2 * bool(s & DROP) - 3 * _booked(trace))
            elif name == "visible_error_overweight":
                p = (0.85 if _booked(trace) else 0.4) if fails else 0.15
        return {"type": "noul", "noul": round(p, 4)}
    if kind == "choice":
        labels = list(q["criteria"])
        chosen = failure_mode_rule(trace) if name == "failure_mode" and trace is not None else None
        if chosen not in labels:
            probs = {l: 1 / len(labels) for l in labels}
            chosen = labels[0]
        else:
            probs = _peaked(labels, chosen)
        return {"type": "choice", "choice": chosen, "confidence": round(max(probs.values()), 4), "probabilities": probs}
    # score
    n = len(q["criteria"])
    if name == "agent_progress" and trace is not None and n == 4:
        probs = _peaked(list(range(n)), progress_level(trace))
    else:
        probs = {str(i): 1 / n for i in range(n)}
    score = sum(int(k) * v for k, v in probs.items())
    return {
        "type": "score",
        "score": round(score, 4),
        "confidence": round(max(probs.values()), 4),
        "legend": {str(i): c for i, c in enumerate(q["criteria"])},
        "probabilities": probs,
    }


def answer(payload: dict) -> dict:
    trace = _trace(payload.get("state"))
    questions = payload["questions"]
    return {
        "model": MOCK_MODEL,
        "answers": {name: _answer_one(name, q, trace) for name, q in questions.items()},
        # ~4 characters per token: a stand-in for real billing counts.
        "usage": {"input_tokens": len(json.dumps(payload, separators=(",", ":"))) // 4, "output_tokens": 4 * len(questions)},
    }


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code: int, body: dict):
        out = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_POST(self):
        if not self.path.rstrip("/").endswith(("/decisions", "/systemone")):
            return self._send(404, {"error": "not found"})
        if not self.headers.get("Authorization", "").startswith("Bearer "):
            return self._send(401, {"error": "missing bearer token"})
        try:
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            missing = [k for k in ("model", "state", "questions") if k not in payload]
            if missing or not payload["questions"]:
                return self._send(422, {"detail": [{"loc": ["body", k], "msg": "Field required"} for k in missing]})
            return self._send(200, answer(payload))
        except (ValueError, KeyError, TypeError) as e:
            return self._send(400, {"error": str(e)})


def make_server(port: int = 8765) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), _Handler)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    port = ap.parse_args().port
    print(f"mock Jev on http://127.0.0.1:{port}/api/alpha/decisions (NOT the real model)")
    make_server(port).serve_forever()
