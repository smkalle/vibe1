"""Synthetic phone-call traces: the tutorial's three seed calls plus a seeded stage machine.

Covers the phone failure modes SGD lacks: silence, refusal, hangup, identity
blocks, slot droughts, agent loops, and recoveries after a visible error.

    python generate_synthetic.py --n 150 --seed 7
"""
import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).parent

SEEDS = [
    {"call_id": "book-001", "site": "north-clinic", "specialty": "dermatology", "label_booked": True, "events": [
        {"t_ms": 0, "actor": "system", "stage": "greeting", "event": "call_started", "ok": True},
        {"t_ms": 4200, "actor": "agent", "stage": "identity", "event": "asked_dob_and_name", "ok": True},
        {"t_ms": 9100, "actor": "caller", "stage": "identity", "event": "provided_identifiers", "ok": True},
        {"t_ms": 11200, "actor": "tool", "stage": "identity", "event": "ehr_patient_lookup", "ok": True, "tool": "find_patient"},
        {"t_ms": 15800, "actor": "agent", "stage": "intent", "event": "confirmed_new_appointment", "ok": True},
        {"t_ms": 20100, "actor": "tool", "stage": "availability", "event": "searched_open_slots", "ok": True, "tool": "search_slots", "n_results": 4},
        {"t_ms": 26400, "actor": "agent", "stage": "offer", "event": "offered_two_slots", "ok": True},
        {"t_ms": 33100, "actor": "caller", "stage": "offer", "event": "accepted_slot", "ok": True},
        {"t_ms": 36200, "actor": "tool", "stage": "booking", "event": "create_appointment", "ok": True, "tool": "book"},
        {"t_ms": 40100, "actor": "agent", "stage": "confirm", "event": "readback_datetime", "ok": True},
        {"t_ms": 44800, "actor": "system", "stage": "complete", "event": "call_ended", "ok": True}]},
    {"call_id": "fail-002", "site": "west-clinic", "specialty": "cardiology", "label_booked": False, "events": [
        {"t_ms": 0, "actor": "system", "stage": "greeting", "event": "call_started", "ok": True},
        {"t_ms": 3800, "actor": "agent", "stage": "identity", "event": "asked_dob_and_name", "ok": True},
        {"t_ms": 14100, "actor": "caller", "stage": "identity", "event": "caller_went_silent", "ok": False},
        {"t_ms": 18200, "actor": "agent", "stage": "identity", "event": "repeated_identity_ask", "ok": True},
        {"t_ms": 24100, "actor": "tool", "stage": "identity", "event": "ehr_patient_lookup", "ok": False, "tool": "find_patient"},
        {"t_ms": 26800, "actor": "agent", "stage": "identity", "event": "identity_check_failed", "ok": False},
        {"t_ms": 32100, "actor": "caller", "stage": "intent", "event": "refused_to_verify", "ok": False},
        {"t_ms": 35500, "actor": "agent", "stage": "handoff", "event": "offered_human_transfer", "ok": True},
        {"t_ms": 40200, "actor": "caller", "stage": "complete", "event": "hung_up", "ok": False},
        {"t_ms": 40350, "actor": "system", "stage": "complete", "event": "call_ended", "ok": True}]},
    {"call_id": "recover-003", "site": "east-clinic", "specialty": "orthopedics", "label_booked": True, "events": [
        {"t_ms": 0, "actor": "system", "stage": "greeting", "event": "call_started", "ok": True},
        {"t_ms": 5100, "actor": "tool", "stage": "identity", "event": "ehr_patient_lookup", "ok": False, "tool": "find_patient"},
        {"t_ms": 7800, "actor": "agent", "stage": "identity", "event": "identity_check_failed", "ok": False},
        {"t_ms": 12400, "actor": "agent", "stage": "identity", "event": "retried_with_phone_match", "ok": True},
        {"t_ms": 16100, "actor": "tool", "stage": "identity", "event": "ehr_patient_lookup", "ok": True, "tool": "find_patient"},
        {"t_ms": 19800, "actor": "agent", "stage": "intent", "event": "confirmed_follow_up", "ok": True},
        {"t_ms": 24200, "actor": "tool", "stage": "availability", "event": "searched_open_slots", "ok": True, "tool": "search_slots", "n_results": 1},
        {"t_ms": 28600, "actor": "agent", "stage": "offer", "event": "offered_one_slot", "ok": True},
        {"t_ms": 33400, "actor": "caller", "stage": "offer", "event": "asked_for_later_week", "ok": True},
        {"t_ms": 37100, "actor": "tool", "stage": "availability", "event": "searched_open_slots", "ok": True, "tool": "search_slots", "n_results": 3},
        {"t_ms": 41900, "actor": "caller", "stage": "offer", "event": "accepted_slot", "ok": True},
        {"t_ms": 44800, "actor": "tool", "stage": "booking", "event": "create_appointment", "ok": True, "tool": "book"},
        {"t_ms": 49200, "actor": "system", "stage": "complete", "event": "call_ended", "ok": True}]},
]

SITES = ["north-clinic", "west-clinic", "east-clinic", "south-clinic"]
SPECIALTIES = ["dermatology", "cardiology", "orthopedics", "pediatrics", "family_medicine"]


class _Call:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.t = 0
        self.events = []
        self.over = False

    def add(self, actor, stage, event, ok=True, gap=(2000, 6000), **extra):
        if self.events:
            self.t += self.rng.randint(*gap)
        self.events.append({"t_ms": self.t, "actor": actor, "stage": stage, "event": event, "ok": ok, **extra})

    def hang_up(self, stage):
        self.add("caller", "complete", "hung_up", ok=False, gap=(500, 3000))
        self.over = True

    def maybe_drop(self, stage, p=0.03):
        if self.rng.random() < p:
            self.hang_up(stage)
        return self.over


def _one(rng: random.Random, call_id: str) -> dict:
    c = _Call(rng)
    r = rng.random
    c.add("system", "greeting", "call_started")
    c.add("agent", "identity", "asked_dob_and_name")

    # identity
    if r() < 0.12:
        c.add("caller", "identity", "caller_went_silent", ok=False, gap=(8000, 12000))
        c.add("agent", "identity", "repeated_identity_ask")
        if r() < 0.5:
            c.add("caller", "intent", "refused_to_verify", ok=False)
            c.add("agent", "handoff", "offered_human_transfer")
            c.hang_up("handoff")
    if not c.over and not c.maybe_drop("identity"):
        c.add("caller", "identity", "provided_identifiers")
        ok = r() < 0.8
        c.add("tool", "identity", "ehr_patient_lookup", ok=ok, tool="find_patient")
        if not ok:
            c.add("agent", "identity", "identity_check_failed", ok=False)
            if r() < 0.6:
                c.add("agent", "identity", "retried_with_phone_match")
                ok = r() < 0.75
                c.add("tool", "identity", "ehr_patient_lookup", ok=ok, tool="find_patient")
            if not ok:
                c.add("agent", "handoff", "offered_human_transfer")
                if r() < 0.7:
                    c.hang_up("handoff")
                else:
                    c.over = True

    # intent + availability + offer
    if not c.over:
        c.add("agent", "intent", rng.choice(["confirmed_new_appointment", "confirmed_follow_up"]))
        if r() < 0.05:  # agent loop
            for _ in range(3):
                c.add("agent", "intent", "agent_repeated_question")
            c.hang_up("intent")
    if not c.over:
        n = 0 if r() < 0.15 else rng.randint(1, 5)
        c.add("tool", "availability", "searched_open_slots", ok=n > 0, tool="search_slots", n_results=n)
        if n == 0 and r() < 0.5:
            n = rng.randint(1, 3)
            c.add("tool", "availability", "searched_open_slots", ok=True, tool="search_slots", n_results=n)
        if n == 0:
            c.add("agent", "handoff", "offered_human_transfer")
            c.over = True
    if not c.over:
        c.add("agent", "offer", "offered_two_slots" if n >= 2 else "offered_one_slot")
        if not c.maybe_drop("offer"):
            x = r()
            if x < 0.2:
                c.add("caller", "offer", "asked_for_later_week")
                n = rng.randint(0, 3)
                c.add("tool", "availability", "searched_open_slots", ok=n > 0, tool="search_slots", n_results=n)
                if n and r() < 0.7:
                    c.add("caller", "offer", "accepted_slot")
                else:
                    c.add("caller", "offer", "declined_slots", ok=False)
                    c.over = True
            elif x < 0.85:
                c.add("caller", "offer", "accepted_slot")
            else:
                c.add("caller", "offer", "declined_slots", ok=False)
                c.over = True

    # booking
    if not c.over:
        ok = r() < 0.93
        c.add("tool", "booking", "create_appointment", ok=ok, tool="book")
        if not ok:
            c.add("agent", "offer", "offered_one_slot")
            if r() < 0.7:
                c.add("caller", "offer", "accepted_slot")
                c.add("tool", "booking", "create_appointment", ok=True, tool="book")
                ok = True
        if ok:
            c.add("agent", "confirm", "readback_datetime")
    c.add("system", "complete", "call_ended", gap=(150, 2500))

    return {
        "call_id": call_id,
        "source": "synthetic",
        "split": "synthetic",
        "site": rng.choice(SITES),
        "specialty": rng.choice(SPECIALTIES),
        "label_booked": any(e["event"] == "create_appointment" and e["ok"] for e in c.events),
        "events": c.events,
    }


def generate(n: int = 150, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    seeds = [{**s, "source": "synthetic", "split": "synthetic"} for s in json.loads(json.dumps(SEEDS))]
    return seeds + [_one(rng, f"syn-{i:04d}") for i in range(1, n + 1)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(HERE / "data" / "synthetic_calls.json"))
    a = ap.parse_args()
    calls = generate(a.n, a.seed)
    Path(a.out).write_text(json.dumps(calls, indent=1))
    booked = sum(c["label_booked"] for c in calls)
    print(f"wrote {len(calls)} calls ({booked} booked / {len(calls) - booked} not) to {a.out}")
