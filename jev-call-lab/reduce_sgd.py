"""Reduce Google SGD Services dialogues to structure-only call traces.

Source: https://github.com/google-research-datasets/dstc8-schema-guided-dialogue
(CC BY-SA 4.0). Only dialogue acts, slot NAMES and service calls are kept;
no utterance text or slot values are copied. Timing is synthetic, from word counts.

    python reduce_sgd.py --fetch     # download into data/raw/, then reduce
    python reduce_sgd.py             # reduce from data/raw/ cache
"""
import argparse
import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
BASE = "https://raw.githubusercontent.com/google-research-datasets/dstc8-schema-guided-dialogue/master"
N_FILES = {"train": 127, "dev": 20, "test": 34}
SPECIALTY = {"Services_1": "salon", "Services_2": "dentist", "Services_3": "doctor", "Services_4": "therapist"}
TIME_SLOTS = {"appointment_date", "appointment_time"}

TURN_BASE_MS = 900
MS_PER_WORD = 320
TOOL_MS = 1500


def fetch(workers: int = 12):
    """Download every split file, keep only dialogues touching a Services_* service."""
    RAW.mkdir(parents=True, exist_ok=True)

    def get(job):
        split, i = job
        with urllib.request.urlopen(f"{BASE}/{split}/dialogues_{i:03d}.json", timeout=120) as r:
            dialogues = json.load(r)
        return split, [d for d in dialogues if any(s.startswith("Services_") for s in d["services"])]

    jobs = [(s, i) for s, n in N_FILES.items() for i in range(1, n + 1)]
    kept = {s: [] for s in N_FILES}
    with ThreadPoolExecutor(workers) as pool:
        for split, ds in pool.map(get, jobs):
            kept[split] += ds
    for split, ds in kept.items():
        (RAW / f"services_{split}.json").write_text(json.dumps(ds))
        print(f"fetched {split}: {len(ds)} dialogues touching Services_*")


def _slots(actions, act):
    out = []
    for a in actions:
        if a["act"] == act and a["slot"] and a["slot"] != "intent" and a["slot"] not in out:
            out.append(a["slot"])
    return out


def _acts(actions):
    seen = []
    for a in actions:
        if a["act"] not in seen:
            seen.append(a["act"])
    return seen


def _user_events(frame, prev_system_acts):
    actions = frame["actions"]
    active = (frame.get("state") or {}).get("active_intent")
    events = []
    for act in _acts(actions):
        slots = _slots(actions, act)
        if act == "INFORM_INTENT":
            intent = next((a.get("values", [None])[0] for a in actions if a["act"] == act and a.get("values")), active)
            if intent == "BookAppointment":
                events.append(("stated_intent_book", "details", True, []))
            else:
                events.append(("stated_intent_find", "discovery", True, []))
        elif act == "INFORM":
            stage = "details" if set(slots) & TIME_SLOTS or active == "BookAppointment" else "discovery"
            events.append(("provided_info", stage, True, slots))
        elif act == "REQUEST":
            events.append(("asked_info", "info", True, slots))
        elif act == "REQUEST_ALTS":
            events.append(("asked_for_alternatives", "offer", True, []))
        elif act == "SELECT":
            events.append(("selected_offer", "offer", True, []))
        elif act == "AFFIRM":
            events.append(("affirmed", "confirm", True, []))
        elif act == "NEGATE":
            if "REQ_MORE" in prev_system_acts:
                events.append(("nothing_else", "wrapup", True, []))
            else:
                events.append(("declined", "confirm", False, []))
        elif act == "AFFIRM_INTENT":
            events.append(("accepted_booking_offer", "details", True, []))
        elif act == "NEGATE_INTENT":
            events.append(("declined_booking_offer", "wrapup", False, []))
        elif act == "THANK_YOU":
            events.append(("thanked", "wrapup", True, []))
        elif act == "GOODBYE":
            events.append(("said_goodbye", "wrapup", True, []))
        else:
            raise ValueError(f"unmapped USER act {act}")
    return events


def _system_events(frame):
    actions = frame["actions"]
    events = []
    for act in _acts(actions):
        slots = _slots(actions, act)
        if act == "OFFER":
            if set(slots) and set(slots) <= TIME_SLOTS:
                events.append(("offered_alternative_time", "offer", True, slots))
            else:
                events.append(("offered_provider", "offer", True, slots))
        elif act == "INFORM_COUNT":
            events.append(("gave_result_count", "discovery", True, []))
        elif act == "REQUEST":
            events.append(("asked_for_info", "details", True, slots))
        elif act == "INFORM":
            events.append(("gave_info", "info", True, slots))
        elif act == "OFFER_INTENT":
            events.append(("offered_to_book", "offer", True, []))
        elif act == "CONFIRM":
            events.append(("readback_details", "confirm", True, slots))
        elif act == "NOTIFY_SUCCESS":
            events.append(("booking_confirmed", "booking", True, []))
        elif act == "NOTIFY_FAILURE":
            events.append(("booking_failed", "booking", False, []))
        elif act == "REQ_MORE":
            events.append(("asked_anything_else", "wrapup", True, []))
        elif act == "GOODBYE":
            events.append(("said_goodbye", "wrapup", True, []))
        else:
            raise ValueError(f"unmapped SYSTEM act {act}")
    return events


def _words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def reduce_dialogue(d: dict, split: str) -> dict:
    service = d["services"][0]
    events = [{"t_ms": 0, "actor": "system", "stage": "greeting", "event": "call_started", "ok": True}]
    t = 400
    prev_system_acts = []

    def add(actor, name, stage, ok, slots=None, **extra):
        nonlocal t
        e = {"t_ms": t, "actor": actor, "stage": stage, "event": name, "ok": ok, **extra}
        if slots:
            e["slots"] = slots
        events.append(e)
        t += 50  # events within one turn are a few ms apart; keeps t_ms strictly increasing

    for turn in d["turns"]:
        start = t
        for frame in turn["frames"]:
            if turn["speaker"] == "USER":
                for name, stage, ok, slots in _user_events(frame, prev_system_acts):
                    add("caller", name, stage, ok, slots)
            else:
                call = frame.get("service_call")
                if call:
                    results = frame.get("service_results") or []
                    if call["method"] == "FindProvider":
                        add("tool", "search_providers", "discovery", bool(results), tool="find_provider", n_results=len(results))
                    else:
                        add("tool", "create_appointment", "booking", bool(results), tool="book")
                    t += TOOL_MS
                for name, stage, ok, slots in _system_events(frame):
                    add("agent", name, stage, ok, slots)
                prev_system_acts = _acts(frame["actions"])
        t = max(t, start + TURN_BASE_MS + MS_PER_WORD * _words(turn["utterance"]))

    events.append({"t_ms": t + 500, "actor": "system", "stage": "complete", "event": "call_ended", "ok": True})
    return {
        "call_id": f"sgd-{split}-{d['dialogue_id']}",
        "source": "sgd",
        "split": split,
        "site": service,
        "specialty": SPECIALTY[service],
        "label_booked": any(e["event"] == "create_appointment" and e["ok"] for e in events),
        "events": events,
    }


def reduce_all():
    vocab = set()
    for split in N_FILES:
        dialogues = json.loads((RAW / f"services_{split}.json").read_text())
        single = [d for d in dialogues if len(d["services"]) == 1]
        calls = [reduce_dialogue(d, split) for d in single]
        for d in single:
            for turn in d["turns"]:
                vocab |= {w.lower() for w in re.findall(r"[A-Za-z]+", turn["utterance"])}
        (HERE / "data" / f"sgd_{split}.json").write_text(json.dumps(calls, indent=None, separators=(",", ":")))
        booked = sum(c["label_booked"] for c in calls)
        print(f"{split}: {len(calls)} single-service calls, {booked} booked / {len(calls) - booked} not")
    # Word list used by eval E2 to prove no transcript word reaches a state.
    (HERE / "data" / "sgd_utterance_vocab.json").write_text(json.dumps(sorted(vocab)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()
    if args.fetch:
        fetch()
    reduce_all()
