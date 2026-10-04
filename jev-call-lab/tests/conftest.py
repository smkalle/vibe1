import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    """Tests never touch the network: mock mode, no key, private fixtures dir."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("JEV_MODE", "mock")
    monkeypatch.setenv("LLM_MODE", "mock")
    monkeypatch.setenv("JEV_FIXTURES", str(tmp_path / "fixtures"))
    monkeypatch.setenv("JEV_WORKSPACE", str(tmp_path / "workspace"))


@pytest.fixture
def seed_calls():
    calls = json.loads((ROOT / "data" / "synthetic_calls.json").read_text())
    return {c["call_id"]: c for c in calls if c["call_id"] in {"book-001", "fail-002", "recover-003"}}


# A trimmed real SGD Services_2 dialogue (train), booked after a failed first attempt.
SGD_RECOVERED = {
    "dialogue_id": "t_1",
    "services": ["Services_2"],
    "turns": [
        {"speaker": "USER", "utterance": "Find a dentist in Gilroy please",
         "frames": [{"service": "Services_2", "actions": [{"act": "INFORM", "slot": "city"}, {"act": "INFORM_INTENT", "slot": "intent"}],
                     "state": {"active_intent": "FindProvider"}}]},
        {"speaker": "SYSTEM", "utterance": "I found 10 dentists. How about Albert Lee?",
         "frames": [{"service": "Services_2", "actions": [{"act": "OFFER", "slot": "dentist_name"}, {"act": "INFORM_COUNT", "slot": "count"}],
                     "service_call": {"method": "FindProvider", "parameters": {"city": "Gilroy"}},
                     "service_results": [{"dentist_name": "Albert Lee"}] * 10}]},
        {"speaker": "USER", "utterance": "Book it for March 13 at 11:30",
         "frames": [{"service": "Services_2", "actions": [{"act": "INFORM_INTENT", "slot": "intent"}, {"act": "INFORM", "slot": "appointment_date"}],
                     "state": {"active_intent": "BookAppointment"}}]},
        {"speaker": "SYSTEM", "utterance": "Please confirm Albert Lee on March 13 at 11:30",
         "frames": [{"service": "Services_2", "actions": [{"act": "CONFIRM", "slot": "dentist_name"}]}]},
        {"speaker": "USER", "utterance": "Yes",
         "frames": [{"service": "Services_2", "actions": [{"act": "AFFIRM", "slot": ""}], "state": {"active_intent": "BookAppointment"}}]},
        {"speaker": "SYSTEM", "utterance": "Sorry that failed. How about 12 pm?",
         "frames": [{"service": "Services_2", "actions": [{"act": "NOTIFY_FAILURE", "slot": ""}, {"act": "OFFER", "slot": "appointment_time"}],
                     "service_call": {"method": "BookAppointment", "parameters": {}}, "service_results": []}]},
        {"speaker": "USER", "utterance": "Sure",
         "frames": [{"service": "Services_2", "actions": [{"act": "AFFIRM", "slot": ""}], "state": {"active_intent": "BookAppointment"}}]},
        {"speaker": "SYSTEM", "utterance": "Booked. Anything else?",
         "frames": [{"service": "Services_2", "actions": [{"act": "NOTIFY_SUCCESS", "slot": ""}, {"act": "REQ_MORE", "slot": ""}],
                     "service_call": {"method": "BookAppointment", "parameters": {}}, "service_results": [{"ok": "True"}]}]},
        {"speaker": "USER", "utterance": "No thanks",
         "frames": [{"service": "Services_2", "actions": [{"act": "NEGATE", "slot": ""}, {"act": "THANK_YOU", "slot": ""}],
                     "state": {"active_intent": "NONE"}}]},
        {"speaker": "SYSTEM", "utterance": "Bye",
         "frames": [{"service": "Services_2", "actions": [{"act": "GOODBYE", "slot": ""}]}]},
    ],
}


@pytest.fixture
def sgd_recovered():
    return json.loads(json.dumps(SGD_RECOVERED))
