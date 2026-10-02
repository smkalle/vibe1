"""The structural event schema: closed vocabulary plus a validator (eval E3)."""
import re

ACTORS = {"system", "agent", "caller", "tool"}

STAGES = {
    "greeting", "identity", "intent", "discovery", "availability", "offer", "details",
    "info", "booking", "confirm", "handoff", "wrapup", "complete",
}

SYNTHETIC_EVENTS = {
    "call_started", "call_ended", "asked_dob_and_name", "provided_identifiers", "ehr_patient_lookup",
    "identity_check_failed", "retried_with_phone_match", "repeated_identity_ask", "refused_to_verify",
    "caller_went_silent", "confirmed_new_appointment", "confirmed_follow_up", "searched_open_slots",
    "offered_two_slots", "offered_one_slot", "accepted_slot", "declined_slots", "asked_for_later_week",
    "create_appointment", "readback_datetime", "offered_human_transfer", "hung_up",
    "agent_repeated_question",
}

SGD_EVENTS = {
    "call_started", "call_ended", "stated_intent_find", "stated_intent_book", "provided_info", "asked_info",
    "asked_for_alternatives", "selected_offer", "affirmed", "declined", "nothing_else",
    "accepted_booking_offer", "declined_booking_offer", "thanked", "said_goodbye", "search_providers",
    "create_appointment", "offered_provider", "offered_alternative_time", "gave_result_count",
    "asked_for_info", "gave_info", "offered_to_book", "readback_details", "booking_confirmed",
    "booking_failed", "asked_anything_else",
}

EVENTS = SYNTHETIC_EVENTS | SGD_EVENTS

SLOTS = {
    "address", "appointment_date", "appointment_time", "average_rating", "city", "count", "dentist_name",
    "doctor_name", "is_unisex", "offers_cosmetic_services", "phone_number", "street_address",
    "stylist_name", "therapist_name", "type",
}

TOOLS = {"find_patient", "search_slots", "book", "find_provider"}

SPECIALTIES = {
    "dermatology", "cardiology", "orthopedics", "pediatrics", "family_medicine",
    "salon", "dentist", "doctor", "therapist",
}

SITES = {"north-clinic", "west-clinic", "east-clinic", "south-clinic",
         "Services_1", "Services_2", "Services_3", "Services_4"}

EVENT_KEYS = {"t_ms", "actor", "stage", "event", "ok", "tool", "n_results", "slots"}


def vocabulary() -> set[str]:
    """Every lowercase word the schema itself can put into a state."""
    words = set()
    for group in (ACTORS, STAGES, EVENTS, SLOTS, TOOLS, SPECIALTIES, SITES, EVENT_KEYS):
        for item in group:
            words |= {w.lower() for w in re.findall(r"[A-Za-z]+", item)}
    return words


def validate_call(call: dict) -> list[str]:
    errors = []
    for key in ("call_id", "site", "specialty", "label_booked", "events"):
        if key not in call:
            errors.append(f"missing {key}")
    events = call.get("events") or []
    if not events:
        return errors + ["no events"]
    if call.get("site") not in SITES:
        errors.append(f"unknown site {call.get('site')!r}")
    if call.get("specialty") not in SPECIALTIES:
        errors.append(f"unknown specialty {call.get('specialty')!r}")
    if events[0]["event"] != "call_started":
        errors.append("first event is not call_started")
    if events[-1]["event"] != "call_ended":
        errors.append("last event is not call_ended")

    last_t = -1
    for i, e in enumerate(events):
        if set(e) - EVENT_KEYS:
            errors.append(f"event {i}: unexpected keys {sorted(set(e) - EVENT_KEYS)}")
        if not isinstance(e.get("t_ms"), int) or e["t_ms"] <= last_t:
            errors.append(f"event {i}: t_ms {e.get('t_ms')} not strictly increasing")
        else:
            last_t = e["t_ms"]
        if e.get("actor") not in ACTORS:
            errors.append(f"event {i}: unknown actor {e.get('actor')!r}")
        if e.get("stage") not in STAGES:
            errors.append(f"event {i}: unknown stage {e.get('stage')!r}")
        if e.get("event") not in EVENTS:
            errors.append(f"event {i}: unknown event {e.get('event')!r}")
        if not isinstance(e.get("ok"), bool):
            errors.append(f"event {i}: ok must be bool")
        if "tool" in e and e["tool"] not in TOOLS:
            errors.append(f"event {i}: unknown tool {e['tool']!r}")
        if (e.get("actor") == "tool") != ("tool" in e):
            errors.append(f"event {i}: tool events need a tool name, and only they")
        if set(e.get("slots", [])) - SLOTS:
            errors.append(f"event {i}: unknown slots {sorted(set(e['slots']) - SLOTS)}")

    booked = any(e["event"] == "create_appointment" and e["ok"] for e in events)
    if call.get("label_booked") is not booked:
        errors.append(f"label_booked={call.get('label_booked')} but ok create_appointment={booked}")
    return errors
