"""Typed question sets for Jev, and the prefix state it sees at each turn."""

WILL_BOOK = {
    "will_book": {
        "type": "noul",
        "instructions": (
            "This state is a structural trace of an AI receptionist call that schedules appointments "
            "(clinic or service provider). There is no transcript. Will this call end with a booked "
            "appointment written to the scheduling system?"
        ),
        "criteria": {
            "true": "A slot will be confirmed and created before the call ends.",
            "false": "The caller will leave without a booked appointment.",
        },
    }
}

# Five typed questions for a full-call review (Sully pattern). One request, one state.
REVIEW = {
    "will_book": WILL_BOOK["will_book"],
    "failure_mode": {
        "type": "choice",
        "instructions": "What is the dominant failure or success mode visible in the structure?",
        "criteria": {
            "clean_book": "Lookup, offer and booking all succeeded with little friction.",
            "recovered_error": "An earlier tool, identity or booking failure was later overcome and the call booked.",
            "identity_block": "Patient lookup or verification never succeeded.",
            "no_availability": "Slot search or booking kept failing and the call stalled there.",
            "info_only": "The caller only gathered information and never tried to book.",
            "declined_offer": "The caller was offered a booking or slot and turned it down.",
            "caller_drop": "Long silence, refusal, or hangup ended the call.",
            "agent_loop": "Repeated stages or tools without forward progress.",
            "other": "None of the above fits.",
        },
    },
    "agent_progress": {
        "type": "score",
        "instructions": "How far did the agent advance the scheduling workflow?",
        "criteria": [
            "Never got past greeting or identity",
            "Reached intent or search but made no real offer",
            "Offered a provider or slot but did not book",
            "Completed a confirmed booking",
        ],
    },
    "needs_human": {
        "type": "noul",
        "instructions": "Should a human receptionist have taken over, based on structure alone?",
        "criteria": {
            "true": "Stuck identity, refusal or silence, or repeated tool failure.",
            "false": "The agent was still making workflow progress.",
        },
    },
    "visible_error_overweight": {
        "type": "noul",
        "instructions": (
            "Does the trace contain an early visible error that a competent agent often recovers from "
            "(a failed lookup then retry, one failed booking then another slot)?"
        ),
    },
}

NOTE = "No audio or transcript is available. Judge from structure only."


def prefix_state(call: dict, events_so_far: list, forecast_after_event_index: int) -> dict:
    """What Jev sees at one turn: call metadata plus the prefix only.

    Unlike the tutorial, the call id stays out of the state: seed ids like
    'book-001' / 'fail-002' would hand Jev the label.
    """
    last = events_so_far[-1]
    return {
        "task": "appointment_scheduling",
        "site": call["site"],
        "specialty": call["specialty"],
        "forecast_after_event_index": forecast_after_event_index,
        "elapsed_ms": last["t_ms"],
        "current_stage": last["stage"],
        "current_event": last["event"],
        "fail_count_so_far": sum(1 for e in events_so_far if not e.get("ok", True)),
        "tool_calls_so_far": sum(1 for e in events_so_far if e.get("actor") == "tool"),
        "trace_so_far": events_so_far,
        "note": NOTE,
    }
