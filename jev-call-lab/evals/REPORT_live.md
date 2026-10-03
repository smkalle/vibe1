# Eval report (live)

**Overall: PASS** (Jev mode: live)


| ID | Eval | Gate | Result | Detail |
|---|---|---|---|---|
| E1 | Contract | gate | PASS | 2879/2879 responses valid |
| E2 | No leak | gate | PASS | 8710 prefix states checked |
| E3 | Reducer fidelity | gate | PASS | 915 calls, 556 booked / 359 not |
| E4 | Seed trajectories | report | ok | book-001 ends > 0.8: ok, book-001 rises over the call: ok, fail-002 ends < 0.2: ok, recover-003 dips after first failure: ok, recover-003 ends > 0.8: ok |
| E5 | Signal on SGD test | report | ok | see below |
| E6 | Record -> replay | gate | PASS | 24 fixtures over HTTP; replay identical=True; replay miss raises=True |
| E7 | Cost accounting | gate | PASS | see below |

## E5 Signal on SGD test

```json
{
  "sgd_test (baselines trained on sgd_train)": {
    "jev": {
      "mid": 0.8508,
      "pre_outcome": 0.7587,
      "end": 0.996,
      "pre_outcome_attempted": null
    },
    "B0_constant": {
      "mid": 0.5,
      "pre_outcome": 0.5,
      "end": 0.5,
      "pre_outcome_attempted": null
    },
    "B1_stage_reached": {
      "mid": 0.8667,
      "pre_outcome": 0.9653,
      "end": 0.9942,
      "pre_outcome_attempted": null
    },
    "B2_logreg_counts": {
      "mid": 0.8699,
      "pre_outcome": 0.947,
      "end": 0.992,
      "pre_outcome_attempted": null
    }
  },
  "synthetic (baselines 5-fold)": {
    "jev": {
      "mid": 0.7146,
      "pre_outcome": 0.8065,
      "end": 1.0,
      "pre_outcome_attempted": 0.7262
    },
    "B0_constant": {
      "mid": 0.5,
      "pre_outcome": 0.5,
      "end": 0.5,
      "pre_outcome_attempted": 0.5
    },
    "B1_stage_reached": {
      "mid": 0.8921,
      "pre_outcome": 0.7733,
      "end": 0.98,
      "pre_outcome_attempted": 0.6792
    },
    "B2_logreg_counts": {
      "mid": 0.7755,
      "pre_outcome": 0.7798,
      "end": 0.9985,
      "pre_outcome_attempted": 0.6984
    }
  }
}
```

## E7 Cost accounting

```json
{
  "this_run": {
    "total_usd": 0.08791087,
    "per_call_usd": 0.00052641,
    "input_tokens": 2093116,
    "source": [
      "usage.cost"
    ]
  },
  "projected_full_sgd": {
    "calls": 762,
    "requests": 8476,
    "input_tokens": 11081274,
    "usd": 0.465414,
    "token_scale_vs_chars_div_4": 2.2
  }
}
```

## SGD test trajectories and review (evaluate.py)

```
call                       y      mid_p   pre_p   end_p   mode (reference)
sgd-test-5_00068           False  0.490   0.360   0.100   info_only (info_only)
  trajectory:
    t=  3.9s  details      0.45 █████████            asked_for_info
    t=  9.2s  details      0.48 █████████            asked_for_info
    t= 15.1s  discovery    0.48 █████████            search_providers
    t= 16.7s  offer        0.49 █████████            offered_provider
    t= 16.7s  discovery    0.42 ████████             gave_result_count
    t= 24.9s  info         0.36 ███████              gave_info
    t= 34.1s  wrapup       0.10 ██                   said_goodbye
  review: progress=2.01 needs_human=0.05 recoverable_error=0.04

sgd-test-5_00069           False  0.360   0.560   0.050   declined_offer (declined_offer)
  trajectory:
    t=  3.9s  discovery    0.48 █████████            search_providers
    t=  5.4s  offer        0.50 ██████████           offered_provider
    t=  5.5s  discovery    0.42 ████████             gave_result_count
    t= 12.1s  offer        0.36 ███████              offered_provider
    t= 19.6s  info         0.33 ██████               gave_info
    t= 26.2s  offer        0.56 ███████████          offered_to_book
    t= 32.8s  wrapup       0.05 █                    said_goodbye
  review: progress=2.00 needs_human=0.32 recoverable_error=0.28

sgd-test-5_00070           False  0.440   0.580   0.050   declined_offer (declined_offer)
  trajectory:
    t=  3.2s  details      0.45 █████████            asked_for_info
    t=  8.9s  discovery    0.48 █████████            search_providers
    t= 10.4s  offer        0.50 ██████████           offered_provider
    t= 10.5s  discovery    0.44 ████████             gave_result_count
    t= 17.4s  offer        0.58 ███████████          offered_to_book
    t= 24.0s  wrapup       0.12 ██                   asked_anything_else
    t= 30.3s  wrapup       0.05 █                    said_goodbye
  review: progress=2.00 needs_human=0.26 recoverable_error=0.28

sgd-test-5_00071           False  0.380   0.380   0.140   info_only (info_only)
  trajectory:
    t=  3.9s  discovery    0.48 █████████            search_providers
    t=  5.4s  offer        0.49 █████████            offered_provider
    t= 11.7s  info         0.38 ███████              gave_info
    t= 19.0s  wrapup       0.14 ██                   said_goodbye
  review: progress=2.03 needs_human=0.06 recoverable_error=0.04

calls=167 booked=81 requests=1730 models=['typesafe/jev-1.13-20260917']
AUC   mid=0.8508 pre_outcome=0.7587 end=0.996 pre_outcome_attempted=None (n=78, not booked=1)
Brier mid=0.1662 pre_outcome=0.2017 end=0.0436
failure_mode agreement with rule reference: 0.6228
latency p50=358.42ms p95=505.67ms
total USD=0.087911 (usage.cost), input tokens=2093116
```

## Seed trajectories (synthetic set summary at the bottom)

```
call                       y      mid_p   pre_p   end_p   mode (reference)
book-001                   True   0.650   0.630   0.930   clean_book (clean_book)
  trajectory:
    t=  4.2s  identity     0.44 ████████             asked_dob_and_name
    t= 11.2s  identity     0.49 █████████            ehr_patient_lookup
    t= 15.8s  intent       0.73 ██████████████       confirmed_new_appointment
    t= 20.1s  availability 0.65 █████████████        searched_open_slots
    t= 26.4s  offer        0.63 ████████████         offered_two_slots
    t= 36.2s  booking      0.91 ██████████████████   create_appointment
    t= 40.1s  confirm      0.93 ██████████████████   readback_datetime
  review: progress=3.00 needs_human=0.03 recoverable_error=0.03

fail-002                   False  0.100   0.070   0.070   caller_drop (caller_drop)
  trajectory:
    t=  3.8s  identity     0.45 █████████            asked_dob_and_name
    t= 18.2s  identity     0.13 ██                   repeated_identity_ask
    t= 24.1s  identity     0.10 ██                   ehr_patient_lookup  ✗
    t= 26.8s  identity     0.08 █                    identity_check_failed  ✗
    t= 35.5s  handoff      0.07 █                    offered_human_transfer
  review: progress=0.29 needs_human=0.94 recoverable_error=0.52

recover-003                True   0.500   0.580   0.920   recovered_error (recovered_error)
  trajectory:
    t=  5.1s  identity     0.17 ███                  ehr_patient_lookup  ✗
    t=  7.8s  identity     0.11 ██                   identity_check_failed  ✗
    t= 12.4s  identity     0.38 ███████              retried_with_phone_match
    t= 16.1s  identity     0.46 █████████            ehr_patient_lookup
    t= 19.8s  intent       0.50 ██████████           confirmed_follow_up
    t= 24.2s  availability 0.57 ███████████          searched_open_slots
    t= 28.6s  offer        0.50 ██████████           offered_one_slot
    t= 37.1s  availability 0.58 ███████████          searched_open_slots
    t= 44.8s  booking      0.92 ██████████████████   create_appointment
  review: progress=3.00 needs_human=0.15 recoverable_error=0.97

call                       y      mid_p   pre_p   end_p   mode (reference)
calls=153 booked=78 requests=1149 models=['typesafe/jev-1.13-20260917']
AUC   mid=0.7146 pre_outcome=0.8065 end=1.0 pre_outcome_attempted=0.7262 (n=131, not booked=53)
Brier mid=0.2118 pre_outcome=0.1678 end=0.082
failure_mode agreement with rule reference: 0.8431
latency p50=362.22ms p95=517.75ms
total USD=0.040808 (usage.cost), input tokens=971624
```
