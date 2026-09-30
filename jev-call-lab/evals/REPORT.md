# Eval report (mock)

**Overall: PASS** (Jev mode: mock)


| ID | Eval | Gate | Result | Detail |
|---|---|---|---|---|
| E1 | Contract | gate | PASS | 2879/2879 responses valid |
| E2 | No leak | gate | PASS | 8710 prefix states checked |
| E3 | Reducer fidelity | gate | PASS | 915 calls, 556 booked / 359 not |
| E4 | Seed trajectories | gate | PASS | book-001 ends > 0.8: ok, book-001 rises over the call: ok, fail-002 ends < 0.2: ok, recover-003 dips after first failure: ok, recover-003 ends > 0.8: ok |
| E5 | Signal on SGD test | gate | PASS | see below |
| E6 | Record -> replay | gate | PASS | 24 fixtures over HTTP; replay identical=True; replay miss raises=True |
| E7 | Cost accounting | gate | PASS | see below |

## E5 Signal on SGD test

```json
{
  "sgd_test (baselines trained on sgd_train)": {
    "jev": {
      "mid": 0.9516,
      "pre_outcome": 0.9569,
      "end": 1.0,
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
      "mid": 0.6601,
      "pre_outcome": 0.7484,
      "end": 1.0,
      "pre_outcome_attempted": 0.6548
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
    "total_usd": 0.04138201,
    "per_call_usd": 0.0002478,
    "input_tokens": 985286,
    "source": [
      "tokens_x_price"
    ]
  },
  "projected_full_sgd": {
    "calls": 762,
    "requests": 8476,
    "input_tokens": 5036600,
    "usd": 0.211537,
    "token_scale_vs_chars_div_4": 1.0
  }
}
```

## SGD test trajectories and review (evaluate.py)

```
call                       y      mid_p   pre_p   end_p   mode (reference)
sgd-test-5_00068           False  0.599   0.550   0.182   info_only (info_only)
  trajectory:
    t=  3.9s  details      0.43 ████████             asked_for_info
    t=  9.2s  details      0.43 ████████             asked_for_info
    t= 15.1s  discovery    0.53 ██████████           search_providers
    t= 16.7s  offer        0.60 ███████████          offered_provider
    t= 16.7s  discovery    0.60 ███████████          gave_result_count
    t= 24.9s  info         0.55 ██████████           gave_info
    t= 34.1s  wrapup       0.18 ███                  said_goodbye
  review: progress=1.80 needs_human=0.12 recoverable_error=0.15

sgd-test-5_00069           False  0.599   0.731   0.015   declined_offer (declined_offer)
  trajectory:
    t=  3.9s  discovery    0.53 ██████████           search_providers
    t=  5.4s  offer        0.60 ███████████          offered_provider
    t=  5.5s  discovery    0.60 ███████████          gave_result_count
    t= 12.1s  offer        0.60 ███████████          offered_provider
    t= 19.6s  info         0.55 ██████████           gave_info
    t= 26.2s  offer        0.73 ██████████████       offered_to_book
    t= 32.8s  wrapup       0.01                      said_goodbye
  review: progress=1.80 needs_human=0.31 recoverable_error=0.40

sgd-test-5_00070           False  0.599   0.768   0.018   declined_offer (declined_offer)
  trajectory:
    t=  3.2s  details      0.43 ████████             asked_for_info
    t=  8.9s  discovery    0.53 ██████████           search_providers
    t= 10.4s  offer        0.60 ███████████          offered_provider
    t= 10.5s  discovery    0.60 ███████████          gave_result_count
    t= 17.4s  offer        0.77 ███████████████      offered_to_book
    t= 24.0s  wrapup       0.02                      asked_anything_else
    t= 30.3s  wrapup       0.02                      said_goodbye
  review: progress=1.80 needs_human=0.31 recoverable_error=0.40

sgd-test-5_00071           False  0.550   0.550   0.182   info_only (info_only)
  trajectory:
    t=  3.9s  discovery    0.53 ██████████           search_providers
    t=  5.4s  offer        0.60 ███████████          offered_provider
    t= 11.7s  info         0.55 ██████████           gave_info
    t= 19.0s  wrapup       0.18 ███                  said_goodbye
  review: progress=1.80 needs_human=0.12 recoverable_error=0.15

calls=167 booked=81 requests=1730 models=['mock-jev-heuristic']
AUC   mid=0.9516 pre_outcome=0.9569 end=1.0 pre_outcome_attempted=None (n=78, not booked=1)
Brier mid=0.1984 pre_outcome=0.2236 end=0.0111
failure_mode agreement with rule reference: 1.0
latency p50=0.04ms p95=0.07ms
total USD=0.041382 (tokens_x_price), input tokens=985286
```

## Seed trajectories (synthetic set summary at the bottom)

```
call                       y      mid_p   pre_p   end_p   mode (reference)
book-001                   True   0.750   0.802   0.990   clean_book (clean_book)
  trajectory:
    t=  4.2s  identity     0.43 ████████             asked_dob_and_name
    t= 11.2s  identity     0.53 ██████████           ehr_patient_lookup
    t= 15.8s  intent       0.75 ███████████████      confirmed_new_appointment
    t= 20.1s  availability 0.75 ███████████████      searched_open_slots
    t= 26.4s  offer        0.80 ████████████████     offered_two_slots
    t= 36.2s  booking      0.99 ███████████████████  create_appointment
    t= 40.1s  confirm      0.99 ███████████████████  readback_datetime
  review: progress=2.40 needs_human=0.01 recoverable_error=0.15

fail-002                   False  0.047   0.010   0.010   caller_drop (caller_drop)
  trajectory:
    t=  3.8s  identity     0.43 ████████             asked_dob_and_name
    t= 18.2s  identity     0.09 █                    repeated_identity_ask
    t= 24.1s  identity     0.05                      ehr_patient_lookup  ✗
    t= 26.8s  identity     0.02                      identity_check_failed  ✗
    t= 35.5s  handoff      0.01                      offered_human_transfer
  review: progress=0.60 needs_human=1.00 recoverable_error=0.40

recover-003                True   0.426   0.500   0.990   recovered_error (recovered_error)
  trajectory:
    t=  5.1s  identity     0.27 █████                ehr_patient_lookup  ✗
    t=  7.8s  identity     0.15 ███                  identity_check_failed  ✗
    t= 12.4s  identity     0.15 ███                  retried_with_phone_match
    t= 16.1s  identity     0.21 ████                 ehr_patient_lookup
    t= 19.8s  intent       0.43 ████████             confirmed_follow_up
    t= 24.2s  availability 0.43 ████████             searched_open_slots
    t= 28.6s  offer        0.50 ██████████           offered_one_slot
    t= 37.1s  availability 0.50 ██████████           searched_open_slots
    t= 44.8s  booking      0.99 ███████████████████  create_appointment
  review: progress=2.40 needs_human=0.07 recoverable_error=0.85

call                       y      mid_p   pre_p   end_p   mode (reference)
calls=153 booked=78 requests=1149 models=['mock-jev-heuristic']
AUC   mid=0.6601 pre_outcome=0.7484 end=1.0 pre_outcome_attempted=0.6548 (n=131, not booked=53)
Brier mid=0.2455 pre_outcome=0.1985 end=0.1394
failure_mode agreement with rule reference: 1.0
latency p50=0.03ms p95=0.04ms
total USD=0.019731 (tokens_x_price), input tokens=469786
```
