# Spec: jev-call-lab, structure-only call analysis with Jev

**Status:** BUILT and EVALUATED on the mock (2026-09-30). Awaiting user sign-off and a live run with a key (§9). Results are in §10.
**Source:** the "Build a Sully-style call-structure analyzer with Jev on OpenRouter" tutorial (the tutorial below).
**Lives in:** `jev-call-lab/` (standalone, no LitServe/MCP coupling, per decision D3).

## 1. Goal

Reproduce the Sully AI experiment in miniature. Jev sees only a call's **structure**: turns, tool calls, stages and timing. It never sees words. It answers typed questions:

1. **Turn-level forecasts.** After every agent or tool event, one `noul`: will this call end with a booked appointment?
2. **Five-question review.** One request per finished call: `will_book`, `failure_mode` (choice), `agent_progress` (score), `needs_human`, `visible_error_overweight`.
3. **Evaluation.** Trajectories, ranking quality (AUC at the midpoint, just before the outcome and at the end), calibration (Brier), cost and latency. Jev is compared against baselines that don't use an LLM.

The data is the tutorial's synthetic traces **plus real open-source dialogues** (Google SGD, §4), reduced to the same structural schema, with ground-truth booking labels.

## 2. Decisions taken with the user (before build)

| # | Question | Decision |
|---|---|---|
| D1 | Real-data verification set | **SGD `Services_1..4`** (salon, dentist, doctor, therapist): real `FindProvider`/`BookAppointment` tool calls with results, and a `NOTIFY_SUCCESS`/`NOTIFY_FAILURE` outcome that gives ground-truth `label_booked`. |
| D2 | No key in this sandbox (OpenRouter is also blocked by egress policy here) | **Heuristic mock + record/replay cache.** Mock answers use the exact wire shape. On the user's first live run, `JEV_MODE=record` stores real responses as fixtures, and `JEV_MODE=replay` re-runs them offline. |
| D3 | Placement | **Standalone `jev-call-lab/`**, laid out as in the tutorial. |
| D4 | Process | **spec → evals → build → test → run evals → sign-off.** This pass ends at a sign-off checklist (§9) for the user. |

## 3. Facts verified vs. the tutorial

The TypeSafe SDK source was checked (`typesafe-sdk` 0.7.2 from PyPI, whose schemas are generated from `api.typesafe.ai/openapi.json`). OpenRouter and the TypeSafe docs are not reachable from this sandbox.

| Tutorial claim | Verified | Consequence |
|---|---|---|
| Request `{model, state, questions}`; `state` is a string, object or array | Yes (`SystemOneRequest`) | Used as is. |
| `noul` answer = `{type, noul}` | Yes | |
| `choice` answer = `{type, choice, confidence, probabilities}` | Yes | |
| `score` answer = `{type, score, probabilities}` | **It also has `confidence` and `legend`**; probabilities are keyed `"0".."n"` | Mock and validator include them. |
| `usage.cost` in the response | **Not in the TypeSafe schema** (only `input_tokens`, `output_tokens`); OpenRouter may add it | Cost = `usage.cost` if present, else `input_tokens × $0.042/1M`. The report says which. |
| TypeSafe endpoint `/v1/systemone`, Bearer auth | Yes (`SYSTEM_ONE_PATH`) | |
| OpenRouter `/api/alpha/decisions`, model `typesafe/jev-1.13` | **Unverified** (blocked) | Both configurable (`JEV_URL`, `JEV_MODEL`). First live call is a smoke test (§8). |
| Retries | The SDK retries 408/429/5xx, 3 times | Client does the same, with exponential backoff. |

**A label leak in the tutorial:** `prefix_state` puts `call_id` into `state`, and the seed ids are `book-001` and `fail-002`. Here the id stays outside `state`. Eval E2 enforces that `state` contains no id, no label and no transcript text.

## 4. Data

### 4.1 Event schema (one call)

```json
{"call_id": "...", "source": "synthetic|sgd", "split": "train|dev|test|synthetic",
 "site": "...", "specialty": "...", "label_booked": true,
 "events": [{"t_ms": 0, "actor": "system|agent|caller|tool", "stage": "...",
             "event": "...", "ok": true, "tool": "...", "n_results": 4, "slots": ["appointment_date"]}]}
```

`tool`, `n_results` and `slots` are optional. `slots` holds schema slot **names**, never values, so there are no names, phones or addresses.

### 4.2 Synthetic (`generate_synthetic.py`)

- The tutorial's three seed calls, verbatim (`book-001`, `fail-002`, `recover-003`).
- About 150 more calls sampled from the stage machine `greeting → identity → intent → availability → offer → booking → confirm → complete`, seeded (`--seed 7`). Failures are injected at fixed rates: identity lookup fails, the caller goes silent, the slot search is empty, the caller asks for another week, the caller hangs up, the agent loops.
- `label_booked` = `create_appointment ok` happened and the caller did not hang up first.

### 4.3 SGD reducer (`reduce_sgd.py`)

- Fetch `dialogues_*.json` for train/dev/test from `github.com/google-research-datasets/dstc8-schema-guided-dialogue`, and keep **single-service** dialogues whose service starts with `Services_`. There are 762: train 3/4 of them, with dev and test the rest; exact counts go in the eval report. Raw files are cached under `data/raw/` (gitignored). The reduced traces are committed, with CC BY-SA 4.0 attribution in `data/SGD_ATTRIBUTION.md`.
- Mapping (dialogue acts and service calls → events; **no utterance text is copied**):

| SGD | actor | event | stage | ok |
|---|---|---|---|---|
| (start) | system | `call_started` | greeting | ✓ |
| USER `INFORM_INTENT` FindProvider / BookAppointment | caller | `stated_intent_find` / `stated_intent_book` | discovery / details | ✓ |
| USER `INFORM` slots | caller | `provided_info` + slots | current | ✓ |
| USER `REQUEST` slots | caller | `asked_info` + slots | info | ✓ |
| USER `REQUEST_ALTS` | caller | `asked_for_alternatives` | offer | ✓ |
| USER `SELECT` | caller | `selected_offer` | offer | ✓ |
| USER `AFFIRM` / `NEGATE` | caller | `affirmed` / `declined` (a `NEGATE` answering `REQ_MORE` becomes `nothing_else`, wrapup, ✓) | confirm | ✓ / ✗ |
| USER `AFFIRM_INTENT` / `NEGATE_INTENT` | caller | `accepted_booking_offer` / `declined_booking_offer` | details / wrapup | ✓ / ✗ |
| USER `THANK_YOU`, `GOODBYE` | caller | `thanked`, `said_goodbye` | wrapup | ✓ |
| SYSTEM service_call FindProvider | tool | `search_providers` (`tool=find_provider`, `n_results`) | discovery | n>0 |
| SYSTEM service_call BookAppointment | tool | `create_appointment` (`tool=book`) | booking | result non-empty |
| SYSTEM `OFFER` provider slots / time slots | agent | `offered_provider` / `offered_alternative_time` | offer | ✓ |
| SYSTEM `INFORM_COUNT` | agent | `gave_result_count` | discovery | ✓ |
| SYSTEM `REQUEST` slots | agent | `asked_for_info` + slots | details | ✓ |
| SYSTEM `INFORM` slots | agent | `gave_info` + slots | info | ✓ |
| SYSTEM `OFFER_INTENT` | agent | `offered_to_book` | offer | ✓ |
| SYSTEM `CONFIRM` | agent | `readback_details` | confirm | ✓ |
| SYSTEM `NOTIFY_SUCCESS` / `NOTIFY_FAILURE` | agent | `booking_confirmed` / `booking_failed` | booking | ✓ / ✗ |
| SYSTEM `REQ_MORE`, `GOODBYE` | agent | `asked_anything_else`, `said_goodbye` | wrapup | ✓ |
| (end) | system | `call_ended` | complete | ✓ |

- In a system turn, the tool event comes **before** the agent's events, because the service call happens before the reply.
- **Timing is synthetic** (SGD is text): turn duration = `900 ms + 320 ms × word_count` and tool latency is 1500 ms. Only word *counts* are used, never words.
- `label_booked` = any `NOTIFY_SUCCESS`. Eval E3 checks that this equals "some `create_appointment` returned ok".
- Built: train 551 (360 booked), dev 44 (37), test 167 (81). The SGD **test split holds unseen services and flows** (salon + therapist only; therapist never appears in train). The profile seen during spec: 431 booked / 331 not; 75 booked calls **recover from a failed booking** (the Sully bias case); 227 unbooked calls never state a booking intent.

### 4.4 Known limitations of SGD as a stand-in

SGD calls are written, cooperative and simulated. They have no silence, no hangups and no identity step. Negative calls are mostly "info only" or "declined the offer", not caller drop-off. Real AUCs on SGD will therefore not match Sully's phone numbers. SGD tests the **method** on real dialogue structure with real labels. Synthetic data covers the phone-specific failure modes.

## 5. Components (`jev-call-lab/`)

| File | Role |
|---|---|
| `jev_client.py` | `decide(state, questions) → {model, answers, usage, _latency_ms, _cost_usd, _cost_source}`. Modes via `JEV_MODE`: `live`, `mock`, `record`, `replay`; `auto` (the default) is `live` if `OPENROUTER_API_KEY` is set, else `mock`. Retries 408/429/5xx with 1s, 2s, 4s backoff. `validate_response()` checks each answer against its question. |
| `mock_jev.py` | Deterministic heuristic engine that returns the exact wire shape, plus `python mock_jev.py --port 8765`, a local HTTP server at `/api/alpha/decisions` so the full HTTP path can be tested with curl. |
| `questions.py` | `WILL_BOOK`, `REVIEW` (5 questions; `failure_mode` gains `info_only`, `declined_offer` and a leftover `other`), `prefix_state()` without `call_id`. |
| `generate_synthetic.py` | §4.2 |
| `reduce_sgd.py` | §4.3 (`--fetch` downloads; otherwise uses the cache) |
| `simulate.py` | Forecasts after every agent/tool event, plus one review per call. `--data`, `--out`, `--workers` (8), `--limit`, `--no-review`. |
| `evaluate.py` | Trajectories (text bars), mid / pre-outcome / end AUC, Brier, a calibration table, a `failure_mode` confusion table vs. derived truth, cost (with its source), p50/p95 latency, and a projected full-dataset cost. `--json` for machine output. |
| `baselines.py` | B0 constant, B1 "furthest stage reached" ordinal, B2 logistic regression (pure Python) on event-count features, trained on SGD train and scored on the same prefixes. |
| `run_evals.py` | Runs E1–E7 and writes `evals/REPORT.md`. |
| `tests/` | pytest unit tests for each module; they use the mock and replay, never the network. |

**Mock heuristic.** It is intentionally simple, it is **not** Jev, and it is never fitted to labels. The logit is the sum of weights for:

- **Positive:** booking intent, details stage reached, readback + affirm, booking confirmed.
- **Negative:** each failure, with a deliberate weight so the "visible-error overweight" dip is visible; declined offer, silence, hangup, and ending without a booking.

Only the known question names get specific answers; any other question gets a neutral answer. Its purpose is to exercise the pipeline and give the evaluator real signal. Its numbers are a pipeline check, not a Jev result.

## 6. Metrics (definitions)

- **Forecast points:** every event with `actor ∈ {agent, tool}`. The state is `events[: i+1]` (prefix only).
- **mid:** the forecast at the middle forecast point.
- **pre-outcome:** the last forecast strictly before the first event whose stage is `booking`, `wrapup` or `complete`. *(Changed during the build. The first definition, "before the first `create_appointment`", cut booked calls early but let unbooked calls run to their goodbye, which inflated AUC to 0.99.)*
- **pre-outcome, attempted subset:** the same point, but only on calls that stated a booking intent before it: "will this attempt land?". It is reported as `null` when either class has fewer than 5 calls.
- **end:** the last forecast. This is a sanity check and should be close to 1.0.
- **AUC:** Mann–Whitney with ties counted as 0.5. **Brier:** mean (p − y)².
- **Cost:** Σ cost over all requests. **Projection:** tokens per request × requests for the full SGD set × price.

## 7. Evals (written before the build)

| ID | Eval | Gate (this pass, mock) |
|---|---|---|
| E1 | **Contract:** every response validates. `noul` ∈ [0,1]; `choice` ∈ criteria keys with probabilities summing to ≈1; `score` ∈ [0, n−1] with `legend`/`probabilities` keys `"0".."n-1"`; `usage.input_tokens` is an int | 100% |
| E2 | **No leak:** each prefix state equals `events[:i+1]`. `state` has no `call_id`, no `label*` and no word from any SGD utterance outside the closed event/slot/stage vocabulary | 100% |
| E3 | **Reducer fidelity:** `label_booked` ⇔ ok `create_appointment`; starts `call_started` and ends `call_ended`; `t_ms` strictly increasing; every event name is in the vocabulary; counts are reported | 100% |
| E4 | **Seed trajectories:** book-001 ends > 0.8 and rises after `accepted_slot`; fail-002 ends < 0.2; recover-003 dips after the first failure, then ends > 0.8 | pass |
| E5 | **Signal on the SGD test split:** pre-outcome AUC ≥ 0.70 and end AUC ≥ 0.95, with baselines B0–B2 (trained on SGD train) reported alongside. Also reported: the full synthetic set, with 5-fold baselines | pass (mock) |
| E6 | **Record → replay:** record through the local HTTP mock server, then replay; results are identical and a replay miss is a hard error | pass |
| E7 | **Cost accounting:** cost source reported; projected live cost of a full SGD run (forecasts + reviews) | < $1 |

**Live run (the user, with a key):** the same E1–E7. E4 and E5 are **reported, not gated** because they measure Jev. Reference points: Sully reported a midpoint AUC of 0.78 and near-end ranking of about 94% on 2,029 real calls. Jev should beat B1 at the pre-outcome point to be interesting.

## 8. Running it live (for the user)

```bash
cd jev-call-lab && pip install -r requirements.txt
export OPENROUTER_API_KEY=sk-or-v1-...
python jev_client.py --smoke                       # 1 request: verifies URL, model, answer shape
JEV_MODE=record python simulate.py --data data/synthetic_calls.json --limit 3 --out results/seed_live.json
python evaluate.py results/seed_live.json
JEV_MODE=record python simulate.py --data data/sgd_test.json --workers 16 --out results/sgd_test_live.json
python run_evals.py --results results/sgd_test_live.json --live
```

If the smoke test returns 404, set `JEV_URL=https://openrouter.ai/api/v1/systemone` and `JEV_MODEL=jev-1.13` (the SDK path).

## 9. Sign-off checklist

- [x] E1–E7 pass on the mock (see `jev-call-lab/evals/REPORT.md`)
- [x] pytest suite green (26 tests, offline)
- [x] No key in the repo; `state` is structure-only (E2); prefixes never include future events (E2)
- [x] Multiple questions share one request when they share one state (the review is one request)
- [x] `usage`, cost (with its source) and latency are logged per request
- [x] Labels come from SGD / generator ground truth, never from Jev (E3)
- [ ] **User:** review this spec and the eval report, and sign off on the approach
- [ ] **User:** `python jev_client.py --smoke` with a key (confirms the unverified endpoint and model id)
- [ ] **User:** a live `record` run on synthetic + SGD test (about $0.06), then `run_evals.py --live`

## 9b. Workbench (added 2026-10-02)

`jev-call-lab/app.py` (Streamlit). The key is entered per browser session and passed per run through `jev_client.Config`
(never through environment variables, so concurrent sessions can't see each other's keys, and `repr(Config)` hides it).
Tabs: Run, Results, Call explorer, Sign-off (E1–E7 via `run_evals.build_report`), Playground. Headless-tested with Streamlit
`AppTest`: mock run, live run against the local mock HTTP server with a fake key, test connection, playground and a full
mock sign-off (PASS). The key is never stored with the runs.

**Live sign-off status:** not run. The sandbox's network policy denies `openrouter.ai`, so the live smoke test and live E1–E7 are
still open (§9) and are meant to be run from the workbench's Sign-off tab.

## 10. Results

All numbers below come from the **mock** (`mock-jev-heuristic`, hand-set weights, not Jev). They validate the pipeline and set the baselines Jev has to beat; they say nothing about Jev yet.

**Gates:** E1 2,879/2,879 responses valid · E2 8,710 prefix states, no id/label/transcript word · E3 915 calls valid, labels consistent · E4 all 5 seed checks · E5 pass · E6 24 fixtures recorded over HTTP, replay identical, replay miss raises · E7 pass.

**AUC** (the mock row is the pipeline check; B-rows are what live Jev has to beat):

| Set | Scorer | mid | pre-outcome | pre-outcome, attempted | end |
|---|---|---|---|---|---|
| SGD test (167) | mock | 0.952 | 0.957 | n/a (1 negative) | 1.000 |
| | B1 stage reached | 0.867 | 0.965 | n/a | 0.994 |
| | B2 logreg (trained on SGD train) | 0.870 | 0.947 | n/a | 0.992 |
| Synthetic (153) | mock | 0.660 | 0.748 | 0.655 | 1.000 |
| | B1 stage reached | 0.892 | 0.773 | 0.679 | 0.980 |
| | B2 logreg (5-fold) | 0.776 | 0.780 | 0.698 | 0.999 |

**Cost:** mock token estimate (4 chars/token): SGD test 1,730 requests, about 985k tokens, **$0.041**; synthetic 1,149 requests, **$0.020**. Projected full SGD (762 calls, 8,476 requests): about 5.0M tokens, **$0.21**. The live run rescales this projection with real token counts.

**What we learned while building**

1. **SGD is easy at pre-outcome.** Unbooked SGD calls are mostly information-only: they never state a booking intent, so "furthest stage reached" already gets 0.97. Only 1 of 78 test calls that attempted a booking failed, so SGD can't measure "will this attempt land?". The synthetic set can (53 of 131 attempts fail), and there baselines reach about 0.70. **That is the number to watch on the live run.**
2. **The logistic baseline needed shuffled SGD training.** Unshuffled, it scored 0.55 mid AUC on test.
3. **The tutorial leaks the label through `call_id`, and its "pre-outcome" point was asymmetric.** Both are fixed, and both are guarded by evals.
4. The mock's `failure_mode` agrees 100% with the rule reference because it *is* the rule. On a live run, that agreement becomes a meaningful measure.
