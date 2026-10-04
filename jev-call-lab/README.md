# jev-call-lab

A small, reproducible version of the Sully AI experiment. **Jev** (TypeSafe, served on OpenRouter)
sees only a call's *structure* (turns, tool calls, stages and timing, no words) and answers typed
questions: "will this call book?" after every agent turn, plus a five-question review per call.

Spec, decisions and results: [`../specs/jev-call-analysis.md`](../specs/jev-call-analysis.md).
Latest eval report: [`evals/REPORT.md`](evals/REPORT.md).

## Data

| File | What | Calls |
|---|---|---|
| `data/synthetic_calls.json` | The tutorial's 3 seed calls + 150 seeded phone-style traces (silence, hangup, identity block, slot drought, recoveries) | 153 |
| `data/sgd_{train,dev,test}.json` | **Real dialogues**: Google SGD `Services_1..4` (salon / dentist / doctor / therapist), reduced to structure with ground-truth booking labels (CC BY-SA 4.0, see `data/SGD_ATTRIBUTION.md`) | 551 / 44 / 167 |

Regenerate them with `python generate_synthetic.py` and `python reduce_sgd.py --fetch`.

## Workbench UI (v2: guided benchmark workflow)

```bash
./run.sh          # creates .venv, installs deps, opens http://localhost:8501
./run.sh check    # offline tests + evals E1-E7 instead
```

`run.sh` uses a venv because Arch-based systems (e.g. Omarchy) block system-wide pip. Manual equivalent:
`python -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/streamlit run app.py`.
Start in **replay** mode (the default) for real recorded Jev and GLM answers with no key and no spend. Paste an OpenRouter
key in step ① only for live, record or the latency probe.

Spec: [`../specs/jev-workbench-v2.md`](../specs/jev-workbench-v2.md). Two views share one app:

- **Workflow** (operators), six steps:
  1. **Connect:** pick a source *per approach*. Replay (real recorded live answers, free), live, record or mock.
     The OpenRouter key is held in session memory only; "Forget key" clears it.
  2. **Corpus:** a built-in corpus, plus a profile card including *how readable the outcome is from structure alone*
     (if a free rule already scores high, LLMs are unlikely to add accuracy).
  3. **Plan:** use cases (in-call escalation, post-call QA, bulk analytics), approaches (two free rules, Jev, GLM),
     a **frozen, stratified, seeded sample** with its expected precision, and expected and worst-case cost with a cap.
  4. **Run:** results are saved per call (resumable) and the run pauses at the cap. The **latency probe** measures 20
     sequential decisions after 3 warm-ups. You can import external results, stating their provenance explicitly.
  5. **Compare:** a plain-language verdict per use case, accuracy vs cost with 95% CIs, latency vs budget, and cost at
     your monthly volume. With technical details on: paired CIs for every metric, a paired-difference plot, and a call overlay.
  6. **Decide & export:** a quality gate per approach (E1 contract, E2 no leak, secret scan of the workspace),
     recorded decisions, and a Markdown report, scorecard JSON and run manifests.
- **Summary** (business readers): the verdict cards and charts for the project's latest benchmark.
- **Tools** (with technical details on): Audit, lab sign-off (E1–E7) and Playground (Jev or GLM).

Benchmark rules the engine enforces (`benchmark.py`, `stats.py`, `sampling.py`):
- Every approach scores the same frozen sample; mismatched samples, corpora or question sets are refused.
- CIs come from a paired cluster bootstrap over calls.
- A winner needs a paired CI that excludes 0. Otherwise it's a tie, broken by cost, then latency.
- Approaches over the latency budget are excluded. Mock and old-replay timings never count as latency.

Work is saved under `workspace/<project>/` (gitignored; `$JEV_WORKSPACE` overrides it): samples, run manifests
(never the key), per-call results. The CLI does the same:
`python benchmark.py --project demo --corpus synthetic --n 120 --approaches rules_b1,rules_b2,jev,glm`.

## Run without a key (mock)

```bash
pip install -r requirements.txt
python -m pytest -q tests                 # unit tests (never touch the network)
python run_evals.py                       # E1-E7 -> evals/REPORT.md
JEV_MODE=mock python simulate.py --data data/synthetic_calls.json --limit 3 --out results/seed.json
python evaluate.py results/seed.json
python mock_jev.py --port 8765            # local HTTP stand-in; point JEV_URL at it
```

The mock is a hand-weighted heuristic, **not Jev**. Its numbers only prove that the pipeline carries signal.

## Run live (your OpenRouter key)

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
python jev_client.py --smoke                                   # 1 request: URL, model, answer shape
JEV_MODE=record python simulate.py --data data/synthetic_calls.json --out results/synthetic_live.json
JEV_MODE=record python simulate.py --data data/sgd_test.json --workers 16 --out results/sgd_test_live.json
python run_evals.py --live --results results/sgd_test_live.json --synthetic-results results/synthetic_live.json
```

`record` saves every response under `fixtures/`. Afterwards, `JEV_MODE=replay` re-runs everything
offline for free, and commit the fixtures if you want the run to be reproducible.
Projected cost of the full SGD set (8,476 requests, about 5M input tokens): about **$0.21**.

GLM scorer (`llm_client.py`, default `z-ai/glm-5.3` on the same key): answers the same typed
questions through OpenRouter chat completions with JSON repair retries, so Jev wire shape,
contract checks and evaluators work unchanged. Reasoning is always on (effort `low` by default);
output/reasoning tokens bill ~$4.40/1M, so benchmark small N first. Fixtures are `glm_`-prefixed
to never collide with Jev's. CLI: `simulate.py --scorer glm [--llm-model ...]`; smoke:
`python llm_client.py --smoke`. The workbench **Compare** step scores Jev vs rules vs GLM on
the same frozen sample (paired AUCs with CIs + latency probe + $/call, verdict rule).

If the smoke test returns 404: `export JEV_URL=https://openrouter.ai/api/v1/systemone JEV_MODEL=jev-1.13`.

| Env | Default | |
|---|---|---|
| `JEV_MODE` | `auto` | `live` if a key is set, else `mock`; also `record`, `replay` |
| `JEV_URL` | `https://openrouter.ai/api/alpha/decisions` | |
| `JEV_MODEL` | `typesafe/jev-1.13` | |
| `JEV_FIXTURES` | `./fixtures` | record/replay directory |

## Files

`jev_client.py` client, retries, record/replay, contract validator · `mock_jev.py` heuristic stand-in and HTTP server ·
`questions.py` `WILL_BOOK`, `REVIEW`, `prefix_state` · `schema.py` closed vocabulary and call validator ·
`generate_synthetic.py` · `reduce_sgd.py` · `simulate.py` forecasts and reviews · `evaluate.py` metrics and trajectories ·
`baselines.py` constant / stage-reached / logistic-regression baselines · `run_evals.py` E1-E7.

## Differences from the tutorial

- `call_id` is **not** sent in `state`. The seed ids (`book-001`, `fail-002`) would leak the label.
- `usage.cost` is not in the TypeSafe response schema. Cost falls back to `input_tokens × $0.042/1M`, and the report says which source was used.
- `score` answers also carry `confidence` and `legend` (verified against the `typesafe-sdk` 0.7.2 schemas).
- `n_slots` → `n_results`; the review's `failure_mode` adds `info_only`, `declined_offer` and a leftover `other`.
- There is a **pre_outcome** forecast point: the last forecast before the call enters booking, wrap-up or completion. There is also an **attempted-booking** subset ("will this booking attempt land?"), which is harder than "will they book at all".
