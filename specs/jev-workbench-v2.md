# Spec: Jev workbench v2, an operator workflow for benchmarking call-analysis approaches

**Status:** APPROVED with defaults (§14). P1 in progress.
**Builds on:** [jev-call-analysis.md](jev-call-analysis.md) (v1 lab, evals E1–E7) and main @ `d93ce7c`
(three-way benchmark Jev vs rules vs GLM-5.3, PR #12 latency provenance).
**Lives in:** `jev-call-lab/` (still standalone; Streamlit stays the UI).

## 1. Goal

Turn the workbench from **eight tool tabs** into a **guided, resumable workflow**. It should answer one question
for a given corpus of calls:

> *Which approach (rules, Jev, GLM, or a cascade of them) should we use for **this kind of call** and
> **this job**? How sure are we, what will it cost at our volume, and how fast is it?*

There are three audiences, served by **one app with two views**:

| Persona | Needs | View |
|---|---|---|
| **Business user** (ops lead, product owner) | A plain-language recommendation per use case, cost at their volume, latency against budget, and a confidence level they can trust. No jargon. | **Summary** view |
| **Operator** (runs the benchmark) | A step-by-step flow: connect → corpus → plan → run → compare. Cost guardrails, resumable runs, clear failures, reproducible exports. | **Workflow** view (default) |
| **Analyst** | CIs, calibration, per-call drill-down, audit, raw JSON | **Technical details** toggle in every step |

## 2. Review of the current workbench (main @ `d93ce7c`)

### 2.1 Correctness issues (benchmark validity)

| # | Finding | Where | Impact |
|---|---|---|---|
| R1 | **Samples are the first N calls**, not random. `sgd_test.json` is grouped by service, so the Benchmark default (first 10) is **10 therapist calls, all unbooked**. The first 30 are too. | `app.py:246, 449` | AUC is **undefined** on the default benchmark. Any small-N run is biased by file order. |
| R2 | **Benchmark can compare different call sets.** "Reuse saved runs" filters Jev/GLM runs by dataset only, not by call ids, question set or limit. The verdict then states "LLM runs on {slider N} calls". | `app.py:493–539` | Unpaired comparison, reported as paired. |
| R3 | **Stale rule scores.** `bench_rules` lives in session state with no key. After the dataset or N changes, the old rule AUCs stay in the table next to new LLM runs. | `app.py:452–459` | Wrong comparison rows. |
| R4 | **No uncertainty anywhere in the UI.** The verdict picks "best" by point estimate. Offline bootstrap on main's fixtures shows several Jev−GLM gaps whose 95% CI includes 0. | `app.py:528–539` | The verdict line flips on noise. |
| R5 | **Loaded results lose their provenance.** Uploads are tagged `Config()` (mode `mock`). The dataset is guessed from the call-id prefix, so dev/train/uploaded results are labelled `synthetic_calls.json`. | `app.py:286–288` | Wrong baselines, wrong mode badge. |
| R6 | **One global mode for both scorers.** GLM inherits Jev's mode, so you can't replay Jev's committed fixtures while running GLM live on the same sample. | `app.py:86–97` | Forces paying twice, or skipping the comparison. |
| R7 | **Latency comes from the accuracy run, at up to 32-way concurrency.** p50 under load isn't the per-decision latency a live call sees, and concurrency isn't recorded. | `run_calls` | Misleading "fastest". |
| R8 | **Sign-off (E1–E7) covers Jev only.** GLM has no contract/no-leak sign-off path in the UI. | `app.py:544–579` | GLM results lack the same gate. |
| R9 | **Fixture keys are shared across calls with identical prefixes.** This is documented in AGENTS.md: last-write-wins under parallel record. | `jev_client.cache_key` | Small replay drift (≤ 0.03 p). Must show in run manifests. |

### 2.2 UX issues

- **No workflow.** Eight peer tabs (Run, Results, Call explorer, Analytics, Benchmark, Sign-off, Audit, Playground).
  Three of them run things (Run, Benchmark, Sign-off) and three compare things (Results, Analytics, Benchmark). Nothing
  says what to do first, and the same controls (dataset, N, workers, cap) are duplicated per tab with different defaults (20 vs 10 calls, $0.50 vs $2.00 caps).
- **The sidebar is overloaded with plumbing:** mode, key, endpoint, URL, model, fixtures dir, GLM model, GLM effort and two buttons.
  Fixture paths and URLs are operator details. They belong in an "Advanced" drawer, not in front of business users.
- **Runs vanish on refresh.** All state is `st.session_state`, with no on-disk run registry or resume.
- **Long runs block the page.** A run executes inside the script, so you can't browse results while GLM runs for minutes.
- **Jargon with no translation:** AUC, `pre_outcome_attempted`, Brier, noul, `visible_error_overweight`.
- **No upload of real calls.** Only results JSON can be uploaded. There is no way to bring your own transcripts or event logs.
- **No cost at volume.** Cost is shown per run ($ for N calls), not per 1,000 calls or per month at the business's volume.
- **No side-by-side call view.** You can't overlay Jev's and GLM's P(book) trajectories on the same call. That overlay is the most intuitive way to see how they differ.

### 2.3 Things to keep

The key is held only in session memory (`Config(repr=False)`). Contract validation, no-leak eval E2, latency provenance (PR #12),
replay fixtures, the audit's secret scan, the stable typed wire shape for both scorers and the
dataviz palette all stay unchanged.

## 3. Design principles

1. **A workflow, with one question per screen.** A stepper shows where you are, what's done and what's next. Each step ends in a clear primary button.
2. **Progressive disclosure.** Lead with a plain-language answer, keep the evidence one click away, and hide the plumbing in "Advanced".
3. **Honest by construction.** Comparisons are always **paired** (same calls, same questions). A winner is only declared when the CI says so; otherwise the screen says "tie" and breaks it on cost and latency.
4. **Guardrails before spend.** Show the estimate before every paid action, enforce a hard cap, confirm in a dialog, and reconcile the estimate with actual cost afterwards.
5. **Never lose work.** Runs persist to disk as they go, resume after a crash or refresh, and run in the background while you browse.
6. **Every number carries its provenance.** Mode (live/record/replay/mock), model version, sample id and latency source are badged next to the number.
7. **Accessible charts.** Use the existing palette rules, never color alone, keep a table view behind every chart, and lean on direct labels.

## 4. Information architecture

```
┌ Sidebar ─────────────────────┐  ┌ Main ───────────────────────────────────────────────────────────────┐
│ Project: clinic-q3   [▼]     │  │ ① Connect  ② Corpus  ③ Plan  ④ Run  ⑤ Compare  ⑥ Decide & export  │
│ View:  (•) Workflow ( ) Summary│ │ ✓ done     ✓ done    ● now   ○      ○          ○                    │
│ ☐ Show technical details     │  │─────────────────────────────────────────────────────────────────────│
│ ─────────────                │  │  step content                                                       │
│ Key: ●●●● set (session only) │  │                                                                     │
│ Background jobs: GLM 41/120  │  │                                    [Back]          [Primary action →]│
│ ▸ Advanced (URLs, fixtures…) │  └─────────────────────────────────────────────────────────────────────┘
└──────────────────────────────┘
```

- **Project.** A named workspace on disk (§10) that holds corpora, frozen samples, runs and decisions. It's created on first use and picked from the sidebar.
- **Stepper.** Six steps. A step is reachable once its inputs exist. Later steps show "needs: …" instead of an error.
- **Tools** (secondary navigation, technical details on): Call explorer, Audit, Sign-off (E1–E7, now for both scorers), Playground.
- Built with `st.navigation` (pages), `st.dialog` (confirmations), `st.fragment(run_every=…)` (job progress polling) and `st.data_editor` (labels and lexicon edits).

## 5. Screens

### ① Connect
- **Key** (password field, session memory only, as today) and **Test connection** for **each** approach: Jev and GLM. Each row shows model version, round-trip ms and contract OK.
- **Source per approach** (fixes R6): `live` · `record` · `replay` · `mock`, chosen per approach. Example: "Jev: replay committed fixtures, GLM: record".
- "Advanced": URLs, model slugs, GLM reasoning effort, fixtures directory, concurrency limits.
- With no key, everything works in replay/mock, and a banner says which numbers are not live.

### ② Corpus
Pick a **built-in corpus** (synthetic phone calls, SGD test/dev/train) or **upload your own** (§6).
After ingest, a **Corpus profile card** shows:

| Field | Example |
|---|---|
| Calls / labelled / booked rate | 412 / 412 / 47% |
| Avg events per call, tool-log coverage | 11.3, 92% |
| Calls that attempted a booking | 63% |
| Vocabulary coverage (events mapped / "other") | 96% / 4% |
| **Structure difficulty** = B1 "furthest stage reached" AUC before outcome | 0.97 → *"Outcome is almost fully visible from structure. LLMs are unlikely to add accuracy here; compare on cost."* |
| PII redaction report (uploaded text only) | 37 phones, 12 emails redacted |

The difficulty line puts the main lesson from v1 in front of the user: on SGD a free rule already gets 0.965,
while on phone-style calls the hard question ("will this attempt land?") is where LLMs can earn their cost.

### ③ Plan
- **Use cases** (multi-select, each with plain-language help and editable thresholds):

  | Use case | What matters | Default constraints |
  |---|---|---|
  | **In-call escalation** (decide during the call) | Ranking accuracy before the outcome; latency | p95 ≤ **800 ms** per decision |
  | **Post-call QA review** | Failure-mode accuracy, needs-human agreement | none on latency |
  | **Bulk analytics / forecasting** | Mid-call ranking, calibration, cost per 1k calls | budget per 1k calls |

- **Approaches:** Rules (B1, needs no training; B2 only if a labelled training split exists) · Jev · GLM (model + effort) ·
  **Cascade** (Jev first, GLM only when Jev is uncertain, 0.35 ≤ p ≤ 0.65; simulated offline from the two paired runs, so it needs **no extra calls**).
- **Sample** (fixes R1/R2): seeded, **stratified** by label × source × specialty. Its size comes from a precision target:
  "±0.05 AUC needs about 300 calls" (live v1 evidence: n = 153 gave ±0.06), shown as a slider with the CI-width estimate next to it.
  The sample is **frozen** as `sample.json` (call ids + seed + corpus hash). Every approach runs on exactly this set.
- **Estimate panel** per approach: requests, tokens, **$ for this sample**, and **$ per 1k calls**. A hard cap applies, with a confirm dialog if live.
  After the run, the panel shows "estimate vs actual".

### ④ Run
- One **job per approach** in a **background worker** (thread with in-memory `Config`; the key never goes to argv, env or disk).
  Results are appended per call to `runs/<id>/results.jsonl`, so the job is **resumable**: on restart, finished calls are skipped.
- A job card shows a progress bar, calls/s, $ so far vs cap (auto-pauses at the cap), failures with retry, and Pause/Resume/Cancel controls.
- **Latency probe** (fixes R7): a separate small job per approach that sends 50 sequential decisions at concurrency 1 after 5 warm-ups.
  It reports p50/p95/p99 with `latency_source=measured` and the concurrency recorded. Throughput at concurrency *c* is reported separately.
- **Repeatability** (optional, live only): re-score 30 sampled calls a second time. Report test-retest mean |Δp| and the AUC delta.

### ⑤ Compare
**Summary view (business).** The numbers in this mock-up are illustrative only; no GLM latency has been measured yet:

```
┌ For IN-CALL ESCALATION on "clinic-q3 uploaded calls" (412 calls, 300 sampled) ─────────────────┐
│ ✅ Recommended: Jev                                                                              │
│   • As accurate as GLM-5.3 (difference +0.02, within noise)                                       │
│   • 11× cheaper: $0.43 vs $4.70 per 1,000 calls  →  $4 vs $47 per month at 10k calls             │
│   • Meets the 800 ms budget (p95 505 ms); GLM does not (p95 3.9 s)                                │
│   Confidence: medium (300 calls; ±0.05)            [Why?]  [See the evidence]                     │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

- **Accuracy vs cost** scatter: x = $ per 1k calls (log), y = ranking accuracy with CI whiskers, one point per approach,
  direct labels, and the Pareto frontier drawn. It's the single most useful chart for choosing an approach.
- **Latency vs budget**: horizontal bars (p50 with a p95 whisker) against a dashed budget line. Approaches over budget are badged ⛔ with an icon and a label, never color alone.
- **Cost at volume**: monthly-calls input (default 10k) → $/month per approach, as a table plus a bar chart.
- Plain-language glossary on hover: "ranking accuracy = how often a call that booked scores above one that didn't (0.5 = coin flip)".

**Operator / analyst view (technical details on):**
- Paired metrics table: AUC at mid / pre-outcome / attempted / end with 95% CIs, Brier, ECE, failure-mode macro-F1 against the reference
  (or against human labels when present), contract-error and repair rates, tokens, $/call, and latency from the probe.
- Paired-difference forest plot (each approach minus the best, with CIs).
- Reliability diagram (calibration) per approach.
- **Call overlay**: pick a call → Jev, GLM and rule P(book) trajectories on one chart, failed events marked, review answers side by side.
- "Disagreements" list: the calls where the approaches disagree most. These are useful for spot-checks and labelling.

### ⑥ Decide & export
- Record the decision per use case (approach, thresholds, operator note).
- Export a **benchmark report** (Markdown + JSON), a **run manifest** (§10), and a **fixture pack** (one compressed JSONL per run, for reproducible replays).
- Sign-off gate: E1–E7 plus the W-evals (§12) for **every approach used**. The report states which numbers were live, replayed or mock.

## 6. Bringing your own corpus

### 6.1 Accepted inputs (templates downloadable from the Corpus step)

| Format | Columns / shape | Path |
|---|---|---|
| **Structured traces** (JSON) | the v1 call schema (`events[]` with actor, stage, event, ok, t_ms, tool…) | validated with `schema.validate_call`, used as is |
| **Transcripts** (CSV or JSONL) | `call_id, turn, speaker, text, start_ms, end_ms` | reduced to events (§6.2) |
| **Tool logs** (optional, CSV/JSONL) | `call_id, t_ms, tool, ok, n_results` | merged as `tool` events |
| **Labels** (optional, CSV) | `call_id, booked` (+ optional `failure_mode`, `needs_human`) | ground truth |

The **column mapper** is a `st.data_editor` that maps the user's column names onto these. A preview of the first five calls
shows the transcript and the reduced events side by side.

### 6.2 Reducer: transcript → structure

- **Rules reducer (default, offline, free).** Speaker → actor. A gap of more than 4 s → `caller_went_silent`. An **editable keyword lexicon** (YAML,
  per corpus) maps phrases to events ("date of birth" → `asked_dob_and_name`, …). Tool logs → tool events. A stage machine assigns stages.
  Utterances that match nothing become `other_caller` / `other_agent`, two new vocabulary entries whose coverage the profile reports.
- **LLM reducer (opt-in, phase 4).** GLM classifies each utterance into the closed vocabulary. The cost is estimated first.
  **Text leaves the machine**, so it's off by default, needs explicit consent per corpus, and runs only after PII redaction.
- **Reducer quality check.** The operator labels 20 random utterances with the correct event, and the UI shows reducer accuracy.
  Below 80%, the Corpus step warns that results reflect the reducer as much as the scorer.
- **Privacy.** Regex redaction (phones, emails, dates, long digit runs) runs before anything is stored. Raw text is stored only if the user ticks "keep originals".
  Jev and the structure-only GLM approach **never receive text**. E2 is extended to uploaded corpora: the vocabulary scan uses the corpus's own utterance words.

### 6.3 Labels
- From a label column, or derived from tool logs (a successful `create_appointment`), or via **label-a-sample**:
  the operator marks booked yes/no on 100 stratified calls in a fast keyboard-driven table.
- **Unlabelled corpus:** accuracy metrics are disabled, with an explanation. The Compare step shows **agreement between approaches**,
  cost and latency only, plus a call to action to label at least 100 calls.

## 7. Benchmark methodology (best practices, enforced by the engine)

1. **Paired design.** Every approach scores the same frozen sample with the same states and the same question set. The engine refuses
   to compare runs whose `sample_id`, `questions_hash` or `corpus_hash` differ (fixes R2).
2. **Held-out evaluation.** Rules train only on a train split that's disjoint from the sample, or use k-fold when there's no split, and say so.
   Nothing is tuned on the sample.
3. **Uncertainty.** A **cluster bootstrap over calls**, never turns (turns within a call are correlated): 1,000 resamples, 95% percentile CIs
   for every metric and every paired difference. A CI wider than ±0.10 triggers an "underpowered" badge.
4. **Verdict rule.** For each use case, approaches that break a hard constraint (latency budget, $ cap) are dropped. Among the rest,
   an approach wins if its paired difference against the runner-up on the primary metric has a CI that excludes 0. Otherwise it's a
   **tie**, broken by $ per 1k calls, then latency p95. The verdict text always states which of those decided it.
5. **Metrics per job:**
   - **Discrimination:** AUC at mid, pre-outcome and attempted-booking.
   - **Calibration:** Brier, ECE (10 bins), reliability diagram.
   - **Decision quality:** precision and recall at the operator's escalation threshold.
   - **Classification:** failure-mode macro-F1.
   - **Reliability:** contract validity, GLM repair rate, failure rate.
   - **Cost:** actual `usage.cost` (estimate vs actual), $ per 1k calls.
   - **Latency:** probe p50/p95/p99 at concurrency 1, plus throughput.
6. **Latency protocol.** Only probe latency drives verdicts. Accuracy-run latency is shown with its concurrency and labelled "under load".
7. **Repeatability.** Optional test-retest on 30 calls. Live endpoints are stochastic, and the spread is reported, not hidden.
8. **Reproducibility.** Every run writes a manifest: corpus hash, sample ids and seed, questions hash, prompt hash (GLM's prompt
   differs from Jev's request; the hash makes that explicit), model slug **and resolved version** (e.g. `typesafe/jev-1.13-20260917`),
   mode per approach, concurrency, git SHA, start/end time. Recording uses per-(call, turn) fixture keys so shared prefixes can't collide (R9).
9. **Robustness (phase 4).** Re-score with event names paraphrased and fields shuffled. The AUC drop shows how much a model relies on surface tokens.
10. **Fair framing.** Both LLMs get the same `state` JSON and the same typed questions. GLM's system prompt is versioned and shown in technical details.

## 8. Approach guide per corpus type (seeded from v1 evidence, recomputed per project)

| Corpus type | v1 evidence | Starting recommendation (to be confirmed by a run) |
|---|---|---|
| **Structured logs where the outcome shows in the structure** (SGD-like, labelled) | B1 rule 0.965 pre-outcome; Jev 0.755, GLM 0.650 | **Rules**: free and best. Use an LLM only for post-call QA labels, where GLM agrees with the reference on 0.92 of calls vs Jev's 0.63. |
| **Phone calls with silence, hangups and identity failures** (synthetic-like) | Attempted-booking subset: GLM 0.739, Jev 0.715, rules 0.698 (a tie within CI) | **Jev for in-call** (cheaper, meets latency); **GLM or a cascade for post-call QA**. |
| **Unlabelled transcripts** | n/a | Label 100 calls first; meanwhile compare agreement, cost and latency only. |
| **High volume, tight budget** | Per 1,000 calls: Jev $0.27 (synthetic) / $0.53 (SGD) vs GLM $3.23 / $5.54 | **Jev or rules**. Simulate the cascade to see whether GLM on the uncertain ~20% pays off. |

The guide is shown on the Corpus step as a hypothesis. The Compare step replaces it with the project's own measured verdict.

## 9. Cost and latency UX details

- Every paid button shows "~$X (cap $Y)". Live runs over $1 need a confirm dialog that states the number of requests.
- A running total is shown per job and per project. The job auto-pauses at the cap.
- "Estimate vs actual" is shown after each run, so estimators improve. For GLM, completion-token assumptions are recalibrated from actual usage.
- Volume projection: `$ per 1k calls × monthly volume`. Latency is shown against the in-call budget, with p95 as the default comparison.

## 10. Persistence and data model

```
jev-call-lab/workspace/<project>/          (gitignored)
  project.json                    name, created, decisions[]
  corpora/<corpus_id>/            calls.json, profile.json, lexicon.yaml, mapping.json, labels.csv, [originals/ if kept]
  samples/<sample_id>.json        corpus_id, seed, strata, call_ids[], corpus_hash
  runs/<run_id>/manifest.json     §7.8 (never the key)
  runs/<run_id>/results.jsonl     one line per call, appended (resumable)
  runs/<run_id>/fixtures.jsonl.gz optional fixture pack (per (call, turn) keys)
  benchmarks/<bench_id>/          scorecard.json, report.md
```

- Key hygiene is unchanged and extended. The audit's secret scan runs over the whole workspace after each run (W10).
- Committed repo fixtures (10.5k files) move to per-run packs in a later cleanup. Replay still reads both formats.

## 11. Code structure

Split UI from logic so each logic module is testable without Streamlit, and the CLI stays at parity:

| Module | Responsibility |
|---|---|
| `workspace.py` | projects, corpora, samples, run registry, manifests, resume |
| `ingest.py` | format detection, column mapping, PII redaction, rules reducer, label import, corpus profile |
| `sampling.py` | stratified, seeded, frozen samples, plus the size ↔ CI-width estimate |
| `stats.py` | cluster bootstrap, paired differences, ECE, reliability bins, precision/recall at a threshold |
| `benchmark.py` | paired-run validation, metrics per use case, cascade simulation, verdict rule, scorecard |
| `jobs.py` | background job runner with pause/resume/cap and progress |
| `app.py` → `ui/` | pages: `connect.py`, `corpus.py`, `plan.py`, `run.py`, `compare.py`, `decide.py`, `tools/*.py` |
| existing | `jev_client.py`, `llm_client.py`, `simulate.py`, `evaluate.py`, `baselines.py`, `audit.py` reused as is |

CLI parity: `python benchmark.py --project clinic-q3 --sample s1 --approaches jev,glm,rules` produces the same scorecard as the UI.

## 12. Evals for v2 (written before the build)

| ID | Eval | Gate |
|---|---|---|
| W1 | **Paired integrity:** the engine rejects comparisons across different sample, question or corpus hashes | 100% |
| W2 | **No stale state:** changing the corpus or sample invalidates rule scores and scorecards (regression test for R3) | pass |
| W3 | **Sampling:** same seed gives the same ids; stratum proportions within ±1 call; never a single-class sample when both classes exist (R1) | pass |
| W4 | **Bootstrap:** resamples calls, not turns; CI covers a known AUC in ≥ 93% of 200 simulated trials | pass |
| W5 | **Verdict rule:** table-driven cases (clear win, tie → cost, tie → latency, constraint violation, underpowered) | 100% |
| W6 | **Ingest:** CSV/JSONL transcripts → schema-valid calls; seeded phones and emails redacted; extended E2 finds no transcript word in any structure state | 100% |
| W7 | **Unlabelled corpus:** accuracy metrics disabled; agreement, cost and latency still computed | pass |
| W8 | **Latency:** only `measured`/`recorded` sources feed verdicts; probe concurrency = 1 is recorded | pass |
| W9 | **Resume:** kill a run mid-way and resume → no duplicates, same final id set, cost not double-counted | pass |
| W10 | **Key hygiene:** after a live run against the local mock server, the workspace scan finds no key in manifests, results, fixtures or logs | 0 hits |
| W11 | **Cascade math:** cost and AUC computed from two paired runs match a hand-computed example | exact |
| W12 | **UI smoke (AppTest):** the full mock-mode happy path through steps ①–⑥, Summary and Workflow views | no exceptions |
| W13 | **Provenance:** uploaded results keep their mode, dataset and model from the manifest (R5) | pass |

E1–E7 from v1 stay and now run per approach (R8).

## 13. Phasing

| Phase | Scope | Exit |
|---|---|---|
| **P1: foundation + fixes** | workspace and manifests, frozen stratified samples, paired benchmark engine with CIs and the verdict rule, per-approach source modes, stepper UI on built-in corpora, Summary/Workflow views, fixes for R1–R8 | W1–W5, W8, W12, W13 |
| **P2: bring your own corpus** | upload of structured traces and transcripts (CSV/JSONL), rules reducer with editable lexicon, PII redaction, labels and label-a-sample, corpus profile and difficulty | W6, W7 |
| **P3: operations** | background jobs with pause/resume/cap, latency probe, repeatability, cascade simulation, volume cost projection, export and fixture packs | W9–W11 |
| **P4: extensions** | opt-in LLM reducer, robustness checks, a "text-aware" approach (GLM on redacted transcript) clearly separated from structure-only | new evals |

## 14. Decisions (user accepted all defaults, 2026-10-04)

1. **In-call latency budget:** default **800 ms p95 per decision**. What's the real budget for your receptionist?
2. **Sending transcript text to GLM** (LLM reducer, text-aware approach): default **off**, opt-in per corpus after redaction. Is that acceptable for your data?
3. **Deployment:** default a **local single-user** app with an on-disk workspace. Shared deployment with auth would be a separate spec.
4. **Volume for cost projection:** default **10,000 calls/month**.
5. **First real corpus to target:** P1 uses the built-in corpora; your own call logs arrive with upload in P2 (the Kaggle booking-calls set is blocked in this sandbox).
6. **Fixture cleanup:** OK to move the 10.5k committed fixture files into per-run compressed packs in P3?

## 15. Sign-off checklist

- [x] User: review the findings (§2) and approve the workflow (§4–§5)
- [x] User: answer or accept the defaults in §14
- [ ] Build P1 → W1–W5, W8, W12, W13 pass; E1–E7 still pass
- [ ] Build P2 → W6, W7 on an uploaded sample corpus
- [ ] Build P3 → W9–W11; a live benchmark with both approaches on one frozen sample; the report exported
