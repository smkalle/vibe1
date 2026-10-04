"""Jev workbench v2: a guided benchmark workflow (spec: specs/jev-workbench-v2.md).

    streamlit run app.py

Steps: ① Connect → ② Corpus → ③ Plan → ④ Run → ⑤ Compare → ⑥ Decide & export.
Views: Workflow (operators) and Summary (business). The OpenRouter key lives only in this
browser session's memory: never on disk, in manifests, results, fixtures or logs.
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

import audit
import benchmark
import jev_client
import llm_client
import sampling
import ui_charts
import ui_tools
import workspace
from jev_client import JevError
from questions import WILL_BOOK

HERE = Path(__file__).parent
STEPS = ["① Connect", "② Corpus", "③ Plan", "④ Run", "⑤ Compare", "⑥ Decide & export"]
TOOLS = "🔧 Tools"
SOURCES = {
    "replay": "Replay: answers recorded from real live runs (free, offline)",
    "live": "Live: call OpenRouter with your key",
    "record": "Record: live, and save answers for free replays later",
    "mock": "Mock: built-in stand-in for testing the workflow (not a real model)",
}
LLM = ("jev", "glm")
GLOSSARY = {
    "ranking accuracy": "How often a call that booked gets a higher score than one that didn't (AUC). 0.5 = coin flip, 1.0 = perfect.",
    "before the outcome": "Scored just before the call enters booking or wrap-up, when the result is not yet visible.",
    "95% CI": "The range the true value most likely falls in, given the sample size. Overlapping ranges mean 'too close to call'.",
    "p95 latency": "95% of decisions are faster than this.",
}

st.set_page_config(page_title="Jev Benchmark Workbench", page_icon="📞", layout="wide")
ss = st.session_state
for k, v in {"step": STEPS[0], "view": "Workflow", "tech": False, "project": "default", "jev_mode": "replay",
             "glm_mode": "replay", "corpus_id": "synthetic", "use_cases": ["in_call", "post_call_qa", "bulk"],
             "approaches": ["rules_b1", "rules_b2", "jev", "glm"], "latency_budget": 800, "threshold": 0.35,
             "volume": 10_000, "cap": 2.0, "workers": 8, "jev_url": jev_client.DEFAULT_URL,
             "jev_model": jev_client.DEFAULT_MODEL, "glm_url": llm_client.DEFAULT_URL,
             "glm_model": llm_client.DEFAULT_MODEL, "glm_effort": llm_client.DEFAULT_EFFORT,
             "fixtures": str(HERE / "fixtures"), "api_key": "", "plan_n": 300, "plan_seed": 7}.items():
    ss.setdefault(k, v)
# Streamlit drops a widget's state when the widget is not rendered (each lives on one step).
# Re-assigning keeps settings, the key and the step across navigation.
for _k in ("step", "view", "tech", "jev_mode", "glm_mode", "corpus_id", "use_cases", "approaches", "latency_budget",
           "threshold", "volume", "cap", "workers", "jev_url", "jev_model", "glm_url", "glm_model", "glm_effort",
           "fixtures", "api_key", "plan_n", "plan_seed"):
    ss[_k] = ss[_k]
workspace.Project(ss.project).ensure()


# ---------------------------------------------------------------- helpers
def jev_cfg() -> jev_client.Config:
    return jev_client.Config(mode=ss.jev_mode, api_key=ss.api_key or None, url=ss.jev_url, model=ss.jev_model,
                             fixtures=Path(ss.fixtures))


def glm_cfg() -> llm_client.Config:
    return llm_client.Config(mode=ss.glm_mode, api_key=ss.api_key or None, url=ss.glm_url, model=ss.glm_model,
                             effort=ss.glm_effort, fixtures=Path(ss.fixtures))


def cfg_for(a):
    return jev_cfg() if a == "jev" else glm_cfg()


def needs_key(a) -> bool:
    return ss[f"{a}_mode"] in {"live", "record"} and not ss.api_key


def project() -> workspace.Project:
    return workspace.Project(ss.project).ensure()


def current_sample():
    """The selected sample, else the latest frozen sample for this corpus (so a refresh never loses the plan)."""
    sid = ss.get("sample_id")
    if sid:
        try:
            s = project().sample(sid)
            if s["corpus_id"] == ss.corpus_id:
                return s
        except FileNotFoundError:
            pass
    mine = [s for s in project().samples() if s["corpus_id"] == ss.corpus_id]
    if mine:
        ss.sample_id = mine[-1]["sample_id"]
        return mine[-1]
    return None


@st.cache_data(show_spinner=False)
def corpus(corpus_id):
    return workspace.load_corpus(corpus_id)


@st.cache_data(show_spinner="Profiling the corpus…")
def corpus_profile(corpus_id):
    return benchmark.profile(workspace.load_corpus(corpus_id))


def get_scorecard(sample, approaches):
    """Rules are scored on the fly for this exact sample, and the cache key holds the sample and run ids (W2)."""
    ents = benchmark.load_entries(project(), sample, approaches)
    key = (sample["sample_id"], tuple(sorted((a, (e["manifest"] or {}).get("run_id"),
                                              (e["manifest"] or {}).get("n_done")) for a, e in ents.items())),
           ss.latency_budget, ss.threshold)
    cache = ss.setdefault("sc_cache", {})
    if key not in cache:
        ucs = {k: {**v, "latency_budget_ms": ss.latency_budget if v["latency_budget_ms"] else None}
               for k, v in benchmark.USE_CASES.items()}
        cache.clear()
        try:
            cache[key] = ("ok", benchmark.scorecard(sample, ents, use_cases=ucs, threshold=ss.threshold), ents)
        except benchmark.PairingError as e:
            cache[key] = ("error", str(e), ents)
    return cache[key]


def step_status():
    s = current_sample()
    llm = [a for a in ss.approaches if a in LLM]
    runs_done = bool(s) and all((project().latest_run(s["sample_id"], a) or {}).get("status") == "done" for a in llm)
    return {STEPS[0]: all(not needs_key(a) for a in llm), STEPS[1]: bool(ss.corpus_id), STEPS[2]: bool(s),
            STEPS[3]: bool(s) and runs_done, STEPS[4]: bool(s) and runs_done,
            STEPS[5]: bool(project().meta()["decisions"])}


def go(step):
    ss.step = step


def nav(prev=None, nxt=None, label="Next"):
    c1, _, c3 = st.columns([1, 4, 1])
    if prev:
        c1.button("← Back", on_click=go, args=(prev,), key=f"back_{prev}")
    if nxt:
        c3.button(f"{label} →", type="primary", on_click=go, args=(nxt,), key=f"next_{nxt}")


def md(text: str) -> str:
    """Streamlit Markdown treats $...$ as math; escape dollars in prose."""
    return text.replace("$", "\\$")


def badge(mode):
    return {"live": "🟢 live", "record": "🟢 live (recording)", "replay": "🔵 replay of live runs",
            "mock": "⚪ mock (not a real model)", "offline": "⚫ offline rule", "unknown": "❔ unknown"}.get(mode, mode)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.subheader("Project")
    existing = workspace.list_projects()
    options = existing + ["➕ New project…"]
    choice = st.selectbox("Open project", options, index=existing.index(ss.project) if ss.project in existing else 0,
                          key="project_pick")
    if choice == "➕ New project…":
        name = st.text_input("Name", placeholder="clinic-q3")
        if st.button("Create project", disabled=not name):
            ss.project = workspace.Project(name).ensure().name
            ss.pop("sample_id", None)
            st.rerun()
    elif choice != ss.project:
        ss.project = choice
        ss.pop("sample_id", None)
    st.radio("View", ["Workflow", "Summary"], key="view", horizontal=True,
             help="Workflow: step-by-step for operators. Summary: the recommendation for business readers.")
    st.toggle("Show technical details", key="tech")
    st.divider()
    st.caption("🔑 Key: " + ("set for this session only" if ss.api_key else "not set (replay/mock only)"))
    for a in LLM:
        st.caption(f"{benchmark.APPROACHES[a]['label']}: {badge(ss[f'{a}_mode'])}")
    with st.expander("Advanced"):
        st.text_input("Jev URL", key="jev_url")
        st.text_input("Jev model", key="jev_model")
        st.text_input("GLM URL", key="glm_url")
        st.text_input("GLM model", key="glm_model")
        st.selectbox("GLM reasoning effort", ["low", "high", "max"], key="glm_effort")
        st.text_input("Fixtures directory", key="fixtures")
        st.slider("Parallel requests", 1, 32, key="workers")


# ---------------------------------------------------------------- summary (shared by both views)
def verdict_card(v, sc):
    icon = {"high": "✅", "medium": "✅", "low": "⚠️", "none": "❔"}[v["confidence"]]
    with st.container(border=True):
        st.markdown(f"**{v['label']}** · {benchmark.USE_CASES[v['use_case']]['question']}")
        if v["winner"]:
            st.markdown(f"### {icon} {sc['approaches'][v['winner']]['label']}")
        st.markdown(md(v["sentence"]))
        conf = {"high": "High: the winner is clearly ahead.", "medium": "Medium: a tie on accuracy, decided on cost or speed.",
                "low": "Low: too few calls for a firm answer. Use a larger sample.", "none": "No data."}[v["confidence"]]
        st.caption(f"Confidence: {conf}")
        for n, why in v["excluded"].items():
            st.caption(f"{'⛔' if 'budget' in why else '–'} {sc['approaches'][n]['label']}: "
                       f"{'not applicable (' + why + ')' if 'does not produce' in why else why}")
        if v["latency_unverified"]:
            st.caption("⏱ Latency not measured yet for " + ", ".join(sc["approaches"][n]["label"] for n in v["latency_unverified"])
                       + ". Run the latency probe in step ④ with a live source.")


def summary_block(sc, sample):
    st.caption(f"Corpus **{workspace.BUILTIN_CORPORA[sample['corpus_id']]['label']}** · sample of {sc['n_calls']} calls "
               f"({sample['n_booked']} booked) · " + " · ".join(f"{a['label']}: {badge(a['mode'])}" for a in sc["approaches"].values()))
    cols = st.columns(len(ss.use_cases) or 1)
    for col, uc in zip(cols, ss.use_cases):
        with col:
            verdict_card(sc["verdicts"][uc], sc)
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**Accuracy vs cost** · higher and further left is better; whiskers are 95% CIs; "
                    "dashed line = best value for money")
        metric = st.selectbox("Accuracy measure", ["auc_pre_outcome", "auc_mid", "auc_attempted"], key="avc_metric",
                              format_func={"auc_pre_outcome": "Ranking accuracy before the outcome",
                                           "auc_mid": "Ranking accuracy mid-call",
                                           "auc_attempted": "Will a booking attempt land?"}.get)
        ch = ui_charts.accuracy_vs_cost(sc, metric, "Ranking accuracy")
        if ch is not None:
            st.altair_chart(ch, width="stretch")
            st.dataframe(pd.DataFrame([{"approach": a["label"], "accuracy": a["metrics"][metric]["point"],
                                        "95% CI": "–" if a["metrics"][metric]["ci"][0] is None else
                                        f"{a['metrics'][metric]['ci'][0]:.2f} – {a['metrics'][metric]['ci'][1]:.2f}",
                                        "USD per 1,000 calls": a["cost_per_1k"]}
                                       for a in sc["approaches"].values() if a["metrics"][metric]["point"] is not None]),
                         hide_index=True, width="stretch",
                         column_config={"accuracy": st.column_config.NumberColumn(format="%.3f"),
                                        "USD per 1,000 calls": st.column_config.NumberColumn(format="$%.3f")})
        else:
            st.info("No accuracy data for this measure on this sample.")
    with c2:
        st.markdown(f"**Latency per decision** vs the {ss.latency_budget} ms in-call budget (red dashed line)")
        ch = ui_charts.latency_vs_budget(sc, ss.latency_budget)
        if ch is not None:
            st.altair_chart(ch, width="stretch")
        else:
            st.info("No measured latency yet for Jev or GLM. Replays of older recordings and mock runs carry no network "
                    "time: run the latency probe (step ④) with a live source.")
        if any(a["latency"].get("source") == "offline" for a in sc["approaches"].values()):
            st.caption("Rules run offline, so they add no network latency.")
        st.number_input("Calls per month", 100, 10_000_000, step=1000, key="volume")
        ch, _ = ui_charts.cost_at_volume(sc, ss.volume)
        if ch is not None:
            st.altair_chart(ch, width="stretch")
    with st.expander("What do these terms mean?"):
        for k, v in GLOSSARY.items():
            st.markdown(f"**{k}**: {v}")


def summary_view():
    st.title("Which approach should we use?")
    s = current_sample()
    if not s:
        samples = project().samples()
        s = samples[-1] if samples else None
        if s:
            ss.corpus_id, ss.sample_id = s["corpus_id"], s["sample_id"]
    if not s:
        st.info("No benchmark in this project yet. Switch to the **Workflow** view to run one.")
        return
    status, sc, _ = get_scorecard(s, ss.approaches)
    if status == "error":
        st.error(sc)
        return
    summary_block(sc, s)
    decisions = project().meta()["decisions"]
    if decisions:
        st.subheader("Decisions recorded")
        st.dataframe(pd.DataFrame([{k: d[k] for k in ("use_case", "approach", "note", "at")} for d in decisions]),
                     hide_index=True, width="stretch")


# ---------------------------------------------------------------- steps
def step_connect():
    st.header("① Connect")
    st.write("Choose where each approach's answers come from. **Replay** uses answers recorded from real live runs "
             "(Synthetic and SGD test corpora), so you can compare without a key or spend.")
    k1, k2 = st.columns([5, 1])
    k1.text_input("OpenRouter API key (needed for live / record)", type="password", key="api_key", placeholder="sk-or-v1-…",
                  help="Held only in this browser session's memory. Never written to disk, results, manifests or logs.")
    k2.button("Forget key", on_click=lambda: ss.update(api_key=""), key="forget_key")
    cols = st.columns(2)
    for col, a in zip(cols, LLM):
        with col, st.container(border=True):
            st.markdown(f"**{benchmark.APPROACHES[a]['label']}** · {benchmark.APPROACHES[a]['about']}")
            st.selectbox("Source", list(SOURCES), format_func=SOURCES.get, key=f"{a}_mode")
            if needs_key(a):
                st.warning("This source needs the key above.")
            if st.button("Test", key=f"test_{a}", disabled=needs_key(a)):
                try:
                    probe_state = {"current_event": "identity_check_failed", "current_stage": "identity", "trace_so_far": []}
                    r = benchmark.decide_fn_for(a, cfg_for(a))(probe_state, WILL_BOOK)
                    errs = jev_client.validate_response(WILL_BOOK, r)
                    (st.success if not errs else st.warning)(
                        f"{r.get('model')} · {r['_latency_ms']:.0f} ms ({r.get('_latency_source')}) · "
                        f"contract {'OK' if not errs else errs}")
                except JevError as e:
                    msg = str(e)
                    st.error(msg + (" Replay only covers answers that were recorded; use mock or live for other states."
                                    if "replay miss" in msg else ""))
    st.caption("Rules (furthest stage, logistic regression) need no connection: they run offline and free.")
    nav(nxt=STEPS[1])


def step_corpus():
    st.header("② Corpus")
    st.radio("Calls to benchmark on", list(workspace.BUILTIN_CORPORA), key="corpus_id", horizontal=True,
             format_func=lambda c: workspace.BUILTIN_CORPORA[c]["label"])
    st.caption(workspace.BUILTIN_CORPORA[ss.corpus_id]["about"])
    p = corpus_profile(ss.corpus_id)
    k = st.columns(5)
    k[0].metric("Calls", f"{p['calls']:,}")
    k[1].metric("Booked", f"{p['booked_rate']:.0%}" if p["booked_rate"] is not None else "–")
    k[2].metric("Events per call", f"{p['avg_events']:.1f}")
    k[3].metric("Calls with tool logs", f"{p['tool_coverage']:.0%}")
    k[4].metric("Tried to book", f"{p['attempted_rate']:.0%}")
    with st.container(border=True):
        d = p["difficulty_auc"]
        st.markdown("**How readable is the outcome from structure alone?**" +
                    (f" A free rule scores **{d:.2f}** ranking accuracy before the outcome." if d is not None else ""))
        st.write(p["difficulty_message"])
    st.info("📤 Upload your own transcripts or event logs: coming in phase 2.", icon="ℹ️")
    nav(STEPS[0], STEPS[2])


def step_plan():
    st.header("③ Plan")
    calls = corpus(ss.corpus_id)
    c1, c2 = st.columns(2)
    with c1:
        st.multiselect("What will the recommendation be used for?", list(benchmark.USE_CASES), key="use_cases",
                       format_func=lambda u: benchmark.USE_CASES[u]["label"])
        for u in ss.use_cases:
            st.caption(f"• **{benchmark.USE_CASES[u]['label']}**: {benchmark.USE_CASES[u]['question']}")
        st.number_input("In-call latency budget, p95 (ms)", 50, 10_000, step=50, key="latency_budget")
        if ss.tech:
            st.slider("Escalation threshold (flag 'won't book' below)", 0.05, 0.95, step=0.05, key="threshold")
    with c2:
        st.multiselect("Approaches to compare", list(benchmark.APPROACHES), key="approaches",
                       format_func=lambda a: benchmark.APPROACHES[a]["label"])
        for a in ss.approaches:
            st.caption(f"• **{benchmark.APPROACHES[a]['label']}**: {benchmark.APPROACHES[a]['about']}")

    st.subheader("Sample")
    booked = sum(c["label_booked"] for c in calls) / len(calls)
    self_trained = workspace.BUILTIN_CORPORA[ss.corpus_id]["train"] == workspace.BUILTIN_CORPORA[ss.corpus_id]["file"]
    if ss.get("_plan_corpus") != ss.corpus_id:  # new corpus: recommend a size that leaves rules something to train on
        ss._plan_corpus = ss.corpus_id
        ss.plan_n = min(300, int(len(calls) * 0.8) if self_trained else len(calls))
    ss.plan_n = max(min(20, len(calls)), min(ss.plan_n, len(calls)))
    n = st.slider("Calls in the sample", min(20, len(calls)), len(calls), key="plan_n")
    if self_trained and n == len(calls):
        st.caption("ℹ️ With the whole corpus in the sample, the logistic-regression rule has no calls left to train on.")
    hw = sampling.auc_halfwidth(n, booked)
    if hw:
        st.caption(f"Expected precision: about **±{hw:.2f}** ranking accuracy (95%). ±0.05 needs about 300 calls.")
    seed = st.number_input("Random seed", 0, 10_000, key="plan_seed") if ss.tech else ss.plan_seed
    existing = [s for s in project().samples() if s["corpus_id"] == ss.corpus_id]
    b1, b2 = st.columns([1, 2])
    if b1.button("Freeze sample", type="primary"):
        s = project().save_sample(sampling.freeze(ss.corpus_id, calls, n, int(seed)))
        ss.sample_id = s["sample_id"]
        st.rerun()
    if existing:
        ids = [s["sample_id"] for s in existing]
        cur = ss.get("sample_id") if ss.get("sample_id") in ids else ids[-1]
        ss.sample_id = b2.selectbox("…or use a frozen sample", ids, index=ids.index(cur))
    s = current_sample()
    if s:
        st.success(f"Sample **{s['sample_id']}**: {s['n']} calls, {s['n_booked']} booked, stratified by outcome, "
                   f"source and specialty. Every approach scores exactly these calls.")
        st.subheader("Estimated cost for this sample")
        by = {c["call_id"]: c for c in calls}
        chosen = [by[i] for i in s["call_ids"]]
        rows = []
        for a in ss.approaches:
            e = benchmark.estimate(a, chosen)
            live = a in LLM and ss[f"{a}_mode"] in {"live", "record"}
            rows.append({"Approach": benchmark.APPROACHES[a]["label"],
                         "Source": badge(ss[f"{a}_mode"]) if a in LLM else badge("offline"),
                         "Requests": e["requests"], "Expected USD (this sample)": e["usd"] if live else 0.0,
                         "Worst case USD": e["usd_max"] if live else 0.0,
                         "Expected USD per 1,000 calls (if live)": e["usd_per_1k"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="$%.3f") for c in
                                    ("Expected USD (this sample)", "Worst case USD", "Expected USD per 1,000 calls (if live)")})
        st.number_input("Spending cap per approach (USD)", 0.0, 100.0, step=0.5, key="cap")
    nav(STEPS[1], STEPS[3] if s else None)


def step_run():
    st.header("④ Run")
    s = current_sample()
    if not s:
        st.warning("Freeze a sample in step ③ first.")
        nav(STEPS[2])
        return
    calls = corpus(ss.corpus_id)
    by = {c["call_id"]: c for c in calls}
    st.caption(f"Sample **{s['sample_id']}** · {s['n']} calls. Results are saved after every call, so you can stop and resume.")
    rules = [a for a in ss.approaches if a.startswith("rules")]
    if rules:
        st.markdown("✅ " + ", ".join(benchmark.APPROACHES[a]["label"] for a in rules) + ": scored instantly in step ⑤ (free, offline).")
    for a in [x for x in ss.approaches if x in LLM]:
        m = project().latest_run(s["sample_id"], a)
        with st.container(border=True):
            label = benchmark.APPROACHES[a]["label"]
            status = (m or {}).get("status", "not started")
            st.markdown(f"**{label}** · {badge(ss[f'{a}_mode'])} · status: **{status}**"
                        + (f" · {m['n_done']}/{m['n_calls']} calls · \\${m.get('cost_usd', 0):.4f}" if m else ""))
            est = benchmark.estimate(a, [by[i] for i in s["call_ids"]])
            live = ss[f"{a}_mode"] in {"live", "record"}
            confirm = True
            if live and est["usd_max"] > 1.0:
                confirm = st.checkbox(f"I confirm spending about \\${est['usd']:.2f} (at most \\${est['usd_max']:.2f}; "
                                      f"{est['requests']:,} requests)", key=f"ok_{a}")
            if live and est["usd_max"] > ss.cap:
                st.warning(f"Worst case \\${est['usd_max']:.2f} is above the \\${ss.cap:.2f} cap; the run will pause at the cap.")
            c1, c2, c3 = st.columns(3)
            resume = bool(m) and m["status"] in {"partial", "paused_cap", "running"} and m.get("mode") == ss[f"{a}_mode"]
            if c1.button("Resume" if resume else ("Run again" if m else "Run"), type="primary", key=f"run_{a}",
                         disabled=needs_key(a) or not confirm):
                bar = st.progress(0.0, text="starting")
                rid = benchmark.run_approach(project(), s, calls, a, cfg_for(a), workers=ss.workers,
                                             cap_usd=ss.cap if live else None,
                                             progress=lambda d, n, usd: bar.progress(d / n, text=f"{d}/{n} calls · ${usd:.4f}"),
                                             run_id=m["run_id"] if resume else None)
                mm = project().manifest(rid)
                errs = project().results(rid, "errors")
                (st.success if mm["status"] == "done" else st.warning)(
                    f"{label}: {mm['status']} · {mm['n_done']}/{mm['n_calls']} calls · \\${mm['cost_usd']:.4f}")
                if errs:
                    st.error(f"{len(errs)} calls failed, e.g. {errs[0]['error'][:200]}")
            if c2.button("Latency probe", key=f"probe_{a}", disabled=needs_key(a) or not m,
                         help="20 decisions one at a time (after 3 warm-ups): real per-decision latency for the in-call budget."):
                with st.spinner("Measuring…"):
                    pr = benchmark.latency_probe(a, cfg_for(a), s, calls)
                project().update_manifest(m["run_id"], probe=pr)
                st.info(f"p50 {pr['p50']:.0f} ms · p95 {pr['p95']:.0f} ms" if pr["p50"] is not None else
                        f"No network timing from a {', '.join(pr['sources'])} source. Use live or record to measure latency.")
            if m and m.get("probe") and m["probe"].get("p50") is not None:
                c3.caption(f"⏱ probe p50 {m['probe']['p50']:.0f} ms / p95 {m['probe']['p95']:.0f} ms")
    with st.expander("Import results produced elsewhere"):
        st.caption("Attach a simulate.py results JSON to this sample. State where it came from: provenance is never guessed.")
        up = st.file_uploader("Results JSON", type="json", key="imp_file")
        ia = st.selectbox("Approach", list(LLM), format_func=lambda x: benchmark.APPROACHES[x]["label"], key="imp_a")
        im = st.selectbox("It was produced in", ["live", "record", "replay", "mock"], key="imp_mode")
        imodel = st.text_input("Model", value=jev_client.DEFAULT_MODEL if ia == "jev" else llm_client.DEFAULT_MODEL, key="imp_model")
        if st.button("Import", disabled=up is None, key="imp_btn"):
            try:
                rid = project().import_results(s, json.loads(up.read()), ia, im, imodel, benchmark.QUESTIONS_HASH)
                st.success(f"Imported as run {rid}.")
            except ValueError as e:
                st.error(str(e))
    nav(STEPS[2], STEPS[4], "Compare")


def step_compare():
    st.header("⑤ Compare")
    s = current_sample()
    if not s:
        st.warning("Freeze a sample in step ③ first.")
        nav(STEPS[2])
        return
    missing = [benchmark.APPROACHES[a]["label"] for a in ss.approaches if a in LLM and not project().latest_run(s["sample_id"], a)]
    if missing:
        st.info("Not run yet (step ④): " + ", ".join(missing) + ". Comparing what is available.")
    status, sc, ents = get_scorecard(s, ss.approaches)
    if status == "error":
        st.error(f"These runs cannot be compared: {sc}")
        nav(STEPS[3])
        return
    for n, why in benchmark.unavailable(s, ss.approaches, ents).items():
        if n not in LLM:
            st.caption(f"ℹ️ {benchmark.APPROACHES[n]['label']} is not included: {why}.")
    summary_block(sc, s)
    if ss.tech:
        st.subheader("Evidence (paired, 95% CIs)")
        rows = []
        for a in sc["approaches"].values():
            row = {"approach": a["label"], "source": badge(a["mode"])}
            for k in benchmark.METRICS:
                m = a["metrics"][k]
                row[k] = "–" if m["point"] is None else f"{m['point']:.3f}" + (
                    f" [{m['ci'][0]:.2f}, {m['ci'][1]:.2f}]" if m["ci"][0] is not None else "")
            lat = a["latency"]
            row.update({"brier_pre": a["brier_pre_outcome"], "ece_pre": a["ece_pre_outcome"],
                        "$ / 1k calls": a["cost_per_1k"], "contract errors": f"{a['contract_error_rate']:.1%}",
                        "latency p50/p95": "–" if lat["p50"] is None else f"{lat['p50']:.0f}/{lat['p95']:.0f} ({lat['source']})"})
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        metric = st.selectbox("Paired differences for", list(benchmark.METRICS), key="forest_metric")
        pts = {n: a["metrics"][metric]["point"] for n, a in sc["approaches"].items() if a["metrics"][metric]["point"] is not None}
        if pts:
            best = max(pts, key=pts.get)
            ch = ui_charts.forest(sc, metric, best)
            if ch is not None:
                st.caption(f"Each approach minus the best ({sc['approaches'][best]['label']}). A range crossing 0 = too close to call.")
                st.altair_chart(ch, width="stretch")
        st.subheader("Call overlay")
        by_call = {n: {r["call_id"]: r for r in e["results"]} for n, e in ents.items()}
        dis = []
        for cid in s["call_ids"]:
            ps = [benchmark._point(by_call[n][cid], "pre_outcome") for n in by_call if cid in by_call[n]]
            if len(ps) > 1:
                dis.append((max(ps) - min(ps), cid))
        dis.sort(reverse=True)
        label_of = {c["call_id"]: c["label_booked"] for c in corpus(ss.corpus_id)}
        cid = st.selectbox("Call (most disagreement first)", [c for _, c in dis] or s["call_ids"], key="overlay_call",
                           format_func=lambda c: f"{c} · booked={label_of.get(c, '?')}")
        per = {n: by_call[n][cid] for n in by_call if cid in by_call[n]}
        ch = ui_charts.trajectory_overlay(per, {n: sc["approaches"][n]["label"] for n in per})
        if ch is not None:
            st.altair_chart(ch, width="stretch")
        reviews = {sc["approaches"][n]["label"]: r["review"] for n, r in per.items() if r.get("review")}
        if reviews:
            st.dataframe(pd.DataFrame([{"approach": k, "failure_mode": v["failure_mode"]["choice"],
                                        "agent_progress": round(v["agent_progress"]["score"], 2),
                                        "needs_human": round(v["needs_human"]["noul"], 2)} for k, v in reviews.items()]),
                         hide_index=True, width="stretch")
    nav(STEPS[3], STEPS[5], "Decide")


def step_decide():
    st.header("⑥ Decide & export")
    s = current_sample()
    if not s:
        st.warning("Freeze a sample in step ③ first.")
        nav(STEPS[2])
        return
    status, sc, ents = get_scorecard(s, ss.approaches)
    if status == "error":
        st.error(sc)
        return
    st.subheader("Quality gate")
    gate_rows, ok_all = [], True
    by = {c["call_id"]: c for c in corpus(ss.corpus_id)}
    for n, e in ents.items():
        if e["manifest"] is None:
            gate_rows.append({"approach": sc["approaches"][n]["label"], "contract (E1)": "n/a (rule)", "no leak (E2)": "n/a (rule)"})
            continue
        c = audit.contract_summary(e["results"])
        lk = audit.no_leak_summary([by[r["call_id"]] for r in e["results"] if r["call_id"] in by])
        ok_all &= c["ok"] and lk["ok"]
        gate_rows.append({"approach": sc["approaches"][n]["label"], "contract (E1)": ("PASS · " if c["ok"] else "FAIL · ") + c["detail"],
                          "no leak (E2)": ("PASS · " if lk["ok"] else "FAIL · ") + lk["detail"]})
    hits = audit.scan_artifact_files([p for p in project().dir.rglob("*") if p.is_file()])
    ok_all &= not hits
    st.dataframe(pd.DataFrame(gate_rows), hide_index=True, width="stretch")
    if ok_all:
        st.success("Gate: PASS. No key material in the project workspace.")
    else:
        st.error(f"Gate: FAIL. {len(hits)} secret-hygiene hits." if hits else "Gate: FAIL. See the table.")

    st.subheader("Record decisions")
    names = list(sc["approaches"])
    for uc in ss.use_cases:
        v = sc["verdicts"][uc]
        with st.container(border=True):
            st.markdown(f"**{v['label']}**: recommended **{sc['approaches'][v['winner']]['label'] if v['winner'] else '–'}** "
                        f"(confidence {v['confidence']})")
            c1, c2, c3 = st.columns([2, 3, 1])
            pick = c1.selectbox("Decision", names, index=names.index(v["winner"]) if v["winner"] in names else 0,
                                key=f"dec_{uc}", format_func=lambda n: sc["approaches"][n]["label"])
            note = c2.text_input("Note", key=f"note_{uc}", placeholder="why, owner, follow-up")
            if c3.button("Save", key=f"save_{uc}"):
                a = sc["approaches"][pick]
                project().record_decision(uc, pick, note, {
                    "sample_id": s["sample_id"], "run_id": a["run_id"], "mode": a["mode"], "metric": v["metric"],
                    "point": a["metrics"][v["metric"]]["point"], "ci": a["metrics"][v["metric"]]["ci"],
                    "cost_per_1k": a["cost_per_1k"], "latency_p95": a["latency"]["p95"],
                    "followed_recommendation": pick == v["winner"]})
                st.success("Saved.")
    st.subheader("Export")
    manifests = [e["manifest"] for e in ents.values() if e["manifest"]]
    report = benchmark.report_markdown(sc, s, ss.use_cases, project().meta()["decisions"], manifests, ss.volume)
    c1, c2, c3 = st.columns(3)
    c1.download_button("Report (Markdown)", report, file_name=f"benchmark-{s['sample_id']}.md", type="primary")
    c2.download_button("Scorecard (JSON)", json.dumps(sc, indent=1, default=str), file_name=f"scorecard-{s['sample_id']}.json")
    c3.download_button("Run manifests (JSON)", json.dumps(manifests, indent=1), file_name=f"manifests-{s['sample_id']}.json")
    with st.expander("Preview report"):
        st.markdown(md(report))
    nav(STEPS[4])


def tools_page():
    st.header("🔧 Tools")
    t1, t2, t3 = st.tabs(["Audit", "Lab sign-off (E1–E7)", "Playground"])
    with t1:
        ui_tools.audit_tool(project(), current_sample())
    with t2:
        ui_tools.signoff_tool(jev_cfg(), needs_key("jev"))
    with t3:
        ui_tools.playground_tool(jev_cfg(), glm_cfg(), needs_key("jev") or needs_key("glm"))


# ---------------------------------------------------------------- main
if ss.view == "Summary":
    summary_view()
else:
    st.title("Call-analysis benchmark")
    st.caption("Compare rules, Jev and GLM on the same calls. Jev and GLM see only call structure (actors, stages, "
               "tool results, timing), never words.")
    done = step_status()
    options = STEPS + ([TOOLS] if ss.tech else [])
    if ss.step not in options:
        ss.step = STEPS[0]
    st.radio("Step", options, key="step", horizontal=True, label_visibility="collapsed",
             format_func=lambda x: f"{x} ✓" if done.get(x) else x)
    st.divider()
    {STEPS[0]: step_connect, STEPS[1]: step_corpus, STEPS[2]: step_plan, STEPS[3]: step_run,
     STEPS[4]: step_compare, STEPS[5]: step_decide, TOOLS: tools_page}[ss.step]()
