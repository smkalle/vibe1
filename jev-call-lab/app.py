"""jev-call-lab workbench (Streamlit).

    streamlit run app.py

Paste an OpenRouter key in the sidebar for a run. The key lives only in this
browser session's memory: it is never written to disk, results, fixtures or logs,
and it is not read from or written to the environment.
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import partial
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import audit
import baselines
import evaluate
import jev_client
import llm_client
import run_evals
from jev_client import Config, JevError, decide, validate_response
from questions import REVIEW, WILL_BOOK, prefix_state
from simulate import forecast_call

HERE = Path(__file__).parent
DATA = HERE / "data"
RESULTS = HERE / "results"

DATASETS = {
    "Synthetic phone calls (153)": "synthetic_calls.json",
    "SGD test, real dialogues (167)": "sgd_test.json",
    "SGD dev (44)": "sgd_dev.json",
    "SGD train (551)": "sgd_train.json",
}
ENDPOINTS = {
    "OpenRouter Decisions API": ("https://openrouter.ai/api/alpha/decisions", "typesafe/jev-1.13"),
    "OpenRouter, TypeSafe SDK path": ("https://openrouter.ai/api/v1/systemone", "jev-1.13"),
    "Local mock server (python mock_jev.py)": ("http://127.0.0.1:8765/api/alpha/decisions", "typesafe/jev-1.13"),
    "Custom": ("", "typesafe/jev-1.13"),
}
MODES = {
    "mock": "Mock: in-process heuristic, no key, free (NOT Jev)",
    "live": "Live: call the endpoint with your key",
    "record": "Record: live, and save every response to fixtures/",
    "replay": "Replay: answer from fixtures/ only, no network",
}

# Reference palette (dataviz skill): categorical slots in fixed order, status red for failures.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
CRITICAL = "#d03b3b"
GRID_GRAY = "#8a8984"

st.set_page_config(page_title="Jev Call Workbench", page_icon="📞", layout="wide")


# ---------------------------------------------------------------- data helpers
@st.cache_data
def load_calls(name: str) -> list:
    return json.loads((DATA / name).read_text())


@st.cache_data(show_spinner="Scoring non-LLM baselines…")
def baseline_table(dataset_file: str, call_ids: tuple) -> dict:
    calls = [c for c in load_calls(dataset_file) if c["call_id"] in set(call_ids)]
    if dataset_file.startswith("sgd_") and dataset_file != "sgd_train.json":
        return {k: v["auc"] for k, v in baselines.evaluate_all(load_calls("sgd_train.json"), calls).items()}
    return baselines.cross_validate(calls) if len(calls) >= 10 else {}


def session_cfg() -> Config:
    s = st.session_state
    return Config(
        mode=s.get("mode", "mock"),
        api_key=s.get("api_key") or None,
        url=s.get("url") or ENDPOINTS["OpenRouter Decisions API"][0],
        model=s.get("model") or "typesafe/jev-1.13",
        fixtures=Path(s.get("fixtures") or HERE / "fixtures"),
    )


def llm_session_cfg() -> llm_client.Config:
    """GLM scorer config. Shares the mode and the session key; never persisted."""
    s = st.session_state
    base = session_cfg()
    return llm_client.Config(
        mode=base.mode,
        api_key=base.api_key,
        url=s.get("glm_url") or llm_client.DEFAULT_URL,
        model=s.get("glm_model") or llm_client.DEFAULT_MODEL,
        effort=s.get("glm_effort") or llm_client.DEFAULT_EFFORT,
        fixtures=base.fixtures,
    )


SCORERS = {"jev": "Jev (Decisions API)", "glm": "GLM (chat completions)"}


def needs_key(cfg: Config) -> bool:
    return cfg.mode in {"live", "record"} and not cfg.api_key


def run_calls(calls, cfg, workers, review, progress, decide_fn=None, scorer="jev"):
    """Forecast every call in parallel; one failing call never sinks the run."""
    results, errors = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(forecast_call, c, review, cfg, decide_fn, scorer): c["call_id"] for c in calls}
        for n, fut in enumerate(as_completed(futs), 1):
            try:
                results.append(fut.result())
            except Exception as e:  # noqa: BLE001 - surfaced in the UI per call
                errors.append((futs[fut], f"{type(e).__name__}: {e}"))
            progress.progress(n / len(calls), text=f"{n}/{len(calls)} calls")
    order = {c["call_id"]: i for i, c in enumerate(calls)}
    results.sort(key=lambda r: order[r["call_id"]])
    return results, errors


def store_run(name, results, errors, cfg, dataset_file, elapsed, scorer="jev"):
    st.session_state.setdefault("runs", {})[name] = {
        "results": results,
        "errors": errors,
        "dataset": dataset_file,
        # Settings only; the key is deliberately not stored with the run.
        "settings": {"mode": cfg.mode, "url": cfg.url, "model": cfg.model, "scorer": scorer},
        "elapsed_s": round(elapsed, 1),
        "at": datetime.now().strftime("%H:%M:%S"),
    }
    st.session_state["current_run"] = name


# ---------------------------------------------------------------- charts
def trajectory_chart(result: dict):
    df = pd.DataFrame(result["turns"])
    if df.empty:
        return None
    df["t_s"] = df["t_ms"] / 1000
    df["status"] = df["ok"].map({True: "ok", False: "failed event"})
    base = alt.Chart(df).encode(
        x=alt.X("t_s:Q", title="Seconds into call"),
        y=alt.Y("p_book:Q", title="P(book)", scale=alt.Scale(domain=[0, 1])),
    )
    line = base.mark_line(color=SERIES[0], strokeWidth=2)
    tooltip = [alt.Tooltip("t_s:Q", title="t (s)", format=".1f"), "stage:N", "event:N",
               alt.Tooltip("p_book:Q", title="P(book)", format=".2f"), "status:N"]
    points = base.mark_point(filled=True, size=90, strokeWidth=2, stroke="white").encode(
        color=alt.Color("status:N", scale=alt.Scale(domain=["ok", "failed event"], range=[SERIES[0], CRITICAL]),
                        legend=alt.Legend(title=None, orient="top")),
        shape=alt.Shape("status:N", scale=alt.Scale(domain=["ok", "failed event"], range=["circle", "cross"]), legend=None),
        tooltip=tooltip,
    )
    half = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(strokeDash=[4, 4], color=GRID_GRAY, strokeWidth=1).encode(y="y:Q")
    return (half + line + points).properties(height=280)


def auc_chart(table: dict):
    rows = [{"scorer": scorer, "point": point, "AUC": v}
            for scorer, aucs in table.items() for point, v in aucs.items() if v is not None]
    if not rows:
        return None
    df = pd.DataFrame(rows)
    scorers = list(table)
    palette = (SERIES * ((len(scorers) + len(SERIES) - 1) // len(SERIES)))[: len(scorers)]
    return alt.Chart(df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("point:N", title=None, sort=["mid", "pre_outcome", "pre_outcome_attempted", "end"]),
        xOffset=alt.XOffset("scorer:N", sort=scorers),
        y=alt.Y("AUC:Q", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("scorer:N", sort=scorers, scale=alt.Scale(domain=scorers, range=palette),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=["scorer:N", "point:N", alt.Tooltip("AUC:Q", format=".3f")],
    ).properties(height=280)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Connection")
    st.radio("Mode", list(MODES), format_func=lambda m: MODES[m], key="mode")
    st.text_input(
        "OpenRouter API key",
        type="password",
        key="api_key",
        placeholder="sk-or-v1-…",
        help="Held only in this browser session's memory for the runs you start. "
             "Never written to disk, results, fixtures or logs.",
        disabled=st.session_state.get("mode", "mock") in {"mock", "replay"},
    )
    ep = st.selectbox("Endpoint", list(ENDPOINTS), key="endpoint")
    if st.session_state.get("_last_ep") != ep:  # reset URL/model when the preset changes
        st.session_state["url"], st.session_state["model"] = ENDPOINTS[ep]
        st.session_state["_last_ep"] = ep
    st.text_input("URL", key="url")
    st.text_input("Model", key="model")
    st.text_input("Fixtures directory", value=str(HERE / "fixtures"), key="fixtures")
    st.text_input("GLM model", value=llm_client.DEFAULT_MODEL, key="glm_model",
                  help="Chat-completions slug for the GLM scorer (e.g. z-ai/glm-5.3). Uses the same key.")
    st.selectbox("GLM reasoning effort", ["low", "high", "max"], key="glm_effort",
                 help="GLM-5.3 always thinks; higher effort costs more output tokens.")

    c1, c2 = st.columns(2)
    if c1.button("Test connection", width="stretch"):
        cfg = session_cfg()
        if needs_key(cfg):
            st.error("Paste a key first.")
        else:
            try:
                r = decide({"current_event": "identity_check_failed", "current_stage": "identity"}, WILL_BOOK, cfg=cfg)
                errs = validate_response(WILL_BOOK, r)
                (st.success if not errs else st.warning)(
                    f"{r.get('model')} · P(book)={r['answers']['will_book']['noul']:.2f} · "
                    f"{r['_latency_ms']:.0f} ms · contract {'OK' if not errs else errs}")
            except JevError as e:
                st.error(str(e))
    if c2.button("Forget key", width="stretch"):
        st.session_state["api_key"] = ""
        st.rerun()

    cfg_now = session_cfg()
    if cfg_now.mode == "mock":
        st.info("Mock mode: numbers prove the pipeline, not Jev.")
    elif needs_key(cfg_now):
        st.warning("This mode needs a key.")


st.title("Jev call-structure workbench")
st.caption("Jev sees only structure (actors, stages, tool results, timing), never words. "
           "Forecast P(book) after every agent turn, review whole calls, and compare against non-LLM baselines.")

tab_run, tab_results, tab_call, tab_analytics, tab_bench, tab_signoff, tab_audit, tab_play = st.tabs(
    ["Run", "Results", "Call explorer", "Analytics", "Benchmark", "Sign-off (E1–E7)", "Audit", "Playground"])

# ---------------------------------------------------------------- Run
with tab_run:
    left, right = st.columns([2, 1])
    with left:
        ds_label = st.selectbox("Dataset", list(DATASETS))
        ds_file = DATASETS[ds_label]
        calls_all = load_calls(ds_file)
        limit = st.slider("Calls", 1, len(calls_all), min(len(calls_all), 20))
        workers = st.slider("Parallel requests", 1, 32, 8, help="Start at 8–16; raise after checking rate limits.")
        review = st.checkbox("Five-question review per call", value=True)
        scorer = st.radio("Scorer", list(SCORERS), format_func=lambda s: SCORERS[s], horizontal=True, key="scorer")
    calls = calls_all[:limit]
    if scorer == "glm":
        glm_est = llm_client.project_cost(calls, review)
        est_requests, est_usd, est_note = glm_est["requests"], glm_est["usd"], \
            f"~{glm_est['est_completion_tokens']:,} completion tokens (reasoning + JSON, rough)"
    else:
        est = evaluate.project_cost([], calls)
        est_requests, est_usd, est_note = est["requests"], est["usd"], f"{est['input_tokens']:,} input tokens"
    with right:
        st.metric("Requests", f"{est_requests if review else est_requests - len(calls):,}")
        st.metric("Est. cost (USD)", f"${est_usd:.4f}" if cfg_now.mode in {"live", "record"} else "$0 (offline)",
                  help=est_note)
        cap = st.number_input("Cost cap (USD)", 0.0, 50.0, 0.50, 0.05)

    run_name = st.text_input("Run name", f"{ds_file.removesuffix('.json')}-{scorer}-{cfg_now.mode}-{limit}")
    blocked = needs_key(cfg_now) or (cfg_now.mode in {"live", "record"} and est_usd > cap)
    if blocked and not needs_key(cfg_now):
        st.warning("Estimated cost is above the cap.")
    if st.button("Run", type="primary", disabled=blocked):
        cfg = session_cfg()
        t0 = time.time()
        bar = st.progress(0.0, text="starting")
        if scorer == "glm":
            eff_cfg = llm_session_cfg()
            results, errors = run_calls(calls, eff_cfg, workers, review, bar,
                                        partial(llm_client.decide, cfg=eff_cfg), "glm")
        else:
            eff_cfg = cfg
            results, errors = run_calls(calls, cfg, workers, review, bar)
        store_run(run_name, results, errors, eff_cfg, ds_file, time.time() - t0, scorer)
        if errors:
            st.error(f"{len(errors)} calls failed. First: {errors[0][1]}")
        st.success(f"{len(results)} calls done in {time.time() - t0:.1f}s. Open the Results tab.")

    st.subheader("Load saved results")
    files = sorted(RESULTS.glob("*.json")) if RESULTS.exists() else []
    up = st.file_uploader("…or upload a results JSON from simulate.py", type="json")
    pick = st.selectbox("Saved in results/", ["—"] + [f.name for f in files])
    if st.button("Load") and (up or pick != "—"):
        data = json.loads(up.read() if up else (RESULTS / pick).read_text())
        guess = "sgd_test.json" if str(data[0]["call_id"]).startswith("sgd-test") else "synthetic_calls.json"
        scorer_guess = "glm" if any(r.get("scorer") == "glm" for r in data) else "jev"
        store_run(up.name if up else pick, data, [], Config(), guess, 0, scorer_guess)
        st.success(f"Loaded {len(data)} calls.")

runs = st.session_state.get("runs", {})


def pick_run(key):
    if not runs:
        st.info("No runs yet. Start one on the Run tab.")
        return None, None
    names = list(runs)
    cur = st.session_state.get("current_run", names[-1])
    name = st.selectbox("Run", names, index=names.index(cur) if cur in names else len(names) - 1, key=key)
    return name, runs[name]


# ---------------------------------------------------------------- Results
with tab_results:
    name, run = pick_run("results_run")
    if run and run["results"]:
        res = run["results"]
        m = evaluate.metrics(res)
        st.caption(f"{run['settings']['mode']} · {run['settings']['model']} · {run['dataset']} · finished {run['at']} "
                   f"in {run['elapsed_s']}s · answered by {', '.join(m['models'])}")
        k = st.columns(5)
        k[0].metric("Calls (booked)", f"{m['n_calls']} ({m['n_booked']})")
        k[1].metric("Requests", f"{m['n_requests']:,}")
        k[2].metric("Cost (USD)", f"${m['cost']['total_usd']:.5f}", help=f"source: {', '.join(m['cost']['source'])}")
        k[3].metric("Latency p50 / p95", f"{m['latency_ms']['p50']:.0f} / {m['latency_ms']['p95']:.0f} ms")
        contract_bad = sum(bool(t.get("errors")) for r in res for t in r["turns"]) + sum(bool(r.get("review_errors")) for r in res)
        k[4].metric("Contract errors", contract_bad)
        if run["errors"]:
            with st.expander(f"{len(run['errors'])} calls failed"):
                st.dataframe(pd.DataFrame(run["errors"], columns=["call_id", "error"]), hide_index=True)

        st.subheader("Ranking quality: Jev vs non-LLM baselines")
        table = {"Jev (this run)": m["auc"]}
        for b, v in baseline_table(run["dataset"], tuple(r["call_id"] for r in res)).items():
            if b != "B0_constant":
                table[b] = v
        chart = auc_chart(table)
        if chart is not None:
            st.altair_chart(chart, width="stretch")
        st.dataframe(pd.DataFrame(table).T, width="stretch")
        st.caption("mid = middle forecast · pre_outcome = last forecast before booking/wrap-up starts · "
                   "pre_outcome_attempted = same, only calls that tried to book (blank if < 5 per class) · end = last forecast.")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Calibration (pre-outcome)")
            st.dataframe(pd.DataFrame(m["calibration_pre_outcome"]), hide_index=True, width="stretch")
            st.write("Brier:", m["brier"])
        with c2:
            st.subheader("Failure mode: Jev vs rule reference")
            rv = [r for r in res if r.get("review")]
            if rv:
                ct = pd.crosstab(pd.Series([r["reference_failure_mode"] for r in rv], name="reference"),
                                 pd.Series([r["review"]["failure_mode"]["choice"] for r in rv], name="Jev"))
                st.dataframe(ct, width="stretch")
                st.caption(f"Agreement: {m['failure_mode']['agreement_with_rule_reference']}")

        st.download_button("Download results JSON", json.dumps(res, indent=1), file_name=f"{name}.json")
        if st.button("Save to results/"):
            RESULTS.mkdir(exist_ok=True)
            (RESULTS / f"{name}.json").write_text(json.dumps(res, indent=1))
            st.success(f"Saved results/{name}.json")

# ---------------------------------------------------------------- Call explorer
with tab_call:
    name, run = pick_run("call_run")
    if run and run["results"]:
        res = run["results"]
        only = st.radio("Show", ["all", "booked", "not booked", "had a failure"], horizontal=True)
        keep = {
            "all": lambda r: True,
            "booked": lambda r: r["label_booked"],
            "not booked": lambda r: not r["label_booked"],
            "had a failure": lambda r: any(not t["ok"] for t in r["turns"]),
        }[only]
        options = [r for r in res if keep(r)]
        if options:
            r = st.selectbox("Call", options, format_func=lambda r: f"{r['call_id']} · booked={r['label_booked']} · "
                                                                    f"end P={r['turns'][-1]['p_book']:.2f}" if r["turns"] else r["call_id"])
            chart = trajectory_chart(r)
            if chart is not None:
                st.altair_chart(chart, width="stretch")
            calls_by_id = {c["call_id"]: c for c in load_calls(run["dataset"])}
            call = calls_by_id.get(r["call_id"])
            c1, c2 = st.columns([3, 2])
            with c1:
                st.subheader("Turn forecasts")
                st.dataframe(pd.DataFrame(r["turns"])[["event_index", "t_ms", "stage", "event", "ok", "p_book", "latency_ms"]],
                             hide_index=True, width="stretch")
            with c2:
                st.subheader("Review")
                if r.get("review"):
                    rv = r["review"]
                    st.write(f"**failure_mode:** {rv['failure_mode']['choice']} "
                             f"(confidence {rv['failure_mode']['confidence']:.2f}; reference: {r['reference_failure_mode']})")
                    st.write(f"**agent_progress:** {rv['agent_progress']['score']:.2f} / 3")
                    st.write(f"**will_book:** {rv['will_book']['noul']:.2f} · **needs_human:** {rv['needs_human']['noul']:.2f} · "
                             f"**visible_error_overweight:** {rv['visible_error_overweight']['noul']:.2f}")
                    st.bar_chart(pd.Series(rv["failure_mode"]["probabilities"], name="probability"), color=SERIES[0], horizontal=True)
            if call and r["turns"]:
                st.subheader("Exactly what Jev saw")
                idx = st.select_slider("After event", [t["event_index"] for t in r["turns"]], value=r["turns"][-1]["event_index"])
                st.json(prefix_state(call, call["events"][: idx + 1], idx), expanded=False)

# ---------------------------------------------------------------- Analytics
with tab_analytics:
    st.markdown("Compare runs side by side: ranking quality, cost and latency. Select any runs started or loaded this session.")
    if not runs:
        st.info("No runs yet. Start one on the Run tab.")
    else:
        sel = st.multiselect("Runs", list(runs), default=list(runs), key="analytics_runs")
        if sel:
            picked = {n: runs[n] for n in sel}
            st.dataframe(pd.DataFrame(audit.summarize_runs(picked)), hide_index=True, width="stretch")
            auc_table = {n: evaluate.metrics(runs[n]["results"])["auc"] for n in sel if runs[n]["results"]}
            chart = auc_chart(auc_table)
            if chart is not None:
                st.subheader("AUC by run")
                st.altair_chart(chart, width="stretch")
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Cost per call (USD)")
                st.bar_chart(pd.Series({n: evaluate.metrics(runs[n]["results"])["cost"]["total_usd"] / max(len(runs[n]["results"]), 1)
                                            for n in sel if runs[n]["results"]}, name="USD / call"), color=SERIES[1], horizontal=True)
            with c2:
                st.subheader("Latency p50 / p95 (ms)")
                st.dataframe(pd.DataFrame([{"run": n, **evaluate.metrics(runs[n]["results"])["latency_ms"]}
                                           for n in sel if runs[n]["results"]]), hide_index=True, width="stretch")

# ---------------------------------------------------------------- Benchmark
def _bench_run(name, calls, cfg, workers, review, bar, scorer):
    """Run one benchmark leg and store it tagged with its scorer."""
    t0 = time.time()
    if scorer == "glm":
        lcfg = llm_session_cfg()
        results, errors = run_calls(calls, lcfg, workers, review, bar, partial(llm_client.decide, cfg=lcfg), "glm")
        store_run(name, results, errors, lcfg, st.session_state["bench_file"], time.time() - t0, "glm")
    else:
        results, errors = run_calls(calls, cfg, workers, review, bar)
        store_run(name, results, errors, cfg, st.session_state["bench_file"], time.time() - t0, "jev")
    return results, errors


with tab_bench:
    st.markdown("Same calls, three approaches: **Jev** (Decisions API) vs **rules** (B0/B1/B2, instant, offline) "
                "vs **GLM** (chat completions). Accuracy = AUC at mid / pre_outcome / end; "
                "performance = latency p50 and $/call.")
    b_ds_label = st.selectbox("Dataset", list(DATASETS), key="bench_ds")
    b_file = DATASETS[b_ds_label]
    st.session_state["bench_file"] = b_file
    b_all = load_calls(b_file)
    b_limit = st.slider("Calls", 1, len(b_all), min(len(b_all), 10), key="bench_n",
                        help="Keep small for LLM cost: each call is ~7 forecasts + 1 review per scorer.")
    b_review = st.checkbox("Five-question review per call", value=True, key="bench_review")
    b_workers = st.slider("Parallel requests", 1, 32, 8, key="bench_workers")
    b_calls = b_all[:b_limit]

    st.subheader("Rules (instant, offline)")
    if st.button("Score rules", key="bench_rules_btn"):
        with st.spinner("Scoring B0/B1/B2…"):
            if b_file.startswith("sgd_") and b_file != "sgd_train.json":
                st.session_state["bench_rules"] = {
                    k: v["auc"] for k, v in baselines.evaluate_all(load_calls("sgd_train.json"), b_calls).items()}
            else:
                st.session_state["bench_rules"] = baselines.cross_validate(b_calls)
    rules = st.session_state.get("bench_rules")
    if rules:
        st.dataframe(pd.DataFrame(rules).T, width="stretch")
        st.caption("B0 constant 0.5 · B1 furthest stage reached · B2 logistic regression on event counts.")

    st.subheader("LLM scorers on the same calls")
    jev_est = evaluate.project_cost([], b_calls)
    glm_est = llm_client.project_cost(b_calls, b_review)
    e1, e2 = st.columns(2)
    e1.metric("Jev estimate", f"${jev_est['usd']:.4f}", f"{jev_est['requests']} requests")
    e2.metric("GLM estimate", f"${glm_est['usd']:.4f}",
              f"{glm_est['requests']} requests · ~{glm_est['est_completion_tokens']:,} completion tokens (rough)",
              help="GLM-5.3 always thinks; completion estimate covers reasoning + JSON. Billed input "
                   f"${llm_client.PRICE_IN_PER_TOKEN * 1e6:.2f}/M + output ${llm_client.PRICE_OUT_PER_TOKEN * 1e6:.2f}/M.")
    bcap = st.number_input("Benchmark cost cap (USD)", 0.0, 50.0, 2.00, 0.25, key="bench_cap")
    live = cfg_now.mode in {"live", "record"}
    b1, b2 = st.columns(2)
    jev_blocked = needs_key(cfg_now) or (live and jev_est["usd"] > bcap)
    glm_blocked = needs_key(cfg_now) or (live and glm_est["usd"] > bcap)
    if b1.button(f"Run Jev ({b_limit} calls)", type="primary", disabled=jev_blocked, key="bench_jev"):
        bar = st.progress(0.0, text="Jev")
        res, errs = _bench_run(f"bench-{b_file.removesuffix('.json')}-jev-{b_limit}",
                               b_calls, session_cfg(), b_workers, b_review, bar, "jev")
        (st.error if errs else st.success)(f"Jev: {len(res)} calls, {len(errs)} failed." if errs else f"Jev: {len(res)} calls done.")
    if b2.button(f"Run GLM ({b_limit} calls)", type="primary", disabled=glm_blocked, key="bench_glm"):
        bar = st.progress(0.0, text="GLM")
        res, errs = _bench_run(f"bench-{b_file.removesuffix('.json')}-glm-{b_limit}",
                               b_calls, session_cfg(), b_workers, b_review, bar, "glm")
        (st.error if errs else st.success)(f"GLM: {len(res)} calls, {len(errs)} failed." if errs else f"GLM: {len(res)} calls done.")
    if (jev_blocked or glm_blocked) and not needs_key(cfg_now):
        st.warning("An estimate is above the cap.")

    st.subheader("…or reuse saved runs")
    r1, r2 = st.columns(2)
    jev_opts = ["—"] + [n for n, r in runs.items()
                        if r["settings"].get("scorer", "jev") == "jev" and r["dataset"] == b_file and r["results"]]
    glm_opts = ["—"] + [n for n, r in runs.items()
                        if r["settings"].get("scorer") == "glm" and r["dataset"] == b_file and r["results"]]
    bench_jev_name = r1.selectbox("Jev run", jev_opts, key="bench_jev_run")
    bench_glm_name = r2.selectbox("GLM run", glm_opts, key="bench_glm_run")
    auto_jev = [n for n in jev_opts[1:] if n.startswith(f"bench-{b_file.removesuffix('.json')}-jev-")]
    auto_glm = [n for n in glm_opts[1:] if n.startswith(f"bench-{b_file.removesuffix('.json')}-glm-")]
    if bench_jev_name == "—" and auto_jev:
        bench_jev_name = auto_jev[-1]
    if bench_glm_name == "—" and auto_glm:
        bench_glm_name = auto_glm[-1]

    st.subheader("Benchmark table")
    brows, bauc = [], {}
    for label, rname in (("Jev", bench_jev_name), ("GLM-5.3", bench_glm_name)):
        if rname and rname != "—":
            m = evaluate.metrics(runs[rname]["results"])
            bad = sum(bool(t.get("errors")) for r in runs[rname]["results"] for t in r["turns"])
            brows.append({"scorer": f"{label} ({rname})", **{f"auc_{k}": v for k, v in m["auc"].items()},
                          "brier_pre_outcome": m["brier"]["pre_outcome"], "latency_p50_ms": m["latency_ms"]["p50"],
                          "usd_per_call": m["cost"]["total_usd"] / max(m["n_calls"], 1),
                          "requests": m["n_requests"], "contract_errors": bad})
            bauc[label] = m["auc"]
    if rules:
        for rn, aucs in rules.items():
            brows.append({"scorer": f"rule {rn}", **{f"auc_{k}": v for k, v in aucs.items()},
                          "brier_pre_outcome": None, "latency_p50_ms": 0.0,
                          "usd_per_call": 0.0, "requests": 0, "contract_errors": 0})
            bauc[f"rule {rn}"] = aucs
    if brows:
        st.dataframe(pd.DataFrame(brows), hide_index=True, width="stretch")
        chart = auc_chart(bauc)
        if chart is not None:
            st.altair_chart(chart, width="stretch")
        scored = [b for b in brows if b["auc_pre_outcome"] is not None]
        if scored:
            best = max(scored, key=lambda b: b["auc_pre_outcome"])
            cheap = min([b for b in brows if b["requests"]], key=lambda b: b["usd_per_call"])
            fast = min([b for b in brows if b["requests"]], key=lambda b: b["latency_p50_ms"])
            st.caption(f"Best pre_outcome AUC: **{best['scorer']}** ({best['auc_pre_outcome']:.3f}) · "
                       f"cheapest: **{cheap['scorer']}** (${cheap['usd_per_call']:.5f}/call) · "
                       f"fastest: **{fast['scorer']}** ({fast['latency_p50_ms']:.0f} ms p50). "
                       f"Rules cost $0 and run offline; LLM runs on {b_limit} calls from {b_file}.")
    else:
        st.info("Score the rules and run (or pick) at least one LLM scorer to fill the table.")

# ---------------------------------------------------------------- Sign-off
with tab_signoff:
    st.markdown("Runs the full sign-off with the sidebar connection: **every synthetic call + the SGD test split** "
                "(about 2,900 requests), then evals E1–E7. In live/record mode E4/E5 measure Jev and are reported, not gated.")
    syn, test = load_calls("synthetic_calls.json"), load_calls("sgd_test.json")
    est_s = evaluate.project_cost([], syn + test)
    st.write(f"Estimated: {est_s['requests']:,} requests · {est_s['input_tokens']:,} input tokens · "
             f"**${est_s['usd']:.3f}** if live")
    so_workers = st.slider("Parallel requests ", 1, 32, 16)
    if st.button("Run sign-off", type="primary", disabled=needs_key(cfg_now)):
        cfg = session_cfg()
        t0 = time.time()
        bar = st.progress(0.0, text="synthetic set")
        syn_res, e1 = run_calls(syn, cfg, so_workers, True, bar)
        bar2 = st.progress(0.0, text="SGD test")
        test_res, e2 = run_calls(test, cfg, so_workers, True, bar2)
        store_run(f"signoff-synthetic-{cfg.mode}", syn_res, e1, cfg, "synthetic_calls.json", time.time() - t0)
        store_run(f"signoff-sgd_test-{cfg.mode}", test_res, e2, cfg, "sgd_test.json", time.time() - t0)
        if e1 or e2:
            st.error(f"{len(e1) + len(e2)} calls failed, first: {(e1 + e2)[0][1]}. Fix the connection and re-run.")
        else:
            with st.spinner("Evaluating E1–E7 (includes a local record/replay check)…"):
                ok, md, rows = run_evals.build_report(syn_res, test_res, live=cfg.mode != "mock", mode_label=cfg.mode)
            st.session_state["signoff"] = (ok, md, rows, cfg.mode)
    if "signoff" in st.session_state:
        ok, md, rows, mode_used = st.session_state["signoff"]
        (st.success if ok else st.error)(f"Overall: {'PASS' if ok else 'FAIL'} ({mode_used})")
        st.dataframe(pd.DataFrame([{"ID": i, "Eval": n, "Gate": "gate" if g else "report",
                                    "Result": ("PASS" if o else "FAIL") if g else ("ok" if o else "below target"),
                                    "Detail": d if isinstance(d, str) else "see report"} for i, n, g, o, d in rows]),
                     hide_index=True, width="stretch")
        with st.expander("Full report (Markdown)"):
            st.markdown(md)
        st.download_button("Download report", md, file_name=f"REPORT_{mode_used}.md")
        if st.button("Save as evals/REPORT_" + mode_used + ".md"):
            (HERE / "evals" / f"REPORT_{mode_used}.md").write_text(md)
            st.success("Saved.")

# ---------------------------------------------------------------- Audit
with tab_audit:
    st.markdown("Per-run audit: contract (E1), no-leak (E2, scoped to the run), replay-risk from shared "
                "request keys, and secret hygiene. Full E1–E7 stays on the Sign-off tab.")
    name, run = pick_run("audit_run")
    if run and run["results"]:
        calls_by_id = {c["call_id"]: c for c in load_calls(run["dataset"])}
        missing = [r["call_id"] for r in run["results"] if r["call_id"] not in calls_by_id]
        if missing:
            st.warning(f"{len(missing)} result calls have no dataset rows (e.g. {missing[0]}); they are skipped.")
        if st.button("Run audit", type="primary"):
            with st.spinner("Auditing (hashing every request state)…"):
                rep = audit.audit_run(run["results"], [calls_by_id[r["call_id"]] for r in run["results"]
                                                       if r["call_id"] in calls_by_id], run["settings"]["model"])
            st.session_state["audit"] = (name, rep)
    if "audit" in st.session_state:
        aname, rep = st.session_state["audit"]
        st.caption(f"Audit of **{aname}** · {rep['n_calls']} calls "
                   f"({rep['n_calls_with_data']} with dataset rows) · answered by {', '.join(rep['metrics'].get('models', []))}")
        k = st.columns(4)
        cstat = rep["contract"]
        k[0].metric("Contract (E1)", "PASS" if cstat["ok"] else "FAIL", cstat["detail"])
        lstat = rep["no_leak"]
        k[1].metric("No leak (E2)", "PASS" if lstat["ok"] else "FAIL", lstat["detail"])
        dup = rep["duplicates"]
        k[2].metric("Shared request keys", dup["n_groups"], help="Distinct (call, turn) locations with identical request keys. "
                    "With record + parallel workers + a stochastic endpoint, the fixture is last-write-wins and replay serves one value to all sharers.")
        k[3].metric("Secrets in results", len(rep["hygiene"]), help="Pattern hits for key material in the run's results JSON.")
        if not cstat["ok"]:
            with st.expander("Contract problems (first 10)"):
                st.write(cstat["problems"])
        if not lstat["ok"]:
            with st.expander("Leak problems (first 10)"):
                st.write(lstat["problems"])
        if dup["n_groups"]:
            with st.expander(f"{dup['n_groups']} shared request keys (first {len(dup['groups'])} shown)"):
                st.dataframe(pd.DataFrame([{"key": g["key"], "shared by": ", ".join(f"{c}@{w}" for c, w in g["locations"])}
                                           for g in dup["groups"]]), hide_index=True, width="stretch")
        if rep["hygiene"]:
            with st.expander("Secret-hygiene hits"):
                st.write(rep["hygiene"])
        else:
            st.success("No key material in this run's results JSON.")
        st.subheader("Fixtures directory scan")
        fxdir = Path(st.session_state.get("fixtures") or HERE / "fixtures")
        if st.button(f"Scan {fxdir}"):
            files = sorted(fxdir.glob("*.json")) if fxdir.exists() else []
            with st.spinner(f"Scanning {len(files)} fixture files…"):
                hits = audit.scan_artifact_files(files)
            if hits:
                st.error(f"{len(hits)} hygiene hits (first {len(hits)} shown).")
                st.write(hits)
            else:
                st.success(f"{len(files)} fixture files clean.")
        if "signoff" in st.session_state:
            ok, _, _, mode_used = st.session_state["signoff"]
            (st.success if ok else st.error)(f"Last full sign-off (E1–E7): {'PASS' if ok else 'FAIL'} ({mode_used}). See the Sign-off tab.")

# ---------------------------------------------------------------- Playground
with tab_play:
    st.markdown("Send any `state` and typed `questions` in one request.")
    seed = {c["call_id"]: c for c in load_calls("synthetic_calls.json")}["recover-003"]
    default_state = prefix_state(seed, seed["events"][:3], 2)
    c1, c2 = st.columns(2)
    state_txt = c1.text_area("state (JSON)", json.dumps(default_state, indent=1), height=360)
    q_txt = c2.text_area("questions (JSON)", json.dumps(REVIEW, indent=1), height=360)
    if st.button("Decide", disabled=needs_key(cfg_now)):
        try:
            state, qs = json.loads(state_txt), json.loads(q_txt)
            r = decide(state, qs, cfg=session_cfg())
            errs = validate_response(qs, r)
            (st.success if not errs else st.warning)(
                f"{r.get('model')} · {r['_latency_ms']:.0f} ms · ${r['_cost_usd']:.7f} ({r['_cost_source']}) · "
                f"contract {'OK' if not errs else errs}")
            st.json({k: v for k, v in r.items() if not k.startswith("_")})
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
        except JevError as e:
            st.error(str(e))
