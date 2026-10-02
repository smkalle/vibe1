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
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import baselines
import evaluate
import jev_client
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


def needs_key(cfg: Config) -> bool:
    return cfg.mode in {"live", "record"} and not cfg.api_key


def run_calls(calls, cfg, workers, review, progress):
    """Forecast every call in parallel; one failing call never sinks the run."""
    results, errors = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(forecast_call, c, review, cfg): c["call_id"] for c in calls}
        for n, fut in enumerate(as_completed(futs), 1):
            try:
                results.append(fut.result())
            except Exception as e:  # noqa: BLE001 - surfaced in the UI per call
                errors.append((futs[fut], f"{type(e).__name__}: {e}"))
            progress.progress(n / len(calls), text=f"{n}/{len(calls)} calls")
    order = {c["call_id"]: i for i, c in enumerate(calls)}
    results.sort(key=lambda r: order[r["call_id"]])
    return results, errors


def store_run(name, results, errors, cfg, dataset_file, elapsed):
    st.session_state.setdefault("runs", {})[name] = {
        "results": results,
        "errors": errors,
        "dataset": dataset_file,
        # Settings only; the key is deliberately not stored with the run.
        "settings": {"mode": cfg.mode, "url": cfg.url, "model": cfg.model},
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
    return alt.Chart(df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("point:N", title=None, sort=["mid", "pre_outcome", "pre_outcome_attempted", "end"]),
        xOffset=alt.XOffset("scorer:N", sort=scorers),
        y=alt.Y("AUC:Q", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("scorer:N", sort=scorers, scale=alt.Scale(domain=scorers, range=SERIES[: len(scorers)]),
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

tab_run, tab_results, tab_call, tab_signoff, tab_play = st.tabs(
    ["Run", "Results", "Call explorer", "Sign-off (E1–E7)", "Playground"])

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
    calls = calls_all[:limit]
    est = evaluate.project_cost([], calls)
    with right:
        st.metric("Requests", f"{est['requests'] if review else est['requests'] - len(calls):,}")
        st.metric("Est. input tokens", f"{est['input_tokens']:,}")
        st.metric("Est. cost (USD)", f"${est['usd']:.4f}" if cfg_now.mode in {"live", "record"} else "$0 (offline)")
        cap = st.number_input("Cost cap (USD)", 0.0, 50.0, 0.50, 0.05)

    run_name = st.text_input("Run name", f"{ds_file.removesuffix('.json')}-{cfg_now.mode}-{limit}")
    blocked = needs_key(cfg_now) or (cfg_now.mode in {"live", "record"} and est["usd"] > cap)
    if blocked and not needs_key(cfg_now):
        st.warning("Estimated cost is above the cap.")
    if st.button("Run", type="primary", disabled=blocked):
        cfg = session_cfg()
        t0 = time.time()
        bar = st.progress(0.0, text="starting")
        results, errors = run_calls(calls, cfg, workers, review, bar)
        store_run(run_name, results, errors, cfg, ds_file, time.time() - t0)
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
        store_run(up.name if up else pick, data, [], Config(), guess, 0)
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
