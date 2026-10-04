"""Technical tools behind the workflow: Audit, Lab sign-off (E1–E7), Playground (ported from the v1 app)."""
import json

import pandas as pd
import streamlit as st

import audit
import llm_client
import run_evals
import workspace
from jev_client import JevError, decide, validate_response
from questions import REVIEW, prefix_state
from simulate import forecast_call



def audit_tool(project, sample):
    st.markdown("Per-run audit: contract (E1), no-leak (E2), shared request keys (replay risk) and secret hygiene.")
    runs = project.runs(sample["sample_id"]) if sample else []
    if not runs:
        st.info("No runs for the current sample yet.")
        return
    rid = st.selectbox("Run", [m["run_id"] for m in runs], key="audit_run")
    m = project.manifest(rid)
    corpus = {c["call_id"]: c for c in workspace.load_corpus(m["corpus_id"])}
    if st.button("Run audit", type="primary", key="audit_btn"):
        res = project.results(rid)
        with st.spinner("Auditing…"):
            rep = audit.audit_run(res, [corpus[r["call_id"]] for r in res if r["call_id"] in corpus], m.get("model"))
            hits = audit.scan_artifact_files([p for p in project.dir.rglob("*") if p.is_file()])
        k = st.columns(4)
        k[0].metric("Contract (E1)", "PASS" if rep["contract"]["ok"] else "FAIL", rep["contract"]["detail"])
        k[1].metric("No leak (E2)", "PASS" if rep["no_leak"]["ok"] else "FAIL", rep["no_leak"]["detail"])
        k[2].metric("Shared request keys", rep["duplicates"]["n_groups"])
        k[3].metric("Secrets in workspace", len(hits))
        if hits:
            st.error(hits)


def signoff_tool(jev_cfg, needs_key):
    st.markdown("The v1 lab sign-off: **every synthetic call + the SGD test split** with Jev (about 2,900 requests), "
                "then evals E1–E7. GLM runs are gated per run in step ⑥.")
    if st.button("Run lab sign-off", type="primary", disabled=needs_key, key="signoff_btn"):
        syn, test = workspace.load_corpus("synthetic"), workspace.load_corpus("sgd_test")
        bar = st.progress(0.0, text="scoring")
        out = []
        for i, calls in enumerate((syn, test)):
            res = []
            for j, c in enumerate(calls):
                res.append(forecast_call(c, True, jev_cfg))
                bar.progress((i + (j + 1) / len(calls)) / 2)
            out.append(res)
        ok, md, rows = run_evals.build_report(out[0], out[1], live=jev_cfg.mode != "mock", mode_label=jev_cfg.mode)
        st.session_state["signoff"] = (ok, md, rows, jev_cfg.mode)
    if "signoff" in st.session_state:
        ok, md, rows, mode_used = st.session_state["signoff"]
        (st.success if ok else st.error)(f"Overall: {'PASS' if ok else 'FAIL'} ({mode_used})")
        st.dataframe(pd.DataFrame([{"ID": i, "Eval": n, "Gate": "gate" if g else "report",
                                    "Result": ("PASS" if o else "FAIL") if g else ("ok" if o else "below target")}
                                   for i, n, g, o, d in rows]), hide_index=True, width="stretch")
        st.download_button("Download report", md, file_name=f"REPORT_{mode_used}.md")


def playground_tool(jev_cfg, glm_cfg, needs_key):
    st.markdown("Send any `state` and typed `questions` in one request, to either approach.")
    seed = {c["call_id"]: c for c in workspace.load_corpus("synthetic")}["recover-003"]
    c1, c2 = st.columns(2)
    state_txt = c1.text_area("state (JSON)", json.dumps(prefix_state(seed, seed["events"][:3], 2), indent=1), height=320)
    q_txt = c2.text_area("questions (JSON)", json.dumps(REVIEW, indent=1), height=320)
    who = st.radio("Approach", ["Jev", "GLM"], horizontal=True, key="play_who")
    if st.button("Decide", disabled=needs_key, key="play_btn"):
        try:
            state, qs = json.loads(state_txt), json.loads(q_txt)
            r = decide(state, qs, cfg=jev_cfg) if who == "Jev" else llm_client.decide(state, qs, cfg=glm_cfg)
            errs = validate_response(qs, r)
            (st.success if not errs else st.warning)(
                f"{r.get('model')} · {r['_latency_ms']:.0f} ms ({r.get('_latency_source')}) · ${r['_cost_usd']:.7f} · "
                f"contract {'OK' if not errs else errs}")
            st.json({k: v for k, v in r.items() if not k.startswith("_")})
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
        except JevError as e:
            st.error(str(e))
