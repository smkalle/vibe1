"""Altair charts for the workbench (dataviz reference palette; color follows the approach, never its rank)."""
import altair as alt
import pandas as pd

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
CRITICAL = "#d03b3b"
GRID_GRAY = "#8a8984"
APPROACH_COLOR = {"jev": SERIES[0], "glm": SERIES[1], "rules_b1": SERIES[2], "rules_b2": SERIES[3]}


def _color(names):
    names = list(names)
    return alt.Scale(domain=names, range=[APPROACH_COLOR.get(n, GRID_GRAY) for n in names])


def accuracy_vs_cost(sc: dict, metric: str, metric_label: str):
    """Ranking accuracy (with 95% CI whiskers) against $ per 1,000 calls, plus the Pareto frontier."""
    rows = []
    for name, a in sc["approaches"].items():
        m = a["metrics"][metric]
        if m["point"] is None or a["cost_per_1k"] is None:
            continue
        lo, hi = m["ci"]
        rows.append({"approach": name, "label": a["label"], "cost": a["cost_per_1k"], "acc": m["point"],
                     "lo": lo if lo is not None else m["point"], "hi": hi if hi is not None else m["point"]})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    frontier, best = [], -1.0
    for r in sorted(rows, key=lambda r: (r["cost"], -r["acc"])):
        if r["acc"] > best:
            frontier.append(r)
            best = r["acc"]
    x = alt.X("cost:Q", title="Cost per 1,000 calls (USD)", scale=alt.Scale(zero=True, nice=True))
    y = alt.Y("acc:Q", title=metric_label, scale=alt.Scale(domain=[max(0.0, df["lo"].min() - 0.05), 1.0]))
    tooltip = ["label:N", alt.Tooltip("cost:Q", title="$ / 1k calls", format="$.3f"),
               alt.Tooltip("acc:Q", title=metric_label, format=".3f"),
               alt.Tooltip("lo:Q", title="95% CI low", format=".3f"), alt.Tooltip("hi:Q", title="95% CI high", format=".3f")]
    color = alt.Color("label:N", title=None, legend=alt.Legend(orient="top"),
                      scale=alt.Scale(domain=list(df["label"]), range=[APPROACH_COLOR.get(n, GRID_GRAY) for n in df["approach"]]))
    whisk = alt.Chart(df).mark_rule(strokeWidth=2).encode(x=x, y="lo:Q", y2="hi:Q", color=color)
    pts = alt.Chart(df).mark_point(filled=True, size=140, stroke="white", strokeWidth=2).encode(x=x, y=y, color=color,
                                                                                               tooltip=tooltip)
    layers = [whisk, pts]
    if len(frontier) > 1:
        layers.insert(0, alt.Chart(pd.DataFrame(frontier)).mark_line(strokeDash=[4, 4], color=GRID_GRAY, strokeWidth=1)
                      .encode(x=x, y=y))
    return alt.layer(*layers).properties(height=320)


def latency_vs_budget(sc: dict, budget_ms: float | None):
    rows = []
    for name, a in sc["approaches"].items():
        lat = a["latency"]
        if lat["p50"] is None or lat.get("source") == "offline":
            continue
        over = budget_ms is not None and lat["p95"] > budget_ms
        rows.append({"approach": name, "label": a["label"] + (" ⛔ over budget" if over else ""),
                     "p50": lat["p50"], "p95": lat["p95"], "source": lat["source"]})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    y = alt.Y("label:N", title=None, sort="-x", axis=alt.Axis(labelLimit=220))
    bars = alt.Chart(df).mark_bar(cornerRadiusEnd=4, height=18).encode(
        x=alt.X("p50:Q", title="Latency per decision (ms): bar p50, tick p95"), y=y,
        color=alt.Color("approach:N", scale=_color(df["approach"]), legend=None),
        tooltip=["label:N", alt.Tooltip("p50:Q", format=".0f"), alt.Tooltip("p95:Q", format=".0f"), "source:N"])
    ticks = alt.Chart(df).mark_tick(thickness=2, size=22, color="#0b0b0b").encode(x="p95:Q", y=y)
    layers = [bars, ticks]
    if budget_ms is not None:
        layers.append(alt.Chart(pd.DataFrame({"b": [budget_ms]})).mark_rule(strokeDash=[4, 4], color=CRITICAL)
                      .encode(x="b:Q"))
    return alt.layer(*layers).properties(height=max(90, 44 * len(rows)))


def cost_at_volume(sc: dict, volume: int):
    rows = [{"approach": n, "label": a["label"], "monthly": (a["cost_per_1k"] or 0) * volume / 1000}
            for n, a in sc["approaches"].items() if a["cost_per_1k"] is not None]
    if not rows:
        return None, []
    df = pd.DataFrame(rows)
    chart = alt.Chart(df).mark_bar(cornerRadiusEnd=4, height=18).encode(
        x=alt.X("monthly:Q", title=f"USD per month at {volume:,} calls"),
        y=alt.Y("label:N", title=None, sort="-x", axis=alt.Axis(labelLimit=220)),
        color=alt.Color("approach:N", scale=_color(df["approach"]), legend=None),
        tooltip=["label:N", alt.Tooltip("monthly:Q", format="$,.2f")],
    ).properties(height=max(90, 44 * len(rows)))
    text = chart.mark_text(align="left", dx=6).encode(text=alt.Text("monthly:Q", format="$,.2f"), color=alt.value("#0b0b0b"))
    return chart + text, rows


def forest(sc: dict, metric: str, best: str):
    rows = []
    for key, (mean, lo, hi) in sc["diffs"].get(metric, {}).items():
        a, b = key.split("|")
        if b == best and mean is not None:
            rows.append({"approach": a, "label": sc["approaches"][a]["label"] + " − " + sc["approaches"][best]["label"],
                         "mean": mean, "lo": lo, "hi": hi})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    y = alt.Y("label:N", title=None)
    zero = alt.Chart(pd.DataFrame({"z": [0]})).mark_rule(color=GRID_GRAY, strokeDash=[4, 4]).encode(x="z:Q")
    ci = alt.Chart(df).mark_rule(strokeWidth=2).encode(x=alt.X("lo:Q", title="Paired difference (95% CI)"), x2="hi:Q", y=y,
                                                       color=alt.Color("approach:N", scale=_color(df["approach"]), legend=None))
    pt = alt.Chart(df).mark_point(filled=True, size=90).encode(
        x="mean:Q", y=y, color=alt.Color("approach:N", scale=_color(df["approach"]), legend=None),
        tooltip=["label:N", alt.Tooltip("mean:Q", format="+.3f"), alt.Tooltip("lo:Q", format="+.3f"),
                 alt.Tooltip("hi:Q", format="+.3f")])
    return alt.layer(zero, ci, pt).properties(height=max(90, 40 * len(rows)))


def trajectory_overlay(per_approach: dict, labels: dict):
    """P(book) over the call for each approach on the same call; failed events marked."""
    rows = []
    for name, r in per_approach.items():
        for t in r["turns"]:
            rows.append({"approach": name, "label": labels.get(name, name), "t_s": t["t_ms"] / 1000, "p": t["p_book"],
                         "event": t["event"], "stage": t["stage"], "status": "ok" if t["ok"] else "failed event"})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    x = alt.X("t_s:Q", title="Seconds into call")
    y = alt.Y("p:Q", title="P(book)", scale=alt.Scale(domain=[0, 1]))
    color = alt.Color("label:N", title=None, scale=alt.Scale(domain=[labels.get(n, n) for n in per_approach],
                                                             range=[APPROACH_COLOR.get(n, GRID_GRAY) for n in per_approach]),
                      legend=alt.Legend(orient="top"))
    lines = alt.Chart(df).mark_line(strokeWidth=2).encode(x=x, y=y, color=color)
    pts = alt.Chart(df).mark_point(filled=True, size=70, stroke="white", strokeWidth=1.5).encode(
        x=x, y=y, color=color, shape=alt.Shape("status:N", scale=alt.Scale(domain=["ok", "failed event"],
                                                                           range=["circle", "cross"]),
                                               legend=alt.Legend(title=None, orient="top")),
        tooltip=["label:N", alt.Tooltip("t_s:Q", format=".1f"), "stage:N", "event:N", alt.Tooltip("p:Q", format=".2f"),
                 "status:N"])
    half = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(strokeDash=[4, 4], color=GRID_GRAY).encode(y="y:Q")
    return alt.layer(half, lines, pts).properties(height=300)
