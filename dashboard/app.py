"""EvalPulse Gradio Dashboard — fully wired 4-tab monitoring interface."""

from __future__ import annotations

import csv
import tempfile
from collections import defaultdict

import gradio as gr
import plotly.graph_objects as go

from dashboard.charts import (
    empty_figure,
    health_gauge_chart,
    radar_chart,
)

# ── Plotly dark theme base ────────────────────────────────────────────

_DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono, monospace", color="#94a3b8", size=11),
    autosize=True,
    margin=dict(l=50, r=20, t=44, b=40),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.04)",
        tickfont=dict(size=10, color="#475569"),
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.04)",
        tickfont=dict(size=10, color="#475569"),
    ),
    legend=dict(font=dict(size=10, color="#64748b"), bgcolor="rgba(0,0,0,0)"),
)


def _dark(fig: go.Figure, **kw) -> go.Figure:
    """Apply dark theme to a Plotly figure."""
    layout = {**_DARK_LAYOUT, **kw}
    fig.update_layout(**layout)
    return fig


# ── Data fetching ─────────────────────────────────────────────────────


def _get_store():
    try:
        from evalpulse.storage.sqlite_store import SQLiteStore

        return SQLiteStore("evalpulse.db")
    except Exception:
        return None


def _fetch_records(limit: int = 500):
    store = _get_store()
    if store is None:
        return []
    try:
        return store.get_latest(limit)
    except Exception:
        return []
    finally:
        store.close()


def _fetch_alerts(limit: int = 20):
    try:
        from evalpulse.alerts import AlertEngine

        engine = AlertEngine()
        return engine.get_recent_alerts(limit)
    except Exception:
        return []


# ── Tab 1: Overview ───────────────────────────────────────────────────


def build_overview():
    records = _fetch_records(500)
    alerts = _fetch_alerts(20)

    if not records:
        return (
            _kpi_card("Health Score", "—", "no data", "#06d6a0"),
            _kpi_card("Hallucination", "—", "no data", "#f59e0b"),
            _kpi_card("Drift", "—", "no data", "#3b82f6"),
            _kpi_card("Evaluations", "0", "", "#a78bfa"),
            health_gauge_chart(None),
            empty_figure("", "No evaluations yet"),
            [["No alerts yet", "", "", "", "", ""]],
        )

    avg_health = int(sum(r.health_score for r in records) / len(records))
    avg_halluc = sum(r.hallucination_score for r in records) / len(records)
    drift_vals = [r.drift_score for r in records if r.drift_score is not None]
    avg_drift = sum(drift_vals) / len(drift_vals) if drift_vals else None

    if avg_health >= 90:
        h_sub = "HEALTHY"
    elif avg_health >= 75:
        h_sub = "MONITORING"
    elif avg_health >= 60:
        h_sub = "DEGRADING"
    else:
        h_sub = "CRITICAL"

    d_val = f"{avg_drift:.3f}" if avg_drift is not None else "..."
    d_sub = (
        "STABLE"
        if avg_drift is not None and avg_drift < 0.15
        else "DRIFTING"
        if avg_drift is not None
        else "BUILDING BASELINE"
    )

    sorted_recs = sorted(records, key=lambda r: r.timestamp)
    times = [r.timestamp.strftime("%m-%d %H:%M") for r in sorted_recs]
    scores = [r.health_score for r in sorted_recs]

    min_score = max(0, min(scores) - 10)
    trend = go.Figure()
    trend.add_trace(
        go.Scatter(
            x=times,
            y=scores,
            mode="lines",
            name="Health Score",
            line=dict(color="#06d6a0", width=2, shape="spline"),
        )
    )
    if min_score <= 75:
        trend.add_hline(y=75, line_dash="dot", line_color="#f59e0b", line_width=1)
    if min_score <= 40:
        trend.add_hline(y=40, line_dash="dot", line_color="#ef4444", line_width=1)
    _dark(trend, title="Health Score Trend", yaxis=dict(range=[min_score, 105], **_DARK_LAYOUT["yaxis"]), height=350)

    alert_rows = [["—", "", "", "", "", "No alerts triggered"]]
    if alerts:
        alert_rows = []
        for a in alerts[:20]:
            alert_rows.append([
                a.timestamp.strftime("%Y-%m-%d %H:%M"),
                a.severity.upper(),
                a.metric,
                f"{a.value:.4f}",
                f"{a.threshold:.4f}",
                a.message,
            ])

    return (
        _kpi_card("Health Score", str(avg_health), h_sub, "#06d6a0"),
        _kpi_card("Hallucination", f"{avg_halluc:.1%}", f"avg of {len(records)}", "#f59e0b"),
        _kpi_card("Drift", d_val, d_sub, "#3b82f6"),
        _kpi_card("Evaluations", f"{len(records):,}", "total tracked", "#a78bfa"),
        health_gauge_chart(avg_health),
        trend,
        alert_rows,
    )


# ── Tab 2: Hallucination ─────────────────────────────────────────────


def build_hallucination():
    records = _fetch_records(500)
    if not records:
        e = empty_figure("", "No data yet")
        return e, e, e, [["No data", "", "", "", ""]]

    sorted_recs = sorted(records, key=lambda r: r.timestamp)
    times = [r.timestamp.strftime("%m-%d %H:%M") for r in sorted_recs]
    h_scores = [r.hallucination_score for r in sorted_recs]

    rate = go.Figure()
    rate.add_trace(
        go.Scatter(
            x=times,
            y=h_scores,
            mode="lines",
            line=dict(color="#ef4444", width=2, shape="spline"),
            fill="tozeroy",
            fillcolor="rgba(239,68,68,0.08)",
        )
    )
    rate.add_hline(
        y=0.3, line_dash="dot", line_color="#f59e0b",
        annotation_text="Threshold 0.3",
        annotation_font_size=9, annotation_font_color="#f59e0b",
    )
    _dark(rate, title="Hallucination Score Over Time", yaxis=dict(range=[0, 1.05], **_DARK_LAYOUT["yaxis"]), height=350)

    dist = go.Figure(
        go.Histogram(x=h_scores, nbinsx=25, marker_color="#ef4444", opacity=0.7, marker_line_width=0)
    )
    dist.add_vline(x=0.3, line_dash="dot", line_color="#f59e0b")
    _dark(dist, title="Score Distribution", height=300, bargap=0.05)

    ms = defaultdict(list)
    for r in records:
        ms[r.model_name].append(r.hallucination_score)
    models = list(ms.keys())
    avgs = [sum(v) / len(v) for v in ms.values()]
    model = go.Figure(
        go.Bar(
            x=models,
            y=avgs,
            marker_color=["#ef4444" if a > 0.3 else "#06d6a0" for a in avgs],
            marker_line_width=0,
        )
    )
    _dark(model, title="Avg Hallucination by Model", height=300)

    top = sorted(records, key=lambda r: r.hallucination_score, reverse=True)[:10]
    rows = [
        [
            r.timestamp.strftime("%H:%M:%S"),
            r.query[:50],
            r.response[:60],
            f"{r.hallucination_score:.3f}",
            ", ".join(r.flagged_claims[:2]) if r.flagged_claims else "",
        ]
        for r in top
    ]

    return rate, dist, model, rows


# ── Tab 3: Drift ─────────────────────────────────────────────────────


def build_drift():
    records = _fetch_records(500)
    if not records:
        e = empty_figure("", "No data yet")
        return e, e, "No data"

    sorted_recs = sorted(records, key=lambda r: r.timestamp)
    drift_recs = [r for r in sorted_recs if r.drift_score is not None]

    emb_recs = [
        r for r in sorted_recs if r.embedding_vector and len(r.embedding_vector) > 2
    ]
    if len(emb_recs) >= 3:
        embed = go.Figure(
            go.Scatter(
                x=[r.embedding_vector[0] for r in emb_recs],
                y=[r.embedding_vector[1] for r in emb_recs],
                mode="markers",
                marker=dict(
                    size=8,
                    color=[r.hallucination_score for r in emb_recs],
                    colorscale=[[0, "#06d6a0"], [0.5, "#f59e0b"], [1, "#ef4444"]],
                    showscale=True,
                    colorbar=dict(
                        title="Halluc",
                        tickfont=dict(size=9, color="#64748b"),
                        titlefont=dict(size=10, color="#64748b"),
                    ),
                    line=dict(width=0),
                ),
                text=[r.query[:30] for r in emb_recs],
                hovertemplate="%{text}<br>Halluc: %{marker.color:.3f}<extra></extra>",
            )
        )
        _dark(embed, title="Response Embedding Space", height=350)
    else:
        embed = empty_figure("", "Need more data for visualization")

    if not drift_recs:
        return (
            empty_figure("", "Building baseline (need 10+ evaluations)"),
            embed,
            "Building baseline...",
        )

    times = [r.timestamp.strftime("%m-%d %H:%M") for r in drift_recs]
    scores = [r.drift_score for r in drift_recs]

    dfig = go.Figure()
    dfig.add_trace(
        go.Scatter(
            x=times,
            y=scores,
            mode="lines",
            line=dict(color="#a78bfa", width=2, shape="spline"),
            fill="tozeroy",
            fillcolor="rgba(167,139,250,0.08)",
        )
    )
    dfig.add_hline(
        y=0.15, line_dash="dot", line_color="#ef4444",
        annotation_text="Threshold 0.15",
        annotation_font_size=9, annotation_font_color="#ef4444",
    )
    y_max = max(max(scores) * 1.2, 0.3)
    _dark(dfig, title="Drift Score Over Time", yaxis=dict(range=[0, y_max], **_DARK_LAYOUT["yaxis"]), height=350)

    avg = sum(scores) / len(scores)
    if avg < 0.1:
        st = "Stable"
    elif avg < 0.2:
        st = "Minor drift"
    else:
        st = "Significant drift!"

    return dfig, embed, st


# ── Tab 4: RAG & Quality ─────────────────────────────────────────────


def build_rag_quality():
    records = _fetch_records(500)
    if not records:
        e = empty_figure("", "No data yet")
        return e, e, e, e

    sorted_recs = sorted(records, key=lambda r: r.timestamp)
    times = [r.timestamp.strftime("%m-%d %H:%M") for r in sorted_recs]

    qfig = go.Figure()
    qfig.add_trace(
        go.Scatter(
            x=times,
            y=[r.sentiment_score for r in sorted_recs],
            mode="lines",
            name="Sentiment",
            line=dict(color="#3b82f6", width=2, shape="spline"),
        )
    )
    qfig.add_trace(
        go.Scatter(
            x=times,
            y=[r.toxicity_score for r in sorted_recs],
            mode="lines",
            name="Toxicity",
            line=dict(color="#ef4444", width=2, shape="spline"),
        )
    )
    _dark(qfig, title="Quality Metrics Over Time", yaxis=dict(range=[0, 1.05], **_DARK_LAYOUT["yaxis"]), height=350)

    rag_recs = [r for r in sorted_recs if r.groundedness_score is not None]
    if rag_recs:
        rt = [r.timestamp.strftime("%m-%d %H:%M") for r in rag_recs]
        rfig = go.Figure()
        rfig.add_trace(
            go.Scatter(
                x=rt, y=[r.faithfulness_score or 0 for r in rag_recs],
                mode="lines", name="Faithfulness",
                line=dict(color="#06d6a0", width=2, shape="spline"),
            )
        )
        rfig.add_trace(
            go.Scatter(
                x=rt, y=[r.context_relevance or 0 for r in rag_recs],
                mode="lines", name="Context Relevance",
                line=dict(color="#3b82f6", width=2, shape="spline"),
            )
        )
        rfig.add_trace(
            go.Scatter(
                x=rt, y=[r.groundedness_score or 0 for r in rag_recs],
                mode="lines", name="Groundedness",
                line=dict(color="#a78bfa", width=2, dash="dash"),
            )
        )
        _dark(rfig, title="RAG Quality Metrics", yaxis=dict(range=[0, 1.05], **_DARK_LAYOUT["yaxis"]), height=350)

        af = sum(r.faithfulness_score or 0 for r in rag_recs) / len(rag_recs)
        ac = sum(r.context_relevance or 0 for r in rag_recs) / len(rag_recs)
        aa = sum(r.answer_relevancy or 0 for r in rag_recs) / len(rag_recs)
        ag = sum(r.groundedness_score or 0 for r in rag_recs) / len(rag_recs)
        radar = radar_chart(
            ["Faithfulness", "Context Relevance", "Answer Relevancy", "Groundedness"],
            [af, ac, aa, ag],
            title="RAG Quality Radar",
        )
    else:
        rfig = empty_figure("", "No RAG calls yet")
        radar = empty_figure("", "No RAG data")

    lang = defaultdict(int)
    denials = 0
    for r in records:
        lang[r.language_detected] += 1
        if r.is_denial:
            denials += 1
    bfig = go.Figure(
        go.Bar(
            x=list(lang.keys()),
            y=list(lang.values()),
            marker_color="#3b82f6",
            marker_line_width=0,
        )
    )
    _dark(
        bfig,
        title=f"Language Distribution  |  Denials: {denials}/{len(records)}",
        height=300,
    )

    return qfig, rfig, radar, bfig


def export_csv():
    records = _fetch_records(1000)
    if not records:
        return gr.update(visible=False)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline=""
    )
    fields = [
        "id", "app_name", "timestamp", "query", "response", "model_name",
        "latency_ms", "health_score", "hallucination_score", "drift_score",
        "faithfulness_score", "context_relevance", "answer_relevancy",
        "groundedness_score", "sentiment_score", "toxicity_score",
        "response_length", "language_detected", "is_denial",
    ]
    writer = csv.DictWriter(tmp, fieldnames=fields)
    writer.writeheader()
    for r in records:
        row = r.model_dump()
        row["timestamp"] = str(row["timestamp"])
        writer.writerow({k: row.get(k, "") for k in fields})
    tmp.close()
    return gr.update(value=tmp.name, visible=True)


# ── KPI card HTML helper ──────────────────────────────────────────────


def _kpi_card(label: str, value: str, sub: str, color: str) -> str:
    return f"""<div style="
        background:linear-gradient(145deg,#111827,#0f172a);
        border:1px solid #1e293b;
        border-radius:14px;
        padding:18px 20px;
        border-top:2.5px solid {color};
        min-height:90px;
        min-width:0;
        width:100%;
        box-sizing:border-box;
        overflow:hidden;
    ">
        <div style="
            font-family:'JetBrains Mono',monospace;
            font-size:0.62em;font-weight:600;
            text-transform:uppercase;letter-spacing:1.5px;
            color:#64748b;margin-bottom:8px;
        ">{label}</div>
        <div style="
            font-family:'Outfit',sans-serif;
            font-size:1.8em;font-weight:700;
            color:{color};line-height:1;margin-bottom:5px;
        ">{value}</div>
        <div style="
            font-family:'JetBrains Mono',monospace;
            font-size:0.68em;color:#475569;
        ">{sub}</div>
    </div>"""


# ── CSS ───────────────────────────────────────────────────────────────

THEME_CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Outfit:wght@300;400;500;600;700;800&display=swap');

body, .gradio-container {
    background: #060a14 !important;
    color: #e2e8f0 !important;
    font-family: 'Outfit', sans-serif !important;
}
.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 20px !important;
    box-sizing: border-box !important;
    overflow-x: hidden !important;
}
.main, .wrap, .contain {
    max-width: 100% !important;
    width: 100% !important;
    overflow-x: hidden !important;
}
.app {
    max-width: 100% !important;
    overflow-x: hidden !important;
}
/* Plotly charts should not overflow */
.js-plotly-plot, .plotly, .plot-container, .svg-container {
    max-width: 100% !important;
    width: 100% !important;
    overflow: hidden !important;
}
.js-plotly-plot .main-svg, .js-plotly-plot .svg-container {
    max-width: 100% !important;
    width: 100% !important;
}
.plot-container.plotly {
    width: 100% !important;
}
/* Gradio plot wrapper */
.gr-plot, .plot-padding {
    max-width: 100% !important;
    overflow: hidden !important;
}

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }

.ep-hdr {
    position: relative;
    padding: 24px 32px;
    margin: 0 -20px 20px -20px;
    background: linear-gradient(135deg, #0a0e1a 0%, #111827 50%, #0f172a 100%);
    border-bottom: 1px solid rgba(6,214,160,0.15);
    overflow: hidden;
    box-sizing: border-box;
}
.ep-hdr::before {
    content:'';position:absolute;inset:0;
    background:
        radial-gradient(ellipse 600px 300px at 15% 50%,rgba(6,214,160,0.06),transparent 70%),
        radial-gradient(ellipse 400px 200px at 85% 30%,rgba(59,130,246,0.04),transparent 70%);
    pointer-events:none;
}
.ep-hdr-in { position:relative;display:flex;align-items:center;justify-content:space-between;z-index:1; }
.ep-brand { display:flex;align-items:center;gap:14px; }
.ep-logo {
    width:40px;height:40px;border-radius:10px;
    background:linear-gradient(135deg,#06d6a0,#3b82f6);
    display:flex;align-items:center;justify-content:center;
    font-size:18px;font-weight:700;color:#060a14;
    font-family:'JetBrains Mono',monospace;
    box-shadow:0 0 20px rgba(6,214,160,0.3);
}
.ep-t { font-family:'Outfit';font-size:1.6em;font-weight:700;letter-spacing:-0.5px;color:#f1f5f9!important;margin:0!important; }
.ep-st { font-family:'JetBrains Mono';font-size:0.7em;color:#64748b!important;margin:3px 0 0!important;letter-spacing:0.5px;text-transform:uppercase; }
.ep-live { display:flex;align-items:center;gap:8px;font-family:'JetBrains Mono';font-size:0.72em;color:#06d6a0;letter-spacing:0.3px; }
.ep-dot {
    width:7px;height:7px;border-radius:50%;background:#06d6a0;
    box-shadow:0 0 8px rgba(6,214,160,0.6);
    animation:pdot 2s ease-in-out infinite;
}
@keyframes pdot { 0%,100%{opacity:1} 50%{opacity:0.4} }

.tab-nav { background:transparent!important;border:none!important;gap:4px!important;padding:0 0 14px!important;border-bottom:1px solid #1e293b!important;margin-bottom:18px!important; }
.tab-nav button {
    font-family:'JetBrains Mono',monospace!important;font-size:0.76em!important;font-weight:500!important;
    letter-spacing:0.5px!important;text-transform:uppercase!important;color:#64748b!important;
    background:transparent!important;border:1px solid transparent!important;border-radius:8px!important;
    padding:8px 18px!important;transition:all 0.2s!important;
}
.tab-nav button:hover { color:#e2e8f0!important;background:rgba(255,255,255,0.03)!important; }
.tab-nav button.selected { color:#06d6a0!important;background:rgba(6,214,160,0.08)!important;border-color:rgba(6,214,160,0.2)!important; }
.tabitem { border:none!important;background:transparent!important;padding:0!important; }

table { background:#111827!important;border:1px solid #1e293b!important;border-radius:10px!important;overflow:hidden!important; }
table thead th {
    background:#0f172a!important;color:#64748b!important;
    font-family:'JetBrains Mono',monospace!important;font-size:0.7em!important;
    font-weight:600!important;letter-spacing:0.8px!important;text-transform:uppercase!important;
    padding:10px 14px!important;border-bottom:1px solid #1e293b!important;
}
table tbody td {
    background:#111827!important;color:#cbd5e1!important;
    font-family:'JetBrains Mono',monospace!important;font-size:0.78em!important;
    padding:8px 14px!important;border-bottom:1px solid rgba(30,41,59,0.5)!important;
}
table tbody tr:hover td { background:rgba(6,214,160,0.03)!important; }

button.primary, button.secondary {
    font-family:'JetBrains Mono',monospace!important;font-size:0.74em!important;
    letter-spacing:0.4px!important;border-radius:8px!important;
}
button.primary { background:rgba(6,214,160,0.12)!important;color:#06d6a0!important;border:1px solid rgba(6,214,160,0.25)!important; }
button.primary:hover { background:rgba(6,214,160,0.2)!important; }
button.secondary { background:rgba(59,130,246,0.1)!important;color:#3b82f6!important;border:1px solid rgba(59,130,246,0.2)!important; }
button.secondary:hover { background:rgba(59,130,246,0.18)!important; }

.gr-row {
    gap:14px!important;
    flex-wrap: wrap !important;
    max-width: 100% !important;
    overflow: hidden !important;
}
/* Remove all white backgrounds from Gradio components */
.gr-block, .block:not(.gr-group) { border:none!important;background:transparent!important; }
.gr-padded { padding:0!important; }
.label-wrap { background:#0a0e1a!important;border:1px solid #1e293b!important;border-radius:8px!important;padding:4px 10px!important; }
.label-wrap span { color:#64748b!important;font-family:'JetBrains Mono',monospace!important;font-size:0.72em!important;letter-spacing:0.5px!important; }
/* Plot containers */
.gr-plot, .plot-wrap, .gradio-plot { background:transparent!important;border:none!important; }
div[class*="plot"] { background:transparent!important; }
/* All panel/group/box backgrounds */
.panel, .gr-panel, .gr-box, .gr-form, .gr-input-label, .gr-check-radio { background:#111827!important;border-color:#1e293b!important;color:#e2e8f0!important; }
/* File download component */
.file-preview, .upload-button { background:#111827!important;border-color:#1e293b!important;color:#94a3b8!important; }
/* Inputs and textboxes */
input, textarea, select, .gr-input { background:#111827!important;border-color:#1e293b!important;color:#e2e8f0!important; }
/* Any remaining white wrapper divs */
.contain > div, .wrap > div { background:transparent!important; }
/* Markdown text areas */
.prose, .markdown-text, .md { background:transparent!important;color:#94a3b8!important; }
/* Accordion headers */
.accordion { background:#111827!important;border-color:#1e293b!important; }
/* Prevent dataframes from causing horizontal scroll */
.dataframe, .table-wrap, .svelte-table {
    max-width: 100% !important;
    overflow-x: auto !important;
    overflow-y: hidden !important;
}
/* KPI card row in HTML shouldn't overflow */
div[style*="display:flex"] {
    flex-wrap: wrap !important;
    max-width: 100% !important;
}

.ep-ftr {
    margin-top:28px;padding:14px 0;border-top:1px solid #1e293b;
    text-align:center;font-family:'JetBrains Mono',monospace;
    font-size:0.68em;color:#334155;letter-spacing:0.3px;
}
.ep-ftr a { color:#475569;text-decoration:none; }
.ep-ftr a:hover { color:#06d6a0; }

.markdown-text h4 { color:#94a3b8!important;font-family:'Outfit',sans-serif!important; }
.markdown-text p, .markdown-text { color:#94a3b8!important; }

@media(max-width:768px) { .ep-hdr-in{flex-direction:column;gap:10px;align-items:flex-start;} }
"""


# ── App ───────────────────────────────────────────────────────────────


def create_app() -> gr.Blocks:
    with gr.Blocks(title="EvalPulse Dashboard", css=THEME_CSS) as app:
        gr.HTML("""
        <div class="ep-hdr"><div class="ep-hdr-in">
            <div class="ep-brand">
                <div class="ep-logo">EP</div>
                <div><div class="ep-t">EvalPulse</div>
                <div class="ep-st">LLM Evaluation &amp; Drift Monitor</div></div>
            </div>
            <div class="ep-live"><div class="ep-dot"></div>MONITORING ACTIVE</div>
        </div></div>
        """)

        with gr.Tabs():
            with gr.TabItem("Overview"):
                with gr.Row():
                    hc = gr.HTML("Loading...")
                    hac = gr.HTML("Loading...")
                    dc = gr.HTML("Loading...")
                    tc = gr.HTML("Loading...")
                with gr.Row():
                    hg = gr.Plot(label="Health Gauge")
                    ht = gr.Plot(label="Health Trend")
                gr.Markdown("#### Recent Alerts")
                at = gr.Dataframe(
                    headers=["Time", "Severity", "Metric", "Value", "Threshold", "Message"],
                    interactive=False,
                )
                gr.Button("Refresh", variant="primary", size="sm").click(
                    fn=build_overview, outputs=[hc, hac, dc, tc, hg, ht, at]
                )

            with gr.TabItem("Hallucination"):
                hr = gr.Plot()
                with gr.Row():
                    hd = gr.Plot()
                    hm = gr.Plot()
                gr.Markdown("#### Highest Hallucination Responses")
                htb = gr.Dataframe(
                    headers=["Time", "Query", "Response", "Score", "Flagged"],
                    interactive=False,
                )
                gr.Button("Refresh", variant="primary", size="sm").click(
                    fn=build_hallucination, outputs=[hr, hd, hm, htb]
                )

            with gr.TabItem("Semantic Drift"):
                ds = gr.Markdown("Loading...")
                dp = gr.Plot()
                de = gr.Plot()
                gr.Button("Refresh", variant="primary", size="sm").click(
                    fn=build_drift, outputs=[dp, de, ds]
                )

            with gr.TabItem("RAG & Quality"):
                qp = gr.Plot()
                with gr.Row():
                    rp = gr.Plot()
                    rr = gr.Plot()
                bp = gr.Plot()
                with gr.Row():
                    gr.Button("Refresh", variant="primary", size="sm").click(
                        fn=build_rag_quality, outputs=[qp, rp, rr, bp]
                    )
                    ef = gr.File(label="Download", visible=False)
                    gr.Button("Export CSV", variant="secondary", size="sm").click(
                        fn=export_csv, outputs=ef
                    )

        gr.HTML("""
        <div class="ep-ftr">
            EvalPulse v0.1.0 &middot; Open Source LLM Evaluation &amp; Drift Monitoring
            &middot; <a href="https://github.com/ninjacode911/evalpulse">GitHub</a>
        </div>
        """)

        app.load(fn=build_overview, outputs=[hc, hac, dc, tc, hg, ht, at])

    return app


if __name__ == "__main__":
    create_app().launch(server_name="0.0.0.0", server_port=7860)
