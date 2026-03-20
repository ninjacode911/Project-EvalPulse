"""EvalPulse Demo Dashboard — self-contained HuggingFace Spaces deployment.

Runs entirely on synthetic data. No external dependencies on evalpulse or
dashboard packages.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import gradio as gr
import plotly.graph_objects as go

# ── Lightweight EvalRecord (replaces pydantic model) ─────────────────


@dataclass
class EvalRecord:
    """Minimal evaluation record for demo purposes."""

    app_name: str = "default"
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    query: str = ""
    context: str | None = None
    response: str = ""
    model_name: str = "unknown"
    latency_ms: int = 0
    tags: list[str] = field(default_factory=list)

    # Hallucination
    hallucination_score: float = 0.0
    hallucination_method: str = "none"
    flagged_claims: list[str] = field(default_factory=list)

    # Drift
    embedding_vector: list[float] = field(default_factory=list)
    drift_score: float | None = None

    # RAG Quality
    faithfulness_score: float | None = None
    context_relevance: float | None = None
    answer_relevancy: float | None = None
    groundedness_score: float | None = None

    # Response Quality
    sentiment_score: float = 0.5
    toxicity_score: float = 0.0
    response_length: int = 0
    language_detected: str = "en"
    is_denial: bool = False

    # Composite
    health_score: int = 0


# ── Demo data generator ─────────────────────────────────────────────


def generate_demo_records(n: int = 200) -> list[EvalRecord]:
    """Generate N synthetic EvalRecords with realistic distributions.

    Simulates an LLM app with:
    - Generally good performance (health 70-95)
    - Occasional hallucination spikes
    - Gradual drift over time
    - Some toxic/denial responses
    """
    random.seed(42)
    records: list[EvalRecord] = []
    now = datetime.now(UTC)

    queries = [
        "What is machine learning?",
        "Explain neural networks",
        "How does RAG work?",
        "What is Python used for?",
        "Describe transformer architecture",
        "What are embeddings?",
        "How do LLMs handle context?",
        "What is fine-tuning?",
        "Explain attention mechanism",
        "What is prompt engineering?",
    ]

    models = ["llama-3.1-70b", "gpt-4o-mini", "gemini-flash"]

    for i in range(n):
        ts = now - timedelta(hours=n - i)
        query = random.choice(queries)
        model = random.choice(models)

        # Simulate drift: later responses drift slightly
        drift_factor = i / n * 0.1

        # Base scores
        halluc = random.gauss(0.12, 0.08) + drift_factor * 0.5
        halluc = max(0.0, min(1.0, halluc))

        drift = random.gauss(0.05, 0.03) + drift_factor
        drift = max(0.0, min(1.0, drift))

        sentiment = random.gauss(0.7, 0.1)
        sentiment = max(0.0, min(1.0, sentiment))

        toxicity = abs(random.gauss(0.02, 0.02))
        toxicity = max(0.0, min(1.0, toxicity))

        is_denial = random.random() < 0.05
        length = random.randint(20, 200)

        # RAG scores (70% of calls are RAG)
        is_rag = random.random() < 0.7
        faith = None
        ctx_rel = None
        ans_rel = None
        ground = None
        context = None

        if is_rag:
            faith = random.gauss(0.75, 0.1)
            faith = max(0.0, min(1.0, faith))
            ctx_rel = random.gauss(0.8, 0.08)
            ctx_rel = max(0.0, min(1.0, ctx_rel))
            ans_rel = random.gauss(0.78, 0.09)
            ans_rel = max(0.0, min(1.0, ans_rel))
            ground = 0.4 * faith + 0.3 * ctx_rel + 0.3 * ans_rel
            context = f"Context for: {query}"

        # Compute health score
        components = [(1 - halluc) * 0.35, (1 - drift) * 0.25]
        if ground is not None:
            components.append(ground * 0.20)
        quality = (1 - toxicity) * 0.5 + sentiment * 0.4 + 0.1
        components.append(quality * 0.15)
        health = int(
            sum(components) / sum([0.35, 0.25] + ([0.20] if ground else []) + [0.15]) * 100
        )
        health = max(0, min(100, health))

        record = EvalRecord(
            app_name="demo-app",
            timestamp=ts,
            query=query,
            context=context,
            response=f"Demo response for: {query}",
            model_name=model,
            latency_ms=random.randint(50, 500),
            tags=["demo"],
            hallucination_score=round(halluc, 4),
            hallucination_method="embedding",
            drift_score=round(drift, 4),
            faithfulness_score=round(faith, 4) if faith else None,
            context_relevance=round(ctx_rel, 4) if ctx_rel else None,
            answer_relevancy=round(ans_rel, 4) if ans_rel else None,
            groundedness_score=round(ground, 4) if ground else None,
            sentiment_score=round(sentiment, 4),
            toxicity_score=round(toxicity, 4),
            response_length=length,
            language_detected="en",
            is_denial=is_denial,
            health_score=health,
        )
        records.append(record)

    return records


# ── Chart helpers (inlined from dashboard/charts.py) ─────────────────

_BG = "#0a0e1a"
_SURFACE = "#111827"
_BORDER = "#1e293b"
_TEXT = "#e2e8f0"
_TEXT_DIM = "#64748b"
_CYAN = "#06d6a0"
_AMBER = "#f59e0b"
_RED = "#ef4444"
_BLUE = "#3b82f6"
_PURPLE = "#a78bfa"
_PINK = "#f472b6"

_LAYOUT_BASE: dict = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono, monospace", color=_TEXT, size=11),
    margin=dict(l=48, r=24, t=48, b=40),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.04)",
        zerolinecolor="rgba(255,255,255,0.06)",
        tickfont=dict(size=10, color=_TEXT_DIM),
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.04)",
        zerolinecolor="rgba(255,255,255,0.06)",
        tickfont=dict(size=10, color=_TEXT_DIM),
    ),
    legend=dict(
        font=dict(size=10, color=_TEXT_DIM),
        bgcolor="rgba(0,0,0,0)",
    ),
)


def _apply_layout(fig: go.Figure, height: int = 320, **kwargs) -> go.Figure:
    layout = {**_LAYOUT_BASE, "height": height}
    layout.update(kwargs)
    fig.update_layout(**layout)
    return fig


def empty_figure(title: str = "", message: str = "No data available") -> go.Figure:
    """Create an empty figure with a message."""
    fig = go.Figure()
    _apply_layout(
        fig,
        height=260,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[
            dict(
                text=f"<i>{message}</i>",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=13, color=_TEXT_DIM),
            )
        ],
    )
    return fig


def health_gauge_chart(score: int | None = None) -> go.Figure:
    """Create a health score gauge chart (0-100)."""
    if score is None:
        return empty_figure("", "Awaiting first evaluation")

    if score >= 75:
        bar_color = _CYAN
    elif score >= 40:
        bar_color = _AMBER
    else:
        bar_color = _RED

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number=dict(
                font=dict(size=48, color=bar_color, family="JetBrains Mono, monospace"),
                suffix="",
            ),
            gauge=dict(
                axis=dict(
                    range=[0, 100],
                    tickcolor=_TEXT_DIM,
                    tickfont=dict(size=9, color=_TEXT_DIM),
                    dtick=25,
                ),
                bgcolor="rgba(255,255,255,0.03)",
                bordercolor="rgba(255,255,255,0.08)",
                bar=dict(color=bar_color, thickness=0.75),
                steps=[
                    dict(range=[0, 40], color="rgba(239,68,68,0.08)"),
                    dict(range=[40, 75], color="rgba(245,158,11,0.06)"),
                    dict(range=[75, 100], color="rgba(6,214,160,0.06)"),
                ],
            ),
        )
    )
    _apply_layout(fig, height=220, margin=dict(l=24, r=24, t=16, b=8))
    return fig


def radar_chart(
    categories: list[str],
    values: list[float],
    title: str = "",
) -> go.Figure:
    """Create a radar/spider chart for multi-dimensional scores."""
    if not categories or not values:
        return empty_figure(title, "No RAG data yet")

    # Close the polygon
    cats = categories + [categories[0]]
    vals = values + [values[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=vals,
            theta=cats,
            fill="toself",
            fillcolor="rgba(6,214,160,0.12)",
            line=dict(color=_CYAN, width=2),
            marker=dict(size=5, color=_CYAN),
        )
    )

    _apply_layout(fig, height=340)
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor="rgba(255,255,255,0.06)",
                tickfont=dict(size=8, color=_TEXT_DIM),
            ),
            angularaxis=dict(
                gridcolor="rgba(255,255,255,0.06)",
                tickfont=dict(size=10, color=_TEXT),
            ),
        ),
        title=dict(text=title, font=dict(size=12, color=_TEXT_DIM), x=0, xanchor="left"),
    )
    return fig


# ── Plotly dark theme for dashboard figures ──────────────────────────

_DARK_LAYOUT: dict = dict(
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


# ── Data layer (demo-only) ───────────────────────────────────────────

_DEMO_RECORDS: list[EvalRecord] | None = None


def _fetch_records(limit: int = 500) -> list[EvalRecord]:
    """Return cached demo records (generated once on first call)."""
    global _DEMO_RECORDS
    if _DEMO_RECORDS is None:
        _DEMO_RECORDS = generate_demo_records(200)
    return _DEMO_RECORDS[:limit]


def _fetch_alerts(limit: int = 20) -> list:
    """No real alerts in demo mode."""
    return []


# ── KPI card HTML helper ────────────────────────────────────────────


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


# ── Tab 1: Overview ─────────────────────────────────────────────────


def build_overview():
    records = _fetch_records(500)
    alerts = _fetch_alerts(20)

    if not records:
        return (
            _kpi_card("Health Score", "---", "no data", "#06d6a0"),
            _kpi_card("Hallucination", "---", "no data", "#f59e0b"),
            _kpi_card("Drift", "---", "no data", "#3b82f6"),
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

    trend = go.Figure()
    min_score = max(0, min(scores) - 10)
    trend.add_trace(
        go.Scatter(
            x=times,
            y=scores,
            mode="lines",
            name="Health Score",
            line=dict(color="#06d6a0", width=2, shape="spline"),
            fill="tonexty" if min_score > 30 else "none",
            fillcolor="rgba(6,214,160,0.06)",
        )
    )
    # Only show threshold lines if they're within visible range
    if min_score <= 75:
        trend.add_hline(
            y=75,
            line_dash="dot",
            line_color="#f59e0b",
            line_width=1,
            annotation_text="Warning: 75",
            annotation_font_size=9,
            annotation_font_color="#f59e0b",
        )
    if min_score <= 40:
        trend.add_hline(
            y=40,
            line_dash="dot",
            line_color="#ef4444",
            line_width=1,
            annotation_text="Critical: 40",
            annotation_font_size=9,
            annotation_font_color="#ef4444",
        )
    _dark(
        trend,
        title="Health Score Trend",
        yaxis=dict(range=[min_score, 105], **_DARK_LAYOUT["yaxis"]),
        height=350,
    )

    alert_rows = [["---", "", "", "", "", "No alerts triggered"]]
    if alerts:
        alert_rows = []
        for a in alerts[:20]:
            alert_rows.append(
                [
                    a.timestamp.strftime("%Y-%m-%d %H:%M"),
                    a.severity.upper(),
                    a.metric,
                    f"{a.value:.4f}",
                    f"{a.threshold:.4f}",
                    a.message,
                ]
            )

    return (
        _kpi_card("Health Score", str(avg_health), h_sub, "#06d6a0"),
        _kpi_card("Hallucination", f"{avg_halluc:.1%}", f"avg of {len(records)}", "#f59e0b"),
        _kpi_card("Drift", d_val, d_sub, "#3b82f6"),
        _kpi_card("Evaluations", f"{len(records):,}", "total tracked", "#a78bfa"),
        health_gauge_chart(avg_health),
        trend,
        alert_rows,
    )


# ── Tab 2: Hallucination ────────────────────────────────────────────


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
        y=0.3,
        line_dash="dot",
        line_color="#f59e0b",
        annotation_text="Threshold 0.3",
        annotation_font_size=9,
        annotation_font_color="#f59e0b",
    )
    _dark(
        rate,
        title="Hallucination Score Over Time",
        yaxis=dict(range=[0, 1.05], **_DARK_LAYOUT["yaxis"]),
        height=350,
    )

    dist = go.Figure(
        go.Histogram(
            x=h_scores,
            nbinsx=25,
            marker_color="#ef4444",
            opacity=0.7,
            marker_line_width=0,
        )
    )
    dist.add_vline(x=0.3, line_dash="dot", line_color="#f59e0b")
    _dark(dist, title="Score Distribution", height=300, bargap=0.05)

    ms: dict[str, list[float]] = defaultdict(list)
    for r in records:
        ms[r.model_name].append(r.hallucination_score)
    model_names = list(ms.keys())
    avgs = [sum(v) / len(v) for v in ms.values()]
    model_fig = go.Figure(
        go.Bar(
            x=model_names,
            y=avgs,
            marker_color=["#ef4444" if a > 0.3 else "#06d6a0" for a in avgs],
            marker_line_width=0,
        )
    )
    _dark(model_fig, title="Avg Hallucination by Model", height=300)

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

    return rate, dist, model_fig, rows


# ── Tab 3: Drift ────────────────────────────────────────────────────


def build_drift():
    records = _fetch_records(500)
    if not records:
        e = empty_figure("", "No data yet")
        return e, e, "No data"

    sorted_recs = sorted(records, key=lambda r: r.timestamp)
    drift_recs = [r for r in sorted_recs if r.drift_score is not None]

    emb_recs = [r for r in sorted_recs if r.embedding_vector and len(r.embedding_vector) > 2]
    if len(emb_recs) >= 3:
        embed = go.Figure(
            go.Scatter(
                x=[r.embedding_vector[0] for r in emb_recs],
                y=[r.embedding_vector[1] for r in emb_recs],
                mode="markers",
                marker=dict(
                    size=8,
                    color=[r.hallucination_score for r in emb_recs],
                    colorscale=[
                        [0, "#06d6a0"],
                        [0.5, "#f59e0b"],
                        [1, "#ef4444"],
                    ],
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
        y=0.15,
        line_dash="dot",
        line_color="#ef4444",
        annotation_text="Threshold 0.15",
        annotation_font_size=9,
        annotation_font_color="#ef4444",
    )
    y_max = max(max(scores) * 1.2, 0.3)
    _dark(
        dfig,
        title="Drift Score Over Time",
        yaxis=dict(range=[0, y_max], **_DARK_LAYOUT["yaxis"]),
        height=350,
    )

    avg = sum(scores) / len(scores)
    if avg < 0.1:
        st = "Stable"
    elif avg < 0.2:
        st = "Minor drift"
    else:
        st = "Significant drift!"

    return dfig, embed, st


# ── Tab 4: RAG & Quality ────────────────────────────────────────────


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
    _dark(
        qfig,
        title="Quality Metrics Over Time",
        yaxis=dict(range=[0, 1.05], **_DARK_LAYOUT["yaxis"]),
        height=350,
    )

    rag_recs = [r for r in sorted_recs if r.groundedness_score is not None]
    if rag_recs:
        rt = [r.timestamp.strftime("%m-%d %H:%M") for r in rag_recs]
        rfig = go.Figure()
        rfig.add_trace(
            go.Scatter(
                x=rt,
                y=[r.faithfulness_score or 0 for r in rag_recs],
                mode="lines",
                name="Faithfulness",
                line=dict(color="#06d6a0", width=2, shape="spline"),
            )
        )
        rfig.add_trace(
            go.Scatter(
                x=rt,
                y=[r.context_relevance or 0 for r in rag_recs],
                mode="lines",
                name="Context Relevance",
                line=dict(color="#3b82f6", width=2, shape="spline"),
            )
        )
        rfig.add_trace(
            go.Scatter(
                x=rt,
                y=[r.groundedness_score or 0 for r in rag_recs],
                mode="lines",
                name="Groundedness",
                line=dict(color="#a78bfa", width=2, dash="dash"),
            )
        )
        _dark(
            rfig,
            title="RAG Quality Metrics",
            yaxis=dict(range=[0, 1.05], **_DARK_LAYOUT["yaxis"]),
            height=350,
        )

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

    lang: dict[str, int] = defaultdict(int)
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


# ── CSS ─────────────────────────────────────────────────────────────

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


# ── App ─────────────────────────────────────────────────────────────


def create_app() -> gr.Blocks:
    with gr.Blocks(title="EvalPulse Dashboard", css=THEME_CSS) as app:
        gr.HTML("""
        <div class="ep-hdr"><div class="ep-hdr-in">
            <div class="ep-brand">
                <div class="ep-logo">EP</div>
                <div><div class="ep-t">EvalPulse</div>
                <div class="ep-st">LLM Evaluation &amp; Drift Monitor</div></div>
            </div>
            <div class="ep-live"><div class="ep-dot"></div>DEMO MODE</div>
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
                    headers=[
                        "Time",
                        "Severity",
                        "Metric",
                        "Value",
                        "Threshold",
                        "Message",
                    ],
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
                gr.Button("Refresh", variant="primary", size="sm").click(
                    fn=build_rag_quality, outputs=[qp, rp, rr, bp]
                )

        gr.HTML("""
        <div class="ep-ftr">
            EvalPulse v0.1.0 &middot; Open Source LLM Evaluation &amp; Drift Monitoring
            &middot; <a href="https://github.com/ninjacode911/Project-EvalPulse">GitHub</a>
        </div>
        """)

        app.load(fn=build_overview, outputs=[hc, hac, dc, tc, hg, ht, at])

    return app


if __name__ == "__main__":
    create_app().launch(server_name="0.0.0.0", server_port=7860)
