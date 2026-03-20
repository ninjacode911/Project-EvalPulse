"""Plotly chart builders for the EvalPulse dashboard.

Dark theme with glowing accents — mission control aesthetic.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

# ── Color palette ──
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

_LAYOUT_BASE = dict(
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


def time_series_chart(
    data: list[dict[str, Any]],
    metric: str = "value",
    title: str = "",
    threshold: float | None = None,
    color: str = _CYAN,
) -> go.Figure:
    """Create a time-series line chart with area fill."""
    if not data:
        return empty_figure(title, "Collecting data...")

    times = [d.get("time_bucket", d.get("timestamp", "")) for d in data]
    values = [d.get("avg_value", d.get(metric, 0)) for d in data]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=times,
            y=values,
            mode="lines",
            name=metric,
            line=dict(color=color, width=2, shape="spline"),
            fill="tozeroy",
            fillcolor=color.replace(")", ",0.08)").replace("rgb", "rgba")
            if "rgb" in color
            else f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.08)",
        )
    )

    if threshold is not None:
        fig.add_hline(
            y=threshold,
            line_dash="dot",
            line_color=_RED,
            line_width=1,
            annotation_text=f"Threshold {threshold}",
            annotation_font_size=9,
            annotation_font_color=_RED,
        )

    _apply_layout(
        fig,
        height=300,
        title=dict(text=title, font=dict(size=12, color=_TEXT_DIM), x=0, xanchor="left"),
    )
    return fig


def multi_line_chart(
    data: list[dict[str, Any]],
    metrics: list[str],
    colors: list[str] | None = None,
    title: str = "",
) -> go.Figure:
    """Create a multi-line time-series chart."""
    if not data:
        return empty_figure(title, "Collecting data...")

    if colors is None:
        colors = [_CYAN, _AMBER, _BLUE, _PURPLE, _PINK]

    times = [d.get("time_bucket", d.get("timestamp", "")) for d in data]
    fig = go.Figure()
    for i, metric in enumerate(metrics):
        values = [d.get(metric, 0) for d in data]
        fig.add_trace(
            go.Scatter(
                x=times,
                y=values,
                mode="lines",
                name=metric.replace("_", " ").title(),
                line=dict(color=colors[i % len(colors)], width=2, shape="spline"),
            )
        )

    _apply_layout(
        fig,
        height=320,
        title=dict(text=title, font=dict(size=12, color=_TEXT_DIM), x=0, xanchor="left"),
    )
    return fig


def bar_chart(
    labels: list[str],
    values: list[float],
    title: str = "",
    color: str = _BLUE,
) -> go.Figure:
    """Create a bar chart."""
    if not labels:
        return empty_figure(title, "No data available yet")

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=color,
            marker_line=dict(width=0),
            opacity=0.85,
        )
    )
    _apply_layout(
        fig,
        height=300,
        title=dict(text=title, font=dict(size=12, color=_TEXT_DIM), x=0, xanchor="left"),
    )
    return fig


def distribution_chart(
    values: list[float],
    title: str = "",
    color: str = _PURPLE,
    threshold: float | None = None,
) -> go.Figure:
    """Create a histogram / distribution chart."""
    if not values:
        return empty_figure(title, "No data yet")

    fig = go.Figure(
        go.Histogram(
            x=values,
            nbinsx=30,
            marker_color=color,
            opacity=0.7,
            marker_line=dict(width=0),
        )
    )

    if threshold is not None:
        fig.add_vline(
            x=threshold,
            line_dash="dot",
            line_color=_RED,
            line_width=1,
            annotation_text=f"Threshold {threshold}",
            annotation_font_size=9,
            annotation_font_color=_RED,
        )

    _apply_layout(
        fig,
        height=280,
        title=dict(text=title, font=dict(size=12, color=_TEXT_DIM), x=0, xanchor="left"),
        bargap=0.05,
    )
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
            fillcolor=f"rgba({int(_CYAN[1:3],16)},{int(_CYAN[3:5],16)},{int(_CYAN[5:7],16)},0.12)",
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


def model_comparison_chart(
    models: list[str],
    scores: list[float],
    title: str = "",
) -> go.Figure:
    """Horizontal bar chart comparing model performance."""
    if not models:
        return empty_figure(title, "No model data")

    colors = [_CYAN if s >= 0.7 else _AMBER if s >= 0.4 else _RED for s in scores]

    fig = go.Figure(
        go.Bar(
            y=models,
            x=scores,
            orientation="h",
            marker_color=colors,
            marker_line=dict(width=0),
            opacity=0.85,
        )
    )

    _apply_layout(
        fig,
        height=max(200, len(models) * 45),
        title=dict(text=title, font=dict(size=12, color=_TEXT_DIM), x=0, xanchor="left"),
        xaxis=dict(range=[0, 1], **_LAYOUT_BASE["xaxis"]),
    )
    return fig
