"""Pick a sensible Plotly chart for a result table. Output is plain JSON for plotly.js."""
from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go

from .models import Evidence

PURPLE, BLUE, GREEN, RED, MUTED = "#a855f7", "#60a5fa", "#34d399", "#f87171", "#9a93c4"
SERIES = [BLUE, PURPLE, "#f472b6", "#fbbf24", GREEN, "#22d3ee"]


def _layout(title: str, fig: go.Figure, ytitle: str | None = None) -> None:
    fig.update_layout(
        title=None, margin=dict(l=48, r=12, t=10, b=44), height=260,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cfc8ee", size=11, family="Inter, Segoe UI, system-ui, sans-serif"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False, automargin=True),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", zeroline=False, title=ytitle, automargin=True),
        legend=dict(orientation="h", y=-0.25, font=dict(size=10)), showlegend=False,
        hoverlabel=dict(bgcolor="#1b1535", font_color="#fff"),
    )


def _dates(s: pd.Series) -> list[str]:
    return pd.to_datetime(s).dt.strftime("%Y-%m-%d").tolist()


def make_chart(ev: Evidence, highlight: str | None = None) -> dict | None:
    df = ev.df
    if len(df) < 2:
        return None
    time_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    cat_cols = [c for c in df.columns if df[c].dtype == object or pd.api.types.is_string_dtype(df[c])]
    num_cols = [c for c in df.select_dtypes("number").columns if c.lower() not in ("year", "rank")]
    if not num_cols:
        return None
    fig = go.Figure()

    if time_cols:
        x = time_cols[0]
        if cat_cols and len(num_cols) >= 1:
            y = num_cols[0]
            for i, (k, g) in enumerate(df.groupby(cat_cols[0], sort=False)):
                fig.add_scatter(x=_dates(g[x]), y=g[y].tolist(), mode="lines", name=str(k),
                                line=dict(width=2, color=SERIES[i % len(SERIES)]))
            fig.update_layout(showlegend=True)
            kind, y_name = "line", y
        else:
            y = num_cols[0]
            fig.add_scatter(x=_dates(df[x]), y=df[y].tolist(), mode="lines", name=y,
                            line=dict(width=2.5, color=BLUE, shape="spline", smoothing=0.5),
                            fill="tozeroy", fillcolor="rgba(168,85,247,0.18)")
            kind, y_name = "area", y
            lo, hi = float(df[y].min()), float(df[y].max())
            if lo > 0 and (hi - lo) < 0.5 * hi:       # zoom in on the interesting range instead of starting at zero
                pad = max(hi - lo, hi * 0.02) * 0.25
                fig.update_yaxes(range=[max(0, lo - pad), hi + pad])
            if highlight:
                fig.add_vline(x=highlight, line=dict(color=RED, width=1.5, dash="dot"))
                row = df[pd.to_datetime(df[x]) == pd.to_datetime(highlight)]
                if len(row):
                    fig.add_scatter(x=[highlight], y=[float(row[y].iloc[0])], mode="markers",
                                    marker=dict(size=10, color=RED), name="flagged", hoverinfo="skip")
    elif cat_cols:
        x = cat_cols[0]
        y = "change" if "change" in num_cols else num_cols[0]
        d = df
        if y == "change":
            colors = [GREEN if v >= 0 else RED for v in d[y]]
        else:
            n = len(d)
            colors = [f"rgb({int(96 + (168 - 96) * i / max(n - 1, 1))},{int(165 + (85 - 165) * i / max(n - 1, 1))},{int(250 + (247 - 250) * i / max(n - 1, 1))})" for i in range(n)]
        fig.add_bar(x=d[x].astype(str).tolist(), y=d[y].tolist(), marker=dict(color=colors, line=dict(width=0)))
        kind, y_name = "bar", y
    else:
        return None

    _layout(ev.purpose, fig, None)
    return {"evidence": ev.id, "title": ev.purpose, "kind": kind, "y": y_name, "figure": json.loads(fig.to_json())}
