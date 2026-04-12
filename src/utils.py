"""
Shared utilities: file I/O, date helpers, and chart styling.
"""

from __future__ import annotations

import os
import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

LABEL_COLORS = {"positive": "#06d6a0", "negative": "#ef233c", "neutral": "#8ecae6"}
CHART_THEME  = "plotly_white"


# ── file I/O ─────────────────────────────────────────────────────────────────

def processed_path(ticker: str) -> str:
    return os.path.join(DATA_DIR, "processed", f"{ticker.lower()}_sentiment.csv")


def save_scored(df: pd.DataFrame, ticker: str) -> str:
    os.makedirs(os.path.join(DATA_DIR, "processed"), exist_ok=True)
    path = processed_path(ticker)
    df.to_csv(path, index=False)
    return path


def load_scored(ticker: str) -> pd.DataFrame | None:
    path = processed_path(ticker)
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["date"])
        return df
    return None


def save_chart(fig: go.Figure, name: str) -> str:
    os.makedirs(os.path.join(OUTPUTS_DIR, "charts"), exist_ok=True)
    path = os.path.join(OUTPUTS_DIR, "charts", f"{name}.png")
    try:
        fig.write_image(path, scale=2)
    except Exception:
        pass   # kaleido might not be installed; non-fatal
    return path


# ── date helpers ──────────────────────────────────────────────────────────────

def date_range_from_days(days: int) -> tuple[str, str]:
    end   = datetime.date.today()
    start = end - datetime.timedelta(days=days)
    return start.isoformat(), end.isoformat()


def parse_date_range(start, end) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(start), pd.Timestamp(end)


# ── plotly chart builders ────────────────────────────────────────────────────

def sentiment_distribution_chart(df: pd.DataFrame) -> go.Figure:
    counts = df["sentiment_label"].value_counts().reindex(
        ["positive", "negative", "neutral"], fill_value=0
    )
    fig = go.Figure(go.Bar(
        x=counts.index,
        y=counts.values,
        marker_color=[LABEL_COLORS[l] for l in counts.index],
        text=counts.values,
        textposition="outside",
    ))
    fig.update_layout(
        title="Sentiment Distribution",
        xaxis_title="Sentiment",
        yaxis_title="Number of Headlines",
        template=CHART_THEME,
        showlegend=False,
        height=380,
    )
    return fig


def sentiment_trend_chart(agg: pd.DataFrame, freq_label: str = "Daily") -> go.Figure:
    fig = go.Figure()

    # Shaded confidence band
    fig.add_trace(go.Scatter(
        x=pd.concat([agg["date"], agg["date"][::-1]]),
        y=pd.concat([
            agg["mean_score"] + agg["std_score"].fillna(0),
            (agg["mean_score"] - agg["std_score"].fillna(0))[::-1],
        ]),
        fill="toself",
        fillcolor="rgba(58,134,255,0.10)",
        line_color="rgba(0,0,0,0)",
        showlegend=False,
        name="±1 SD",
    ))

    # Score line
    fig.add_trace(go.Scatter(
        x=agg["date"], y=agg["mean_score"],
        mode="lines",
        name="Mean Score",
        line=dict(color="#3a86ff", width=1.5),
        opacity=0.7,
    ))

    # 7-period MA
    fig.add_trace(go.Scatter(
        x=agg["date"], y=agg["score_ma7"],
        mode="lines",
        name="7-period MA",
        line=dict(color="#ff006e", width=2.5),
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        title=f"{freq_label} Sentiment Score with 7-Period Moving Average",
        xaxis_title="Date",
        yaxis_title="Sentiment Score (−1 to +1)",
        template=CHART_THEME,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
    )
    return fig


def stacked_sentiment_chart(agg: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for label, color in LABEL_COLORS.items():
        col = f"pct_{label}"
        if col in agg.columns:
            fig.add_trace(go.Bar(
                x=agg["date"], y=agg[col],
                name=label.capitalize(),
                marker_color=color,
            ))
    fig.update_layout(
        barmode="stack",
        title="Sentiment Composition Over Time (%)",
        xaxis_title="Date",
        yaxis_title="Percentage",
        template=CHART_THEME,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=380,
    )
    return fig


def price_vs_sentiment_chart(merged: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=merged["Date"], y=merged["price_norm"],
        name=f"{ticker} Price (normalised)",
        line=dict(color="#3a86ff", width=2),
        yaxis="y1",
    ))

    fig.add_trace(go.Scatter(
        x=merged["Date"], y=merged["score_ma7"],
        name="Sentiment MA-7",
        line=dict(color="#ff006e", width=2, dash="dot"),
        yaxis="y2",
    ))

    fig.update_layout(
        title=f"{ticker} Normalised Price vs. Sentiment Trend",
        xaxis_title="Date",
        template=CHART_THEME,
        height=420,
        yaxis=dict(title="Price (base 100)", side="left"),
        yaxis2=dict(title="Sentiment Score", side="right", overlaying="y",
                    zeroline=True, zerolinecolor="lightgray"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def correlation_scatter_chart(merged: pd.DataFrame, ticker: str) -> go.Figure:
    df = merged.dropna(subset=["mean_score", "return_next"]).copy()
    df["return_next_pct"] = df["return_next"] * 100

    fig = px.scatter(
        df,
        x="mean_score",
        y="return_next_pct",
        trendline="ols",
        color_discrete_sequence=["#3a86ff"],
        labels={
            "mean_score":      "Daily Mean Sentiment Score",
            "return_next_pct": "Next-Day Return (%)",
        },
        title=f"{ticker} — Sentiment Score vs. Next-Day Return",
        template=CHART_THEME,
        opacity=0.65,
    )
    fig.update_traces(marker=dict(size=7))
    fig.update_layout(height=420)
    return fig
