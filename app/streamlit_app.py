"""
Financial News Sentiment Analysis — Streamlit Dashboard

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
import sys
import warnings
import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader    import load_news
from src.sentiment_model import score_dataframe, LABEL_COLORS, LABEL_EMOJIS
from src.stock_data     import load_prices, price_summary
from src.analysis       import aggregate_sentiment, merge_sentiment_prices, compute_correlation
from src.utils          import (
    load_scored, save_scored,
    sentiment_distribution_chart, sentiment_trend_chart,
    stacked_sentiment_chart, price_vs_sentiment_chart,
    correlation_scatter_chart, CHART_THEME,
)

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Sentiment Analyzer",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .big-title {
        font-size: 2.3rem; font-weight: 900;
        background: linear-gradient(90deg, #3a86ff, #ff006e);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .subtitle { color: #6c757d; font-size: 1.0rem; margin-bottom: 1.5rem; }
    .metric-box {
        background: #f8f9fa; border-radius: 10px;
        padding: 1rem 1.2rem; border-left: 4px solid #3a86ff;
    }
    .pos-tag  { color: #06d6a0; font-weight: 700; }
    .neg-tag  { color: #ef233c; font-weight: 700; }
    .neu-tag  { color: #8ecae6; font-weight: 700; }
    [data-testid="stSidebar"] { background: #f0f4ff; }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    ticker = st.selectbox(
        "Ticker",
        ["AAPL", "MSFT", "TSLA", "GOOGL", "AMZN", "NVDA"],
        index=0,
    )

    date_col1, date_col2 = st.columns(2)
    default_start = datetime.date.today() - datetime.timedelta(days=180)
    default_end   = datetime.date.today()
    with date_col1:
        start_date = st.date_input("From", value=default_start)
    with date_col2:
        end_date = st.date_input("To", value=default_end)

    news_source = st.selectbox(
        "News Source",
        ["auto", "sample", "yfinance", "alphavantage"],
        index=0,
        help="'auto' tries Alpha Vantage → yfinance → sample dataset",
    )

    batch_size = st.slider("Inference Batch Size", 4, 32, 16, step=4,
                            help="Lower if you experience memory errors")

    run_btn = st.button("🔍 Analyze Sentiment", use_container_width=True, type="primary")
    st.caption("⚡ First run downloads FinBERT (~400 MB). Subsequent runs use cache.")

    st.markdown("---")
    st.markdown("**Model:** [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)")
    st.markdown("**Stock data:** [yfinance](https://pypi.org/project/yfinance/)")
    st.markdown("**Author:** Elif Naz Turan")


# ── header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="big-title">📰 Financial News Sentiment Analyzer</p>',
            unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">FinBERT transformer model · News sentiment trends · '
    'Stock price correlation · Interactive exploration</p>',
    unsafe_allow_html=True,
)


# ── data loading / scoring ────────────────────────────────────────────────────
@st.cache_resource
def get_finbert_ready():
    """Pre-load FinBERT so it's ready for inference."""
    from src.sentiment_model import load_finbert
    load_finbert()


@st.cache_data(show_spinner=False)
def get_scored_data(ticker: str, source: str, start: str, end: str, bs: int) -> pd.DataFrame:
    cached = load_scored(ticker)
    if cached is not None and not cached.empty:
        # filter to selected date range
        cached["date"] = pd.to_datetime(cached["date"])
        mask = (cached["date"] >= pd.Timestamp(start)) & (cached["date"] <= pd.Timestamp(end))
        if mask.sum() > 5:
            return cached[mask].reset_index(drop=True)

    news = load_news(ticker, days=(pd.Timestamp(end) - pd.Timestamp(start)).days + 30,
                     source=source)
    scored = score_dataframe(news, batch_size=bs)
    save_scored(scored, ticker)

    scored["date"] = pd.to_datetime(scored["date"])
    mask = (scored["date"] >= pd.Timestamp(start)) & (scored["date"] <= pd.Timestamp(end))
    return scored[mask].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def get_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    return load_prices(ticker, start=start, end=end)


# ── session state ─────────────────────────────────────────────────────────────
if "df_scored" not in st.session_state:
    st.session_state.df_scored  = None
    st.session_state.df_prices  = None
    st.session_state.df_daily   = None
    st.session_state.df_weekly  = None
    st.session_state.df_merged  = None
    st.session_state.corr_stats = None
    st.session_state.ticker     = None

if run_btn:
    start_str = start_date.isoformat()
    end_str   = end_date.isoformat()

    with st.spinner("Loading FinBERT and scoring headlines …"):
        get_finbert_ready()
        df_scored = get_scored_data(ticker, news_source, start_str, end_str, batch_size)

    with st.spinner("Fetching stock price data …"):
        try:
            df_prices = get_prices(ticker, start_str, end_str)
        except Exception as e:
            st.error(f"Price data error: {e}")
            df_prices = pd.DataFrame()

    if not df_scored.empty:
        df_daily  = aggregate_sentiment(df_scored, freq="D")
        df_weekly = aggregate_sentiment(df_scored, freq="W")

        if not df_prices.empty:
            df_merged  = merge_sentiment_prices(df_daily, df_prices)
            corr_stats = compute_correlation(df_merged)
        else:
            df_merged  = pd.DataFrame()
            corr_stats = {}

        st.session_state.df_scored  = df_scored
        st.session_state.df_prices  = df_prices
        st.session_state.df_daily   = df_daily
        st.session_state.df_weekly  = df_weekly
        st.session_state.df_merged  = df_merged
        st.session_state.corr_stats = corr_stats
        st.session_state.ticker     = ticker
        st.success(f"✅ Scored **{len(df_scored)}** headlines for **{ticker}**")

# ── main content ──────────────────────────────────────────────────────────────
if st.session_state.df_scored is None:
    st.info("👈 Select a ticker and date range in the sidebar, then click **Analyze Sentiment**.")

    st.markdown("---")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("### 🤖 Model\n**ProsusAI/FinBERT** — BERT fine-tuned on 10 000+ financial news articles, earnings call transcripts, and analyst reports. Assigns *positive*, *negative*, or *neutral* with probability scores.")
    with cols[1]:
        st.markdown("### 📊 Analysis\nAggregates daily and weekly sentiment trends, computes a **sentiment score** = P(positive) − P(negative), applies a 7-period moving average, and identifies bullish/bearish regimes.")
    with cols[2]:
        st.markdown("### 📈 Stock Correlation\nCorrelates today's sentiment with **tomorrow's stock return** using Pearson and Spearman metrics. ⚠️ *Correlation ≠ causation* — this is exploratory analysis only.")
    st.stop()

# ── load session data ─────────────────────────────────────────────────────────
df_scored  = st.session_state.df_scored
df_prices  = st.session_state.df_prices
df_daily   = st.session_state.df_daily
df_weekly  = st.session_state.df_weekly
df_merged  = st.session_state.df_merged
corr_stats = st.session_state.corr_stats
current_ticker = st.session_state.ticker

# ── top KPI row ───────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
n_pos = (df_scored["sentiment_label"] == "positive").sum()
n_neg = (df_scored["sentiment_label"] == "negative").sum()
n_neu = (df_scored["sentiment_label"] == "neutral").sum()
mean_score = df_scored["sentiment_score"].mean()

k1.metric("Headlines Scored",   len(df_scored))
k2.metric("🟢 Positive",  f"{n_pos}  ({n_pos/len(df_scored)*100:.0f}%)")
k3.metric("🔴 Negative",  f"{n_neg}  ({n_neg/len(df_scored)*100:.0f}%)")
k4.metric("🔵 Neutral",   f"{n_neu}  ({n_neu/len(df_scored)*100:.0f}%)")
k5.metric("Mean Sentiment Score", f"{mean_score:+.3f}")

st.markdown("---")

# ── tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Distribution",
    "📈 Sentiment Trends",
    "💹 Stock Price",
    "🔗 Correlation",
    "📰 Headlines",
    "ℹ️ About",
])

# ── Tab 1: Distribution ───────────────────────────────────────────────────────
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        st.plotly_chart(sentiment_distribution_chart(df_scored), use_container_width=True)

    with c2:
        pie = go.Figure(go.Pie(
            labels=["Positive", "Negative", "Neutral"],
            values=[n_pos, n_neg, n_neu],
            marker_colors=[LABEL_COLORS["positive"], LABEL_COLORS["negative"], LABEL_COLORS["neutral"]],
            hole=0.45,
            textinfo="label+percent",
        ))
        pie.update_layout(title="Sentiment Share", template=CHART_THEME, height=380,
                          showlegend=False)
        st.plotly_chart(pie, use_container_width=True)

    # Confidence distribution
    st.markdown("#### Confidence Score Distribution")
    fig_conf = px.histogram(
        df_scored, x="confidence", color="sentiment_label",
        color_discrete_map=LABEL_COLORS,
        nbins=30, barmode="overlay", opacity=0.75,
        labels={"confidence": "Model Confidence", "sentiment_label": "Sentiment"},
        template=CHART_THEME, height=320,
        title="FinBERT Prediction Confidence by Sentiment Class",
    )
    st.plotly_chart(fig_conf, use_container_width=True)

    # Probability triangle
    st.markdown("#### Probability Space (Positive vs Negative, sized by Confidence)")
    fig_tri = px.scatter(
        df_scored.sample(min(500, len(df_scored)), random_state=42),
        x="prob_positive", y="prob_negative",
        color="sentiment_label",
        size="confidence",
        color_discrete_map=LABEL_COLORS,
        labels={"prob_positive": "P(Positive)", "prob_negative": "P(Negative)",
                "sentiment_label": "Sentiment"},
        template=CHART_THEME, height=380,
        title="FinBERT Output Probability Space",
        opacity=0.65,
    )
    fig_tri.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                      line=dict(dash="dash", color="gray"))
    st.plotly_chart(fig_tri, use_container_width=True)


# ── Tab 2: Sentiment Trends ───────────────────────────────────────────────────
with tab2:
    freq_choice = st.radio("Aggregation", ["Daily", "Weekly"], horizontal=True)
    agg = df_daily if freq_choice == "Daily" else df_weekly

    st.plotly_chart(sentiment_trend_chart(agg, freq_label=freq_choice), use_container_width=True)
    st.plotly_chart(stacked_sentiment_chart(agg), use_container_width=True)

    # Regime table
    st.markdown("#### Sentiment Regime Summary")
    regime_counts = agg["regime"].value_counts().rename_axis("Regime").reset_index(name="Periods")
    st.dataframe(regime_counts, use_container_width=True, hide_index=True)

    st.markdown("#### Aggregated Sentiment Table")
    display_agg = agg.copy()
    display_agg["date"] = display_agg["date"].dt.strftime("%Y-%m-%d")
    display_agg["mean_score"] = display_agg["mean_score"].round(4)
    st.dataframe(
        display_agg[["date", "n_headlines", "mean_score", "score_ma7",
                     "pct_positive", "pct_negative", "pct_neutral", "regime"]],
        use_container_width=True, hide_index=True,
    )


# ── Tab 3: Stock Price ────────────────────────────────────────────────────────
with tab3:
    if df_prices.empty:
        st.warning("No price data available. Check your internet connection.")
    else:
        summary = price_summary(df_prices)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Start Price",    f"${summary['start_price']}")
        m2.metric("End Price",      f"${summary['end_price']}")
        m3.metric("Total Return",   f"{summary['total_return']:+.2f}%")
        m4.metric("Ann. Volatility",f"{summary['ann_volatility']:.2f}%")
        m5.metric("Max Drawdown",   f"{summary['max_drawdown']:.2f}%")

        # OHLCV candlestick
        fig_candle = go.Figure(go.Candlestick(
            x=df_prices["Date"],
            open=df_prices["Open"], high=df_prices["High"],
            low=df_prices["Low"],   close=df_prices["Close"],
            increasing_line_color="#06d6a0",
            decreasing_line_color="#ef233c",
            name=current_ticker,
        ))
        fig_candle.update_layout(
            title=f"{current_ticker} Price (OHLC)",
            xaxis_title="Date", yaxis_title="Price ($)",
            template=CHART_THEME, height=420,
            xaxis_rangeslider_visible=False,
        )
        st.plotly_chart(fig_candle, use_container_width=True)

        # Overlay sentiment
        if not df_merged.empty:
            st.plotly_chart(
                price_vs_sentiment_chart(df_merged, current_ticker),
                use_container_width=True,
            )

        # Volume bar
        fig_vol = go.Figure(go.Bar(
            x=df_prices["Date"], y=df_prices["Volume"],
            marker_color="#3a86ff", opacity=0.6, name="Volume",
        ))
        fig_vol.update_layout(title=f"{current_ticker} Trading Volume",
                               xaxis_title="Date", yaxis_title="Volume",
                               template=CHART_THEME, height=280)
        st.plotly_chart(fig_vol, use_container_width=True)


# ── Tab 4: Correlation ────────────────────────────────────────────────────────
with tab4:
    st.markdown("### Sentiment → Next-Day Return Correlation")
    st.info(
        "⚠️ **Disclaimer:** The following analysis is **exploratory only**. "
        "Correlation between news sentiment and stock returns does **not** imply "
        "causation and should **not** be used as a basis for trading decisions."
    )

    if df_merged.empty:
        st.warning("Insufficient merged data for correlation analysis.")
    else:
        c = corr_stats
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Observations",    c.get("n_observations", "—"))
        col2.metric("Pearson r",       f"{c.get('pearson_r',  0):+.4f}" if c.get("pearson_r")  else "—")
        col3.metric("Spearman r",      f"{c.get('spearman_r', 0):+.4f}" if c.get("spearman_r") else "—")
        col4.metric("Pearson p-value", f"{c.get('pearson_p',  1):.4f}"  if c.get("pearson_p")  else "—")

        sig_msg = (
            "✅ Pearson correlation is **statistically significant** (p < 0.05)."
            if c.get("pearson_sig") else
            "❌ Pearson correlation is **not statistically significant** (p ≥ 0.05)."
        )
        st.markdown(sig_msg)

        st.plotly_chart(
            correlation_scatter_chart(df_merged, current_ticker),
            use_container_width=True,
        )

        # Rolling correlation
        st.markdown("#### 20-Day Rolling Pearson Correlation (Sentiment vs. Next-Day Return)")
        roll_corr = (
            df_merged[["mean_score", "return_next"]]
            .dropna()
            .rolling(20)
            .apply(lambda x: x.iloc[:, 0].corr(x.iloc[:, 1]) if len(x) > 5 else np.nan,
                   raw=False)
        )
        # simpler rolling
        s = df_merged["mean_score"].reset_index(drop=True)
        r = df_merged["return_next"].reset_index(drop=True)
        dates = df_merged["Date"].reset_index(drop=True)
        roll_vals = [
            s.iloc[max(0, i-19):i+1].corr(r.iloc[max(0, i-19):i+1])
            if i >= 5 else np.nan
            for i in range(len(s))
        ]

        fig_roll = go.Figure()
        fig_roll.add_trace(go.Scatter(
            x=dates, y=roll_vals,
            mode="lines", name="Rolling r",
            line=dict(color="#3a86ff", width=2),
        ))
        fig_roll.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_roll.update_layout(
            xaxis_title="Date", yaxis_title="Pearson r",
            template=CHART_THEME, height=320,
        )
        st.plotly_chart(fig_roll, use_container_width=True)

        st.markdown(
            "**How to read this:** A positive rolling correlation means that, in that window, "
            "higher sentiment scores tended to precede higher next-day returns — and vice versa. "
            "The relationship is unstable over time, which is typical in financial markets."
        )


# ── Tab 5: Headlines ──────────────────────────────────────────────────────────
with tab5:
    st.markdown(f"### {current_ticker} Scored Headlines")

    sentiment_filter = st.multiselect(
        "Filter by sentiment",
        ["positive", "negative", "neutral"],
        default=["positive", "negative", "neutral"],
    )
    keyword = st.text_input("Search headlines", placeholder="e.g. earnings, layoffs, AI …")

    filtered = df_scored[df_scored["sentiment_label"].isin(sentiment_filter)].copy()
    if keyword:
        filtered = filtered[filtered["headline"].str.contains(keyword, case=False, na=False)]

    filtered["date"] = pd.to_datetime(filtered["date"]).dt.strftime("%Y-%m-%d")
    filtered["sentiment_score"] = filtered["sentiment_score"].round(3)
    filtered["confidence"]      = filtered["confidence"].round(3)

    st.caption(f"Showing {len(filtered)} of {len(df_scored)} headlines")
    st.dataframe(
        filtered[["date", "headline", "sentiment_label", "sentiment_score",
                  "prob_positive", "prob_negative", "prob_neutral", "confidence", "source"]]
        .sort_values("date", ascending=False)
        .reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
        height=500,
    )

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download filtered results as CSV",
        data=csv,
        file_name=f"{current_ticker}_sentiment_filtered.csv",
        mime="text/csv",
    )


# ── Tab 6: About ──────────────────────────────────────────────────────────────
with tab6:
    st.markdown("""
### About This Project

**Financial News Sentiment Analysis** is a portfolio data science project that
applies a fine-tuned transformer model to measure the emotional tone of financial
news and investigates its relationship with stock price movements.

---

#### Model — ProsusAI/FinBERT

FinBERT is a BERT-base model fine-tuned on a large corpus of financial text
including news articles, earnings call transcripts, and analyst reports.
It classifies text into three sentiment classes and outputs calibrated probabilities.

| Label    | Meaning                              |
|----------|--------------------------------------|
| Positive | Optimistic / bullish tone            |
| Negative | Pessimistic / bearish tone           |
| Neutral  | Factual or non-directional reporting |

**Sentiment Score** = P(positive) − P(negative) ∈ [−1, +1]

---

#### Data Sources

| Source            | Description                                     |
|-------------------|-------------------------------------------------|
| Alpha Vantage API | Real-time financial news (free tier, 25 req/day)|
| yfinance news     | Recent ticker-linked headlines, no key needed   |
| Sample dataset    | 240 curated headlines for offline demo          |
| yfinance prices   | Daily OHLCV stock data, always available        |

---

#### Sentiment → Return Correlation

The dashboard correlates **today's mean sentiment score** with **tomorrow's stock return**.
This forward-looking design tests whether news sentiment has predictive signal.

⚠️ Results are **statistically weak and unstable** — consistent with academic literature
showing markets are largely efficient. This analysis is for educational purposes only.

---

#### Author
**Elif Naz Turan** — Data Science Portfolio Project
""")
