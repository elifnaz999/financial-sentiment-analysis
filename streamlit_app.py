"""
Financial News Sentiment Analysis — Streamlit Dashboard
Entry point for Streamlit Community Cloud deployment.

Local development:
    streamlit run streamlit_app.py

The src/ package is importable directly because this file sits at the repo root.
"""

from __future__ import annotations

import os
import warnings
import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

warnings.filterwarnings("ignore")

from src.data_loader     import load_news
from src.sentiment_model import score_dataframe, LABEL_COLORS, LABEL_EMOJIS
from src.stock_data      import load_prices, price_summary
from src.analysis        import aggregate_sentiment, merge_sentiment_prices, compute_correlation
from src.utils           import (
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
    [data-testid="stSidebar"] { background: #f0f4ff; }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ── ticker lists ─────────────────────────────────────────────────────────────
_US_TICKERS   = ["AAPL", "MSFT", "TSLA", "GOOGL", "AMZN", "NVDA"]
_BIST_TICKERS = ["THYAO", "ASELS", "GARAN", "AKBNK", "KCHOL",
                 "SISE", "EREGL", "TUPRS", "BIMAS", "ISCTR"]

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    market = st.selectbox("Market", ["US", "BIST"], index=0)
    st.markdown("---")
    _ticker_list = _US_TICKERS if market == "US" else _BIST_TICKERS
    ticker = st.selectbox("Ticker", _ticker_list, index=0)

    date_col1, date_col2 = st.columns(2)
    default_start = datetime.date.today() - datetime.timedelta(days=180)
    default_end   = datetime.date.today()
    with date_col1:
        start_date = st.date_input("From", value=default_start)
    with date_col2:
        end_date = st.date_input("To", value=default_end)

    if market == "US":
        _source_opts = ["auto", "sample", "yfinance", "alphavantage"]
        _source_help = "'auto' tries Alpha Vantage → yfinance → sample dataset"
    else:
        _source_opts = ["auto", "sample", "kap", "bigpara", "investing_tr"]
        _source_help = "'auto' tries KAP RSS → Bigpara → Investing.com TR → sample dataset"

    news_source = st.selectbox(
        "News Source",
        _source_opts,
        index=0,
        help=_source_help,
    )

    batch_size = st.slider(
        "Inference Batch Size", 4, 16, 8, step=4,
        help="Lower values use less memory. Recommended: 8 on cloud.",
    )

    run_btn = st.button("🔍 Analyze Sentiment", use_container_width=True, type="primary")
    st.caption("⚡ First run downloads FinBERT (~440 MB). Subsequent runs use cache.")

    st.markdown("---")
    st.markdown("**Model:** [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)")
    st.markdown("**Stock data:** [yfinance](https://pypi.org/project/yfinance/)")
    st.markdown("**Author:** Elif Naz Turan")


# ── header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="big-title">📰 Financial News Sentiment Analyzer</p>',
            unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">FinBERT transformer · Sentiment trends · '
    'Stock price correlation · Interactive exploration</p>',
    unsafe_allow_html=True,
)


# ── cached model load ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading FinBERT model …")
def load_finbert_cached():
    """Load once per app session; persists across reruns."""
    from src.sentiment_model import load_finbert
    load_finbert()


@st.cache_data(show_spinner=False, ttl=3600)
def get_scored_data(ticker: str, source: str, start: str, end: str, bs: int,
                    market: str = "US") -> pd.DataFrame:
    """Fetch + score headlines. Cached for 1 hour per (ticker, source, dates, market) combo."""
    # try disk cache first (within same session/container)
    cached = load_scored(ticker)
    if cached is not None and not cached.empty:
        cached["date"] = pd.to_datetime(cached["date"])
        mask = (cached["date"] >= pd.Timestamp(start)) & (cached["date"] <= pd.Timestamp(end))
        if mask.sum() > 5:
            return cached[mask].reset_index(drop=True)

    days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 30
    news   = load_news(ticker, days=days, source=source, market=market)
    scored = score_dataframe(news, batch_size=bs)
    try:
        save_scored(scored, ticker)
    except Exception:
        pass  # ephemeral filesystem on cloud; non-fatal

    scored["date"] = pd.to_datetime(scored["date"])
    mask = (scored["date"] >= pd.Timestamp(start)) & (scored["date"] <= pd.Timestamp(end))
    return scored[mask].reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=3600)
def get_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    return load_prices(ticker, start=start, end=end)


# ── session state initialisation ──────────────────────────────────────────────
for key in ("df_scored", "df_prices", "df_daily", "df_weekly",
            "df_merged", "corr_stats", "active_ticker", "active_market"):
    if key not in st.session_state:
        st.session_state[key] = None


# ── run analysis ──────────────────────────────────────────────────────────────
if run_btn:
    start_str = start_date.isoformat()
    end_str   = end_date.isoformat()

    load_finbert_cached()

    with st.spinner("Scoring headlines with FinBERT …"):
        try:
            df_scored = get_scored_data(ticker, news_source, start_str, end_str, batch_size, market)
        except Exception as e:
            st.error(f"Sentiment scoring failed: {e}")
            st.stop()

    # BIST tickers require the ".IS" suffix on Yahoo Finance
    yf_ticker = f"{ticker}.IS" if market == "BIST" else ticker

    with st.spinner("Fetching stock prices …"):
        try:
            df_prices = get_prices(yf_ticker, start_str, end_str)
        except Exception as e:
            st.warning(f"Price data unavailable: {e}")
            df_prices = pd.DataFrame()

    if df_scored.empty:
        st.error("No headlines found for the selected ticker and date range.")
        st.stop()

    df_daily  = aggregate_sentiment(df_scored, freq="D")
    df_weekly = aggregate_sentiment(df_scored, freq="W")
    df_merged = merge_sentiment_prices(df_daily, df_prices) if not df_prices.empty else pd.DataFrame()
    corr_stats = compute_correlation(df_merged) if not df_merged.empty else {}

    st.session_state.df_scored    = df_scored
    st.session_state.df_prices    = df_prices
    st.session_state.df_daily     = df_daily
    st.session_state.df_weekly    = df_weekly
    st.session_state.df_merged    = df_merged
    st.session_state.corr_stats   = corr_stats
    st.session_state.active_ticker = ticker
    st.session_state.active_market = market
    st.success(f"✅ Scored **{len(df_scored)}** headlines for **{ticker}**")
    if market == "BIST":
        st.info(
            "ℹ️ **BIST mode:** Headlines are sourced from Turkish news feeds. "
            "FinBERT is trained on English financial text — sentiment scores for "
            "Turkish headlines are approximate. Turkish model support is planned for Phase 3."
        )


# ── landing screen (before first run) ─────────────────────────────────────────
if st.session_state.df_scored is None:
    st.info("👈 Select a ticker and date range in the sidebar, then click **Analyze Sentiment**.")
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 🤖 Model\n**ProsusAI/FinBERT** — BERT fine-tuned on 10 000+ financial texts. Assigns *positive*, *negative*, or *neutral* with probability scores.")
    with c2:
        st.markdown("### 📊 Analysis\nAggregates daily and weekly trends, computes **sentiment score** = P(positive) − P(negative), with 7-period moving average and bullish/bearish regime labels.")
    with c3:
        st.markdown("### 📈 Correlation\nCorrelates today's sentiment with **tomorrow's stock return** (Pearson + Spearman). ⚠️ *Correlation ≠ causation* — exploratory only.")
    st.stop()


# ── unpack session state ──────────────────────────────────────────────────────
df_scored      = st.session_state.df_scored
df_prices      = st.session_state.df_prices
df_daily       = st.session_state.df_daily
df_weekly      = st.session_state.df_weekly
df_merged      = st.session_state.df_merged
corr_stats     = st.session_state.corr_stats
current_ticker = st.session_state.active_ticker
current_market = st.session_state.active_market or "US"

n_total = len(df_scored)
n_pos   = (df_scored["sentiment_label"] == "positive").sum()
n_neg   = (df_scored["sentiment_label"] == "negative").sum()
n_neu   = (df_scored["sentiment_label"] == "neutral").sum()
mean_sc = df_scored["sentiment_score"].mean()


# ── KPI row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Headlines Scored",    n_total)
k2.metric("🟢 Positive",  f"{n_pos} ({n_pos/n_total*100:.0f}%)")
k3.metric("🔴 Negative",  f"{n_neg} ({n_neg/n_total*100:.0f}%)")
k4.metric("🔵 Neutral",   f"{n_neu} ({n_neu/n_total*100:.0f}%)")
k5.metric("Mean Sentiment Score", f"{mean_sc:+.3f}")
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

# ── Tab 1 — Distribution ──────────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(sentiment_distribution_chart(df_scored), use_container_width=True)
    with col2:
        pie = go.Figure(go.Pie(
            labels=["Positive", "Negative", "Neutral"],
            values=[n_pos, n_neg, n_neu],
            marker_colors=[LABEL_COLORS["positive"], LABEL_COLORS["negative"], LABEL_COLORS["neutral"]],
            hole=0.45, textinfo="label+percent",
        ))
        pie.update_layout(title="Sentiment Share", template=CHART_THEME, height=380, showlegend=False)
        st.plotly_chart(pie, use_container_width=True)

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

    st.markdown("#### Probability Space")
    sample_df = df_scored.sample(min(500, n_total), random_state=42)
    fig_tri = px.scatter(
        sample_df, x="prob_positive", y="prob_negative",
        color="sentiment_label", size="confidence",
        color_discrete_map=LABEL_COLORS,
        labels={"prob_positive": "P(Positive)", "prob_negative": "P(Negative)",
                "sentiment_label": "Sentiment"},
        template=CHART_THEME, height=380,
        title="FinBERT Output Probability Space (sample of 500)",
        opacity=0.65,
    )
    fig_tri.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="gray"))
    st.plotly_chart(fig_tri, use_container_width=True)


# ── Tab 2 — Sentiment Trends ──────────────────────────────────────────────────
with tab2:
    freq_choice = st.radio("Aggregation", ["Daily", "Weekly"], horizontal=True)
    agg = df_daily if freq_choice == "Daily" else df_weekly

    st.plotly_chart(sentiment_trend_chart(agg, freq_label=freq_choice), use_container_width=True)
    st.plotly_chart(stacked_sentiment_chart(agg), use_container_width=True)

    st.markdown("#### Sentiment Regime Summary")
    regime_counts = agg["regime"].value_counts().rename_axis("Regime").reset_index(name="Periods")
    st.dataframe(regime_counts, use_container_width=True, hide_index=True)

    st.markdown("#### Aggregated Data Table")
    display_agg = agg.copy()
    display_agg["date"] = display_agg["date"].dt.strftime("%Y-%m-%d")
    for col in ["mean_score", "score_ma7", "pct_positive", "pct_negative", "pct_neutral"]:
        display_agg[col] = display_agg[col].round(3)
    st.dataframe(
        display_agg[["date", "n_headlines", "mean_score", "score_ma7",
                     "pct_positive", "pct_negative", "pct_neutral", "regime"]],
        use_container_width=True, hide_index=True,
    )


# ── Tab 3 — Stock Price ───────────────────────────────────────────────────────
with tab3:
    if df_prices is None or df_prices.empty:
        st.warning("No price data available. Check your internet connection.")
    else:
        summary = price_summary(df_prices)
        _currency = "₺" if current_market == "BIST" else "$"
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Start Price",     f"{_currency}{summary['start_price']}")
        m2.metric("End Price",       f"{_currency}{summary['end_price']}")
        m3.metric("Total Return",    f"{summary['total_return']:+.2f}%")
        m4.metric("Ann. Volatility", f"{summary['ann_volatility']:.2f}%")
        m5.metric("Max Drawdown",    f"{summary['max_drawdown']:.2f}%")

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
            xaxis_title="Date", yaxis_title=f"Price ({_currency})",
            template=CHART_THEME, height=420,
            xaxis_rangeslider_visible=False,
        )
        st.plotly_chart(fig_candle, use_container_width=True)

        if df_merged is not None and not df_merged.empty:
            st.plotly_chart(price_vs_sentiment_chart(df_merged, current_ticker), use_container_width=True)

        fig_vol = go.Figure(go.Bar(
            x=df_prices["Date"], y=df_prices["Volume"],
            marker_color="#3a86ff", opacity=0.6,
        ))
        fig_vol.update_layout(title=f"{current_ticker} Trading Volume",
                               xaxis_title="Date", yaxis_title="Volume",
                               template=CHART_THEME, height=280)
        st.plotly_chart(fig_vol, use_container_width=True)


# ── Tab 4 — Correlation ───────────────────────────────────────────────────────
with tab4:
    st.markdown("### Sentiment → Next-Day Return Correlation")
    st.info(
        "⚠️ **Disclaimer:** This analysis is **exploratory only**. "
        "Correlation does **not** imply causation and should **not** be used for trading decisions."
    )

    if df_merged is None or df_merged.empty:
        st.warning("Insufficient merged data for correlation analysis.")
    else:
        c = corr_stats
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Observations",    c.get("n_observations", "—"))
        col2.metric("Pearson r",       f"{c['pearson_r']:+.4f}" if c.get("pearson_r") is not None else "—")
        col3.metric("Spearman r",      f"{c['spearman_r']:+.4f}" if c.get("spearman_r") is not None else "—")
        col4.metric("Pearson p-value", f"{c['pearson_p']:.4f}"   if c.get("pearson_p") is not None else "—")

        if c.get("pearson_sig") is not None:
            if c["pearson_sig"]:
                st.success("✅ Pearson correlation is **statistically significant** (p < 0.05).")
            else:
                st.warning("❌ Pearson correlation is **not statistically significant** (p ≥ 0.05).")

        st.plotly_chart(correlation_scatter_chart(df_merged, current_ticker), use_container_width=True)

        # 20-day rolling correlation
        st.markdown("#### 20-Day Rolling Pearson Correlation")
        s_col  = df_merged["mean_score"].reset_index(drop=True)
        r_col  = df_merged["return_next"].reset_index(drop=True)
        dates  = df_merged["Date"].reset_index(drop=True)
        roll_vals = [
            s_col.iloc[max(0, i-19):i+1].corr(r_col.iloc[max(0, i-19):i+1])
            if i >= 5 else float("nan")
            for i in range(len(s_col))
        ]
        fig_roll = go.Figure(go.Scatter(
            x=dates, y=roll_vals, mode="lines",
            line=dict(color="#3a86ff", width=2),
        ))
        fig_roll.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_roll.update_layout(
            xaxis_title="Date", yaxis_title="Pearson r",
            template=CHART_THEME, height=320,
            title="Rolling 20-Day Correlation: Sentiment vs. Next-Day Return",
        )
        st.plotly_chart(fig_roll, use_container_width=True)
        st.caption(
            "A positive value means higher sentiment tended to precede higher returns in that window. "
            "The relationship is unstable — typical for financial markets."
        )


# ── Tab 5 — Headlines ─────────────────────────────────────────────────────────
with tab5:
    st.markdown(f"### {current_ticker} — Scored Headlines")

    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
        sentiment_filter = st.multiselect(
            "Filter by sentiment",
            ["positive", "negative", "neutral"],
            default=["positive", "negative", "neutral"],
        )
    with col_f2:
        keyword = st.text_input("Search headlines", placeholder="e.g. earnings, AI, layoffs …")

    filtered = df_scored[df_scored["sentiment_label"].isin(sentiment_filter)].copy()
    if keyword:
        filtered = filtered[filtered["headline"].str.contains(keyword, case=False, na=False)]

    filtered = filtered.copy()
    filtered["date"]            = pd.to_datetime(filtered["date"]).dt.strftime("%Y-%m-%d")
    filtered["sentiment_score"] = filtered["sentiment_score"].round(3)
    filtered["confidence"]      = filtered["confidence"].round(3)

    st.caption(f"Showing {len(filtered)} of {n_total} headlines")
    st.dataframe(
        filtered[["date", "headline", "sentiment_label", "sentiment_score",
                  "prob_positive", "prob_negative", "prob_neutral", "confidence", "source"]]
        .sort_values("date", ascending=False)
        .reset_index(drop=True),
        use_container_width=True, hide_index=True, height=500,
    )

    st.download_button(
        "⬇️ Download as CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name=f"{current_ticker}_sentiment.csv",
        mime="text/csv",
    )


# ── Tab 6 — About ─────────────────────────────────────────────────────────────
with tab6:
    st.markdown("""
### About This Project

**Financial News Sentiment Analysis** is a portfolio data science project that applies
a fine-tuned transformer model to measure the sentiment of financial news headlines,
tracks how sentiment evolves over time, and investigates its statistical relationship
with stock price movements.

---

#### Model — ProsusAI/FinBERT

FinBERT is a BERT-base model fine-tuned on ~10 000 financial news articles, earnings
call transcripts, and analyst reports. It is more accurate on financial language than
general-purpose sentiment models.

| Label    | Meaning                              |
|----------|--------------------------------------|
| Positive | Optimistic / bullish tone            |
| Negative | Pessimistic / bearish tone           |
| Neutral  | Factual or non-directional reporting |

**Sentiment Score** = P(positive) − P(negative) ∈ [−1, +1]

---

#### Data Sources — US Market

| Source              | Key required? | Notes                           |
|---------------------|:-------------:|--------------------------------|
| Alpha Vantage API   | Yes (free)    | 25 requests/day free tier       |
| yfinance news       | No            | ~20 recent headlines per ticker |
| Built-in sample set | No            | 240 curated headlines, offline  |
| yfinance prices     | No            | Full OHLCV history              |

#### Data Sources — BIST Market

| Source              | Key required? | Notes                                        |
|---------------------|:-------------:|---------------------------------------------|
| KAP RSS             | No            | Public company disclosures (kap.org.tr)      |
| Bigpara RSS         | No            | Hurriyet financial news feed                 |
| Investing.com TR    | No            | Turkish market news RSS                      |
| BIST sample set     | No            | 90+ Turkish-language headlines, offline      |
| yfinance prices     | No            | Full OHLCV via Yahoo Finance `.IS` suffix    |

> ⚠️ **BIST sentiment note:** FinBERT is trained on English text. Sentiment scores for
> Turkish headlines are approximate until a Turkish financial NLP model is integrated (Phase 3).

---

#### Disclaimer

⚠️ This project is for **educational and portfolio purposes only**.
Nothing here constitutes financial advice or a trading signal.

---

**Author:** Elif Naz Turan
""")
