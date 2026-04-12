# Financial News Sentiment Analysis

**Author:** Elif Naz Turan

An end-to-end data science portfolio project that analyzes the sentiment of financial
news headlines using the **FinBERT** transformer model, visualizes sentiment trends,
and investigates the correlation between news sentiment and stock price movements —
all served through an interactive **Streamlit dashboard**.

🚀 Live Demo: [Try the App Here](https://financial-sentiment-analysis-dkzxfbszakm8xn5hdmio4x.streamlit.app)

---

## Table of Contents
- [Overview](#overview)
- [Business Problem](#business-problem)
- [Model](#model)
- [Data Sources](#data-sources)
- [Project Structure](#project-structure)
- [Local Installation](#local-installation)
- [Deploy on Streamlit Community Cloud](#deploy-on-streamlit-community-cloud)
- [Dashboard Features](#dashboard-features)
- [Stock Price Correlation](#stock-price-correlation)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)

---

## Overview

Automated NLP pipeline that:
1. Fetches financial news headlines (Alpha Vantage API / yfinance / built-in sample)
2. Scores each headline with FinBERT → `positive`, `negative`, `neutral` + probability scores
3. Aggregates daily/weekly sentiment trends with a 7-period moving average
4. Fetches stock OHLCV data via yfinance and overlays it with sentiment
5. Computes Pearson + Spearman correlation (sentiment → next-day return)
6. Presents everything in an interactive Streamlit dashboard

---

## Business Problem

Financial markets react to information. Being able to automatically measure the tone
of thousands of headlines per day gives analysts a structured view of market narrative momentum.

> ⚠️ Correlation between news sentiment and stock returns is weak in efficient markets.
> This project is for **educational purposes only** — not financial advice.

---

## Model

**[ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)**

FinBERT is a BERT-base model fine-tuned on ~10 000 financial news articles, earnings
call transcripts, and analyst reports. It outperforms general-purpose sentiment models
on financial text because domain-specific terms ("headwinds", "correction", "beat") carry
different connotations than in everyday English.

| Label    | Meaning                    | Score contribution |
|----------|----------------------------|--------------------|
| Positive | Bullish / optimistic       | +P(positive)       |
| Negative | Bearish / pessimistic      | −P(negative)       |
| Neutral  | Factual / non-directional  | 0                  |

**Sentiment Score** = P(positive) − P(negative) ∈ [−1, +1]

---

## Data Sources

| Source | Description | API Key |
|--------|-------------|---------|
| Alpha Vantage News API | Real-time financial news, ticker-filtered | Yes (free, 25 req/day) |
| yfinance `ticker.news` | Recent headlines per ticker, ~20 items | No |
| Built-in sample dataset | 240 curated headlines across 6 tickers | No |
| yfinance prices | Full OHLCV history | No |

The app tries sources in order: **Alpha Vantage → yfinance → sample**. It works without
any API key using the built-in dataset.

---

## Project Structure

```
financial-sentiment/
├── streamlit_app.py          ← Streamlit Cloud entry point (repo root)
├── requirements.txt
├── .gitignore
├── .env.example
├── README.md
│
├── .streamlit/
│   ├── config.toml           ← Theme + server settings
│   └── secrets.toml.example  ← Template for Streamlit Cloud secrets
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py        ← News fetching (Alpha Vantage / yfinance / sample)
│   ├── sentiment_model.py    ← FinBERT loading + batch inference
│   ├── stock_data.py         ← yfinance OHLCV + return computation
│   ├── analysis.py           ← Sentiment aggregation + correlation
│   └── utils.py              ← File I/O + Plotly chart builders
│
├── app/
│   └── streamlit_app.py      ← Legacy entry point (local use with sys.path)
│
├── data/
│   ├── raw/                  ← Downloaded raw news (gitignored)
│   └── processed/            ← Scored CSVs, e.g. aapl_sentiment.csv (gitignored)
│
├── outputs/
│   ├── charts/               ← Saved chart images (gitignored)
│   └── tables/               ← Exported tables (gitignored)
│
└── notebooks/
    └── 01_sentiment_analysis.ipynb
```

---

## Local Installation

```bash
# 1. Clone the repo
git clone https://github.com/elifnaz999/financial-sentiment-analysis.git
cd financial-sentiment-analysis

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Add your Alpha Vantage API key
cp .env.example .env
# Edit .env → set ALPHA_VANTAGE_API_KEY=your_key_here

# 5. Run the app
streamlit run streamlit_app.py
```

> **Note:** The first run downloads FinBERT (~440 MB) from HuggingFace Hub.
> Subsequent runs use the local cache (`~/.cache/huggingface/`).

---

## Deploy on Streamlit Community Cloud

### Step 1 — Push your code to GitHub

Make sure your latest code is on GitHub:

```bash
git add .
git commit -m "Ready for Streamlit Cloud deployment"
git push origin main
```

### Step 2 — Create a Streamlit Cloud account

Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with your GitHub account.

### Step 3 — Create a new app

1. Click **"New app"**
2. Fill in the form:

| Field | Value |
|-------|-------|
| **Repository** | `elifnaz999/financial-sentiment-analysis` |
| **Branch** | `main` |
| **Main file path** | `streamlit_app.py` |
| **App URL** | choose a slug, e.g. `financial-sentiment` |

3. Click **"Deploy!"**

Streamlit Cloud will install `requirements.txt` and start the app. The first cold start
takes **3–5 minutes** (installing torch + downloading FinBERT). After that, warm starts
are near-instant.

### Step 4 — (Optional) Add your Alpha Vantage API key

To use real news instead of the sample dataset:

1. In your Streamlit Cloud app, click **⋮ → Settings → Secrets**
2. Paste the following (replace the value with your real key):

```toml
ALPHA_VANTAGE_API_KEY = "your_key_here"
```

3. Click **"Save"**. The app restarts and picks up the key automatically.

> Get a free key at [alphavantage.co](https://www.alphavantage.co/support/#api-key).
> Free tier: 25 requests/day. No credit card required.

### Step 5 — Share your app

Your app will be live at:
```
https://your-slug.streamlit.app
```

Add this URL to your portfolio and GitHub README.

---

## Dashboard Features

| Tab | Content |
|-----|---------|
| **📊 Distribution** | Sentiment counts, pie chart, confidence histogram, probability scatter |
| **📈 Sentiment Trends** | Daily/weekly score + 7-period MA, stacked composition bars, regime table |
| **💹 Stock Price** | OHLC candlestick, volume bar, normalised price vs. sentiment overlay |
| **🔗 Correlation** | Scatter + OLS fit, Pearson/Spearman stats, 20-day rolling correlation |
| **📰 Headlines** | Searchable/filterable scored headline table + CSV download |
| **ℹ️ About** | Model explanation, data source details, methodology notes |

---

## Stock Price Correlation

**Method:**
```
Today's mean sentiment score  →  Tomorrow's stock return
```

Both Pearson (linear) and Spearman (rank-based) correlations are reported with p-values.
A 20-day rolling correlation shows how the relationship evolves over time.

⚠️ Consistent with academic literature (Tetlock 2007; Loughran & McDonald 2011),
the relationship is weak and unstable — markets are largely efficient.

---

## Limitations

- Free Alpha Vantage tier is limited to 25 requests/day
- yfinance fallback returns only ~20 recent headlines per ticker
- FinBERT may misclassify very short headlines or sarcastic phrasing
- Streamlit Cloud free tier: cold start takes ~3–5 min due to model download
- Stock price correlation uses close-to-close returns; intraday news timing is ignored

---

## Future Improvements

- Entity-level sentiment (positive about revenue, negative about guidance)
- Multi-ticker sector comparison dashboard
- Named entity recognition for automatic company/event extraction
- Earnings date overlay on sentiment trend chart
- Granger causality test for more rigorous signal analysis
- Reddit (r/wallstreetbets) and SEC filing integration

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| [ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert) | Financial sentiment classification |
| [Transformers](https://huggingface.co/docs/transformers/) | Model loading and inference |
| [PyTorch](https://pytorch.org/) | Deep learning backend |
| [yfinance](https://pypi.org/project/yfinance/) | Stock price data |
| [Streamlit](https://streamlit.io/) | Interactive dashboard |
| [Plotly](https://plotly.com/python/) | Interactive charts |
| [statsmodels](https://www.statsmodels.org/) | OLS trendline for Plotly |
| [pandas](https://pandas.pydata.org/) / [numpy](https://numpy.org/) | Data manipulation |
| [scipy](https://scipy.org/) | Pearson and Spearman correlation |

---

## License

Code: MIT License.
FinBERT model: [HuggingFace model license](https://huggingface.co/ProsusAI/finbert).
