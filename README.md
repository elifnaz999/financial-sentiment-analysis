# Financial News Sentiment Analysis

**Author:** Elif Naz Turan

An end-to-end data science portfolio project that analyzes the sentiment of financial news headlines using the **FinBERT** transformer model, visualizes sentiment trends over time, and investigates the correlation between news sentiment and stock price movements — all served through an interactive **Streamlit dashboard**.

---

## Table of Contents
- [Business Problem](#business-problem)
- [Why Sentiment Analysis in Finance](#why-sentiment-analysis-in-finance)
- [Model](#model)
- [Data Sources](#data-sources)
- [Stock Price Integration](#stock-price-integration)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Sample Workflow](#sample-workflow)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)

---

## Business Problem

Financial markets are driven partly by information — and news sentiment is one of the fastest-moving signals available. Being able to automatically measure the emotional tone of thousands of headlines per day, track how that tone shifts over time, and compare it to stock price behaviour gives analysts and researchers a structured view of market narrative momentum.

This project answers three questions:
1. What is the prevailing sentiment in the news for a given stock over a selected period?
2. How does that sentiment evolve over time, and what regimes (bullish/bearish/neutral) emerge?
3. Is there a measurable statistical relationship between news sentiment and subsequent stock returns?

---

## Why Sentiment Analysis in Finance

- **Speed**: Automated NLP processes hundreds of headlines in seconds vs. hours for human analysts.
- **Consistency**: A model applies the same scoring logic uniformly; human annotation varies.
- **Scale**: A single model can monitor hundreds of tickers simultaneously.
- **Signal research**: Sentiment indices built from news have been studied in academic literature as potential leading indicators of price movements (Tetlock, 2007; Loughran & McDonald, 2011).

> ⚠️ **Important disclaimer**: News sentiment is a weak and noisy signal in efficient markets. Correlation between sentiment and returns should not be interpreted as causation, and this project should **not** be used as a basis for trading decisions.

---

## Model

**[ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)**

FinBERT is a BERT-base-uncased model fine-tuned on approximately 10,000 financial news sentences, earnings call transcripts, and analyst reports. It was chosen over a general-purpose sentiment model (e.g. `distilbert-sst-2`) because:

- It is trained on **domain-specific financial language** where words like "volatile", "correction", and "headwinds" carry different connotations than in everyday English.
- It produces three classes — **positive, negative, neutral** — which is more appropriate for financial reporting than a binary positive/negative split.
- It outputs **calibrated probability scores**, not just class labels.

**Sentiment Score** = P(positive) − P(negative) ∈ [−1, +1]

| Range | Interpretation |
|-------|---------------|
| +0.05 to +1.0 | Bullish / positive tone |
| −0.05 to +0.05 | Neutral / ambiguous |
| −1.0 to −0.05 | Bearish / negative tone |

---

## Data Sources

| Source | Description | API Key Required |
|--------|-------------|-----------------|
| **Alpha Vantage News Sentiment API** | Real-time financial news with ticker filtering. Free tier: 25 req/day. | Yes (`.env`) |
| **yfinance `ticker.news`** | Recent news headlines linked to a ticker. ~20 most recent items. | No |
| **Built-in sample dataset** | 240 curated realistic headlines across major tickers. Used as offline fallback. | No |

The project automatically tries sources in priority order (`alphavantage → yfinance → sample`) so it **always runs**, even without any API key.

To use Alpha Vantage:
```bash
cp .env.example .env
# Edit .env and add your key
ALPHA_VANTAGE_API_KEY=your_key_here
```

---

## Stock Price Integration

Stock price data is fetched via **yfinance** (no key required). For each ticker, the module downloads:
- Daily OHLCV bars for the selected date range
- 1-day percentage return (`return_1d`)
- Next-day return (`return_next`) — used as the forward-looking correlation target
- Normalised price series (base = 100 at start) for trend comparison

**Supported tickers** (examples): `AAPL`, `MSFT`, `TSLA`, `GOOGL`, `AMZN`, `NVDA` — and any other valid Yahoo Finance ticker.

**Correlation method:**

```
Today's mean sentiment score → Tomorrow's stock return
```

Both Pearson (linear) and Spearman (rank-based) correlations are computed with p-values. A 20-day rolling correlation is also plotted to show how the relationship evolves over time.

---

## Project Structure

```
financial-sentiment/
├── app/
│   └── streamlit_app.py        # interactive Streamlit dashboard
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # news fetching (Alpha Vantage / yfinance / sample)
│   ├── sentiment_model.py      # FinBERT loading + headline scoring
│   ├── stock_data.py           # yfinance OHLCV fetching + return computation
│   ├── analysis.py             # sentiment aggregation + correlation analysis
│   └── utils.py                # file I/O, date helpers, Plotly chart builders
├── data/
│   ├── raw/                    # downloaded raw news files
│   └── processed/              # scored CSVs ({ticker}_sentiment.csv)
├── outputs/
│   ├── charts/                 # saved chart images
│   └── tables/                 # exported result tables
├── notebooks/
│   └── 01_sentiment_analysis.ipynb   # full EDA and analysis notebook
├── README.md
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Installation

**Prerequisites:** Python 3.9+

```bash
# 1. Navigate to the project folder
cd financial-sentiment

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Set up API key
cp .env.example .env
# Edit .env and add ALPHA_VANTAGE_API_KEY=your_key_here
```

> **Note:** The first run downloads the FinBERT model weights (~400 MB) from HuggingFace Hub. Subsequent runs use the local cache.

---

## How to Run

### Option A — Streamlit Dashboard (recommended)
```bash
streamlit run app/streamlit_app.py
```
Then open `http://localhost:8501` in your browser.

### Option B — Jupyter Notebook
```bash
jupyter notebook notebooks/01_sentiment_analysis.ipynb
```

### Option C — Python scripts directly
```python
from src.data_loader     import load_news
from src.sentiment_model import score_dataframe
from src.stock_data      import load_prices
from src.analysis        import aggregate_sentiment, compute_correlation, merge_sentiment_prices

# Fetch + score
news   = load_news("AAPL", days=180)
scored = score_dataframe(news)

# Aggregate + correlate
daily   = aggregate_sentiment(scored, freq="D")
prices  = load_prices("AAPL", days=180)
merged  = merge_sentiment_prices(daily, prices)
corr    = compute_correlation(merged)
print(corr)
```

---

## Streamlit Dashboard

The dashboard has 6 tabs:

| Tab | Content |
|-----|---------|
| **Distribution** | Sentiment counts, pie chart, confidence histogram, probability scatter |
| **Sentiment Trends** | Daily/weekly score with 7-period MA, stacked composition bars, regime table |
| **Stock Price** | OHLC candlestick, volume bar, normalised price vs. sentiment overlay |
| **Correlation** | Scatter with OLS trendline, Pearson/Spearman stats, 20-day rolling correlation |
| **Headlines** | Searchable/filterable headline table with download button |
| **About** | Model explanation, data source details, methodology notes |

---

## Sample Workflow

```
1. Open the dashboard: streamlit run app/streamlit_app.py
2. Select ticker: AAPL
3. Select date range: last 6 months
4. Click "Analyze Sentiment"
   → FinBERT scores all headlines
   → Results cached to data/processed/aapl_sentiment.csv
5. Explore Distribution tab: 55% neutral, 28% positive, 17% negative
6. Explore Trends tab: sentiment dipped sharply in Feb, recovered in March
7. Explore Stock Price tab: price overlay shows price rose while sentiment improved
8. Explore Correlation tab: Pearson r = +0.12, p = 0.08 (not significant)
9. Download filtered results from Headlines tab
```

---

## Limitations

- **Sample dataset**: The built-in offline dataset is simulated from 240 curated headlines. Real results require a live news API.
- **Free API limits**: Alpha Vantage free tier is limited to 25 requests per day.
- **Short history**: yfinance fallback returns only ~20 recent headlines, limiting trend analysis.
- **Correlation is weak**: Academic research consistently shows that news sentiment has limited predictive power in liquid markets. Results here are exploratory.
- **Model limitations**: FinBERT may misclassify very short headlines, idiomatic financial language, or sarcastic phrasing.
- **Lookahead bias**: The correlation analysis uses close-to-close returns; intraday timing of news publication is not considered.

---

## Future Improvements

- **Finer-grained sentiment**: Entity-level sentiment (e.g. positive about revenue but negative about guidance).
- **Multi-ticker comparison**: Side-by-side sentiment comparison across a sector.
- **Named entity recognition**: Automatically extract mentioned companies, people, and events.
- **Earnings event overlay**: Mark earnings dates on the sentiment trend chart.
- **Granger causality test**: A more rigorous test of whether sentiment leads or lags returns.
- **Extended news coverage**: Integrate Reddit (r/wallstreetbets), SEC filings, or earnings call transcripts.
- **Alert system**: Notify when sentiment crosses a threshold (e.g. drops 2 standard deviations in a day).

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
| [pandas](https://pandas.pydata.org/) / [numpy](https://numpy.org/) | Data manipulation |
| [scipy](https://scipy.org/) | Pearson and Spearman correlation |
| [matplotlib](https://matplotlib.org/) / [seaborn](https://seaborn.pydata.org/) | Notebook visualizations |

---

## License

Code released under the MIT License.  
FinBERT model weights are subject to the [HuggingFace model license](https://huggingface.co/ProsusAI/finbert).
