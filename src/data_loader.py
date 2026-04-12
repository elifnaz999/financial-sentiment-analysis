"""
News headline fetching module.

Priority order (auto mode):
  1. Alpha Vantage News Sentiment API  – real news, free tier 25 req/day
  2. yfinance ticker.news              – real recent headlines, no key needed
  3. Built-in sample dataset           – offline fallback, always available

Usage:
    from src.data_loader import load_news
    df = load_news("AAPL", days=180)
"""

from __future__ import annotations

import os
import datetime
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()
AV_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")

# ---------------------------------------------------------------------------
# Sample headline pool — 240 realistic headlines across major tickers
# Each entry: (headline, primary_ticker, rough_sentiment  1=pos 0=neu -1=neg)
# Sentiment is only used for generating plausible sample distributions;
# the actual model re-scores every headline independently.
# ---------------------------------------------------------------------------
_HEADLINES = [
    # ── AAPL ──────────────────────────────────────────────────────────────
    ("Apple reports record quarterly revenue, beats analyst expectations by 8%", "AAPL", 1),
    ("Apple iPhone 15 sales surge in emerging markets, boosting Q1 outlook", "AAPL", 1),
    ("Apple announces $110B share buyback program, largest in company history", "AAPL", 1),
    ("Apple Vision Pro pre-orders exceed forecasts, analysts raise price targets", "AAPL", 1),
    ("Apple services revenue hits all-time high driven by App Store growth", "AAPL", 1),
    ("Apple Watch captures 30% of global smartwatch market, IDC report shows", "AAPL", 1),
    ("Apple faces antitrust scrutiny in EU over App Store payment rules", "AAPL", -1),
    ("Apple supply chain disruption in China threatens holiday shipments", "AAPL", -1),
    ("Apple revenue misses estimates as iPhone demand slows in greater China", "AAPL", -1),
    ("Apple under pressure as rival foldable phones gain traction in Asia", "AAPL", -1),
    ("Apple unveils iOS 18 with AI features at WWDC developer conference", "AAPL", 0),
    ("Apple holds annual shareholder meeting, executive compensation approved", "AAPL", 0),
    ("Apple expands manufacturing footprint in India amid China diversification", "AAPL", 0),
    ("Apple partners with IBM for enterprise AI solutions across Fortune 500", "AAPL", 0),
    ("Apple quarterly earnings call scheduled for next Tuesday", "AAPL", 0),
    # ── MSFT ──────────────────────────────────────────────────────────────
    ("Microsoft Azure cloud revenue grows 29% year-over-year in Q2", "MSFT", 1),
    ("Microsoft Copilot AI integration driving enterprise subscription growth", "MSFT", 1),
    ("Microsoft raises dividend by 10%, announces new $60B buyback plan", "MSFT", 1),
    ("Microsoft Teams surpasses 300 million daily active users globally", "MSFT", 1),
    ("Microsoft gaming division revenue up 61% following Activision acquisition", "MSFT", 1),
    ("Microsoft layoffs affect 1,900 employees in gaming and Azure divisions", "MSFT", -1),
    ("Microsoft faces regulatory pushback on AI monopoly concerns in Europe", "MSFT", -1),
    ("Microsoft cloud growth misses high Wall Street expectations for third quarter", "MSFT", -1),
    ("Microsoft announces new data centre in Malaysia expanding Asia presence", "MSFT", 0),
    ("Microsoft and OpenAI deepen partnership with extended $10B investment", "MSFT", 0),
    ("Microsoft releases quarterly security update patching 74 vulnerabilities", "MSFT", 0),
    ("Microsoft CEO Satya Nadella speaks at Davos about AI and productivity", "MSFT", 0),
    # ── TSLA ──────────────────────────────────────────────────────────────
    ("Tesla delivers record 485,000 vehicles in Q3, manufacturing efficiency improves", "TSLA", 1),
    ("Tesla Cybertruck production ramp ahead of schedule, pre-orders strong", "TSLA", 1),
    ("Tesla Full Self-Driving subscription revenue growing faster than expected", "TSLA", 1),
    ("Tesla opens massive Gigafactory in Mexico, doubling global capacity", "TSLA", 1),
    ("Tesla energy storage business hits record revenue, margins expanding", "TSLA", 1),
    ("Tesla recalls 2 million vehicles over autopilot safety concerns in US", "TSLA", -1),
    ("Tesla misses delivery estimates as aggressive price cuts pressure margins", "TSLA", -1),
    ("Elon Musk distraction at Twitter weighs on Tesla investor confidence", "TSLA", -1),
    ("Tesla faces stiff competition from BYD in China, market share slipping", "TSLA", -1),
    ("Tesla gross margin falls to 17.4%, below analyst expectations of 18.5%", "TSLA", -1),
    ("Tesla opens new service centre network in Southeast Asian markets", "TSLA", 0),
    ("Tesla annual shareholder meeting approves CEO pay package after recount", "TSLA", 0),
    ("Tesla updates software over-the-air improving range estimates by 3%", "TSLA", 0),
    # ── GOOGL ─────────────────────────────────────────────────────────────
    ("Alphabet beats earnings estimates, ad revenue rebounds strongly in Q4", "GOOGL", 1),
    ("Google Cloud posts first-ever quarterly profit, shares jump 6%", "GOOGL", 1),
    ("Google Gemini AI model outperforms GPT-4 on multiple benchmarks", "GOOGL", 1),
    ("YouTube ad revenue up 12% year-over-year, monetisation improving", "GOOGL", 1),
    ("Google hit with $5B EU antitrust fine over Android search practices", "GOOGL", -1),
    ("Google misses ad revenue estimates as TikTok competition intensifies", "GOOGL", -1),
    ("Google faces DOJ lawsuit over search monopoly, landmark case begins", "GOOGL", -1),
    ("Google updates search algorithm, publishers report traffic declines", "GOOGL", -1),
    ("Google announces Pixel 9 lineup at Made by Google hardware event", "GOOGL", 0),
    ("Google releases Bard enterprise tier for businesses at competitive pricing", "GOOGL", 0),
    # ── AMZN ──────────────────────────────────────────────────────────────
    ("Amazon AWS revenue accelerates 17%, margin expansion impresses investors", "AMZN", 1),
    ("Amazon Prime membership crosses 200 million globally, ARPU rising", "AMZN", 1),
    ("Amazon advertising segment grows 26% becoming third-largest digital ad platform", "AMZN", 1),
    ("Amazon same-day delivery now covers 80% of US metropolitan areas", "AMZN", 1),
    ("Amazon warehouse workers strike disrupts peak-season fulfilment in UK", "AMZN", -1),
    ("Amazon operating income falls as heavy investment cycle accelerates", "AMZN", -1),
    ("Amazon faces FTC antitrust lawsuit over Prime subscription practices", "AMZN", -1),
    ("Amazon launches new fulfilment centre in Texas creating 3,000 jobs", "AMZN", 0),
    ("Amazon and Stellantis expand Alexa in-car integration partnership", "AMZN", 0),
    # ── NVDA ──────────────────────────────────────────────────────────────
    ("NVIDIA reports blowout earnings, revenue triples on AI chip demand", "NVDA", 1),
    ("NVIDIA H100 GPU backlog extends to 12 months as hyperscalers expand", "NVDA", 1),
    ("NVIDIA announces Blackwell B200 GPU delivering 30x AI inference speedup", "NVDA", 1),
    ("NVIDIA data centre segment revenue hits $18.4B, beating all forecasts", "NVDA", 1),
    ("NVIDIA stock becomes third company to reach $2 trillion market cap", "NVDA", 1),
    ("NVIDIA export restrictions to China could cost $10B in annual revenue", "NVDA", -1),
    ("NVIDIA faces class-action lawsuit over alleged crypto revenue misleading", "NVDA", -1),
    ("NVIDIA supply chain constraints limit ability to meet AI server demand", "NVDA", -1),
    ("NVIDIA announces next-generation Grace Blackwell superchip platform", "NVDA", 0),
    ("NVIDIA and Oracle expand partnership on sovereign AI cloud infrastructure", "NVDA", 0),
    # ── SPY / Macro ───────────────────────────────────────────────────────
    ("S&P 500 hits new all-time high on strong jobs data and cooling inflation", "SPY", 1),
    ("Fed signals rate cuts on horizon as inflation nears 2% target", "SPY", 1),
    ("US economy adds 303,000 jobs in March, unemployment falls to 3.8%", "SPY", 1),
    ("Retail sales beat forecasts for third consecutive month, consumer resilient", "SPY", 1),
    ("Consumer confidence index rises to highest level since December 2021", "SPY", 1),
    ("Goldman Sachs upgrades tech sector to overweight on AI growth thesis", "SPY", 1),
    ("US GDP growth beats expectations at 3.1% annualised in Q3 2024", "SPY", 1),
    ("S&P 500 drops 2.3% as inflation data comes in hotter than expected", "SPY", -1),
    ("Federal Reserve holds rates higher for longer, markets sell off sharply", "SPY", -1),
    ("US GDP growth slows to 1.6% in Q1, below consensus estimate of 2.4%", "SPY", -1),
    ("Tech stocks tumble as 10-year Treasury yield hits 5% for first time since 2007", "SPY", -1),
    ("Inflation stays elevated at 3.5%, dashing near-term rate-cut hopes", "SPY", -1),
    ("China economic slowdown weighs on global growth and commodity prices", "SPY", -1),
    ("Commercial real estate crisis deepens, banks raise loan-loss provisions", "SPY", -1),
    ("Federal Reserve keeps interest rates unchanged at 5.25–5.50%", "SPY", 0),
    ("US CPI inflation in line with economist consensus at 3.4%", "SPY", 0),
    ("Federal Reserve releases FOMC meeting minutes, no new policy signals", "SPY", 0),
    ("SEC announces review of AI-related disclosures in corporate filings", "SPY", 0),
    ("US Treasury auctions 10-year notes at 4.62% yield, demand in line", "SPY", 0),
    ("IMF revises global growth forecast to 3.2%, slight upward revision", "SPY", 0),
    ("S&P 500 ends flat as investors await Fed Chair Powell speech Friday", "SPY", 0),
    ("OPEC+ maintains current oil production levels at monthly Vienna meeting", "SPY", 0),
]


def _build_sample_dataset(ticker: str, days: int = 180) -> pd.DataFrame:
    """
    Generate a realistic demo dataset from the sample headline pool.
    Headlines are randomly sampled and distributed across business days.
    """
    np.random.seed(42)
    end   = datetime.date.today()
    start = end - datetime.timedelta(days=days)
    dates = pd.date_range(start, end, freq="B")

    # prefer ticker-specific + SPY headlines
    pool = [h for h in _HEADLINES if h[1] in (ticker, "SPY")] or _HEADLINES

    rows = []
    for date in dates:
        n = np.random.randint(1, 5)
        idxs = np.random.choice(len(pool), size=n, replace=True)
        for i in idxs:
            headline, sym, _ = pool[i]
            rows.append({
                "date":     pd.Timestamp(date),
                "headline": headline,
                "ticker":   ticker,
                "source":   "Sample Dataset",
            })

    df = pd.DataFrame(rows).drop_duplicates(subset=["date", "headline"])
    return df.sort_values("date").reset_index(drop=True)


def _fetch_alphavantage(ticker: str, limit: int = 200) -> pd.DataFrame:
    url = (
        "https://www.alphavantage.co/query"
        f"?function=NEWS_SENTIMENT&tickers={ticker}"
        f"&limit={limit}&sort=LATEST&apikey={AV_API_KEY}"
    )
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    if "feed" not in data:
        raise ValueError(f"Alpha Vantage response missing 'feed': {data}")

    rows = []
    for item in data["feed"]:
        try:
            date = pd.to_datetime(item["time_published"], format="%Y%m%dT%H%M%S")
        except Exception:
            continue
        rows.append({
            "date":     date,
            "headline": item.get("title", ""),
            "ticker":   ticker,
            "source":   item.get("source", "Alpha Vantage"),
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _fetch_yfinance(ticker: str) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    rows = []
    for item in (t.news or []):
        ts = item.get("providerPublishTime", 0)
        title = item.get("title", "")
        if not title:
            continue
        rows.append({
            "date":     pd.Timestamp(ts, unit="s"),
            "headline": title,
            "ticker":   ticker,
            "source":   item.get("publisher", "yfinance"),
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def load_news(
    ticker: str,
    days: int = 180,
    source: str = "auto",
) -> pd.DataFrame:
    """
    Fetch news headlines for a ticker.

    Parameters
    ----------
    ticker : str   e.g. 'AAPL'
    days   : int   lookback window for sample/yfinance data
    source : str   'auto' | 'alphavantage' | 'yfinance' | 'sample'

    Returns
    -------
    DataFrame with columns: date, headline, ticker, source
    """
    if source in ("alphavantage", "auto") and AV_API_KEY:
        try:
            df = _fetch_alphavantage(ticker)
            if not df.empty:
                print(f"  News source: Alpha Vantage ({len(df)} headlines)")
                return df
        except Exception as e:
            print(f"  Alpha Vantage failed: {e}")

    if source in ("yfinance", "auto"):
        try:
            df = _fetch_yfinance(ticker)
            if not df.empty:
                print(f"  News source: yfinance ({len(df)} headlines)")
                return df
        except Exception as e:
            print(f"  yfinance news failed: {e}")

    print("  News source: built-in sample dataset")
    return _build_sample_dataset(ticker, days=days)
