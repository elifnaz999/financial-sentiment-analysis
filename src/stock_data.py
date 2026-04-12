"""
Stock price fetching and return computation via yfinance.

Usage:
    from src.stock_data import load_prices
    prices = load_prices("AAPL", days=180)
"""

from __future__ import annotations

import datetime
import pandas as pd
import yfinance as yf


def load_prices(
    ticker: str,
    start:  str | None = None,
    end:    str | None = None,
    days:   int = 180,
) -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance.

    Parameters
    ----------
    ticker : str       e.g. 'AAPL'
    start  : str|None  ISO date string; defaults to today − days
    end    : str|None  ISO date string; defaults to today
    days   : int       fallback lookback window when start/end are None

    Returns
    -------
    DataFrame with columns:
        Date, Open, High, Low, Close, Volume,
        return_1d      (same-day % change),
        return_next    (next-day % change — the predictive target),
        price_norm     (Close normalised to first observation = 100)
    """
    if start is None:
        start = (datetime.date.today() - datetime.timedelta(days=days)).isoformat()
    if end is None:
        end = datetime.date.today().isoformat()

    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No price data returned for ticker '{ticker}' ({start} → {end})")

    df = raw.reset_index()
    # flatten MultiIndex columns that yfinance sometimes returns
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()

    df = df.sort_values("Date").reset_index(drop=True)
    df["return_1d"]   = df["Close"].pct_change()
    df["return_next"] = df["return_1d"].shift(-1)
    df["price_norm"]  = df["Close"] / df["Close"].iloc[0] * 100

    return df


def price_summary(df: pd.DataFrame) -> dict:
    """Return a dict of summary statistics for the price series."""
    total_return = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
    volatility   = df["return_1d"].std() * (252 ** 0.5) * 100
    max_drawdown = ((df["Close"] / df["Close"].cummax()) - 1).min() * 100

    return {
        "start_price":    round(float(df["Close"].iloc[0]),  2),
        "end_price":      round(float(df["Close"].iloc[-1]), 2),
        "total_return":   round(total_return, 2),
        "ann_volatility": round(volatility, 2),
        "max_drawdown":   round(max_drawdown, 2),
        "trading_days":   len(df),
    }
