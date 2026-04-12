"""
Sentiment trend aggregation and stock-sentiment correlation analysis.

Usage:
    from src.analysis import aggregate_sentiment, compute_correlation
    daily  = aggregate_sentiment(df_scored, freq="D")
    weekly = aggregate_sentiment(df_scored, freq="W")
    corr   = compute_correlation(daily, df_prices)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def aggregate_sentiment(
    df: pd.DataFrame,
    freq: str = "D",
) -> pd.DataFrame:
    """
    Aggregate headline-level sentiment scores to a time series.

    Parameters
    ----------
    df   : scored DataFrame with columns [date, sentiment_score,
           sentiment_label, prob_positive, prob_negative, prob_neutral]
    freq : 'D' for daily, 'W' for weekly, 'ME' for monthly

    Returns
    -------
    DataFrame indexed by period with aggregated sentiment metrics.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    agg = (
        df.groupby(pd.Grouper(key="date", freq=freq))
        .agg(
            mean_score      =("sentiment_score",  "mean"),
            median_score    =("sentiment_score",  "median"),
            std_score       =("sentiment_score",  "std"),
            n_headlines     =("sentiment_score",  "count"),
            n_positive      =("sentiment_label",  lambda s: (s == "positive").sum()),
            n_negative      =("sentiment_label",  lambda s: (s == "negative").sum()),
            n_neutral       =("sentiment_label",  lambda s: (s == "neutral").sum()),
            mean_confidence =("confidence",       "mean"),
        )
        .reset_index()
    )

    # Derived fields
    total = agg["n_headlines"].replace(0, np.nan)
    agg["pct_positive"] = agg["n_positive"] / total * 100
    agg["pct_negative"] = agg["n_negative"] / total * 100
    agg["pct_neutral"]  = agg["n_neutral"]  / total * 100

    # Smoothed score (7-period rolling mean, forward-fill gaps first)
    agg = agg[agg["n_headlines"] > 0].copy()
    agg["score_ma7"] = agg["mean_score"].rolling(window=7, min_periods=1).mean()

    # Bullish/bearish regime flag
    agg["regime"] = agg["score_ma7"].apply(
        lambda x: "bullish" if x > 0.05 else ("bearish" if x < -0.05 else "neutral")
    )

    return agg.sort_values("date").reset_index(drop=True)


def merge_sentiment_prices(
    sentiment_daily: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join daily sentiment onto price data on the Date column.

    Returns a merged DataFrame aligned on business days.
    """
    s = sentiment_daily.rename(columns={"date": "Date"})
    s["Date"] = pd.to_datetime(s["Date"]).dt.normalize()
    p = prices.copy()
    p["Date"] = pd.to_datetime(p["Date"]).dt.normalize()

    merged = pd.merge(p, s, on="Date", how="left")
    merged["mean_score"] = merged["mean_score"].ffill()
    return merged.dropna(subset=["mean_score", "return_1d"])


def compute_correlation(
    merged: pd.DataFrame,
    sentiment_col: str = "mean_score",
    return_col:    str = "return_next",
) -> dict:
    """
    Compute Pearson and Spearman correlations between sentiment and stock returns.

    Uses next-day return (return_next) by default so the comparison is
    forward-looking: does today's news sentiment predict tomorrow's move?

    ⚠️ Correlation does not imply causation. This analysis is exploratory only.
    """
    df = merged[[sentiment_col, return_col]].dropna()

    if len(df) < 10:
        return {
            "n_observations": len(df),
            "pearson_r":  None, "pearson_p":  None,
            "spearman_r": None, "spearman_p": None,
            "note": "Insufficient data for correlation.",
        }

    pearson_r,  pearson_p  = stats.pearsonr(df[sentiment_col], df[return_col])
    spearman_r, spearman_p = stats.spearmanr(df[sentiment_col], df[return_col])

    return {
        "n_observations": len(df),
        "pearson_r":      round(pearson_r,  4),
        "pearson_p":      round(pearson_p,  4),
        "spearman_r":     round(spearman_r, 4),
        "spearman_p":     round(spearman_p, 4),
        "pearson_sig":    pearson_p  < 0.05,
        "spearman_sig":   spearman_p < 0.05,
    }


def rolling_correlation(
    merged: pd.DataFrame,
    window: int = 20,
    sentiment_col: str = "mean_score",
    return_col:    str = "return_next",
) -> pd.Series:
    """Compute rolling Pearson correlation between sentiment and next-day return."""
    return (
        merged[[sentiment_col, return_col]]
        .dropna()
        .apply(lambda col: col)
        [sentiment_col]
        .rolling(window)
        .corr(merged[return_col])
    )
