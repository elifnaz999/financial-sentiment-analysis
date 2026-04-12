"""
FinBERT sentiment scoring.

Model  : ProsusAI/finbert
         A BERT model fine-tuned on financial communications (earnings calls,
         analyst reports, financial news) — more accurate on finance text than
         general-purpose models like distilbert-base-uncased-finetuned-sst-2.

Labels : positive | negative | neutral
Score  : P(positive) − P(negative)  ∈ [−1, +1]
         +1 = strongly positive, −1 = strongly negative, 0 = neutral

Usage:
    from src.sentiment_model import score_dataframe
    df_scored = score_dataframe(df, text_col="headline")
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification

MODEL_NAME = "ProsusAI/finbert"
_LABEL_MAP = {0: "positive", 1: "negative", 2: "neutral"}

_tokenizer = None
_model     = None


def load_finbert() -> tuple:
    """Load FinBERT tokenizer and model (lazy, cached after first call)."""
    global _tokenizer, _model
    if _tokenizer is None:
        print(f"  Loading {MODEL_NAME} from HuggingFace Hub …")
        _tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        _model     = BertForSequenceClassification.from_pretrained(MODEL_NAME)
        _model.eval()
        print("  FinBERT ready.")
    return _tokenizer, _model


def _score_batch(texts: list[str]) -> list[dict]:
    tokenizer, model = load_finbert()
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).numpy()

    results = []
    for p in probs:
        label = _LABEL_MAP[int(np.argmax(p))]
        results.append({
            "sentiment_label": label,
            "sentiment_score": round(float(p[0] - p[1]), 4),   # P(pos) - P(neg)
            "prob_positive":   round(float(p[0]), 4),
            "prob_negative":   round(float(p[1]), 4),
            "prob_neutral":    round(float(p[2]), 4),
            "confidence":      round(float(np.max(p)), 4),
        })
    return results


def score_headline(text: str) -> dict:
    """Score a single headline. Returns dict with label + probabilities."""
    return _score_batch([text])[0]


def score_dataframe(
    df: pd.DataFrame,
    text_col: str = "headline",
    batch_size: int = 16,
) -> pd.DataFrame:
    """
    Score all headlines in a DataFrame using FinBERT.

    Parameters
    ----------
    df         : DataFrame containing a column with headline text
    text_col   : name of the text column
    batch_size : number of texts per forward pass (lower if OOM)

    Returns
    -------
    Input DataFrame with sentiment columns appended.
    """
    texts   = df[text_col].fillna("").tolist()
    records = []

    for i in range(0, len(texts), batch_size):
        batch   = texts[i : i + batch_size]
        records.extend(_score_batch(batch))

    scored = df.copy().reset_index(drop=True)
    scored = pd.concat([scored, pd.DataFrame(records)], axis=1)
    return scored


# Colour / emoji helpers used by both Streamlit and notebooks
LABEL_COLORS  = {"positive": "#06d6a0", "negative": "#ef233c", "neutral": "#8ecae6"}
LABEL_EMOJIS  = {"positive": "🟢", "negative": "🔴", "neutral": "🔵"}
LABEL_NUMERIC = {"positive": 1, "negative": -1, "neutral": 0}
