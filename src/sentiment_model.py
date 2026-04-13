"""
Sentiment scoring — FinBERT (US) and Turkish BERT (BIST).

US market
  Model  : ProsusAI/finbert
           BERT fine-tuned on financial communications (earnings calls,
           analyst reports, financial news).
  Labels : positive | negative | neutral  (fixed, model-intrinsic)

BIST market
  Model  : savasy/bert-base-turkish-sentiment-cased
           dbmdz/bert-base-turkish-cased fine-tuned on Turkish sentiment.
  Labels : normalized to positive | negative | neutral via _build_label_map()
           (the raw model emits "notr" for neutral; we standardise here)

Output schema (identical for both models)
  sentiment_label  : str    positive | negative | neutral
  sentiment_score  : float  P(positive) − P(negative)  ∈ [−1, +1]
  prob_positive    : float  [0, 1]
  prob_negative    : float  [0, 1]
  prob_neutral     : float  [0, 1]
  confidence       : float  max(probabilities)

Usage:
    from src.sentiment_model import score_dataframe
    df_scored = score_dataframe(df, text_col="headline", market="US")
    df_scored = score_dataframe(df, text_col="headline", market="BIST")
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

# ── model identifiers ─────────────────────────────────────────────────────────
FINBERT_MODEL_NAME = "ProsusAI/finbert"
TR_MODEL_NAME      = "savasy/bert-base-turkish-sentiment-cased"

# FinBERT label order is fixed by the checkpoint
_FINBERT_LABEL_MAP: dict[int, str] = {0: "positive", 1: "negative", 2: "neutral"}

# ── module-level lazy caches ──────────────────────────────────────────────────
_tokenizer     = None   # FinBERT
_model         = None   # FinBERT

_tr_tokenizer  = None   # Turkish model
_tr_model      = None   # Turkish model
_tr_label_map  = None   # normalised label map built at load time


# ── label normalisation ───────────────────────────────────────────────────────

def _build_label_map(id2label: dict) -> dict[int, str]:
    """
    Convert a HuggingFace id2label dict to the standard three-class schema.

    Rules (applied to lower-cased label string):
      contains "pos"              → "positive"
      contains "neg"              → "negative"
      anything else (notr, neutral, nötr, …) → "neutral"

    Works for 2-class models too: the absent class gets prob = 0 at score time.
    """
    label_map = {}
    for idx, raw in id2label.items():
        ll = raw.lower().strip()
        if "pos" in ll:
            label_map[int(idx)] = "positive"
        elif "neg" in ll:
            label_map[int(idx)] = "negative"
        else:
            label_map[int(idx)] = "neutral"
    return label_map


# ── model loaders (lazy, idempotent) ─────────────────────────────────────────

def load_finbert() -> tuple:
    """Load FinBERT tokenizer + model. Cached after first call."""
    global _tokenizer, _model
    if _tokenizer is None:
        print(f"  Loading {FINBERT_MODEL_NAME} from HuggingFace Hub …")
        _tokenizer = BertTokenizer.from_pretrained(FINBERT_MODEL_NAME)
        _model     = BertForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)
        _model.eval()
        print("  FinBERT ready.")
    return _tokenizer, _model


def load_turkish_model() -> tuple:
    """
    Load savasy/bert-base-turkish-sentiment-cased.
    Builds a normalised label map from the model's own id2label config.
    Cached after first call.
    """
    global _tr_tokenizer, _tr_model, _tr_label_map
    if _tr_tokenizer is None:
        print(f"  Loading {TR_MODEL_NAME} from HuggingFace Hub …")
        _tr_tokenizer = AutoTokenizer.from_pretrained(TR_MODEL_NAME)
        _tr_model     = AutoModelForSequenceClassification.from_pretrained(TR_MODEL_NAME)
        _tr_model.eval()
        _tr_label_map = _build_label_map(_tr_model.config.id2label)
        print(f"  Turkish model ready. Normalised label map: {_tr_label_map}")
    return _tr_tokenizer, _tr_model, _tr_label_map


# ── generic batch scorer ──────────────────────────────────────────────────────

def _score_batch(
    texts: list[str],
    tokenizer,
    model,
    label_map: dict[int, str],
) -> list[dict]:
    """
    Run a forward pass and convert logits to the standard output schema.

    Parameters
    ----------
    texts     : list of raw text strings
    tokenizer : HuggingFace tokenizer compatible with `model`
    model     : HuggingFace sequence-classification model (eval mode)
    label_map : normalised {class_index: 'positive'|'negative'|'neutral'}

    Returns
    -------
    List of dicts, one per input text, with keys:
        sentiment_label, sentiment_score,
        prob_positive, prob_negative, prob_neutral, confidence
    """
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

    # Precompute class indices once per batch
    pos_idx = next((k for k, v in label_map.items() if v == "positive"), None)
    neg_idx = next((k for k, v in label_map.items() if v == "negative"), None)
    neu_idx = next((k for k, v in label_map.items() if v == "neutral"),  None)

    results = []
    for p in probs:
        label    = label_map[int(np.argmax(p))]
        prob_pos = float(p[pos_idx]) if pos_idx is not None else 0.0
        prob_neg = float(p[neg_idx]) if neg_idx is not None else 0.0
        prob_neu = (
            float(p[neu_idx])
            if neu_idx is not None
            else max(0.0, 1.0 - prob_pos - prob_neg)   # 2-class model fallback
        )
        results.append({
            "sentiment_label": label,
            "sentiment_score": round(prob_pos - prob_neg, 4),
            "prob_positive":   round(prob_pos, 4),
            "prob_negative":   round(prob_neg, 4),
            "prob_neutral":    round(prob_neu, 4),
            "confidence":      round(float(np.max(p)), 4),
        })
    return results


# ── public API ────────────────────────────────────────────────────────────────

def score_headline(text: str, market: str = "US") -> dict:
    """Score a single headline. Returns dict with label + probabilities."""
    if market == "BIST":
        tok, mdl, lmap = load_turkish_model()
    else:
        tok, mdl = load_finbert()
        lmap     = _FINBERT_LABEL_MAP
    return _score_batch([text], tok, mdl, lmap)[0]


def score_dataframe(
    df: pd.DataFrame,
    text_col: str = "headline",
    batch_size: int = 16,
    market: str = "US",
) -> pd.DataFrame:
    """
    Score all headlines in a DataFrame.

    Routes to FinBERT for US market, Turkish BERT for BIST market.
    Output schema is identical regardless of the underlying model.

    Parameters
    ----------
    df         : DataFrame containing a text column
    text_col   : name of the text column  (default: 'headline')
    batch_size : texts per forward pass   (lower = less memory)
    market     : 'US' | 'BIST'

    Returns
    -------
    Input DataFrame with sentiment columns appended:
        sentiment_label, sentiment_score,
        prob_positive, prob_negative, prob_neutral, confidence
    """
    if market == "BIST":
        tok, mdl, lmap = load_turkish_model()
    else:
        tok, mdl = load_finbert()
        lmap     = _FINBERT_LABEL_MAP

    texts   = df[text_col].fillna("").tolist()
    records = []

    for i in range(0, len(texts), batch_size):
        records.extend(_score_batch(texts[i : i + batch_size], tok, mdl, lmap))

    scored = df.copy().reset_index(drop=True)
    scored = pd.concat([scored, pd.DataFrame(records)], axis=1)
    return scored


# ── colour / emoji helpers ────────────────────────────────────────────────────
LABEL_COLORS  = {"positive": "#06d6a0", "negative": "#ef233c", "neutral": "#8ecae6"}
LABEL_EMOJIS  = {"positive": "🟢", "negative": "🔴", "neutral": "🔵"}
LABEL_NUMERIC = {"positive": 1, "negative": -1, "neutral": 0}
