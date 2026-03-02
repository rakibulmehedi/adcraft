"""
HuggingFace sentiment scoring using RoBERTa fine-tuned on Twitter data.

Model: cardiffnlp/twitter-roberta-base-sentiment-latest
Labels: positive, neutral, negative
"""

from __future__ import annotations
import logging
from functools import lru_cache

from transformers import pipeline

from app.core.config import settings

logger = logging.getLogger(__name__)

# Map raw model labels to clean labels
_LABEL_MAP = {
    "positive": "positive",
    "neutral": "neutral",
    "negative": "negative",
    # Some model versions use LABEL_0/1/2 or Positive/Negative/Neutral
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
    "Positive": "positive",
    "Neutral": "neutral",
    "Negative": "negative",
}


@lru_cache(maxsize=1)
def _get_sentiment_pipeline():
    """Lazy-load and cache the HuggingFace sentiment pipeline."""
    logger.info("Loading sentiment model: %s", settings.hf_sentiment_model)
    return pipeline(
        "sentiment-analysis",
        model=settings.hf_sentiment_model,
        device=settings.hf_device,
        truncation=True,
        max_length=512,
    )


def score_sentiment(text: str) -> dict[str, str | float]:
    """
    Score the sentiment of the given text.

    Args:
        text: The text to analyze (headlines + body copy combined).

    Returns:
        {"sentiment": "positive"|"neutral"|"negative", "sentiment_score": float}
    """
    try:
        pipe = _get_sentiment_pipeline()
        result = pipe(text[:512])[0]  # truncate to model max length

        raw_label = result["label"]
        score = float(result["score"])

        clean_label = _LABEL_MAP.get(raw_label, raw_label.lower())

        return {
            "sentiment": clean_label,
            "sentiment_score": round(score, 4),
        }
    except Exception as exc:
        logger.warning("Sentiment scoring failed: %s — returning neutral fallback", exc)
        return {
            "sentiment": "neutral",
            "sentiment_score": 0.5,
        }
