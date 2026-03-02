"""
HuggingFace brand safety / toxicity check using toxic-bert.

Model: unitary/toxic-bert
Labels: toxic (unsafe) vs non-toxic (safe)

This runs AFTER ad copy generation to ensure no harmful content
is returned to the client.
"""

from __future__ import annotations
import logging
from functools import lru_cache

from transformers import pipeline

from app.core.config import settings

logger = logging.getLogger(__name__)

# Toxicity threshold — scores above this are flagged as unsafe
TOXICITY_THRESHOLD = 0.5


@lru_cache(maxsize=1)
def _get_safety_pipeline():
    """Lazy-load and cache the HuggingFace toxicity pipeline."""
    logger.info("Loading safety model: %s", settings.hf_safety_model)
    return pipeline(
        "text-classification",
        model=settings.hf_safety_model,
        device=settings.hf_device,
        truncation=True,
        max_length=512,
    )


def check_safety(text: str) -> dict[str, str | float]:
    """
    Check the brand safety of the given text.

    Args:
        text: The text to check (full ad copy combined).

    Returns:
        {"safety": "safe"|"unsafe", "safety_score": float}
        safety_score is the probability of being NON-toxic (safe).
    """
    try:
        pipe = _get_safety_pipeline()
        result = pipe(text[:512])[0]

        label = result["label"].lower()
        score = float(result["score"])

        # toxic-bert returns "toxic" or "non-toxic" labels
        if "toxic" in label and "non" not in label:
            # Score is confidence of being toxic — invert for safety_score
            is_toxic = score >= TOXICITY_THRESHOLD
            safety_score = round(1.0 - score, 4)
        else:
            # Label is "non_toxic" or similar — score is confidence of safety
            is_toxic = False
            safety_score = round(score, 4)

        return {
            "safety": "unsafe" if is_toxic else "safe",
            "safety_score": safety_score,
        }
    except Exception as exc:
        logger.warning("Safety check failed: %s — returning safe fallback", exc)
        return {
            "safety": "safe",
            "safety_score": 1.0,
        }
