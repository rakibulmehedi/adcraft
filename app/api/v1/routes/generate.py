"""
POST /api/v1/generate — Main ad copy generation endpoint.
"""

import asyncio
import logging

from fastapi import APIRouter, HTTPException, status

from app.core.chains.ad_copy_chain import generate_ad_copy
from app.core.chains.hashtag_chain import generate_hashtags
from app.core.config import settings
from app.core.hf.safety import check_safety
from app.core.hf.sentiment import score_sentiment
from app.core.models.request import GenerateRequest
from app.core.models.response import GenerateResponse, PlatformCompliance, QualityScores
from app.utils.platform_specs import check_compliance

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/generate",
    response_model=GenerateResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate AI-powered ad copy",
    description=(
        "Generate platform-specific ad copy using LangChain + OpenAI GPT. "
        "Output is scored for sentiment and brand safety via HuggingFace models."
    ),
    tags=["generation"],
)
async def generate(request: GenerateRequest) -> GenerateResponse:
    logger.info(
        "Generation request: product=%r platform=%s tone=%s",
        request.product,
        request.platform,
        request.tone,
    )

    # --- Step 1: Generate ad copy + hashtags concurrently ---
    try:
        ad_copy_task = generate_ad_copy(
            product=request.product,
            description=request.description,
            target_audience=request.target_audience,
            platform=request.platform.value,
            tone=request.tone.value,
            language=request.language,
            num_variations=request.num_variations,
        )
        hashtag_task = generate_hashtags(
            product=request.product,
            description=request.description,
            target_audience=request.target_audience,
            platform=request.platform.value,
            tone=request.tone.value,
        )

        ad_copy_result, hashtags = await asyncio.gather(ad_copy_task, hashtag_task)

    except Exception as exc:
        logger.error("LLM generation failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Ad copy generation failed: {exc}",
        ) from exc

    # --- Step 2: Extract fields ---
    headlines: list[str] = ad_copy_result.get("headlines", [])
    body_copy: str = ad_copy_result.get("body_copy", "")
    cta: str = ad_copy_result.get("cta", "")

    if not headlines:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="LLM returned empty headlines. Please retry.",
        )

    # --- Step 3: Quality scoring (HuggingFace) ---
    combined_text = " ".join(headlines) + " " + body_copy
    sentiment_result = score_sentiment(combined_text)
    safety_result = check_safety(combined_text)

    quality_scores = QualityScores(
        sentiment=sentiment_result["sentiment"],
        sentiment_score=sentiment_result["sentiment_score"],
        safety=safety_result["safety"],
        safety_score=safety_result["safety_score"],
    )

    # --- Step 4: Platform compliance check ---
    compliance_data = check_compliance(
        platform=request.platform.value,
        headlines=headlines,
        body=body_copy,
        hashtags=hashtags,
    )

    platform_compliance = PlatformCompliance(
        headline_within_limit=compliance_data["headline_within_limit"],
        body_within_limit=compliance_data["body_within_limit"],
        hashtag_count_valid=compliance_data["hashtag_count_valid"],
        details=compliance_data["details"],
    )

    return GenerateResponse(
        headlines=headlines,
        body_copy=body_copy,
        cta=cta,
        hashtags=hashtags,
        quality_scores=quality_scores,
        platform_compliance=platform_compliance,
        platform=request.platform.value,
        tone=request.tone.value,
        model_used=settings.openai_model,
    )
