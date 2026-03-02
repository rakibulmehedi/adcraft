from pydantic import BaseModel, Field


class QualityScores(BaseModel):
    sentiment: str = Field(description="Sentiment label: positive, neutral, negative")
    sentiment_score: float = Field(ge=0.0, le=1.0, description="Confidence score")
    safety: str = Field(description="Safety label: safe, unsafe")
    safety_score: float = Field(ge=0.0, le=1.0, description="Safety confidence score")


class PlatformCompliance(BaseModel):
    headline_within_limit: bool
    body_within_limit: bool
    hashtag_count_valid: bool
    details: dict[str, str | int] = Field(default_factory=dict)


class GenerateResponse(BaseModel):
    headlines: list[str] = Field(description="Headline variations")
    body_copy: str = Field(description="Main body copy")
    cta: str = Field(description="Call-to-action text")
    hashtags: list[str] = Field(default_factory=list, description="Hashtags (if applicable)")
    quality_scores: QualityScores
    platform_compliance: PlatformCompliance
    platform: str
    tone: str
    model_used: str = Field(description="LLM model used for generation")
