from enum import Enum
from pydantic import BaseModel, Field, field_validator


class Platform(str, Enum):
    FACEBOOK = "facebook"
    GOOGLE = "google"
    INSTAGRAM = "instagram"
    LINKEDIN = "linkedin"
    TWITTER = "twitter"
    TIKTOK = "tiktok"


class Tone(str, Enum):
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    URGENT = "urgent"
    INSPIRATIONAL = "inspirational"
    HUMOROUS = "humorous"
    EMPATHETIC = "empathetic"


class GenerateRequest(BaseModel):
    product: str = Field(
        ...,
        min_length=3,
        max_length=200,
        description="Product or service name",
        examples=["Flutter expense tracker app"],
    )
    description: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Brief product description",
        examples=["Track daily expenses with beautiful charts"],
    )
    target_audience: str = Field(
        ...,
        min_length=5,
        max_length=200,
        description="Target audience description",
        examples=["Young professionals, 22-35"],
    )
    platform: Platform = Field(
        ...,
        description="Ad platform to generate copy for",
    )
    tone: Tone = Field(
        default=Tone.PROFESSIONAL,
        description="Desired tone of the ad copy",
    )
    language: str = Field(
        default="en",
        min_length=2,
        max_length=5,
        description="ISO 639-1 language code",
    )
    num_variations: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of headline variations to generate",
    )

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        return v.lower().strip()
