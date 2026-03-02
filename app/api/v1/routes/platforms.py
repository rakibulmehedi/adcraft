"""
GET /api/v1/platforms — List all supported platforms with their specs.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from app.utils.platform_specs import PLATFORM_SPECS

router = APIRouter()


class PlatformInfo(BaseModel):
    key: str
    name: str
    headline_char_limit: int
    body_char_limit: int
    headline_count: int
    body_count: int
    hashtag_min: int
    hashtag_max: int
    notes: str


class PlatformsResponse(BaseModel):
    platforms: list[PlatformInfo]
    total: int


@router.get(
    "/platforms",
    response_model=PlatformsResponse,
    summary="List all supported ad platforms",
    description="Returns all supported platforms with their character limits and rules.",
    tags=["platforms"],
)
async def list_platforms() -> PlatformsResponse:
    platforms = [
        PlatformInfo(
            key=key,
            name=spec.name,
            headline_char_limit=spec.headline_char_limit,
            body_char_limit=spec.body_char_limit,
            headline_count=spec.headline_count,
            body_count=spec.body_count,
            hashtag_min=spec.hashtag_min,
            hashtag_max=spec.hashtag_max,
            notes=spec.notes,
        )
        for key, spec in PLATFORM_SPECS.items()
    ]
    return PlatformsResponse(platforms=platforms, total=len(platforms))
