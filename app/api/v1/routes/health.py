"""
GET /api/v1/health — Health check endpoint.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import settings

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    version: str
    app: str


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["system"],
)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        app=settings.app_name,
    )
