"""
FastAPI application factory — registers routers, middleware, and CORS.
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.v1.routes import generate, health, platforms

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

app = FastAPI(
    title="AdCraft API",
    description=(
        "AI-powered ad copy generator for 6 platforms. "
        "Built with LangChain LCEL, OpenAI GPT, and HuggingFace Transformers."
    ),
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
API_PREFIX = "/api/v1"
app.include_router(health.router, prefix=API_PREFIX)
app.include_router(platforms.router, prefix=API_PREFIX)
app.include_router(generate.router, prefix=API_PREFIX)


@app.get("/", include_in_schema=False)
async def root():
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
    }
