"""
Shared pytest fixtures for adcraft tests.
"""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport

# Set dummy env vars BEFORE any app imports so pydantic-settings doesn't fail
os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy-key-for-testing")


@pytest.fixture(scope="session")
def mock_llm_response():
    """Standard mock LLM response for ad copy generation."""
    return {
        "headlines": [
            "Take Control of Your Money",
            "Track Every Penny Easily",
            "Beautiful Charts, Clear Budget",
            "Stop Wondering Where It Went",
            "Your Finance App, Reimagined",
        ],
        "body_copy": "Tired of wondering where your money goes? Track daily expenses effortlessly.",
        "cta": "Download Free Today",
    }


@pytest.fixture(scope="session")
def mock_hashtag_response():
    """Standard mock hashtag response."""
    return {
        "hashtags": [
            "#PersonalFinance",
            "#BudgetApp",
            "#MoneyManagement",
            "#ExpenseTracker",
            "#FinancialFreedom",
        ]
    }


@pytest.fixture
def mock_ad_copy_chain(mock_llm_response):
    """Patch the ad copy chain to return mock data without calling OpenAI."""
    with patch(
        "app.core.chains.ad_copy_chain.generate_ad_copy",
        new=AsyncMock(return_value=mock_llm_response),
    ) as mock:
        yield mock


@pytest.fixture
def mock_hashtag_chain(mock_hashtag_response):
    """Patch the hashtag chain to return mock data."""
    with patch(
        "app.core.chains.hashtag_chain.generate_hashtags",
        new=AsyncMock(return_value=mock_hashtag_response["hashtags"]),
    ) as mock:
        yield mock


@pytest.fixture
def mock_sentiment():
    """Patch HF sentiment scorer."""
    with patch(
        "app.api.v1.routes.generate.score_sentiment",
        return_value={"sentiment": "positive", "sentiment_score": 0.94},
    ) as mock:
        yield mock


@pytest.fixture
def mock_safety():
    """Patch HF safety checker."""
    with patch(
        "app.api.v1.routes.generate.check_safety",
        return_value={"safety": "safe", "safety_score": 0.98},
    ) as mock:
        yield mock


@pytest.fixture
async def async_client(mock_ad_copy_chain, mock_hashtag_chain, mock_sentiment, mock_safety):
    """Async test client with all external services mocked."""
    from app.api.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client
