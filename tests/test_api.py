"""
FastAPI integration tests for adcraft API.

All external services (OpenAI, HuggingFace) are mocked.
"""

import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch


VALID_REQUEST = {
    "product": "Flutter expense tracker app",
    "description": "Track daily expenses with beautiful charts and insights",
    "target_audience": "Young professionals, 22-35",
    "platform": "facebook",
    "tone": "professional",
    "language": "en",
    "num_variations": 5,
}


# ─── Health Check ─────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_ok(self):
        from app.api.main import app

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["app"] == "adcraft"
        assert "version" in data

    @pytest.mark.asyncio
    async def test_root_returns_app_info(self):
        from app.api.main import app

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["app"] == "adcraft"
        assert "docs" in data


# ─── Platforms Endpoint ───────────────────────────────────────────────────────

class TestPlatformsEndpoint:
    @pytest.mark.asyncio
    async def test_platforms_returns_all_six(self):
        from app.api.main import app

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/api/v1/platforms")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 6
        assert len(data["platforms"]) == 6

    @pytest.mark.asyncio
    async def test_platforms_have_required_fields(self):
        from app.api.main import app

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/api/v1/platforms")

        data = response.json()
        required_fields = {
            "key", "name", "headline_char_limit", "body_char_limit",
            "hashtag_min", "hashtag_max", "notes",
        }
        for platform in data["platforms"]:
            assert required_fields.issubset(set(platform.keys()))

    @pytest.mark.asyncio
    async def test_facebook_specs_correct(self):
        from app.api.main import app

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/api/v1/platforms")

        data = response.json()
        fb = next(p for p in data["platforms"] if p["key"] == "facebook")
        assert fb["headline_char_limit"] == 40
        assert fb["body_char_limit"] == 125

    @pytest.mark.asyncio
    async def test_all_platform_keys_present(self):
        from app.api.main import app

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/api/v1/platforms")

        data = response.json()
        keys = {p["key"] for p in data["platforms"]}
        assert keys == {"facebook", "google", "instagram", "linkedin", "twitter", "tiktok"}


# ─── Generate Endpoint ────────────────────────────────────────────────────────

class TestGenerateEndpoint:
    @pytest.mark.asyncio
    async def test_generate_success(self, async_client: AsyncClient):
        response = await async_client.post("/api/v1/generate", json=VALID_REQUEST)
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_generate_response_structure(self, async_client: AsyncClient):
        response = await async_client.post("/api/v1/generate", json=VALID_REQUEST)
        data = response.json()

        assert "headlines" in data
        assert "body_copy" in data
        assert "cta" in data
        assert "hashtags" in data
        assert "quality_scores" in data
        assert "platform_compliance" in data
        assert "platform" in data
        assert "tone" in data
        assert "model_used" in data

    @pytest.mark.asyncio
    async def test_generate_quality_scores_structure(self, async_client: AsyncClient):
        response = await async_client.post("/api/v1/generate", json=VALID_REQUEST)
        scores = response.json()["quality_scores"]

        assert "sentiment" in scores
        assert "sentiment_score" in scores
        assert "safety" in scores
        assert "safety_score" in scores
        assert 0.0 <= scores["sentiment_score"] <= 1.0
        assert 0.0 <= scores["safety_score"] <= 1.0

    @pytest.mark.asyncio
    async def test_generate_platform_compliance_structure(self, async_client: AsyncClient):
        response = await async_client.post("/api/v1/generate", json=VALID_REQUEST)
        compliance = response.json()["platform_compliance"]

        assert "headline_within_limit" in compliance
        assert "body_within_limit" in compliance
        assert "hashtag_count_valid" in compliance
        assert "details" in compliance

    @pytest.mark.asyncio
    async def test_generate_returns_correct_platform(self, async_client: AsyncClient):
        response = await async_client.post("/api/v1/generate", json=VALID_REQUEST)
        data = response.json()
        assert data["platform"] == "facebook"

    @pytest.mark.asyncio
    async def test_generate_headlines_is_list(self, async_client: AsyncClient):
        response = await async_client.post("/api/v1/generate", json=VALID_REQUEST)
        assert isinstance(response.json()["headlines"], list)
        assert len(response.json()["headlines"]) > 0

    @pytest.mark.asyncio
    async def test_generate_all_platforms(self, mock_ad_copy_chain, mock_hashtag_chain, mock_sentiment, mock_safety):
        from app.api.main import app

        platforms = ["facebook", "google", "instagram", "linkedin", "twitter", "tiktok"]

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            for platform in platforms:
                req = {**VALID_REQUEST, "platform": platform}
                response = await client.post("/api/v1/generate", json=req)
                assert response.status_code == 200, f"Failed for platform: {platform}"

    @pytest.mark.asyncio
    async def test_generate_all_tones(self, mock_ad_copy_chain, mock_hashtag_chain, mock_sentiment, mock_safety):
        from app.api.main import app

        tones = ["professional", "casual", "urgent", "inspirational", "humorous", "empathetic"]

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            for tone in tones:
                req = {**VALID_REQUEST, "tone": tone}
                response = await client.post("/api/v1/generate", json=req)
                assert response.status_code == 200, f"Failed for tone: {tone}"


# ─── Validation Tests ─────────────────────────────────────────────────────────

class TestGenerateValidation:
    @pytest.mark.asyncio
    async def test_missing_required_field(self):
        from app.api.main import app

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/generate",
                json={"product": "Test"},  # missing required fields
            )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_invalid_platform(self):
        from app.api.main import app

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/generate",
                json={**VALID_REQUEST, "platform": "myspace"},
            )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_invalid_tone(self):
        from app.api.main import app

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/generate",
                json={**VALID_REQUEST, "tone": "angry_ranting"},
            )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_product_too_short(self):
        from app.api.main import app

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/generate",
                json={**VALID_REQUEST, "product": "AB"},  # min 3 chars
            )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_num_variations_out_of_range(self):
        from app.api.main import app

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/generate",
                json={**VALID_REQUEST, "num_variations": 99},  # max is 10
            )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_default_tone_is_professional(self):
        from app.api.main import app

        req_no_tone = {k: v for k, v in VALID_REQUEST.items() if k != "tone"}

        with (
            patch("app.core.chains.ad_copy_chain.generate_ad_copy", new=AsyncMock(return_value={
                "headlines": ["Test"], "body_copy": "Body", "cta": "Click"
            })),
            patch("app.core.chains.hashtag_chain.generate_hashtags", new=AsyncMock(return_value=[])),
            patch("app.api.v1.routes.generate.score_sentiment", return_value={"sentiment": "positive", "sentiment_score": 0.9}),
            patch("app.api.v1.routes.generate.check_safety", return_value={"safety": "safe", "safety_score": 0.99}),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.post("/api/v1/generate", json=req_no_tone)

        assert response.status_code == 200
        assert response.json()["tone"] == "professional"
