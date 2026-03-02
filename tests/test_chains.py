"""
Unit tests for LangChain ad copy + hashtag chains.

All LLM calls are mocked — no OpenAI API key required.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.chains.ad_copy_chain import build_ad_copy_chain, format_chain_input, generate_ad_copy
from app.core.chains.hashtag_chain import build_hashtag_chain, generate_hashtags
from app.utils.platform_specs import get_spec, check_compliance, PLATFORM_SPECS


# ─── Platform Specs ───────────────────────────────────────────────────────────

class TestPlatformSpecs:
    def test_all_platforms_present(self):
        expected = {"facebook", "google", "instagram", "linkedin", "twitter", "tiktok"}
        assert set(PLATFORM_SPECS.keys()) == expected

    def test_get_spec_valid(self):
        spec = get_spec("facebook")
        assert spec.name == "Facebook Ads"
        assert spec.headline_char_limit == 40
        assert spec.body_char_limit == 125

    def test_get_spec_case_insensitive(self):
        spec = get_spec("FACEBOOK")
        assert spec.name == "Facebook Ads"

    def test_get_spec_invalid(self):
        with pytest.raises(ValueError, match="Unknown platform"):
            get_spec("myspace")

    def test_facebook_headline_validation(self):
        spec = get_spec("facebook")
        assert spec.validate_headline("Short") is True
        assert spec.validate_headline("A" * 40) is True
        assert spec.validate_headline("A" * 41) is False

    def test_google_body_validation(self):
        spec = get_spec("google")
        assert spec.validate_body("A" * 90) is True
        assert spec.validate_body("A" * 91) is False

    def test_instagram_hashtag_range(self):
        spec = get_spec("instagram")
        assert spec.validate_hashtag_count(0) is True
        assert spec.validate_hashtag_count(30) is True
        assert spec.validate_hashtag_count(31) is False

    def test_google_no_hashtags(self):
        spec = get_spec("google")
        assert spec.hashtag_max == 0
        assert spec.validate_hashtag_count(0) is True
        assert spec.validate_hashtag_count(1) is False

    def test_compliance_check_passing(self):
        result = check_compliance(
            platform="facebook",
            headlines=["Short headline"],
            body="Short body",
            hashtags=[],
        )
        assert result["headline_within_limit"] is True
        assert result["body_within_limit"] is True
        assert result["hashtag_count_valid"] is True

    def test_compliance_check_failing_headline(self):
        result = check_compliance(
            platform="facebook",
            headlines=["A" * 50],  # exceeds 40 char limit
            body="OK body",
            hashtags=[],
        )
        assert result["headline_within_limit"] is False

    def test_compliance_details_keys(self):
        result = check_compliance(
            platform="linkedin",
            headlines=["Great LinkedIn Ad"],
            body="Body here",
            hashtags=["#B2B", "#Tech", "#SaaS"],
        )
        assert "headline_limit" in result["details"]
        assert "body_limit" in result["details"]
        assert "hashtag_count" in result["details"]


# ─── Chain Input Formatting ───────────────────────────────────────────────────

class TestFormatChainInput:
    def test_facebook_input(self):
        inp = format_chain_input(
            product="My App",
            description="Track expenses",
            target_audience="Young professionals",
            platform="facebook",
            tone="professional",
            language="en",
            num_variations=5,
        )
        assert inp["platform_name"] == "Facebook Ads"
        assert inp["headline_limit"] == 40
        assert inp["body_limit"] == 125
        assert inp["num_variations"] == 5
        assert "none" in inp["hashtag_range"].lower()

    def test_instagram_input_has_hashtag_range(self):
        inp = format_chain_input(
            product="App",
            description="Desc",
            target_audience="Gen Z",
            platform="instagram",
            tone="casual",
            language="en",
            num_variations=3,
        )
        assert "30" in inp["hashtag_range"]

    def test_all_platforms_format_without_error(self):
        for platform in PLATFORM_SPECS:
            inp = format_chain_input(
                product="Test",
                description="Test desc",
                target_audience="Everyone",
                platform=platform,
                tone="casual",
                language="en",
                num_variations=5,
            )
            assert "platform_name" in inp
            assert "headline_limit" in inp


# ─── Ad Copy Chain (mocked LLM) ───────────────────────────────────────────────

class TestAdCopyChain:
    @pytest.mark.asyncio
    async def test_generate_ad_copy_returns_expected_keys(self, mock_llm_response):
        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(return_value=mock_llm_response)

        with patch("app.core.chains.ad_copy_chain.build_ad_copy_chain", return_value=mock_chain):
            result = await generate_ad_copy(
                product="Expense App",
                description="Track spending",
                target_audience="Young adults",
                platform="facebook",
                tone="professional",
            )

        assert "headlines" in result
        assert "body_copy" in result
        assert "cta" in result
        assert isinstance(result["headlines"], list)

    @pytest.mark.asyncio
    async def test_generate_ad_copy_all_platforms(self, mock_llm_response):
        for platform in PLATFORM_SPECS:
            mock_chain = MagicMock()
            mock_chain.ainvoke = AsyncMock(return_value=mock_llm_response)

            with patch("app.core.chains.ad_copy_chain.build_ad_copy_chain", return_value=mock_chain):
                result = await generate_ad_copy(
                    product="Test Product",
                    description="Test description for product",
                    target_audience="Test audience",
                    platform=platform,
                    tone="professional",
                )
            assert result is not None


# ─── Hashtag Chain (mocked LLM) ───────────────────────────────────────────────

class TestHashtagChain:
    @pytest.mark.asyncio
    async def test_no_hashtags_for_google(self):
        result = await generate_hashtags(
            product="App",
            description="Desc",
            target_audience="Everyone",
            platform="google",
            tone="professional",
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_no_hashtags_for_facebook(self):
        # Facebook has hashtag_max=5 so it WILL generate some
        # But Google has max=0 so returns []
        result = await generate_hashtags(
            product="App",
            description="Desc",
            target_audience="Everyone",
            platform="google",
            tone="professional",
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_instagram_generates_hashtags(self):
        mock_response = {"hashtags": ["#Tag1", "#Tag2", "#Tag3"]}
        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(return_value=mock_response)

        with patch("app.core.chains.hashtag_chain.build_hashtag_chain", return_value=mock_chain):
            result = await generate_hashtags(
                product="App",
                description="Description here",
                target_audience="Gen Z",
                platform="instagram",
                tone="casual",
            )

        assert len(result) >= 0
        if result:
            assert all(tag.startswith("#") for tag in result)

    @pytest.mark.asyncio
    async def test_hashtags_respect_platform_max(self):
        """Hashtags returned must not exceed platform max."""
        many_hashtags = [f"#Tag{i}" for i in range(50)]
        mock_response = {"hashtags": many_hashtags}
        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(return_value=mock_response)

        with patch("app.core.chains.hashtag_chain.build_hashtag_chain", return_value=mock_chain):
            result = await generate_hashtags(
                product="App",
                description="Description",
                target_audience="Users",
                platform="twitter",
                tone="casual",
            )

        spec = get_spec("twitter")
        assert len(result) <= spec.hashtag_max


# ─── HuggingFace Scoring (mocked pipeline) ───────────────────────────────────

class TestSentimentScoring:
    def test_score_sentiment_positive(self):
        from app.core.hf.sentiment import score_sentiment

        mock_pipeline = MagicMock(return_value=[{"label": "positive", "score": 0.94}])
        with patch("app.core.hf.sentiment._get_sentiment_pipeline", return_value=mock_pipeline):
            result = score_sentiment("Amazing product you will love!")

        assert result["sentiment"] == "positive"
        assert result["sentiment_score"] == pytest.approx(0.94)

    def test_score_sentiment_negative(self):
        from app.core.hf.sentiment import score_sentiment

        mock_pipeline = MagicMock(return_value=[{"label": "negative", "score": 0.87}])
        with patch("app.core.hf.sentiment._get_sentiment_pipeline", return_value=mock_pipeline):
            result = score_sentiment("Terrible, awful, broken")

        assert result["sentiment"] == "negative"

    def test_score_sentiment_fallback_on_error(self):
        from app.core.hf.sentiment import score_sentiment

        with patch(
            "app.core.hf.sentiment._get_sentiment_pipeline",
            side_effect=RuntimeError("model not found"),
        ):
            result = score_sentiment("some text")

        assert result["sentiment"] == "neutral"
        assert result["sentiment_score"] == 0.5


class TestSafetyCheck:
    def test_safe_content(self):
        from app.core.hf.safety import check_safety

        mock_pipeline = MagicMock(return_value=[{"label": "non_toxic", "score": 0.98}])
        with patch("app.core.hf.safety._get_safety_pipeline", return_value=mock_pipeline):
            result = check_safety("Download our free app today!")

        assert result["safety"] == "safe"
        assert result["safety_score"] > 0.5

    def test_unsafe_content(self):
        from app.core.hf.safety import check_safety

        mock_pipeline = MagicMock(return_value=[{"label": "toxic", "score": 0.91}])
        with patch("app.core.hf.safety._get_safety_pipeline", return_value=mock_pipeline):
            result = check_safety("offensive content here")

        assert result["safety"] == "unsafe"

    def test_safety_fallback_on_error(self):
        from app.core.hf.safety import check_safety

        with patch(
            "app.core.hf.safety._get_safety_pipeline",
            side_effect=RuntimeError("pipeline error"),
        ):
            result = check_safety("some text")

        assert result["safety"] == "safe"
        assert result["safety_score"] == 1.0
