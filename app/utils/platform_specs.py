from dataclasses import dataclass, field


@dataclass
class PlatformSpec:
    name: str
    headline_char_limit: int
    body_char_limit: int
    headline_count: int          # how many headline slots (e.g. Google has 3)
    body_count: int              # how many body slots
    hashtag_min: int
    hashtag_max: int
    notes: str = ""

    def validate_headline(self, headline: str) -> bool:
        return len(headline) <= self.headline_char_limit

    def validate_body(self, body: str) -> bool:
        return len(body) <= self.body_char_limit

    def validate_hashtag_count(self, count: int) -> bool:
        return self.hashtag_min <= count <= self.hashtag_max


PLATFORM_SPECS: dict[str, PlatformSpec] = {
    "facebook": PlatformSpec(
        name="Facebook Ads",
        headline_char_limit=40,
        body_char_limit=125,
        headline_count=1,
        body_count=1,
        hashtag_min=0,
        hashtag_max=5,
        notes="Primary text shown above creative. Truncated after 125 chars on most placements.",
    ),
    "google": PlatformSpec(
        name="Google Ads",
        headline_char_limit=30,
        body_char_limit=90,
        headline_count=3,
        body_count=2,
        hashtag_min=0,
        hashtag_max=0,
        notes="Responsive Search Ads: up to 15 headlines (30 chars each), 4 descriptions (90 chars).",
    ),
    "instagram": PlatformSpec(
        name="Instagram",
        headline_char_limit=2200,
        body_char_limit=2200,
        headline_count=1,
        body_count=1,
        hashtag_min=0,
        hashtag_max=30,
        notes="Caption limit 2200 chars total. First 125 chars visible before 'more'. Hashtags in caption.",
    ),
    "linkedin": PlatformSpec(
        name="LinkedIn",
        headline_char_limit=150,
        body_char_limit=600,
        headline_count=1,
        body_count=1,
        hashtag_min=3,
        hashtag_max=5,
        notes="Introductory text limit 600 chars. Headline 150 chars. Professional tone performs best.",
    ),
    "twitter": PlatformSpec(
        name="Twitter/X",
        headline_char_limit=280,
        body_char_limit=280,
        headline_count=1,
        body_count=1,
        hashtag_min=2,
        hashtag_max=3,
        notes="Total tweet including hashtags must fit 280 chars. Brevity is key.",
    ),
    "tiktok": PlatformSpec(
        name="TikTok",
        headline_char_limit=100,
        body_char_limit=100,
        headline_count=1,
        body_count=1,
        hashtag_min=5,
        hashtag_max=10,
        notes="Ad caption 100 chars. Hook in first 3 seconds. High-energy, trend-aware copy works best.",
    ),
}


def get_spec(platform: str) -> PlatformSpec:
    """Return PlatformSpec for a platform key. Raises ValueError if unknown."""
    spec = PLATFORM_SPECS.get(platform.lower())
    if spec is None:
        raise ValueError(f"Unknown platform: {platform!r}. Valid: {list(PLATFORM_SPECS)}")
    return spec


def check_compliance(
    platform: str,
    headlines: list[str],
    body: str,
    hashtags: list[str],
) -> dict:
    """Return compliance dict for given platform + content."""
    spec = get_spec(platform)

    headline_within = all(spec.validate_headline(h) for h in headlines)
    body_within = spec.validate_body(body)
    hashtag_valid = spec.validate_hashtag_count(len(hashtags))

    longest_headline = max((len(h) for h in headlines), default=0)

    return {
        "headline_within_limit": headline_within,
        "body_within_limit": body_within,
        "hashtag_count_valid": hashtag_valid,
        "details": {
            "headline_limit": spec.headline_char_limit,
            "longest_headline": longest_headline,
            "body_limit": spec.body_char_limit,
            "body_length": len(body),
            "hashtag_count": len(hashtags),
            "hashtag_min": spec.hashtag_min,
            "hashtag_max": spec.hashtag_max,
        },
    }
