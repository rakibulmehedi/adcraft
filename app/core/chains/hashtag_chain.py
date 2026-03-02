"""
Dedicated LangChain LCEL chain for hashtag generation.

Separate from ad copy chain so hashtags can be regenerated independently
or skipped entirely for platforms that don't use them (Google, Facebook).
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.utils.platform_specs import get_spec

_SYSTEM_PROMPT = """\
You are a social media hashtag strategist. You craft hashtag sets that maximize \
organic reach, are trending, and are highly relevant to the content and target audience.

You MUST respond with valid JSON only — no markdown, no extra text.
"""

_HUMAN_PROMPT = """\
Generate hashtags for a social media post about:

Product: {product}
Description: {description}
Target Audience: {target_audience}
Platform: {platform_name}
Tone: {tone}

Hashtag requirements:
- Count: exactly {hashtag_count} hashtags
- Mix of: broad reach tags, niche/specific tags, branded potential tags
- NO spam hashtags (avoid overused generic ones like #love #follow)
- Relevant to {target_audience}

Respond with this exact JSON:
{{
  "hashtags": ["#Tag1", "#Tag2", ...]
}}

Rules:
1. Each hashtag starts with #
2. CamelCase for multi-word hashtags (e.g. #PersonalFinance)
3. Return exactly {hashtag_count} hashtags
"""


def build_hashtag_chain(llm: ChatOpenAI | None = None):
    """Build and return the LCEL hashtag generation chain."""
    if llm is None:
        llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.5,  # slightly lower temp for consistent hashtag quality
            max_tokens=512,
            api_key=settings.openai_api_key,
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            ("human", _HUMAN_PROMPT),
        ]
    )

    chain = prompt | llm | JsonOutputParser()
    return chain


async def generate_hashtags(
    product: str,
    description: str,
    target_audience: str,
    platform: str,
    tone: str,
    llm: ChatOpenAI | None = None,
) -> list[str]:
    """
    Generate hashtags for the given platform.

    Returns empty list for platforms that don't use hashtags (google, facebook).
    """
    spec = get_spec(platform)

    # Platforms that don't use hashtags
    if spec.hashtag_max == 0:
        return []

    # Use midpoint of allowed range as target count
    target_count = (spec.hashtag_min + spec.hashtag_max) // 2
    target_count = max(target_count, spec.hashtag_min)

    chain = build_hashtag_chain(llm)
    result = await chain.ainvoke(
        {
            "product": product,
            "description": description,
            "target_audience": target_audience,
            "platform_name": spec.name,
            "tone": tone,
            "hashtag_count": target_count,
        }
    )

    hashtags = result.get("hashtags", [])

    # Enforce platform limits
    hashtags = hashtags[: spec.hashtag_max]

    return hashtags
