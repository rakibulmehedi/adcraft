"""
LangChain LCEL pipeline for ad copy generation.

Pipeline:
    PromptTemplate → ChatOpenAI → JsonOutputParser
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.utils.platform_specs import get_spec

_SYSTEM_PROMPT = """\
You are AdCraft — an expert digital marketing copywriter with 15 years of experience \
writing high-converting ads for Facebook, Google, Instagram, LinkedIn, Twitter/X, and TikTok.

You understand direct-response copywriting, AIDA frameworks, and platform-specific best practices.
You always write copy that respects character limits strictly.

You MUST respond with valid JSON only — no markdown, no code fences, no extra text.
"""

_HUMAN_PROMPT = """\
Generate ad copy for the following brief:

Product: {product}
Description: {description}
Target Audience: {target_audience}
Platform: {platform_name}
Tone: {tone}
Language: {language}

Platform Rules:
- Headline character limit: {headline_limit} characters (HARD LIMIT — never exceed)
- Body copy character limit: {body_limit} characters (HARD LIMIT — never exceed)
- Hashtags: {hashtag_range}
- Platform notes: {platform_notes}

Generate {num_variations} headline variations.

Respond with this exact JSON structure:
{{
  "headlines": ["headline 1", "headline 2", ...],
  "body_copy": "main body copy text here",
  "cta": "call to action text"
}}

Rules:
1. Every headline MUST be under {headline_limit} characters
2. body_copy MUST be under {body_limit} characters
3. cta should be 2-5 words, punchy and action-oriented
4. Match the {tone} tone throughout
5. Write for {target_audience} specifically
6. Respond in {language} language
"""


def build_ad_copy_chain(llm: ChatOpenAI | None = None):
    """Build and return the LCEL ad copy generation chain."""
    if llm is None:
        llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.openai_temperature,
            max_tokens=settings.openai_max_tokens,
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


def format_chain_input(
    product: str,
    description: str,
    target_audience: str,
    platform: str,
    tone: str,
    language: str,
    num_variations: int,
) -> dict:
    """Build the input dict for the ad copy chain."""
    spec = get_spec(platform)

    if spec.hashtag_max == 0:
        hashtag_range = "none (hashtags not used on this platform)"
    else:
        hashtag_range = f"{spec.hashtag_min}–{spec.hashtag_max} hashtags"

    return {
        "product": product,
        "description": description,
        "target_audience": target_audience,
        "platform_name": spec.name,
        "tone": tone,
        "language": language,
        "headline_limit": spec.headline_char_limit,
        "body_limit": spec.body_char_limit,
        "hashtag_range": hashtag_range,
        "platform_notes": spec.notes,
        "num_variations": num_variations,
    }


async def generate_ad_copy(
    product: str,
    description: str,
    target_audience: str,
    platform: str,
    tone: str,
    language: str = "en",
    num_variations: int = 5,
    llm: ChatOpenAI | None = None,
) -> dict:
    """
    Run the ad copy chain and return parsed JSON dict.

    Returns:
        {"headlines": [...], "body_copy": "...", "cta": "..."}
    """
    chain = build_ad_copy_chain(llm)
    inputs = format_chain_input(
        product=product,
        description=description,
        target_audience=target_audience,
        platform=platform,
        tone=tone,
        language=language,
        num_variations=num_variations,
    )
    result = await chain.ainvoke(inputs)
    return result
