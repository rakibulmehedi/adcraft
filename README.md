# AdCraft — AI Ad Copy Generator

> Generate high-converting ad copy for 6 platforms in seconds using LangChain + OpenAI GPT + HuggingFace.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-orange)](https://python.langchain.com)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)](https://huggingface.co)
[![Docker](https://img.shields.io/badge/Docker-ready-blue?logo=docker)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## What It Does

AdCraft is a production-ready SaaS API that generates platform-aware ad copy using a 3-stage AI pipeline:

```
1. LangChain LCEL Chain  →  GPT generates headlines + body + CTA
2. HuggingFace Sentiment →  Ensures positive, engaging tone
3. HuggingFace Safety    →  Brand safety toxicity check
```

**Supported Platforms:**

| Platform | Headline Limit | Body Limit | Hashtags |
|---|---|---|---|
| Facebook Ads | 40 chars | 125 chars | optional (0–5) |
| Google Ads | 30 chars × 3 | 90 chars × 2 | none |
| Instagram | 2200 chars | 2200 chars | up to 30 |
| LinkedIn | 150 chars | 600 chars | 3–5 |
| Twitter/X | 280 chars | 280 chars | 2–3 |
| TikTok | 100 chars | 100 chars | 5–10 |

---

## Architecture

```
adcraft/
├── app/
│   ├── api/
│   │   ├── v1/
│   │   │   └── routes/
│   │   │       ├── generate.py     # POST /generate — main endpoint
│   │   │       ├── platforms.py    # GET  /platforms — platform specs
│   │   │       └── health.py       # GET  /health
│   │   └── main.py                 # FastAPI app, CORS, router registration
│   ├── core/
│   │   ├── chains/
│   │   │   ├── ad_copy_chain.py    # LCEL: PromptTemplate → ChatOpenAI → JsonOutputParser
│   │   │   └── hashtag_chain.py    # Dedicated hashtag generation chain
│   │   ├── hf/
│   │   │   ├── sentiment.py        # RoBERTa sentiment scoring
│   │   │   └── safety.py           # toxic-bert brand safety check
│   │   ├── models/
│   │   │   ├── request.py          # Pydantic input model
│   │   │   └── response.py         # Pydantic output model
│   │   └── config.py               # pydantic-settings environment config
│   └── utils/
│       └── platform_specs.py       # Per-platform limits + compliance checker
├── tests/
│   ├── conftest.py                 # Shared fixtures (all LLM calls mocked)
│   ├── test_chains.py              # Unit tests: chains, specs, HF scoring
│   └── test_api.py                 # Integration tests: all endpoints
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))

### 1. Clone & Install

```bash
git clone https://github.com/MehedisGits/adcraft.git
cd adcraft
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY
```

### 3. Run

```bash
uvicorn app.api.main:app --reload
```

API is now running at `http://localhost:8000`
- Swagger docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## API Reference

### `POST /api/v1/generate`

Generate ad copy for a platform.

**Request:**
```json
{
  "product": "Flutter expense tracker app",
  "description": "Track daily expenses with beautiful charts",
  "target_audience": "Young professionals, 22-35",
  "platform": "facebook",
  "tone": "professional",
  "language": "en",
  "num_variations": 5
}
```

**Response:**
```json
{
  "headlines": [
    "Take Control of Your Money",
    "Track Every Penny Easily",
    "Beautiful Charts, Clear Budget",
    "Stop Wondering Where It Went",
    "Your Finance App, Reimagined"
  ],
  "body_copy": "Tired of wondering where your money goes? Track daily expenses effortlessly with beautiful charts and instant insights.",
  "cta": "Download Free Today",
  "hashtags": [],
  "quality_scores": {
    "sentiment": "positive",
    "sentiment_score": 0.94,
    "safety": "safe",
    "safety_score": 0.98
  },
  "platform_compliance": {
    "headline_within_limit": true,
    "body_within_limit": true,
    "hashtag_count_valid": true,
    "details": {
      "headline_limit": 40,
      "longest_headline": 29,
      "body_limit": 125,
      "body_length": 115,
      "hashtag_count": 0,
      "hashtag_min": 0,
      "hashtag_max": 5
    }
  },
  "platform": "facebook",
  "tone": "professional",
  "model_used": "gpt-4o-mini"
}
```

**Platform options:** `facebook` · `google` · `instagram` · `linkedin` · `twitter` · `tiktok`

**Tone options:** `professional` · `casual` · `urgent` · `inspirational` · `humorous` · `empathetic`

---

### `GET /api/v1/platforms`

Returns all supported platforms with their specs and limits.

### `GET /api/v1/health`

Health check — returns `{"status": "ok"}`.

---

## Running with Docker

```bash
# Build and start (API + Redis)
docker-compose up --build

# API available at http://localhost:8000
```

---

## Running Tests

```bash
# All tests — no API keys required (all LLM calls mocked)
pytest

# With coverage report
pytest --cov=app --cov-report=term-missing
```

Test coverage includes:
- Platform spec validation (all 6 platforms)
- LangChain chain input formatting
- Ad copy + hashtag generation (mocked LLM)
- HuggingFace sentiment + safety scoring (mocked pipeline)
- FastAPI endpoint integration tests (all routes, all platforms, all tones)
- Request validation (422s for invalid input)

---

## LangChain LCEL Pipeline

The core generation uses LangChain Expression Language (LCEL) — the modern composable pattern:

```python
chain = prompt | llm | JsonOutputParser()
result = await chain.ainvoke(inputs)
```

Ad copy and hashtags are generated **concurrently** using `asyncio.gather()`:

```python
ad_copy, hashtags = await asyncio.gather(
    generate_ad_copy(...),
    generate_hashtags(...),
)
```

---

## HuggingFace Models

| Model | Purpose | Pipeline |
|---|---|---|
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | Sentiment scoring | `sentiment-analysis` |
| `unitary/toxic-bert` | Brand safety / toxicity check | `text-classification` |

Models are lazy-loaded and cached with `@lru_cache` — downloaded on first request, zero overhead after.

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | — | Your OpenAI API key |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | GPT model to use |
| `OPENAI_TEMPERATURE` | No | `0.7` | LLM creativity (0–1) |
| `HF_SENTIMENT_MODEL` | No | `cardiffnlp/twitter-roberta-base-sentiment-latest` | Sentiment model |
| `HF_SAFETY_MODEL` | No | `unitary/toxic-bert` | Safety model |
| `HF_DEVICE` | No | `-1` (CPU) | `-1` for CPU, `0+` for GPU |
| `DEBUG` | No | `false` | Enable debug logging |

---

## Built By

**Rakibul Islam Mehedi** — Flutter developer, Python engineer, and retired digital marketer.

> "I don't just build products — I understand how to grow them."

Connect: [GitHub](https://github.com/MehedisGits) · [LinkedIn](https://linkedin.com/in/mehedisgits)

---

## License

MIT © 2026 Rakibul Islam Mehedi
