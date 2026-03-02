from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # OpenAI
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model name")
    openai_temperature: float = Field(default=0.7, description="LLM temperature")
    openai_max_tokens: int = Field(default=1024, description="Max tokens per request")

    # HuggingFace
    hf_sentiment_model: str = Field(
        default="cardiffnlp/twitter-roberta-base-sentiment-latest",
        description="Sentiment analysis model",
    )
    hf_safety_model: str = Field(
        default="unitary/toxic-bert",
        description="Toxicity detection model",
    )
    hf_device: int = Field(default=-1, description="-1 for CPU, 0+ for GPU")

    # App
    app_name: str = Field(default="adcraft", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    cors_origins: list[str] = Field(
        default=["*"], description="Allowed CORS origins"
    )


settings = Settings()
