from enum import StrEnum
import sys
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AliasChoices
from typing import Optional, FrozenSet


# Base model names for faster prefix matching

# Models that support reasoning features (e.g., reasoning_effort)
SUPPORT_REASONING_EFFORT_MODELS: FrozenSet[str] = frozenset({
    "o3",
    "o3-2025-04-16",
    "o4-mini",
    "o4-mini-2025-04-16",
    "gpt-5-mini-2025-08-07",
    "gpt-5-mini",
    "gpt-5-2025-08-07",
    "gpt-5",
})

# Models that do not support temperature (e.g., temperature)
NO_SUPPORT_TEMPERATURE_MODELS: FrozenSet[str] = frozenset({
    "o3",
    "o3-2025-04-16",
    "o4-mini",
    "o4-mini-2025-04-16",
    "gpt-5-mini-2025-08-07",
    "gpt-5-mini",
    "gpt-5-2025-08-07",
    "gpt-5",
})

# Models that support developer messages (e.g., developer_message)
SUPPORT_DEVELOPER_MESSAGE_MODELS: FrozenSet[str] = frozenset({
    "o3",
    "o3-2025-04-16",
    "o4-mini",
    "o4-mini-2025-04-16",
    "gpt-5-mini-2025-08-07",
    "gpt-5-mini",
    "gpt-5-2025-08-07",
    "gpt-5",
})

# Models that are in the top tier of the OpenAI API
TOP_TIER_MODELS: FrozenSet[str] = frozenset({
    "o3",
    "o3-2025-04-16",
    "gpt-5-2025-08-07",
    "gpt-5",
})


class MessageRoles(StrEnum):
    Developer = "developer"
    System = "system"
    User = "user"

class ReasoningEfforts(StrEnum):
    High = "high"
    Medium = "medium"
    Low = "low"

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_prefix="", case_sensitive=False)

    # Required
    openai_api_key: str = Field(
        default=...,
        validation_alias=AliasChoices("OPENAI_API_KEY"),
    )
    big_model_name: str = Field(default="gpt-5-2025-08-07", validation_alias=AliasChoices("BIG_MODEL_NAME"))
    small_model_name: str = Field(default="gpt-5-mini-2025-08-07", validation_alias=AliasChoices("SMALL_MODEL_NAME"))

    # Optional with defaults
    base_url: str = Field(default="https://api.openai.com/v1", validation_alias=AliasChoices("OPENAI_BASE_URL"))
    referer_url: str = "http://localhost:8082/claude_proxy"
    app_name: str = "CCProxy"
    app_version: str = "0.1.1"
    log_level: str = Field(default="INFO", validation_alias=AliasChoices("LOG_LEVEL"))
    log_file_path: Optional[str] = "log.jsonl"
    host: str = "127.0.0.1"
    port: int = Field(default=8082, validation_alias=AliasChoices("PORT"))
    reload: bool = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._validate_required_models()

    def _validate_required_models(self):
        """Validate that required model settings are configured."""
        errors = []

        if not (self.openai_api_key and self.openai_api_key.strip()):
            errors.append("API key is required. Set OPENAI_API_KEY in your environment or .env.")

        if not (self.big_model_name and self.big_model_name.strip()):
            errors.append("BIG_MODEL_NAME is required. Set it in your environment or .env.")

        if not (self.small_model_name and self.small_model_name.strip()):
            errors.append("SMALL_MODEL_NAME is required. Set it in your environment or .env.")

        if errors:
            error_message = "\n".join(errors)
            print(f"\nConfiguration Error:\n{error_message}\n")
            sys.exit(1)
