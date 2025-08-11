from enum import StrEnum
import sys
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AliasChoices, field_validator
from typing import Optional, FrozenSet, List, Union

from urllib.parse import urlparse

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

MODEL_INPUT_TOKEN_LIMIT: FrozenSet[tuple[str, int]] = frozenset({
    ("o3", 200_000),
    ("o3-2025-04-16", 200_000),
    ("o4-mini", 200_000),
    ("o4-mini-2025-04-16", 200_000),
    ("gpt-5-2025-08-07", 272_000),
    ("gpt-5", 272_000),
    ("gpt-5-mini-2025-08-07", 272_000),
    ("gpt-5-mini", 272_000),
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
    app_version: str = "0.2.0"
    log_level: str = Field(default="INFO", validation_alias=AliasChoices("LOG_LEVEL"))
    log_file_path: Optional[str] = "log.jsonl"
    host: str = "127.0.0.1"
    port: int = Field(default=8082, validation_alias=AliasChoices("PORT"))
    reload: bool = True

    rate_limit_enabled: bool = Field(default=True, validation_alias=AliasChoices("RATE_LIMIT_ENABLED"))
    rate_limit_per_minute: int = Field(default=60, validation_alias=AliasChoices("RATE_LIMIT_PER_MINUTE"))
    rate_limit_burst: int = Field(default=30, validation_alias=AliasChoices("RATE_LIMIT_BURST"))

    security_headers_enabled: bool = Field(default=True, validation_alias=AliasChoices("SECURITY_HEADERS_ENABLED"))
    enable_hsts: bool = Field(default=False, validation_alias=AliasChoices("ENABLE_HSTS"))

    enable_cors: bool = Field(default=False, validation_alias=AliasChoices("ENABLE_CORS"))
    cors_allow_origins: Union[List[str], str] = Field(default_factory=list, validation_alias=AliasChoices("CORS_ALLOW_ORIGINS"))
    cors_allow_methods: Union[List[str], str] = Field(default_factory=lambda: ["POST", "OPTIONS"], validation_alias=AliasChoices("CORS_ALLOW_METHODS"))
    cors_allow_headers: Union[List[str], str] = Field(default_factory=lambda: ["Authorization", "Content-Type", "X-Requested-With"], validation_alias=AliasChoices("CORS_ALLOW_HEADERS"))

    allowed_hosts: Union[List[str], str] = Field(default_factory=list, validation_alias=AliasChoices("ALLOWED_HOSTS"))

    restrict_base_url: bool = Field(default=True, validation_alias=AliasChoices("RESTRICT_BASE_URL"))
    allowed_base_url_hosts: Union[List[str], str] = Field(default_factory=lambda: ["api.openai.com"], validation_alias=AliasChoices("ALLOWED_BASE_URL_HOSTS"))

    redact_log_fields: Union[List[str], str] = Field(default_factory=lambda: ["openai_api_key", "authorization"], validation_alias=AliasChoices("REDACT_LOG_FIELDS"))

    max_stream_seconds: int = Field(default=600, validation_alias=AliasChoices("MAX_STREAM_SECONDS"))

    @field_validator('cors_allow_origins', 'cors_allow_methods', 'cors_allow_headers', 'allowed_hosts', 'allowed_base_url_hosts', 'redact_log_fields')
    @classmethod
    def parse_comma_separated(cls, v):
        if isinstance(v, str):
            if v.strip() == "":
                return []
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._validate_required_models()
        self._validate_security()

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

    def _validate_security(self):
        errors = []
        if self.restrict_base_url:
            try:
                parsed = urlparse(self.base_url)
                if parsed.scheme.lower() != "https":
                    errors.append("OPENAI_BASE_URL must use https when RESTRICT_BASE_URL is enabled.")
                host = parsed.hostname or ""
                if host not in set(self.allowed_base_url_hosts or []):
                    errors.append("OPENAI_BASE_URL host is not in ALLOWED_BASE_URL_HOSTS.")
            except Exception:
                errors.append("OPENAI_BASE_URL is invalid.")
        if errors:
            error_message = "\n".join(errors)
            print(f"\nSecurity Configuration Error:\n{error_message}\n")
            sys.exit(1)
