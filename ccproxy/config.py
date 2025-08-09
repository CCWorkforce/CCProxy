from enum import StrEnum
import sys
import os
from pydantic_settings import BaseSettings
from typing import Optional, FrozenSet


# Models that support reasoning features (e.g., reasoning_effort)
SUPPORT_REASONING_EFFORT_MODELS: FrozenSet[str] = frozenset({
    "gpt-5-mini-2025-08-07",
    "gpt-5-mini",
    "gpt-5-2025-08-07",
    "gpt-5",
})

# Models that do not support temperature (e.g., temperature)
NO_SUPPORT_TEMPERATURE_MODELS: FrozenSet[str] = frozenset({
    "gpt-5-mini-2025-08-07",
    "gpt-5-mini",
    "gpt-5-2025-08-07",
    "gpt-5",
})

# Models that support developer messages (e.g., developer_message)
SUPPORT_DEVELOPER_MESSAGE_MODELS: FrozenSet[str] = frozenset({
    "gpt-5-mini-2025-08-07",
    "gpt-5-mini",
    "gpt-5-2025-08-07",
    "gpt-5",
})

class MessageRoles(StrEnum):
    Developer = "developer"
    System = "system"
    User = "user"

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Required
    openai_api_key: str
    big_model_name: str
    small_model_name: str

    # Optional with defaults
    base_url: str = "https://openrouter.ai/api/v1"
    referer_url: str = "http://localhost:8082/claude_proxy"
    app_name: str = "ClaudeCodeProxy"
    app_version: str = "0.1.0"
    log_level: str = "INFO"
    log_file_path: Optional[str] = "log.jsonl"
    host: str = "127.0.0.1"
    port: int = 8082
    reload: bool = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._validate_required_models()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.big_model_name = os.getenv("BIG_MODEL_NAME")
        self.small_model_name = os.getenv("SMALL_MODEL_NAME")

    def _validate_required_models(self):
        """Validate that required model settings are configured."""
        errors = []

        if not self.openai_api_key.strip():
            errors.append(
                "OPENAI_API_KEY is required but not configured. "
                "Please set the OPENAI_API_KEY environment variable or add it to your .env file."
            )

        if not self.big_model_name.strip():
            errors.append(
                "BIG_MODEL_NAME is required but not configured. "
                "Please set the BIG_MODEL_NAME environment variable or add it to your .env file."
            )

        if not self.small_model_name.strip():
            errors.append(
                "SMALL_MODEL_NAME is required but not configured. "
                "Please set the SMALL_MODEL_NAME environment variable or add it to your .env file."
            )

        if len(errors) > 0:
            error_message = "\n".join([f"❌ {error}" for error in errors])
            print(f"\n[bold red]Configuration Error:[/bold red]\n{error_message}\n")
            sys.exit(1)
