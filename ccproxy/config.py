from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Required
    openai_api_key: str
    big_model_name: str
    small_model_name: str

    # Optional with defaults
    base_url: str = "https://openrouter.ai/api/v1"
    referer_url: str = "http://localhost:8082/claude_proxy"
    app_name: str = "ClaudeCodeProxy"
    app_version: str = "0.2.0"
    log_level: str = "INFO"
    log_file_path: Optional[str] = "log.jsonl"
    host: str = "127.0.0.1"
    port: int = 8082
    reload: bool = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        missing = []
        if not self.big_model_name or not self.big_model_name.strip():
            missing.append("BIG_MODEL_NAME")
        if not self.small_model_name or not self.small_model_name.strip():
            missing.append("SMALL_MODEL_NAME")
        if missing:
            raise ValueError(f"Missing required settings: {', '.join(missing)}")
