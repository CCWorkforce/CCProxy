import sys
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AliasChoices, field_validator
from typing import Optional, List, Union, Any

from urllib.parse import urlparse

from ccproxy.enums import TruncationConfig



class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", env_prefix="", case_sensitive=False
    )

    # Required
    openai_api_key: str = Field(
        default=...,
        validation_alias=AliasChoices("OPENAI_API_KEY"),
    )
    big_model_name: str = Field(
        default="gpt-5-2025-08-07", validation_alias=AliasChoices("BIG_MODEL_NAME")
    )
    small_model_name: str = Field(
        default="gpt-5-mini-2025-08-07",
        validation_alias=AliasChoices("SMALL_MODEL_NAME"),
    )

    # Optional with defaults
    base_url: str = Field(
        default="https://api.openai.com/v1",
        validation_alias=AliasChoices("OPENAI_BASE_URL"),
    )
    referer_url: str = "http://localhost:11434/claude_proxy"
    app_name: str = "CCProxy"
    app_version: str = "1.1.0"
    log_level: str = Field(default="INFO", validation_alias=AliasChoices("LOG_LEVEL"))
    log_file_path: Optional[str] = "log.jsonl"
    error_log_file_path: Optional[str] = Field(
        default="error.jsonl", validation_alias=AliasChoices("ERROR_LOG_FILE_PATH")
    )
    log_pretty_console: bool = Field(
        default=False, validation_alias=AliasChoices("LOG_PRETTY_CONSOLE")
    )
    host: str = "127.0.0.1"
    port: int = Field(default=11434, validation_alias=AliasChoices("PORT"))
    reload: bool = True

    rate_limit_enabled: bool = Field(
        default=True, validation_alias=AliasChoices("RATE_LIMIT_ENABLED")
    )
    rate_limit_per_minute: int = Field(
        default=60, validation_alias=AliasChoices("RATE_LIMIT_PER_MINUTE")
    )
    rate_limit_burst: int = Field(
        default=30, validation_alias=AliasChoices("RATE_LIMIT_BURST")
    )

    security_headers_enabled: bool = Field(
        default=True, validation_alias=AliasChoices("SECURITY_HEADERS_ENABLED")
    )
    enable_hsts: bool = Field(
        default=False, validation_alias=AliasChoices("ENABLE_HSTS")
    )

    enable_cors: bool = Field(
        default=False, validation_alias=AliasChoices("ENABLE_CORS")
    )
    cors_allow_origins: Union[List[str], str] = Field(
        default_factory=list, validation_alias=AliasChoices("CORS_ALLOW_ORIGINS")
    )
    cors_allow_methods: Union[List[str], str] = Field(
        default_factory=lambda: ["POST", "OPTIONS"],
        validation_alias=AliasChoices("CORS_ALLOW_METHODS"),
    )
    cors_allow_headers: Union[List[str], str] = Field(
        default_factory=lambda: ["Authorization", "Content-Type", "X-Requested-With"],
        validation_alias=AliasChoices("CORS_ALLOW_HEADERS"),
    )

    allowed_hosts: Union[List[str], str] = Field(
        default_factory=list, validation_alias=AliasChoices("ALLOWED_HOSTS")
    )

    restrict_base_url: bool = Field(
        default=True, validation_alias=AliasChoices("RESTRICT_BASE_URL")
    )
    allowed_base_url_hosts: Union[List[str], str] = Field(
        default_factory=lambda: ["api.openai.com"],
        validation_alias=AliasChoices("ALLOWED_BASE_URL_HOSTS"),
    )

    redact_log_fields: Union[List[str], str] = Field(
        default_factory=lambda: ["openai_api_key", "authorization"],
        validation_alias=AliasChoices("REDACT_LOG_FIELDS"),
    )

    # Error tracking configuration
    error_tracking_enabled: bool = Field(
        default=True, validation_alias=AliasChoices("ERROR_TRACKING_ENABLED")
    )
    error_tracking_file: str = Field(
        default="errors_detailed.jsonl",
        validation_alias=AliasChoices("ERROR_TRACKING_FILE"),
    )
    error_tracking_max_size_mb: int = Field(
        default=100, validation_alias=AliasChoices("ERROR_TRACKING_MAX_SIZE_MB")
    )
    error_tracking_retention_days: int = Field(
        default=30, validation_alias=AliasChoices("ERROR_TRACKING_RETENTION_DAYS")
    )
    error_tracking_capture_request: bool = Field(
        default=True, validation_alias=AliasChoices("ERROR_TRACKING_CAPTURE_REQUEST")
    )
    error_tracking_capture_response: bool = Field(
        default=True, validation_alias=AliasChoices("ERROR_TRACKING_CAPTURE_RESPONSE")
    )
    error_tracking_max_body_size: int = Field(
        default=10000, validation_alias=AliasChoices("ERROR_TRACKING_MAX_BODY_SIZE")
    )

    max_stream_seconds: int = Field(
        default=600, validation_alias=AliasChoices("MAX_STREAM_SECONDS")
    )

    cache_token_counts_enabled: bool = Field(
        default=True, validation_alias=AliasChoices("CACHE_TOKEN_COUNTS_ENABLED")
    )
    cache_token_counts_ttl_s: int = Field(
        default=300, validation_alias=AliasChoices("CACHE_TOKEN_COUNTS_TTL_S")
    )
    cache_token_counts_max: int = Field(
        default=2048, validation_alias=AliasChoices("CACHE_TOKEN_COUNTS_MAX")
    )

    cache_converters_enabled: bool = Field(
        default=True, validation_alias=AliasChoices("CACHE_CONVERTERS_ENABLED")
    )
    cache_converters_max: int = Field(
        default=256, validation_alias=AliasChoices("CACHE_CONVERTERS_MAX")
    )

    stream_dedupe_enabled: bool = Field(
        default=True, validation_alias=AliasChoices("STREAM_DEDUPE_ENABLED")
    )
    truncate_long_requests: bool = Field(
        default=True, validation_alias=AliasChoices("TRUNCATE_LONG_REQUESTS")
    )
    truncation_config: TruncationConfig = Field(
        default_factory=TruncationConfig,
        validation_alias=AliasChoices("TRUNCATION_CONFIG"),
    )
    metrics_cache_enabled: bool = Field(
        default=True, validation_alias=AliasChoices("METRICS_CACHE_ENABLED")
    )

    # Cache warmup configuration
    cache_warmup_enabled: bool = Field(
        default=False, validation_alias=AliasChoices("CACHE_WARMUP_ENABLED")
    )
    cache_warmup_file_path: Optional[str] = Field(
        default="cache_warmup.json", validation_alias=AliasChoices("CACHE_WARMUP_FILE_PATH")
    )
    cache_warmup_max_items: int = Field(
        default=100, validation_alias=AliasChoices("CACHE_WARMUP_MAX_ITEMS")
    )
    cache_warmup_on_startup: bool = Field(
        default=True, validation_alias=AliasChoices("CACHE_WARMUP_ON_STARTUP")
    )
    cache_warmup_preload_common: bool = Field(
        default=True, validation_alias=AliasChoices("CACHE_WARMUP_PRELOAD_COMMON")
    )
    cache_warmup_auto_save_popular: bool = Field(
        default=True, validation_alias=AliasChoices("CACHE_WARMUP_AUTO_SAVE_POPULAR")
    )
    cache_warmup_popularity_threshold: int = Field(
        default=3, validation_alias=AliasChoices("CACHE_WARMUP_POPULARITY_THRESHOLD")
    )
    cache_warmup_save_interval_seconds: int = Field(
        default=3600, validation_alias=AliasChoices("CACHE_WARMUP_SAVE_INTERVAL_SECONDS")
    )

    provider_max_retries: int = Field(
        default=3, validation_alias=AliasChoices("PROVIDER_MAX_RETRIES")
    )
    provider_retry_base_delay: float = Field(
        default=1.0, validation_alias=AliasChoices("PROVIDER_RETRY_BASE_DELAY")
    )
    provider_retry_jitter: float = Field(
        default=0.5, validation_alias=AliasChoices("PROVIDER_RETRY_JITTER")
    )

    # Circuit breaker configuration
    circuit_breaker_failure_threshold: int = Field(
        default=5, validation_alias=AliasChoices("CIRCUIT_BREAKER_FAILURE_THRESHOLD")
    )
    circuit_breaker_recovery_timeout: int = Field(
        default=60, validation_alias=AliasChoices("CIRCUIT_BREAKER_RECOVERY_TIMEOUT")
    )
    circuit_breaker_half_open_requests: int = Field(
        default=3, validation_alias=AliasChoices("CIRCUIT_BREAKER_HALF_OPEN_REQUESTS")
    )

    # Connection pool configuration
    pool_max_keepalive_connections: int = Field(
        default=50, validation_alias=AliasChoices("POOL_MAX_KEEPALIVE_CONNECTIONS")
    )
    pool_max_connections: int = Field(
        default=500, validation_alias=AliasChoices("POOL_MAX_CONNECTIONS")
    )
    pool_keepalive_expiry: int = Field(
        default=120, validation_alias=AliasChoices("POOL_KEEPALIVE_EXPIRY")
    )

    # HTTP timeout configuration
    http_connect_timeout: float = Field(
        default=10.0, validation_alias=AliasChoices("HTTP_CONNECT_TIMEOUT")
    )
    http_write_timeout: float = Field(
        default=30.0, validation_alias=AliasChoices("HTTP_WRITE_TIMEOUT")
    )
    http_pool_timeout: float = Field(
        default=10.0, validation_alias=AliasChoices("HTTP_POOL_TIMEOUT")
    )

    # Distributed tracing configuration
    tracing_enabled: bool = Field(
        default=False, validation_alias=AliasChoices("TRACING_ENABLED")
    )
    tracing_exporter: str = Field(
        default="console", validation_alias=AliasChoices("TRACING_EXPORTER")
    )
    tracing_endpoint: str = Field(
        default="", validation_alias=AliasChoices("TRACING_ENDPOINT")
    )
    tracing_service_name: str = Field(
        default="ccproxy", validation_alias=AliasChoices("TRACING_SERVICE_NAME")
    )

    # Client-side rate limiting configuration
    client_rate_limit_enabled: bool = Field(
        default=True, validation_alias=AliasChoices("CLIENT_RATE_LIMIT_ENABLED")
    )
    client_rate_limit_rpm: int = Field(
        default=500, validation_alias=AliasChoices("CLIENT_RATE_LIMIT_RPM")
    )
    client_rate_limit_tpm: int = Field(
        default=90000, validation_alias=AliasChoices("CLIENT_RATE_LIMIT_TPM")
    )
    client_rate_limit_burst: int = Field(
        default=100, validation_alias=AliasChoices("CLIENT_RATE_LIMIT_BURST")
    )
    client_rate_limit_adaptive: bool = Field(
        default=True, validation_alias=AliasChoices("CLIENT_RATE_LIMIT_ADAPTIVE")
    )

    @field_validator(
        "cors_allow_origins",
        "cors_allow_methods",
        "cors_allow_headers",
        "allowed_hosts",
        "allowed_base_url_hosts",
        "redact_log_fields",
    )
    @classmethod
    def parse_comma_separated(cls, v: Union[List[str], str]) -> List[str]:
        """Parse comma-separated string values into lists for configuration fields.

        Handles both string inputs (splitting by commas) and already-list inputs.
        Empty strings are converted to empty lists.

        Args:
            v: Input value which can be a string or list

        Returns:
            List of stripped, non-empty items
        """
        if isinstance(v, str):
            if v.strip() == "":
                return []
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    def __init__(self, **kwargs: Any) -> None:
        """Initialize settings object and perform validation.

        Calls parent constructor with provided keyword arguments, then validates
        required model settings and security configurations.

        Args:
            **kwargs: Keyword arguments for settings initialization

        Raises:
            SystemExit: If required settings are missing or security validation fails
        """
        super().__init__(**kwargs)
        self._validate_required_models()
        self._validate_security()

    def _validate_required_models(self) -> None:
        """Validate that required model settings are configured."""
        errors = []

        if not (self.openai_api_key and self.openai_api_key.strip()):
            errors.append(
                "API key is required. Set OPENAI_API_KEY in your environment or .env."
            )

        if not (self.big_model_name and self.big_model_name.strip()):
            errors.append(
                "BIG_MODEL_NAME is required. Set it in your environment or .env."
            )

        if not (self.small_model_name and self.small_model_name.strip()):
            errors.append(
                "SMALL_MODEL_NAME is required. Set it in your environment or .env."
            )

        if errors:
            error_message = "\n".join(errors)
            # Use logging instead of print to ensure proper error handling
            import logging

            logging.error(f"Configuration Error:\n{error_message}\n")
            sys.exit(1)

    def _validate_security(self) -> None:
        """Validates security configurations when RESTRICT_BASE_URL is enabled.

        Checks that OPENAI_BASE_URL uses HTTPS and the host is in ALLOWED_BASE_URL_HOSTS.
        Exits with error message if validation fails.
        """
        errors = []
        if self.restrict_base_url:
            try:
                parsed = urlparse(self.base_url)
                if parsed.scheme.lower() != "https":
                    errors.append(
                        "OPENAI_BASE_URL must use https when RESTRICT_BASE_URL is enabled."
                    )
                host = parsed.hostname or ""
                if host not in set(self.allowed_base_url_hosts or []):
                    errors.append(
                        "OPENAI_BASE_URL host is not in ALLOWED_BASE_URL_HOSTS."
                    )
            except Exception:
                errors.append("OPENAI_BASE_URL is invalid.")
        if errors:
            error_message = "\n".join(errors)
            # Use logging instead of print to ensure proper error handling
            import logging

            logging.error(f"Security Configuration Error:\n{error_message}\n")
            sys.exit(1)
