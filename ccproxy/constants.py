"""Constants module for CCProxy configuration.

Contains all constant values used throughout the application including model capabilities,
token limits, configuration messages, and system defaults.
"""

from typing import Dict, Tuple, FrozenSet

# UTF-8 enforcement message for models that support developer role
UTF8_ENFORCEMENT_MESSAGE = "IMPORTANT: All responses must use proper UTF-8 encoding. Ensure all characters, including special characters and non-ASCII text, are properly encoded in UTF-8 format."

# Models that support reasoning features (e.g., reasoning_effort)
SUPPORT_REASONING_EFFORT_MODELS: FrozenSet[str] = frozenset(
    {
        "o3",
        "o3-2025-04-16",
        "o4-mini",
        "o4-mini-2025-04-16",
        "gpt-5-mini-2025-08-07",
        "gpt-5-mini",
        "gpt-5-2025-08-07",
        "gpt-5",
        "openai/gpt-5",
        "openai/gpt-5-mini",
    }
)

OPENROUTER_SUPPORT_REASONING_EFFORT_MODELS: FrozenSet[str] = frozenset(
    {
        "deepseek/deepseek-chat-v3.1",
        "x-ai/grok-code-fast-1",
        "x-ai/grok-4-fast:free",
        "z-ai/glm-4.5",
    }
)

# Models that do not support temperature (e.g., temperature)
NO_SUPPORT_TEMPERATURE_MODELS: FrozenSet[str] = frozenset(
    {
        "o3",
        "o3-2025-04-16",
        "o4-mini",
        "o4-mini-2025-04-16",
        "gpt-5-mini-2025-08-07",
        "gpt-5-mini",
        "gpt-5-2025-08-07",
        "gpt-5",
        "qwen/qwen3-235b-a22b-thinking-2507",
        "deepseek-reasoner",
        "openai/gpt-5",
        "openai/gpt-5-mini",
    }
)

# Models that support developer messages (e.g., developer_message)
SUPPORT_DEVELOPER_MESSAGE_MODELS: FrozenSet[str] = frozenset(
    {
        "o3",
        "o3-2025-04-16",
        "o4-mini",
        "o4-mini-2025-04-16",
        "gpt-5-mini-2025-08-07",
        "gpt-5-mini",
        "gpt-5-2025-08-07",
        "gpt-5",
    }
)

# Models that are in the top tier of the OpenAI API
TOP_TIER_OPENAI_MODELS: FrozenSet[str] = frozenset(
    {"o3", "o3-2025-04-16", "gpt-5-2025-08-07", "gpt-5", "openai/gpt-5"}
)

# Models that are in the top tier of the Anthropic API
TOP_TIER_ANTHROPIC_MODELS: FrozenSet[str] = frozenset(
    {
        "claude-opus-4-1-20250805",
    }
)

# Model input token limits (model_name, max_tokens)
MODEL_INPUT_TOKEN_LIMIT: FrozenSet[Tuple[str, int]] = frozenset(
    {
        ("o3", 200_000),
        ("o3-2025-04-16", 200_000),
        ("o4-mini", 200_000),
        ("o4-mini-2025-04-16", 200_000),
        ("gpt-5-2025-08-07", 400_000),
        ("gpt-5", 400_000),
        ("gpt-5-mini-2025-08-07", 400_000),
        ("gpt-5-mini", 400_000),
        ("openai/gpt-5", 400_000),
        ("openai/gpt-5-mini", 400_000),
        ("qwen/qwen3-coder", 262_144),
        ("qwen/qwen3-235b-a22b-thinking-2507", 262_144),
        ("qwen/qwen3-coder-480b-a35b-07-25", 262_144),
        ("z-ai/glm-4.5", 131_072),
        ("deepseek-reasoner", 131_072),
        ("deepseek-chat", 131_072),
        ("deepseek/deepseek-chat-v3.1", 163_840),
        ("x-ai/grok-code-fast-1", 256_000),
        ("qwen/qwen3-next-80b-a3b-thinking", 262_144),
        ("qwen/qwen3-coder-plus", 128_000),
        ("qwen/qwen3-coder-flash", 128_000),
        ("x-ai/grok-4-fast:free", 2_000_000),
        ("z-ai/glm-4.5", 131_072),
        ("moonshotai/kimi-k2-0905", 262_144),
    }
)

MODEL_INPUT_TOKEN_LIMIT_MAP: Dict[str, int] = dict(MODEL_INPUT_TOKEN_LIMIT)

# Model maximum output token limits (model_name, max_tokens)
MODEL_MAX_OUTPUT_TOKEN_LIMIT: FrozenSet[Tuple[str, int]] = frozenset(
    {
        ("o3", 200_000),
        ("o3-2025-04-16", 200_000),
        ("o4-mini", 200_000),
        ("gpt-5-2025-08-07", 128_000),
        ("gpt-5", 128_000),
        ("gpt-5-mini-2025-08-07", 128_000),
        ("gpt-5-mini", 128_000),
        ("openai/gpt-5", 128_000),
        ("openai/gpt-5-mini", 128_000),
        ("deepseek-reasoner", 65_536),
        ("deepseek-chat", 8_192),
        ("qwen/qwen3-coder", 66_560),
        ("qwen/qwen3-coder-480b-a35b-07-25", 66_560),
        ("qwen/qwen3-next-80b-a3b-thinking", 66_560),
        ("deepseek/deepseek-chat-v3.1", 134_144),
        ("x-ai/grok-code-fast-1", 10_000),
        ("qwen/qwen3-coder-plus", 66_560),
        ("qwen/qwen3-coder-flash", 66_560),
        ("x-ai/grok-4-fast:free", 30_720),
        ("z-ai/glm-4.5", 30_720),
        ("moonshotai/kimi-k2-0905", 30_720),
    }
)

MODEL_MAX_OUTPUT_TOKEN_LIMIT_MAP: Dict[str, int] = dict(MODEL_MAX_OUTPUT_TOKEN_LIMIT)

# ============================================================================
# Cache Configuration Constants
# ============================================================================

# Response Cache Settings
DEFAULT_CACHE_MAX_SIZE = 1000  # Maximum number of cached responses
DEFAULT_CACHE_MAX_MEMORY_MB = 500  # Maximum memory usage in MB
DEFAULT_CACHE_TTL_SECONDS = 3600  # Time-to-live for cached responses (1 hour)
DEFAULT_CACHE_CLEANUP_INTERVAL_SECONDS = 300  # Cleanup interval (5 minutes)
DEFAULT_CACHE_VALIDATION_FAILURE_THRESHOLD = 25  # Circuit breaker threshold
DEFAULT_CACHE_VALIDATION_FAILURE_RESET_TIME = 300  # Reset after 5 minutes

# Request Validator Cache Settings
REQUEST_VALIDATOR_CACHE_SIZE = 10_000  # LRU cache capacity
REQUEST_HASH_ALGORITHM = "sha256"  # Cryptographic hash for deduplication

# Token Count Cache Settings
TOKEN_COUNT_CACHE_TTL_SECONDS = 300  # TTL for token count cache (5 minutes)
TOKEN_COUNT_CACHE_MAX_SIZE = 1000  # Maximum entries in token count cache

# ============================================================================
# HTTP/Network Configuration Constants
# ============================================================================

# Connection Pool Settings
HTTP_MAX_CONNECTIONS = 500  # Maximum HTTP connections
HTTP_MAX_KEEPALIVE_CONNECTIONS = 50  # Keepalive connections
HTTP_KEEPALIVE_EXPIRY_SECONDS = 120  # Keepalive expiry (2 minutes)

# Timeout Settings
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120  # Default request timeout (2 minutes)
DEFAULT_CONNECT_TIMEOUT_SECONDS = 5  # Connection timeout
DEFAULT_READ_TIMEOUT_SECONDS = 30  # Read timeout
DEFAULT_WRITE_TIMEOUT_SECONDS = 10  # Write timeout
DEFAULT_POOL_TIMEOUT_SECONDS = 5  # Pool timeout

# Retry Settings
DEFAULT_MAX_RETRIES = 3  # Maximum retry attempts
DEFAULT_RETRY_BACKOFF_BASE = 2.0  # Exponential backoff base
DEFAULT_RETRY_BACKOFF_MAX = 60.0  # Maximum backoff time (1 minute)

# ============================================================================
# Server Configuration Constants
# ============================================================================

# Default server settings
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11434
DEFAULT_WORKER_CONNECTIONS = 1000
DEFAULT_MAX_REQUESTS = 1000
DEFAULT_MAX_REQUESTS_JITTER = 50
DEFAULT_WEB_CONCURRENCY = 4

# ============================================================================
# Logging and Monitoring Constants
# ============================================================================

# Log file paths
DEFAULT_LOG_FILE_PATH = "log.jsonl"
DEFAULT_ERROR_LOG_FILE_PATH = "error.jsonl"

# Monitoring settings
MONITORING_RECENT_DURATIONS_MAXLEN = 1000  # Recent request duration tracking
MONITORING_LATENCY_WINDOW_SECONDS = 60  # Latency calculation window

# String length limits for logging
LOG_STRING_MAX_LENGTH = 5000  # Maximum string length before truncation
LOG_STACK_TRACE_MAX_LENGTH = 10000  # Maximum stack trace length

# ============================================================================
# API/Protocol Constants
# ============================================================================

# Default API endpoints
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Allowed hosts for base URL
DEFAULT_ALLOWED_BASE_URL_HOSTS = ["api.openai.com", "openrouter.ai"]

# CORS settings
DEFAULT_CORS_ALLOW_METHODS = ["POST", "OPTIONS"]
DEFAULT_CORS_ALLOW_HEADERS = ["Authorization", "Content-Type", "X-Requested-With"]

# Security headers
SECURITY_HEADER_CSP = "default-src 'none'; frame-ancestors 'none'"
SECURITY_HEADER_X_CONTENT_TYPE = "nosniff"
SECURITY_HEADER_X_FRAME = "DENY"

# ============================================================================
# Message Processing Constants
# ============================================================================

# Message size limits
MAX_MESSAGE_CONTENT_LENGTH = 1_000_000  # 1MB max content per message
MAX_TOOL_INPUT_LENGTH = 100_000  # 100KB max tool input
MAX_IMAGE_DATA_LENGTH = 10_000_000  # 10MB max image data

# Truncation settings
DEFAULT_TRUNCATION_STRATEGY = "oldest_first"  # Default truncation strategy
TRUNCATION_BUFFER_TOKENS = 100  # Safety buffer for token limits

# ============================================================================
# Error Messages
# ============================================================================

ERROR_INVALID_REQUEST = "Invalid request format"
ERROR_AUTHENTICATION_FAILED = "Authentication failed"
ERROR_RATE_LIMIT_EXCEEDED = "Rate limit exceeded"
ERROR_UPSTREAM_TIMEOUT = "Upstream service timeout"
ERROR_CACHE_VALIDATION_FAILED = "Cache validation failed"
ERROR_UTF8_DECODE = "UTF-8 decoding error"
ERROR_MODEL_NOT_SUPPORTED = "Model not supported"
ERROR_TOKEN_LIMIT_EXCEEDED = "Token limit exceeded"
