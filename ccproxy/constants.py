"""Constants module for CCProxy configuration.

Contains all constant values used throughout the application including model capabilities,
token limits, and configuration messages.
"""

from typing import Dict, Tuple, FrozenSet

# UTF-8 enforcement message for models that support developer role
UTF8_ENFORCEMENT_MESSAGE = "IMPORTANT: All responses must use proper UTF-8 encoding. Ensure all characters, including special characters and non-ASCII text, are properly encoded in UTF-8 format."

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
    "qwen/qwen3-235b-a22b-thinking-2507",
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
    "qwen/qwen3-235b-a22b-thinking-2507",
    "deepseek-reasoner",
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
TOP_TIER_OPENAI_MODELS: FrozenSet[str] = frozenset({
    "o3",
    "o3-2025-04-16",
    "gpt-5-2025-08-07",
    "gpt-5",
})

# Models that are in the top tier of the Anthropic API
TOP_TIER_ANTHROPIC_MODELS: FrozenSet[str] = frozenset({
    "claude-opus-4-1-20250805",
})

# Model input token limits (model_name, max_tokens)
MODEL_INPUT_TOKEN_LIMIT: FrozenSet[Tuple[str, int]] = frozenset({
    ("o3", 200_000),
    ("o3-2025-04-16", 200_000),
    ("o4-mini", 200_000),
    ("o4-mini-2025-04-16", 200_000),
    ("gpt-5-2025-08-07", 400_000),
    ("gpt-5", 400_000),
    ("gpt-5-mini-2025-08-07", 400_000),
    ("gpt-5-mini", 400_000),
    ("qwen/qwen3-coder", 262_144),
    ("qwen/qwen3-235b-a22b-thinking-2507", 262_144),
    ("qwen/qwen3-coder-480b-a35b-07-25", 262_144),
    ("z-ai/glm-4.5", 131_072),
    ("deepseek-reasoner", 131_072),
    ("deepseek-chat", 131_072),
    ("qwen/qwen3-coder", 262_144),
    ("deepseek/deepseek-chat-v3.1", 163_840),
    ("x-ai/grok-code-fast-1", 256_000),
})

MODEL_INPUT_TOKEN_LIMIT_MAP: Dict[str, int] = dict(MODEL_INPUT_TOKEN_LIMIT)

# Model maximum output token limits (model_name, max_tokens)
MODEL_MAX_OUTPUT_TOKEN_LIMIT: FrozenSet[Tuple[str, int]] = frozenset({
    ("o3", 200_000),
    ("o3-2025-04-16", 200_000),
    ("o4-mini", 200_000),
    ("gpt-5-2025-08-07", 128_000),
    ("gpt-5", 128_000),
    ("gpt-5-mini-2025-08-07", 128_000),
    ("gpt-5-mini", 128_000),
    ("deepseek-reasoner", 65_536),
    ("deepseek-chat", 8_192),
    ("qwen/qwen3-coder", 66_560),
    ("qwen/qwen3-coder-480b-a35b-07-25", 66_560),
    ("deepseek/deepseek-chat-v3.1", 134_144),
    ("x-ai/grok-code-fast-1", 10_000),
})

MODEL_MAX_OUTPUT_TOKEN_LIMIT_MAP: Dict[str, int] = dict(MODEL_MAX_OUTPUT_TOKEN_LIMIT)