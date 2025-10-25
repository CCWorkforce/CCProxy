"""Benchmarks for token counting operations."""

import anyio
from typing import Any
from ccproxy.application.tokenizer import (
    count_tokens_for_anthropic_request,
    count_tokens_for_openai_request,
    get_token_encoder,
)
from ccproxy.domain.models import Message, Tool
from ccproxy.config import Settings


# Sample data for benchmarks
SIMPLE_MESSAGES = [
    Message(role="user", content="Hello, how are you?"),
    Message(role="assistant", content="I'm doing well, thank you!"),
]

COMPLEX_MESSAGES = [
    Message(role="user", content="What's the weather like in San Francisco?"),
    Message(
        role="assistant",
        content=[
            {
                "type": "tool_use",
                "id": "toolu_123",
                "name": "get_weather",
                "input": {"location": "San Francisco", "unit": "celsius"},
            }
        ],
    ),
    Message(
        role="user",
        content=[
            {
                "type": "tool_result",
                "tool_use_id": "toolu_123",
                "content": "The weather in San Francisco is 18°C and sunny with light winds.",
            }
        ],
    ),
]

TOOLS = [
    Tool(
        name="get_weather",
        description="Get the current weather in a given location",
        input_schema={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The unit of temperature",
                },
            },
            "required": ["location"],
        },
    )
]


class TestTokenCountingPerformance:
    """Benchmark token counting operations."""

    def test_anthropic_simple_messages(self: Any, benchmark: Any) -> Any:
        """Benchmark token counting for simple Anthropic messages."""

        async def count_tokens() -> Any:
            return await count_tokens_for_anthropic_request(
                messages=SIMPLE_MESSAGES,
                system=None,
                model_name="claude-3-5-sonnet-20241022",
                tools=None,
                request_id="bench",
                settings=Settings(cache_token_counts_enabled=False),
            )

        result = benchmark(anyio.run, count_tokens)
        assert result > 0

    def test_anthropic_complex_messages_with_tools(self: Any, benchmark: Any) -> Any:
        """Benchmark token counting for complex messages with tools."""

        async def count_tokens() -> Any:
            return await count_tokens_for_anthropic_request(
                messages=COMPLEX_MESSAGES,
                system="You are a helpful assistant.",
                model_name="claude-3-5-sonnet-20241022",
                tools=TOOLS,
                request_id="bench",
                settings=Settings(cache_token_counts_enabled=False),
            )

        result = benchmark(anyio.run, count_tokens)
        assert result > 0

    def test_anthropic_with_cache_hit(self: Any, benchmark: Any) -> Any:
        """Benchmark token counting with cache hit."""

        async def count_tokens_cached() -> Any:
            # First call to populate cache
            await count_tokens_for_anthropic_request(
                messages=SIMPLE_MESSAGES,
                system=None,
                model_name="claude-3-5-sonnet-20241022",
                tools=None,
                request_id="bench",
                settings=Settings(cache_token_counts_enabled=True),
            )

            # Second call should hit cache
            return await count_tokens_for_anthropic_request(
                messages=SIMPLE_MESSAGES,
                system=None,
                model_name="claude-3-5-sonnet-20241022",
                tools=None,
                request_id="bench",
                settings=Settings(cache_token_counts_enabled=True),
            )

        result = benchmark(anyio.run, count_tokens_cached)
        assert result > 0

    def test_openai_simple_messages(self: Any, benchmark: Any) -> Any:
        """Benchmark token counting for simple OpenAI messages."""

        async def count_tokens() -> Any:
            return await count_tokens_for_openai_request(
                messages=[
                    {"role": "user", "content": "Hello, how are you?"},
                    {"role": "assistant", "content": "I'm doing well, thank you!"},
                ],
                model_name="gpt-4",
                tools=None,
                request_id="bench",
            )

        result = benchmark(anyio.run, count_tokens)
        assert result > 0

    def test_openai_with_tools(self: Any, benchmark: Any) -> Any:
        """Benchmark token counting for OpenAI messages with tools."""

        async def count_tokens() -> Any:
            return await count_tokens_for_openai_request(
                messages=[
                    {"role": "user", "content": "What's the weather like?"},
                    {
                        "role": "assistant",
                        "content": None,  # type: ignore[dict-item]
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "San Francisco", "unit": "celsius"}',
                                },
                            }
                        ],
                    },
                ],
                model_name="gpt-4",
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get the current weather",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "location": {"type": "string"},
                                    "unit": {"type": "string"},
                                },
                                "required": ["location"],
                            },
                        },
                    }
                ],
                request_id="bench",
            )

        result = benchmark(anyio.run, count_tokens)
        assert result > 0


class TestTokenEncoderPerformance:
    """Benchmark token encoder operations."""

    def test_get_token_encoder(self, benchmark) -> None:  # type: ignore[no-untyped-def]
        """Benchmark token encoder retrieval (with caching)."""
        result = benchmark(get_token_encoder, "gpt-4")
        assert result is not None

    def test_tiktoken_encode_small(self, benchmark) -> None:  # type: ignore[no-untyped-def]
        """Benchmark tiktoken encoding for small text."""
        encoder = get_token_encoder("gpt-4")
        text = "Hello, world!"

        result = benchmark(encoder.encode, text)
        assert len(result) > 0

    def test_tiktoken_encode_medium(self, benchmark) -> None:  # type: ignore[no-untyped-def]
        """Benchmark tiktoken encoding for medium text."""
        encoder = get_token_encoder("gpt-4")
        text = "This is a longer piece of text that contains multiple sentences. " * 10

        result = benchmark(encoder.encode, text)
        assert len(result) > 0

    def test_tiktoken_encode_large(self, benchmark) -> None:  # type: ignore[no-untyped-def]
        """Benchmark tiktoken encoding for large text."""
        encoder = get_token_encoder("gpt-4")
        text = "This is a very long piece of text with many repetitions. " * 100

        result = benchmark(encoder.encode, text)
        assert len(result) > 0
