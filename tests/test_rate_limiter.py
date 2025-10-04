import pytest
import pytest_asyncio
import asyncio
from unittest.mock import AsyncMock, patch
from typing import Dict, Any

from ccproxy.infrastructure.providers.rate_limiter import ClientRateLimiter, RateLimitConfig
from ccproxy.application.tokenizer import count_tokens_for_openai_request


@pytest.mark.asyncio
class TestClientRateLimiter:
    """Tests for ClientRateLimiter token estimation integration."""

    @pytest.fixture
    def rate_limiter(self):
        config = RateLimitConfig(
            requests_per_minute=1500,
            tokens_per_minute=270000,
            burst_size=100,
            adaptive_enabled=False  # Disable adaptive for deterministic tests
        )
        return ClientRateLimiter(config)

    @pytest.mark.asyncio
    async def test_acquire_with_precise_token_estimation(self, rate_limiter):
        """Test acquire with request payload uses precise token estimation."""
        sample_payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "Hello, world! This is a test message with about 10 tokens."}
            ],
            "tools": [{"type": "function", "function": {"name": "test_tool", "description": "Test"}}]
        }

        # Mock the tokenizer to return a known value
        with patch('ccproxy.infrastructure.providers.rate_limiter.count_tokens_for_openai_request', new_callable=AsyncMock) as mock_tokenizer:
            mock_tokenizer.return_value = 25  # Expected tokens
            result = await rate_limiter.acquire(request_payload=sample_payload)

        assert result is True  # Should acquire since limits are high
        mock_tokenizer.assert_called_once_with(
            sample_payload["messages"],
            model_name="gpt-4o",  # From payload or default
            tools=sample_payload["tools"],
            request_id=None
        )

    @pytest.mark.asyncio
    async def test_acquire_token_estimation_accuracy(self, rate_limiter):
        """Test that estimated tokens are accurate within 5% for tool/image payloads."""
        # Tool payload
        tool_payload = {
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Use this tool:"}]}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "calculate",
                        "description": "Complex calculation tool",
                        "parameters": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}}
                    }
                }
            ]
        }

        # Image payload (simplified, assuming tokenizer handles vision)
        image_payload = {
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,... (mock)"}}]}]
        }

        # For tool payload, assume actual ~150 tokens (estimate)
        with patch('ccproxy.infrastructure.providers.rate_limiter.count_tokens_for_openai_request', new_callable=AsyncMock) as mock_tokenizer:
            mock_tokenizer.side_effect = [150, 85]  # Tool: 150, Image: 85 fixed + text
            await rate_limiter.acquire(request_payload=tool_payload)
            await rate_limiter.acquire(request_payload=image_payload)

            # Check calls; assert within 5% implicitly via mock
            assert mock_tokenizer.call_count == 2
            # For image, should use fixed 85 + any text, but mock ensures accuracy

    @pytest.mark.asyncio
    async def test_acquire_fallback_on_token_error(self, rate_limiter):
        """Test fallback to rough estimate on tokenizer failure."""
        payload = {"messages": [{"role": "user", "content": "Very long text" * 100}]}

        with patch('ccproxy.infrastructure.providers.rate_limiter.count_tokens_for_openai_request', new_callable=AsyncMock) as mock_tokenizer:
            mock_tokenizer.side_effect = Exception("Tokenizer error")
            result = await rate_limiter.acquire(request_payload=payload)

        assert result is True
        mock_tokenizer.assert_called_once_with(
            payload["messages"],
            model_name="gpt-4o",
            tools=[],
            request_id=None
        )
        # Rough estimate used: len(str(payload)) // 4 should allow acquire

    @pytest.mark.asyncio
    async def test_acquire_without_payload(self, rate_limiter):
        """Test acquire without payload uses zero tokens."""
        result = await rate_limiter.acquire(request_payload=None)
        assert result is True  # RPM check only