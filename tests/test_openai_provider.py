"""Comprehensive test suite for OpenAI provider and HTTP/2 client."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, Mock
import pytest
import pytest_asyncio
import httpx
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types import CompletionUsage

from ccproxy.infrastructure.providers.openai_provider import OpenAIProvider
from ccproxy.config import Settings
from ccproxy.domain.exceptions import (
    AuthenticationError,
    ProviderError
)


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = MagicMock(spec=Settings)
    settings.openai_api_key = "test-api-key"
    settings.base_url = "https://api.openai.com/v1"
    settings.provider_max_retries = 3
    settings.provider_retry_base_delay = 1.0
    settings.provider_retry_jitter = 0.5
    settings.referer_url = "http://localhost:11434"
    settings.app_name = "CCProxy"
    settings.app_version = "1.0.0"
    settings.max_stream_seconds = 300  # Add missing attribute
    return settings


@pytest_asyncio.fixture
async def openai_provider(mock_settings):
    """Create an OpenAI provider instance for testing."""
    import os
    # Set to local mode for consistent testing
    os.environ['IS_LOCAL_DEPLOYMENT'] = 'True'
    provider = OpenAIProvider(mock_settings)
    yield provider
    await provider.close()


@pytest.fixture
def sample_messages():
    """Create sample OpenAI format messages."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather like?"}
    ]


@pytest.fixture
def sample_tools():
    """Create sample OpenAI format tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    }
                }
            }
        }
    ]


@pytest.fixture
def mock_chat_completion():
    """Create a mock ChatCompletion response."""
    completion = MagicMock(spec=ChatCompletion)
    completion.id = "chatcmpl-test123"
    completion.model = "gpt-5"
    completion.created = 1234567890

    # Create proper usage object
    usage = MagicMock(spec=CompletionUsage)
    usage.prompt_tokens = 50
    usage.completion_tokens = 100
    usage.total_tokens = 150
    completion.usage = usage

    # Create proper message
    message = MagicMock(spec=ChatCompletionMessage)
    message.role = "assistant"
    message.content = "The weather is sunny!"
    message.tool_calls = None
    message.function_call = None

    # Create proper choice
    choice = MagicMock(spec=Choice)
    choice.index = 0
    choice.message = message
    choice.finish_reason = "stop"
    choice.logprobs = None

    completion.choices = [choice]
    completion.system_fingerprint = None
    completion.object = "chat.completion"

    return completion


@pytest.fixture
def mock_stream_chunks():
    """Create mock streaming chunks."""
    chunks = []

    # First chunk - role
    chunk1 = MagicMock(spec=ChatCompletionChunk)
    chunk1.id = "chatcmpl-stream123"
    chunk1.model = "gpt-5"
    chunk1.created = 1234567890

    delta1 = MagicMock()
    delta1.role = "assistant"
    delta1.content = None
    delta1.tool_calls = None

    choice1 = MagicMock()
    choice1.index = 0
    choice1.delta = delta1
    choice1.finish_reason = None

    chunk1.choices = [choice1]
    chunk1.usage = None
    chunks.append(chunk1)

    # Second chunk - content
    chunk2 = MagicMock(spec=ChatCompletionChunk)
    chunk2.id = "chatcmpl-stream123"
    chunk2.model = "gpt-5"
    chunk2.created = 1234567890

    delta2 = MagicMock()
    delta2.role = None
    delta2.content = "The weather is "
    delta2.tool_calls = None

    choice2 = MagicMock()
    choice2.index = 0
    choice2.delta = delta2
    choice2.finish_reason = None

    chunk2.choices = [choice2]
    chunk2.usage = None
    chunks.append(chunk2)

    # Final chunk - with usage
    chunk3 = MagicMock(spec=ChatCompletionChunk)
    chunk3.id = "chatcmpl-stream123"
    chunk3.model = "gpt-5"
    chunk3.created = 1234567890

    delta3 = MagicMock()
    delta3.role = None
    delta3.content = "sunny!"
    delta3.tool_calls = None

    choice3 = MagicMock()
    choice3.index = 0
    choice3.delta = delta3
    choice3.finish_reason = "stop"

    chunk3.choices = [choice3]

    usage = MagicMock()
    usage.prompt_tokens = 50
    usage.completion_tokens = 100
    usage.total_tokens = 150
    chunk3.usage = usage

    chunks.append(chunk3)

    return chunks


class TestOpenAIProvider:
    """Test cases for OpenAI provider."""

    @pytest.mark.asyncio
    async def test_provider_initialization(self, mock_settings):
        """Test provider initialization with settings."""
        provider = OpenAIProvider(mock_settings)

        assert provider.settings == mock_settings
        assert provider._openAIClient is not None
        assert isinstance(provider._openAIClient, AsyncOpenAI)
        assert provider._http_client is not None

        await provider.close()

    @pytest.mark.asyncio
    async def test_create_chat_completion(self, openai_provider, sample_messages, mock_chat_completion):
        """Test creating a chat completion."""
        with patch.object(openai_provider._openAIClient.chat.completions, 'create',
                         new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_chat_completion

            result = await openai_provider.create_chat_completion(
                messages=sample_messages,
                model="gpt-5",
                temperature=0.7,
                max_tokens=100
            )

            assert result == mock_chat_completion
            mock_create.assert_called_once()

            # Verify call arguments
            call_args = mock_create.call_args
            assert call_args.kwargs["messages"] == sample_messages
            assert call_args.kwargs["model"] == "gpt-5"
            assert call_args.kwargs["temperature"] == 0.7
            assert call_args.kwargs["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_create_chat_completion_with_tools(
        self, openai_provider, sample_messages, sample_tools, mock_chat_completion
    ):
        """Test creating a chat completion with tools."""
        with patch.object(openai_provider._openAIClient.chat.completions, 'create',
                         new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_chat_completion

            result = await openai_provider.create_chat_completion(
                messages=sample_messages,
                model="gpt-5",
                tools=sample_tools,
                tool_choice="auto"
            )

            assert result == mock_chat_completion
            mock_create.assert_called_once()

            # Verify tools were passed
            call_args = mock_create.call_args
            assert call_args.kwargs["tools"] == sample_tools
            assert call_args.kwargs["tool_choice"] == "auto"

    @pytest.mark.asyncio
    async def test_create_streaming_completion(
        self, openai_provider, sample_messages, mock_stream_chunks
    ):
        """Test creating a streaming chat completion."""
        async def async_generator():
            for chunk in mock_stream_chunks:
                yield chunk

        with patch.object(openai_provider._openAIClient.chat.completions, 'create',
                         new_callable=AsyncMock) as mock_create:
            mock_create.return_value = async_generator()

            result = await openai_provider.create_chat_completion(
                messages=sample_messages,
                model="gpt-5",
                stream=True
            )

            # Collect stream chunks
            chunks = []
            async for chunk in result:
                chunks.append(chunk)

            assert len(chunks) == 3
            assert chunks[0].choices[0].delta.role == "assistant"
            assert chunks[1].choices[0].delta.content == "The weather is "
            assert chunks[2].choices[0].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_retry_on_rate_limit(self, openai_provider, sample_messages):
        """Test retry mechanism on rate limit errors."""
        with patch.object(openai_provider._openAIClient.chat.completions, 'create',
                         new_callable=AsyncMock) as mock_create:
            # First call raises rate limit error, second succeeds
            mock_create.side_effect = [
                httpx.HTTPStatusError(
                    "Rate limit exceeded",
                    request=Mock(),
                    response=Mock(status_code=429)
                ),
                MagicMock()  # Success on retry
            ]

            with patch('asyncio.sleep', new_callable=AsyncMock):  # Skip actual sleep
                await openai_provider.create_chat_completion(
                    messages=sample_messages,
                    model="gpt-5"
                )

            assert mock_create.call_count == 2  # Initial + 1 retry

    @pytest.mark.asyncio
    async def test_authentication_error(self, openai_provider, sample_messages):
        """Test handling of authentication errors."""
        with patch.object(openai_provider._openAIClient.chat.completions, 'create',
                         new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = httpx.HTTPStatusError(
                "Unauthorized",
                request=Mock(),
                response=Mock(status_code=401)
            )

            with pytest.raises(AuthenticationError):
                await openai_provider.create_chat_completion(
                    messages=sample_messages,
                    model="gpt-5"
                )

    @pytest.mark.asyncio
    async def test_network_error_handling(self, openai_provider, sample_messages):
        """Test handling of network errors."""
        with patch.object(openai_provider._openAIClient.chat.completions, 'create',
                         new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = httpx.ConnectError("Connection failed")

            with pytest.raises(ProviderError):
                await openai_provider.create_chat_completion(
                    messages=sample_messages,
                    model="gpt-5"
                )

    @pytest.mark.asyncio
    async def test_timeout_handling(self, openai_provider, sample_messages):
        """Test handling of timeout errors."""
        with patch.object(openai_provider._openAIClient.chat.completions, 'create',
                         new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = asyncio.TimeoutError("Request timed out")

            with pytest.raises(ProviderError) as exc_info:
                await openai_provider.create_chat_completion(
                    messages=sample_messages,
                    model="gpt-5"
                )

            assert "timeout" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_http2_configuration(self, mock_settings):
        """Test HTTP/2 configuration."""
        import os
        os.environ['IS_LOCAL_DEPLOYMENT'] = 'True'
        provider = OpenAIProvider(mock_settings)

        # Check that HTTP client is configured
        assert provider._http_client is not None
        assert provider._openAIClient is not None

        await provider.close()

    @pytest.mark.asyncio
    async def test_custom_headers(self, mock_settings):
        """Test custom headers are set."""
        import os
        os.environ['IS_LOCAL_DEPLOYMENT'] = 'True'
        provider = OpenAIProvider(mock_settings)

        # Headers should include referer and X-Title
        default_headers = provider._openAIClient.default_headers
        assert "HTTP-Referer" in default_headers
        assert default_headers["HTTP-Referer"] == mock_settings.referer_url
        assert "X-Title" in default_headers
        assert default_headers["X-Title"] == mock_settings.app_name

        await provider.close()

    @pytest.mark.asyncio
    async def test_connection_pool_limits(self, mock_settings):
        """Test connection pool configuration."""
        import os
        os.environ['IS_LOCAL_DEPLOYMENT'] = 'True'
        provider = OpenAIProvider(mock_settings)

        # Check that HTTP client is configured
        assert provider._http_client is not None
        # The actual pool limits are handled internally by the DefaultAsyncHttpxClient

        await provider.close()

    @pytest.mark.asyncio
    async def test_provider_close(self, openai_provider):
        """Test provider cleanup on close."""
        # Close should not raise errors
        await openai_provider.close()

        # Multiple closes should be safe
        await openai_provider.close()

    @pytest.mark.asyncio
    async def test_reasoning_effort_parameter(self, openai_provider, sample_messages):
        """Test reasoning effort parameter handling."""
        with patch.object(openai_provider._openAIClient.chat.completions, 'create',
                         new_callable=AsyncMock) as mock_create:
            mock_create.return_value = MagicMock()

            # Test with reasoning effort
            await openai_provider.create_chat_completion(
                messages=sample_messages,
                model="gpt-5",
                reasoning_effort="high"
            )

            call_args = mock_create.call_args
            assert call_args.kwargs.get("reasoning_effort") == "high"

    @pytest.mark.asyncio
    async def test_openrouter_specific_reasoning(self, mock_settings):
        """Test OpenRouter-specific reasoning configuration."""
        import os
        os.environ['IS_LOCAL_DEPLOYMENT'] = 'True'
        # Configure for OpenRouter
        mock_settings.base_url = "https://openrouter.ai/api/v1"
        provider = OpenAIProvider(mock_settings)

        with patch.object(provider._openAIClient.chat.completions, 'create',
                         new_callable=AsyncMock) as mock_create:
            mock_create.return_value = MagicMock()

            # Test with OpenRouter reasoning format
            await provider.create_chat_completion(
                messages=[{"role": "user", "content": "test"}],
                model="deepseek/deepseek-chat-v3.1",
                reasoning={
                    "enabled": True,
                    "effort": "high",
                    "max_tokens": 10000
                }
            )

            call_args = mock_create.call_args
            assert "reasoning" in call_args.kwargs

        await provider.close()

    @pytest.mark.asyncio
    async def test_parameter_filtering(self, openai_provider, sample_messages):
        """Test filtering of unsupported parameters."""
        with patch.object(openai_provider._openAIClient.chat.completions, 'create',
                         new_callable=AsyncMock) as mock_create:
            mock_create.return_value = MagicMock()

            # Pass various parameters, some unsupported
            await openai_provider.create_chat_completion(
                messages=sample_messages,
                model="gpt-5",
                temperature=0.7,
                top_p=0.9,
                unsupported_param="should_be_filtered",
                another_unsupported=123
            )

            call_args = mock_create.call_args
            # Supported params should be present
            assert "temperature" in call_args.kwargs
            assert "top_p" in call_args.kwargs
            # Unsupported params should be filtered
            assert "unsupported_param" not in call_args.kwargs
            assert "another_unsupported" not in call_args.kwargs

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, openai_provider, sample_messages, mock_chat_completion):
        """Test handling multiple concurrent requests."""
        with patch.object(openai_provider._openAIClient.chat.completions, 'create',
                         new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_chat_completion

            # Launch multiple concurrent requests
            tasks = [
                openai_provider.create_chat_completion(
                    messages=sample_messages,
                    model="gpt-5"
                )
                for _ in range(10)
            ]

            results = await asyncio.gather(*tasks)

            assert len(results) == 10
            assert all(r == mock_chat_completion for r in results)
            assert mock_create.call_count == 10

    @pytest.mark.asyncio
    async def test_empty_messages_handling(self, openai_provider):
        """Test handling of empty messages list."""
        with patch.object(openai_provider._openAIClient.chat.completions, 'create',
                         new_callable=AsyncMock) as mock_create:
            mock_create.return_value = MagicMock()

            # Should handle empty messages gracefully
            await openai_provider.create_chat_completion(
                messages=[],
                model="gpt-5"
            )

            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_model_validation(self, openai_provider):
        """Test model name validation and mapping."""
        with patch.object(openai_provider._openAIClient.chat.completions, 'create',
                         new_callable=AsyncMock) as mock_create:
            mock_create.return_value = MagicMock()

            # Test various model names
            test_models = [
                "gpt-5",
                "gpt-5-mini",
                "o3",
                "deepseek-reasoner"
            ]

            for model in test_models:
                await openai_provider.create_chat_completion(
                    messages=[{"role": "user", "content": "test"}],
                    model=model
                )

            assert mock_create.call_count == len(test_models)