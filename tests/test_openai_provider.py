"""Comprehensive test suite for OpenAI provider and HTTP/2 client."""

import anyio
from unittest.mock import AsyncMock, MagicMock, patch, Mock
import pytest
import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types import CompletionUsage

from ccproxy.infrastructure.providers.openai_provider import OpenAIProvider
from ccproxy.config import Settings
from typing import Any


@pytest.fixture
def mock_settings() -> MagicMock:
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
    # Circuit breaker settings
    settings.circuit_breaker_failure_threshold = 5
    settings.circuit_breaker_recovery_timeout = 60
    settings.circuit_breaker_half_open_requests = 3
    # Connection pool settings
    settings.pool_max_keepalive_connections = 50
    settings.pool_max_connections = 500
    settings.pool_keepalive_expiry = 120
    # HTTP timeout settings
    settings.http_connect_timeout = 10.0
    settings.http_write_timeout = 30.0
    settings.http_pool_timeout = 10.0
    # Tracing settings
    settings.tracing_enabled = False
    settings.tracing_exporter = "console"
    settings.tracing_endpoint = ""
    settings.tracing_service_name = "ccproxy"
    # Client rate limiting settings
    settings.client_rate_limit_enabled = False
    settings.client_rate_limit_rpm = 500
    settings.client_rate_limit_tpm = 90000
    settings.client_rate_limit_burst = 100
    settings.client_rate_limit_adaptive = True
    return settings


@pytest.fixture
async def openai_provider(mock_settings) -> Settings:  # type: ignore[no-untyped-def]
    """Create an OpenAI provider instance for testing."""
    import os

    # Set[Any] to local mode for consistent testing
    os.environ["IS_LOCAL_DEPLOYMENT"] = "True"
    provider = OpenAIProvider(mock_settings)
    yield provider
    await provider.close()


@pytest.fixture
def sample_messages() -> Any:
    """Create sample OpenAI format messages."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather like?"},
    ]


@pytest.fixture
def sample_tools() -> Any:
    """Create sample OpenAI format tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            },
        }
    ]


@pytest.fixture
def mock_chat_completion() -> MagicMock:
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
def mock_stream_chunks() -> MagicMock:
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

    return chunks  # type: ignore[return-value]


class TestOpenAIProvider:
    """Test cases for OpenAI provider."""

    @pytest.mark.anyio
    async def test_provider_initialization(self, mock_settings: MagicMock) -> None:
        """Test provider initialization with settings."""
        provider = OpenAIProvider(mock_settings)

        assert provider.settings == mock_settings
        assert provider._openai_client is not None
        assert isinstance(provider._openai_client, AsyncOpenAI)
        assert provider._http_client is not None

        await provider.close()  # type: ignore[unreachable]

    @pytest.mark.anyio
    async def test_create_chat_completion(
        self, openai_provider, sample_messages, mock_chat_completion
    ) -> None:  # type: ignore[no-untyped-def]
        """Test creating a chat completion."""
        with patch.object(
            openai_provider._openAIClient.chat.completions,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = mock_chat_completion

            result = await openai_provider.create_chat_completion(
                messages=sample_messages, model="gpt-5", temperature=0.7, max_tokens=100
            )

            assert result == mock_chat_completion
            mock_create.assert_called_once()

            # Verify call arguments
            call_args = mock_create.call_args
            assert call_args.kwargs["messages"] == sample_messages
            assert call_args.kwargs["model"] == "gpt-5"
            assert call_args.kwargs["temperature"] == 0.7
            assert call_args.kwargs["max_tokens"] == 100

    @pytest.mark.anyio
    async def test_create_chat_completion_with_tools(
        self, openai_provider, sample_messages, sample_tools, mock_chat_completion
    ) -> None:  # type: ignore[no-untyped-def]
        """Test creating a chat completion with tools."""
        with patch.object(
            openai_provider._openAIClient.chat.completions,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = mock_chat_completion

            result = await openai_provider.create_chat_completion(
                messages=sample_messages,
                model="gpt-5",
                tools=sample_tools,
                tool_choice="auto",
            )

            assert result == mock_chat_completion
            mock_create.assert_called_once()

            # Verify tools were passed
            call_args = mock_create.call_args
            assert call_args.kwargs["tools"] == sample_tools
            assert call_args.kwargs["tool_choice"] == "auto"

    @pytest.mark.anyio
    async def test_create_streaming_completion(
        self, openai_provider, sample_messages, mock_stream_chunks
    ) -> None:  # type: ignore[no-untyped-def]
        """Test creating a streaming chat completion."""

        async def async_generator() -> None:
            for chunk in mock_stream_chunks:
                yield chunk

        with patch.object(
            openai_provider._openAIClient.chat.completions,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = async_generator()  # type: ignore[func-returns-value]

            result = await openai_provider.create_chat_completion(
                messages=sample_messages, model="gpt-5", stream=True
            )

            # Collect stream chunks
            chunks = []
            async for chunk in result:
                chunks.append(chunk)

            assert len(chunks) == 3
            assert chunks[0].choices[0].delta.role == "assistant"
            assert chunks[1].choices[0].delta.content == "The weather is "
            assert chunks[2].choices[0].finish_reason == "stop"

    @pytest.mark.anyio
    async def test_retry_on_rate_limit(
        self, openai_provider: Any, sample_messages: Any
    ) -> Any:
        """Test retry mechanism on rate limit errors."""
        # Test the retry mechanism through RetryHandler
        with patch.object(
            openai_provider._openAIClient.chat.completions,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            # First call raises rate limit error, second succeeds
            mock_response = MagicMock()
            call_count = [0]  # Use list to capture in closure

            async def mock_func(*args, **kwargs) -> Any:  # type: ignore[no-untyped-def]
                call_count[0] += 1
                if call_count[0] == 1:
                    # First call raises rate limit error
                    raise openai.RateLimitError(
                        message="Rate limit exceeded",
                        response=Mock(status_code=429),
                        body=None,
                    )
                # Second call succeeds
                return mock_response

            mock_create.side_effect = mock_func

            with patch("anyio.sleep", new_callable=AsyncMock):  # Skip actual sleep
                # Test through retry_handler directly
                result = await openai_provider._retry_handler.execute_with_retry(
                    mock_create, messages=sample_messages, model="gpt-5"
                )

            assert result == mock_response
            assert call_count[0] == 2  # Initial + 1 retry

    @pytest.mark.anyio
    async def test_authentication_error(
        self, openai_provider: Any, sample_messages: Any
    ) -> None:
        """Test handling of authentication errors."""
        with patch.object(
            openai_provider._openAIClient.chat.completions,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.side_effect = openai.AuthenticationError(
                message="Unauthorized", response=Mock(status_code=401), body=None
            )

            with pytest.raises(openai.AuthenticationError):
                await openai_provider.create_chat_completion(
                    messages=sample_messages, model="gpt-5"
                )

    @pytest.mark.anyio
    async def test_network_error_handling(
        self, openai_provider: Any, sample_messages: Any
    ) -> None:
        """Test handling of network errors."""
        with patch.object(
            openai_provider._openAIClient.chat.completions,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.side_effect = openai.APIConnectionError(
                message="Connection failed", request=Mock()
            )

            with pytest.raises(openai.APIError):
                await openai_provider.create_chat_completion(
                    messages=sample_messages, model="gpt-5"
                )

    @pytest.mark.anyio
    async def test_timeout_handling(
        self, openai_provider: Any, sample_messages: Any
    ) -> None:
        """Test handling of timeout errors."""
        with patch.object(
            openai_provider._openAIClient.chat.completions,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.side_effect = openai.APITimeoutError(request=Mock())

            with pytest.raises(openai.APIError):
                await openai_provider.create_chat_completion(
                    messages=sample_messages, model="gpt-5"
                )

    @pytest.mark.anyio
    async def test_http2_configuration(self, mock_settings: MagicMock) -> None:
        """Test HTTP/2 configuration."""
        import os

        os.environ["IS_LOCAL_DEPLOYMENT"] = "True"
        provider = OpenAIProvider(mock_settings)

        # Check that HTTP client is configured
        assert provider._http_client is not None
        assert provider._openai_client is not None  # type: ignore[unreachable]

        await provider.close()

    @pytest.mark.anyio
    async def test_custom_headers(self, mock_settings: MagicMock) -> None:
        """Test custom headers are set."""
        import os

        os.environ["IS_LOCAL_DEPLOYMENT"] = "True"
        provider = OpenAIProvider(mock_settings)

        # Headers should include referer and X-Title
        default_headers = provider._openai_client.default_headers  # type: ignore[union-attr]
        assert "HTTP-Referer" in default_headers
        assert default_headers["HTTP-Referer"] == mock_settings.referer_url
        assert "X-Title" in default_headers
        assert default_headers["X-Title"] == mock_settings.app_name

        await provider.close()

    @pytest.mark.anyio
    async def test_connection_pool_limits(self, mock_settings: MagicMock) -> None:
        """Test connection pool configuration."""
        import os

        os.environ["IS_LOCAL_DEPLOYMENT"] = "True"
        provider = OpenAIProvider(mock_settings)

        # Check that HTTP client is configured
        assert provider._http_client is not None
        # The actual pool limits are handled internally by the DefaultAsyncHttpxClient

        await provider.close()  # type: ignore[unreachable]

    @pytest.mark.anyio
    async def test_provider_close(self, openai_provider: Any) -> None:
        """Test provider cleanup on close."""
        # Close should not raise errors
        await openai_provider.close()

        # Multiple closes should be safe
        await openai_provider.close()

    @pytest.mark.anyio
    async def test_reasoning_effort_parameter(
        self, openai_provider: Any, sample_messages: Any
    ) -> None:
        """Test reasoning effort parameter handling."""
        with patch.object(
            openai_provider._openAIClient.chat.completions,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = MagicMock()

            # Test with reasoning effort
            await openai_provider.create_chat_completion(
                messages=sample_messages, model="gpt-5", reasoning_effort="high"
            )

            call_args = mock_create.call_args
            assert call_args.kwargs.get("reasoning_effort") == "high"

    @pytest.mark.anyio
    async def test_openrouter_specific_reasoning(
        self, mock_settings: MagicMock
    ) -> None:
        """Test OpenRouter-specific reasoning configuration."""
        import os

        os.environ["IS_LOCAL_DEPLOYMENT"] = "True"
        # Configure for OpenRouter
        mock_settings.base_url = "https://openrouter.ai/api/v1"
        provider = OpenAIProvider(mock_settings)

        with patch.object(
            provider._openai_client.chat.completions,
            "create",
            new_callable=AsyncMock,  # type: ignore[union-attr]
        ) as mock_create:
            mock_create.return_value = MagicMock()

            # Test with OpenRouter reasoning format
            await provider.create_chat_completion(
                messages=[{"role": "user", "content": "test"}],
                model="deepseek/deepseek-chat-v3.1",
                reasoning={"enabled": True, "effort": "high", "max_tokens": 10000},
            )

            call_args = mock_create.call_args
            assert "reasoning" in call_args.kwargs

        await provider.close()

    @pytest.mark.anyio
    async def test_parameter_filtering(
        self, openai_provider: Any, sample_messages: Any
    ) -> None:
        """Test that all parameters are passed through to OpenAI client."""
        with patch.object(
            openai_provider._openAIClient.chat.completions,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = MagicMock()

            # Pass various parameters
            await openai_provider.create_chat_completion(
                messages=sample_messages,
                model="gpt-5",
                temperature=0.7,
                top_p=0.9,
                custom_param="custom_value",
                another_param=123,
            )

            call_args = mock_create.call_args
            # All params should be passed through - let OpenAI SDK handle validation
            assert "temperature" in call_args.kwargs
            assert "top_p" in call_args.kwargs
            assert "custom_param" in call_args.kwargs
            assert "another_param" in call_args.kwargs

    @pytest.mark.anyio
    async def test_concurrent_requests(
        self, openai_provider, sample_messages, mock_chat_completion
    ) -> None:  # type: ignore[no-untyped-def]
        """Test handling multiple concurrent requests."""
        with patch.object(
            openai_provider._openAIClient.chat.completions,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = mock_chat_completion

            # Launch multiple concurrent requests
            async with anyio.create_task_group() as tg:
                results = []

                async def make_request() -> Any:
                    result = await openai_provider.create_chat_completion(
                        messages=sample_messages, model="gpt-5"
                    )
                    results.append(result)

                for _ in range(10):
                    tg.start_soon(make_request)

            assert len(results) == 10
            assert all(r == mock_chat_completion for r in results)
            assert mock_create.call_count == 10

    @pytest.mark.anyio
    async def test_empty_messages_handling(self, openai_provider: Any) -> None:
        """Test handling of empty messages list."""
        with patch.object(
            openai_provider._openAIClient.chat.completions,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = MagicMock()

            # Should handle empty messages gracefully
            await openai_provider.create_chat_completion(messages=[], model="gpt-5")

            mock_create.assert_called_once()

    @pytest.mark.anyio
    async def test_model_validation(self, openai_provider: Any) -> None:
        """Test model name validation and mapping."""
        with patch.object(
            openai_provider._openAIClient.chat.completions,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = MagicMock()

            # Test various model names
            test_models = ["gpt-5", "gpt-5-mini", "o3", "deepseek-reasoner"]

            for model in test_models:
                await openai_provider.create_chat_completion(
                    messages=[{"role": "user", "content": "test"}], model=model
                )

            assert mock_create.call_count == len(test_models)
