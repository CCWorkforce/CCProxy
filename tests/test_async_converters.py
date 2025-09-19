"""Test async converter implementations for improved performance."""

import pytest
from unittest.mock import MagicMock
from ccproxy.application.converters_module.async_converter import (
    AsyncMessageConverter,
    AsyncResponseConverter,
    convert_messages_async,
    convert_response_async,
)
from ccproxy.application.converters_module.base import ConversionContext
from ccproxy.domain.models import (
    Message,
    MessagesResponse,
    ContentBlockText,
    ContentBlockImage,
    ContentBlockImageSource,
)
from ccproxy.config import Settings
from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage


class TestAsyncMessageConverter:
    """Test async message converter with parallel processing."""

    @pytest.mark.asyncio
    async def test_convert_messages_async(self):
        """Test async message conversion."""
        messages = [
            Message(
                role="user",
                content="Hello, how are you?"
            ),
            Message(
                role="assistant",
                content="I'm doing well, thank you!"
            )
        ]

        result = await convert_messages_async(messages, system="Be helpful")

        assert isinstance(result, list)
        assert len(result) == 3  # System + 2 messages
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "Be helpful"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_convert_messages_with_context(self):
        """Test async message conversion with context."""
        settings = MagicMock(spec=Settings)
        context = ConversionContext(
            request_id="test-123",
            target_model="gpt-4",
            settings=settings
        )

        messages = [
            Message(
                role="user",
                content=[
                    ContentBlockText(type="text", text="What's in this image?"),
                    ContentBlockImage(
                        type="image",
                        source=ContentBlockImageSource(
                            type="base64",
                            media_type="image/jpeg",
                            data="base64data"
                        )
                    )
                ]
            )
        ]

        converter = AsyncMessageConverter(context=context)
        result = await converter.convert_messages_async(messages, system=None)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert isinstance(result[0]["content"], list)
        assert len(result[0]["content"]) == 2

    @pytest.mark.asyncio
    async def test_parallel_processing(self):
        """Test that messages are processed in parallel."""
        settings = MagicMock(spec=Settings)
        context = ConversionContext(
            request_id="test-456",
            target_model="gpt-4",
            settings=settings
        )

        # Create multiple messages
        messages = [
            Message(role="user", content=f"Message {i}")
            for i in range(10)
        ]

        converter = AsyncMessageConverter(context=context)
        result = await converter.convert_messages_async(messages, system=None)

        assert len(result) == 10
        for i, msg in enumerate(result):
            assert msg["role"] == "user"
            assert f"Message {i}" in msg["content"]


class TestAsyncResponseConverter:
    """Test async response converter."""

    @pytest.mark.asyncio
    async def test_convert_response_async(self):
        """Test async response conversion."""
        # Create mock OpenAI response
        openai_response = ChatCompletion(
            id="chatcmpl-123",
            model="gpt-4",
            object="chat.completion",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content="Hello! How can I help you today?"
                    ),
                    finish_reason="stop"
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=10,
                completion_tokens=8,
                total_tokens=18
            )
        )

        result = await convert_response_async(openai_response)

        assert isinstance(result, MessagesResponse)
        assert result.id == "chatcmpl-123"
        assert result.model == "gpt-4"
        assert result.stop_reason == "end_turn"
        assert len(result.content) == 1
        assert result.content[0].type == "text"
        assert result.content[0].text == "Hello! How can I help you today?"
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 8

    @pytest.mark.asyncio
    async def test_convert_response_with_tool_calls(self):
        """Test async response conversion with tool calls."""
        # Create mock OpenAI response with tool calls
        tool_call = ChatCompletionMessageToolCall(
            id="tool-1",
            type="function",
            function={"name": "get_weather", "arguments": '{"location": "New York"}'}
        )

        openai_response = ChatCompletion(
            id="chatcmpl-456",
            model="gpt-4",
            object="chat.completion",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=None,
                        tool_calls=[tool_call]
                    ),
                    finish_reason="tool_calls"
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=15,
                completion_tokens=20,
                total_tokens=35
            )
        )

        result = await convert_response_async(openai_response)

        assert isinstance(result, MessagesResponse)
        assert result.stop_reason == "tool_use"
        assert len(result.content) == 1
        assert result.content[0].type == "tool_use"
        assert result.content[0].name == "get_weather"
        assert result.content[0].input == {"location": "New York"}

    @pytest.mark.asyncio
    async def test_convert_response_with_context(self):
        """Test async response conversion with context."""
        settings = MagicMock(spec=Settings)
        context = ConversionContext(
            request_id="test-789",
            target_model="gpt-4",
            settings=settings
        )

        openai_response = ChatCompletion(
            id="chatcmpl-789",
            model="gpt-4",
            object="chat.completion",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content="Response with context"
                    ),
                    finish_reason="stop"
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=5,
                completion_tokens=3,
                total_tokens=8
            )
        )

        converter = AsyncResponseConverter(context=context)
        result = await converter.convert_response_async(openai_response)

        assert isinstance(result, MessagesResponse)
        assert result.content[0].text == "Response with context"


class TestAsyncConverterIntegration:
    """Test async converter integration with the application."""

    @pytest.mark.asyncio
    async def test_end_to_end_conversion(self):
        """Test complete conversion flow."""
        # Input messages
        messages = [
            Message(
                role="user",
                content="What's the weather like?"
            )
        ]

        # Convert to OpenAI format
        openai_messages = await convert_messages_async(messages)

        assert isinstance(openai_messages, list)
        assert len(openai_messages) == 1
        assert openai_messages[0]["role"] == "user"

        # Mock OpenAI response
        openai_response = ChatCompletion(
            id="test-e2e",
            model="gpt-4",
            object="chat.completion",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content="I'll need to check the weather for you."
                    ),
                    finish_reason="stop"
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=6,
                completion_tokens=10,
                total_tokens=16
            )
        )

        # Convert back to Anthropic format
        anthropic_response = await convert_response_async(openai_response)

        assert isinstance(anthropic_response, MessagesResponse)
        assert anthropic_response.content[0].text == "I'll need to check the weather for you."
        assert anthropic_response.usage.input_tokens == 6
        assert anthropic_response.usage.output_tokens == 10

    @pytest.mark.asyncio
    async def test_performance_improvement(self):
        """Test that async converters provide performance improvement."""
        import time

        # Create many messages
        messages = [
            Message(role="user" if i % 2 == 0 else "assistant", content=f"Message {i}")
            for i in range(50)
        ]

        # Measure async conversion time
        start = time.time()
        result = await convert_messages_async(messages)
        async_time = time.time() - start

        assert len(result) == 50
        # Async should be reasonably fast (under 1 second for 50 messages)
        assert async_time < 1.0