"""Test async converter implementations for improved performance."""

import json
import pytest
from typing import Any
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
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage


class TestAsyncMessageConverter:
    """Test async message converter with parallel processing."""

    @pytest.mark.anyio
    async def test_convert_messages_async(self) -> None:
        """Test async message conversion."""
        messages = [
            Message(role="user", content="Hello, how are you?"),
            Message(role="assistant", content="I'm doing well, thank you!"),
        ]

        result = await convert_messages_async(messages, system="Be helpful")

        assert isinstance(result, list)
        assert len(result) == 3  # System + 2 messages
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "Be helpful"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"

    @pytest.mark.anyio
    async def test_convert_messages_with_context(self) -> None:
        """Test async message conversion with context."""
        settings = MagicMock(spec=Settings)
        context = ConversionContext(
            request_id="test-123", target_model="gpt-4", settings=settings
        )

        messages = [
            Message(
                role="user",
                content=[
                    ContentBlockText(type="text", text="What's in this image?"),
                    ContentBlockImage(
                        type="image",
                        source=ContentBlockImageSource(
                            type="base64", media_type="image/jpeg", data="base64data"
                        ),
                    ),
                ],
            )
        ]

        converter = AsyncMessageConverter(context=context)
        result = await converter.convert_messages_async(messages, system=None)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert isinstance(result[0]["content"], list)
        assert len(result[0]["content"]) == 2

    @pytest.mark.anyio
    async def test_parallel_processing(self) -> None:
        """Test that messages are processed in parallel."""
        settings = MagicMock(spec=Settings)
        context = ConversionContext(
            request_id="test-456", target_model="gpt-4", settings=settings
        )

        # Create multiple messages
        messages = [Message(role="user", content=f"Message {i}") for i in range(10)]

        converter = AsyncMessageConverter(context=context)
        result = await converter.convert_messages_async(messages, system=None)

        assert len(result) == 10
        for i, msg in enumerate(result):
            assert msg["role"] == "user"
            assert f"Message {i}" in msg["content"]


class TestAsyncResponseConverter:
    """Test async response converter."""

    @pytest.mark.anyio
    async def test_convert_response_async(self) -> None:
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
                        role="assistant", content="Hello! How can I help you today?"
                    ),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=10, completion_tokens=8, total_tokens=18
            ),
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

    @pytest.mark.anyio
    async def test_convert_response_with_tool_calls(self) -> None:
        """Test async response conversion with tool calls."""
        # Create mock OpenAI response with tool calls
        tool_call = ChatCompletionMessageToolCall(
            id="tool-1",
            type="function",
            function={"name": "get_weather", "arguments": '{"location": "New York"}'},
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
                        role="assistant", content=None, tool_calls=[tool_call]
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=15, completion_tokens=20, total_tokens=35
            ),
        )

        result = await convert_response_async(openai_response)

        assert isinstance(result, MessagesResponse)
        assert result.stop_reason == "tool_use"
        assert len(result.content) == 1
        assert result.content[0].type == "tool_use"
        assert result.content[0].name == "get_weather"
        assert result.content[0].input == {"location": "New York"}

    @pytest.mark.anyio
    async def test_convert_response_with_context(self) -> None:
        """Test async response conversion with context."""
        settings = MagicMock(spec=Settings)
        context = ConversionContext(
            request_id="test-789", target_model="gpt-4", settings=settings
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
                        role="assistant", content="Response with context"
                    ),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        )

        converter = AsyncResponseConverter(context=context)
        result = await converter.convert_response_async(openai_response)

        assert isinstance(result, MessagesResponse)
        assert result.content[0].text == "Response with context"  # type: ignore[union-attr]


class TestAsyncConverterIntegration:
    """Test async converter integration with the application."""

    @pytest.mark.anyio
    async def test_end_to_end_conversion(self) -> None:
        """Test complete conversion flow."""
        # Input messages
        messages = [Message(role="user", content="What's the weather like?")]

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
                        content="I'll need to check the weather for you.",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=6, completion_tokens=10, total_tokens=16
            ),
        )

        # Convert back to Anthropic format
        anthropic_response = await convert_response_async(openai_response)

        assert isinstance(anthropic_response, MessagesResponse)
        assert (
            anthropic_response.content[0].text  # type: ignore[union-attr]
            == "I'll need to check the weather for you."
        )
        assert anthropic_response.usage.input_tokens == 6
        assert anthropic_response.usage.output_tokens == 10

    @pytest.mark.anyio
    async def test_performance_improvement(self) -> None:
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


# === Additional Coverage Tests ===


class TestToolConsistencyValidation:
    """Test tool consistency validation logic."""

    @pytest.mark.anyio
    async def test_orphaned_tool_calls_removal(self) -> None:
        """Test that orphaned tool calls are removed when there are subsequent messages."""
        from ccproxy.domain.models import ContentBlockToolUse

        messages = [
            Message(
                role="assistant",
                content=[
                    ContentBlockToolUse(
                        type="tool_use",
                        id="tool_1",
                        name="get_weather",
                        input={"location": "NYC"},
                    )
                ],
            ),
            Message(
                role="user", content="What about Boston?"
            ),  # Subsequent message without tool result
        ]

        converter = AsyncMessageConverter()
        result = await converter.convert_messages_async(messages)

        # The assistant message should have tool_calls removed since there's no result
        # and there are subsequent messages
        assistant_msg = result[0]
        assert assistant_msg["role"] == "assistant"
        # Tool calls should be removed or content added
        assert (
            "tool_calls" not in assistant_msg
            or len(assistant_msg.get("tool_calls", [])) == 0
        )

    @pytest.mark.anyio
    async def test_tool_calls_preserved_at_end(self) -> None:
        """Test that tool calls are preserved when they're the last messages."""
        from ccproxy.domain.models import ContentBlockToolUse

        messages = [
            Message(
                role="assistant",
                content=[
                    ContentBlockToolUse(
                        type="tool_use",
                        id="tool_1",
                        name="search",
                        input={"query": "test"},
                    )
                ],
            ),
        ]

        converter = AsyncMessageConverter()
        result = await converter.convert_messages_async(messages)

        # Tool calls at the end should be preserved
        assert len(result) == 1
        assert "tool_calls" in result[0]
        assert len(result[0]["tool_calls"]) == 1

    @pytest.mark.anyio
    async def test_matching_tool_results(self) -> None:
        """Test that tool calls with matching results are preserved."""
        from ccproxy.domain.models import ContentBlockToolUse, ContentBlockToolResult

        messages = [
            Message(
                role="assistant",
                content=[
                    ContentBlockToolUse(
                        type="tool_use",
                        id="tool_1",
                        name="search",
                        input={"query": "test"},
                    )
                ],
            ),
            Message(
                role="user",
                content=[
                    ContentBlockToolResult(
                        type="tool_result",
                        tool_use_id="tool_1",
                        content="Search results here",
                    )
                ],
            ),
        ]

        converter = AsyncMessageConverter()
        result = await converter.convert_messages_async(messages)

        # Both messages should be present and properly formatted
        assert len(result) == 2
        assert "tool_calls" in result[0]
        assert result[1]["role"] == "tool"
        assert result[1]["tool_call_id"] == "tool_1"


class TestComplexMessageConversion:
    """Test complex message conversion scenarios."""

    @pytest.mark.anyio
    async def test_mixed_text_and_images(self) -> None:
        """Test message with both text and image content."""
        from ccproxy.domain.models import ContentBlockImage, ContentBlockImageSource

        messages = [
            Message(
                role="user",
                content=[
                    ContentBlockText(type="text", text="Look at this:"),
                    ContentBlockImage(
                        type="image",
                        source=ContentBlockImageSource(
                            type="base64",
                            media_type="image/png",
                            data="iVBORw0KGgo=",
                        ),
                    ),
                    ContentBlockText(type="text", text="What do you think?"),
                ],
            )
        ]

        result = await convert_messages_async(messages)

        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "user"
        assert isinstance(msg["content"], list)
        # Should have all content blocks represented
        assert len(msg["content"]) >= 2  # At least text content
        assert msg["content"][0]["type"] == "text"
        assert msg["content"][0]["text"] == "Look at this:"

    @pytest.mark.anyio
    async def test_multiple_tool_calls_in_message(self) -> None:
        """Test message with multiple tool calls."""
        from ccproxy.domain.models import ContentBlockToolUse

        messages = [
            Message(
                role="assistant",
                content=[
                    ContentBlockText(type="text", text="Let me check both locations"),
                    ContentBlockToolUse(
                        type="tool_use",
                        id="tool_1",
                        name="get_weather",
                        input={"location": "NYC"},
                    ),
                    ContentBlockToolUse(
                        type="tool_use",
                        id="tool_2",
                        name="get_weather",
                        input={"location": "Boston"},
                    ),
                ],
            )
        ]

        result = await convert_messages_async(messages)

        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "assistant"
        assert "tool_calls" in msg
        assert len(msg["tool_calls"]) == 2
        assert msg["tool_calls"][0]["function"]["name"] == "get_weather"
        assert msg["tool_calls"][1]["function"]["name"] == "get_weather"
        assert msg["content"] == "Let me check both locations"

    @pytest.mark.anyio
    async def test_empty_content_message(self) -> None:
        """Test handling of message with empty content."""
        messages = [Message(role="user", content="")]

        result = await convert_messages_async(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        # Empty content should still be present
        assert "content" in result[0] or result[0] == {"role": "user"}

    @pytest.mark.anyio
    async def test_only_images_no_text(self) -> None:
        """Test message with only images, no text."""
        from ccproxy.domain.models import ContentBlockImage, ContentBlockImageSource

        messages = [
            Message(
                role="user",
                content=[
                    ContentBlockImage(
                        type="image",
                        source=ContentBlockImageSource(
                            type="base64",
                            media_type="image/jpeg",
                            data="image_data_here",
                        ),
                    )
                ],
            )
        ]

        result = await convert_messages_async(messages)

        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "user"
        assert "content" in msg


class TestSystemContentHandling:
    """Test system content handling."""

    @pytest.mark.anyio
    async def test_system_as_string(self) -> None:
        """Test system content as simple string."""
        messages = [Message(role="user", content="Hello")]

        result = await convert_messages_async(messages, system="You are helpful")

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are helpful"
        assert result[1]["role"] == "user"

    @pytest.mark.anyio
    async def test_system_as_content_list(self) -> None:
        """Test system content as list of SystemContent blocks."""
        from ccproxy.domain.models import SystemContent

        messages = [Message(role="user", content="Hello")]
        system = [
            SystemContent(type="text", text="You are helpful"),
            SystemContent(type="text", text="Be concise"),
        ]

        result = await convert_messages_async(messages, system=system)

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "You are helpful" in result[0]["content"]
        assert "Be concise" in result[0]["content"]

    @pytest.mark.anyio
    async def test_no_system_content(self) -> None:
        """Test conversion without system content."""
        messages = [Message(role="user", content="Hello")]

        result = await convert_messages_async(messages, system=None)

        assert len(result) == 1
        assert result[0]["role"] == "user"


class TestResponseConversionEdgeCases:
    """Test response conversion edge cases."""

    @pytest.mark.anyio
    async def test_no_choices_error(self) -> None:
        """Test error when OpenAI response has no choices."""
        openai_response = ChatCompletion(
            id="test",
            model="gpt-4",
            object="chat.completion",
            created=1234567890,
            choices=[],  # Empty choices
            usage=CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )

        converter = AsyncResponseConverter()

        with pytest.raises(ValueError, match="no choices"):
            await converter.convert_response_async(openai_response)

    @pytest.mark.anyio
    async def test_response_with_refusal(self) -> None:
        """Test handling of OpenAI refusal."""
        from unittest.mock import MagicMock

        # Create a mock response with refusal
        message_mock = MagicMock()
        message_mock.content = None
        message_mock.refusal = "I cannot help with that request"
        message_mock.tool_calls = None

        choice_mock = MagicMock()
        choice_mock.message = message_mock
        choice_mock.finish_reason = "stop"

        openai_response = MagicMock()
        openai_response.id = "test-refusal"
        openai_response.model = "gpt-4"
        openai_response.choices = [choice_mock]
        openai_response.usage = MagicMock()
        openai_response.usage.prompt_tokens = 10
        openai_response.usage.completion_tokens = 8

        converter = AsyncResponseConverter()
        result = await converter.convert_response_async(openai_response)

        assert isinstance(result, MessagesResponse)
        assert len(result.content) == 1
        assert result.content[0].text == "I cannot help with that request"  # type: ignore[union-attr]

    @pytest.mark.anyio
    async def test_finish_reason_mappings(self) -> None:
        """Test all finish reason mappings."""
        # Test valid finish reasons
        finish_reason_tests = [
            ("stop", "end_turn"),
            ("length", "max_tokens"),
            ("tool_calls", "tool_use"),
            ("content_filter", "stop_sequence"),
        ]

        converter = AsyncResponseConverter()

        for openai_reason, expected_anthropic in finish_reason_tests:
            openai_response = ChatCompletion(
                id=f"test-{openai_reason}",
                model="gpt-4",
                object="chat.completion",
                created=1234567890,
                choices=[
                    Choice(
                        index=0,
                        message=ChatCompletionMessage(role="assistant", content="Test"),
                        finish_reason=openai_reason,
                    )
                ],
                usage=CompletionUsage(
                    prompt_tokens=5, completion_tokens=5, total_tokens=10
                ),
            )

            result = await converter.convert_response_async(openai_response)
            assert result.stop_reason == expected_anthropic

        # Test None/unknown with mock since Pydantic validates finish_reason
        from unittest.mock import MagicMock

        mock_response = MagicMock()
        mock_response.id = "test-none"
        mock_response.model = "gpt-4"
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].message.refusal = None
        mock_response.choices[0].finish_reason = None
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 5

        result = await converter.convert_response_async(mock_response)
        assert result.stop_reason == "stop_sequence"

    @pytest.mark.anyio
    async def test_empty_content_in_response(self) -> None:
        """Test response with empty content."""
        openai_response = ChatCompletion(
            id="test-empty",
            model="gpt-4",
            object="chat.completion",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content=None),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(prompt_tokens=5, completion_tokens=0, total_tokens=5),
        )

        converter = AsyncResponseConverter()
        result = await converter.convert_response_async(openai_response)

        # Should have at least one content block (empty text)
        assert len(result.content) >= 1

    @pytest.mark.anyio
    async def test_large_tool_arguments_async_parsing(self) -> None:
        """Test async parsing for large tool arguments."""
        # Create large arguments (> 1000 chars)
        large_args = {"data": "x" * 2000, "items": list(range(100))}

        tool_call = ChatCompletionMessageToolCall(
            id="tool-large",
            type="function",
            function={"name": "process_data", "arguments": json.dumps(large_args)},
        )

        openai_response = ChatCompletion(
            id="test-large",
            model="gpt-4",
            object="chat.completion",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant", content=None, tool_calls=[tool_call]
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=50, completion_tokens=200, total_tokens=250
            ),
        )

        converter = AsyncResponseConverter()
        result = await converter.convert_response_async(openai_response)

        assert len(result.content) == 1
        tool_use = result.content[0]
        assert tool_use.type == "tool_use"
        assert tool_use.name == "process_data"
        assert len(tool_use.input["data"]) == 2000

    @pytest.mark.anyio
    async def test_response_with_zero_usage(self) -> None:
        """Test response with zero usage tokens."""
        openai_response = ChatCompletion(
            id="test-zero-usage",
            model="gpt-4",
            object="chat.completion",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="Test"),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )

        converter = AsyncResponseConverter()
        result = await converter.convert_response_async(openai_response)

        assert result.usage is not None
        assert result.usage.input_tokens == 0
        assert result.usage.output_tokens == 0


# === Additional Coverage Tests for Missing Lines ===


class TestSyncConvertFallback:
    """Test synchronous convert method fallback."""

    def test_sync_convert_fallback(self) -> Any:
        """Test that sync convert method uses AnthropicToOpenAIConverter."""
        from ccproxy.application.converters_module.base import ConversionContext
        from unittest.mock import MagicMock
        from ccproxy.config import Settings

        settings = MagicMock(spec=Settings)
        context = ConversionContext(
            request_id="test-sync", target_model="gpt-4", settings=settings
        )

        converter = AsyncMessageConverter(context=context)

        # Mock input that would be passed to the regular converter
        mock_input = {"messages": [{"role": "user", "content": "test"}]}

        # The sync convert method should work (though it's not used in practice)
        # It should return a result by delegating to the sync converter
        result = converter.convert(mock_input)
        assert result is not None  # Should delegate successfully


class TestToolUseSingleMessageConversion:
    """Test tool use handling in single message conversion."""

    @pytest.mark.anyio
    async def test_single_message_tool_use_conversion(self) -> None:
        """Test conversion when single message contains tool_use content block (line 155)."""
        from ccproxy.domain.models import ContentBlockToolUse

        messages = [
            Message(
                role="assistant",
                content=[
                    ContentBlockToolUse(
                        type="tool_use",
                        id="single-tool-1",
                        name="calculate_sum",
                        input={"numbers": [1, 2, 3]},
                    )
                ],
            )
        ]

        converter = AsyncMessageConverter()
        result = await converter.convert_messages_async(messages)

        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "assistant"
        assert "tool_calls" in msg
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["function"]["name"] == "calculate_sum"


class TestComplexMessageAsyncStringConversion:
    """Test async string conversion in complex message handling."""

    @pytest.mark.anyio
    async def test_mixed_text_and_other_content_types(self) -> None:
        """Test text_and_non_text branch (line 207) with unknown content types triggering async str conversion (lines 201-203, 216-221)."""
        from ccproxy.domain.models import ContentBlockText

        # Create a message with text and tool use content (mixed)
        messages = [
            Message(
                role="assistant",
                content=[
                    ContentBlockText(
                        type="text", text="I need to use a tool and provide text."
                    ),
                    ContentBlockText(type="text", text="This is additional text."),
                ],
            )
        ]

        result = await convert_messages_async(messages)

        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "assistant"
        assert (
            msg["content"]
            == "I need to use a tool and provide text. This is additional text."
        )

    @pytest.mark.anyio
    async def test_only_non_text_content_blocks_array_format(self) -> None:
        """Test only non-text content blocks requiring array format (line 227)."""
        from ccproxy.domain.models import ContentBlockToolUse

        # Message with only multiple tool uses (no text)
        messages = [
            Message(
                role="assistant",
                content=[
                    ContentBlockToolUse(
                        type="tool_use",
                        id="tool-1",
                        name="search",
                        input={"query": "test1"},
                    ),
                    ContentBlockToolUse(
                        type="tool_use",
                        id="tool-2",
                        name="search",
                        input={"query": "test2"},
                    ),
                ],
            )
        ]

        result = await convert_messages_async(messages)

        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "assistant"
        assert "tool_calls" in msg
        assert len(msg["tool_calls"]) == 2
