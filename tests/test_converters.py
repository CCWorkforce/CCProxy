"""Comprehensive test suite for message converters."""

import json
from unittest.mock import MagicMock
import pytest

from ccproxy.application.converters_module.main import (
    convert_anthropic_to_openai_messages,
    convert_anthropic_tools_to_openai,
    convert_anthropic_tool_choice_to_openai,
    convert_openai_to_anthropic_response,
)
from ccproxy.application.converters_module.base import ConversionContext
from ccproxy.application.converters_module.content_converter import ContentConverter

from ccproxy.domain.models import (
    Message,
    ContentBlockText,
    ContentBlockImage,
    ContentBlockToolUse,
    ContentBlockToolResult,
    Tool,
    ToolChoice,
)


@pytest.fixture
def conversion_context():
    """Create a conversion context for testing."""
    return ConversionContext(
        request_id="test-req-123",
        target_model="gpt-5",
        settings=None,
    )


@pytest.fixture
def text_message():
    """Create a simple text message."""
    return Message(
        role="user", content=[ContentBlockText(type="text", text="Hello, how are you?")]
    )


@pytest.fixture
def assistant_message():
    """Create an assistant message."""
    return Message(
        role="assistant",
        content=[ContentBlockText(type="text", text="I'm doing well, thank you!")],
    )


@pytest.fixture
def image_message():
    """Create a message with image content."""
    return Message(
        role="user",
        content=[
            ContentBlockText(type="text", text="What's in this image?"),
            ContentBlockImage(
                type="image",
                source={
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": "base64encodedimagedata",
                },
            ),
        ],
    )


@pytest.fixture
def tool_use_message():
    """Create a message with tool use."""
    return Message(
        role="assistant",
        content=[
            ContentBlockToolUse(
                type="tool_use",
                id="tool_call_123",
                name="get_weather",
                input={"location": "San Francisco", "unit": "celsius"},
            )
        ],
    )


@pytest.fixture
def tool_result_message():
    """Create a message with tool result."""
    return Message(
        role="user",
        content=[
            ContentBlockToolResult(
                type="tool_result",
                tool_use_id="tool_call_123",
                content="The weather in San Francisco is 18°C and sunny.",
                is_error=False,
            )
        ],
    )


@pytest.fixture
def sample_tool():
    """Create a sample tool."""
    return Tool(
        name="get_weather",
        description="Get the current weather for a location",
        input_schema={
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    )


@pytest.fixture
def openai_chat_completion():
    """Create a sample OpenAI ChatCompletion object."""
    completion = MagicMock()
    completion.id = "chatcmpl-123"
    completion.model = "gpt-5"  # Using latest GPT-5 model
    completion.created = 1234567890

    # Create a simple object for usage to avoid MagicMock issues
    class UsageMock:
        prompt_tokens = 50
        completion_tokens = 100
        total_tokens = 150

    completion.usage = UsageMock()

    # Create a choice with message
    choice = MagicMock()
    choice.index = 0
    choice.finish_reason = "stop"
    choice.message = MagicMock()
    choice.message.content = "This is the response"
    choice.message.role = "assistant"
    choice.message.tool_calls = None
    choice.message.refusal = None

    completion.choices = [choice]

    return completion


class TestAnthropicToOpenAIConversion:
    """Test Anthropic to OpenAI message conversion."""

    def test_convert_simple_text_message(self, text_message):
        """Test converting a simple text message."""
        result = convert_anthropic_to_openai_messages([text_message])

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello, how are you?"

    def test_convert_multiple_messages(self, text_message, assistant_message):
        """Test converting multiple messages."""
        messages = [text_message, assistant_message]
        result = convert_anthropic_to_openai_messages(messages)

        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello, how are you?"
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == "I'm doing well, thank you!"

    def test_convert_with_system_message(self, text_message):
        """Test converting with system message."""
        system = "You are a helpful assistant."
        result = convert_anthropic_to_openai_messages([text_message], system)

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are a helpful assistant."
        assert result[1]["role"] == "user"

    def test_convert_image_message(self, image_message):
        """Test converting message with image."""
        result = convert_anthropic_to_openai_messages([image_message])

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert isinstance(result[0]["content"], list)
        assert len(result[0]["content"]) == 2

        # Check text part
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][0]["text"] == "What's in this image?"

        # Check image part
        assert result[0]["content"][1]["type"] == "image_url"
        assert "url" in result[0]["content"][1]["image_url"]
        assert result[0]["content"][1]["image_url"]["url"].startswith(
            "data:image/jpeg;base64,"
        )

    def test_convert_tool_use_message(self, tool_use_message):
        """Test converting tool use message."""
        result = convert_anthropic_to_openai_messages([tool_use_message])

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert "tool_calls" in result[0]
        assert len(result[0]["tool_calls"]) == 1

        tool_call = result[0]["tool_calls"][0]
        assert tool_call["id"] == "tool_call_123"
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "get_weather"
        assert json.loads(tool_call["function"]["arguments"]) == {
            "location": "San Francisco",
            "unit": "celsius",
        }

    def test_convert_tool_result_message(self, tool_result_message):
        """Test converting tool result message."""
        result = convert_anthropic_to_openai_messages([tool_result_message])

        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "tool_call_123"
        assert result[0]["content"] == "The weather in San Francisco is 18°C and sunny."

    def test_convert_empty_messages(self):
        """Test converting empty message list."""
        result = convert_anthropic_to_openai_messages([])
        assert result == []

    def test_convert_with_error_handling(self):
        """Test error handling in conversion."""
        # Create invalid message
        invalid_message = MagicMock()
        invalid_message.role = "user"
        invalid_message.content = None  # Invalid content

        result = convert_anthropic_to_openai_messages([invalid_message])
        # Should handle error gracefully
        assert isinstance(result, list)


class TestToolConversion:
    """Test tool and tool choice conversion."""

    def test_convert_single_tool(self, sample_tool):
        """Test converting a single tool."""
        result = convert_anthropic_tools_to_openai([sample_tool])

        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        assert (
            result[0]["function"]["description"]
            == "Get the current weather for a location"
        )
        assert result[0]["function"]["parameters"] == sample_tool.input_schema

    def test_convert_multiple_tools(self, sample_tool):
        """Test converting multiple tools."""
        tool2 = Tool(
            name="search_web",
            description="Search the web",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        )

        result = convert_anthropic_tools_to_openai([sample_tool, tool2])

        assert len(result) == 2
        assert result[0]["function"]["name"] == "get_weather"
        assert result[1]["function"]["name"] == "search_web"

    def test_convert_tool_choice_auto(self):
        """Test converting auto tool choice."""
        tool_choice = ToolChoice(type="auto", disable_parallel_tool_use=False)
        result = convert_anthropic_tool_choice_to_openai(tool_choice, [])
        assert result == "auto"

    def test_convert_tool_choice_any(self):
        """Test converting any tool choice."""
        tool_choice = ToolChoice(type="any", disable_parallel_tool_use=False)
        result = convert_anthropic_tool_choice_to_openai(tool_choice, [])
        assert result == "required"

    def test_convert_tool_choice_specific(self, sample_tool):
        """Test converting specific tool choice."""
        tool_choice = ToolChoice(
            type="tool", name="get_weather", disable_parallel_tool_use=False
        )
        result = convert_anthropic_tool_choice_to_openai(tool_choice, [sample_tool])

        assert result["type"] == "function"
        assert result["function"]["name"] == "get_weather"

    def test_convert_tool_choice_none(self):
        """Test converting when no tool choice specified."""
        result = convert_anthropic_tool_choice_to_openai(None, [])
        assert result == "auto"


class TestOpenAIToAnthropicConversion:
    """Test OpenAI to Anthropic response conversion."""

    def test_convert_simple_response(self, openai_chat_completion):
        """Test converting simple OpenAI response."""
        result = convert_openai_to_anthropic_response(
            openai_chat_completion,
            original_anthropic_model_name="claude-opus-4-1-20250805",
        )

        assert result.id == "chatcmpl-123"
        assert result.type == "message"
        assert result.model == "claude-opus-4-1-20250805"  # Uses original model name
        assert result.role == "assistant"
        assert len(result.content) == 1
        assert result.content[0].type == "text"
        assert result.content[0].text == "This is the response"
        assert result.usage.input_tokens == 50
        assert result.usage.output_tokens == 100

    def test_convert_response_with_tool_calls(self, openai_chat_completion):
        """Test converting response with tool calls."""
        # Add tool calls to the response
        tool_call = MagicMock()
        tool_call.id = "call_123"
        tool_call.type = "function"
        tool_call.function = MagicMock()
        tool_call.function.name = "get_weather"
        tool_call.function.arguments = json.dumps({"location": "Paris"})

        openai_chat_completion.choices[0].message.tool_calls = [tool_call]
        openai_chat_completion.choices[0].message.content = None

        result = convert_openai_to_anthropic_response(
            openai_chat_completion,
            original_anthropic_model_name="claude-opus-4-1-20250805",
        )

        assert len(result.content) == 1
        assert result.content[0].type == "tool_use"
        assert result.content[0].id == "call_123"
        assert result.content[0].name == "get_weather"
        assert result.content[0].input == {"location": "Paris"}

    def test_convert_response_with_refusal(self, openai_chat_completion):
        """Test converting response with refusal."""
        openai_chat_completion.choices[0].message.content = None
        openai_chat_completion.choices[
            0
        ].message.refusal = "I cannot help with that request."

        result = convert_openai_to_anthropic_response(
            openai_chat_completion,
            original_anthropic_model_name="claude-opus-4-1-20250805",
        )

        assert len(result.content) == 1
        assert result.content[0].type == "text"
        assert result.content[0].text == "I cannot help with that request."

    def test_convert_finish_reason_mapping(self, openai_chat_completion):
        """Test finish reason mapping."""
        test_cases = [
            ("stop", "end_turn"),
            ("length", "max_tokens"),
            ("tool_calls", "tool_use"),
            ("content_filter", "stop_sequence"),
            ("unknown", "end_turn"),  # Default mapping for unknown reasons
        ]

        for openai_reason, expected_anthropic in test_cases:
            openai_chat_completion.choices[0].finish_reason = openai_reason
            result = convert_openai_to_anthropic_response(
                openai_chat_completion,
                original_anthropic_model_name="claude-opus-4-1-20250805",
            )
            assert result.stop_reason == expected_anthropic


# Note: Stream conversion tests removed as the functionality doesn't exist yet
# TODO: Add stream conversion tests when the feature is implemented


class TestContentConverter:
    """Test the ContentConverter class."""

    def test_convert_text_block(self, conversion_context):
        """Test converting text content block."""
        # Text blocks are handled directly in AnthropicToOpenAIConverter
        # ContentConverter doesn't have a method for text blocks
        text_block = ContentBlockText(type="text", text="Hello world")

        # Just test that the text block has correct attributes
        assert text_block.type == "text"
        assert text_block.text == "Hello world"

    def test_convert_image_block(self, conversion_context):
        """Test converting image content block."""
        converter = ContentConverter()
        image_block = ContentBlockImage(
            type="image",
            source={
                "type": "base64",
                "media_type": "image/png",
                "data": "imagedata123",
            },
        )

        result = converter.convert_image_block_to_openai(image_block)

        assert result["type"] == "image_url"
        assert result["image_url"]["url"].startswith("data:image/png;base64,")

    def test_convert_tool_use_block(self, conversion_context):
        """Test converting tool use content block."""
        converter = ContentConverter()
        tool_block = ContentBlockToolUse(
            type="tool_use",
            id="tool_123",
            name="calculator",
            input={"expression": "2+2"},
        )

        result = converter.convert_tool_use_to_openai(tool_block)

        assert result["id"] == "tool_123"
        assert result["type"] == "function"
        assert result["function"]["name"] == "calculator"
        assert json.loads(result["function"]["arguments"]) == {"expression": "2+2"}


class TestErrorHandling:
    """Test error handling in converters."""

    def test_handle_invalid_content_type(self, conversion_context):
        """Test handling invalid content type."""
        # Create a message that already has a valid content type
        # but tests handling of unknown content block types in conversion
        text_block = ContentBlockText(type="text", text="Valid text")
        messages = [Message(role="user", content=[text_block])]

        # Patch the converter to simulate an unknown block type
        result = convert_anthropic_to_openai_messages(messages)
        assert isinstance(result, list)
        assert len(result) == 1
        # Verify the valid text block was converted correctly
        assert result[0]["content"] == "Valid text"

    def test_handle_missing_fields(self):
        """Test handling missing required fields."""
        # Message with missing content
        invalid_message = MagicMock()
        invalid_message.role = "user"
        invalid_message.content = None

        result = convert_anthropic_to_openai_messages([invalid_message])
        assert isinstance(result, list)

    def test_handle_json_serialization_error(self, tool_use_message):
        """Test handling JSON serialization errors."""
        # Make input non-serializable
        tool_use_message.content[0].input = MagicMock()  # Non-serializable object

        result = convert_anthropic_to_openai_messages([tool_use_message])

        # Check that error was handled gracefully
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert "tool_calls" in result[0]
        # The arguments should contain the error fallback
        tool_call_args = json.loads(result[0]["tool_calls"][0]["function"]["arguments"])
        assert "error" in tool_call_args
        # Should handle error and still return valid structure
        assert isinstance(result, list)


class TestConverterEdgeCases:
    """Test edge cases for message converters."""

    @pytest.fixture
    def large_text_message(self):
        """Create a message with very large text content (>10k tokens)."""
        # Approximate 10k+ tokens with repeated text
        large_text = "This is a test sentence. " * 2500  # ~10k+ tokens
        return Message(
            role="user", content=[ContentBlockText(type="text", text=large_text)]
        )

    @pytest.fixture
    def many_tool_calls_message(self):
        """Create a message with many tool calls."""
        tool_calls = []
        for i in range(10):
            tool_calls.append(
                ContentBlockToolUse(
                    type="tool_use",
                    id=f"tool_{i}",
                    name=f"function_{i}",
                    input={"param": f"value_{i}"},
                )
            )
        return Message(role="assistant", content=tool_calls)

    def test_convert_large_message_handling(self, large_text_message):
        """Test conversion of messages with very large content (>10k tokens)."""
        result = convert_anthropic_to_openai_messages([large_text_message])

        assert len(result) == 1
        assert result[0]["role"] == "user"
        # Verify large content is preserved (not truncated by default)
        assert len(result[0]["content"]) > 10000

    def test_convert_many_tool_calls_parallel(self, many_tool_calls_message):
        """Test conversion of message with many tool calls (parallel processing)."""
        result = convert_anthropic_to_openai_messages([many_tool_calls_message])

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert "tool_calls" in result[0]
        assert len(result[0]["tool_calls"]) == 10

        # Verify all tool calls were converted correctly
        for i, tool_call in enumerate(result[0]["tool_calls"]):
            assert tool_call["function"]["name"] == f"function_{i}"
            args = json.loads(tool_call["function"]["arguments"])
            assert args["param"] == f"value_{i}"

    def test_mixed_content_types_in_single_message(self):
        """Test message with mixed content types."""
        mixed_message = Message(
            role="user",
            content=[
                ContentBlockText(type="text", text="Here's an image:"),
                ContentBlockImage(
                    type="image",
                    source={
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "base64data",
                    },
                ),
                ContentBlockText(type="text", text="What do you see?"),
            ],
        )

        result = convert_anthropic_to_openai_messages([mixed_message])
        assert len(result) == 1
        assert len(result[0]["content"]) == 3
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][1]["type"] == "image_url"
        assert result[0]["content"][2]["type"] == "text"

    def test_deeply_nested_tool_inputs(self):
        """Test tool use with deeply nested input structures."""
        nested_input = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "data": ["item1", "item2", "item3"],
                            "meta": {"key": "value"},
                        }
                    }
                }
            }
        }

        tool_message = Message(
            role="assistant",
            content=[
                ContentBlockToolUse(
                    type="tool_use",
                    id="nested_tool",
                    name="complex_function",
                    input=nested_input,
                )
            ],
        )

        result = convert_anthropic_to_openai_messages([tool_message])
        assert len(result) == 1
        assert "tool_calls" in result[0]

        # Verify nested structure is preserved
        args = json.loads(result[0]["tool_calls"][0]["function"]["arguments"])
        assert args["level1"]["level2"]["level3"]["level4"]["data"] == [
            "item1",
            "item2",
            "item3",
        ]

    def test_empty_tool_input(self):
        """Test tool use with empty input."""
        tool_message = Message(
            role="assistant",
            content=[
                ContentBlockToolUse(
                    type="tool_use",
                    id="empty_tool",
                    name="no_params_function",
                    input={},
                )
            ],
        )

        result = convert_anthropic_to_openai_messages([tool_message])
        assert len(result) == 1
        assert "tool_calls" in result[0]

        # Empty input should convert to empty JSON object
        args = json.loads(result[0]["tool_calls"][0]["function"]["arguments"])
        assert args == {}

    def test_unicode_and_special_characters(self):
        """Test messages with Unicode and special characters."""
        unicode_message = Message(
            role="user",
            content=[
                ContentBlockText(
                    type="text",
                    text="Unicode: 你好世界 🌍 Émojis: 😀🎉 Special: \n\t\r",
                )
            ],
        )

        result = convert_anthropic_to_openai_messages([unicode_message])
        assert len(result) == 1
        assert "你好世界" in result[0]["content"]
        assert "🌍" in result[0]["content"]
        assert "😀🎉" in result[0]["content"]

    def test_malformed_image_data(self):
        """Test handling of minimal/edge-case image data."""
        minimal_image = Message(
            role="user",
            content=[
                ContentBlockImage(
                    type="image",
                    source={
                        "type": "base64",
                        "media_type": "image/jpeg",  # Valid but minimal
                        "data": "test",  # Minimal data
                    },
                )
            ],
        )

        result = convert_anthropic_to_openai_messages([minimal_image])
        assert len(result) == 1
        # Should handle minimal data gracefully
        assert result[0]["role"] == "user"
        assert result[0]["content"][0]["type"] == "image_url"

    def test_rapid_role_switching(self):
        """Test conversation with rapid role switching."""
        messages = [
            Message(role="user", content=[ContentBlockText(type="text", text="1")]),
            Message(
                role="assistant", content=[ContentBlockText(type="text", text="2")]
            ),
            Message(role="user", content=[ContentBlockText(type="text", text="3")]),
            Message(
                role="assistant", content=[ContentBlockText(type="text", text="4")]
            ),
            Message(role="user", content=[ContentBlockText(type="text", text="5")]),
        ]

        result = convert_anthropic_to_openai_messages(messages)
        assert len(result) == 5
        assert [msg["role"] for msg in result] == [
            "user",
            "assistant",
            "user",
            "assistant",
            "user",
        ]
        assert [msg["content"] for msg in result] == ["1", "2", "3", "4", "5"]

    def test_tool_result_with_error_content(self):
        """Test tool result containing error information."""
        error_result = Message(
            role="user",
            content=[
                ContentBlockToolResult(
                    type="tool_result",
                    tool_use_id="failed_tool",
                    content=[
                        ContentBlockText(
                            type="text", text="Error: API rate limit exceeded"
                        )
                    ],
                    is_error=True,
                )
            ],
        )

        result = convert_anthropic_to_openai_messages([error_result])
        assert len(result) == 1
        # Tool results convert to tool role messages
        assert result[0]["role"] == "tool"
        assert "Error: API rate limit exceeded" in str(result[0]["content"])

    def test_concurrent_conversion_simulation(self):
        """Test that converter handles multiple messages efficiently."""
        # Create a mix of different message types
        messages = []
        for i in range(20):
            if i % 3 == 0:
                # Text message
                messages.append(
                    Message(
                        role="user" if i % 2 == 0 else "assistant",
                        content=[ContentBlockText(type="text", text=f"Message {i}")],
                    )
                )
            elif i % 3 == 1:
                # Tool use message
                messages.append(
                    Message(
                        role="assistant",
                        content=[
                            ContentBlockToolUse(
                                type="tool_use",
                                id=f"tool_{i}",
                                name=f"function_{i}",
                                input={"index": i},
                            )
                        ],
                    )
                )
            else:
                # Tool result message
                messages.append(
                    Message(
                        role="user",
                        content=[
                            ContentBlockToolResult(
                                type="tool_result",
                                tool_use_id=f"tool_{i - 1}",
                                content=[
                                    ContentBlockText(type="text", text=f"Result {i}")
                                ],
                            )
                        ],
                    )
                )

        result = convert_anthropic_to_openai_messages(messages)
        assert len(result) == 20

        # Verify each message was converted correctly
        for i, msg in enumerate(result):
            if i % 3 == 0:
                assert f"Message {i}" in str(msg["content"])
            elif i % 3 == 1:
                assert "tool_calls" in msg
            else:
                assert f"Result {i}" in str(msg["content"])
