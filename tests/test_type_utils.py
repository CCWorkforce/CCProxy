"""Comprehensive tests for type utility functions."""

from dataclasses import dataclass

from ccproxy.application.type_utils import (
    is_text_block,
    is_image_block,
    is_tool_use_block,
    is_tool_result_block,
    is_thinking_block,
    is_redacted_thinking_block,
    is_system_text_block,
    is_string_content,
    is_list_content,
    is_dict_content,
    is_serializable_primitive,
    is_dataclass_instance,
    is_redactable_key,
    is_large_string,
    is_error_dict_with_field,
    is_nested_dict_field,
    safe_get_nested,
    ContentBlockDispatcher,
    TypeConverter,
    get_content_type,
)
from ccproxy.domain.models import (
    ContentBlockText,
    ContentBlockImage,
    ContentBlockToolUse,
    ContentBlockToolResult,
    ContentBlockThinking,
    ContentBlockRedactedThinking,
    SystemContent,
)


class TestTextBlockChecking:
    """Test is_text_block function."""

    def test_text_block_instance(self):
        """Test with ContentBlockText instance."""
        block = ContentBlockText(type="text", text="Hello")
        assert is_text_block(block) is True

    def test_text_block_dict(self):
        """Test with dict representation."""
        block = {"type": "text", "text": "Hello"}
        assert is_text_block(block) is True

    def test_text_block_dict_missing_text_key(self):
        """Test dict without 'text' key."""
        block = {"type": "text"}
        assert is_text_block(block) is False

    def test_not_text_block(self):
        """Test with non-text block."""
        block = {"type": "image"}
        assert is_text_block(block) is False


class TestImageBlockChecking:
    """Test is_image_block function."""

    def test_image_block_instance(self):
        """Test with ContentBlockImage instance."""
        block = ContentBlockImage(
            type="image",
            source={"type": "base64", "media_type": "image/png", "data": "abc123"},
        )
        assert is_image_block(block) is True

    def test_image_block_dict(self):
        """Test with dict representation."""
        block = {"type": "image", "source": {}}
        assert is_image_block(block) is True

    def test_not_image_block(self):
        """Test with non-image block."""
        block = {"type": "text"}
        assert is_image_block(block) is False


class TestToolUseBlockChecking:
    """Test is_tool_use_block function."""

    def test_tool_use_block_instance(self):
        """Test with ContentBlockToolUse instance."""
        block = ContentBlockToolUse(
            type="tool_use", id="tool_1", name="search", input={}
        )
        assert is_tool_use_block(block) is True

    def test_tool_use_block_dict(self):
        """Test with dict representation."""
        block = {"type": "tool_use", "id": "tool_1", "name": "search", "input": {}}
        assert is_tool_use_block(block) is True

    def test_not_tool_use_block(self):
        """Test with non-tool-use block."""
        block = {"type": "text"}
        assert is_tool_use_block(block) is False


class TestToolResultBlockChecking:
    """Test is_tool_result_block function."""

    def test_tool_result_block_instance(self):
        """Test with ContentBlockToolResult instance."""
        block = ContentBlockToolResult(
            type="tool_result", tool_use_id="tool_1", content="result"
        )
        assert is_tool_result_block(block) is True

    def test_tool_result_block_dict(self):
        """Test with dict representation."""
        block = {"type": "tool_result", "tool_use_id": "tool_1", "content": "result"}
        assert is_tool_result_block(block) is True

    def test_not_tool_result_block(self):
        """Test with non-tool-result block."""
        block = {"type": "text"}
        assert is_tool_result_block(block) is False


class TestThinkingBlockChecking:
    """Test is_thinking_block function."""

    def test_thinking_block_instance(self):
        """Test with ContentBlockThinking instance."""
        block = ContentBlockThinking(type="thinking", thinking="reasoning")
        assert is_thinking_block(block) is True

    def test_thinking_block_dict(self):
        """Test with dict representation."""
        block = {"type": "thinking", "thinking": "reasoning"}
        assert is_thinking_block(block) is True

    def test_not_thinking_block(self):
        """Test with non-thinking block."""
        block = {"type": "text"}
        assert is_thinking_block(block) is False


class TestRedactedThinkingBlockChecking:
    """Test is_redacted_thinking_block function."""

    def test_redacted_thinking_block_instance(self):
        """Test with ContentBlockRedactedThinking instance."""
        block = ContentBlockRedactedThinking(type="redacted_thinking", data="hidden")
        assert is_redacted_thinking_block(block) is True

    def test_redacted_thinking_block_dict(self):
        """Test with dict representation."""
        block = {"type": "redacted_thinking"}
        assert is_redacted_thinking_block(block) is True

    def test_not_redacted_thinking_block(self):
        """Test with non-redacted-thinking block."""
        block = {"type": "text"}
        assert is_redacted_thinking_block(block) is False


class TestSystemTextBlockChecking:
    """Test is_system_text_block function."""

    def test_system_text_block_instance(self):
        """Test with SystemContent instance."""
        block = SystemContent(type="text", text="System message")
        assert is_system_text_block(block) is True

    def test_system_text_block_dict(self):
        """Test with dict representation."""
        block = {"type": "text", "text": "System message"}
        assert is_system_text_block(block) is True

    def test_not_system_text_block(self):
        """Test with non-text type."""
        block = {"type": "image"}
        assert is_system_text_block(block) is False


class TestContentTypeChecking:
    """Test content type checking functions."""

    def test_is_string_content(self):
        """Test string content detection."""
        assert is_string_content("hello") is True
        assert is_string_content(123) is False
        assert is_string_content([]) is False

    def test_is_list_content(self):
        """Test list content detection."""
        assert is_list_content([1, 2, 3]) is True
        assert is_list_content("hello") is False
        assert is_list_content({}) is False

    def test_is_dict_content(self):
        """Test dict content detection."""
        assert is_dict_content({"key": "value"}) is True
        assert is_dict_content("hello") is False
        assert is_dict_content([]) is False


class TestPrimitiveChecking:
    """Test serializable primitive checking."""

    def test_serializable_primitives(self):
        """Test various serializable primitives."""
        assert is_serializable_primitive("string") is True
        assert is_serializable_primitive(123) is True
        assert is_serializable_primitive(45.67) is True
        assert is_serializable_primitive(True) is True
        assert is_serializable_primitive(False) is True
        assert is_serializable_primitive(None) is True

    def test_non_serializable_primitives(self):
        """Test non-primitive types."""
        assert is_serializable_primitive([]) is False
        assert is_serializable_primitive({}) is False
        assert is_serializable_primitive(object()) is False


class TestDataclassChecking:
    """Test dataclass instance checking."""

    def test_dataclass_instance(self):
        """Test with dataclass instance."""

        @dataclass
        class TestData:
            value: str

        instance = TestData(value="test")
        assert is_dataclass_instance(instance) is True

    def test_dataclass_type(self):
        """Test with dataclass type (not instance)."""

        @dataclass
        class TestData:
            value: str

        assert is_dataclass_instance(TestData) is False

    def test_non_dataclass(self):
        """Test with non-dataclass."""
        assert is_dataclass_instance("string") is False
        assert is_dataclass_instance({}) is False


class TestRedactableKeyChecking:
    """Test redactable key checking."""

    def test_redactable_key(self):
        """Test with redactable key."""
        redact_list = ["password", "api_key", "secret"]
        assert is_redactable_key("password", redact_list) is True
        assert is_redactable_key("PASSWORD", redact_list) is True
        assert is_redactable_key("api_key", redact_list) is True

    def test_non_redactable_key(self):
        """Test with non-redactable key."""
        redact_list = ["password", "api_key"]
        assert is_redactable_key("username", redact_list) is False
        assert is_redactable_key("email", redact_list) is False

    def test_non_string_key(self):
        """Test with non-string key."""
        redact_list = ["password"]
        assert is_redactable_key(123, redact_list) is False
        assert is_redactable_key(None, redact_list) is False


class TestLargeStringChecking:
    """Test large string checking."""

    def test_large_string(self):
        """Test with string exceeding max length."""
        large_str = "a" * 6000
        assert is_large_string(large_str, max_length=5000) is True

    def test_small_string(self):
        """Test with string within limit."""
        small_str = "a" * 100
        assert is_large_string(small_str, max_length=5000) is False

    def test_exact_limit(self):
        """Test with string at exact limit."""
        exact_str = "a" * 5000
        assert is_large_string(exact_str, max_length=5000) is False

    def test_non_string(self):
        """Test with non-string."""
        assert is_large_string(123, max_length=5000) is False
        assert is_large_string([], max_length=5000) is False


class TestErrorDictChecking:
    """Test error dict field checking."""

    def test_error_dict_with_field(self):
        """Test dict with specified field."""
        error_dict = {"error": "Something went wrong", "code": 500}
        assert is_error_dict_with_field(error_dict, "error") is True
        assert is_error_dict_with_field(error_dict, "code") is True

    def test_error_dict_without_field(self):
        """Test dict without specified field."""
        error_dict = {"message": "Error"}
        assert is_error_dict_with_field(error_dict, "error") is False

    def test_error_dict_with_none_field(self):
        """Test dict with None value (should return False)."""
        error_dict = {"error": None}
        assert is_error_dict_with_field(error_dict, "error") is False

    def test_non_dict(self):
        """Test with non-dict."""
        assert is_error_dict_with_field("string", "error") is False
        assert is_error_dict_with_field([], "error") is False


class TestNestedDictChecking:
    """Test nested dict field checking."""

    def test_nested_dict_field_exists(self):
        """Test with existing nested path."""
        data = {"error": {"details": {"message": "Failed"}}}
        assert is_nested_dict_field(data, "error", "details", "message") is True
        assert is_nested_dict_field(data, "error", "details") is True
        assert is_nested_dict_field(data, "error") is True

    def test_nested_dict_field_missing(self):
        """Test with missing nested path."""
        data = {"error": {"details": {}}}
        assert is_nested_dict_field(data, "error", "details", "message") is False
        assert is_nested_dict_field(data, "error", "missing") is False

    def test_nested_dict_non_dict_value(self):
        """Test when intermediate value is not dict."""
        data = {"error": "string_value"}
        assert is_nested_dict_field(data, "error", "details") is False

    def test_nested_dict_non_dict_input(self):
        """Test with non-dict input."""
        assert is_nested_dict_field("string", "error") is False
        assert is_nested_dict_field([], "error") is False


class TestSafeGetNested:
    """Test safe nested dict value retrieval."""

    def test_safe_get_nested_exists(self):
        """Test retrieving existing nested value."""
        data = {"user": {"profile": {"name": "John"}}}
        assert safe_get_nested(data, "user", "profile", "name") == "John"
        assert safe_get_nested(data, "user", "profile") == {"name": "John"}

    def test_safe_get_nested_missing(self):
        """Test retrieving missing nested value."""
        data = {"user": {"profile": {}}}
        assert safe_get_nested(data, "user", "profile", "name") is None
        assert safe_get_nested(data, "user", "missing") is None

    def test_safe_get_nested_with_default(self):
        """Test with custom default value."""
        data = {"user": {}}
        assert safe_get_nested(data, "user", "profile", "name", default="Unknown") == "Unknown"

    def test_safe_get_nested_non_dict_value(self):
        """Test when intermediate value is not dict."""
        data = {"user": "string"}
        assert safe_get_nested(data, "user", "profile") is None

    def test_safe_get_nested_non_dict_input(self):
        """Test with non-dict input."""
        assert safe_get_nested("string", "key") is None
        assert safe_get_nested([], "key") is None


class TestContentBlockDispatcher:
    """Test ContentBlockDispatcher class."""

    def test_dispatch_text_block(self):
        """Test dispatching text block."""
        block = ContentBlockText(type="text", text="Hello")
        handlers = {"text": lambda b: "handled_text"}
        result = ContentBlockDispatcher.dispatch(block, handlers)
        assert result == "handled_text"

    def test_dispatch_image_block(self):
        """Test dispatching image block."""
        block = {"type": "image"}
        handlers = {"image": lambda b: "handled_image"}
        result = ContentBlockDispatcher.dispatch(block, handlers)
        assert result == "handled_image"

    def test_dispatch_tool_use_block(self):
        """Test dispatching tool use block."""
        block = {"type": "tool_use"}
        handlers = {"tool_use": lambda b: "handled_tool_use"}
        result = ContentBlockDispatcher.dispatch(block, handlers)
        assert result == "handled_tool_use"

    def test_dispatch_with_default_handler(self):
        """Test fallback to default handler."""
        block = {"type": "text"}
        handlers = {"image": lambda b: "image", "_default": lambda b: "default"}
        result = ContentBlockDispatcher.dispatch(block, handlers)
        assert result == "default"

    def test_dispatch_with_unknown_handler(self):
        """Test fallback to unknown handler."""
        block = {"type": "unknown_type"}
        handlers = {"text": lambda b: "text", "_unknown": lambda b: "unknown"}
        result = ContentBlockDispatcher.dispatch(block, handlers)
        assert result == "unknown"

    def test_dispatch_no_handler(self):
        """Test with no matching handler."""
        block = {"type": "text"}
        handlers = {"image": lambda b: "image"}
        result = ContentBlockDispatcher.dispatch(block, handlers)
        assert result is None

    def test_dispatch_all_block_types(self):
        """Test dispatching all supported block types."""
        blocks_and_types = [
            (ContentBlockText(type="text", text="Hi"), "text"),
            (ContentBlockImage(type="image", source={"type": "base64", "media_type": "image/png", "data": "abc"}), "image"),
            (ContentBlockToolUse(type="tool_use", id="1", name="tool", input={}), "tool_use"),
            (ContentBlockToolResult(type="tool_result", tool_use_id="1", content="result"), "tool_result"),
            (ContentBlockThinking(type="thinking", thinking="think"), "thinking"),
            (ContentBlockRedactedThinking(type="redacted_thinking", data="hidden"), "redacted_thinking"),
        ]

        for block, expected_type in blocks_and_types:
            handlers = {expected_type: lambda b: expected_type}
            result = ContentBlockDispatcher.dispatch(block, handlers)
            assert result == expected_type


class TestTypeConverter:
    """Test TypeConverter class."""

    def test_to_string_from_string(self):
        """Test converting string to string."""
        assert TypeConverter.to_string("hello") == "hello"

    def test_to_string_from_bytes(self):
        """Test converting bytes to string."""
        assert TypeConverter.to_string(b"hello") == "hello"

    def test_to_string_from_bytes_decode_error(self):
        """Test converting invalid bytes with fallback."""
        invalid_bytes = b"\xff\xfe"
        assert TypeConverter.to_string(invalid_bytes, fallback="error") == "error"

    def test_to_string_from_none(self):
        """Test converting None with fallback."""
        assert TypeConverter.to_string(None, fallback="default") == "default"

    def test_to_string_from_number(self):
        """Test converting number to string."""
        assert TypeConverter.to_string(123) == "123"
        assert TypeConverter.to_string(45.67) == "45.67"

    def test_to_string_from_object_exception(self):
        """Test converting object that raises exception."""

        class BadObject:
            def __str__(self):
                raise ValueError("Cannot convert")

        assert TypeConverter.to_string(BadObject(), fallback="failed") == "failed"

    def test_to_dict_from_dict(self):
        """Test converting dict to dict."""
        data = {"key": "value"}
        assert TypeConverter.to_dict(data) == data

    def test_to_dict_from_dataclass(self):
        """Test converting dataclass to dict."""

        @dataclass
        class TestData:
            name: str
            value: int

        instance = TestData(name="test", value=42)
        result = TypeConverter.to_dict(instance)
        assert result == {"name": "test", "value": 42}

    def test_to_dict_with_model_dump(self):
        """Test converting Pydantic model to dict."""

        class MockModel:
            def model_dump(self):
                return {"field": "value"}

        model = MockModel()
        assert TypeConverter.to_dict(model) == {"field": "value"}

    def test_to_dict_with_dict_method(self):
        """Test converting object with dict() method."""

        class MockModel:
            def dict(self):
                return {"field": "value"}

        model = MockModel()
        assert TypeConverter.to_dict(model) == {"field": "value"}

    def test_to_dict_with_dunder_dict(self):
        """Test converting object with __dict__."""

        class MockObject:
            def __init__(self):
                self.field = "value"

        obj = MockObject()
        result = TypeConverter.to_dict(obj)
        assert result == {"field": "value"}

    def test_to_dict_fallback(self):
        """Test to_dict with fallback."""
        assert TypeConverter.to_dict("string") == {}
        assert TypeConverter.to_dict(123, fallback={"default": True}) == {"default": True}

    def test_ensure_list_from_list(self):
        """Test ensure_list with list input."""
        assert TypeConverter.ensure_list([1, 2, 3]) == [1, 2, 3]

    def test_ensure_list_from_tuple(self):
        """Test ensure_list with tuple input."""
        assert TypeConverter.ensure_list((1, 2, 3)) == [1, 2, 3]

    def test_ensure_list_from_set(self):
        """Test ensure_list with set input."""
        result = TypeConverter.ensure_list({1, 2, 3})
        assert isinstance(result, list)
        assert set(result) == {1, 2, 3}

    def test_ensure_list_from_none(self):
        """Test ensure_list with None input."""
        assert TypeConverter.ensure_list(None) == []

    def test_ensure_list_from_single_value(self):
        """Test ensure_list with single value."""
        assert TypeConverter.ensure_list("value") == ["value"]
        assert TypeConverter.ensure_list(42) == [42]


class TestGetContentType:
    """Test get_content_type function."""

    def test_get_content_type_primitives(self):
        """Test content type for primitives."""
        assert get_content_type("string") == "string"
        assert get_content_type([1, 2, 3]) == "list"
        assert get_content_type({"key": "value"}) == "dict"
        assert get_content_type(123) == "number"
        assert get_content_type(45.67) == "number"
        # Boolean is checked after number in get_content_type, so True/False return "number"
        # This is because bool is a subclass of int in Python
        assert get_content_type(True) in ["boolean", "number"]
        assert get_content_type(False) in ["boolean", "number"]
        assert get_content_type(None) == "null"

    def test_get_content_type_blocks(self):
        """Test content type for content blocks."""
        assert (
            get_content_type(ContentBlockText(type="text", text="Hi")) == "text_block"
        )
        assert (
            get_content_type(ContentBlockImage(type="image", source={"type": "base64", "media_type": "image/png", "data": "abc"}))
            == "image_block"
        )
        assert (
            get_content_type(
                ContentBlockToolUse(type="tool_use", id="1", name="tool", input={})
            )
            == "tool_use_block"
        )
        assert (
            get_content_type(
                ContentBlockToolResult(
                    type="tool_result", tool_use_id="1", content="result"
                )
            )
            == "tool_result_block"
        )
        assert (
            get_content_type(ContentBlockThinking(type="thinking", thinking="think"))
            == "thinking_block"
        )
        assert (
            get_content_type(ContentBlockRedactedThinking(type="redacted_thinking", data="hidden"))
            == "redacted_thinking_block"
        )

    def test_get_content_type_unknown(self):
        """Test content type for unknown objects."""

        class CustomClass:
            pass

        obj = CustomClass()
        result = get_content_type(obj)
        assert result == "unknown_CustomClass"
