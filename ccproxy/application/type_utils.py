from typing import Any, Callable, TypeVar, Protocol, runtime_checkable, cast

import dataclasses

from typing import Union

from ..domain.models import (
    ContentBlockText,
    ContentBlockImage,
    ContentBlockToolUse,
    ContentBlockToolResult,
    ContentBlockThinking,
    ContentBlockRedactedThinking,
    ContentBlock,
    SystemContent,
)

from .._cython import CYTHON_ENABLED

T = TypeVar("T")


@runtime_checkable
class HasType(Protocol):
    """Protocol for objects with a 'type' attribute."""

    type: str


# Try to import Cython-optimized functions
if CYTHON_ENABLED:
    try:
        from .._cython.type_checks import (
            is_text_block,
            is_image_block,
            is_tool_use_block,
            is_tool_result_block,
            is_thinking_block,
            is_redacted_thinking_block,
            is_serializable_primitive,
            is_redactable_key,
            is_large_string,
            get_content_type,
            dispatch_block_type,
        )

        _USING_CYTHON = True
    except ImportError:
        _USING_CYTHON = False
else:
    _USING_CYTHON = False

# Fallback to pure Python implementations if Cython not available
if not _USING_CYTHON:

    def is_text_block(obj: Any) -> bool:
        """Check if object is a text content block."""
        return isinstance(obj, ContentBlockText) or (
            isinstance(obj, dict) and obj.get("type") == "text" and "text" in obj
        )

    def is_image_block(obj: Any) -> bool:
        """Check if object is an image content block."""
        return isinstance(obj, ContentBlockImage) or (
            isinstance(obj, dict) and obj.get("type") == "image"
        )

    def is_tool_use_block(obj: Any) -> bool:
        """Check if object is a tool use content block."""
        return isinstance(obj, ContentBlockToolUse) or (
            isinstance(obj, dict) and obj.get("type") == "tool_use"
        )

    def is_tool_result_block(obj: Any) -> bool:
        """Check if object is a tool result content block."""
        return isinstance(obj, ContentBlockToolResult) or (
            isinstance(obj, dict) and obj.get("type") == "tool_result"
        )

    def is_thinking_block(obj: Any) -> bool:
        """Check if object is a thinking content block."""
        return isinstance(obj, ContentBlockThinking) or (
            isinstance(obj, dict) and obj.get("type") == "thinking"
        )

    def is_redacted_thinking_block(obj: Any) -> bool:
        """Check if object is a redacted thinking content block."""
        return isinstance(obj, ContentBlockRedactedThinking) or (
            isinstance(obj, dict) and obj.get("type") == "redacted_thinking"
        )

    def is_serializable_primitive(obj: Any) -> bool:
        """Check if object is a JSON-serializable primitive."""
        return isinstance(obj, (str, int, float, bool, type(None)))

    def is_redactable_key(key: Any, redact_keys: list[str]) -> bool:
        """Check if a key should be redacted based on redact list."""
        return isinstance(key, str) and key.lower() in redact_keys

    def is_large_string(obj: Any, max_length: int = 5000) -> bool:
        """Check if object is a string exceeding max length."""
        return isinstance(obj, str) and len(obj) > max_length

    def get_content_type(content: Any) -> str:
        """Get the type of content as a string descriptor."""
        if isinstance(content, str):
            return "string"
        elif isinstance(content, list):
            return "list"
        elif isinstance(content, dict):
            return "dict"
        elif isinstance(content, ContentBlockText):
            return "text_block"
        elif isinstance(content, ContentBlockImage):
            return "image_block"
        elif isinstance(content, ContentBlockToolUse):
            return "tool_use_block"
        elif isinstance(content, ContentBlockToolResult):
            return "tool_result_block"
        elif isinstance(content, ContentBlockThinking):
            return "thinking_block"
        elif isinstance(content, ContentBlockRedactedThinking):
            return "redacted_thinking_block"
        elif isinstance(content, (int, float)):
            return "number"
        elif isinstance(content, bool):
            return "boolean"
        elif content is None:
            return "null"
        else:
            return f"unknown_{type(content).__name__}"

    def dispatch_block_type(block: Union[ContentBlock, dict[str, Any]]) -> str:
        """Fast dispatch to determine block type string."""
        if isinstance(block, ContentBlockText) or is_text_block(block):
            return "text"
        elif isinstance(block, ContentBlockImage) or is_image_block(block):
            return "image"
        elif isinstance(block, ContentBlockToolUse) or is_tool_use_block(block):
            return "tool_use"
        elif isinstance(block, ContentBlockToolResult) or is_tool_result_block(block):
            return "tool_result"
        elif isinstance(block, ContentBlockThinking) or is_thinking_block(block):
            return "thinking"
        elif isinstance(
            block, ContentBlockRedactedThinking
        ) or is_redacted_thinking_block(block):
            return "redacted_thinking"
        else:
            return "_unknown"


def is_system_text_block(obj: Any) -> bool:
    """Check if object is a system text content block."""
    return (isinstance(obj, SystemContent) and obj.type == "text") or (
        isinstance(obj, dict) and obj.get("type") == "text"
    )


def is_string_content(obj: Any) -> bool:
    """Check if object is string content."""
    return isinstance(obj, str)


def is_list_content(obj: Any) -> bool:
    """Check if object is list content."""
    return isinstance(obj, list)


def is_dict_content(obj: Any) -> bool:
    """Check if object is dictionary content."""
    return isinstance(obj, dict)


def is_dataclass_instance(obj: Any) -> bool:
    """Check if object is a dataclass instance (not a type)."""
    return dataclasses.is_dataclass(obj) and not isinstance(obj, type)


def is_error_dict_with_field(obj: Any, field: str) -> bool:
    """Check if object is a dict with specified field."""
    return isinstance(obj, dict) and obj.get(field) is not None


def is_nested_dict_field(obj: Any, *path: str) -> bool:
    """Check if object is a dict with nested field path.

    Example: is_nested_dict_field(obj, "error", "stack_trace")
    checks for obj["error"]["stack_trace"] existence.
    """
    if not isinstance(obj, dict):
        return False

    current: dict[str, Any] = obj
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]  # type: ignore
    return True


def safe_get_nested(obj: dict[str, Any], *path: str, default: Any = None) -> Any:
    """Safely get nested dictionary value with default."""
    if not isinstance(obj, dict):
        return default

    current: dict[str, Any] = obj
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)  # type: ignore
        if current is default:
            return default
    return current


class ContentBlockDispatcher:
    """Dispatcher for handling different content block types."""

    @staticmethod
    def dispatch(
        block: Union[ContentBlock, dict[str, Any]],
        handlers: dict[str, Callable[[Any], Any]],
    ) -> Any:
        """Dispatch to appropriate handler based on block type.

        Args:
            block: Content block to dispatch
            handlers: Dict mapping block type to handler function
                     Special keys: '_default' for fallback, '_unknown' for unknown types

        Returns:
            Result of handler function
        """
        # Determine block type
        if isinstance(block, ContentBlockText) or is_text_block(block):
            handler = handlers.get("text", handlers.get("_default"))
        elif isinstance(block, ContentBlockImage) or is_image_block(block):
            handler = handlers.get("image", handlers.get("_default"))
        elif isinstance(block, ContentBlockToolUse) or is_tool_use_block(block):
            handler = handlers.get("tool_use", handlers.get("_default"))
        elif isinstance(block, ContentBlockToolResult) or is_tool_result_block(block):
            handler = handlers.get("tool_result", handlers.get("_default"))
        elif isinstance(block, ContentBlockThinking) or is_thinking_block(block):
            handler = handlers.get("thinking", handlers.get("_default"))
        elif isinstance(
            block, ContentBlockRedactedThinking
        ) or is_redacted_thinking_block(block):
            handler = handlers.get("redacted_thinking", handlers.get("_default"))
        else:
            handler = handlers.get("_unknown", handlers.get("_default"))

        if handler:
            return handler(block)
        return None


class TypeConverter:
    """Generic type conversion utilities."""

    @staticmethod
    def to_string(obj: Any, fallback: str = "") -> str:
        """Convert object to string with fallback."""
        if isinstance(obj, str):
            return obj
        if isinstance(obj, bytes):
            try:
                return obj.decode("utf-8")
            except UnicodeDecodeError:
                return fallback
        if obj is None:
            return fallback
        try:
            return str(obj)
        except Exception:
            return fallback

    @staticmethod
    def to_dict(obj: Any, fallback: dict[Any, Any] | None = None) -> dict[Any, Any]:
        """Convert object to dict with fallback."""
        if isinstance(obj, dict):
            return obj
        if is_dataclass_instance(obj):
            return dataclasses.asdict(obj)
        if hasattr(obj, "model_dump"):
            return cast(dict[Any, Any], obj.model_dump())
        if hasattr(obj, "dict"):
            return cast(dict[Any, Any], obj.dict())
        if hasattr(obj, "__dict__"):
            return cast(dict[Any, Any], obj.__dict__)
        return fallback or {}

    @staticmethod
    def ensure_list(obj: Any) -> list[Any]:
        """Ensure object is a list."""
        if isinstance(obj, list):
            return obj
        if isinstance(obj, (tuple, set)):
            return list(obj)
        if obj is None:
            return []
        return [obj]
