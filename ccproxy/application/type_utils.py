"""Type checking utilities for consolidated isinstance patterns and type dispatching."""

from typing import Any, Dict, List, Optional, Union, TypeVar, Protocol, runtime_checkable
import dataclasses

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

T = TypeVar('T')


@runtime_checkable
class HasType(Protocol):
    """Protocol for objects with a 'type' attribute."""
    type: str


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


def is_serializable_primitive(obj: Any) -> bool:
    """Check if object is a JSON-serializable primitive."""
    return isinstance(obj, (str, int, float, bool, type(None)))


def is_dataclass_instance(obj: Any) -> bool:
    """Check if object is a dataclass instance (not a type)."""
    return dataclasses.is_dataclass(obj) and not isinstance(obj, type)


def is_redactable_key(key: Any, redact_keys: List[str]) -> bool:
    """Check if a key should be redacted based on redact list."""
    return isinstance(key, str) and key.lower() in redact_keys


def is_large_string(obj: Any, max_length: int = 5000) -> bool:
    """Check if object is a string exceeding max length."""
    return isinstance(obj, str) and len(obj) > max_length


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

    current = obj
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return False
        current = current.get(key)
    return True


def safe_get_nested(obj: Dict[str, Any], *path: str, default: Any = None) -> Any:
    """Safely get nested dictionary value with default."""
    if not isinstance(obj, dict):
        return default

    current = obj
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
        if current is default:
            return default
    return current


class ContentBlockDispatcher:
    """Dispatcher for handling different content block types."""

    @staticmethod
    def dispatch(block: Union[ContentBlock, Dict[str, Any]], handlers: Dict[str, callable]) -> Any:
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
            handler = handlers.get('text', handlers.get('_default'))
        elif isinstance(block, ContentBlockImage) or is_image_block(block):
            handler = handlers.get('image', handlers.get('_default'))
        elif isinstance(block, ContentBlockToolUse) or is_tool_use_block(block):
            handler = handlers.get('tool_use', handlers.get('_default'))
        elif isinstance(block, ContentBlockToolResult) or is_tool_result_block(block):
            handler = handlers.get('tool_result', handlers.get('_default'))
        elif isinstance(block, ContentBlockThinking) or is_thinking_block(block):
            handler = handlers.get('thinking', handlers.get('_default'))
        elif isinstance(block, ContentBlockRedactedThinking) or is_redacted_thinking_block(block):
            handler = handlers.get('redacted_thinking', handlers.get('_default'))
        else:
            handler = handlers.get('_unknown', handlers.get('_default'))

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
                return obj.decode('utf-8')
            except UnicodeDecodeError:
                return fallback
        if obj is None:
            return fallback
        try:
            return str(obj)
        except Exception:
            return fallback

    @staticmethod
    def to_dict(obj: Any, fallback: Optional[Dict] = None) -> Dict:
        """Convert object to dict with fallback."""
        if isinstance(obj, dict):
            return obj
        if is_dataclass_instance(obj):
            return dataclasses.asdict(obj)
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        if hasattr(obj, 'dict'):
            return obj.dict()
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return fallback or {}

    @staticmethod
    def ensure_list(obj: Any) -> List:
        """Ensure object is a list."""
        if isinstance(obj, list):
            return obj
        if isinstance(obj, (tuple, set)):
            return list(obj)
        if obj is None:
            return []
        return [obj]


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