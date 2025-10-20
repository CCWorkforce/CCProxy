# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: profile=True
# cython: linetrace=True
"""Cython-optimized type checking functions for CCProxy.

This module provides high-performance type checking and dispatch operations
for content blocks and other domain objects. The functions are compiled to C
for maximum performance with minimal Python object overhead.

Performance Targets:
    - Type checks: < 0.15ms per call (50% improvement from ~0.3ms)
    - Dispatch: < 0.2ms per call
    - Content type detection: < 0.1ms per call
"""

from typing import Any
cimport cython

# Define type name constants at C level for faster comparisons
cdef str TYPE_TEXT = "text"
cdef str TYPE_IMAGE = "image"
cdef str TYPE_TOOL_USE = "tool_use"
cdef str TYPE_TOOL_RESULT = "tool_result"
cdef str TYPE_THINKING = "thinking"
cdef str TYPE_REDACTED_THINKING = "redacted_thinking"


@cython.cfunc
@cython.inline
cdef bint _is_dict_with_type(object obj, str type_str):
    """Internal C function to check if dict has specific type field.

    This is a cdef function that's not exposed to Python but used internally
    for fast type checking at the C level.
    """
    if not isinstance(obj, dict):
        return False
    return obj.get("type") == type_str


cpdef bint is_text_block(object obj):
    """Check if object is a text content block.

    Args:
        obj: Object to check

    Returns:
        True if object is a ContentBlockText or dict with type="text"
    """
    # Check for ContentBlockText class (fastest path)
    cdef str obj_type_name = type(obj).__name__
    if obj_type_name == "ContentBlockText":
        return True

    # Check for dict with type="text" and text field
    if isinstance(obj, dict):
        return obj.get("type") == TYPE_TEXT and "text" in obj

    return False


cpdef bint is_image_block(object obj):
    """Check if object is an image content block.

    Args:
        obj: Object to check

    Returns:
        True if object is a ContentBlockImage or dict with type="image"
    """
    cdef str obj_type_name = type(obj).__name__
    if obj_type_name == "ContentBlockImage":
        return True

    return _is_dict_with_type(obj, TYPE_IMAGE)


cpdef bint is_tool_use_block(object obj):
    """Check if object is a tool use content block.

    Args:
        obj: Object to check

    Returns:
        True if object is a ContentBlockToolUse or dict with type="tool_use"
    """
    cdef str obj_type_name = type(obj).__name__
    if obj_type_name == "ContentBlockToolUse":
        return True

    return _is_dict_with_type(obj, TYPE_TOOL_USE)


cpdef bint is_tool_result_block(object obj):
    """Check if object is a tool result content block.

    Args:
        obj: Object to check

    Returns:
        True if object is a ContentBlockToolResult or dict with type="tool_result"
    """
    cdef str obj_type_name = type(obj).__name__
    if obj_type_name == "ContentBlockToolResult":
        return True

    return _is_dict_with_type(obj, TYPE_TOOL_RESULT)


cpdef bint is_thinking_block(object obj):
    """Check if object is a thinking content block.

    Args:
        obj: Object to check

    Returns:
        True if object is a ContentBlockThinking or dict with type="thinking"
    """
    cdef str obj_type_name = type(obj).__name__
    if obj_type_name == "ContentBlockThinking":
        return True

    return _is_dict_with_type(obj, TYPE_THINKING)


cpdef bint is_redacted_thinking_block(object obj):
    """Check if object is a redacted thinking content block.

    Args:
        obj: Object to check

    Returns:
        True if object is a ContentBlockRedactedThinking or dict with type="redacted_thinking"
    """
    cdef str obj_type_name = type(obj).__name__
    if obj_type_name == "ContentBlockRedactedThinking":
        return True

    return _is_dict_with_type(obj, TYPE_REDACTED_THINKING)


cpdef str get_content_type(object content):
    """Get the type of content as a string descriptor.

    Optimized version that uses C-level type checking for better performance.

    Args:
        content: Content object to classify

    Returns:
        String descriptor of content type
    """
    # Use C-level type name lookup
    cdef str type_name = type(content).__name__

    # Fast path for primitive types
    if isinstance(content, str):
        return "string"
    if isinstance(content, list):
        return "list"
    if isinstance(content, dict):
        return "dict"
    if content is None:
        return "null"
    if isinstance(content, bool):
        return "boolean"
    if isinstance(content, (int, float)):
        return "number"

    # Content block types (using type name for speed)
    if type_name == "ContentBlockText":
        return "text_block"
    if type_name == "ContentBlockImage":
        return "image_block"
    if type_name == "ContentBlockToolUse":
        return "tool_use_block"
    if type_name == "ContentBlockToolResult":
        return "tool_result_block"
    if type_name == "ContentBlockThinking":
        return "thinking_block"
    if type_name == "ContentBlockRedactedThinking":
        return "redacted_thinking_block"

    return f"unknown_{type_name}"


cpdef str dispatch_block_type(object block):
    """Fast dispatch to determine block type string.

    This is a helper function optimized for the dispatcher pattern,
    returning the type string directly without handlers.

    Args:
        block: Content block to classify

    Returns:
        Type string: "text", "image", "tool_use", "tool_result", "thinking",
                     "redacted_thinking", or "_unknown"
    """
    # Fast type name lookup
    cdef str type_name = type(block).__name__

    # Check each type in order of likelihood (most common first)
    if type_name == "ContentBlockText" or is_text_block(block):
        return TYPE_TEXT
    if type_name == "ContentBlockImage" or is_image_block(block):
        return TYPE_IMAGE
    if type_name == "ContentBlockToolUse" or is_tool_use_block(block):
        return TYPE_TOOL_USE
    if type_name == "ContentBlockToolResult" or is_tool_result_block(block):
        return TYPE_TOOL_RESULT
    if type_name == "ContentBlockThinking" or is_thinking_block(block):
        return TYPE_THINKING
    if type_name == "ContentBlockRedactedThinking" or is_redacted_thinking_block(block):
        return TYPE_REDACTED_THINKING

    return "_unknown"


cpdef bint is_serializable_primitive(object obj):
    """Check if object is a JSON-serializable primitive.

    Args:
        obj: Object to check

    Returns:
        True if object is str, int, float, bool, or None
    """
    return isinstance(obj, (str, int, float, bool, type(None)))


cpdef bint is_redactable_key(object key, list redact_keys):
    """Check if a key should be redacted based on redact list.

    Args:
        key: Key to check
        redact_keys: List of lowercase keys to redact

    Returns:
        True if key is in redact list (case-insensitive)
    """
    if not isinstance(key, str):
        return False
    return key.lower() in redact_keys


cpdef bint is_large_string(object obj, Py_ssize_t max_length=5000):
    """Check if object is a string exceeding max length.

    Args:
        obj: Object to check
        max_length: Maximum allowed string length

    Returns:
        True if object is a string longer than max_length
    """
    if not isinstance(obj, str):
        return False
    return len(<str>obj) > max_length
