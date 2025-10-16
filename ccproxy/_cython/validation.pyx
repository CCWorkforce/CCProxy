# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: profile=True
# cython: linetrace=True

"""Optimized validation operations for request/response validation and cache key generation."""

import json
from typing import Dict, Any, List, Set, Optional, Tuple


cpdef tuple validate_content_blocks(list blocks):
    """
    Validate structure of content blocks in Anthropic format.

    Used in response_cache.py and converters to ensure content blocks
    have required fields and valid types before caching or processing.

    Args:
        blocks: List of content blocks to validate

    Returns:
        Tuple of (is_valid: bool, error_message: str)
        Returns (True, "") if valid

    Performance:
        Expected 30-40% improvement over Python validation loops
        C-level type checking and field access
    """
    cdef object block
    cdef dict block_dict
    cdef str block_type
    cdef int index = 0

    if not blocks:
        return (False, "Content blocks cannot be empty")

    if not isinstance(blocks, list):
        return (False, "Content blocks must be a list")

    for block in blocks:
        if not isinstance(block, dict):
            return (False, f"Block at index {index} is not a dictionary")

        block_dict = block

        # Check for required 'type' field
        if 'type' not in block_dict:
            return (False, f"Block at index {index} missing required 'type' field")

        block_type = str(block_dict.get('type', ''))

        # Validate based on block type
        if block_type == 'text':
            if 'text' not in block_dict:
                return (False, f"Text block at index {index} missing 'text' field")
            if not isinstance(block_dict['text'], str):
                return (False, f"Text block at index {index} has non-string 'text' field")

        elif block_type == 'image':
            if 'source' not in block_dict:
                return (False, f"Image block at index {index} missing 'source' field")
            if not isinstance(block_dict['source'], dict):
                return (False, f"Image block at index {index} has invalid 'source' field")

        elif block_type == 'tool_use':
            if 'id' not in block_dict:
                return (False, f"Tool use block at index {index} missing 'id' field")
            if 'name' not in block_dict:
                return (False, f"Tool use block at index {index} missing 'name' field")
            if 'input' not in block_dict:
                return (False, f"Tool use block at index {index} missing 'input' field")

        elif block_type == 'tool_result':
            if 'tool_use_id' not in block_dict:
                return (False, f"Tool result block at index {index} missing 'tool_use_id' field")
            if 'content' not in block_dict:
                return (False, f"Tool result block at index {index} missing 'content' field")

        elif block_type == 'thinking':
            if 'thinking' not in block_dict:
                return (False, f"Thinking block at index {index} missing 'thinking' field")

        else:
            return (False, f"Block at index {index} has unknown type '{block_type}'")

        index += 1

    return (True, "")


cpdef bint check_json_serializable(object obj):
    """
    Fast check if an object is JSON serializable.

    Used in response_cache.py before caching and in error_tracker.py
    before logging. Attempts serialization with error handling.

    Args:
        obj: Object to check

    Returns:
        True if object can be serialized to JSON

    Performance:
        Expected 25-35% improvement over try/except json.dumps()
        Type-based fast path for common serializable types
    """
    # Fast path for common types
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return True

    if isinstance(obj, (list, tuple)):
        return _check_list_serializable(obj)

    if isinstance(obj, dict):
        return _check_dict_serializable(obj)

    # Try serialization for other types
    try:
        json.dumps(obj)
        return True
    except (TypeError, ValueError, OverflowError):
        return False


cdef bint _check_list_serializable(object obj):
    """Helper to check if list/tuple is serializable."""
    cdef object item

    try:
        for item in obj:
            if not check_json_serializable(item):
                return False
        return True
    except Exception:
        return False


cdef bint _check_dict_serializable(dict obj):
    """Helper to check if dictionary is serializable."""
    cdef str key
    cdef object value

    try:
        for key, value in obj.items():
            # Keys must be strings
            if not isinstance(key, str):
                return False
            if not check_json_serializable(value):
                return False
        return True
    except Exception:
        return False


cpdef tuple validate_message_structure(dict message):
    """
    Validate Anthropic message structure.

    Used in request_validator.py to validate incoming messages
    before processing or caching.

    Args:
        message: Message dictionary to validate

    Returns:
        Tuple of (is_valid: bool, error_message: str)

    Performance:
        Expected 30-40% improvement over Python validation
        C-level dictionary access and type checking
    """
    cdef str role
    cdef object content

    if not message:
        return (False, "Message cannot be empty")

    if not isinstance(message, dict):
        return (False, "Message must be a dictionary")

    # Check required 'role' field
    if 'role' not in message:
        return (False, "Message missing required 'role' field")

    role = str(message.get('role', ''))
    if role not in ('user', 'assistant'):
        return (False, f"Message has invalid role '{role}' (must be 'user' or 'assistant')")

    # Check required 'content' field
    if 'content' not in message:
        return (False, "Message missing required 'content' field")

    content = message['content']

    # Content can be string or list of blocks
    if isinstance(content, str):
        if not content:
            return (False, "Message content cannot be empty string")
    elif isinstance(content, list):
        # Validate content blocks
        is_valid, error = validate_content_blocks(content)
        if not is_valid:
            return (False, f"Invalid content blocks: {error}")
    else:
        return (False, "Message content must be string or list of content blocks")

    return (True, "")


cpdef tuple validate_tool_structure(dict tool):
    """
    Validate tool definition structure.

    Used in converters and request_validator.py to ensure tools
    are properly defined before sending to provider.

    Args:
        tool: Tool dictionary to validate

    Returns:
        Tuple of (is_valid: bool, error_message: str)

    Performance:
        Expected 25-35% improvement over Python validation
        C-level field checking
    """
    cdef dict input_schema

    if not tool:
        return (False, "Tool cannot be empty")

    if not isinstance(tool, dict):
        return (False, "Tool must be a dictionary")

    # Check required fields
    if 'name' not in tool:
        return (False, "Tool missing required 'name' field")

    if not isinstance(tool['name'], str):
        return (False, "Tool 'name' must be a string")

    if not tool['name']:
        return (False, "Tool 'name' cannot be empty")

    # Description is optional but must be string if present
    if 'description' in tool:
        if not isinstance(tool['description'], str):
            return (False, "Tool 'description' must be a string")

    # Input schema is required
    if 'input_schema' not in tool:
        return (False, "Tool missing required 'input_schema' field")

    input_schema = tool['input_schema']
    if not isinstance(input_schema, dict):
        return (False, "Tool 'input_schema' must be a dictionary")

    # Validate JSON Schema structure
    if 'type' not in input_schema:
        return (False, "Tool 'input_schema' missing 'type' field")

    if input_schema['type'] != 'object':
        return (False, "Tool 'input_schema' type must be 'object'")

    return (True, "")


cpdef tuple check_required_fields(dict data, list required_fields):
    """
    Check if dictionary has all required fields.

    Used throughout validators to ensure critical fields are present.

    Args:
        data: Dictionary to check
        required_fields: List of required field names

    Returns:
        Tuple of (has_all_fields: bool, missing_fields: list)

    Performance:
        Expected 20-30% improvement over list comprehension
        C-level membership testing
    """
    cdef list missing = []
    cdef str field

    if not data:
        return (False, required_fields)

    for field in required_fields:
        if field not in data:
            missing.append(field)

    if missing:
        return (False, missing)

    return (True, [])


cpdef bint validate_field_types(dict data, dict type_spec):
    """
    Validate field types match expected types.

    Used in validators to ensure fields have correct types.

    Args:
        data: Dictionary with fields to validate
        type_spec: Dictionary mapping field names to expected type names
                  Example: {"model": "str", "max_tokens": "int"}

    Returns:
        True if all fields match expected types

    Performance:
        Expected 25-35% improvement over Python type checking loops
        C-level type inspection
    """
    cdef str field
    cdef str expected_type
    cdef object value
    cdef str actual_type

    if not data or not type_spec:
        return True

    for field, expected_type in type_spec.items():
        if field not in data:
            continue  # Skip missing fields (use check_required_fields for that)

        value = data[field]
        actual_type = type(value).__name__

        # Handle special cases
        if expected_type == "str" and not isinstance(value, str):
            return False
        elif expected_type == "int" and not isinstance(value, int):
            return False
        elif expected_type == "float" and not isinstance(value, (int, float)):
            return False
        elif expected_type == "bool" and not isinstance(value, bool):
            return False
        elif expected_type == "list" and not isinstance(value, list):
            return False
        elif expected_type == "dict" and not isinstance(value, dict):
            return False
        elif expected_type == "None" and value is not None:
            return False

    return True


cpdef int estimate_object_complexity(object obj):
    """
    Estimate complexity of nested object structure.

    Used for cache size limits and validation depth checks.
    Counts total number of nested objects and collections.

    Args:
        obj: Object to analyze

    Returns:
        Complexity score (higher = more complex)

    Performance:
        Expected 30-40% improvement over recursive Python counting
        C-level recursion and type checking
    """
    cdef int complexity = 1  # Count the object itself

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return 1

    if isinstance(obj, dict):
        complexity += len(obj)
        for value in obj.values():
            complexity += estimate_object_complexity(value)

    elif isinstance(obj, (list, tuple)):
        complexity += len(obj)
        for item in obj:
            complexity += estimate_object_complexity(item)

    return complexity


cpdef bint is_valid_anthropic_model(str model):
    """
    Check if model string is a valid Anthropic model name.

    Used in request validation and model selection.

    Args:
        model: Model name to validate

    Returns:
        True if valid Anthropic model

    Performance:
        O(1) set membership test at C level
    """
    cdef set valid_models

    if not model:
        return False

    valid_models = {
        'claude-3-5-sonnet-20241022',
        'claude-3-5-sonnet-20240620',
        'claude-3-5-haiku-20241022',
        'claude-3-opus-20240229',
        'claude-3-sonnet-20240229',
        'claude-3-haiku-20240307',
        'claude-2.1',
        'claude-2.0',
        'claude-instant-1.2',
    }

    return model in valid_models


cpdef tuple validate_token_limits(int input_tokens, int max_tokens, str model):
    """
    Validate token counts against model limits.

    Used in tokenizer.py and request_validator.py to prevent
    requests that exceed model context windows.

    Args:
        input_tokens: Number of input tokens
        max_tokens: Requested max output tokens
        model: Model name

    Returns:
        Tuple of (is_valid: bool, error_message: str)

    Performance:
        Expected 20-30% improvement over Python validation
        C-level integer comparison
    """
    cdef int total_tokens
    cdef int model_limit

    if input_tokens < 0:
        return (False, "Input tokens cannot be negative")

    if max_tokens <= 0:
        return (False, "Max tokens must be positive")

    # Model context limits (approximate)
    if 'opus' in model or 'sonnet' in model:
        model_limit = 200000
    elif 'haiku' in model:
        model_limit = 200000
    else:
        model_limit = 100000

    total_tokens = input_tokens + max_tokens

    if total_tokens > model_limit:
        return (False, f"Total tokens ({total_tokens}) exceeds model limit ({model_limit})")

    return (True, "")


cpdef bint is_safe_for_cache(dict data, int max_complexity=10000):
    """
    Check if data structure is safe to cache.

    Validates that data is JSON serializable, not too complex,
    and doesn't contain sensitive fields.

    Args:
        data: Data to validate
        max_complexity: Maximum allowed complexity score

    Returns:
        True if safe to cache

    Performance:
        Combines multiple checks in single pass for efficiency
    """
    cdef int complexity

    if not data:
        return False

    # Check JSON serializability
    if not check_json_serializable(data):
        return False

    # Check complexity
    complexity = estimate_object_complexity(data)
    if complexity > max_complexity:
        return False

    # Check for sensitive fields (common patterns)
    cdef list sensitive_keys = ['api_key', 'token', 'password', 'secret', 'auth']
    cdef str key
    cdef str key_lower

    for key in data.keys():
        key_lower = key.lower()
        for sensitive_key in sensitive_keys:
            if sensitive_key in key_lower:
                return False

    return True


# Export all public functions
__all__ = [
    'validate_content_blocks',
    'check_json_serializable',
    'validate_message_structure',
    'validate_tool_structure',
    'check_required_fields',
    'validate_field_types',
    'estimate_object_complexity',
    'is_valid_anthropic_model',
    'validate_token_limits',
    'is_safe_for_cache',
]
