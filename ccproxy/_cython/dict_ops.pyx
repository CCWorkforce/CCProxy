# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: profile=True
# cython: linetrace=True

"""Optimized dictionary operations for error tracking, logging, and data sanitization."""

import json
from typing import Dict, Any, List, Set, Optional


cpdef dict recursive_redact(dict data, list sensitive_keys):
    """
    Recursively redact sensitive fields from nested dictionaries.

    Used in error_tracker.py and logging.py to remove secrets, tokens, and
    passwords from log entries and error reports. Walks the entire dictionary
    tree and replaces sensitive values with "[REDACTED]".

    Args:
        data: Dictionary to redact (will not be modified)
        sensitive_keys: List of key names to redact (case-insensitive)

    Returns:
        New dictionary with sensitive fields redacted

    Performance:
        Expected 35-45% improvement over recursive Python dict comprehension
        C-level iteration and string comparison reduce overhead
    """
    cdef dict result = {}
    cdef str key
    cdef object value
    cdef str key_lower
    cdef str sensitive_key

    if not data:
        return {}

    if not sensitive_keys:
        return data.copy()

    # Convert sensitive keys to lowercase for case-insensitive matching
    cdef set sensitive_set = {k.lower() for k in sensitive_keys}

    for key, value in data.items():
        key_lower = key.lower()

        # Check if this key should be redacted
        if key_lower in sensitive_set:
            result[key] = "[REDACTED]"
        elif isinstance(value, dict):
            # Recursively redact nested dictionaries
            result[key] = recursive_redact(value, sensitive_keys)
        elif isinstance(value, list):
            # Recursively redact dictionaries in lists
            result[key] = _redact_list(value, sensitive_keys)
        else:
            # Keep non-sensitive values as-is
            result[key] = value

    return result


cdef list _redact_list(list items, list sensitive_keys):
    """Helper function to redact dictionaries within lists."""
    cdef list result = []
    cdef object item

    for item in items:
        if isinstance(item, dict):
            result.append(recursive_redact(item, sensitive_keys))
        elif isinstance(item, list):
            result.append(_redact_list(item, sensitive_keys))
        else:
            result.append(item)

    return result


cpdef dict recursive_filter_none(dict data):
    """
    Recursively remove None values from nested dictionaries.

    Used in content_converter.py and response formatting to clean up
    optional fields that weren't set. Reduces payload size and improves
    JSON serialization performance.

    Args:
        data: Dictionary to filter

    Returns:
        New dictionary with None values removed

    Performance:
        Expected 30-40% improvement over recursive comprehension
        C-level None checking is faster than Python
    """
    cdef dict result = {}
    cdef str key
    cdef object value

    if not data:
        return {}

    for key, value in data.items():
        if value is None:
            continue
        elif isinstance(value, dict):
            filtered_dict = recursive_filter_none(value)
            if filtered_dict:  # Only include non-empty dicts
                result[key] = filtered_dict
        elif isinstance(value, list):
            filtered_list = _filter_none_list(value)
            if filtered_list:  # Only include non-empty lists
                result[key] = filtered_list
        else:
            result[key] = value

    return result


cdef list _filter_none_list(list items):
    """Helper function to filter None values from lists."""
    cdef list result = []
    cdef object item

    for item in items:
        if item is None:
            continue
        elif isinstance(item, dict):
            filtered_dict = recursive_filter_none(item)
            if filtered_dict:
                result.append(filtered_dict)
        elif isinstance(item, list):
            filtered_list = _filter_none_list(item)
            if filtered_list:
                result.append(filtered_list)
        else:
            result.append(item)

    return result


cpdef dict deep_merge_dicts(dict base, dict update):
    """
    Deep merge two dictionaries with conflict resolution.

    Used in configuration merging and response composition. Recursively
    merges nested dictionaries, with values from 'update' taking precedence.

    Args:
        base: Base dictionary
        update: Dictionary with updates (takes precedence)

    Returns:
        New merged dictionary

    Performance:
        Expected 25-35% improvement over recursive Python merge
        C-level dictionary operations reduce overhead
    """
    cdef dict result = base.copy()
    cdef str key
    cdef object value

    if not update:
        return result

    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = deep_merge_dicts(result[key], value)
        else:
            # Override with new value
            result[key] = value

    return result


cpdef dict extract_dict_subset(dict data, list keys):
    """
    Extract subset of dictionary with specified keys.

    Used in request validation and logging to extract only relevant fields
    from large request/response objects.

    Args:
        data: Source dictionary
        keys: List of keys to extract

    Returns:
        New dictionary with only specified keys

    Performance:
        Expected 20-30% improvement over dict comprehension
        C-level membership testing
    """
    cdef dict result = {}
    cdef str key

    if not data or not keys:
        return {}

    for key in keys:
        if key in data:
            result[key] = data[key]

    return result


cpdef int count_nested_keys(dict data):
    """
    Count total number of keys in nested dictionary.

    Used for complexity estimation and cache size limits.

    Args:
        data: Dictionary to count

    Returns:
        Total number of keys at all nesting levels

    Performance:
        Expected 35-45% improvement over recursive Python counting
        C-level integer arithmetic
    """
    cdef int count = 0
    cdef object value

    if not data:
        return 0

    count = len(data)

    for value in data.values():
        if isinstance(value, dict):
            count += count_nested_keys(value)
        elif isinstance(value, list):
            count += _count_keys_in_list(value)

    return count


cdef int _count_keys_in_list(list items):
    """Helper function to count keys in dictionaries within lists."""
    cdef int count = 0
    cdef object item

    for item in items:
        if isinstance(item, dict):
            count += count_nested_keys(item)
        elif isinstance(item, list):
            count += _count_keys_in_list(item)

    return count


cpdef list flatten_dict_keys(dict data, str separator='.'):
    """
    Flatten nested dictionary keys into dot-notation paths.

    Used in logging and error tracking to create flat field names.
    Example: {"user": {"name": "alice"}} -> ["user.name"]

    Args:
        data: Dictionary to flatten
        separator: Separator for key paths (default: '.')

    Returns:
        List of flattened key paths

    Performance:
        Expected 30-40% improvement over recursive generator
        C-level string concatenation
    """
    cdef list result = []
    cdef str key
    cdef object value

    if not data:
        return []

    for key, value in data.items():
        if isinstance(value, dict):
            nested_keys = flatten_dict_keys(value, separator)
            for nested_key in nested_keys:
                result.append(f"{key}{separator}{nested_key}")
        else:
            result.append(key)

    return result


cpdef dict sanitize_for_logging(dict data, list sensitive_keys, int max_string_length=1000):
    """
    Comprehensive sanitization for logging: redact + filter + truncate.

    Combines multiple operations into one pass for efficiency:
    - Redacts sensitive fields
    - Removes None values
    - Truncates long strings

    Args:
        data: Dictionary to sanitize
        sensitive_keys: Keys to redact
        max_string_length: Maximum string length before truncation

    Returns:
        Sanitized dictionary safe for logging

    Performance:
        Expected 40-50% improvement by combining operations
        Single-pass iteration vs multiple recursive passes
    """
    cdef dict result = {}
    cdef str key
    cdef object value
    cdef str key_lower
    cdef set sensitive_set

    if not data:
        return {}

    # Convert sensitive keys to set for O(1) lookup
    sensitive_set = {k.lower() for k in sensitive_keys} if sensitive_keys else set()

    for key, value in data.items():
        if value is None:
            continue

        key_lower = key.lower()

        # Redact sensitive fields
        if key_lower in sensitive_set:
            result[key] = "[REDACTED]"
        elif isinstance(value, str):
            # Truncate long strings
            if len(value) > max_string_length:
                result[key] = value[:max_string_length] + "...[truncated]"
            else:
                result[key] = value
        elif isinstance(value, dict):
            # Recursively sanitize nested dicts
            sanitized = sanitize_for_logging(value, sensitive_keys, max_string_length)
            if sanitized:
                result[key] = sanitized
        elif isinstance(value, list):
            # Sanitize lists
            sanitized_list = _sanitize_list(value, sensitive_keys, max_string_length)
            if sanitized_list:
                result[key] = sanitized_list
        else:
            result[key] = value

    return result


cdef list _sanitize_list(list items, list sensitive_keys, int max_string_length):
    """Helper function to sanitize lists."""
    cdef list result = []
    cdef object item

    for item in items:
        if item is None:
            continue
        elif isinstance(item, dict):
            sanitized = sanitize_for_logging(item, sensitive_keys, max_string_length)
            if sanitized:
                result.append(sanitized)
        elif isinstance(item, str):
            if len(item) > max_string_length:
                result.append(item[:max_string_length] + "...[truncated]")
            else:
                result.append(item)
        elif isinstance(item, list):
            sanitized_list = _sanitize_list(item, sensitive_keys, max_string_length)
            if sanitized_list:
                result.append(sanitized_list)
        else:
            result.append(item)

    return result


cpdef bint dict_has_key_path(dict data, str key_path, str separator='.'):
    """
    Check if a nested key path exists in dictionary.

    Example: dict_has_key_path(data, "user.profile.name") checks data["user"]["profile"]["name"]

    Args:
        data: Dictionary to check
        key_path: Dot-separated key path
        separator: Path separator (default: '.')

    Returns:
        True if path exists

    Performance:
        Expected 25-35% improvement over repeated dict.get() calls
        C-level path traversal
    """
    cdef list keys
    cdef str key
    cdef object current

    if not data or not key_path:
        return False

    keys = key_path.split(separator)
    current = data

    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]

    return True


cpdef object get_nested_value(dict data, str key_path, object default=None, str separator='.'):
    """
    Safely get value from nested dictionary using key path.

    Example: get_nested_value(data, "user.profile.name", "unknown")

    Args:
        data: Dictionary to query
        key_path: Dot-separated key path
        default: Default value if path doesn't exist
        separator: Path separator (default: '.')

    Returns:
        Value at key path or default

    Performance:
        Expected 30-40% improvement over chained .get() calls
        C-level path traversal with early exit
    """
    cdef list keys
    cdef str key
    cdef object current

    if not data or not key_path:
        return default

    keys = key_path.split(separator)
    current = data

    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]

    return current


# Export all public functions
__all__ = [
    'recursive_redact',
    'recursive_filter_none',
    'deep_merge_dicts',
    'extract_dict_subset',
    'count_nested_keys',
    'flatten_dict_keys',
    'sanitize_for_logging',
    'dict_has_key_path',
    'get_nested_value',
]
