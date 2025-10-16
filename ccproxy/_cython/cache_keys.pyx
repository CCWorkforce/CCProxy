# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: profile=True
# cython: linetrace=True
"""Cython-optimized cache key generation and hashing for CCProxy.

This module provides high-performance cache key generation operations including:
- SHA256 hashing with minimal overhead
- String concatenation and formatting for cache keys
- JSON normalization for consistent key generation
- Model name normalization

Performance Targets:
    - SHA256 hashing: < 8ms for 50KB input
    - String concatenation: < 0.5ms for typical keys
    - Cache key generation: < 9ms total (25% improvement from ~12ms)
"""

import hashlib
from typing import Any, List, Dict, Optional
cimport cython


cpdef str compute_sha256_hex(bytes data):
    """Compute SHA256 hash of bytes and return hexadecimal digest.

    Optimized wrapper around hashlib.sha256 with minimal Python overhead.

    Args:
        data: Bytes to hash

    Returns:
        Hexadecimal digest string (64 characters)
    """
    return hashlib.sha256(data).hexdigest()


cpdef str compute_sha256_hex_from_str(str text):
    """Compute SHA256 hash of string and return hexadecimal digest.

    Convenience method that handles UTF-8 encoding.

    Args:
        text: String to hash

    Returns:
        Hexadecimal digest string (64 characters)
    """
    cdef bytes data = text.encode('utf-8')
    return hashlib.sha256(data).hexdigest()


cpdef str generate_request_hash(str json_str):
    """Generate hash for a JSON request string.

    Optimized for request validation cache keys.

    Args:
        json_str: JSON string (should be sorted keys)

    Returns:
        SHA256 hexadecimal digest
    """
    return compute_sha256_hex_from_str(json_str)


cpdef str join_cache_key_parts(list parts, str separator='|'):
    """Join cache key parts with separator.

    Optimized string concatenation for building cache keys from multiple parts.

    Args:
        parts: List of strings to join
        separator: Separator string (default: '|')

    Returns:
        Joined string
    """
    return separator.join(parts)


cpdef str normalize_model_name(str model_name):
    """Normalize model name for consistent cache keys.

    Converts to lowercase and removes common variations.

    Args:
        model_name: Raw model name

    Returns:
        Normalized model name
    """
    cdef str normalized = model_name.lower().strip()

    # Remove common prefixes/suffixes for consistency
    if normalized.startswith('anthropic/'):
        normalized = normalized[10:]  # len('anthropic/')
    elif normalized.startswith('openai/'):
        normalized = normalized[7:]   # len('openai/')

    return normalized


cpdef str create_token_cache_key(str model_name, str messages_hash, str system_hash, str tools_hash):
    """Create cache key for token counting.

    Optimized for token count caching with pre-computed hashes.

    Args:
        model_name: Model name
        messages_hash: Hash of messages
        system_hash: Hash of system content
        tools_hash: Hash of tools

    Returns:
        Cache key string
    """
    cdef list parts = [model_name, messages_hash, system_hash, tools_hash]
    return join_cache_key_parts(parts)


cpdef str create_response_cache_key(str model_name, str request_hash):
    """Create cache key for response caching.

    Args:
        model_name: Model name
        request_hash: Hash of full request

    Returns:
        Cache key string
    """
    cdef list parts = [normalize_model_name(model_name), request_hash]
    return join_cache_key_parts(parts)


cpdef bint is_valid_cache_key(str key):
    """Validate cache key format.

    Checks that key is non-empty and has reasonable length.

    Args:
        key: Cache key to validate

    Returns:
        True if key is valid
    """
    cdef Py_ssize_t length = len(key)
    return length > 0 and length < 10000  # Reasonable max length


cpdef str truncate_for_logging(str value, Py_ssize_t max_length=100):
    """Truncate string for logging purposes.

    Args:
        value: String to truncate
        max_length: Maximum length (default: 100)

    Returns:
        Truncated string with '...' if necessary
    """
    if len(value) <= max_length:
        return value
    return value[:max_length] + '...'


cpdef str extract_cache_key_prefix(str key, Py_ssize_t prefix_length=8):
    """Extract prefix from cache key for logging.

    Args:
        key: Full cache key
        prefix_length: Length of prefix to extract

    Returns:
        Key prefix (or full key if shorter than prefix_length)
    """
    if len(key) <= prefix_length:
        return key
    return key[:prefix_length]


@cython.cdivision(True)
cpdef Py_ssize_t estimate_json_size(object obj):
    """Estimate serialized JSON size without actually serializing.

    Rough estimation for memory management decisions.

    Args:
        obj: Object to estimate

    Returns:
        Estimated size in bytes
    """
    cdef Py_ssize_t size = 0

    if isinstance(obj, str):
        size = len(<str>obj) + 2  # Add quotes
    elif isinstance(obj, (int, float)):
        size = 16  # Rough average
    elif isinstance(obj, bool):
        size = 5  # "true" or "false"
    elif obj is None:
        size = 4  # "null"
    elif isinstance(obj, list):
        size = 2  # Brackets
        for item in <list>obj:
            size += estimate_json_size(item) + 1  # +1 for comma
    elif isinstance(obj, dict):
        size = 2  # Braces
        for key, value in (<dict>obj).items():
            size += len(str(key)) + 3  # key + quotes + colon
            size += estimate_json_size(value) + 1  # +1 for comma
    else:
        # Unknown type, use repr length as estimate
        size = len(repr(obj))

    return size


cpdef bytes encode_utf8(str text):
    """Encode string to UTF-8 bytes with minimal overhead.

    Args:
        text: String to encode

    Returns:
        UTF-8 encoded bytes
    """
    return text.encode('utf-8')


cpdef str decode_utf8(bytes data):
    """Decode UTF-8 bytes to string with error handling.

    Args:
        data: UTF-8 bytes

    Returns:
        Decoded string (or empty string on error)
    """
    try:
        return data.decode('utf-8')
    except UnicodeDecodeError:
        return ''


cpdef bint compare_cache_keys(str key1, str key2):
    """Compare two cache keys for equality.

    Optimized comparison at C level.

    Args:
        key1: First cache key
        key2: Second cache key

    Returns:
        True if keys are equal
    """
    # Cython optimizes string comparison to C level
    return key1 == key2


cpdef list split_cache_key(str key, str separator='|'):
    """Split cache key into parts.

    Args:
        key: Cache key to split
        separator: Separator string

    Returns:
        List of key parts
    """
    return key.split(separator)


cpdef str sanitize_cache_key(str key):
    """Sanitize cache key by removing invalid characters.

    Args:
        key: Raw cache key

    Returns:
        Sanitized cache key
    """
    # Remove control characters and whitespace
    cdef list chars = []
    cdef str char

    for char in key:
        if ord(char) >= 32 and ord(char) != 127:  # Printable ASCII
            chars.append(char)

    return ''.join(chars)
