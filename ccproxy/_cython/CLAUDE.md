# Cython Optimization Module - CLAUDE.md

**Scope**: High-performance Cython-compiled modules for CPU-bound operations in CCProxy

## Overview

This module contains 9 Cython-optimized `.pyx` modules that provide 15-35% performance improvements for CPU-intensive operations. All modules include pure Python fallbacks and are controlled via the `CCPROXY_ENABLE_CYTHON` environment variable (enabled by default).

**For comprehensive implementation details, see `/CYTHON_INTEGRATION.md` in the project root.**

## Module Organization

### Phase 1: Core Optimizations (Complete)

#### `type_checks.pyx` - Type Checking Optimization
**Target**: 30-50% improvement in type checking throughput
**Integration**: `ccproxy/application/type_utils.py`

**Optimized Functions**:
- `is_text_block()`, `is_image_block()`, `is_tool_use_block()`, `is_tool_result_block()`
- `is_thinking_block()`, `is_redacted_thinking_block()`
- `get_content_type()` - Fast content type classification
- `dispatch_block_type()` - Type-based dispatch with minimal overhead
- `is_serializable_primitive()` - Quick serializability check
- `is_redactable_key()` - Sensitive key detection for redaction
- `is_large_string()` - Large string detection for truncation

**Key Optimizations**:
- C-level type name lookups via `type(obj).__name__`
- Reduced Python object overhead with `cpdef` functions
- Inlined helper functions with `@cython.cfunc`
- Fast path for common types (str, list, dict, primitives)

#### `lru_ops.pyx` - LRU Cache Operation Optimization
**Target**: 20-40% improvement in cache management
**Integration**: Can be used in `tokenizer.py`, `request_validator.py`, `rate_limiter.py`

**Optimized Functions**:
- `get_shard_index()` - Consistent hashing for cache sharding
- `is_expired()` - TTL expiry checks with C-level arithmetic
- `filter_recent_timestamps()` - Fast timestamp filtering
- `filter_request_times()` - Rate limiter request cleanup
- `filter_token_counts()` - TPM rate limiting token count cleanup
- `sum_token_counts()` - Fast token sum calculation for TPM
- `get_expired_keys()`, `count_expired_entries()` - Cache cleanup operations
- `calculate_hit_rate()` - Cache statistics calculation
- `calculate_max_per_shard()` - Shard capacity calculation

**Key Optimizations**:
- C-level timestamp arithmetic using `double` precision
- Optimized list filtering without list comprehensions
- Fast shard index calculation with modulo operations
- Zero-copy operations where possible

#### `cache_keys.pyx` - Cache Key Generation Optimization
**Target**: 15-25% improvement in cache key generation
**Integration**: Can be used in `response_cache.py`, `request_validator.py`, `tokenizer.py`

**Optimized Functions**:
- `compute_sha256_hex()`, `compute_sha256_hex_from_str()` - SHA256 hashing
- `generate_request_hash()` - Request validation cache key generation
- `join_cache_key_parts()` - Optimized string concatenation for keys
- `normalize_model_name()` - Model name normalization
- `create_token_cache_key()` - Token count cache key generation
- `create_response_cache_key()` - Response cache key generation
- `estimate_json_size()` - Quick size estimation without serialization
- `encode_utf8()`, `decode_utf8()` - Fast UTF-8 encoding/decoding

**Key Optimizations**:
- Minimal overhead SHA256 wrappers around hashlib
- C-level string concatenation for key building
- Fast UTF-8 encoding/decoding without Python overhead
- Efficient size estimation without full JSON serialization

### Phase 2: Advanced JSON & String Operations (Complete)

#### `json_ops.pyx` - JSON Operation Optimization
**Target**: 30-40% improvement for JSON serialization (90% for size estimation)
**Integration**: Can be used in `streaming.py` (13 calls), `content_converter.py` (9 calls), `tokenizer.py` (6 calls)

**Optimized Functions**:
- `json_dumps_compact()` - Compact JSON with minimal separators
- `json_dumps_sorted()` - Sorted keys for cache consistency
- `json_loads_safe()` - Safe parsing with error handling
- `estimate_json_size()` - Fast size estimation (10.7x speedup measured)
- `json_dumps_with_default()` - Custom default handler support
- `is_valid_json()` - Quick JSON validation without parsing

**Key Optimizations**:
- Optimized separators for minimal output size
- Consistent sorting for cache key generation
- 10.7x speedup for size estimation (measured: 232,963 ops/sec vs 21,772 ops/sec)
- Safe fallback for non-serializable objects

**Benchmark Results** (10K iterations):
- `estimate_json_size()`: **10.70x faster** (+90.7%)

#### `string_ops.pyx` - String & Pattern Matching Optimization
**Target**: 40-50% improvement for regex matching, 25-35% for hashing
**Integration**: Can be used in `guardrails.py` (8 regex ops), `request_validator.py` (hashing), `error_tracker.py` (redaction)

**Optimized Functions**:
- `regex_multi_match()` - Multi-pattern regex with early termination
- `string_hash_sha256()` - Optimized SHA-256 hashing
- `safe_decode_utf8()` - Fast UTF-8 decoding with fallback
- `contains_sensitive_keyword()` - Fast keyword scanning for redaction
- `truncate_string()` - Efficient string truncation
- `escape_for_json()` - Fast JSON string escaping
- `split_by_newlines()` - Optimized newline splitting
- `join_with_separator()` - Fast string joining
- `string_to_utf8_bytes()` - Optimized UTF-8 encoding

**Key Optimizations**:
- C-level pattern matching with early exits
- Minimal SHA-256 wrapper overhead
- Fast keyword lookup for sensitive data detection
- Reduced allocations in string operations

#### `serialization.pyx` - Content Serialization Optimization
**Target**: 25-35% improvement for content serialization
**Integration**: Can be used in `content_converter.py` (serialize_tool_result_content, extract_system_text)

**Optimized Functions**:
- `serialize_primitive()` - Fast primitive serialization (int, float, bool, None)
- `serialize_list_to_text()` - Optimized list-to-text conversion
- `join_with_newline()` - Fast newline joining with filtering
- `extract_text_from_blocks()` - Text extraction from content blocks
- `serialize_dict_compact()` - Fast compact dict serialization
- `is_text_content_block()` - Quick block type checking
- `filter_text_blocks()` - Filter for text blocks only
- `serialize_with_fallback()` - Generic serialization with fallbacks

**Key Optimizations**:
- Type-specific fast paths for common cases
- Pre-allocated string buffers
- Minimal Python object allocations
- C-level filtering and iteration

### Phase 3: Streaming & Dictionary Operations (Complete)

#### `stream_state.pyx` - SSE Event Formatting Optimization
**Target**: 20-30% improvement for SSE event generation
**Integration**: Can be used in `streaming.py` (StreamProcessor class)

**Optimized Functions**:
- `build_sse_event()` - Fast SSE (Server-Sent Events) event formatting
- `increment_token_count()` - Efficient token count increment
- `format_content_block_start()` - Format content_block_start SSE event
- `format_content_block_delta()` - Format content_block_delta SSE event (hot path)
- `format_content_block_stop()` - Format content_block_stop SSE event
- `format_message_start()`, `format_message_delta()`, `format_message_stop()`
- `format_ping_event()`, `format_error_event()` - Utility SSE events
- `batch_format_events()` - Batch format multiple SSE events

**Key Optimizations**:
- Pre-allocated buffer and optimized JSON serialization
- Type-specific fast paths for common block types
- Minimal allocation for simple stop events
- Reduced overhead by batching event formatting

**Note**: Mixed results due to already-optimized json.dumps() in Python 3.13+. Most benefit from batch operations and high-frequency streaming scenarios.

#### `dict_ops.pyx` - Dictionary Operations Optimization
**Target**: 35-45% improvement for recursive operations, up to 7.8x for counting
**Integration**: Can be used in `error_tracker.py` (redaction), `logging.py` (sanitization)

**Optimized Functions**:
- `recursive_redact()` - Recursively redact sensitive fields from nested dicts
- `recursive_filter_none()` - Recursively remove None values
- `deep_merge_dicts()` - Deep merge two dictionaries with conflict resolution
- `extract_dict_subset()` - Extract subset of dictionary with specified keys
- `count_nested_keys()` - Count total number of keys in nested dictionary (7.83x faster)
- `flatten_dict_keys()` - Flatten nested dictionary keys into dot-notation paths
- `sanitize_for_logging()` - Comprehensive sanitization: redact + filter + truncate
- `dict_has_key_path()` - Check if nested key path exists
- `get_nested_value()` - Safely get value from nested dictionary

**Key Optimizations**:
- C-level type name lookups and iteration
- Single-pass combined operations (sanitize_for_logging)
- Optimized set membership testing for sensitive keys
- Minimal Python object allocations

**Benchmark Results** (measured):
- `count_nested_keys()`: **7.83x faster** (+87.2%) ⭐
- `recursive_filter_none()`: **2.88x faster** (+65.3%)
- `deep_merge_dicts()`: **1.72x faster** (+41.9%)
- `recursive_redact()`: **1.27x faster** (+21.2%)
- `sanitize_for_logging()`: **1.12x faster** (+10.6%)

### Phase 4: Validation Operations (Complete)

#### `validation.pyx` - Validation Operations Optimization
**Target**: 30-40% improvement for validation throughput
**Integration**: Can be used in `request_validator.py`, `response_cache.py`, `converters.py`

**Optimized Functions**:
- `validate_content_blocks()` - Validate structure of content blocks
- `check_json_serializable()` - Fast check if object is JSON serializable
- `validate_message_structure()` - Validate Anthropic message structure
- `validate_tool_structure()` - Validate tool definition structure
- `check_required_fields()` - Check if dictionary has all required fields
- `validate_field_types()` - Validate field types match expected types
- `estimate_object_complexity()` - Estimate complexity of nested object structure
- `is_valid_anthropic_model()` - Check if model string is valid Anthropic model
- `validate_token_limits()` - Validate token counts against model limits
- `is_safe_for_cache()` - Check if data structure is safe to cache

**Key Optimizations**:
- C-level type checking and field access
- Type-based fast path for common serializable types
- O(1) set membership test for model validation
- Single-pass complexity estimation with recursion

**Expected Performance**:
- Content block validation: 30-40% improvement
- JSON serializability check: 25-35% improvement
- Message validation: 30-40% improvement

## Environment Configuration

### `CCPROXY_ENABLE_CYTHON`
**Default**: `true` (ENABLED)

Controls whether Cython modules are used at runtime:
```bash
# Enable Cython (DEFAULT - no need to set)
export CCPROXY_ENABLE_CYTHON=true

# Disable Cython (use pure Python fallback)
export CCPROXY_ENABLE_CYTHON=false
```

**Note**: Can be toggled at runtime without rebuilding.

### `CCPROXY_BUILD_CYTHON`
**Default**: `true` (ENABLED)

Controls whether Cython extensions are built during installation:
```bash
# Skip Cython build (useful for CI/CD without compilers)
export CCPROXY_BUILD_CYTHON=false
uv pip install -e .
```

## Compiler Directives

All `.pyx` files use these optimized compiler directives:
```python
# cython: language_level=3
# cython: boundscheck=False      # Disable array bounds checking
# cython: wraparound=False        # Disable negative indexing
# cython: cdivision=True          # Use C division semantics
# cython: profile=True            # Enable profiling
# cython: linetrace=True          # Enable line tracing
```

## Function Types

- `cdef`: Pure C functions (not callable from Python, maximum performance)
- `cpdef`: Hybrid functions (callable from both Python and C)
- `@cython.cfunc`: Decorator for C functions with type inference
- `@cython.inline`: Force inline expansion for small functions

## Guidelines

### When to Use Cython Modules

**✅ Ideal Use Cases**:
- High-frequency operations called in tight loops
- CPU-bound operations (type checking, arithmetic, hashing)
- List/dict operations with known types
- String operations (parsing, concatenation, encoding)
- Operations profiled as bottlenecks

**❌ Not Beneficial For**:
- I/O-bound operations (already async)
- Operations using optimized C libraries (tiktoken, orjson, hashlib directly)
- Pydantic model operations (Pydantic v2 uses Rust)
- Complex async operations with extensive Python object interactions

### Integration Pattern

```python
from .._cython import CYTHON_ENABLED

if CYTHON_ENABLED:
    try:
        from .._cython.type_checks import is_text_block
        _USING_CYTHON = True
    except ImportError:
        _USING_CYTHON = False
else:
    _USING_CYTHON = False

# Pure Python fallback
if not _USING_CYTHON:
    def is_text_block(obj: Any) -> bool:
        """Pure Python implementation."""
        return isinstance(obj, dict) and obj.get("type") == "text"
```

### Performance Testing

```bash
# Run all benchmarks
pytest benchmarks/ --benchmark-only

# Compare before/after Cython
export CCPROXY_ENABLE_CYTHON=false
pytest benchmarks/ --benchmark-only --benchmark-save=baseline

export CCPROXY_ENABLE_CYTHON=true
pytest benchmarks/ --benchmark-only --benchmark-save=optimized

pytest-benchmark compare baseline optimized
```

### Debugging Cython Code

**View Cython Annotations**:
HTML annotation files show Python/C interaction overhead:
```bash
open ccproxy/_cython/*.html
```
Yellow highlights = Python overhead, White = pure C (fast)

**Enable Line Profiling**:
```bash
pip install line-profiler
kernprof -l -v script_using_cython.py
```

**Using cProfile**:
```python
import cProfile
from ccproxy.application.type_utils import is_text_block
cProfile.run('is_text_block(test_object)', sort='cumtime')
```

## Build System

### Building Extensions

```bash
# Install with Cython extensions (development)
uv pip install -e .

# Force rebuild
pip install -e . --force-reinstall --no-deps
```

### Verifying Installation

```python
from ccproxy._cython import CYTHON_ENABLED

if CYTHON_ENABLED:
    try:
        from ccproxy._cython.type_checks import is_text_block
        print("✓ Cython type_checks module loaded")
    except ImportError:
        print("✗ Cython not available, using pure Python")
else:
    print("ℹ Cython disabled via environment variable")
```

### Checking Compiled Modules

```bash
# List compiled .so files
ls -lh ccproxy/_cython/*.so

# Should see 9 files like:
# type_checks.cpython-314-darwin.so
# lru_ops.cpython-314-darwin.so
# ... (7 more)
```

## Integration Status

### Fully Integrated
- ✅ `type_utils.py` - Uses `type_checks.pyx` functions

### Ready for Integration (modules exist but not yet wired)
- ⏳ `tokenizer.py` - Can use `lru_ops` and `cache_keys`
- ⏳ `request_validator.py` - Can use `lru_ops`, `cache_keys`, `validation`
- ⏳ `rate_limiter.py` - Can use `lru_ops` functions
- ⏳ `response_cache.py` - Can use `cache_keys`, `validation`
- ⏳ `streaming.py` - Can use `stream_state` for SSE events
- ⏳ `error_tracker.py` - Can use `dict_ops` for redaction
- ⏳ `logging.py` - Can use `dict_ops` for sanitization
- ⏳ `guardrails.py` - Can use `string_ops` for regex matching
- ⏳ `content_converter.py` - Can use `serialization`, `json_ops`

## Troubleshooting

### Build Errors

**"Cython not found"**:
```bash
pip install cython>=3.0.0
```

**"compiler not found"**:
- macOS: `xcode-select --install`
- Linux: `apt-get install build-essential python3-dev`
- Windows: Install Visual Studio Build Tools

### Runtime Issues

**ImportError when importing Cython modules**:
- Check that `uv pip install -e .` completed successfully
- Verify `.so` files exist: `ls ccproxy/_cython/*.so`
- Try rebuilding: `pip install -e . --force-reinstall --no-deps`

**Performance not improving**:
- Verify Cython is enabled: `echo $CCPROXY_ENABLE_CYTHON`
- Check that Cython modules are actually being imported
- Profile to identify actual bottlenecks
- Remember: I/O-bound operations won't benefit

## Performance Summary

**Overall Expected Impact**: 15-35% reduction in CPU-bound operation latency

**Measured Performance Gains**:
- `json_ops.estimate_json_size()`: **10.7x faster** (+90.7%)
- `dict_ops.count_nested_keys()`: **7.83x faster** (+87.2%)
- `dict_ops.recursive_filter_none()`: **2.88x faster** (+65.3%)
- `dict_ops.deep_merge_dicts()`: **1.72x faster** (+41.9%)

**Module Statistics**:
- Total Cython modules: 9
- Total optimized functions: 60+
- Lines of Cython code: ~2,400
- Compiled .so files: 9
- Build time: ~20 seconds
- Size on disk: ~1.2 MB

## References

- **Comprehensive Documentation**: `/CYTHON_INTEGRATION.md`
- **Benchmarks**: `/benchmarks/` directory with README.md
- **Cython Documentation**: https://cython.readthedocs.io/
- **Compiler Directives**: https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html
- **Performance Tips**: https://cython.readthedocs.io/en/latest/src/userguide/pyrex_differences.html
