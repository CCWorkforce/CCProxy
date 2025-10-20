# Benchmarks - CLAUDE.md

**Scope**: Performance benchmarks for CCProxy Cython optimizations

## Overview

This directory contains comprehensive performance benchmarks using pytest-benchmark to measure the impact of Cython optimizations on CPU-bound operations. The benchmarks compare Cython-compiled modules against their pure Python equivalents.

**For detailed usage instructions, see `README.md` in this directory.**

## Benchmark Files

### Core Benchmarks (Phase 1)
- `bench_type_checks.py`: Type checking and dispatch operations (30-50% expected improvement)
  - Tests for `is_text_block()`, `is_image_block()`, `is_tool_use_block()`, etc.
  - ContentBlockDispatcher performance
  - Multiple type checks in sequence

- `bench_lru_ops.py`: LRU cache operations (20-40% expected improvement)
  - Cache hit/miss latency
  - LRU eviction operations
  - Shard index calculation
  - Timestamp filtering and cleanup

- `bench_tokenizer.py`: Token counting operations (15-25% indirect improvement)
  - Anthropic request token counting
  - OpenAI request token counting
  - Cache hit performance
  - tiktoken encoding operations

### Advanced Benchmarks (Phase 2-4)
- `bench_json_ops.py`: JSON serialization and operations (10.7x faster for size estimation)
  - Compact JSON dumps
  - Size estimation (90%+ improvement measured)
  - JSON validation and parsing

- `bench_stream_state.py`: SSE event formatting (20-30% expected improvement)
  - SSE event generation
  - Content block event formatting
  - Batch event processing

- `bench_dict_ops.py`: Dictionary operations (7.83x faster for nested key counting)
  - Recursive redaction (1.27x faster measured)
  - Recursive filter None (2.88x faster measured)
  - Deep merge dicts (1.72x faster measured)
  - Nested key counting (7.83x faster measured)
  - Sanitize for logging (1.12x faster measured)

- `bench_validation.py`: Validation operations (30-40% expected improvement)
  - Content block validation
  - Message structure validation
  - Tool structure validation
  - JSON serializability checks
  - Complexity estimation

## Running Benchmarks

### Basic Usage
```bash
# Run all benchmarks
pytest benchmarks/ --benchmark-only

# Run specific benchmark module
pytest benchmarks/bench_type_checks.py --benchmark-only -v

# Verbose output with statistics
pytest benchmarks/ --benchmark-only --benchmark-verbose
```

### Comparing Before/After Cython
```bash
# Baseline (without Cython)
export CCPROXY_ENABLE_CYTHON=false
pytest benchmarks/ --benchmark-only --benchmark-save=baseline

# Optimized (with Cython)
export CCPROXY_ENABLE_CYTHON=true
pytest benchmarks/ --benchmark-only --benchmark-save=optimized

# Compare results
pytest-benchmark compare baseline optimized
```

### Generate Reports
```bash
# JSON output
pytest benchmarks/ --benchmark-only --benchmark-json=results.json

# Histogram
pytest benchmarks/ --benchmark-only --benchmark-histogram=histogram

# Compare multiple saved benchmarks
pytest-benchmark compare baseline optimized --csv=comparison.csv
```

## Performance Targets

### Phase 1 Targets
- Type checks: < 0.15ms per call (50% improvement from ~0.3ms)
- LRU operations: < 0.9ms per eviction (40% improvement from ~1.5ms)
- Cache key generation: < 9ms per key (25% improvement from ~12ms)

### Phase 2-4 Measured Results
- JSON size estimation: **10.7x faster** (+90.7%) ⭐⭐
- Nested key counting: **7.83x faster** (+87.2%) ⭐⭐
- Filter None values: **2.88x faster** (+65.3%) ⭐
- Deep merge dicts: **1.72x faster** (+41.9%)
- Recursive redaction: **1.27x faster** (+21.2%)

### Overall Expected Impact
- **15-35% reduction** in CPU-bound operation latency
- Real-world impact depends on request/response complexity and cache hit rates

## Guidelines for Writing Benchmarks

### Benchmark Structure
```python
import pytest
from ccproxy._cython import CYTHON_ENABLED

# Import Cython module if available
if CYTHON_ENABLED:
    try:
        from ccproxy._cython.module_name import optimized_func
        _USING_CYTHON = True
    except ImportError:
        _USING_CYTHON = False
else:
    _USING_CYTHON = False

# Pure Python fallback
if not _USING_CYTHON:
    from ccproxy.application.module_name import optimized_func

def test_benchmark_operation(benchmark):
    """Benchmark description."""
    # Setup
    test_data = create_test_data()

    # Benchmark
    result = benchmark(optimized_func, test_data)

    # Assertions to verify correctness
    assert result is not None
```

### Best Practices
- **Deterministic data**: Use fixed test data, not random values
- **Verify correctness**: Always assert that results are correct
- **Realistic scenarios**: Use data structures that match production usage
- **Warm-up**: pytest-benchmark handles warm-up automatically
- **Isolation**: Each benchmark should test one specific operation
- **Documentation**: Include docstrings explaining what is being measured

### Naming Conventions
- File: `bench_<module_name>.py`
- Test: `test_<operation>_<scenario>(benchmark)`
- Be descriptive: `test_recursive_redact_nested_dict()` not `test_redact()`

### Interpreting Results

**Key Metrics**:
- **Mean**: Average time per operation (primary metric)
- **StdDev**: Standard deviation (lower is more consistent)
- **Min/Max**: Range of observed times
- **Ops/sec**: Operations per second (higher is better)

**Example Output**:
```
Name (time in us)                    Min        Max       Mean    StdDev    Median      IQR   Outliers  OPS
test_json_size_estimation         4.29      45.83       4.62      0.89      4.50     0.12    190;295  216,450
```

### Troubleshooting

**High variance**:
- Close other applications
- Run multiple rounds: `--benchmark-warmup=on --benchmark-rounds=100`
- Check CPU frequency scaling

**Benchmarks too fast**:
- pytest-benchmark automatically increases iterations for fast operations

**Async benchmarks**:
- Ensure Python 3.13+ with proper asyncio support
- Use `benchmark.pedantic()` for more control over async operations

## Integration with CI/CD

### Regression Testing
```bash
# Save baseline on main branch
git checkout main
pytest benchmarks/ --benchmark-only --benchmark-save=main

# Compare PR branch
git checkout feature-branch
pytest benchmarks/ --benchmark-only --benchmark-compare=main --benchmark-compare-fail=mean:10%
```

### Performance Monitoring
- Store benchmark results as artifacts
- Track performance trends over time
- Alert on significant regressions (>10% slowdown)

## Contributing

When adding new benchmarks:
1. Follow the existing naming convention
2. Include comprehensive docstrings
3. Test both Cython and pure Python paths
4. Document expected performance targets
5. Verify results are correct with assertions
6. Add to this CLAUDE.md file

## References

- **pytest-benchmark**: https://pytest-benchmark.readthedocs.io/
- **Cython Performance Tips**: See `/CYTHON_INTEGRATION.md`
- **Benchmark README**: See `README.md` in this directory for detailed usage
