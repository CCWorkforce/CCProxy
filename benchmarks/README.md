# CCProxy Cython Optimization Benchmarks

This directory contains performance benchmarks for measuring the impact of Cython optimizations on CPU-bound operations in CCProxy.

## Prerequisites

Install development dependencies including pytest-benchmark:

```bash
uv pip install --dev
```

## Running Benchmarks

### Run all benchmarks
```bash
pytest benchmarks/ --benchmark-only
```

### Run specific benchmark module
```bash
pytest benchmarks/bench_type_checks.py --benchmark-only -v
pytest benchmarks/bench_lru_ops.py --benchmark-only -v
pytest benchmarks/bench_tokenizer.py --benchmark-only -v
```

### Compare before/after optimization
```bash
# Before Cython optimization
pytest benchmarks/ --benchmark-only --benchmark-save=before

# After Cython optimization
pytest benchmarks/ --benchmark-only --benchmark-save=after

# Compare results
pytest-benchmark compare before after
```

### Generate detailed reports
```bash
# JSON output
pytest benchmarks/ --benchmark-only --benchmark-json=results.json

# Histogram
pytest benchmarks/ --benchmark-only --benchmark-histogram=histogram

# Verbose statistics
pytest benchmarks/ --benchmark-only --benchmark-verbose
```

## Benchmark Modules

### bench_type_checks.py
Measures performance of type checking and dispatch operations:
- `is_text_block()`, `is_image_block()`, etc.
- `ContentBlockDispatcher` performance
- Multiple type checks in sequence

**Expected Impact**: 30-50% improvement with Cython

### bench_lru_ops.py
Measures LRU cache operation performance:
- Cache hit/miss latency
- LRU eviction operations
- Hash generation for cache keys

**Expected Impact**: 20-40% improvement with Cython

### bench_tokenizer.py
Measures token counting operation performance:
- Anthropic request token counting
- OpenAI request token counting
- Cache hit performance
- tiktoken encoding operations

**Expected Impact**: 15-25% improvement with Cython (indirect via reduced overhead)

## Interpreting Results

### Key Metrics
- **Mean**: Average time per operation
- **StdDev**: Standard deviation (consistency)
- **Min/Max**: Range of observed times
- **Iterations**: Number of test runs
- **Rounds**: Number of calibration rounds

### Performance Targets
For Cython optimizations to be worthwhile, we expect:
- Type checks: < 0.15ms per call (50% improvement from ~0.3ms)
- LRU operations: < 0.9ms per eviction (40% improvement from ~1.5ms)
- Cache key generation: < 9ms per key (25% improvement from ~12ms)

### Example Output
```
---------------------------------------------------------------------------------------- benchmark: 12 tests ----------------------------------------------------------------------------------------
Name (time in us)                                Min                   Max                  Mean              StdDev                Median                 IQR            Outliers       OPS            Rounds  Iterations
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_is_text_block                            2.5800 (1.0)          45.8330 (1.0)          2.7825 (1.0)        0.8891 (1.0)          2.7080 (1.0)        0.1248 (1.0)       190;295  359,4058.6503 (1.0)      37467           1
test_is_image_block                           2.6250 (1.02)         47.0840 (1.03)         2.8523 (1.03)       0.9234 (1.04)         2.7917 (1.03)       0.1460 (1.17)      196;311  350,5882.3529 (0.97)     36891           1
...
```

## Baseline Results (Pre-Cython)

Run these commands to establish baseline performance:

```bash
# Type checking baseline
pytest benchmarks/bench_type_checks.py --benchmark-only --benchmark-save=baseline_type_checks

# LRU operations baseline
pytest benchmarks/bench_lru_ops.py --benchmark-only --benchmark-save=baseline_lru_ops

# Token counting baseline
pytest benchmarks/bench_tokenizer.py --benchmark-only --benchmark-save=baseline_tokenizer
```

## Troubleshooting

### Benchmarks are too fast to measure
If operations complete in < 1μs, pytest-benchmark will automatically increase iterations.

### High variance in results
- Close other applications
- Disable CPU frequency scaling: `sudo cpupower frequency-set --governor performance`
- Run multiple rounds: `--benchmark-warmup=on --benchmark-rounds=100`

### Async benchmarks fail
Ensure you're using Python 3.13+ with proper asyncio support.

## Contributing

When adding new benchmarks:
1. Follow the existing naming convention: `bench_<module>.py`
2. Use descriptive test names: `test_<operation>_<scenario>`
3. Include docstrings explaining what is being measured
4. Add assertions to verify results are correct
5. Document expected performance targets
