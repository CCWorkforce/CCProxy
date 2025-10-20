"""Benchmark suite for CCProxy Cython optimizations.

This package contains performance benchmarks to measure the impact of
Cython optimizations on CPU-bound operations.

Benchmark Modules:
    - bench_type_checks: Type checking and dispatch operations
    - bench_lru_ops: LRU cache management operations
    - bench_cache_keys: Cache key generation and hashing
    - bench_tokenizer: Token counting operations

Usage:
    pytest benchmarks/ --benchmark-only
    pytest benchmarks/bench_type_checks.py --benchmark-only -v
"""

__all__ = []
