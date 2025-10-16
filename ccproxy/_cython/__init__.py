"""Cython-optimized performance modules for CCProxy.

This package contains compiled Cython extensions that optimize CPU-bound
operations in the CCProxy application. Each module has a pure Python fallback
implementation in case Cython extensions are not available or disabled.

Environment Configuration:
    CCPROXY_ENABLE_CYTHON: Set to 'false' to disable Cython modules (default: 'true')

Modules:
    - type_checks: Optimized content block type checking
    - lru_ops: Optimized LRU cache operations
    - cache_keys: Optimized cache key generation
"""

import os
from typing import Final

# Feature flag for enabling/disabling Cython modules
CYTHON_ENABLED: Final[bool] = os.getenv(
    "CCPROXY_ENABLE_CYTHON", "true"
).lower() not in (
    "false",
    "0",
    "no",
)

__all__ = ["CYTHON_ENABLED"]
