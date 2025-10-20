"""Cython-optimized performance modules for CCProxy.

This package contains compiled Cython extensions that optimize CPU-bound
operations in the CCProxy application. Each module has a pure Python fallback
implementation in case Cython extensions are not available or disabled.

Environment Configuration:
    CCPROXY_ENABLE_CYTHON: Set to 'false' to disable Cython modules (default: 'true')
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

# Try to import Phase 2 modules if available
try:
    from . import json_ops
except ImportError:
    json_ops = None  # type: ignore

try:
    from . import string_ops
except ImportError:
    string_ops = None  # type: ignore

try:
    from . import serialization
except ImportError:
    serialization = None  # type: ignore

# Try to import Phase 3 modules if available
try:
    from . import stream_state
except ImportError:
    stream_state = None  # type: ignore

try:
    from . import dict_ops
except ImportError:
    dict_ops = None  # type: ignore

# Try to import Phase 4 modules if available
try:
    from . import validation
except ImportError:
    validation = None  # type: ignore

__all__ = [
    "CYTHON_ENABLED",
    "json_ops",
    "string_ops",
    "serialization",
    "stream_state",
    "dict_ops",
    "validation",
]
