"""Setup script for CCProxy with Cython extensions."""

from setuptools import setup, find_packages, Extension
import os

# Check if Cython extensions should be built
enable_cython = os.getenv("CCPROXY_BUILD_CYTHON", "true").lower() not in (
    "false",
    "0",
    "no",
)

# Define potential Cython extension modules
potential_extensions = [
    ("ccproxy._cython.type_checks", "ccproxy/_cython/type_checks.pyx"),
    ("ccproxy._cython.lru_ops", "ccproxy/_cython/lru_ops.pyx"),
    ("ccproxy._cython.cache_keys", "ccproxy/_cython/cache_keys.pyx"),
]

# Check which .pyx files actually exist
existing_extensions = [
    (name, path) for name, path in potential_extensions if os.path.exists(path)
]

# Only cythonize if we have Cython extensions and they exist
ext_modules = []
if enable_cython and existing_extensions:
    try:
        from Cython.Build import cythonize

        ext_modules = cythonize(
            [
                Extension(name, [path], language="c")
                for name, path in existing_extensions
            ],
            compiler_directives={
                "language_level": "3",
                "boundscheck": False,
                "wraparound": False,
                "cdivision": True,
                "profile": True,  # Enable profiling
                "linetrace": True,  # Enable line tracing for line_profiler
            },
            annotate=True,  # Generate HTML annotation files
        )
        print(f"Building {len(ext_modules)} Cython extension(s)")
    except ImportError:
        print("Cython not available, skipping extension build")
elif not existing_extensions:
    print("No Cython source files (.pyx) found, skipping extension build")

setup(
    packages=find_packages(include=["ccproxy", "ccproxy.*"]),
    ext_modules=ext_modules,
)
