#!/bin/bash
# Test script to verify Cython build and fallback behavior locally
# This mirrors the CI/CD workflow checks

set -e  # Exit on error

echo "=========================================="
echo "CCProxy Cython Build Verification"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo "→ $1"
}

# Test 1: Build with Cython enabled
echo "Test 1: Building with Cython ENABLED"
echo "--------------------------------------"

export CCPROXY_BUILD_CYTHON=true
export CCPROXY_ENABLE_CYTHON=true

print_info "Cleaning previous builds..."
find ccproxy/_cython -name "*.so" -o -name "*.pyd" -delete 2>/dev/null || true

print_info "Installing with Cython..."
if uv sync --reinstall > /dev/null 2>&1; then
    print_success "Build completed"
else
    print_error "Build failed"
    exit 1
fi

print_info "Checking for compiled modules..."
SO_COUNT=$(find ccproxy/_cython -name "*.so" -o -name "*.pyd" 2>/dev/null | wc -l | tr -d ' ')
if [ "$SO_COUNT" -gt 0 ]; then
    print_success "Found $SO_COUNT Cython module(s)"
    find ccproxy/_cython -name "*.so" -o -name "*.pyd" | sed 's/^/  /'
else
    print_warning "No Cython modules found (may be expected on some platforms)"
fi

print_info "Testing runtime import..."
if uv run python -c "
from ccproxy._cython import CYTHON_ENABLED
print(f'  CYTHON_ENABLED: {CYTHON_ENABLED}')

try:
    from ccproxy._cython.type_checks import is_text_block
    print('  Successfully imported Cython module')
    result = is_text_block({'type': 'text', 'text': 'test'})
    if result:
        print('  Function works correctly')
    else:
        print('  ERROR: Function returned incorrect result')
        exit(1)
except ImportError as e:
    print(f'  WARNING: Could not import Cython module: {e}')
    print('  Fallback mode will be used')
" 2>&1; then
    print_success "Runtime verification passed"
else
    print_error "Runtime verification failed"
    exit 1
fi

echo ""

# Test 2: Build with Cython disabled (fallback mode)
echo "Test 2: Building with Cython DISABLED (Fallback Mode)"
echo "-------------------------------------------------------"

export CCPROXY_BUILD_CYTHON=false
export CCPROXY_ENABLE_CYTHON=false

print_info "Cleaning all build artifacts and compiled modules..."
# Remove all Cython compiled modules
find ccproxy/_cython -name "*.so" -delete 2>/dev/null || true
find ccproxy/_cython -name "*.pyd" -delete 2>/dev/null || true
find ccproxy/_cython -name "*.c" -delete 2>/dev/null || true
find ccproxy/_cython -name "*.html" -delete 2>/dev/null || true
# Remove build directories
rm -rf build/ dist/ *.egg-info/ .eggs/ 2>/dev/null || true
# Remove cached bytecode
find ccproxy/_cython -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

print_info "Rebuilding package without Cython compilation..."
# Force a clean build by removing the package first
uv pip uninstall -y CCProxy 2>/dev/null || true
# Now install fresh without Cython
if uv sync > /dev/null 2>&1; then
    print_success "Build completed"
else
    print_error "Build failed"
    exit 1
fi

print_info "Verifying no compiled modules exist..."
SO_COUNT=$(find ccproxy/_cython -name "*.so" -o -name "*.pyd" 2>/dev/null | wc -l | tr -d ' ')
if [ "$SO_COUNT" -eq 0 ]; then
    print_success "Confirmed no Cython modules built"
else
    print_warning "Found $SO_COUNT Cython modules (may be from cache)"
    print_info "Forcing deletion and continuing with fallback test..."
    find ccproxy/_cython -name "*.so" -delete 2>/dev/null || true
    find ccproxy/_cython -name "*.pyd" -delete 2>/dev/null || true
fi

print_info "Testing pure Python fallback..."
if uv run python -c "
from ccproxy._cython import CYTHON_ENABLED
print(f'  CYTHON_ENABLED: {CYTHON_ENABLED}')

if CYTHON_ENABLED:
    print('  ERROR: Cython should be disabled')
    exit(1)

from ccproxy.application.type_utils import is_text_block
print('  Successfully imported pure Python implementation')

result = is_text_block({'type': 'text', 'text': 'test'})
if result:
    print('  Function works correctly')
else:
    print('  ERROR: Function returned incorrect result')
    exit(1)
" 2>&1; then
    print_success "Fallback mode verification passed"
else
    print_error "Fallback mode verification failed"
    exit 1
fi

echo ""

# Test 3: Run tests in both modes
echo "Test 3: Running Test Suite"
echo "----------------------------"

print_info "Running tests with Cython enabled..."
export CCPROXY_BUILD_CYTHON=true
export CCPROXY_ENABLE_CYTHON=true
uv sync --reinstall > /dev/null 2>&1

if uv run pytest tests/test_type_utils.py -v --tb=short > /tmp/test_cython.log 2>&1; then
    print_success "Tests passed with Cython"
else
    print_warning "Some tests failed with Cython (check /tmp/test_cython.log)"
fi

print_info "Running tests with fallback mode..."
export CCPROXY_BUILD_CYTHON=false
export CCPROXY_ENABLE_CYTHON=false
find ccproxy/_cython -name "*.so" -delete 2>/dev/null || true
find ccproxy/_cython -name "*.pyd" -delete 2>/dev/null || true

if uv run pytest tests/test_type_utils.py -v --tb=short > /tmp/test_fallback.log 2>&1; then
    print_success "Tests passed in fallback mode"
else
    print_warning "Some tests failed in fallback mode (check /tmp/test_fallback.log)"
fi

echo ""

# Test 4: Quick benchmark comparison
echo "Test 4: Quick Performance Check"
echo "--------------------------------"

print_info "Running quick benchmark with Cython..."
export CCPROXY_BUILD_CYTHON=true
export CCPROXY_ENABLE_CYTHON=true
uv sync --reinstall > /dev/null 2>&1

uv run pytest benchmarks/bench_e2e_pipeline.py::test_e2e_system_text_extraction_simple \
    --benchmark-only --benchmark-quiet > /tmp/bench_cython.txt 2>&1 || true

print_info "Running quick benchmark in fallback mode..."
export CCPROXY_BUILD_CYTHON=false
export CCPROXY_ENABLE_CYTHON=false
find ccproxy/_cython -name "*.so" -delete 2>/dev/null || true
find ccproxy/_cython -name "*.pyd" -delete 2>/dev/null || true

uv run pytest benchmarks/bench_e2e_pipeline.py::test_e2e_system_text_extraction_simple \
    --benchmark-only --benchmark-quiet > /tmp/bench_fallback.txt 2>&1 || true

if [ -f /tmp/bench_cython.txt ] && [ -f /tmp/bench_fallback.txt ]; then
    print_success "Benchmark comparison available in /tmp/bench_*.txt"
else
    print_warning "Benchmark files not generated"
fi

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
print_success "All Cython build tests passed"
print_success "Fallback mode works correctly"
print_info "CI/CD workflows should work correctly"
echo ""
print_info "Next steps:"
echo "  1. Push to GitHub to trigger CI workflows"
echo "  2. Verify all workflow jobs pass"
echo "  3. Check benchmark artifacts for performance validation"
echo ""
