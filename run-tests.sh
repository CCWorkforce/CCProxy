#!/bin/bash

# CCProxy - Test Runner Script
# This script runs all unit tests using pytest with uv for virtual environment management

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
TEST_DIR="tests"
PYTEST_CONFIG_FILE="pyproject.toml"
MIN_COVERAGE=70  # Minimum coverage percentage required

echo -e "${BLUE}🧪 CCProxy - Test Runner${NC}"
echo -e "${BLUE}=====================================${NC}"
echo ""

# Function to check if uv is installed and install pytest if needed
check_pytest() {
    # Check if uv is available
    if ! command -v uv &> /dev/null; then
        echo -e "${RED}❌ uv is not installed${NC}"
        echo -e "${YELLOW}Please install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh${NC}"
        exit 1
    fi

    # Use uv to ensure pytest is available
    if ! uv run pytest --version &> /dev/null; then
        echo -e "${YELLOW}📦 Installing pytest via uv...${NC}"
        uv add --dev pytest pytest-asyncio pytest-cov pytest-mock respx
    fi

    echo -e "${GREEN}✅ pytest is available via uv${NC}"
    echo -e "${CYAN}uv version: $(uv --version)${NC}"
    echo -e "${CYAN}pytest version: $(uv run pytest --version)${NC}"
}

# Function to find test files
find_test_files() {
    echo -e "${BLUE}🔍 Finding test files...${NC}"

    # Find all test files
    TEST_FILES=$(find "$TEST_DIR" -name "test_*.py" -o -name "*_test.py" | sort)

    if [ -z "$TEST_FILES" ]; then
        echo -e "${YELLOW}⚠️  No test files found in $TEST_DIR${NC}"
        exit 1
    fi

    FILE_COUNT=$(echo "$TEST_FILES" | wc -l | tr -d ' ')
    echo -e "${GREEN}✅ Found $FILE_COUNT test files${NC}"
    echo ""
}

# Function to show pytest configuration
show_config() {
    echo -e "${BLUE}⚙️  Pytest Configuration:${NC}"

    if [ -f "$PYTEST_CONFIG_FILE" ]; then
        echo -e "${GREEN}✅ Using configuration from $PYTEST_CONFIG_FILE${NC}"

        # Show relevant pytest configuration if it exists
        if grep -q "\[tool.pytest.ini_options\]" "$PYTEST_CONFIG_FILE" 2>/dev/null; then
            echo -e "${CYAN}Configuration preview:${NC}"
            sed -n '/\[tool\.pytest\.ini_options\]/,/^\[/p' "$PYTEST_CONFIG_FILE" | head -10
        fi
    else
        echo -e "${YELLOW}⚠️  No pytest configuration found, using defaults${NC}"
    fi
    echo ""
}

# Function to run tests with basic output
run_tests_basic() {
    echo -e "${BLUE}🧪 Running tests...${NC}"
    echo ""

    if uv run pytest "$TEST_DIR" -q --tb=short; then
        echo ""
        echo -e "${GREEN}✅ All tests passed!${NC}"
        return 0
    else
        echo ""
        echo -e "${RED}❌ Some tests failed${NC}"
        return 1
    fi
}

# Function to run tests with verbose output
run_tests_verbose() {
    echo -e "${BLUE}🧪 Running tests (verbose)...${NC}"
    echo ""

    uv run pytest "$TEST_DIR" -v --tb=long
}

# Function to run tests with coverage
run_tests_coverage() {
    echo -e "${BLUE}📊 Running tests with coverage...${NC}"
    echo ""

    # Run pytest with coverage
    if uv run pytest "$TEST_DIR" \
        --cov=ccproxy \
        --cov-report=term-missing \
        --cov-report=html \
        --cov-fail-under=$MIN_COVERAGE; then
        echo ""
        echo -e "${GREEN}✅ Tests passed with sufficient coverage (>=$MIN_COVERAGE%)${NC}"
        echo -e "${CYAN}Coverage report saved to htmlcov/index.html${NC}"
        return 0
    else
        echo ""
        if [ $? -eq 5 ]; then
            echo -e "${RED}❌ Coverage is below $MIN_COVERAGE%${NC}"
        else
            echo -e "${RED}❌ Tests failed${NC}"
        fi
        return 1
    fi
}

# Function to run specific test file or test
run_specific_test() {
    local test_path="$1"

    echo -e "${BLUE}🧪 Running specific test: $test_path${NC}"
    echo ""

    if uv run pytest "$test_path" -v; then
        echo ""
        echo -e "${GREEN}✅ Test passed!${NC}"
        return 0
    else
        echo ""
        echo -e "${RED}❌ Test failed${NC}"
        return 1
    fi
}

# Function to run tests in parallel
run_tests_parallel() {
    echo -e "${BLUE}⚡ Running tests in parallel...${NC}"
    echo ""

    # Check if pytest-xdist is installed
    if ! uv run pytest --version 2>&1 | grep -q "pytest-xdist"; then
        echo -e "${YELLOW}📦 Installing pytest-xdist for parallel execution...${NC}"
        uv add --dev pytest-xdist
    fi

    # Run tests in parallel using all available CPU cores
    if uv run pytest "$TEST_DIR" -n auto -q; then
        echo ""
        echo -e "${GREEN}✅ All tests passed (parallel execution)!${NC}"
        return 0
    else
        echo ""
        echo -e "${RED}❌ Some tests failed${NC}"
        return 1
    fi
}

# Function to run tests and watch for changes
run_tests_watch() {
    echo -e "${BLUE}👁️  Running tests in watch mode...${NC}"
    echo ""

    # Check if pytest-watch is installed
    if ! uv run ptw --version &> /dev/null; then
        echo -e "${YELLOW}📦 Installing pytest-watch...${NC}"
        uv add --dev pytest-watch
    fi

    echo -e "${CYAN}Watching for file changes... (Press Ctrl+C to stop)${NC}"
    uv run ptw "$TEST_DIR" -- -q
}

# Function to run only failed tests from last run
run_failed_tests() {
    echo -e "${BLUE}🔧 Running previously failed tests...${NC}"
    echo ""

    if uv run pytest "$TEST_DIR" --lf -v; then
        echo ""
        echo -e "${GREEN}✅ All previously failed tests now pass!${NC}"
        return 0
    else
        echo ""
        echo -e "${RED}❌ Some tests still failing${NC}"
        return 1
    fi
}

# Function to list available tests
list_tests() {
    echo -e "${BLUE}📁 Available test files:${NC}"
    echo ""

    find_test_files

    echo -e "${CYAN}Test files:${NC}"
    echo "$TEST_FILES" | while read -r file; do
        if [ -n "$file" ]; then
            # Count tests in file
            test_count=$(grep -c "def test_\|async def test_" "$file" || echo "0")
            echo "  $file (${test_count} tests)"
        fi
    done

    echo ""
    total_tests=$(grep -h "def test_\|async def test_" $TEST_FILES | wc -l | tr -d ' ')
    echo -e "${GREEN}Total: $FILE_COUNT files, $total_tests tests${NC}"
}

# Function to generate test report
generate_report() {
    echo -e "${BLUE}📄 Generating test report...${NC}"
    echo ""

    # Create reports directory
    mkdir -p reports

    # Run tests with multiple output formats
    uv run pytest "$TEST_DIR" \
        --junitxml=reports/junit.xml \
        --html=reports/test-report.html \
        --self-contained-html \
        --cov=ccproxy \
        --cov-report=html:reports/coverage \
        --cov-report=xml:reports/coverage.xml \
        --cov-report=term \
        -v

    echo ""
    echo -e "${GREEN}✅ Reports generated:${NC}"
    echo -e "${CYAN}  • HTML Test Report: reports/test-report.html${NC}"
    echo -e "${CYAN}  • Coverage Report: reports/coverage/index.html${NC}"
    echo -e "${CYAN}  • JUnit XML: reports/junit.xml${NC}"
    echo -e "${CYAN}  • Coverage XML: reports/coverage.xml${NC}"
}

# Function to run benchmark tests
run_benchmarks() {
    echo -e "${BLUE}⏱️  Running benchmark tests...${NC}"
    echo ""

    # Run only benchmark tests
    if uv run pytest "$TEST_DIR" -k benchmark -v; then
        echo ""
        echo -e "${GREEN}✅ Benchmark tests completed${NC}"
    else
        echo ""
        echo -e "${YELLOW}⚠️  Some benchmark tests failed${NC}"
    fi
}

# Function to show detailed help
show_help() {
    echo "Usage: $0 [OPTIONS] [TEST_PATH]"
    echo ""
    echo "Options:"
    echo "  --all         Run all tests with basic output (default)"
    echo "  --verbose     Run tests with verbose output"
    echo "  --coverage    Run tests with coverage report"
    echo "  --parallel    Run tests in parallel"
    echo "  --watch       Watch for changes and re-run tests"
    echo "  --failed      Run only previously failed tests"
    echo "  --benchmark   Run only benchmark tests"
    echo "  --report      Generate detailed HTML and XML reports"
    echo "  --list        List all available test files"
    echo "  --config      Show pytest configuration"
    echo "  --help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                           # Run all tests"
    echo "  $0 --coverage               # Run with coverage report"
    echo "  $0 --parallel               # Run tests in parallel"
    echo "  $0 tests/test_cache.py      # Run specific test file"
    echo "  $0 tests/test_cache.py::TestResponseCache::test_cache_initialization"
    echo "                              # Run specific test"
    echo ""
    echo "Environment Variables:"
    echo "  PYTEST_ARGS    Additional arguments to pass to pytest"
    echo "  MIN_COVERAGE   Minimum coverage percentage (default: 70)"
    echo ""
    echo "Quick Test Selection:"
    echo "  Unit tests:        pytest -k 'not integration and not benchmark'"
    echo "  Integration tests: pytest -k integration"
    echo "  Benchmark tests:   pytest -k benchmark"
}

# Main execution
main() {
    # Parse command line arguments
    ACTION="all"
    TEST_PATH=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            --all)
                ACTION="all"
                shift
                ;;
            --verbose|-v)
                ACTION="verbose"
                shift
                ;;
            --coverage)
                ACTION="coverage"
                shift
                ;;
            --parallel)
                ACTION="parallel"
                shift
                ;;
            --watch)
                ACTION="watch"
                shift
                ;;
            --failed|--lf)
                ACTION="failed"
                shift
                ;;
            --benchmark)
                ACTION="benchmark"
                shift
                ;;
            --report)
                ACTION="report"
                shift
                ;;
            --list)
                list_tests
                exit 0
                ;;
            --config)
                check_pytest
                show_config
                exit 0
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            --*)
                echo -e "${RED}Unknown option: $1${NC}"
                echo "Use --help for usage information"
                exit 1
                ;;
            *)
                # Assume it's a test path
                TEST_PATH="$1"
                ACTION="specific"
                shift
                ;;
        esac
    done

    # Execute main workflow
    check_pytest

    # If running specific test, don't need to find all files
    if [ "$ACTION" != "specific" ]; then
        find_test_files
        show_config
    fi

    case $ACTION in
        "all")
            if run_tests_basic; then
                echo -e "${GREEN}🎉 All tests passed successfully${NC}"
                exit 0
            else
                echo -e "${YELLOW}💡 Run with --verbose for detailed output${NC}"
                exit 1
            fi
            ;;
        "verbose")
            run_tests_verbose
            ;;
        "coverage")
            if run_tests_coverage; then
                echo -e "${GREEN}🎉 Tests completed with coverage${NC}"
                exit 0
            else
                echo -e "${YELLOW}💡 Check coverage report for details${NC}"
                exit 1
            fi
            ;;
        "parallel")
            if run_tests_parallel; then
                echo -e "${GREEN}🎉 All tests passed (parallel)${NC}"
                exit 0
            else
                exit 1
            fi
            ;;
        "watch")
            run_tests_watch
            ;;
        "failed")
            if run_failed_tests; then
                echo -e "${GREEN}🎉 Previously failed tests now pass${NC}"
                exit 0
            else
                exit 1
            fi
            ;;
        "benchmark")
            run_benchmarks
            ;;
        "report")
            generate_report
            echo -e "${GREEN}🎉 Test reports generated${NC}"
            ;;
        "specific")
            if [ -z "$TEST_PATH" ]; then
                echo -e "${RED}❌ No test path specified${NC}"
                exit 1
            fi
            if run_specific_test "$TEST_PATH"; then
                echo -e "${GREEN}🎉 Test passed${NC}"
                exit 0
            else
                exit 1
            fi
            ;;
    esac
}

# Handle Ctrl+C gracefully
trap 'echo -e "\n${YELLOW}Tests interrupted${NC}"; exit 1' INT

# Run main function
main "$@"