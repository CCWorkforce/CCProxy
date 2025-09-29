# Tests - CLAUDE.md

**Scope**: Comprehensive test suite for CCProxy application

## Test Files:
- `conftest.py`: Pytest configuration and shared fixtures
- `test_tokenizer.py`: Tests for async-aware tokenization functionality
- `test_streaming.py`: Tests for SSE streaming and real-time responses
- `test_utf8.py`: UTF-8 handling and encoding validation tests
- `test_benchmarks.py`: Performance and benchmarking tests
- `test_error_tracker.py`: Comprehensive error tracking tests (18 test cases)
- `test_converters.py`: Message converter tests with full coverage (24 test cases)
- `test_cache.py`: Cache implementation tests including circuit breaker (31 test cases)
- `test_openai_provider.py`: OpenAI provider and HTTP/2 client tests (18 test cases)
- `test_routes.py`: HTTP routes and middleware tests
- `test_async_converters.py`: Async converter tests using Asyncer library (8 test cases)
- `test_cache_warmup.py`: Cache warmup manager tests (8 test cases)

## Testing Framework:
- **pytest**: Main testing framework (configured via pyproject.toml)
- **pytest-asyncio**: For async function testing
- **respx**: For httpx mocking and HTTP simulation

## Guidelines:
- **Environment**: Tests assume Settings defaults and test-specific configurations
- **Network isolation**: Avoid hitting external networks; use mocking instead
- **Async testing**: Tokenizer tests must be async due to async locks implementation
- **Mocking**: Use AsyncMock/MagicMock for external dependencies
- **Deterministic data**: Prefer deterministic test data over random values
- **Logging**: Initialize logging if needed for test scenarios
- **Performance**: Include benchmark tests for critical performance paths
- **Coverage**: Comprehensive test coverage with 120+ test cases
- **Type safety**: Tests validate type annotations and strict mypy compliance

## Running Tests:
- Test runner script: `./run-tests.sh` (uses uv for virtual environment)
- All tests with uv: `uv run pytest -q`
- With coverage: `./run-tests.sh --coverage`
- Parallel execution: `./run-tests.sh --parallel`
- Watch mode: `./run-tests.sh --watch`
- Single test file: `uv run pytest -q test_specific_file.py`
- Specific test: `uv run pytest -q test_file.py::test_function_name`
- Verbose output: `uv run pytest -xvs test_file.py`
