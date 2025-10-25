# Tests - CLAUDE.md

**Scope**: Comprehensive test suite for CCProxy application

## Test Files (27 total):
- `conftest.py`: Pytest configuration and shared fixtures
- `test_tokenizer.py`: Tests for async-aware tokenization functionality with parallel token encoding using anyio.create_task_group
- `test_streaming.py`: Tests for SSE streaming and real-time responses
- `test_utf8.py`: UTF-8 handling and encoding validation tests
- `test_benchmarks.py`: Performance and benchmarking tests
- `test_error_tracker.py`: Comprehensive error tracking tests including async redaction and JSON serialization (18 test cases)
- `test_converters.py`: Message converter tests with full coverage (24 test cases)
- `test_cache.py`: Cache implementation tests including circuit breaker (31 test cases)
- `test_openai_provider.py`: OpenAI provider and HTTP/2 client tests (18 test cases)
- `test_routes.py`: HTTP routes and middleware tests
- `test_async_converters.py`: Async converter tests using Asyncer library with asyncify and parallel processing (8 test cases)
- `test_rate_limiter.py`: Tests for ClientRateLimiter with tiktoken-based token estimation integration (5 test cases)
- `test_cache_warmup.py`: Cache warmup manager tests for preloading and popularity tracking
- `test_thread_pool.py`: Thread pool management and CPU monitoring tests
- `test_type_utils.py`: Type utility function tests with Cython integration
- `test_guardrails.py`: Security guardrails and injection protection tests
- `test_request_validator.py`: Request validation and LRU cache tests
- `test_request_pipeline.py`: Request pipeline and lifecycle tests
- `test_request_lifecycle_observer.py`: Request lifecycle observation tests
- `test_integration.py`: End-to-end integration tests
- `test_health_monitoring_routes.py`: Health and monitoring endpoint tests
- `test_openai_provider_monitoring.py`: Provider metrics and monitoring tests
- `test_provider_components_factory.py`: Provider component factory tests
- `test_upstream_limits.py`: Upstream rate limiting tests
- `test_circuit_breaker.py`: Circuit breaker pattern tests
- `test_middleware.py`: HTTP middleware chain tests
- `test_model_selection.py`: Model mapping logic tests
- `test_errors.py`: Error handling and formatting tests
- `test_exceptions.py`: Exception type and hierarchy tests

## Testing Framework:
- **pytest**: Main testing framework (configured via pyproject.toml)
- **pytest-anyio**: For async function testing (migrating from pytest-asyncio)
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
