# Tests - CLAUDE.md

**Scope**: Comprehensive test suite for CCProxy application

## Test Files:
- `conftest.py`: Pytest configuration and shared fixtures
- `test_tokenizer.py`: Tests for async-aware tokenization functionality
- `test_streaming.py`: Tests for SSE streaming and real-time responses
- `test_utf8.py`: UTF-8 handling and encoding validation tests
- `test_benchmarks.py`: Performance and benchmarking tests

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

## Running Tests:
- All tests: `pytest -q`
- Single test file: `pytest -q test_specific_file.py`
- Specific test: `pytest -q test_file.py::test_function_name`
