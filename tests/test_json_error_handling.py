"""Tests for JSON error handling and response validation."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from json.decoder import JSONDecodeError

from ccproxy.infrastructure.providers.resilience import (
    CircuitBreaker,
    RetryHandler,
    ResilientExecutor,
)
from ccproxy.infrastructure.providers.response_handlers import (
    ErrorResponseHandler,
    ResponseValidator,
)


class TestResponseValidator:
    """Test cases for ResponseValidator."""

    def test_validate_json_response_valid_json_string(self) -> None:
        """Test validation of valid JSON string."""
        valid_json = '{"message": "Hello, World!", "count": 42}'
        assert (
            ResponseValidator.validate_json_response(valid_json, "test response")
            is True
        )

    def test_validate_json_response_valid_json_bytes(self) -> None:
        """Test validation of valid JSON bytes."""
        valid_json = b'{"message": "Hello, World!", "count": 42}'
        assert (
            ResponseValidator.validate_json_response(valid_json, "test response")
            is True
        )

    def test_validate_json_response_invalid_json(self) -> None:
        """Test validation of invalid JSON."""
        invalid_json = '{"message": "unclosed object"'
        assert (
            ResponseValidator.validate_json_response(invalid_json, "test response")
            is False
        )

    def test_validate_json_response_empty_string(self) -> None:
        """Test validation of empty string."""
        assert ResponseValidator.validate_json_response("", "test response") is False

    def test_validate_json_response_none_value(self) -> None:
        """Test validation with None value."""
        with pytest.raises(ValueError, match="Response content must be str or bytes"):
            ResponseValidator.validate_json_response(None, "test response")  # type: ignore[arg-type]

    def test_validate_json_response_invalid_type(self) -> None:
        """Test validation with invalid type."""
        with pytest.raises(ValueError, match="Response content must be str or bytes"):
            ResponseValidator.validate_json_response(123, "test response")  # type: ignore[arg-type]

    def test_validate_json_response_malformed_utf8_bytes(self) -> None:
        """Test validation of malformed UTF-8 bytes."""
        malformed_bytes = b"\xff\xfe\x00\x00"  # Invalid UTF-8 sequence
        assert (
            ResponseValidator.validate_json_response(malformed_bytes, "test response")
            is False
        )

    def test_safe_parse_json_valid_json(self) -> None:
        """Test safe parsing of valid JSON."""
        valid_json = '{"message": "Hello", "data": [1, 2, 3]}'
        result = ResponseValidator.safe_parse_json(valid_json, "test response")
        expected = {"message": "Hello", "data": [1, 2, 3]}
        assert result == expected

    def test_safe_parse_json_invalid_json(self) -> None:
        """Test safe parsing of invalid JSON."""
        invalid_json = '{"message": "unclosed object"'
        result = ResponseValidator.safe_parse_json(invalid_json, "test response")
        assert result is None

    def test_safe_parse_json_malformed_bytes(self) -> None:
        """Test safe parsing of malformed bytes."""
        malformed_bytes = b"\xff\xfe\x00\x00"
        result = ResponseValidator.safe_parse_json(malformed_bytes, "test response")
        assert result is None

    def test_detect_json_corruption_patterns_empty_response(self) -> None:
        """Test detection of empty response pattern."""
        patterns = ResponseValidator.detect_json_corruption_patterns("")
        assert "empty_response" in patterns

    def test_detect_json_corruption_patterns_html_response(self) -> None:
        """Test detection of HTML response pattern."""
        html_response = "<!DOCTYPE html><html><body>Error</body></html>"
        patterns = ResponseValidator.detect_json_corruption_patterns(html_response)
        assert "html_response_instead_of_json" in patterns

    def test_detect_json_corruption_patterns_xml_response(self) -> None:
        """Test detection of XML response pattern."""
        xml_response = "<?xml version='1.0'?><error>Something went wrong</error>"
        patterns = ResponseValidator.detect_json_corruption_patterns(xml_response)
        assert "xml_response_instead_of_json" in patterns

    def test_detect_json_corruption_patterns_truncated_json(self) -> None:
        """Test detection of truncated JSON pattern."""
        truncated_json = '{"message": "test"'
        patterns = ResponseValidator.detect_json_corruption_patterns(truncated_json)
        assert "unmatched_braces_truncated_json" in patterns

    def test_detect_json_corruption_patterns_server_error(self) -> None:
        """Test detection of server error pattern."""
        error_response = "Error 500: Internal Server Error"
        patterns = ResponseValidator.detect_json_corruption_patterns(error_response)
        assert "server_error_response" in patterns

    def test_detect_json_corruption_patterns_short_response(self) -> None:
        """Test detection of response that's too short for valid JSON."""
        short_response = "abc"
        patterns = ResponseValidator.detect_json_corruption_patterns(short_response)
        assert "response_too_short_for_valid_json" in patterns

    def test_create_validation_error_response(self) -> None:
        """Test creation of validation error response."""
        original_error = JSONDecodeError("Expecting value", '{"invalid": ', 15)
        response_content = '{"invalid": '  # This will trigger unmatched_braces pattern
        context = "API response"

        error_response = ResponseValidator.create_validation_error_response(
            original_error, response_content, context
        )

        assert error_response["error"]["type"] == "json_validation_error"
        assert "Expecting value" in error_response["error"]["message"]
        assert error_response["error"]["original_error_type"] == "JSONDecodeError"
        assert error_response["error"]["validation_context"] == context
        assert "corruption_patterns" in error_response["error"]
        assert "response_preview" in error_response["error"]
        assert (
            "unmatched_braces_truncated_json"
            in error_response["error"]["corruption_patterns"]
        )

    def test_create_validation_error_response_without_content(self) -> None:
        """Test creation of validation error response without content."""
        original_error = JSONDecodeError("Expecting value", "", 0)

        error_response = ResponseValidator.create_validation_error_response(
            original_error, None, "test context"
        )

        assert error_response["error"]["type"] == "json_validation_error"
        assert "response_preview" not in error_response["error"]
        assert "corruption_patterns" not in error_response["error"]


class TestErrorResponseHandler:
    """Test cases for ErrorResponseHandler error classification."""

    def test_classify_error_json_decode_error(self) -> None:
        """Test classification of JSON decode error."""
        error = JSONDecodeError("Expecting value", '{"invalid": }', 15)
        category = ErrorResponseHandler.classify_error(error)
        assert category == "json_parse_error"

    def test_classify_error_json_decode_error_in_string(self) -> None:
        """Test classification of error with 'expecting value' in string."""
        error = ValueError("Expecting value: line 1 column 1 (char 0) in JSON")
        category = ErrorResponseHandler.classify_error(error)
        assert category == "json_parse_error"

    def test_should_retry_json_parse_error(self) -> None:
        """Test that JSON parse errors are retryable."""
        error = JSONDecodeError("Expecting value", '{"invalid": }', 15)
        should_retry = ErrorResponseHandler.should_retry(error)
        assert should_retry is True


class TestRetryHandler:
    """Test cases for RetryHandler with JSON errors."""

    @pytest.mark.anyio
    async def test_retry_handler_retries_json_decode_error(self) -> None:
        """Test that RetryHandler retries JSONDecodeError."""
        retry_handler = RetryHandler(max_retries=2, base_delay=0.01)

        call_count = 0

        async def failing_function() -> str:
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                raise JSONDecodeError("Expecting value", '{"invalid": }', 15)
            return "success"

        result = await retry_handler.execute_with_retry(failing_function)
        assert result == "success"
        assert call_count == 2

    @pytest.mark.anyio
    async def test_retry_handler_gives_up_after_max_retries_json_error(self) -> None:
        """Test that RetryHandler gives up after max retries for JSON errors."""
        retry_handler = RetryHandler(max_retries=2, base_delay=0.01)

        call_count = 0

        async def always_failing_function() -> str:
            nonlocal call_count
            call_count += 1
            raise JSONDecodeError("Expecting value", '{"invalid": }', 15)

        with pytest.raises(JSONDecodeError):
            await retry_handler.execute_with_retry(always_failing_function)

        assert call_count == 3  # Initial call + 2 retries

    @pytest.mark.anyio
    async def test_retry_handler_json_error_logging(self) -> None:
        """Test that JSON error retry attempts are logged."""
        retry_handler = RetryHandler(max_retries=1, base_delay=0.01)

        with patch(
            "ccproxy.infrastructure.providers.resilience.logging"
        ) as mock_logging:

            async def failing_function() -> str:
                raise JSONDecodeError("Expecting value", '{"invalid": }', 15)

            with pytest.raises(JSONDecodeError):
                await retry_handler.execute_with_retry(failing_function)

            # Check that debug logging was called for JSON decode error
            mock_logging.debug.assert_called()
            calls = mock_logging.debug.call_args_list
            json_retry_call = any(
                "JSON decode error, retrying" in str(call) for call in calls
            )
            assert json_retry_call is True


class TestCircuitBreaker:
    """Test cases for CircuitBreaker with JSON errors."""

    @pytest.mark.anyio
    async def test_circuit_breaker_triggers_on_json_errors(self) -> None:
        """Test that circuit breaker treats JSON errors as failures."""
        circuit_breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1)

        # First two JSON errors should open the circuit
        for _ in range(2):
            try:
                await circuit_breaker.call(self._failing_function_with_json_error)
            except JSONDecodeError:
                pass  # Expected

        # Circuit should be open
        assert circuit_breaker.is_open is True

        # Next call should be blocked
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            await circuit_breaker.call(self._successful_function)

    async def _failing_function_with_json_error(self) -> str:
        """Helper function that raises JSONDecodeError."""
        raise JSONDecodeError("Expecting value", '{"invalid": }', 15)

    async def _successful_function(self) -> str:
        """Helper function that returns successfully."""
        return "success"


class TestIntegration:
    """Integration tests for JSON error handling across components."""

    @pytest.mark.anyio
    async def test_end_to_end_json_error_retry_flow(self) -> None:
        """Test end-to-end flow of JSON error handling with retries."""
        from ccproxy.infrastructure.providers.resilience import ResilientExecutor

        resilient_executor = ResilientExecutor(
            circuit_breaker=CircuitBreaker(failure_threshold=3),
            retry_handler=RetryHandler(max_retries=2, base_delay=0.01),
        )

        call_count = 0

        async def simulated_openai_call() -> dict:
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                # Simulate JSON parsing error from OpenAI client
                raise JSONDecodeError("Expecting value", '{"invalid": }', 15)
            return {"choices": [{"message": {"content": "Success!"}}]}

        # Should succeed after retry
        result = await resilient_executor.execute(simulated_openai_call)
        assert result == {"choices": [{"message": {"content": "Success!"}}]}
        assert call_count == 2  # Initial call + 1 retry

    @pytest.mark.anyio
    async def test_response_validation_in_pipeline(self) -> None:
        """Test response validation within request pipeline."""
        from ccproxy.infrastructure.providers.request_pipeline import RequestPipeline
        from ccproxy.infrastructure.providers.response_handlers import ResponseProcessor

        # Mock dependencies
        mock_client = AsyncMock()
        mock_circuit_breaker = CircuitBreaker()
        mock_resilient_executor = ResilientExecutor()
        mock_rate_limiter = None
        mock_request_logger = MagicMock()
        mock_response_processor = ResponseProcessor()

        pipeline = RequestPipeline(
            client=mock_client,
            circuit_breaker=mock_circuit_breaker,
            resilient_executor=mock_resilient_executor,
            rate_limiter=mock_rate_limiter,
            request_logger=mock_request_logger,
            response_processor=mock_response_processor,
        )

        # Mock response with invalid JSON content
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = b'{"invalid": }'  # Invalid JSON
        mock_client.chat.completions.create.return_value = mock_response

        # Process request
        params = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "test"}],
        }
        response = await pipeline.process_request(params, "test-correlation-id")

        # Response should be returned even with validation warning
        assert response is not None

        # Mock client should have been called
        mock_client.chat.completions.create.assert_called_once_with(**params)
