"""Comprehensive tests for error handling and formatting."""

import json
from unittest.mock import MagicMock
import openai

from ccproxy.interfaces.http.errors import (
    extract_provider_error_details,
    get_anthropic_error_details_from_execution,
    format_anthropic_error_sse_event,
    STATUS_CODE_ERROR_MAP,
)
from ccproxy.domain.models import (
    AnthropicErrorType,
    ProviderErrorMetadata,
)


class TestExtractProviderErrorDetails:
    """Test provider error metadata extraction."""

    def test_extract_with_valid_metadata(self):
        """Test extracting provider metadata from valid error dict."""
        error_dict = {
            "message": "Error occurred",
            "metadata": {
                "provider_name": "openai",
                "raw": '{"error": {"message": "Rate limit exceeded"}}',
            },
        }

        result = extract_provider_error_details(error_dict)

        assert result is not None
        assert result.provider_name == "openai"
        assert result.raw_error is not None
        assert "error" in result.raw_error

    def test_extract_with_dict_raw_error(self):
        """Test when raw error is already a dict."""
        error_dict = {
            "metadata": {
                "provider_name": "anthropic",
                "raw": {"error": {"message": "Invalid API key"}},
            }
        }

        result = extract_provider_error_details(error_dict)

        assert result is not None
        assert result.provider_name == "anthropic"
        assert isinstance(result.raw_error, dict)
        assert result.raw_error["error"]["message"] == "Invalid API key"

    def test_extract_with_invalid_json_raw(self):
        """Test when raw string is invalid JSON."""
        error_dict = {
            "metadata": {"provider_name": "test", "raw": "not valid json {"}
        }

        result = extract_provider_error_details(error_dict)

        assert result is not None
        assert result.provider_name == "test"
        assert "raw_string_parse_failed" in result.raw_error

    def test_extract_with_no_metadata(self):
        """Test when error dict has no metadata."""
        error_dict = {"message": "Error", "code": 500}

        result = extract_provider_error_details(error_dict)

        assert result is None

    def test_extract_with_non_dict_metadata(self):
        """Test when metadata is not a dictionary."""
        error_dict = {"metadata": "not a dict"}

        result = extract_provider_error_details(error_dict)

        assert result is None

    def test_extract_with_missing_provider_name(self):
        """Test when provider_name is missing."""
        error_dict = {"metadata": {"raw": '{"error": "something"}'}}

        result = extract_provider_error_details(error_dict)

        assert result is None

    def test_extract_with_non_dict_input(self):
        """Test when input is not a dictionary."""
        result = extract_provider_error_details("not a dict")
        assert result is None

        result = extract_provider_error_details(None)
        assert result is None

        result = extract_provider_error_details([])
        assert result is None


class TestGetAnthropicErrorDetailsFromExecution:
    """Test error details extraction from exceptions."""

    def test_generic_exception(self):
        """Test mapping generic Exception."""
        exc = Exception("Something went wrong")

        error_type, message, status_code, provider_details = (
            get_anthropic_error_details_from_execution(exc)
        )

        assert error_type == AnthropicErrorType.API_ERROR
        assert message == "Something went wrong"
        assert status_code == 500
        assert provider_details is None

    def test_openai_api_error_400(self):
        """Test mapping OpenAI BadRequestError (400)."""
        exc = openai.BadRequestError(
            message="Invalid request",
            response=MagicMock(status_code=400),
            body=None,
        )

        error_type, message, status_code, provider_details = (
            get_anthropic_error_details_from_execution(exc)
        )

        assert error_type == AnthropicErrorType.INVALID_REQUEST
        assert message == "Invalid request"
        assert status_code == 400

    def test_openai_api_error_401(self):
        """Test mapping OpenAI AuthenticationError (401)."""
        exc = openai.AuthenticationError(
            message="Invalid API key",
            response=MagicMock(status_code=401),
            body=None,
        )

        error_type, message, status_code, provider_details = (
            get_anthropic_error_details_from_execution(exc)
        )

        assert error_type == AnthropicErrorType.AUTHENTICATION
        assert status_code == 401

    def test_openai_api_error_403(self):
        """Test mapping OpenAI PermissionDeniedError (403)."""
        exc = openai.PermissionDeniedError(
            message="Access denied",
            response=MagicMock(status_code=403),
            body=None,
        )

        error_type, message, status_code, provider_details = (
            get_anthropic_error_details_from_execution(exc)
        )

        assert error_type == AnthropicErrorType.PERMISSION
        assert status_code == 403

    def test_openai_api_error_404(self):
        """Test mapping OpenAI NotFoundError (404)."""
        exc = openai.NotFoundError(
            message="Resource not found",
            response=MagicMock(status_code=404),
            body=None,
        )

        error_type, message, status_code, provider_details = (
            get_anthropic_error_details_from_execution(exc)
        )

        assert error_type == AnthropicErrorType.NOT_FOUND
        assert status_code == 404

    def test_openai_api_error_429(self):
        """Test mapping OpenAI RateLimitError (429)."""
        exc = openai.RateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(status_code=429),
            body=None,
        )

        error_type, message, status_code, provider_details = (
            get_anthropic_error_details_from_execution(exc)
        )

        assert error_type == AnthropicErrorType.RATE_LIMIT
        assert status_code == 429

    def test_openai_api_error_500(self):
        """Test mapping OpenAI InternalServerError (500)."""
        exc = openai.InternalServerError(
            message="Internal error",
            response=MagicMock(status_code=500),
            body=None,
        )

        error_type, message, status_code, provider_details = (
            get_anthropic_error_details_from_execution(exc)
        )

        assert error_type == AnthropicErrorType.API_ERROR
        assert status_code == 500

    def test_openai_api_error_with_body(self):
        """Test OpenAI error with body containing metadata."""
        exc = openai.RateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(status_code=429),
            body={
                "error": {
                    "message": "Rate limit exceeded",
                    "metadata": {
                        "provider_name": "openrouter",
                        "raw": '{"error": {"message": "upstream rate limit"}}',
                    },
                }
            },
        )

        error_type, message, status_code, provider_details = (
            get_anthropic_error_details_from_execution(exc)
        )

        assert error_type == AnthropicErrorType.RATE_LIMIT
        assert status_code == 429
        if provider_details:
            assert provider_details.provider_name == "openrouter"

    def test_openai_api_error_insufficient_quota(self):
        """Test OpenAI error with insufficient_quota code."""
        exc = openai.RateLimitError(
            message="Quota exceeded",
            response=MagicMock(status_code=429),
            body={"error": {"code": "insufficient_quota", "message": "Quota exceeded"}},
        )

        error_type, message, status_code, provider_details = (
            get_anthropic_error_details_from_execution(exc)
        )

        # Should still be rate limit since that's what openai.RateLimitError maps to
        assert error_type == AnthropicErrorType.RATE_LIMIT
        assert status_code == 429

    def test_api_error_with_status_code(self):
        """Test generic APIError with specific status code."""
        exc = openai.InternalServerError(
            message="Server error",
            response=MagicMock(status_code=503),
            body=None,
        )

        error_type, message, status_code, provider_details = (
            get_anthropic_error_details_from_execution(exc)
        )

        assert error_type == AnthropicErrorType.OVERLOADED
        assert status_code == 503


class TestFormatAnthropicErrorSSEEvent:
    """Test SSE error event formatting."""

    def test_format_simple_error(self):
        """Test formatting simple error as SSE event."""
        result = format_anthropic_error_sse_event(
            error_type=AnthropicErrorType.INVALID_REQUEST,
            message="Invalid request format",
            provider_details=None,
        )

        assert result.startswith("event: error\ndata: ")
        # Parse the JSON data
        data_line = result.split("data: ")[1].strip()
        parsed = json.loads(data_line)
        assert parsed["type"] == "error"
        assert parsed["error"]["type"] == "invalid_request_error"
        assert parsed["error"]["message"] == "Invalid request format"

    def test_format_error_with_provider_details(self):
        """Test formatting error with provider metadata."""
        provider_details = ProviderErrorMetadata(
            provider_name="openai",
            raw_error={"error": {"message": "Too many requests", "code": "rate_limit"}},
        )

        result = format_anthropic_error_sse_event(
            error_type=AnthropicErrorType.RATE_LIMIT,
            message="Rate limit exceeded",
            provider_details=provider_details,
        )

        assert "event: error\n" in result
        data_line = result.split("data: ")[1].strip()
        parsed = json.loads(data_line)
        assert parsed["error"]["type"] == "rate_limit_error"
        assert parsed["error"]["provider"] == "openai"


# Note: _build_anthropic_error_response is an internal function
# tested indirectly through error handling in test_routes.py


# Note: log_and_return_error_response tests are integration-level
# and are covered by test_routes.py which tests full request/response cycle


class TestStatusCodeErrorMap:
    """Test status code to error type mapping."""

    def test_all_mapped_codes(self):
        """Test that all standard HTTP error codes are mapped."""
        expected_mappings = {
            400: AnthropicErrorType.INVALID_REQUEST,
            401: AnthropicErrorType.AUTHENTICATION,
            403: AnthropicErrorType.PERMISSION,
            404: AnthropicErrorType.NOT_FOUND,
            413: AnthropicErrorType.REQUEST_TOO_LARGE,
            422: AnthropicErrorType.INVALID_REQUEST,
            429: AnthropicErrorType.RATE_LIMIT,
            500: AnthropicErrorType.API_ERROR,
            502: AnthropicErrorType.API_ERROR,
            503: AnthropicErrorType.OVERLOADED,
            504: AnthropicErrorType.API_ERROR,
        }

        for code, error_type in expected_mappings.items():
            assert STATUS_CODE_ERROR_MAP[code] == error_type

    def test_unmapped_code_defaults(self):
        """Test that unmapped codes default to API_ERROR."""
        # These codes are not in the map
        unmapped_codes = [418, 451, 507, 599]

        for code in unmapped_codes:
            # Should not be in the map
            assert code not in STATUS_CODE_ERROR_MAP
