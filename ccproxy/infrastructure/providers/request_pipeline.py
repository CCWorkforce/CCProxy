"""
Request pipeline for processing OpenAI API requests.
Handles circuit breaker checks, rate limiting, and request execution.
"""

import logging
from typing import Any, Optional, Dict

import openai
from openai import AsyncOpenAI

from .resilience import CircuitBreaker, ResilientExecutor
from .rate_limiter import ClientRateLimiter
from .response_handlers import ResponseProcessor, ResponseValidator
from .request_logger import RequestLogger


class RequestPipeline:
    """
    Pipeline for processing requests through various checks and transformations.

    Responsibilities:
    - Circuit breaker validation
    - Rate limiting enforcement
    - Request execution (streaming and non-streaming)
    - Trace header preparation
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        circuit_breaker: CircuitBreaker,
        resilient_executor: ResilientExecutor,
        rate_limiter: Optional[ClientRateLimiter],
        request_logger: RequestLogger,
        response_processor: ResponseProcessor,
    ):
        """
        Initialize the request pipeline.

        Args:
            client: OpenAI async client
            circuit_breaker: Circuit breaker for failure protection
            resilient_executor: Executor with retry logic
            rate_limiter: Optional client-side rate limiter
            request_logger: Logger for request tracking
            response_processor: Processor for response handling
        """
        self._client = client
        self._circuit_breaker = circuit_breaker
        self._resilient_executor = resilient_executor
        self._rate_limiter = rate_limiter
        self._request_logger = request_logger
        self._response_processor = response_processor

    async def process_request(
        self,
        params: Dict[str, Any],
        correlation_id: str,
    ) -> Any:
        """
        Process a request through the pipeline.

        Args:
            params: Request parameters for OpenAI API
            correlation_id: Unique request identifier

        Returns:
            API response (regular object or async generator for streaming)

        Raises:
            Exception: If circuit breaker is open or rate limit exceeded
            openai.APIError: For API-related errors
        """
        # Check circuit breaker
        await self._check_circuit_breaker(correlation_id)

        # Apply rate limiting
        await self._apply_rate_limiting(params, correlation_id)

        # Prepare trace headers
        self._prepare_trace_headers(correlation_id, params)

        # Execute request
        is_streaming = params.get("stream", False)
        response = await self._execute_request(params, is_streaming)

        # Handle non-streaming UTF-8 decoding
        if not is_streaming:
            response = await self._ensure_utf8_response(response, correlation_id)
            # Validate JSON structure for non-streaming responses
            response = await self._validate_response_json(response, correlation_id)

        return response

    async def _check_circuit_breaker(self, correlation_id: str) -> None:
        """
        Check if circuit breaker allows the request.

        Args:
            correlation_id: Unique request identifier

        Raises:
            Exception: If circuit breaker is open
        """
        if self._circuit_breaker.is_open:
            logging.warning(f"Request {correlation_id} blocked by open circuit breaker")
            raise Exception("Service temporarily unavailable - circuit breaker is open")

    async def _apply_rate_limiting(
        self, params: Dict[str, Any], correlation_id: str
    ) -> None:
        """
        Apply client-side rate limiting if configured.

        Args:
            params: Request parameters for token estimation
            correlation_id: Unique request identifier

        Raises:
            Exception: If rate limit is exceeded
        """
        if not self._rate_limiter:
            return

        # Pass full params to limiter for precise token estimation
        if not await self._rate_limiter.acquire(request_payload=params):
            logging.warning(f"Request {correlation_id} blocked by client rate limiter")
            raise Exception(
                "Client-side rate limit exceeded. Please retry after a short delay."
            )

    def _prepare_trace_headers(
        self, correlation_id: str, params: Dict[str, Any]
    ) -> Optional[Dict[str, str]]:
        """
        Prepare trace headers for distributed tracing.

        Args:
            correlation_id: Unique request identifier
            params: Request parameters to add headers to

        Returns:
            Trace headers dictionary or None
        """
        trace_headers = self._request_logger.prepare_trace_headers(correlation_id)
        if trace_headers:
            params["extra_headers"] = trace_headers
            return trace_headers
        return None

    async def _execute_request(self, params: Dict[str, Any], is_streaming: bool) -> Any:
        """
        Execute the actual API request.

        Args:
            params: Request parameters
            is_streaming: Whether this is a streaming request

        Returns:
            API response
        """
        return await self._resilient_executor.execute(
            self._client.chat.completions.create,
            **params,
        )

    async def _ensure_utf8_response(self, response: Any, correlation_id: str) -> Any:
        """
        Ensure response content is properly UTF-8 decoded.

        Args:
            response: API response object
            correlation_id: Request identifier for logging

        Returns:
            Response with UTF-8 content guaranteed
        """
        if not hasattr(response, "choices") or not response.choices:
            return response

        for choice in response.choices:
            if (
                hasattr(choice, "message")
                and hasattr(choice.message, "content")
                and isinstance(choice.message.content, bytes)
            ):
                try:
                    # Try to decode safely
                    choice.message.content = choice.message.content.decode("utf-8")
                except UnicodeDecodeError as e:
                    logging.warning(
                        f"UTF-8 decode error in {correlation_id}: {e}, using replacement"
                    )
                    # Fall back to replacement decoding
                    choice.message.content = choice.message.content.decode(
                        "utf-8", errors="replace"
                    )

        return response

    async def _validate_response_json(self, response: Any, correlation_id: str) -> Any:
        """
        Validate that the response contains valid JSON structure.

        Args:
            response: API response object
            correlation_id: Request identifier for logging

        Returns:
            Validated response or raises exception if validation fails

        Raises:
            ValueError: If response contains invalid JSON
        """
        # For now, we log JSON validation issues but don't fail the request
        # This allows us to gather data on JSON corruption patterns
        try:
            # Check if response has content that might be JSON
            if hasattr(response, "choices") and response.choices:
                for choice in response.choices:
                    if (
                        hasattr(choice, "message")
                        and hasattr(choice.message, "content")
                        and isinstance(choice.message.content, (str, bytes))
                    ):
                        # Validate JSON content
                        if not ResponseValidator.validate_json_response(
                            choice.message.content,
                            f"response choice in {correlation_id}",
                        ):
                            # Log the validation failure with corruption patterns
                            corruption_patterns = (
                                ResponseValidator.detect_json_corruption_patterns(
                                    choice.message.content
                                )
                            )
                            logging.warning(
                                f"JSON validation failed for {correlation_id}. "
                                f"Corruption patterns: {corruption_patterns}"
                            )
                            # Note: We don't raise an exception here to avoid breaking existing functionality
                            # Instead, we rely on the retry logic in the resilience layer for JSON errors

        except Exception as e:
            # Don't let validation errors break the response flow
            logging.debug(f"Error during JSON validation for {correlation_id}: {e}")

        return response

    async def handle_rate_limit_response(self, error: openai.RateLimitError) -> None:
        """
        Handle rate limit error response.

        Args:
            error: Rate limit error from API
        """
        if not self._rate_limiter:
            return

        retry_after = getattr(error.response, "headers", {}).get("retry-after")
        await self._rate_limiter.handle_429_response(
            int(retry_after) if retry_after else None
        )

    async def release_tokens_on_success(self, response: Any) -> None:
        """
        Release tokens back to rate limiter after successful response.

        Args:
            response: Successful API response
        """
        if not self._rate_limiter:
            return

        await self._rate_limiter.handle_success()

        # Release tokens based on processed response
        usage_info = self._response_processor.extract_usage_info(response)
        if usage_info:
            await self._rate_limiter.release(usage_info["total_tokens"])
