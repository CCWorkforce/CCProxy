"""
Optimized OpenAI provider with high-performance HTTP client configuration.
Supports multiple HTTP client backends for maximum performance.
"""

import json
import logging
import time
from typing import Any, Optional, Dict, List

import openai
from openai import AsyncOpenAI

from ...config import Settings
from ...application.error_tracker import ErrorType

from .provider_components_factory import ProviderComponentsFactory
from .request_pipeline import RequestPipeline
from .request_lifecycle_observer import RequestLifecycleObserver
from .http_client_factory import HttpClientFactory
from .metrics import ProviderMetrics


class OpenAIProvider:
    """
    High-performance OpenAI provider with monitoring and resilience.

    Key features:
    - HTTP/2 support for multiplexing
    - Circuit breaker for failure protection
    - Comprehensive metrics collection
    - Request/response logging with correlation IDs
    - Adaptive timeout based on performance
    - Health monitoring and scoring

    This class has been refactored to use modular components for better
    maintainability and testability.
    """

    def __init__(self, settings: Settings) -> None:
        """
        Initialize the OpenAI provider with modular components.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self._openai_client: Optional[AsyncOpenAI] = None
        self._http_client = None

        # Create components using factory
        self._resilience = ProviderComponentsFactory.create_resilience_components(
            settings
        )
        self._monitoring = ProviderComponentsFactory.create_monitoring_components(
            settings
        )
        self._logging = ProviderComponentsFactory.create_logging_components()
        self._processing = ProviderComponentsFactory.create_processing_components()
        self._rate_limiter = ProviderComponentsFactory.create_rate_limiter(settings)

        # Initialize HTTP and OpenAI clients
        self._initialize_clients()

        # Create request pipeline
        self._pipeline = RequestPipeline(
            client=self._openai_client,  # type: ignore[arg-type]
            circuit_breaker=self._resilience.circuit_breaker,
            resilient_executor=self._resilience.resilient_executor,
            rate_limiter=self._rate_limiter,
            request_logger=self._logging.request_logger,
            response_processor=self._processing.response_processor,
        )

        # Create lifecycle observer
        self._lifecycle_observer = RequestLifecycleObserver(
            performance_monitor=self._monitoring.performance_monitor,
            performance_tracker=self._logging.performance_tracker,
            metrics_collector=self._monitoring.metrics_collector,
            error_tracker=self._monitoring.error_tracker,
            circuit_breaker=self._resilience.circuit_breaker,
            request_logger=self._logging.request_logger,
            response_processor=self._processing.response_processor,
        )

    def _initialize_clients(self) -> None:
        """Initialize HTTP and OpenAI clients."""
        try:
            # Create HTTP client using factory
            self._http_client = HttpClientFactory.create_client(self.settings)  # type: ignore[assignment]
            HttpClientFactory.log_client_configuration(self._http_client, self.settings)  # type: ignore[arg-type]

            # Initialize OpenAI client with the configured HTTP client
            self._openai_client = AsyncOpenAI(
                api_key=self.settings.openai_api_key,
                base_url=self.settings.base_url,
                default_headers=HttpClientFactory.get_default_headers(self.settings),
                timeout=float(self.settings.max_stream_seconds),
                http_client=self._http_client,
                max_retries=self.settings.provider_max_retries,
            )
        except Exception:
            # Clean up if initialization fails
            self._http_client = None
            raise

    async def create_chat_completion(self, **params: Any) -> Any:
        """
        Create a chat completion with monitoring and resilience.

        Args:
            **params: Parameters to pass to the OpenAI chat completion API

        Returns:
            The API response from OpenAI (regular object for non-streaming,
            async generator for streaming)

        Raises:
            openai.APIError: If there are issues with the API call after retries
            Exception: If circuit breaker is open or rate limit exceeded
        """
        if not self._openai_client:
            raise ValueError("OpenAI client not properly initialized")

        # Generate correlation ID
        correlation_id = self._logging.request_logger.generate_correlation_id()
        request_start = time.monotonic()

        # Extract trace context
        trace_headers = self._logging.request_logger.prepare_trace_headers(
            correlation_id
        )
        trace_id = trace_headers.get("X-Trace-ID") if trace_headers else None

        # Start lifecycle tracking
        await self._lifecycle_observer.on_request_start(
            correlation_id, params, trace_id
        )

        try:
            # Process request through pipeline
            response = await self._pipeline.process_request(params, correlation_id)

            # Handle streaming case
            if params.get("stream", False):
                await self._lifecycle_observer.on_success(correlation_id, request_start)
                return response

            # Handle non-streaming case
            await self._lifecycle_observer.on_success(
                correlation_id, request_start, response
            )
            await self._pipeline.release_tokens_on_success(response)

            return response

        except UnicodeDecodeError as e:
            # Handle Unicode errors
            await self._lifecycle_observer.on_failure(
                correlation_id, request_start, e, ErrorType.CONVERSION_ERROR
            )
            logging.warning(
                f"Recovered from decode error in {correlation_id} with partial replacement"
            )
            raise openai.APIError(  # type: ignore[call-arg]
                message=f"Partial decode recovery applied: {str(e)}",
                status_code=500,
                body={
                    "error": {
                        "message": "Response partially recovered from encoding error"
                    }
                },
            ) from e

        except openai.RateLimitError as e:
            await self._lifecycle_observer.on_failure(
                correlation_id, request_start, e, ErrorType.RATE_LIMIT_ERROR
            )
            await self._pipeline.handle_rate_limit_response(e)
            raise

        except openai.AuthenticationError as e:
            await self._lifecycle_observer.on_failure(
                correlation_id, request_start, e, ErrorType.AUTH_ERROR
            )
            raise

        except Exception as e:
            # Classify and handle other errors
            error_type = self._lifecycle_observer.classify_error(e)
            await self._lifecycle_observer.on_failure(
                correlation_id, request_start, e, error_type
            )

            if error_type == ErrorType.CONVERSION_ERROR:
                raise ValueError(
                    f"Received malformed response from API that could not be decoded as UTF-8: {str(e)}"
                ) from e
            raise

    async def get_metrics(self) -> ProviderMetrics:
        """
        Get current metrics snapshot.

        Returns:
            Current provider metrics
        """
        metrics = await self._monitoring.metrics_collector.get_snapshot()
        await self._monitoring.metrics_collector.update_circuit_state(
            self._resilience.circuit_breaker.state
        )
        return metrics

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check and return status.

        Returns:
            Health check status dictionary
        """
        return await self._monitoring.health_monitor.get_health_check(
            circuit_state=self._resilience.circuit_breaker.state,
            active_requests=self._monitoring.performance_monitor.metrics.active_requests,
        )

    def get_adaptive_timeout(self) -> float:
        """
        Calculate adaptive timeout based on recent performance.

        Returns:
            Calculated timeout in seconds
        """
        return self._monitoring.timeout_calculator.calculate_timeout()

    async def _estimate_tokens(
        self, messages: List[Dict[str, Any]], model_name: Optional[str] = None
    ) -> int:
        """
        Accurately count tokens for messages using tiktoken.

        Args:
            messages: List of message dictionaries
            model_name: Optional model name for accurate encoding

        Returns:
            Accurate token count
        """
        if not messages:
            return 0

        try:
            # Import tokenizer function
            from ...application.tokenizer import count_tokens_for_openai_request

            # Use the provided model or default from settings
            model = model_name or self.settings.big_model_name

            # Get accurate token count
            token_count = await count_tokens_for_openai_request(
                messages=messages,
                model_name=model,
                request_id=None,  # Could pass correlation_id if available
            )

            return token_count

        except Exception as e:
            logging.warning(
                f"Failed to count tokens accurately, falling back to rough estimate: {e}"
            )

            # Fallback to rough estimation: ~4 characters per token
            total_chars = 0
            for message in messages:
                content = message.get("content", "")
                if isinstance(content, str):
                    total_chars += len(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict):
                            if part.get("type") == "text":
                                total_chars += len(part.get("text", ""))
                            else:
                                total_chars += len(json.dumps(part, ensure_ascii=False))
                        else:
                            total_chars += len(str(part))
                elif content is not None:
                    total_chars += len(str(content))

                tool_calls = message.get("tool_calls") or []
                if tool_calls:
                    total_chars += len(json.dumps(tool_calls, ensure_ascii=False))

                function_call = message.get("function_call")
                if function_call:
                    total_chars += len(json.dumps(function_call, ensure_ascii=False))

            # Add overhead for message structure
            total_chars += len(messages) * 10

            return total_chars // 4

    async def close(self) -> Any:
        """Clean up resources when shutting down."""
        # Stop rate limiter if running
        if self._rate_limiter:
            try:
                await self._rate_limiter.stop()
                logging.info("Rate limiter stopped successfully")
            except Exception as e:
                logging.error(f"Error stopping rate limiter: {e}")

        # Close HTTP client using factory method
        await HttpClientFactory.close_client(self._http_client)
        self._http_client = None
        self._openai_client = None

    async def __aenter__(self) -> "OpenAIProvider":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()
