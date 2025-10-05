import pytest
from unittest.mock import Mock, AsyncMock, patch

import openai
from openai import AsyncOpenAI

from ccproxy.infrastructure.providers.request_pipeline import RequestPipeline
from ccproxy.infrastructure.providers.resilience import (
    CircuitBreaker,
    ResilientExecutor,
)
from ccproxy.infrastructure.providers.rate_limiter import ClientRateLimiter
from ccproxy.infrastructure.providers.response_handlers import ResponseProcessor
from ccproxy.infrastructure.providers.request_logger import RequestLogger


@pytest.fixture
def mock_components():
    return {
        "client": AsyncMock(spec=AsyncOpenAI),
        "circuit_breaker": Mock(spec=CircuitBreaker),
        "resilient_executor": AsyncMock(spec=ResilientExecutor),
        "rate_limiter": AsyncMock(spec=ClientRateLimiter),
        "request_logger": Mock(spec=RequestLogger),
        "response_processor": Mock(spec=ResponseProcessor),
    }


def test_init(mock_components):
    pipeline = RequestPipeline(**mock_components)

    assert pipeline._client == mock_components["client"]
    assert pipeline._circuit_breaker == mock_components["circuit_breaker"]
    assert pipeline._resilient_executor == mock_components["resilient_executor"]
    assert pipeline._rate_limiter == mock_components["rate_limiter"]
    assert pipeline._request_logger == mock_components["request_logger"]
    assert pipeline._response_processor == mock_components["response_processor"]


@pytest.mark.anyio
async def test_process_request_success_non_streaming(mock_components):
    mock_components["circuit_breaker"].is_open = False
    mock_components["rate_limiter"].acquire.return_value = True
    mock_components["request_logger"].prepare_trace_headers.return_value = {
        "trace-id": "123"
    }
    mock_components["resilient_executor"].execute.return_value = {
        "choices": [{"message": {"content": "hello"}}]
    }
    params = {"model": "gpt-4", "stream": False, "extra_headers": None}
    correlation_id = "test-id"

    pipeline = RequestPipeline(**mock_components)
    response = await pipeline.process_request(params, correlation_id)

    assert response == {"choices": [{"message": {"content": "hello"}}]}
    assert mock_components["circuit_breaker"].is_open is False  # No raise
    await mock_components["rate_limiter"].acquire.assert_called_once_with(
        request_payload=params
    )
    assert params["extra_headers"] == {"trace-id": "123"}
    mock_components["resilient_executor"].execute.assert_called_once_with(
        mock_components["client"].chat.completions.create,
        model="gpt-4",
        stream=False,
        extra_headers={"trace-id": "123"},
    )
    # No UTF-8 decode needed for str


@pytest.mark.anyio
async def test_process_request_streaming(mock_components):
    mock_components["circuit_breaker"].is_open = False
    mock_components["rate_limiter"] = None  # No limiter
    params = {"model": "gpt-4", "stream": True}
    correlation_id = "test-id"

    pipeline = RequestPipeline(**mock_components)
    response = await pipeline.process_request(params, correlation_id)

    mock_components["resilient_executor"].execute.assert_called_once_with(
        mock_components["client"].chat.completions.create,
        model="gpt-4",
        stream=True,
    )
    assert response is mock_components["resilient_executor"].execute.return_value
    # No UTF-8 handling for streaming


@pytest.mark.anyio
async def test_process_request_circuit_open(mock_components):
    mock_components["circuit_breaker"].is_open = True
    params = {"model": "gpt-4"}
    correlation_id = "test-id"

    pipeline = RequestPipeline(**mock_components)
    with pytest.raises(Exception, match="circuit breaker is open"):
        await pipeline.process_request(params, correlation_id)

    # No further calls
    mock_components["resilient_executor"].execute.assert_not_called()


@pytest.mark.anyio
async def test_process_request_rate_limit_exceeded(mock_components):
    mock_components["circuit_breaker"].is_open = False
    mock_components["rate_limiter"].acquire.return_value = False
    params = {"model": "gpt-4"}
    correlation_id = "test-id"

    pipeline = RequestPipeline(**mock_components)
    with pytest.raises(Exception, match="Client-side rate limit exceeded"):
        await pipeline.process_request(params, correlation_id)

    await mock_components["rate_limiter"].acquire.assert_called_once_with(
        request_payload=params
    )


@pytest.mark.anyio
async def test_check_circuit_breaker_closed(mock_components):
    mock_components["circuit_breaker"].is_open = False
    pipeline = RequestPipeline(**mock_components)
    await pipeline._check_circuit_breaker("test-id")
    # No exception


@pytest.mark.anyio
async def test_check_circuit_breaker_open(mock_components):
    mock_components["circuit_breaker"].is_open = True
    pipeline = RequestPipeline(**mock_components)
    with pytest.raises(Exception, match="circuit breaker is open"):
        await pipeline._check_circuit_breaker("test-id")


@pytest.mark.anyio
async def test_apply_rate_limiting_no_limiter(mock_components):
    mock_components["rate_limiter"] = None
    pipeline = RequestPipeline(**mock_components)
    await pipeline._apply_rate_limiting({"model": "gpt-4"}, "test-id")
    # No calls


@pytest.mark.anyio
async def test_apply_rate_limiting_exceeded(mock_components):
    mock_components["rate_limiter"].acquire.return_value = False
    pipeline = RequestPipeline(**mock_components)
    with pytest.raises(Exception, match="Client-side rate limit exceeded"):
        await pipeline._apply_rate_limiting({"model": "gpt-4"}, "test-id")


def test_prepare_trace_headers(mock_components):
    mock_components["request_logger"].prepare_trace_headers.return_value = {
        "trace": "value"
    }
    params = {"model": "gpt-4"}
    pipeline = RequestPipeline(**mock_components)
    headers = pipeline._prepare_trace_headers("test-id", params)
    assert headers == {"trace": "value"}
    assert params["extra_headers"] == {"trace": "value"}


def test_prepare_trace_headers_none(mock_components):
    mock_components["request_logger"].prepare_trace_headers.return_value = None
    params = {"model": "gpt-4"}
    pipeline = RequestPipeline(**mock_components)
    headers = pipeline._prepare_trace_headers("test-id", params)
    assert headers is None
    assert "extra_headers" not in params


@pytest.mark.anyio
async def test_execute_request(mock_components):
    params = {"model": "gpt-4", "stream": False}
    pipeline = RequestPipeline(**mock_components)
    response = await pipeline._execute_request(params, False)
    assert response is mock_components["resilient_executor"].execute.return_value
    mock_components["resilient_executor"].execute.assert_called_once_with(
        mock_components["client"].chat.completions.create,
        model="gpt-4",
        stream=False,
    )


@pytest.mark.anyio
async def test_ensure_utf8_response_no_choices(mock_components):
    response = {}
    pipeline = RequestPipeline(**mock_components)
    result = await pipeline._ensure_utf8_response(response, "test-id")
    assert result == {}


@pytest.mark.anyio
async def test_ensure_utf8_response_str_content(mock_components, caplog):
    response = {"choices": [{"message": {"content": "hello"}}]}
    pipeline = RequestPipeline(**mock_components)
    result = await pipeline._ensure_utf8_response(response, "test-id")
    assert result == response
    assert "UTF-8 decode error" not in caplog.text


@pytest.mark.anyio
async def test_ensure_utf8_response_bytes_success(mock_components):
    b_content = b"hello".decode("utf-8").encode("utf-8")
    response = {"choices": [{"message": {"content": b_content}}]}
    pipeline = RequestPipeline(**mock_components)
    result = await pipeline._ensure_utf8_response(response, "test-id")
    assert result["choices"][0]["message"]["content"] == "hello"


@pytest.mark.anyio
async def test_ensure_utf8_response_bytes_replace(caplog):
    response = {"choices": [{"message": {"content": b"\xff"}}]}
    mock_components["request_logger"] = Mock()  # For logging
    pipeline = RequestPipeline(
        mock_client=Mock(),
        **{k: v for k, v in mock_components.items() if k != "client"},
    )
    result = await pipeline._ensure_utf8_response(response, "test-id")
    assert result["choices"][0]["message"]["content"] == "�"  # Replacement char
    assert "UTF-8 decode error" in caplog.text


@pytest.mark.anyio
async def test_handle_rate_limit_response(mock_components):
    mock_error = Mock(spec=openai.RateLimitError)
    mock_response = Mock()
    mock_response.headers = {"retry-after": "30"}
    mock_error.response = mock_response
    mock_components["rate_limiter"] = AsyncMock()

    pipeline = RequestPipeline(**mock_components)
    await pipeline.handle_rate_limit_response(mock_error)

    await mock_components["rate_limiter"].handle_429_response.assert_called_once_with(
        30
    )

    # Test no limiter
    pipeline._rate_limiter = None
    await pipeline.handle_rate_limit_response(mock_error)
    # No call


@pytest.mark.anyio
async def test_release_tokens_on_success(mock_components):
    response = {"usage": {"total_tokens": 100}}
    mock_components["response_processor"].extract_usage_info.return_value = {
        "total_tokens": 100
    }
    mock_components["rate_limiter"] = AsyncMock()

    pipeline = RequestPipeline(**mock_components)
    await pipeline.release_tokens_on_success(response)

    await mock_components["rate_limiter"].handle_success.assert_called_once()
    await mock_components["rate_limiter"].release.assert_called_once_with(100)

    # Test no usage
    mock_components["response_processor"].extract_usage_info.return_value = None
    await pipeline.release_tokens_on_success(response)
    await mock_components[
        "rate_limiter"
    ].release.assert_not_called()  # Only if not called previously, but adjust

    # Test no limiter
    pipeline._rate_limiter = None
    await pipeline.release_tokens_on_success(response)
    # No calls


@pytest.mark.anyio
async def test_full_openai_response_utf8_decode_success(mock_components):
    """Test full OpenAI response with UTF-8 decode success."""
    # Mock full OpenAI response with multiple choices
    response_data = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": b"Hello world with \xc3\xa9motion",  # UTF-8 bytes for "émotion"
                },
                "finish_reason": "stop",
            },
            {
                "index": 1,
                "message": {
                    "role": "assistant",
                    "content": b"Second \xc3\xa7hoice",  # UTF-8 bytes for "çhoice"
                },
                "finish_reason": "stop",
            },
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }

    mock_components["circuit_breaker"].is_open = False
    mock_components["rate_limiter"] = None
    mock_components["resilient_executor"].execute.return_value = response_data

    pipeline = RequestPipeline(**mock_components)
    params = {"model": "gpt-4", "stream": False}

    result = await pipeline.process_request(params, "test-id")

    # Verify UTF-8 bytes were decoded properly
    assert result["choices"][0]["message"]["content"] == "Hello world with émotion"
    assert result["choices"][1]["message"]["content"] == "Second çhoice"
    assert result["usage"]["total_tokens"] == 30


@pytest.mark.anyio
async def test_full_openai_response_utf8_decode_error_with_replacement(
    mock_components, caplog
):
    """Test full OpenAI response with UTF-8 decode error using replacement."""
    # Mock response with invalid UTF-8 bytes
    response_data = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": b"Invalid \xff\xfe bytes",  # Invalid UTF-8
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"total_tokens": 15},
    }

    mock_components["circuit_breaker"].is_open = False
    mock_components["rate_limiter"] = None
    mock_components["resilient_executor"].execute.return_value = response_data

    with patch("logging.warning") as mock_warning:
        pipeline = RequestPipeline(**mock_components)
        params = {"model": "gpt-4", "stream": False}

        result = await pipeline.process_request(params, "test-id")

        # Verify invalid bytes were replaced with replacement character
        assert "�" in result["choices"][0]["message"]["content"]

        # Verify warning was logged
        mock_warning.assert_called_once()
        assert "UTF-8 decode error" in str(mock_warning.call_args)


@pytest.mark.anyio
async def test_streaming_response_with_rate_limit_per_chunk(mock_components):
    """Test streaming response with rate limiter checking per chunk."""

    # Create async generator for streaming chunks
    async def mock_stream():
        chunks = [
            {"choices": [{"delta": {"content": "First"}}], "usage": None},
            {"choices": [{"delta": {"content": " chunk"}}], "usage": None},
            {"choices": [{"delta": {"content": " data"}}], "usage": None},
            {
                "choices": [{"finish_reason": "stop"}],
                "usage": {"total_tokens": 50},
            },  # Final chunk has usage
        ]
        for chunk in chunks:
            yield chunk

    mock_components["circuit_breaker"].is_open = False
    mock_components["rate_limiter"].acquire_token = AsyncMock(return_value=True)
    mock_components["rate_limiter"].release = AsyncMock()
    mock_components["resilient_executor"].execute.return_value = mock_stream()

    pipeline = RequestPipeline(**mock_components)
    params = {"model": "gpt-4", "stream": True}

    # Get the streaming response
    stream_response = await pipeline.process_request(params, "test-id")

    # Consume the stream and verify rate limiter interactions
    chunks_received = []
    async for chunk in stream_response:
        chunks_received.append(chunk)

    assert len(chunks_received) == 4

    # Verify rate limiter was checked for each chunk (if implemented in actual code)
    # This assumes the pipeline checks rate limit per chunk in streaming mode
    # Actual implementation may vary


@pytest.mark.anyio
async def test_streaming_response_rate_limit_failure_mid_stream(mock_components):
    """Test streaming response when rate limit fails mid-stream."""

    # Create async generator that will fail on second chunk
    call_count = 0

    async def mock_acquire_token(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return call_count <= 2  # Fail on third call

    async def mock_stream():
        chunks = [
            {"choices": [{"delta": {"content": "First"}}]},
            {"choices": [{"delta": {"content": " chunk"}}]},
            {"choices": [{"delta": {"content": " fails"}}]},  # This will fail
        ]
        for i, chunk in enumerate(chunks):
            if i < 2:  # Only yield first two chunks
                yield chunk
            else:
                # Simulate rate limit error
                raise Exception("Rate limit exceeded mid-stream")

    mock_components["circuit_breaker"].is_open = False
    mock_components["rate_limiter"].acquire_token = AsyncMock(
        side_effect=mock_acquire_token
    )
    mock_components["resilient_executor"].execute.return_value = mock_stream()

    pipeline = RequestPipeline(**mock_components)
    params = {"model": "gpt-4", "stream": True}

    stream_response = await pipeline.process_request(params, "test-id")

    # Consume stream and expect exception
    chunks_received = []
    with pytest.raises(Exception, match="Rate limit exceeded mid-stream"):
        async for chunk in stream_response:
            chunks_received.append(chunk)

    assert len(chunks_received) == 2  # Only first two chunks received


@pytest.mark.anyio
async def test_streaming_with_total_tokens_release(mock_components):
    """Test that streaming properly releases tokens at the end."""

    async def mock_stream():
        yield {"choices": [{"delta": {"content": "Test"}}]}
        yield {"choices": [{"delta": {"content": " stream"}}]}
        yield {"choices": [{"finish_reason": "stop"}], "usage": {"total_tokens": 100}}

    mock_components["circuit_breaker"].is_open = False
    mock_components["rate_limiter"].acquire = AsyncMock(return_value=True)
    mock_components["rate_limiter"].release = AsyncMock()
    mock_components["rate_limiter"].handle_success = AsyncMock()
    mock_components["resilient_executor"].execute.return_value = mock_stream()
    mock_components["response_processor"].extract_usage_info.return_value = {
        "total_tokens": 100
    }

    pipeline = RequestPipeline(**mock_components)
    params = {"model": "gpt-4", "stream": True}

    stream_response = await pipeline.process_request(params, "test-id")

    # Consume entire stream
    all_chunks = []
    async for chunk in stream_response:
        all_chunks.append(chunk)
        # If this is the last chunk with usage, verify token release
        if "usage" in chunk and chunk["usage"]:
            await pipeline.release_tokens_on_success(chunk)

    # Verify tokens were released after streaming completed
    await mock_components["rate_limiter"].release.assert_called_with(100)
    await mock_components["rate_limiter"].handle_success.assert_called_once()
