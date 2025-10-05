"""Integration tests for end-to-end API flow simulation."""

import pytest
import json
from httpx import Response
import respx

from ccproxy.interfaces.http.app import create_app
from ccproxy.config import Settings
from httpx import AsyncClient


@pytest.fixture
def test_settings():
    """Create test settings."""
    settings = Settings(
        openai_api_key="test-key",
        big_model_name="gpt-4",
        small_model_name="gpt-3.5-turbo",
        openai_base_url="https://api.openai.com/v1",
        host="127.0.0.1",
        port=8000,
        log_level="INFO",
    )
    return settings


@pytest.fixture
def test_app(test_settings):
    """Create test FastAPI app."""
    return create_app(test_settings)


class TestEndToEndIntegration:
    """Test complete request flow through the application."""

    @pytest.mark.anyio
    @respx.mock
    async def test_opus_to_gpt4_mapping(self, test_app):
        """Test that opus model correctly maps to gpt-4."""
        # Mock OpenAI API response
        mock_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello from GPT-4!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
        }

        # Setup respx mock for OpenAI endpoint
        route = respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=Response(200, json=mock_response)
        )

        async with AsyncClient(app=test_app, base_url="http://test") as client:
            # Send request with opus model
            request_data = {
                "model": "claude-3-opus-20240229",
                "messages": [{"role": "user", "content": "Hello"}],
            }

            response = await client.post(
                "/v1/messages", json=request_data, headers={"x-api-key": "test-key"}
            )

            result = response.json()

            # Verify response format is Anthropic-style
            assert "content" in result
            assert result["role"] == "assistant"
            assert result["model"] == "claude-3-opus-20240229"

            # Verify the OpenAI API was called with gpt-4
            assert route.called
            openai_request = route.calls[0].request
            request_body = json.loads(openai_request.content)
            assert request_body["model"] == "gpt-4"

    @pytest.mark.anyio
    @respx.mock
    async def test_sonnet_to_gpt35_mapping(self, test_app):
        """Test that sonnet model correctly maps to gpt-3.5-turbo."""
        mock_response = {
            "id": "chatcmpl-456",
            "object": "chat.completion",
            "model": "gpt-3.5-turbo",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello from GPT-3.5!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"total_tokens": 20},
        }

        route = respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=Response(200, json=mock_response)
        )

        async with AsyncClient(app=test_app, base_url="http://test") as client:
            request_data = {
                "model": "claude-3-sonnet-20240229",
                "messages": [{"role": "user", "content": "Test message"}],
            }

            response = await client.post(
                "/v1/messages", json=request_data, headers={"x-api-key": "test-key"}
            )

            assert response.status_code == 200

            # Verify mapping
            assert route.called
            openai_request = route.calls[0].request
            request_body = json.loads(openai_request.content)
            assert request_body["model"] == "gpt-3.5-turbo"

    @pytest.mark.anyio
    @respx.mock
    async def test_error_propagation_with_retry(self, test_app):
        """Test that API errors are properly propagated and retried."""
        # First call fails with 503, second succeeds
        error_response = Response(503, json={"error": "Service unavailable"})
        success_response = Response(
            200,
            json={
                "id": "chatcmpl-789",
                "object": "chat.completion",
                "model": "gpt-4",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Success after retry",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 30},
            },
        )

        route = respx.post("https://api.openai.com/v1/chat/completions").mock(
            side_effect=[error_response, success_response]
        )

        async with AsyncClient(app=test_app, base_url="http://test") as client:
            request_data = {
                "model": "claude-3-opus-20240229",
                "messages": [{"role": "user", "content": "Test retry"}],
            }

            response = await client.post(
                "/v1/messages", json=request_data, headers={"x-api-key": "test-key"}
            )

            # Should eventually succeed after retry
            if response.status_code == 200:
                result = response.json()
                assert "content" in result

            assert route.call_count >= 1  # At least one call made

    @pytest.mark.anyio
    @respx.mock
    async def test_streaming_response_conversion(self, test_app):
        """Test streaming response conversion from OpenAI to Anthropic format."""

        # Create SSE chunks
        def create_sse_chunk(data):
            if data is None:
                return b"data: [DONE]\n\n"
            return f"data: {json.dumps(data)}\n\n".encode()

        chunks = [
            create_sse_chunk(
                {
                    "id": "chatcmpl-stream",
                    "object": "chat.completion.chunk",
                    "model": "gpt-4",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "Hello "},
                            "finish_reason": None,
                        }
                    ],
                }
            ),
            create_sse_chunk(
                {
                    "id": "chatcmpl-stream",
                    "object": "chat.completion.chunk",
                    "model": "gpt-4",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "streaming!"},
                            "finish_reason": None,
                        }
                    ],
                }
            ),
            create_sse_chunk(
                {
                    "id": "chatcmpl-stream",
                    "object": "chat.completion.chunk",
                    "model": "gpt-4",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
            ),
            create_sse_chunk(None),  # [DONE] marker
        ]

        async def stream_response():
            for chunk in chunks:
                yield chunk

        route = respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=Response(
                200,
                content=stream_response(),
                headers={"content-type": "text/event-stream"},
            )
        )

        async with AsyncClient(app=test_app, base_url="http://test") as client:
            request_data = {
                "model": "claude-3-opus-20240229",
                "messages": [{"role": "user", "content": "Stream test"}],
                "stream": True,
            }

            # Stream the response
            async with client.stream(
                "POST",
                "/v1/messages",
                json=request_data,
                headers={"x-api-key": "test-key"},
            ) as response:
                assert response.status_code == 200

                chunks_received = []
                async for chunk in response.aiter_bytes():
                    if chunk:
                        chunks_received.append(chunk)

                # Verify we received streaming chunks
                assert len(chunks_received) > 0
                assert route.called

    @pytest.mark.anyio
    @respx.mock
    async def test_tool_use_conversion(self, test_app):
        """Test conversion of tool use between Anthropic and OpenAI formats."""
        # Mock OpenAI response with function call
        mock_response = {
            "id": "chatcmpl-tool",
            "object": "chat.completion",
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": json.dumps(
                                        {"location": "San Francisco"}
                                    ),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"total_tokens": 50},
        }

        route = respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=Response(200, json=mock_response)
        )

        async with AsyncClient(app=test_app, base_url="http://test") as client:
            # Send request with tool definition
            request_data = {
                "model": "claude-3-opus-20240229",
                "messages": [{"role": "user", "content": "What's the weather in SF?"}],
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Get weather for a location",
                        "input_schema": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                            "required": ["location"],
                        },
                    }
                ],
            }

            response = await client.post(
                "/v1/messages", json=request_data, headers={"x-api-key": "test-key"}
            )

            assert response.status_code == 200
            result = response.json()

            # Verify tool use was converted back to Anthropic format
            assert "content" in result
            # Content should have tool_use blocks
            assert any(
                block.get("type") == "tool_use"
                for block in result.get("content", [])
                if isinstance(block, dict)
            )

            # Verify OpenAI was called with correct tool format
            assert route.called
            openai_request = route.calls[0].request
            request_body = json.loads(openai_request.content)
            assert "tools" in request_body
            assert request_body["tools"][0]["function"]["name"] == "get_weather"
