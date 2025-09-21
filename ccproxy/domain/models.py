from enum import StrEnum
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field


class ContentBlockText(BaseModel):
    """Represents a text content block in a message.

    Attributes:
        type (Literal["text"]): The content block type, always 'text'.
        text (str): The text content of the block.
    """

    type: Literal["text"]
    text: str


class ContentBlockImageSource(BaseModel):
    """Represents the source of an image content block.

    Attributes:
        type (str): The type of source (e.g., 'base64').
        media_type (str): The MIME type of the image (e.g., 'image/png').
        data (str): The base64-encoded image data.
    """

    type: str
    media_type: str
    data: str


class ContentBlockImage(BaseModel):
    """Represents an image content block in a message.

    Attributes:
        type (Literal["image"]): The content block type, always 'image'.
        source (ContentBlockImageSource): The image source details including media type and data.
    """

    type: Literal["image"]
    source: ContentBlockImageSource


class ContentBlockToolUse(BaseModel):
    """Represents a tool use request within a message.

    Attributes:
        type (Literal["tool_use"]): The content block type, always 'tool_use'.
        id (str): Unique identifier for this tool use instance.
        name (str): The name of the tool being invoked.
        input (Dict[str, Any]): Parameters to pass to the tool.
    """

    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]


class ContentBlockToolResult(BaseModel):
    """Represents the result of a tool use operation.

    Attributes:
        type (Literal["tool_result"]): The content block type, always 'tool_result'.
        tool_use_id (str): ID of the tool use request this result corresponds to.
        content (Union[str, List[Dict[str, Any]], List[Any]]): The tool's output data.
        is_error (Optional[bool]): Whether this result indicates an error (default: None).
    """

    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], List[Any]]
    is_error: Optional[bool] = None


class ContentBlockThinking(BaseModel):
    """Represents a model's internal reasoning process block.

    Attributes:
        type (Literal["thinking"]): The content block type, always 'thinking'.
        thinking (str): The model's internal reasoning or thought process text.
        signature (Optional[str]): Optional cryptographic signature for the thinking process (default: None).
    """

    type: Literal["thinking"]
    thinking: str
    signature: Optional[str] = None


class ContentBlockRedactedThinking(BaseModel):
    """Represents redacted thinking content where internal reasoning is partially hidden.

    Attributes:
        type (Literal["redacted_thinking"]): The content block type, always 'redacted_thinking'.
        data (str): The redacted or masked reasoning content.
    """

    type: Literal["redacted_thinking"]
    data: str


ContentBlock = Union[
    ContentBlockText,
    ContentBlockImage,
    ContentBlockToolUse,
    ContentBlockToolResult,
    ContentBlockThinking,
    ContentBlockRedactedThinking,
]


class SystemContent(BaseModel):
    """Represents system message content in a conversation.

    Attributes:
        type (Literal["text"]): The content type, always 'text'.
        text (str): The actual system message content.
    """

    type: Literal["text"]
    text: str


class Message(BaseModel):
    """Represents a single message in the conversation.

    Attributes:
        role (Literal["user", "assistant"]): The speaker's role, either 'user' or 'assistant'.
        content (Union[str, List[ContentBlock]]): The message content, either as a string or structured content blocks.
    """

    role: Literal["user", "assistant"]
    content: Union[str, List[ContentBlock]]


class Tool(BaseModel):
    """Represents a tool that can be invoked by the model.

    Attributes:
        name (str): The unique name/identifier of the tool.
        description (Optional[str]): Brief explanation of the tool's purpose (default: None).
        input_schema (Optional[Dict[str, Any]]): JSON schema defining expected input parameters.
            Required for regular tools, optional for built-in tools.
        type (Optional[str]): Type identifier for built-in tools (e.g., 'web_search_20250305').
        max_uses (Optional[int]): Maximum number of times the tool can be used in a single request.
    """

    name: str
    description: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = Field(None, alias="input_schema")
    type: Optional[str] = None
    max_uses: Optional[int] = None


class ToolChoice(BaseModel):
    """Defines the strategy for tool selection during model execution.

    Attributes:
        type (Literal["auto", "any", "tool"]): The tool selection strategy.
            'auto': Model decides whether to use tools.
            'any': Model must use at least one tool.
            'tool': Model must use the specific tool named in 'name'.
        name (Optional[str]): The name of the tool to use when `type` is 'tool'; required in that case, otherwise ignored.
    """

    type: Literal["auto", "any", "tool"]
    name: Optional[str] = None


class ThinkingConfig(BaseModel):
    """Configuration for enabling model thinking (internal reasoning) with token constraints.

    Attributes:
        type (Literal["enabled"]): Configuration type, must be 'enabled' (default: 'enabled').
        budget_tokens (int): Token budget for thinking process; minimum 1024.
    """

    type: Literal["enabled"] = "enabled"
    budget_tokens: int = Field(
        ge=1024, description="Token budget for thinking, minimum 1024"
    )


class MessagesRequest(BaseModel):
    """Request model for the Messages API endpoint.

    Attributes:
        model (str): The model to use for generating the response.
        max_tokens (int): Maximum number of tokens to generate.
        messages (List[Message]): Conversation history as a list of messages.
        system (Optional[Union[str, List[SystemContent]]]): System instructions, either as a string or structured content.
        stop_sequences (Optional[List[str]]): Sequences where the model should stop generating.
        stream (Optional[bool]): Whether to stream the response (default: False).
        temperature (Optional[float]): Sampling temperature for generation.
        top_p (Optional[float]): Nucleus sampling parameter.
        top_k (Optional[int]): Top-k sampling parameter.
        metadata (Optional[Dict[str, Any]]): Additional metadata for the request.
        tools (Optional[List[Tool]]): Tools available for the model to use.
        tool_choice (Optional[ToolChoice]): Strategy for tool selection.
        thinking (Optional[ThinkingConfig]): Configuration for model's internal reasoning.
    """

    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[ToolChoice] = None
    thinking: Optional[ThinkingConfig] = None


class TokenCountRequest(BaseModel):
    """Request model for token counting endpoint.

    Attributes:
        model (str): The model to use for token counting.
        messages (List[Message]): Conversation history to count tokens for.
        system (Optional[Union[str, List[SystemContent]]]): System instructions to include in count.
        tools (Optional[List[Tool]]): Tools definitions to include in count.
    """

    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None


class TokenCountResponse(BaseModel):
    """Response model for token counting endpoint.

    Attributes:
        input_tokens (int): Total number of tokens in the input messages.
    """

    input_tokens: int


class Usage(BaseModel):
    """Token usage statistics for an API request.

    Attributes:
        input_tokens (int): Number of tokens in the input messages.
        output_tokens (int): Number of tokens in the generated response.
    """

    input_tokens: int
    output_tokens: int


class ProviderErrorMetadata(BaseModel):
    """Metadata associated with an error response from a model provider.

    Attributes:
        provider_name (str): Name of the provider that generated the error.
        raw_error (Optional[Dict[str, Any]]): Original error response from the provider.
    """

    provider_name: str
    raw_error: Optional[Dict[str, Any]] = None


class AnthropicErrorType(StrEnum):
    """Enumeration of possible error types returned by the Anthropic API.

    Attributes:
        INVALID_REQUEST: Parameters were incorrect
        AUTHENTICATION: Missing or invalid credentials
        PERMISSION: User lacks required permissions
        NOT_FOUND: Resource not found
        RATE_LIMIT: Rate limit exceeded
        API_ERROR: Generic API error
        OVERLOADED: Service is overloaded
        REQUEST_TOO_LARGE: Request payload too large
    """

    INVALID_REQUEST = "invalid_request_error"
    AUTHENTICATION = "authentication_error"
    PERMISSION = "permission_error"
    NOT_FOUND = "not_found_error"
    RATE_LIMIT = "rate_limit_error"
    API_ERROR = "api_error"
    OVERLOADED = "overloaded_error"
    REQUEST_TOO_LARGE = "request_too_large_error"


class AnthropicErrorDetail(BaseModel):
    """Detailed information about an error encountered during API processing.

    Attributes:
        type (AnthropicErrorType): Categorized error type
        message (str): Human-readable error description
        provider (Optional[str]): Underlying provider name, if different from Anthropic
        provider_message (Optional[str]): Error message from the provider
        provider_code (Optional[Union[str, int]]): Provider-specific error code
    """

    type: AnthropicErrorType
    message: str
    provider: Optional[str] = None
    provider_message: Optional[str] = None
    provider_code: Optional[Union[str, int]] = None


class AnthropicErrorResponse(BaseModel):
    """Structured error response format following Anthropic API specifications.

    Attributes:
        type (Literal[\"error\"])): Fixed type indicator for error responses
        error (AnthropicErrorDetail): Detailed error information
    """

    type: Literal["error"] = "error"
    error: AnthropicErrorDetail


class MessagesResponse(BaseModel):
    """Response model for successful Messages API requests.

    Attributes:
        id (str): Unique identifier for the message
        type (Literal[\"message\"])): Fixed response type
        role (Literal[\"assistant\"])): Role of the message sender
        model (str): Model that generated the response
        content (List[ContentBlock]): Generated content blocks
        stop_reason (Optional[str]): Reason generation stopped
        stop_sequence (Optional[str]): Stopping sequence that ended generation
        usage (Usage): Token usage statistics for the request
    """

    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    model: str
    content: List[ContentBlock]
    stop_reason: Optional[
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use", "error"]
    ] = None
    stop_sequence: Optional[str] = None
    usage: Usage
