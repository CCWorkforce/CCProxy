from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field


class ContentBlockText(BaseModel):
    type: Literal["text"]
    text: str


class ContentBlockImageSource(BaseModel):
    type: str
    media_type: str
    data: str


class ContentBlockImage(BaseModel):
    type: Literal["image"]
    source: ContentBlockImageSource


class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]


class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], List[Any]]
    is_error: Optional[bool] = None


ContentBlock = Union[
    ContentBlockText, ContentBlockImage, ContentBlockToolUse, ContentBlockToolResult
]


class SystemContent(BaseModel):
    type: Literal["text"]
    text: str


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, List[ContentBlock]]


class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any] = Field(..., alias="input_schema")


class ToolChoice(BaseModel):
    type: Literal["auto", "any", "tool"]
    name: Optional[str] = None


class MessagesRequest(BaseModel):
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


class TokenCountRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None


class TokenCountResponse(BaseModel):
    input_tokens: int


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class ProviderErrorMetadata(BaseModel):
    provider_name: str
    raw_error: Optional[Dict[str, Any]] = None


import enum


class AnthropicErrorType(str, enum.Enum):
    INVALID_REQUEST = "invalid_request_error"
    AUTHENTICATION = "authentication_error"
    PERMISSION = "permission_error"
    NOT_FOUND = "not_found_error"
    RATE_LIMIT = "rate_limit_error"
    API_ERROR = "api_error"
    OVERLOADED = "overloaded_error"
    REQUEST_TOO_LARGE = "request_too_large_error"


class AnthropicErrorDetail(BaseModel):
    type: AnthropicErrorType
    message: str
    provider: Optional[str] = None
    provider_message: Optional[str] = None
    provider_code: Optional[Union[str, int]] = None


class AnthropicErrorResponse(BaseModel):
    type: Literal["error"] = "error"
    error: AnthropicErrorDetail


class MessagesResponse(BaseModel):
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
