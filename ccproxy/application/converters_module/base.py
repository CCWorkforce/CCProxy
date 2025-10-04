"""Base classes and interfaces for message conversion."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ...domain.models import Message, MessagesResponse
from ...config import Settings


@dataclass
class ConversionContext:
    """Context information for conversion operations."""

    request_id: Optional[str] = None
    target_model: Optional[str] = None
    original_model: Optional[str] = None
    settings: Optional[Settings] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseConverter(ABC):
    """Abstract base class for message converters."""

    def __init__(self, context: Optional[ConversionContext] = None):
        """Initialize converter with optional context."""
        self.context = context or ConversionContext()

    @abstractmethod
    def convert(self, source: Any) -> Any:
        """
        Convert from source format to target format.

        Args:
            source: Source data to convert

        Returns:
            Converted data in target format
        """
        pass

    def with_context(self, **kwargs: Any) -> "BaseConverter":
        """
        Create a new converter with updated context.

        Returns:
            New converter instance with updated context
        """
        new_context = ConversionContext(
            request_id=kwargs.get("request_id", self.context.request_id),
            target_model=kwargs.get("target_model", self.context.target_model),
            original_model=kwargs.get("original_model", self.context.original_model),
            settings=kwargs.get("settings", self.context.settings),
            metadata={**self.context.metadata, **kwargs.get("metadata", {})},
        )
        return self.__class__(new_context)


class MessageConverter(BaseConverter):
    """Base class for message format conversion."""

    @abstractmethod
    def convert_message(self, message: Message) -> Dict[str, Any]:
        """Convert a single message."""
        pass

    @abstractmethod
    def convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert a list of messages."""
        pass


class ResponseConverter(BaseConverter):
    """Base class for response format conversion."""

    def convert(self, source: Any) -> Any:
        """
        Default convert implementation that delegates to convert_response.

        Args:
            source: Source data to convert

        Returns:
            Converted data
        """
        return self.convert_response(source)

    @abstractmethod
    def convert_response(self, response: Any) -> MessagesResponse:
        """Convert a response to Anthropic format."""
        pass
