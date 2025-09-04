"""Enums module for CCProxy configuration.

Contains all enumeration classes used throughout the application.
"""

from typing import Literal
from enum import StrEnum
from pydantic import BaseModel

# Truncation strategy options
TruncationStrategy = Literal["oldest_first", "newest_first", "system_priority"]

class TruncationConfig(BaseModel):
    strategy: TruncationStrategy = "oldest_first"
    min_tokens: int = 100
    system_message_priority: bool = True


class MessageRoles(StrEnum):
    """Message role constants for chat completion."""
    Developer = "developer"
    System = "system"
    User = "user"


class ReasoningEfforts(StrEnum):
    """Reasoning effort levels for supported models."""
    High = "high"
    Medium = "medium"
    Low = "low"