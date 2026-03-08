"""Core type definitions for the agent scaffold.

Frozen dataclasses representing tool definitions, tool calls/results,
and model responses.  All types have ``to_dict()`` for serialisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolDef:
    """A tool available for the model to call."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_openai_schema(self) -> dict[str, Any]:
        """Return the OpenAI-style function-calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


@dataclass(frozen=True)
class ToolCall:
    """A model's request to invoke a tool."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
        }


@dataclass(frozen=True)
class ToolResult:
    """Result from executing a tool call."""

    tool_call_id: str
    content: str
    is_error: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "content": self.content,
            "is_error": self.is_error,
        }


@dataclass(frozen=True)
class ModelResponse:
    """Parsed response from an LLM API call."""

    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    finish_reason: str = ""
    model: str = ""
    reasoning: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "reasoning": self.reasoning,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "usage": self.usage,
            "finish_reason": self.finish_reason,
            "model": self.model,
        }


# ---- Default constants (shared by AgentConfig and OpenRouter) ----
DEFAULT_TEMPERATURE: float = 1.0
DEFAULT_MAX_TOKENS: int = 4096
DEFAULT_MAX_RETRIES: int = 3
DEFAULT_API_TIMEOUT: float = 300.0
DEFAULT_COMPACTION_RATIO: float = 0.75
FALLBACK_CONTEXT_LENGTH: int = 50_000


@dataclass(frozen=True)
class ModelInfo:
    """Model metadata fetched from OpenRouter's /api/v1/models endpoint."""

    model_id: str
    context_length: int
    max_completion_tokens: int
    supported_parameters: tuple[str, ...]
    pricing_prompt: float
    pricing_completion: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "context_length": self.context_length,
            "max_completion_tokens": self.max_completion_tokens,
            "supported_parameters": list(self.supported_parameters),
            "pricing_prompt": self.pricing_prompt,
            "pricing_completion": self.pricing_completion,
        }


@dataclass(frozen=True)
class AgentConfig:
    """Configuration for agent sessions backed by OpenRouter or similar providers.

    Fields with None use auto-resolution from model metadata at construction time.
    """

    temperature: float = DEFAULT_TEMPERATURE
    max_retries: int = DEFAULT_MAX_RETRIES
    api_timeout: float = DEFAULT_API_TIMEOUT
    max_tokens: int | None = None
    compaction_ratio: float = DEFAULT_COMPACTION_RATIO
    compaction_threshold: int | None = None

    def __post_init__(self) -> None:
        if not 0 < self.compaction_ratio < 1:
            raise ValueError(
                f"compaction_ratio must be in (0, 1), got {self.compaction_ratio}",
            )
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")
        if self.api_timeout <= 0:
            raise ValueError(f"api_timeout must be > 0, got {self.api_timeout}")
