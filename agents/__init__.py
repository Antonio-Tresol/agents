"""Agent core — custom agent scaffold with provider-based architecture."""

from agents.agent import Agent, RunResult
from agents.middleware import Compactor, Context, Middleware, Transcript
from agents.providers import (
    DiskStore,
    LocalTools,
    McpTools,
    MemoryStore,
    Model,
    OpenRouter,
    Store,
    Tools,
    ToolSearch,
    _reset_model_info_cache,  # noqa: F401  (re-exported for tests)
    fetch_openrouter_model_info,
)
from agents.types import (
    DEFAULT_API_TIMEOUT,
    DEFAULT_COMPACTION_RATIO,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    FALLBACK_CONTEXT_LENGTH,
    AgentConfig,
    ModelInfo,
    ModelResponse,
    ToolCall,
    ToolDef,
    ToolResult,
)

__all__ = [
    # Agent
    "Agent",
    "RunResult",
    # Types
    "ToolDef",
    "ToolCall",
    "ToolResult",
    "ModelResponse",
    "ModelInfo",
    "AgentConfig",
    # Constants
    "DEFAULT_TEMPERATURE",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_API_TIMEOUT",
    "DEFAULT_COMPACTION_RATIO",
    "FALLBACK_CONTEXT_LENGTH",
    # Protocols
    "Model",
    "Tools",
    "Store",
    # Provider implementations
    "OpenRouter",
    "McpTools",
    "LocalTools",
    "ToolSearch",
    "DiskStore",
    "MemoryStore",
    # Model metadata
    "fetch_openrouter_model_info",
    # Middleware
    "Context",
    "Middleware",
    "Transcript",
    "Compactor",
]
