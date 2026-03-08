# agents — A Very Simple Agent Framework

A very simple agent framework for LLM-based agents research, as self-contained
as possible. Protocol-driven, pluggable middleware, frozen data types. Connects
to any OpenAI-compatible API via OpenRouter, discovers and calls tools over MCP,
and persists state to disk.

## Design Principles

- **Protocol-driven** — `Model`, `Tools`, and `Store` are runtime-checkable
  protocols. Swap implementations without touching agent logic.
- **No framework lock-in** — Pure Python, no heavy dependencies. Core requires
  only `httpx` and standard library.
- **Middleware, not monolith** — Cross-cutting concerns (logging, compaction)
  are middleware hooks, not baked into the agent loop.
- **Frozen data** — All message types (`ToolDef`, `ToolCall`, `ToolResult`,
  `ModelResponse`, `ModelInfo`, `AgentConfig`) are frozen dataclasses.

## Package Structure

```
agents/
├── core/                    # Standalone agent scaffold
│   ├── types.py             # Frozen dataclasses + default constants
│   ├── providers.py         # Protocol interfaces + implementations
│   │   ├── Model            # Protocol: generate(messages, tools) → ModelResponse
│   │   ├── OpenRouter       # OpenAI-compatible API via OpenRouter
│   │   ├── Tools            # Protocol: open/close, list_tools, call_tool
│   │   ├── McpTools         # MCP server tool provider (streamable-http)
│   │   ├── LocalTools       # Built-in file I/O and task tools
│   │   ├── ToolSearch       # Lazy tool discovery wrapper
│   │   ├── Store            # Protocol: async key-value read/write/append
│   │   ├── DiskStore        # File-backed store with path containment
│   │   └── MemoryStore      # In-memory store (testing)
│   ├── agent.py             # ReAct loop (Agent class, RunResult)
│   ├── middleware.py         # Middleware base, Context, Transcript, Compactor
│   ├── tools.py             # Built-in tool executors for LocalTools
│   └── __init__.py          # Public API re-exports
├── backends.py              # Session wrappers (ClaudeCode, OpenCode, AgentSession)
├── orchestrator.py          # Turn-driven game loop (consumer code)
├── prompts/                 # System prompt templates (consumer code)
└── AGENTS.md
```

`core/` is the reusable framework. `backends.py`, `orchestrator.py`, and
`prompts/` are consumer code specific to the host application.

## Core API

### Agent

```python
from agents.core import Agent, OpenRouter, McpTools, DiskStore, Transcript, Compactor

agent = Agent(
    model=OpenRouter(model="openai/gpt-5-mini", temperature=0.7),
    tools=[McpTools(url="http://localhost:8000/mcp/")],
    middleware=[
        Transcript(store=DiskStore("./logs"), key="transcript.jsonl"),
        Compactor(threshold_tokens=40_000),
    ],
    system_prompt="You are a helpful assistant.",
)

result = await agent.run("What files are in the project?", max_turns=10)
print(result.final_content)
print(f"Tokens: {result.total_input_tokens} in, {result.total_output_tokens} out")
```

### AgentConfig

Typed configuration with validation. All fields have sensible defaults.

```python
from agents.core import AgentConfig

config = AgentConfig(
    temperature=0.7,
    max_tokens=4096,
    max_retries=3,
    api_timeout=300.0,
    compaction_ratio=0.75,
)
```

### Model Metadata

Fetches per-model context windows and pricing from OpenRouter's API.
Thread-safe cache, graceful degradation if API is unreachable.

```python
from agents.core import fetch_openrouter_model_info

info = await fetch_openrouter_model_info("openai/gpt-5-mini")
if info:
    print(f"Context: {info.context_length}, Max output: {info.max_completion_tokens}")
```

### Protocols

```python
class Model(Protocol):
    async def generate(self, messages: list[dict], tools: list[ToolDef]) -> ModelResponse: ...

class Tools(Protocol):
    async def open(self) -> None: ...
    async def close(self) -> None: ...
    async def list_tools(self) -> list[ToolDef]: ...
    async def call_tool(self, name: str, arguments: dict) -> ToolResult: ...

class Store(Protocol):
    async def read(self, key: str) -> str | None: ...
    async def write(self, key: str, content: str) -> None: ...
    async def append(self, key: str, content: str) -> None: ...
    async def list_keys(self, prefix: str = "") -> list[str]: ...
```

### Middleware Hooks

Middleware receives lifecycle events. Implement any subset:

```python
class Middleware(Protocol):
    async def before_agent(self, ctx: Context) -> None: ...
    async def before_model(self, ctx: Context, messages: list, tools: list) -> None: ...
    async def after_model(self, ctx: Context, response: ModelResponse) -> None: ...
    async def before_tool(self, ctx: Context, call: ToolCall) -> None: ...
    async def after_tool(self, ctx: Context, call: ToolCall, result: ToolResult) -> None: ...
    async def after_agent(self, ctx: Context, result: RunResult) -> None: ...
```

Built-in middleware:
- **Transcript** — Writes JSONL event stream to a Store.
- **Compactor** — Summarises old messages when token count exceeds threshold.

## Installation (uv)

```bash
# Add as a dependency (with MCP support)
uv add "agents[mcp] @ git+https://github.com/Antonio-Tresol/agents.git"

# Or add to pyproject.toml manually:
# dependencies = [
#     "agents[mcp] @ git+https://github.com/Antonio-Tresol/agents.git",
# ]
# Then: uv sync

# Note: if using hatchling as build backend, add:
# [tool.hatch.metadata]
# allow-direct-references = true
```

## Dependencies

- `httpx` — HTTP client for OpenRouter API and MCP transport
- `mcp` — MCP SDK (only needed if using McpTools)
- Standard library only for everything else

## Conventions

- Ruff for linting and formatting. Trailing commas enforced (COM812). Line length 120.
- Frozen dataclasses for all value types. `to_dict()` methods for serialization.
- Async-first — all provider methods are async. Blocking I/O wrapped in `asyncio.to_thread()`.
- Path containment enforced in DiskStore (no directory traversal).
- Errors in middleware are caught and logged, never crash the agent loop.
- `_reset_model_info_cache()` exists for testing only; not part of public API.
