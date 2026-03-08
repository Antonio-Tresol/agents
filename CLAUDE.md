# agents

Simple agent framework for LLM-based research. Protocol-driven, pluggable middleware, frozen data types.

## Quick Reference

- **Deps**: `httpx` (core), `mcp` (optional extra)
- **Tests**: `uv run pytest tests/ -v` (73)
- **Lint/format**: `uvx ruff check --fix` / `uvx ruff format`

## Conventions

- Trailing commas enforced (COM812). Line length 120.
- Frozen dataclasses with `to_dict()` for all value types.
- Async-first — blocking I/O wrapped in `asyncio.to_thread()`.
- Path containment enforced in DiskStore.
- Middleware errors are caught and logged, never crash the agent loop.
- `_reset_model_info_cache()` is for testing only; not public API.

## Files

| File | Purpose |
|------|---------|
| `agents/types.py` | Frozen dataclasses (ToolDef, ToolCall, ToolResult, ModelResponse, ModelInfo, AgentConfig) |
| `agents/providers.py` | Protocols (Model, Tools, Store) + implementations (OpenRouter, McpTools, LocalTools, ToolSearch, DiskStore, MemoryStore) |
| `agents/agent.py` | ReAct loop (Agent class, RunResult) |
| `agents/middleware.py` | Middleware base, Context, Transcript (JSONL), Compactor |
| `agents/tools.py` | Built-in tool executors for LocalTools |

Full docs, API examples, and protocols: [`README.md`](README.md)
