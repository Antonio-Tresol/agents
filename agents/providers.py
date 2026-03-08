"""Protocols and implementations for Model, Tools, and Store providers."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import httpx

from agents.types import (
    DEFAULT_API_TIMEOUT,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    ModelInfo,
    ModelResponse,
    ToolCall,
    ToolDef,
    ToolResult,
)

logger = logging.getLogger(__name__)

# ─── Protocols ───


@runtime_checkable
class Model(Protocol):
    async def generate(
        self,
        messages: list[dict],
        tools: list[ToolDef],
    ) -> ModelResponse: ...

    async def close(self) -> None: ...


@runtime_checkable
class Tools(Protocol):
    async def list_tools(self) -> list[ToolDef]: ...
    async def call_tool(self, name: str, arguments: dict) -> ToolResult: ...
    async def open(self) -> None: ...
    async def close(self) -> None: ...


@runtime_checkable
class Store(Protocol):
    async def read(self, key: str) -> str | None: ...
    async def write(self, key: str, content: str) -> None: ...
    async def append(self, key: str, content: str) -> None: ...
    async def list_keys(self, prefix: str = "") -> list[str]: ...


# ─── Model Implementations ───


class OpenRouter:
    """OpenRouter / OpenAI-compatible chat-completions client."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_base: str = "https://openrouter.ai/api/v1",
        *,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        api_timeout: float = DEFAULT_API_TIMEOUT,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.api_base = api_base.rstrip("/")
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._client = httpx.AsyncClient(timeout=api_timeout)

    async def generate(
        self,
        messages: list[dict],
        tools: list[ToolDef],
    ) -> ModelResponse:
        if self._max_retries < 1:
            raise ValueError("max_retries must be >= 1")

        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        if tools:
            body["tools"] = [t.to_openai_schema() for t in tools]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        url = f"{self.api_base}/chat/completions"
        delays = [min(2**i, 8) for i in range(self._max_retries)]

        for attempt in range(self._max_retries):
            resp = await self._client.post(url, json=body, headers=headers)

            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < self._max_retries - 1:
                    logger.warning(
                        "OpenRouter %s (attempt %d), retrying in %ds",
                        resp.status_code,
                        attempt + 1,
                        delays[attempt],
                    )
                    await asyncio.sleep(delays[attempt])
                    continue
                resp.raise_for_status()

            if 400 <= resp.status_code < 500:
                resp.raise_for_status()

            break

        data = resp.json()
        choice = data["choices"][0]
        message = choice["message"]

        # Parse tool calls
        tool_calls: list[ToolCall] = []
        for tc in message.get("tool_calls") or []:
            fn = tc["function"]
            args = fn.get("arguments", "{}")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    logger.warning(
                        "Malformed tool call JSON for %s: %s",
                        fn["name"],
                        args,
                    )
                    args = {"_raw": args}
            tool_calls.append(ToolCall(id=tc["id"], name=fn["name"], arguments=args))

        # Capture reasoning/thinking tokens
        reasoning = message.get("reasoning") or message.get("reasoning_content")

        return ModelResponse(
            content=message.get("content"),
            tool_calls=tool_calls,
            usage=data.get("usage", {}),
            finish_reason=choice.get("finish_reason", ""),
            model=data.get("model", self.model),
            reasoning=reasoning,
            raw=data,
        )

    async def close(self) -> None:
        await self._client.aclose()


# ---- OpenRouter model metadata (async-safe, cached per-process) ----

_model_info_cache: dict[str, ModelInfo] = {}
_models_fetched: bool = False
_fetch_lock = asyncio.Lock()


async def fetch_openrouter_model_info(
    model: str,
    api_base: str = "https://openrouter.ai/api/v1",
) -> ModelInfo | None:
    """Fetch model metadata from OpenRouter. Async-safe, cached per-process.

    Cache is populated once per process lifetime. For long-running processes,
    restart to pick up new models.

    Returns None (with a WARNING log) if the API is unreachable, the response
    is malformed, or the model ID is not found.
    """
    global _models_fetched

    async with _fetch_lock:
        if _models_fetched:
            info = _model_info_cache.get(model)
            if info is None:
                logger.warning(
                    "Model %r not found in OpenRouter catalog (possible typo)",
                    model,
                )
            return info

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(f"{api_base}/models")
                resp.raise_for_status()
                data = resp.json()

            for entry in data.get("data", []):
                mid = entry.get("id")
                ctx = entry.get("context_length")
                tp = entry.get("top_provider") or {}
                pricing = entry.get("pricing") or {}
                params = entry.get("supported_parameters") or []

                if not mid or not isinstance(ctx, int):
                    continue

                max_comp = tp.get("max_completion_tokens")
                if not isinstance(max_comp, int):
                    max_comp = DEFAULT_MAX_TOKENS

                _model_info_cache[mid] = ModelInfo(
                    model_id=mid,
                    context_length=ctx,
                    max_completion_tokens=max_comp,
                    supported_parameters=tuple(params),
                    pricing_prompt=float(pricing.get("prompt", 0)),
                    pricing_completion=float(pricing.get("completion", 0)),
                )

            _models_fetched = True
            logger.info(
                "Cached metadata for %d OpenRouter models",
                len(_model_info_cache),
            )

        except Exception as exc:
            logger.warning("Failed to fetch OpenRouter model catalog: %s", exc)
            return None

    info = _model_info_cache.get(model)
    if info is None:
        logger.warning(
            "Model %r not found in OpenRouter catalog (possible typo)",
            model,
        )
    return info


def _reset_model_info_cache() -> None:
    """Clear the model info cache. Test-only, not concurrency-safe."""
    global _models_fetched
    _model_info_cache.clear()
    _models_fetched = False


# ─── Tools Implementations ───


class McpTools:
    """Tools provider that connects to an MCP server via streamable-http."""

    def __init__(self, url: str) -> None:
        self._url = url
        self._stack: contextlib.AsyncExitStack | None = None
        self._session: Any = None

    async def open(self) -> None:
        from mcp.client.session import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        self._stack = contextlib.AsyncExitStack()
        streams = await self._stack.enter_async_context(
            streamablehttp_client(self._url),
        )
        # streamablehttp_client yields (read_stream, write_stream, _)
        read_stream, write_stream = streams[0], streams[1]
        self._session = await self._stack.enter_async_context(
            ClientSession(read_stream, write_stream),
        )
        await self._session.initialize()

    async def list_tools(self) -> list[ToolDef]:
        if self._session is None:
            raise RuntimeError("McpTools.open() must be called before list_tools()")
        result = await self._session.list_tools()
        return [
            ToolDef(
                name=t.name,
                description=t.description or "",
                parameters=t.inputSchema if hasattr(t, "inputSchema") else {},
            )
            for t in result.tools
        ]

    async def call_tool(self, name: str, arguments: dict) -> ToolResult:
        if self._session is None:
            raise RuntimeError("McpTools.open() must be called before call_tool()")
        result = await self._session.call_tool(name, arguments)
        content_parts = []
        for block in result.content:
            if hasattr(block, "text"):
                content_parts.append(block.text)
            else:
                content_parts.append(str(block))
        return ToolResult(
            tool_call_id="",
            content="\n".join(content_parts),
            is_error=bool(result.isError),
        )

    async def close(self) -> None:
        if self._stack:
            await self._stack.aclose()
            self._stack = None
            self._session = None


class LocalTools:
    """Tools backed by Store + built-in executor functions from tools.py."""

    _TOOL_DEFS: list[ToolDef] = [
        ToolDef(
            name="read_file",
            description="Read a file by relative path.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative file path"},
                },
                "required": ["path"],
            },
        ),
        ToolDef(
            name="write_file",
            description="Write content to a file by relative path.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative file path"},
                    "content": {"type": "string", "description": "File content"},
                },
                "required": ["path", "content"],
            },
        ),
        ToolDef(
            name="create_task",
            description="Create a new task with a title and optional details.",
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Task title"},
                    "details": {
                        "type": "string",
                        "description": "Task details",
                        "default": "",
                    },
                },
                "required": ["title"],
            },
        ),
        ToolDef(
            name="list_tasks",
            description="List all tasks.",
            parameters={"type": "object", "properties": {}},
        ),
        ToolDef(
            name="update_task",
            description="Update a task's status or details.",
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Task UUID"},
                    "status": {"type": "string", "description": "New status"},
                    "details": {"type": "string", "description": "New details"},
                },
                "required": ["task_id"],
            },
        ),
        ToolDef(
            name="get_task",
            description="Get a single task by ID.",
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Task UUID"},
                },
                "required": ["task_id"],
            },
        ),
    ]

    def __init__(self, store: Store, prefix: str) -> None:
        self._store = store
        self._prefix = prefix

    async def list_tools(self) -> list[ToolDef]:
        return list(self._TOOL_DEFS)

    async def call_tool(self, name: str, arguments: dict) -> ToolResult:
        from agents import tools as _tools

        dispatch = {
            "read_file": lambda: _tools.read_file(
                self._store,
                self._prefix,
                arguments["path"],
            ),
            "write_file": lambda: _tools.write_file(
                self._store,
                self._prefix,
                arguments["path"],
                arguments["content"],
            ),
            "create_task": lambda: _tools.create_task(
                self._store,
                self._prefix,
                arguments["title"],
                arguments.get("details", ""),
            ),
            "list_tasks": lambda: _tools.list_tasks(self._store, self._prefix),
            "update_task": lambda: _tools.update_task(
                self._store,
                self._prefix,
                arguments["task_id"],
                arguments.get("status"),
                arguments.get("details"),
            ),
            "get_task": lambda: _tools.get_task(
                self._store,
                self._prefix,
                arguments["task_id"],
            ),
        }

        fn = dispatch.get(name)
        if fn is None:
            return ToolResult(
                tool_call_id="",
                content=f"Unknown tool: {name}",
                is_error=True,
            )

        try:
            result = await fn()
            return ToolResult(tool_call_id="", content=result)
        except (ValueError, KeyError) as exc:
            return ToolResult(tool_call_id="", content=str(exc), is_error=True)

    async def open(self) -> None:
        pass

    async def close(self) -> None:
        pass


class ToolSearch:
    """Wraps a Tools provider, exposing a meta search_tools tool that lazily activates inner tools."""

    _META_TOOL = ToolDef(
        name="search_tools",
        description="Search for available tools by keyword. Returns matching tool names and descriptions.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords to match against tool names and descriptions",
                },
            },
            "required": ["query"],
        },
    )

    def __init__(self, inner: Tools) -> None:
        self._inner = inner
        self._all_tools: list[ToolDef] = []
        self._activated: dict[str, ToolDef] = {}

    async def open(self) -> None:
        await self._inner.open()
        self._all_tools = await self._inner.list_tools()

    async def list_tools(self) -> list[ToolDef]:
        return [self._META_TOOL] + list(self._activated.values())

    async def call_tool(self, name: str, arguments: dict) -> ToolResult:
        if name == "search_tools":
            return self._search(arguments.get("query", ""))

        if name in self._activated:
            return await self._inner.call_tool(name, arguments)

        return ToolResult(
            tool_call_id="",
            content=f"Tool '{name}' not found. Use search_tools to discover available tools.",
            is_error=True,
        )

    def _search(self, query: str) -> ToolResult:
        query_lower = query.lower()
        keywords = query_lower.split()
        matches: list[ToolDef] = []

        for tool in self._all_tools:
            searchable = f"{tool.name} {tool.description}".lower()
            if any(kw in searchable for kw in keywords):
                matches.append(tool)
                self._activated[tool.name] = tool

        if not matches:
            return ToolResult(tool_call_id="", content="No tools matched the query.")

        lines = [f"- {t.name}: {t.description}" for t in matches]
        return ToolResult(
            tool_call_id="",
            content=f"Found {len(matches)} tool(s):\n" + "\n".join(lines),
        )

    async def close(self) -> None:
        await self._inner.close()


# ─── Store Implementations ───


class DiskStore:
    """File-system backed store."""

    def __init__(self, base_dir: Path | str) -> None:
        self._base = Path(base_dir)

    def _safe_path(self, key: str) -> Path:
        """Resolve key to a path, ensuring it stays within base_dir."""
        path = (self._base / key).resolve()
        if not str(path).startswith(str(self._base.resolve())):
            raise ValueError(f"Path escapes base directory: {key}")
        return path

    async def read(self, key: str) -> str | None:
        path = self._safe_path(key)
        try:
            return await asyncio.to_thread(path.read_text, "utf-8")
        except FileNotFoundError:
            return None

    async def write(self, key: str, content: str) -> None:
        path = self._safe_path(key)

        def _write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

        await asyncio.to_thread(_write)

    async def append(self, key: str, content: str) -> None:
        path = self._safe_path(key)

        def _append() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(content)

        await asyncio.to_thread(_append)

    async def list_keys(self, prefix: str = "") -> list[str]:
        search_dir = self._safe_path(prefix) if prefix else self._base.resolve()

        def _scan() -> list[str]:
            if not search_dir.exists():
                return []
            keys: list[str] = []
            for p in search_dir.rglob("*"):
                if p.is_file():
                    keys.append(
                        str(p.relative_to(self._base.resolve())).replace("\\", "/"),
                    )
            return sorted(keys)

        return await asyncio.to_thread(_scan)


class MemoryStore:
    """In-memory dict-backed store, useful for testing."""

    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    async def read(self, key: str) -> str | None:
        return self._data.get(key)

    async def write(self, key: str, content: str) -> None:
        self._data[key] = content

    async def append(self, key: str, content: str) -> None:
        self._data[key] = self._data.get(key, "") + content

    async def list_keys(self, prefix: str = "") -> list[str]:
        return sorted(k for k in self._data if k.startswith(prefix))
