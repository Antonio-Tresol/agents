"""Comprehensive unit tests for agents — all mocked, no real API calls."""

from __future__ import annotations

import dataclasses
import json
import unittest.mock
from dataclasses import FrozenInstanceError

import httpx
import pytest

from agents.agent import Agent, RunResult
from agents.middleware import Compactor, Context, Middleware, Transcript
from agents.providers import (
    DiskStore,
    LocalTools,
    MemoryStore,
    OpenRouter,
    ToolSearch,
    _reset_model_info_cache,
    fetch_openrouter_model_info,
)
from agents.types import (
    AgentConfig,
    ModelInfo,
    ModelResponse,
    ToolCall,
    ToolDef,
    ToolResult,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_model_cache():
    """Reset model info cache between tests."""
    yield
    _reset_model_info_cache()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockModel:
    """Fake Model that returns pre-programmed responses in order."""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)
        self._call_count = 0

    async def generate(
        self,
        messages: list[dict],
        tools: list[ToolDef],
    ) -> ModelResponse:
        resp = self._responses[self._call_count]
        self._call_count += 1
        return resp

    async def close(self) -> None:
        pass


class MockTools:
    """Fake Tools provider returning pre-configured results."""

    def __init__(
        self,
        defs: list[ToolDef],
        results: dict[str, str],
    ) -> None:
        self._defs = defs
        self._results = results
        self.opened = False
        self.closed = False

    async def list_tools(self) -> list[ToolDef]:
        return list(self._defs)

    async def call_tool(self, name: str, arguments: dict) -> ToolResult:
        return ToolResult(
            tool_call_id="",
            content=self._results.get(name, "ok"),
        )

    async def open(self) -> None:
        self.opened = True

    async def close(self) -> None:
        self.closed = True


class SpyMiddleware(Middleware):
    """Records every hook call name in order."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def before_agent(self, ctx, *, system_prompt, user_prompt, tools):
        self.calls.append("before_agent")

    async def before_model(self, ctx):
        self.calls.append("before_model")

    async def after_model(self, ctx, *, response):
        self.calls.append("after_model")

    async def before_tool(self, ctx, *, tool_call):
        self.calls.append("before_tool")

    async def after_tool(self, ctx, *, tool_call, result):
        self.calls.append("after_tool")

    async def after_agent(self, ctx, *, result):
        self.calls.append("after_agent")


class CrashMiddleware(Middleware):
    """Raises on every hook — agent must survive."""

    async def before_agent(self, ctx, **kw):
        raise RuntimeError("boom before_agent")

    async def before_model(self, ctx):
        raise RuntimeError("boom before_model")

    async def after_model(self, ctx, **kw):
        raise RuntimeError("boom after_model")

    async def before_tool(self, ctx, **kw):
        raise RuntimeError("boom before_tool")

    async def after_tool(self, ctx, **kw):
        raise RuntimeError("boom after_tool")

    async def after_agent(self, ctx, **kw):
        raise RuntimeError("boom after_agent")


def _text_response(text: str, usage: dict | None = None) -> ModelResponse:
    return ModelResponse(content=text, usage=usage or {})


def _tool_call_response(
    name: str,
    arguments: dict | None = None,
    call_id: str = "call_1",
    usage: dict | None = None,
) -> ModelResponse:
    return ModelResponse(
        content=None,
        tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments or {})],
        usage=usage or {},
    )


# ===================================================================
# types.py — existing types
# ===================================================================


class TestToolDef:
    def test_to_dict_roundtrip(self):
        td = ToolDef(name="foo", description="bar", parameters={"type": "object"})
        d = td.to_dict()
        assert d == {"name": "foo", "description": "bar", "parameters": {"type": "object"}}
        td2 = ToolDef(**d)
        assert td2 == td

    def test_to_openai_schema(self):
        td = ToolDef(name="search", description="Search stuff", parameters={"type": "object", "properties": {}})
        schema = td.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "search"
        assert schema["function"]["description"] == "Search stuff"
        assert schema["function"]["parameters"] == {"type": "object", "properties": {}}

    def test_frozen(self):
        td = ToolDef(name="x", description="y")
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            td.name = "z"  # type: ignore[misc]


class TestToolCall:
    def test_to_dict_roundtrip(self):
        tc = ToolCall(id="c1", name="do_thing", arguments={"a": 1})
        d = tc.to_dict()
        assert d == {"id": "c1", "name": "do_thing", "arguments": {"a": 1}}
        tc2 = ToolCall(**d)
        assert tc2 == tc

    def test_frozen(self):
        tc = ToolCall(id="c1", name="x")
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            tc.name = "y"  # type: ignore[misc]


class TestToolResult:
    def test_to_dict_roundtrip(self):
        tr = ToolResult(tool_call_id="t1", content="ok", is_error=False)
        d = tr.to_dict()
        assert d == {"tool_call_id": "t1", "content": "ok", "is_error": False}
        tr2 = ToolResult(**d)
        assert tr2 == tr

    def test_frozen(self):
        tr = ToolResult(tool_call_id="t1", content="ok")
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            tr.content = "no"  # type: ignore[misc]


class TestModelResponse:
    def test_to_dict_roundtrip(self):
        tc = ToolCall(id="c1", name="fn", arguments={"x": 1})
        mr = ModelResponse(
            content="hello",
            tool_calls=[tc],
            usage={"prompt_tokens": 10},
            finish_reason="stop",
            model="test-model",
            reasoning="thought",
        )
        d = mr.to_dict()
        assert d["content"] == "hello"
        assert d["reasoning"] == "thought"
        assert d["tool_calls"] == [tc.to_dict()]
        assert d["usage"] == {"prompt_tokens": 10}
        assert d["finish_reason"] == "stop"
        assert d["model"] == "test-model"

    def test_frozen(self):
        mr = ModelResponse(content="hi")
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            mr.content = "bye"  # type: ignore[misc]


# ===================================================================
# types.py — ModelInfo
# ===================================================================


class TestModelInfo:
    def test_frozen(self):
        info = ModelInfo(
            model_id="test/model",
            context_length=200000,
            max_completion_tokens=4096,
            supported_parameters=("temperature", "top_p"),
            pricing_prompt=0.001,
            pricing_completion=0.002,
        )
        with pytest.raises(FrozenInstanceError):
            info.model_id = "other"  # type: ignore[misc]

    def test_to_dict(self):
        info = ModelInfo(
            model_id="test/model",
            context_length=200000,
            max_completion_tokens=4096,
            supported_parameters=("temperature", "top_p"),
            pricing_prompt=0.001,
            pricing_completion=0.002,
        )
        d = info.to_dict()
        assert d["model_id"] == "test/model"
        assert d["context_length"] == 200000
        assert d["supported_parameters"] == ["temperature", "top_p"]  # tuple -> list


# ===================================================================
# types.py — AgentConfig
# ===================================================================


class TestAgentConfig:
    def test_defaults(self):
        cfg = AgentConfig()
        assert cfg.temperature == 1.0
        assert cfg.max_retries == 3
        assert cfg.api_timeout == 300.0
        assert cfg.max_tokens is None
        assert cfg.compaction_ratio == 0.75
        assert cfg.compaction_threshold is None

    def test_frozen(self):
        cfg = AgentConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.temperature = 0.5  # type: ignore[misc]

    def test_from_dict(self):
        cfg = AgentConfig(**{"temperature": 0.7, "max_retries": 5})
        assert cfg.temperature == 0.7
        assert cfg.max_retries == 5

    def test_unknown_keys_raise(self):
        with pytest.raises(TypeError):
            AgentConfig(**{"temperature": 0.7, "bogus_key": 42})

    def test_invalid_compaction_ratio_zero(self):
        with pytest.raises(ValueError, match="compaction_ratio"):
            AgentConfig(compaction_ratio=0.0)

    def test_invalid_compaction_ratio_one(self):
        with pytest.raises(ValueError, match="compaction_ratio"):
            AgentConfig(compaction_ratio=1.0)

    def test_invalid_compaction_ratio_negative(self):
        with pytest.raises(ValueError, match="compaction_ratio"):
            AgentConfig(compaction_ratio=-0.5)

    def test_invalid_max_retries(self):
        with pytest.raises(ValueError, match="max_retries"):
            AgentConfig(max_retries=-1)

    def test_invalid_api_timeout(self):
        with pytest.raises(ValueError, match="api_timeout"):
            AgentConfig(api_timeout=0)


# ===================================================================
# providers.py — OpenRouter
# ===================================================================


def _make_openai_response(
    content: str = "Hello",
    tool_calls: list | None = None,
    reasoning_content: str | None = None,
) -> dict:
    message: dict = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    if reasoning_content:
        message["reasoning_content"] = reasoning_content
    return {
        "choices": [{"message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 10},
        "model": "test-model",
    }


class TestOpenRouter:
    @pytest.mark.asyncio
    async def test_generate_basic(self):
        payload = _make_openai_response(content="Hi there")
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            body = json.loads(request.content)
            assert body["model"] == "test/model"
            assert body["messages"] == [{"role": "user", "content": "Hello"}]
            assert "Authorization" in request.headers
            return httpx.Response(200, json=payload)

        transport = httpx.MockTransport(handler)
        client = OpenRouter(model="test/model", api_key="sk-test")
        client._client = httpx.AsyncClient(transport=transport)

        resp = await client.generate(
            [{"role": "user", "content": "Hello"}],
            [],
        )
        assert resp.content == "Hi there"
        assert resp.model == "test-model"
        assert resp.usage == {"prompt_tokens": 5, "completion_tokens": 10}
        assert call_count == 1
        await client.close()

    @pytest.mark.asyncio
    async def test_generate_with_tools(self):
        tc_payload = [
            {
                "id": "call_abc",
                "type": "function",
                "function": {"name": "my_tool", "arguments": '{"x": 1}'},
            },
        ]
        payload = _make_openai_response(content=None, tool_calls=tc_payload)
        payload["choices"][0]["message"]["content"] = None

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert "tools" in body
            assert body["tools"][0]["type"] == "function"
            return httpx.Response(200, json=payload)

        transport = httpx.MockTransport(handler)
        client = OpenRouter(model="test/model", api_key="sk-test")
        client._client = httpx.AsyncClient(transport=transport)

        tool = ToolDef(name="my_tool", description="A tool", parameters={"type": "object"})
        resp = await client.generate(
            [{"role": "user", "content": "use tool"}],
            [tool],
        )
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "my_tool"
        assert resp.tool_calls[0].arguments == {"x": 1}
        assert resp.tool_calls[0].id == "call_abc"
        await client.close()

    @pytest.mark.asyncio
    async def test_retry_on_429(self):
        attempt = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                return httpx.Response(429, json={"error": "rate limited"})
            return httpx.Response(200, json=_make_openai_response("ok"))

        transport = httpx.MockTransport(handler)
        client = OpenRouter(model="m", api_key="k")
        client._client = httpx.AsyncClient(transport=transport)

        # Patch asyncio.sleep to avoid real delays
        import asyncio

        original_sleep = asyncio.sleep

        async def fast_sleep(seconds):
            pass

        asyncio.sleep = fast_sleep
        try:
            resp = await client.generate([{"role": "user", "content": "hi"}], [])
            assert resp.content == "ok"
            assert attempt == 2
        finally:
            asyncio.sleep = original_sleep
            await client.close()

    @pytest.mark.asyncio
    async def test_retry_on_500(self):
        attempt = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempt
            attempt += 1
            if attempt <= 2:
                return httpx.Response(500, json={"error": "server error"})
            return httpx.Response(200, json=_make_openai_response("recovered"))

        transport = httpx.MockTransport(handler)
        client = OpenRouter(model="m", api_key="k")
        client._client = httpx.AsyncClient(transport=transport)

        import asyncio

        original_sleep = asyncio.sleep

        async def fast_sleep(seconds):
            pass

        asyncio.sleep = fast_sleep
        try:
            resp = await client.generate([{"role": "user", "content": "hi"}], [])
            assert resp.content == "recovered"
            assert attempt == 3
        finally:
            asyncio.sleep = original_sleep
            await client.close()

    @pytest.mark.asyncio
    async def test_captures_reasoning(self):
        payload = _make_openai_response(content="answer", reasoning_content="I think step by step")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=payload)

        transport = httpx.MockTransport(handler)
        client = OpenRouter(model="m", api_key="k")
        client._client = httpx.AsyncClient(transport=transport)

        resp = await client.generate([{"role": "user", "content": "think"}], [])
        assert resp.reasoning == "I think step by step"
        await client.close()

    @pytest.mark.asyncio
    async def test_4xx_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(403, json={"error": "forbidden"})

        transport = httpx.MockTransport(handler)
        client = OpenRouter(model="m", api_key="k")
        client._client = httpx.AsyncClient(transport=transport)

        with pytest.raises(httpx.HTTPStatusError):
            await client.generate([{"role": "user", "content": "hi"}], [])
        await client.close()

    @pytest.mark.asyncio
    async def test_temperature_applied(self):
        """Temperature from constructor is used in request body."""
        captured_body = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured_body.update(json.loads(request.content))
            return httpx.Response(200, json=_make_openai_response("ok"))

        transport = httpx.MockTransport(handler)
        client = OpenRouter(model="m", api_key="k", temperature=0.5)
        client._client = httpx.AsyncClient(transport=transport)

        await client.generate([{"role": "user", "content": "hi"}], [])
        assert captured_body["temperature"] == 0.5
        await client.close()

    @pytest.mark.asyncio
    async def test_max_tokens_applied(self):
        """max_tokens from constructor is used in request body."""
        captured_body = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured_body.update(json.loads(request.content))
            return httpx.Response(200, json=_make_openai_response("ok"))

        transport = httpx.MockTransport(handler)
        client = OpenRouter(model="m", api_key="k", max_tokens=8192)
        client._client = httpx.AsyncClient(transport=transport)

        await client.generate([{"role": "user", "content": "hi"}], [])
        assert captured_body["max_tokens"] == 8192
        await client.close()

    @pytest.mark.asyncio
    async def test_max_retries_controls_attempts(self):
        """max_retries controls how many attempts are made on 429."""
        attempt = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempt
            attempt += 1
            return httpx.Response(429, json={"error": "rate limited"})

        transport = httpx.MockTransport(handler)
        client = OpenRouter(model="m", api_key="k", max_retries=2)
        client._client = httpx.AsyncClient(transport=transport)

        import asyncio

        original_sleep = asyncio.sleep

        async def fast_sleep(seconds):
            pass

        asyncio.sleep = fast_sleep
        try:
            with pytest.raises(httpx.HTTPStatusError):
                await client.generate([{"role": "user", "content": "hi"}], [])
            assert attempt == 2
        finally:
            asyncio.sleep = original_sleep
            await client.close()


# ===================================================================
# providers.py — fetch_openrouter_model_info
# ===================================================================


class TestFetchModelInfo:
    """Tests for fetch_openrouter_model_info."""

    SAMPLE_MODELS_RESPONSE = {
        "data": [
            {
                "id": "anthropic/claude-haiku-4.5",
                "context_length": 200000,
                "top_provider": {"max_completion_tokens": 64000},
                "supported_parameters": ["temperature", "max_tokens"],
                "pricing": {"prompt": "0.000001", "completion": "0.000005"},
            },
            {
                "id": "openai/gpt-4o",
                "context_length": 128000,
                "top_provider": {"max_completion_tokens": 16384},
                "supported_parameters": ["temperature"],
                "pricing": {"prompt": "0.000005", "completion": "0.000015"},
            },
            {
                "id": "malformed/no-context",
                # missing context_length — should be skipped
                "top_provider": {},
                "pricing": {},
            },
        ],
    }

    @pytest.fixture(autouse=True)
    def _reset_cache(self):
        """Extra reset before each test in this class."""
        _reset_model_info_cache()
        yield
        _reset_model_info_cache()

    def _mock_response(self, status_code=200, json_data=None):
        """Create a mock httpx.Response with a request set (avoids raise_for_status error)."""
        resp = httpx.Response(status_code, json=json_data)
        resp._request = httpx.Request("GET", "https://openrouter.ai/api/v1/models")
        return resp

    @pytest.mark.asyncio
    async def test_returns_model_info(self):
        """Fetches and returns correct ModelInfo for a known model."""
        mock_response = self._mock_response(200, self.SAMPLE_MODELS_RESPONSE)
        with unittest.mock.patch("agents.providers.httpx.AsyncClient") as MockClient:
            instance = MockClient.return_value
            instance.__aenter__ = unittest.mock.AsyncMock(return_value=instance)
            instance.__aexit__ = unittest.mock.AsyncMock(return_value=False)
            instance.get = unittest.mock.AsyncMock(return_value=mock_response)

            info = await fetch_openrouter_model_info("anthropic/claude-haiku-4.5")

        assert info is not None
        assert info.model_id == "anthropic/claude-haiku-4.5"
        assert info.context_length == 200000
        assert info.max_completion_tokens == 64000
        assert "temperature" in info.supported_parameters

    @pytest.mark.asyncio
    async def test_model_not_found(self):
        """Returns None for a model not in the catalog."""
        mock_response = self._mock_response(200, self.SAMPLE_MODELS_RESPONSE)
        with unittest.mock.patch("agents.providers.httpx.AsyncClient") as MockClient:
            instance = MockClient.return_value
            instance.__aenter__ = unittest.mock.AsyncMock(return_value=instance)
            instance.__aexit__ = unittest.mock.AsyncMock(return_value=False)
            instance.get = unittest.mock.AsyncMock(return_value=mock_response)

            info = await fetch_openrouter_model_info("nonexistent/model")

        assert info is None

    @pytest.mark.asyncio
    async def test_api_unreachable(self):
        """Returns None when API is unreachable."""
        with unittest.mock.patch("agents.providers.httpx.AsyncClient") as MockClient:
            instance = MockClient.return_value
            instance.__aenter__ = unittest.mock.AsyncMock(return_value=instance)
            instance.__aexit__ = unittest.mock.AsyncMock(return_value=False)
            instance.get = unittest.mock.AsyncMock(side_effect=httpx.ConnectError("Connection refused"))

            info = await fetch_openrouter_model_info("anthropic/claude-haiku-4.5")

        assert info is None

    @pytest.mark.asyncio
    async def test_cache_prevents_second_fetch(self):
        """Second call uses cache, doesn't make HTTP request."""
        mock_response = self._mock_response(200, self.SAMPLE_MODELS_RESPONSE)
        with unittest.mock.patch("agents.providers.httpx.AsyncClient") as MockClient:
            instance = MockClient.return_value
            instance.__aenter__ = unittest.mock.AsyncMock(return_value=instance)
            instance.__aexit__ = unittest.mock.AsyncMock(return_value=False)
            instance.get = unittest.mock.AsyncMock(return_value=mock_response)

            info1 = await fetch_openrouter_model_info("anthropic/claude-haiku-4.5")
            info2 = await fetch_openrouter_model_info("openai/gpt-4o")

        # Only one HTTP call made (the first populates cache for all models)
        assert instance.get.call_count == 1
        assert info1 is not None and info1.model_id == "anthropic/claude-haiku-4.5"
        assert info2 is not None and info2.model_id == "openai/gpt-4o"

    @pytest.mark.asyncio
    async def test_malformed_entry_skipped(self):
        """Entries missing context_length are skipped gracefully."""
        mock_response = self._mock_response(200, self.SAMPLE_MODELS_RESPONSE)
        with unittest.mock.patch("agents.providers.httpx.AsyncClient") as MockClient:
            instance = MockClient.return_value
            instance.__aenter__ = unittest.mock.AsyncMock(return_value=instance)
            instance.__aexit__ = unittest.mock.AsyncMock(return_value=False)
            instance.get = unittest.mock.AsyncMock(return_value=mock_response)

            info = await fetch_openrouter_model_info("malformed/no-context")

        assert info is None  # skipped due to missing context_length


# ===================================================================
# providers.py — LocalTools (sandboxing + task CRUD)
# ===================================================================


class TestLocalTools:
    @pytest.mark.asyncio
    async def test_read_write_file(self):
        store = MemoryStore()
        lt = LocalTools(store, "player1")
        await lt.open()

        result = await lt.call_tool("write_file", {"path": "notes.txt", "content": "hello"})
        assert not result.is_error
        assert "Written" in result.content

        result = await lt.call_tool("read_file", {"path": "notes.txt"})
        assert result.content == "hello"
        await lt.close()

    @pytest.mark.asyncio
    async def test_path_traversal_blocked(self):
        store = MemoryStore()
        lt = LocalTools(store, "player1")

        result = await lt.call_tool("read_file", {"path": "../etc/passwd"})
        assert result.is_error

        result = await lt.call_tool("write_file", {"path": "../../evil.txt", "content": "bad"})
        assert result.is_error

    @pytest.mark.asyncio
    async def test_absolute_path_blocked(self):
        store = MemoryStore()
        lt = LocalTools(store, "player1")

        result = await lt.call_tool("read_file", {"path": "/etc/passwd"})
        assert result.is_error

    @pytest.mark.asyncio
    async def test_task_crud(self):
        store = MemoryStore()
        lt = LocalTools(store, "p1")

        # Create
        result = await lt.call_tool("create_task", {"title": "Do thing", "details": "extra"})
        assert not result.is_error
        task_id = result.content

        # List
        result = await lt.call_tool("list_tasks", {})
        tasks = json.loads(result.content)
        assert len(tasks) == 1
        assert tasks[0]["title"] == "Do thing"
        assert tasks[0]["id"] == task_id

        # Get
        result = await lt.call_tool("get_task", {"task_id": task_id})
        task = json.loads(result.content)
        assert task["status"] == "pending"

        # Update
        result = await lt.call_tool(
            "update_task",
            {"task_id": task_id, "status": "done", "details": "finished"},
        )
        assert result.content == "Updated."

        # Verify update
        result = await lt.call_tool("get_task", {"task_id": task_id})
        task = json.loads(result.content)
        assert task["status"] == "done"
        assert task["details"] == "finished"

    @pytest.mark.asyncio
    async def test_unknown_tool(self):
        store = MemoryStore()
        lt = LocalTools(store, "p1")
        result = await lt.call_tool("nonexistent", {})
        assert result.is_error
        assert "Unknown tool" in result.content

    @pytest.mark.asyncio
    async def test_list_tools_returns_all_defs(self):
        store = MemoryStore()
        lt = LocalTools(store, "p1")
        defs = await lt.list_tools()
        names = {d.name for d in defs}
        assert names == {"read_file", "write_file", "create_task", "list_tasks", "update_task", "get_task"}


# ===================================================================
# providers.py — ToolSearch
# ===================================================================


class TestToolSearch:
    @pytest.mark.asyncio
    async def test_list_tools_returns_only_meta(self):
        inner = MockTools(
            defs=[ToolDef(name="alpha", description="Alpha tool")],
            results={"alpha": "ok"},
        )
        ts = ToolSearch(inner)
        await ts.open()

        tools = await ts.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "search_tools"
        await ts.close()

    @pytest.mark.asyncio
    async def test_search_activates_tools(self):
        inner = MockTools(
            defs=[
                ToolDef(name="alpha", description="Alpha tool"),
                ToolDef(name="beta", description="Beta runner"),
            ],
            results={"alpha": "alpha_result", "beta": "beta_result"},
        )
        ts = ToolSearch(inner)
        await ts.open()

        # Search for alpha
        result = await ts.call_tool("search_tools", {"query": "alpha"})
        assert "alpha" in result.content

        # Now alpha should be in listed tools
        tools = await ts.list_tools()
        names = {t.name for t in tools}
        assert "search_tools" in names
        assert "alpha" in names
        assert "beta" not in names

        # Calling activated tool works
        result = await ts.call_tool("alpha", {})
        assert result.content == "alpha_result"

        await ts.close()

    @pytest.mark.asyncio
    async def test_unactivated_tool_returns_error(self):
        inner = MockTools(
            defs=[ToolDef(name="alpha", description="Alpha tool")],
            results={},
        )
        ts = ToolSearch(inner)
        await ts.open()

        result = await ts.call_tool("alpha", {})
        assert result.is_error
        assert "not found" in result.content
        await ts.close()

    @pytest.mark.asyncio
    async def test_search_no_matches(self):
        inner = MockTools(
            defs=[ToolDef(name="alpha", description="Alpha tool")],
            results={},
        )
        ts = ToolSearch(inner)
        await ts.open()

        result = await ts.call_tool("search_tools", {"query": "zzz_nothing"})
        assert "No tools matched" in result.content
        await ts.close()


# ===================================================================
# providers.py — DiskStore
# ===================================================================


class TestDiskStore:
    @pytest.mark.asyncio
    async def test_write_read(self, tmp_path):
        store = DiskStore(tmp_path)
        await store.write("hello.txt", "world")
        content = await store.read("hello.txt")
        assert content == "world"

    @pytest.mark.asyncio
    async def test_read_missing(self, tmp_path):
        store = DiskStore(tmp_path)
        assert await store.read("no_such_file") is None

    @pytest.mark.asyncio
    async def test_append(self, tmp_path):
        store = DiskStore(tmp_path)
        await store.append("log.txt", "line1\n")
        await store.append("log.txt", "line2\n")
        content = await store.read("log.txt")
        assert content == "line1\nline2\n"

    @pytest.mark.asyncio
    async def test_list_keys(self, tmp_path):
        store = DiskStore(tmp_path)
        await store.write("a/one.txt", "1")
        await store.write("a/two.txt", "2")
        await store.write("b/three.txt", "3")

        all_keys = await store.list_keys()
        assert sorted(all_keys) == ["a/one.txt", "a/two.txt", "b/three.txt"]

        a_keys = await store.list_keys("a")
        assert sorted(a_keys) == ["a/one.txt", "a/two.txt"]

    @pytest.mark.asyncio
    async def test_nested_write(self, tmp_path):
        store = DiskStore(tmp_path)
        await store.write("deep/nested/file.txt", "data")
        assert await store.read("deep/nested/file.txt") == "data"


# ===================================================================
# providers.py — MemoryStore
# ===================================================================


class TestMemoryStore:
    @pytest.mark.asyncio
    async def test_write_read(self):
        store = MemoryStore()
        await store.write("k", "v")
        assert await store.read("k") == "v"

    @pytest.mark.asyncio
    async def test_read_missing(self):
        store = MemoryStore()
        assert await store.read("nope") is None

    @pytest.mark.asyncio
    async def test_append(self):
        store = MemoryStore()
        await store.append("log", "a")
        await store.append("log", "b")
        assert await store.read("log") == "ab"

    @pytest.mark.asyncio
    async def test_list_keys(self):
        store = MemoryStore()
        await store.write("prefix/a", "1")
        await store.write("prefix/b", "2")
        await store.write("other/c", "3")

        assert await store.list_keys("prefix/") == ["prefix/a", "prefix/b"]
        assert sorted(await store.list_keys()) == ["other/c", "prefix/a", "prefix/b"]


# ===================================================================
# middleware.py
# ===================================================================


class TestMiddlewareBase:
    @pytest.mark.asyncio
    async def test_all_hooks_are_noop(self):
        mw = Middleware()
        model = MockModel([])
        ctx = Context(messages=[], model=model)

        # None of these should raise
        await mw.before_agent(ctx, system_prompt="", user_prompt="", tools=[])
        await mw.before_model(ctx)
        await mw.after_model(ctx, response=_text_response("hi"))
        await mw.before_tool(ctx, tool_call=ToolCall(id="c", name="t"))
        await mw.after_tool(
            ctx,
            tool_call=ToolCall(id="c", name="t"),
            result=ToolResult(tool_call_id="c", content="ok"),
        )
        await mw.after_agent(ctx, result=RunResult())


class TestContext:
    def test_messages_mutable(self):
        msgs: list[dict] = [{"role": "user", "content": "hi"}]
        model = MockModel([])
        ctx = Context(messages=msgs, model=model)
        ctx.messages.append({"role": "assistant", "content": "hello"})
        assert len(ctx.messages) == 2
        assert msgs is ctx.messages  # same object

    def test_turn_and_usage_tracked(self):
        model = MockModel([])
        ctx = Context(messages=[], model=model, turn=0)
        assert ctx.turn == 0
        assert ctx.last_usage is None
        ctx.turn = 3
        ctx.last_usage = {"prompt_tokens": 100}
        assert ctx.turn == 3
        assert ctx.last_usage == {"prompt_tokens": 100}


class TestTranscript:
    @pytest.mark.asyncio
    async def test_writes_events(self):
        store = MemoryStore()
        transcript = Transcript(store, "game_log")
        model = MockModel([])
        ctx = Context(messages=[{"role": "user", "content": "hi"}], model=model)

        await transcript.before_agent(ctx, system_prompt="sys", user_prompt="hi", tools=[])
        await transcript.before_model(ctx)

        response = _text_response("hello")
        await transcript.after_model(ctx, response=response)

        tc = ToolCall(id="c1", name="do_thing")
        await transcript.before_tool(ctx, tool_call=tc)

        tr = ToolResult(tool_call_id="c1", content="done")
        await transcript.after_tool(ctx, tool_call=tc, result=tr)

        ctx.turn = 1
        await transcript.after_agent(ctx, result=RunResult(turns=1))

        raw = await store.read("game_log")
        assert raw is not None
        lines = [line for line in raw.strip().split("\n") if line]
        assert len(lines) == 6

        events = [json.loads(line) for line in lines]
        event_types = [e["type"] for e in events]
        assert event_types == [
            "agent_start",
            "model_request",
            "model_response",
            "tool_call",
            "tool_result",
            "agent_end",
        ]

        # Verify timestamps exist
        for e in events:
            assert "timestamp" in e

        # Verify specific fields
        assert events[0]["system_prompt"] == "sys"
        assert events[1]["message_count"] == 1
        assert events[5]["turn"] == 1


class TestCompactor:
    @pytest.mark.asyncio
    async def test_triggers_at_threshold(self):
        summary_model = MockModel([_text_response("Summary of conversation")])
        compactor = Compactor(
            threshold_tokens=100,
            keep_recent=2,
            compaction_model=summary_model,
        )

        model = MockModel([])
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "reply1"},
            {"role": "user", "content": "msg2"},
            {"role": "assistant", "content": "reply2"},
            {"role": "user", "content": "msg3"},
            {"role": "assistant", "content": "reply3"},
        ]
        ctx = Context(messages=messages, model=model)

        response = ModelResponse(
            content="latest",
            usage={"prompt_tokens": 200},  # above threshold
        )

        await compactor.after_model(ctx, response=response)

        # Should have: system + summary + 2 recent
        assert len(ctx.messages) == 4
        assert ctx.messages[0]["role"] == "system"
        assert ctx.messages[0]["content"] == "You are helpful"
        assert "[Compacted conversation summary]" in ctx.messages[1]["content"]
        assert "Summary of conversation" in ctx.messages[1]["content"]
        assert ctx.messages[2] == {"role": "user", "content": "msg3"}
        assert ctx.messages[3] == {"role": "assistant", "content": "reply3"}

    @pytest.mark.asyncio
    async def test_no_compaction_below_threshold(self):
        compactor = Compactor(threshold_tokens=1000, keep_recent=2)

        model = MockModel([])
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "reply1"},
        ]
        ctx = Context(messages=messages, model=model)

        response = ModelResponse(content="ok", usage={"prompt_tokens": 50})
        await compactor.after_model(ctx, response=response)

        # Messages unchanged
        assert len(ctx.messages) == 3

    @pytest.mark.asyncio
    async def test_no_compaction_too_few_messages(self):
        compactor = Compactor(threshold_tokens=10, keep_recent=4)

        model = MockModel([])
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "reply1"},
        ]
        ctx = Context(messages=messages, model=model)

        response = ModelResponse(content="ok", usage={"prompt_tokens": 500})
        await compactor.after_model(ctx, response=response)

        # Not enough messages to compact (3 <= keep_recent+1)
        assert len(ctx.messages) == 3


# ===================================================================
# agent.py
# ===================================================================


class TestAgent:
    @pytest.mark.asyncio
    async def test_simple_text_response(self):
        model = MockModel([_text_response("Hello world")])
        agent = Agent(model=model, system_prompt="You are helpful")
        result = await agent.run("Say hello")

        assert result.final_content == "Hello world"
        assert result.turns == 1
        assert result.error is None
        assert result.tool_calls_made == 0
        await agent.close()

    @pytest.mark.asyncio
    async def test_single_tool_call(self):
        tool_def = ToolDef(name="greet", description="Greet someone")
        mock_tools = MockTools(defs=[tool_def], results={"greet": "Greeted!"})

        model = MockModel(
            [
                _tool_call_response("greet", {"name": "Alice"}),
                _text_response("Done greeting Alice"),
            ]
        )
        agent = Agent(model=model, tools=[mock_tools])
        result = await agent.run("Greet Alice")

        assert result.final_content == "Done greeting Alice"
        assert result.tool_calls_made == 1
        assert result.turns == 2
        assert mock_tools.opened
        assert mock_tools.closed

    @pytest.mark.asyncio
    async def test_multi_turn_tool_chain(self):
        tool_a = ToolDef(name="tool_a", description="Tool A")
        tool_b = ToolDef(name="tool_b", description="Tool B")
        mock_tools = MockTools(
            defs=[tool_a, tool_b],
            results={"tool_a": "result_a", "tool_b": "result_b"},
        )

        model = MockModel(
            [
                _tool_call_response("tool_a", {}, call_id="c1"),
                _tool_call_response("tool_b", {}, call_id="c2"),
                _text_response("All done"),
            ]
        )
        agent = Agent(model=model, tools=[mock_tools])
        result = await agent.run("Do both")

        assert result.final_content == "All done"
        assert result.tool_calls_made == 2
        assert result.turns == 3

    @pytest.mark.asyncio
    async def test_max_turns_exceeded(self):
        tool_def = ToolDef(name="loop", description="Looping tool")
        mock_tools = MockTools(defs=[tool_def], results={"loop": "looped"})

        # Model always calls tool — never finishes
        responses = [_tool_call_response("loop", {}, call_id=f"c{i}") for i in range(5)]
        model = MockModel(responses)
        agent = Agent(model=model, tools=[mock_tools])
        result = await agent.run("loop forever", max_turns=3)

        assert result.turns == 3
        assert result.error is not None
        assert "Max turns" in result.error

    @pytest.mark.asyncio
    async def test_persistent_conversation(self):
        model = MockModel(
            [
                _text_response("First reply"),
                _text_response("Second reply"),
            ]
        )
        agent = Agent(model=model, system_prompt="You are helpful")

        r1 = await agent.run("Message 1")
        assert r1.final_content == "First reply"

        r2 = await agent.run("Message 2")
        assert r2.final_content == "Second reply"

        # Messages should contain both conversations
        # system + user1 + assistant1 + user2 + assistant2
        assert len(agent._messages) == 5
        assert agent._messages[0]["role"] == "system"
        assert agent._messages[1]["content"] == "Message 1"
        assert agent._messages[3]["content"] == "Message 2"

    @pytest.mark.asyncio
    async def test_system_prompt_not_duplicated(self):
        model = MockModel(
            [
                _text_response("r1"),
                _text_response("r2"),
            ]
        )
        agent = Agent(model=model, system_prompt="sys prompt")

        await agent.run("m1")
        await agent.run("m2")

        system_messages = [m for m in agent._messages if m["role"] == "system"]
        assert len(system_messages) == 1
        assert system_messages[0]["content"] == "sys prompt"

    @pytest.mark.asyncio
    async def test_submit_action_causes_early_exit(self):
        tool_def = ToolDef(name="submit_action", description="Submit game action")
        mock_tools = MockTools(defs=[tool_def], results={"submit_action": "submitted"})

        model = MockModel(
            [
                _tool_call_response("submit_action", {"action": "vote_yes"}, call_id="c1"),
                # This response should NOT be consumed
                _text_response("should not reach here"),
            ]
        )
        agent = Agent(model=model, tools=[mock_tools])
        result = await agent.run("Do action")

        assert result.tool_calls_made == 1
        assert result.turns == 1
        # Model was called only once
        assert model._call_count == 1

    @pytest.mark.asyncio
    async def test_middleware_hooks_fire_in_order(self):
        spy = SpyMiddleware()
        tool_def = ToolDef(name="my_tool", description="A tool")
        mock_tools = MockTools(defs=[tool_def], results={"my_tool": "ok"})

        model = MockModel(
            [
                _tool_call_response("my_tool", {}, call_id="c1"),
                _text_response("done"),
            ]
        )
        agent = Agent(model=model, tools=[mock_tools], middleware=[spy])
        await agent.run("do it")

        assert spy.calls == [
            "before_agent",
            # Turn 1: tool call
            "before_model",
            "after_model",
            "before_tool",
            "after_tool",
            # Turn 2: text response
            "before_model",
            "after_model",
            # End
            "after_agent",
        ]

    @pytest.mark.asyncio
    async def test_middleware_error_doesnt_crash_agent(self):
        crash = CrashMiddleware()
        model = MockModel([_text_response("still works")])
        agent = Agent(model=model, middleware=[crash])
        result = await agent.run("hello")

        # Agent completes despite middleware errors
        assert result.final_content == "still works"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_tool_routing_to_correct_provider(self):
        """Tools from different providers are routed correctly."""
        tool_a = ToolDef(name="tool_a", description="From provider A")
        tool_b = ToolDef(name="tool_b", description="From provider B")

        provider_a = MockTools(defs=[tool_a], results={"tool_a": "from_a"})
        provider_b = MockTools(defs=[tool_b], results={"tool_b": "from_b"})

        model = MockModel(
            [
                _tool_call_response("tool_a", {}, call_id="c1"),
                _tool_call_response("tool_b", {}, call_id="c2"),
                _text_response("all done"),
            ]
        )
        agent = Agent(model=model, tools=[provider_a, provider_b])
        result = await agent.run("use both tools")

        assert result.tool_calls_made == 2
        assert result.final_content == "all done"

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self):
        model = MockModel(
            [
                _tool_call_response("nonexistent_tool", {}, call_id="c1"),
                _text_response("recovered"),
            ]
        )
        agent = Agent(model=model)
        result = await agent.run("try missing tool")

        assert result.tool_calls_made == 1
        # Agent should recover after unknown tool error
        assert result.final_content == "recovered"


class TestRunResult:
    def test_to_dict(self):
        r = RunResult(
            turns=3,
            timed_out=False,
            total_input_tokens=100,
            total_output_tokens=50,
            tool_calls_made=2,
            error=None,
            final_content="done",
        )
        d = r.to_dict()
        assert d == {
            "turns": 3,
            "timed_out": False,
            "total_input_tokens": 100,
            "total_output_tokens": 50,
            "tool_calls_made": 2,
            "error": None,
            "final_content": "done",
        }

    def test_frozen(self):
        r = RunResult()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            r.turns = 5  # type: ignore[misc]
