"""Middleware infrastructure for the agent loop."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from agents.types import ModelResponse, ToolCall, ToolResult

if TYPE_CHECKING:
    from agents.providers import Model, Store


# ---------------------------------------------------------------------------
# Context — mutable state threaded through all middleware hooks
# ---------------------------------------------------------------------------


class Context:
    """Mutable state passed to every middleware hook."""

    def __init__(
        self,
        messages: list[dict],
        model: Model,
        turn: int = 0,
        last_usage: dict | None = None,
    ) -> None:
        self.messages = messages
        self.model = model
        self.turn = turn
        self.last_usage = last_usage


# ---------------------------------------------------------------------------
# Middleware — base class (all hooks async no-ops by default)
# ---------------------------------------------------------------------------


class Middleware:
    """Base middleware — override only the hooks you need."""

    async def before_agent(
        self,
        ctx: Context,
        *,
        system_prompt: str,
        user_prompt: str,
        tools: list,
    ) -> None:
        pass

    async def before_model(self, ctx: Context) -> None:
        pass

    async def after_model(self, ctx: Context, *, response: ModelResponse) -> None:
        pass

    async def before_tool(self, ctx: Context, *, tool_call: ToolCall) -> None:
        pass

    async def after_tool(
        self,
        ctx: Context,
        *,
        tool_call: ToolCall,
        result: ToolResult,
    ) -> None:
        pass

    async def after_agent(self, ctx: Context, *, result: object) -> None:
        pass


# ---------------------------------------------------------------------------
# Transcript — JSONL event recorder
# ---------------------------------------------------------------------------


class Transcript(Middleware):
    """Records agent events as JSONL lines via a Store."""

    def __init__(self, store: Store, key: str) -> None:
        self._store = store
        self._key = key

    async def _emit(self, event_type: str, data: dict[str, Any]) -> None:
        event = {
            "type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **data,
        }
        await self._store.append(self._key, json.dumps(event, default=str) + "\n")

    async def before_agent(
        self,
        ctx: Context,
        *,
        system_prompt: str,
        user_prompt: str,
        tools: list,
    ) -> None:
        await self._emit(
            "agent_start",
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            },
        )

    async def before_model(self, ctx: Context) -> None:
        await self._emit("model_request", {"message_count": len(ctx.messages)})

    async def after_model(self, ctx: Context, *, response: ModelResponse) -> None:
        await self._emit("model_response", {"response": response.to_dict()})

    async def before_tool(self, ctx: Context, *, tool_call: ToolCall) -> None:
        await self._emit("tool_call", {"tool_call": tool_call.to_dict()})

    async def after_tool(
        self,
        ctx: Context,
        *,
        tool_call: ToolCall,
        result: ToolResult,
    ) -> None:
        await self._emit("tool_result", {"result": result.to_dict()})

    async def after_agent(self, ctx: Context, *, result: object) -> None:
        info: dict[str, Any] = {"turn": ctx.turn}
        if hasattr(result, "to_dict"):
            info["result"] = result.to_dict()  # type: ignore[union-attr]
        await self._emit("agent_end", info)


# ---------------------------------------------------------------------------
# Compactor — context-window compaction
# ---------------------------------------------------------------------------

_COMPACTION_PROMPT = (
    "Summarize the following conversation history concisely. "
    "Preserve all key facts, decisions, tool results, and game state "
    "information. Be thorough but brief."
)


class Compactor(Middleware):
    """Compacts the context window when prompt tokens exceed a threshold."""

    def __init__(
        self,
        threshold_tokens: int = 25_000,
        keep_recent: int = 6,
        compaction_model: Model | None = None,
    ) -> None:
        self._threshold = threshold_tokens
        self._keep_recent = keep_recent
        self._compaction_model = compaction_model

    async def after_model(self, ctx: Context, *, response: ModelResponse) -> None:
        prompt_tokens = response.usage.get("prompt_tokens", 0)
        if prompt_tokens < self._threshold:
            return

        # Need at least: system msg + some old msgs + keep_recent
        if len(ctx.messages) <= self._keep_recent + 1:
            return

        system_msg = ctx.messages[0]
        old_messages = ctx.messages[1 : -self._keep_recent]
        recent_messages = ctx.messages[-self._keep_recent :]

        # Build the compaction request
        old_text = json.dumps(old_messages, default=str)
        compaction_messages = [
            {"role": "system", "content": _COMPACTION_PROMPT},
            {"role": "user", "content": old_text},
        ]

        model = self._compaction_model or ctx.model
        summary_response = await model.generate(compaction_messages, [])

        summary_msg = {
            "role": "system",
            "content": ("[Compacted conversation summary]\n" + (summary_response.content or "")),
        }

        # Update ctx.messages in place
        ctx.messages[:] = [system_msg, summary_msg, *recent_messages]
