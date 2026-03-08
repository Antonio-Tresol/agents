"""Agent — the ReAct loop that drives model + tool interactions."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from agents.middleware import Context, Middleware
from agents.providers import Model, Tools
from agents.types import ModelResponse, ToolDef, ToolResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunResult:
    """Outcome of a single ``Agent.run()`` invocation."""

    turns: int = 0
    timed_out: bool = False
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    tool_calls_made: int = 0
    error: str | None = None
    final_content: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "turns": self.turns,
            "timed_out": self.timed_out,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "tool_calls_made": self.tool_calls_made,
            "error": self.error,
            "final_content": self.final_content,
        }


class Agent:
    """ReAct agent that drives a model with tool providers and middleware.

    Conversation history (``_messages``) persists across ``run()`` calls so the
    agent maintains context across turns in a game.
    """

    def __init__(
        self,
        model: Model,
        tools: list[Tools] | None = None,
        middleware: list[Middleware] | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self._model = model
        self._tools: list[Tools] = tools or []
        self._middleware: list[Middleware] = middleware or []
        self._system_prompt = system_prompt
        self._messages: list[dict[str, Any]] = []
        self._system_set = False

    # ── public ──

    async def run(
        self,
        user_prompt: str,
        *,
        max_turns: int = 10,
        timeout: float = 120.0,
    ) -> RunResult:
        """Execute one agent invocation (may span many model calls)."""

        # Ensure system message is present exactly once
        if not self._system_set and self._system_prompt:
            self._messages.insert(0, {"role": "system", "content": self._system_prompt})
            self._system_set = True

        # Append user message
        self._messages.append({"role": "user", "content": user_prompt})

        # Open all tool providers
        failed_providers: list[str] = []
        for tp in self._tools:
            try:
                await tp.open()
            except Exception:
                logger.exception("Failed to open tool provider %s", tp)
                failed_providers.append(str(tp))
        if failed_providers:
            if len(failed_providers) == len(self._tools):
                raise RuntimeError(
                    f"All tool providers failed to open: {', '.join(failed_providers)}",
                )
            logger.warning(
                "Some tool providers failed to open: %s",
                ", ".join(failed_providers),
            )

        # Collect tools + build routing map
        tool_defs, routing = await self._collect_tools()

        # Build context
        ctx = Context(messages=self._messages, model=self._model)

        # Fire before_agent
        await self._fire(
            "before_agent",
            ctx,
            system_prompt=self._system_prompt or "",
            user_prompt=user_prompt,
            tools=tool_defs,
        )

        deadline = time.monotonic() + timeout
        total_in = 0
        total_out = 0
        total_tool_calls = 0
        last_content: str | None = None
        error: str | None = None
        timed_out = False
        turn = 0

        try:
            for turn in range(1, max_turns + 1):
                # Timeout check
                if time.monotonic() >= deadline:
                    timed_out = True
                    error = "Timeout exceeded"
                    break

                ctx.turn = turn

                # before_model
                await self._fire("before_model", ctx)

                # Generate
                remaining = max(deadline - time.monotonic(), 1.0)
                try:
                    response = await asyncio.wait_for(
                        self._model.generate(ctx.messages, tool_defs),
                        timeout=remaining,
                    )
                except asyncio.TimeoutError:
                    timed_out = True
                    error = "Timeout exceeded during model generation"
                    break

                # Update stats
                ctx.last_usage = response.usage
                total_in += response.usage.get("prompt_tokens", 0)
                total_out += response.usage.get("completion_tokens", 0)

                # after_model (Compactor runs here)
                await self._fire("after_model", ctx, response=response)

                # Append assistant message
                assistant_msg = self._build_assistant_message(response)
                ctx.messages.append(assistant_msg)

                last_content = response.content

                # No tool calls → done
                if not response.tool_calls:
                    break

                # Execute tool calls
                submit_action_called = False
                for tc in response.tool_calls:
                    total_tool_calls += 1

                    await self._fire("before_tool", ctx, tool_call=tc)

                    provider = routing.get(tc.name)
                    if provider is not None:
                        result = await provider.call_tool(tc.name, tc.arguments)
                        result = ToolResult(
                            tool_call_id=tc.id,
                            content=result.content,
                            is_error=result.is_error,
                        )
                    else:
                        result = ToolResult(
                            tool_call_id=tc.id,
                            content=f"Unknown tool: {tc.name}",
                            is_error=True,
                        )

                    await self._fire("after_tool", ctx, tool_call=tc, result=result)

                    ctx.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result.content,
                        },
                    )

                    if tc.name == "submit_action":
                        submit_action_called = True

                if submit_action_called:
                    break

                # Re-list tools (ToolSearch may have activated new ones)
                tool_defs, routing = await self._collect_tools()

        finally:
            # Close all tool providers
            for tp in self._tools:
                try:
                    await tp.close()
                except Exception:
                    logger.exception("Failed to close tool provider %s", tp)

        if turn >= max_turns and not timed_out and error is None:
            # Check if we actually ran out of turns (loop exhausted without break)
            if response.tool_calls:  # type: ignore[possibly-undefined]
                error = f"Max turns ({max_turns}) exceeded"

        result = RunResult(
            turns=turn,
            timed_out=timed_out,
            total_input_tokens=total_in,
            total_output_tokens=total_out,
            tool_calls_made=total_tool_calls,
            error=error,
            final_content=last_content,
        )

        await self._fire("after_agent", ctx, result=result)
        return result

    async def close(self) -> None:
        """Release the underlying model client."""
        await self._model.close()

    # ── private ──

    async def _collect_tools(self) -> tuple[list[ToolDef], dict[str, Tools]]:
        """Gather tool definitions from all providers, build routing map."""
        all_defs: list[ToolDef] = []
        routing: dict[str, Tools] = {}
        for tp in self._tools:
            try:
                defs = await tp.list_tools()
                for d in defs:
                    all_defs.append(d)
                    routing[d.name] = tp
            except Exception:
                logger.exception("Failed to list tools from %s", tp)
        return all_defs, routing

    async def _fire(self, hook_name: str, ctx: Context, **kwargs: Any) -> None:
        """Call a middleware hook on all middleware, swallowing errors."""
        for mw in self._middleware:
            try:
                method = getattr(mw, hook_name, None)
                if method is not None:
                    await method(ctx, **kwargs)
            except Exception:
                logger.exception(
                    "Middleware %s.%s failed",
                    type(mw).__name__,
                    hook_name,
                )

    @staticmethod
    def _build_assistant_message(response: ModelResponse) -> dict[str, Any]:
        """Convert a ModelResponse to an OpenAI-style assistant message dict."""
        msg: dict[str, Any] = {"role": "assistant"}
        if response.content:
            msg["content"] = response.content
        if response.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in response.tool_calls
            ]
        return msg
