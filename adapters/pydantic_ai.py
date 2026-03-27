"""
PydanticAI adapter for agentlens.

PydanticAI exposes an `instrument` context manager / decorator.
We wrap it to emit AgentEvents before/after each agent call and tool call.

Usage
-----
from agentlens.adapters.pydantic_ai import AgentFlowInstrument
from pydantic_ai import Agent

instrument = AgentFlowInstrument(session_id="my-run")

agent = Agent("openai:gpt-4o", ...)
with instrument.run_context():
    result = await agent.run("Hello")

# OR — patch the agent directly (works for sync and async):
instrument.patch(agent)
result = await agent.run("Hello")
"""
from __future__ import annotations
import functools
import time
import uuid
from typing import Any

from ..bus import get_bus
from ..schema import AgentEvent, EventType


class AgentFlowInstrument:
    """
    Lightweight wrapper around a PydanticAI Agent that emits AgentEvents.

    PydanticAI does not (as of v0.x) expose a native callback protocol, so we
    monkey-patch the agent's `run` / `run_sync` methods and register tool
    wrappers via the agent's tool registry.
    """

    def __init__(
        self,
        session_id: str | None = None,
        session_label: str = "",
        bus=None,
    ):
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.session_label = session_label or f"PydanticAI run {self.session_id}"
        self._bus = bus or get_bus()
        self._emit(EventType.SESSION_START, metadata={"label": self.session_label})

    # ------------------------------------------------------------------
    def _emit(self, etype: EventType, node_id: str = "", node_label: str = "", **kwargs):
        self._bus.publish(AgentEvent(
            type=etype,
            session_id=self.session_id,
            node_id=node_id or self.session_id,
            node_label=node_label,
            **kwargs,
        ))

    # ------------------------------------------------------------------
    def patch(self, agent) -> None:
        """
        Monkey-patch agent.run and agent.run_sync to emit lifecycle events.
        Also wraps any already-registered tools.
        """
        self._patch_run(agent)
        self._patch_tools(agent)

    def _patch_run(self, agent) -> None:
        original_run = agent.run
        original_run_sync = getattr(agent, "run_sync", None)
        sid = self.session_id

        @functools.wraps(original_run)
        async def patched_run(user_prompt, **kw):
            nid = f"agent:{sid}"
            self._emit(EventType.AGENT_START, node_id=nid, node_label="agent.run",
                       metadata={"prompt_preview": str(user_prompt)[:120]})
            t0 = time.time()
            try:
                result = await original_run(user_prompt, **kw)
                self._emit(EventType.AGENT_END, node_id=nid,
                            duration_ms=round((time.time() - t0) * 1000, 2))
                return result
            except Exception as e:
                self._emit(EventType.AGENT_ERROR, node_id=nid,
                            metadata={"error": str(e)},
                            duration_ms=round((time.time() - t0) * 1000, 2))
                raise

        agent.run = patched_run

        if original_run_sync:
            @functools.wraps(original_run_sync)
            def patched_run_sync(user_prompt, **kw):
                import asyncio
                return asyncio.get_event_loop().run_until_complete(patched_run(user_prompt, **kw))
            agent.run_sync = patched_run_sync

    def _patch_tools(self, agent) -> None:
        """
        Wrap each tool's function to emit TOOL_START / TOOL_END events.
        Works with PydanticAI's internal _function_tools list.
        """
        tools = getattr(agent, "_function_tools", {})
        for tool_name, tool_obj in tools.items():
            self._wrap_tool(tool_obj, tool_name)

    def _wrap_tool(self, tool_obj, tool_name: str) -> None:
        original_fn = getattr(tool_obj, "function", None) or getattr(tool_obj, "_function", None)
        if original_fn is None:
            return

        @functools.wraps(original_fn)
        async def wrapped(*args, **kwargs):
            nid = f"tool:{tool_name}:{self.session_id}"
            self._emit(EventType.TOOL_START, node_id=nid, node_label=tool_name,
                        tool_name=tool_name,
                        tool_input=str(kwargs or args)[:300])
            t0 = time.time()
            try:
                result = await original_fn(*args, **kwargs)
                self._emit(EventType.TOOL_END, node_id=nid,
                            tool_name=tool_name,
                            tool_output=str(result)[:300],
                            duration_ms=round((time.time() - t0) * 1000, 2))
                return result
            except Exception as e:
                self._emit(EventType.TOOL_ERROR, node_id=nid,
                            tool_name=tool_name,
                            metadata={"error": str(e)},
                            duration_ms=round((time.time() - t0) * 1000, 2))
                raise

        # Patch back onto the tool object
        for attr in ("function", "_function"):
            if hasattr(tool_obj, attr):
                setattr(tool_obj, attr, wrapped)
                break

    # ------------------------------------------------------------------
    def manual_tool_start(self, tool_name: str, tool_input: Any = None) -> str:
        """
        Use this when auto-patching is not possible.
        Returns a call_id you pass to manual_tool_end().
        """
        call_id = str(uuid.uuid4())[:8]
        self._emit(EventType.TOOL_START,
                    node_id=f"tool:{tool_name}:{call_id}",
                    node_label=tool_name,
                    tool_name=tool_name,
                    tool_input=str(tool_input)[:300])
        return call_id

    def manual_tool_end(self, tool_name: str, call_id: str, output: Any = None, duration_ms: float = 0):
        self._emit(EventType.TOOL_END,
                    node_id=f"tool:{tool_name}:{call_id}",
                    tool_name=tool_name,
                    tool_output=str(output)[:300],
                    duration_ms=duration_ms)
