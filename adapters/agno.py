"""
Agno adapter for agentlens.

Agno agents support a `hooks` parameter.  We provide an AgentFlowHook class
that implements the Agno hook protocol and emits AgentEvents.

Usage
-----
from agentlens.adapters.agno import AgentFlowHook
from agno.agent import Agent

hook = AgentFlowHook(session_id="my-run")
agent = Agent(model=..., tools=[...], hooks=[hook])
agent.print_response("Hello")

If you need to instrument an already-created agent:
    hook.attach(agent)
"""
from __future__ import annotations
import functools
import time
import uuid
from typing import Any

from ..bus import get_bus
from ..schema import AgentEvent, EventType


class AgentFlowHook:
    """
    Agno hook that emits AgentEvents to the agentlens bus.

    Implements the Agno hook interface:
      - on_run_start / on_run_end
      - on_tool_call_start / on_tool_call_end
      - on_model_response
    """

    def __init__(
        self,
        session_id: str | None = None,
        session_label: str = "",
        bus=None,
    ):
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.session_label = session_label or f"Agno run {self.session_id}"
        self._bus = bus or get_bus()
        self._run_starts: dict[str, float] = {}
        self._tool_starts: dict[str, float] = {}
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
    # Agno hook protocol
    # ------------------------------------------------------------------

    def on_run_start(self, agent, message: str, **kwargs) -> None:
        nid = f"agent:{self.session_id}"
        self._run_starts[nid] = time.time()
        self._emit(
            EventType.AGENT_START,
            node_id=nid,
            node_label=getattr(agent, "name", "agent"),
            metadata={"message_preview": str(message)[:120]},
        )

    def on_run_end(self, agent, response, **kwargs) -> None:
        nid = f"agent:{self.session_id}"
        t0 = self._run_starts.pop(nid, time.time())
        self._emit(
            EventType.AGENT_END,
            node_id=nid,
            node_label=getattr(agent, "name", "agent"),
            duration_ms=round((time.time() - t0) * 1000, 2),
        )

    def on_run_error(self, agent, error: Exception, **kwargs) -> None:
        nid = f"agent:{self.session_id}"
        t0 = self._run_starts.pop(nid, time.time())
        self._emit(
            EventType.AGENT_ERROR,
            node_id=nid,
            metadata={"error": str(error)},
            duration_ms=round((time.time() - t0) * 1000, 2),
        )

    def on_tool_call_start(self, agent, tool_name: str, tool_input: Any, **kwargs) -> None:
        call_id = str(uuid.uuid4())[:8]
        nid = f"tool:{tool_name}:{call_id}"
        self._tool_starts[nid] = time.time()
        self._emit(
            EventType.TOOL_START,
            node_id=nid,
            node_label=tool_name,
            tool_name=tool_name,
            tool_input=str(tool_input)[:300],
        )

    def on_tool_call_end(
        self, agent, tool_name: str, tool_input: Any, tool_output: Any, **kwargs
    ) -> None:
        # Reconstruct most-recent nid for this tool
        matching = [k for k in self._tool_starts if k.startswith(f"tool:{tool_name}:")]
        nid = matching[-1] if matching else f"tool:{tool_name}:?"
        t0 = self._tool_starts.pop(nid, time.time())
        self._emit(
            EventType.TOOL_END,
            node_id=nid,
            node_label=tool_name,
            tool_name=tool_name,
            tool_output=str(tool_output)[:300],
            duration_ms=round((time.time() - t0) * 1000, 2),
        )

    def on_model_response(self, agent, response, **kwargs) -> None:
        usage = getattr(response, "usage", None) or {}
        if hasattr(usage, "__dict__"):
            usage = usage.__dict__
        self._emit(
            EventType.LLM_END,
            node_id=f"llm:{self.session_id}",
            node_label="llm",
            model=str(getattr(response, "model", "")),
            prompt_tokens=usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0) or usage.get("completion_tokens", 0),
        )

    # ------------------------------------------------------------------
    # Attach to an existing agent (monkey-patch)
    # ------------------------------------------------------------------
    def attach(self, agent) -> None:
        """
        Attach this hook to an already-instantiated Agno agent.
        Works by appending to agent.hooks if the list exists,
        otherwise falls back to patching agent.run.
        """
        if hasattr(agent, "hooks") and isinstance(agent.hooks, list):
            agent.hooks.append(self)
        else:
            self._patch_run(agent)

    def _patch_run(self, agent) -> None:
        original = agent.run
        hook = self

        @functools.wraps(original)
        def patched(message, **kw):
            hook.on_run_start(agent, message)
            t0 = time.time()
            try:
                result = original(message, **kw)
                hook.on_run_end(agent, result)
                return result
            except Exception as e:
                hook.on_run_error(agent, e)
                raise

        agent.run = patched
