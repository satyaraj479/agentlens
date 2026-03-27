"""
LangGraph adapter for agentlens.

Usage
-----
from agentlens.adapters.langgraph_adapter import AgentFlowCallback
from langgraph.graph import StateGraph

callback = AgentFlowCallback(session_id="my-run")
graph.invoke(input, config={"callbacks": [callback]})
"""
from __future__ import annotations
import time
import uuid
from typing import Any, Union

from ..bus import get_bus
from ..schema import AgentEvent, EventType

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.outputs import LLMResult
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False
    # Provide a no-op base so the class can still be imported
    class BaseCallbackHandler:  # type: ignore
        def __init__(self, *a, **kw): pass


class AgentFlowCallback(BaseCallbackHandler):
    """
    Drop-in LangChain/LangGraph callback handler that emits AgentEvents to
    the agentlens event bus.

    Parameters
    ----------
    session_id:
        Stable identifier for this graph run. Auto-generated if not provided.
    session_label:
        Human-readable name shown in the UI tab.
    bus:
        Override the module-level event bus (useful for testing).
    """

    def __init__(
        self,
        session_id: str | None = None,
        session_label: str = "",
        bus=None,
    ):
        super().__init__()
        if not _LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain-core is required for AgentFlowCallback. "
                "Install it with: pip install langchain-core"
            )
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.session_label = session_label or f"LangGraph run {self.session_id}"
        self._bus = bus or get_bus()
        # Track per-run_id start times for duration calculation
        self._start: dict[str, float] = {}
        # Emit session start
        self._emit(EventType.SESSION_START, metadata={"label": self.session_label})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _emit(self, etype: EventType, run_id: str = "", **kwargs) -> None:
        start = self._start.pop(run_id, None) if run_id else None
        duration = (time.time() - start) * 1000 if start else 0.0
        event = AgentEvent(
            type=etype,
            session_id=self.session_id,
            duration_ms=round(duration, 2),
            **kwargs,
        )
        self._bus.publish(event)

    def _node_id(self, run_id: str | None, name: str) -> str:
        return f"{name}:{str(run_id)[:8]}" if run_id else name

    # ------------------------------------------------------------------
    # Chain (= graph node) callbacks
    # ------------------------------------------------------------------
    def on_chain_start(
        self,
        serialized: dict,
        inputs: dict,
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        metadata=None,
        **kwargs,
    ) -> None:
        rid = str(run_id)
        self._start[rid] = time.time()
        name = (serialized or {}).get("name", "") or (metadata or {}).get("langgraph_node", "node")
        self._emit(
            EventType.AGENT_START,
            run_id=rid,
            node_id=self._node_id(run_id, name),
            node_label=name,
            parent_node_id=self._node_id(parent_run_id, "") if parent_run_id else "",
            metadata={"inputs_keys": list(inputs.keys()) if isinstance(inputs, dict) else []},
        )
        # restore start time (was popped by _emit)
        self._start[rid] = time.time()

    def on_chain_end(self, outputs: dict, *, run_id, **kwargs) -> None:
        rid = str(run_id)
        self._emit(
            EventType.AGENT_END,
            run_id=rid,
            node_id=self._node_id(run_id, ""),
            metadata={"output_keys": list(outputs.keys()) if isinstance(outputs, dict) else []},
        )

    def on_chain_error(self, error: Exception, *, run_id, **kwargs) -> None:
        rid = str(run_id)
        self._emit(
            EventType.AGENT_ERROR,
            run_id=rid,
            node_id=self._node_id(run_id, ""),
            metadata={"error": str(error)},
        )

    # ------------------------------------------------------------------
    # LLM callbacks
    # ------------------------------------------------------------------
    def on_llm_start(
        self,
        serialized: dict,
        prompts: list[str],
        *,
        run_id,
        parent_run_id=None,
        **kwargs,
    ) -> None:
        rid = str(run_id)
        self._start[rid] = time.time()
        model = (serialized or {}).get("kwargs", {}).get("model_name", "") or \
                (serialized or {}).get("name", "llm")
        self._emit(
            EventType.LLM_START,
            run_id=rid,
            node_id=self._node_id(run_id, model),
            node_label=model,
            parent_node_id=self._node_id(parent_run_id, "") if parent_run_id else "",
            model=model,
            metadata={"num_prompts": len(prompts)},
        )
        self._start[rid] = time.time()

    def on_llm_end(self, response: "LLMResult", *, run_id, **kwargs) -> None:
        rid = str(run_id)
        usage = {}
        try:
            usage = response.llm_output.get("token_usage", {}) or {}
        except Exception:
            pass
        self._emit(
            EventType.LLM_END,
            run_id=rid,
            node_id=self._node_id(run_id, ""),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )

    def on_llm_new_token(self, token: str, *, run_id, **kwargs) -> None:
        self._emit(
            EventType.LLM_STREAM_TOKEN,
            node_id=self._node_id(run_id, ""),
            metadata={"token": token},
        )

    # ------------------------------------------------------------------
    # Tool callbacks
    # ------------------------------------------------------------------
    def on_tool_start(
        self,
        serialized: dict,
        input_str: str,
        *,
        run_id,
        parent_run_id=None,
        **kwargs,
    ) -> None:
        rid = str(run_id)
        self._start[rid] = time.time()
        name = (serialized or {}).get("name", "tool")
        self._emit(
            EventType.TOOL_START,
            run_id=rid,
            node_id=self._node_id(run_id, name),
            node_label=name,
            parent_node_id=self._node_id(parent_run_id, "") if parent_run_id else "",
            tool_name=name,
            tool_input=input_str,
        )
        self._start[rid] = time.time()

    def on_tool_end(self, output: str, *, run_id, **kwargs) -> None:
        rid = str(run_id)
        self._emit(
            EventType.TOOL_END,
            run_id=rid,
            node_id=self._node_id(run_id, ""),
            tool_output=str(output)[:500],  # truncate large outputs
        )

    def on_tool_error(self, error: Exception, *, run_id, **kwargs) -> None:
        rid = str(run_id)
        self._emit(
            EventType.TOOL_ERROR,
            run_id=rid,
            node_id=self._node_id(run_id, ""),
            metadata={"error": str(error)},
        )
