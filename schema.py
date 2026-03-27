"""
Unified event schema for agentlens.
All framework adapters emit AgentEvent instances.
"""
from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class EventType(str, Enum):
    # Lifecycle
    SESSION_START    = "session_start"
    SESSION_END      = "session_end"
    # Agent / node
    AGENT_START      = "agent_start"
    AGENT_END        = "agent_end"
    AGENT_ERROR      = "agent_error"
    # LLM
    LLM_START        = "llm_start"
    LLM_END          = "llm_end"
    LLM_STREAM_TOKEN = "llm_stream_token"
    # Tool calls
    TOOL_START       = "tool_start"
    TOOL_END         = "tool_end"
    TOOL_ERROR       = "tool_error"
    # Edges / routing
    EDGE             = "edge"
    # Generic
    CUSTOM           = "custom"


@dataclass
class AgentEvent:
    type: EventType
    session_id: str
    # Node / agent that emitted this event
    node_id: str              = ""
    node_label: str           = ""
    parent_node_id: str       = ""
    # Tool info (for TOOL_* events)
    tool_name: str            = ""
    tool_input: Any           = None
    tool_output: Any          = None
    # LLM info
    model: str                = ""
    prompt_tokens: int        = 0
    completion_tokens: int    = 0
    # Edge info
    edge_from: str            = ""
    edge_to: str              = ""
    edge_label: str           = ""
    # Timing
    duration_ms: float        = 0.0
    timestamp: float          = field(default_factory=time.time)
    # Arbitrary extra data
    metadata: dict            = field(default_factory=dict)
    # Auto-assigned unique id
    event_id: str             = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def to_dict(self) -> dict:
        d = asdict(self)
        d["type"] = self.type.value
        return d
