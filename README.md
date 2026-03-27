# ⬡ AgentLens

> Real-time visualizer for **LangGraph**, **PydanticAI**, and **Agno** agents.

Stop debugging AI pipelines with `print()` statements. AgentLens gives you a live, interactive node graph of your agent's execution — right in your browser, as it runs. Entirely on `localhost`. Nothing leaves your machine.

Inspired by [patoles/agent-flow](https://github.com/patoles/agent-flow) — built for Python.

---

## Features

- **Live node graph** — D3 force-directed graph of agent nodes, tool calls, and LLM calls, updating in real time
- **SSE streaming** — zero-latency event push to the browser via Server-Sent Events
- **Multi-session tabs** — run multiple agents concurrently; each gets its own tab
- **Timeline panel** — horizontal bars showing duration of every node (spot slow tools instantly)
- **Event log** — full timestamped event stream with click-through detail view
- **JSONL export** — optional file-based logging for offline replay and analysis
- **No cloud** — runs entirely on `localhost`, zero external services required

---

## Install

```bash
git clone https://github.com/satyaraj479/agentlens.git
cd agentlens

pip install -e ".[langgraph]"    # LangGraph / LangChain only
pip install -e ".[pydantic-ai]"  # PydanticAI only
pip install -e ".[agno]"         # Agno only
pip install -e ".[all]"          # All frameworks
```

---

## Quick Start

### 1. Start the server

```python
from agentlens import start_server
start_server(open_browser=True)   # → opens http://127.0.0.1:7788
```

Or from the terminal:

```bash
python -m agentlens.server --open
```

### 2. LangGraph

```python
from agentlens import start_server
from agentlens.adapters.langgraph_adapter import AgentFlowCallback

start_server(open_browser=True, background=True)

callback = AgentFlowCallback(session_label="My RAG pipeline")
result = graph.invoke(
    {"question": "What is RAG?"},
    config={"callbacks": [callback]},
)
```

### 3. PydanticAI

```python
from agentlens import start_server
from agentlens.adapters.pydantic_ai import AgentFlowInstrument
from pydantic_ai import Agent

start_server(open_browser=True, background=True)

instrument = AgentFlowInstrument(session_label="PydanticAI agent")
agent = Agent("openai:gpt-4o", ...)
instrument.patch(agent)

result = await agent.run("Summarise this document")
```

### 4. Agno

```python
from agentlens import start_server
from agentlens.adapters.agno import AgentFlowHook
from agno.agent import Agent

start_server(open_browser=True, background=True)

hook = AgentFlowHook(session_label="Agno research agent")
agent = Agent(model=..., tools=[...], hooks=[hook])
agent.print_response("Find the latest AI papers")
```

### 5. Run the zero-API-key demo

```bash
python examples/langgraph_example.py
# → opens http://127.0.0.1:7788 and simulates a 5-stage research pipeline
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Browser UI |
| GET | `/stream?session_id=X` | SSE event stream |
| POST | `/event` | Push a raw event (JSON body) |
| GET | `/sessions` | List active session IDs |
| GET | `/sessions/{id}/events` | Full event history for a session |
| DELETE | `/sessions/{id}` | Clear a session |

---

## Event Schema

Every adapter emits `AgentEvent` instances with these core fields:

```python
AgentEvent(
    type           = EventType.TOOL_START,   # enum
    session_id     = "abc123",
    node_id        = "tool:web_search:001",
    node_label     = "web_search",
    parent_node_id = "planner:001",
    tool_name      = "web_search",
    tool_input     = "LangGraph 2025",
    duration_ms    = 0.0,                    # filled on *_end events
    timestamp      = 1712345678.123,
    metadata       = {},
)
```

---

## Project Structure

```
agentlens/
├── __init__.py              start_server() convenience entry point
├── schema.py                AgentEvent dataclass + EventType enum
├── bus.py                   Thread-safe event bus + SSE pub/sub
├── server.py                FastAPI app + embedded HTML/JS UI
└── adapters/
    ├── langgraph_adapter.py  LangChain BaseCallbackHandler subclass
    ├── pydantic_ai.py        Monkey-patch wrapper for PydanticAI agents
    └── agno.py               Agno hook protocol implementation
examples/
└── langgraph_example.py     Simulated graph run (no API key needed)
```

---

## License

MIT — see [LICENSE](LICENSE).
