"""
agentlens — real-time visualizer for LangGraph, PydanticAI, and Agno agents.
"""
from .bus import get_bus, init_bus
from .schema import AgentEvent, EventType


def start_server(
    host: str = "127.0.0.1",
    port: int = 7788,
    open_browser: bool = False,
    background: bool = False,
    jsonl_path: str | None = None,
):
    """
    Start the Agent Flow FastAPI server.

    Parameters
    ----------
    host, port:
        Where to bind the server.
    open_browser:
        Automatically open the UI in the default browser.
    background:
        Run uvicorn in a daemon thread so the caller can continue.
    jsonl_path:
        Optional path to append events as JSONL for offline replay.
    """
    import uvicorn
    from .server import create_app

    app = create_app(jsonl_path=jsonl_path)

    if open_browser:
        import threading, webbrowser
        threading.Timer(1.2, lambda: webbrowser.open(f"http://{host}:{port}")).start()

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    if background:
        import threading
        t = threading.Thread(target=server.run, daemon=True)
        t.start()
    else:
        server.run()


__all__ = ["start_server", "get_bus", "init_bus", "AgentEvent", "EventType"]
