"""
Thread-safe event bus.
The FastAPI SSE endpoint subscribes here; all framework adapters publish here.
"""
from __future__ import annotations
import asyncio
import json
import queue
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import AsyncGenerator

from .schema import AgentEvent


class EventBus:
    def __init__(self, jsonl_path: str | None = None):
        self._lock = threading.Lock()
        # session_id -> list of events (history)
        self._history: dict[str, list[AgentEvent]] = defaultdict(list)
        # active sessions metadata
        self._sessions: dict[str, dict] = {}
        # subscriber queues  (for SSE)
        self._queues: list[asyncio.Queue] = []
        # optional JSONL log
        self._jsonl_path = Path(jsonl_path) if jsonl_path else None
        if self._jsonl_path:
            self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Publish (called from framework adapters, may be in any thread)
    # ------------------------------------------------------------------
    def publish(self, event: AgentEvent) -> None:
        with self._lock:
            self._history[event.session_id].append(event)
            if event.session_id not in self._sessions:
                self._sessions[event.session_id] = {
                    "session_id": event.session_id,
                    "started_at": event.timestamp,
                    "event_count": 0,
                }
            self._sessions[event.session_id]["event_count"] += 1
            self._sessions[event.session_id]["last_event_at"] = event.timestamp

        if self._jsonl_path:
            with open(self._jsonl_path, "a") as f:
                f.write(json.dumps(event.to_dict()) + "\n")

        # Push to all waiting SSE subscribers (thread-safe via call_soon_threadsafe)
        payload = json.dumps(event.to_dict())
        for q in list(self._queues):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                pass

    # ------------------------------------------------------------------
    # Subscribe (async generator for FastAPI SSE)
    # ------------------------------------------------------------------
    async def subscribe(
        self,
        session_id: str | None = None,
        replay: bool = True,
    ) -> AsyncGenerator[str, None]:
        q: asyncio.Queue = asyncio.Queue(maxsize=500)
        with self._lock:
            self._queues.append(q)
            # Replay history for the requested session (or all if None)
            history = []
            if replay:
                if session_id:
                    history = list(self._history.get(session_id, []))
                else:
                    for evts in self._history.values():
                        history.extend(evts)
                history.sort(key=lambda e: e.timestamp)

        try:
            for evt in history:
                yield json.dumps(evt.to_dict())

            while True:
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=15.0)
                    if session_id is None or json.loads(payload).get("session_id") == session_id:
                        yield payload
                except asyncio.TimeoutError:
                    yield "__heartbeat__"
        finally:
            with self._lock:
                self._queues.remove(q)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def get_sessions(self) -> list[dict]:
        with self._lock:
            return sorted(self._sessions.values(), key=lambda s: s["started_at"])

    def get_history(self, session_id: str) -> list[dict]:
        with self._lock:
            return [e.to_dict() for e in self._history.get(session_id, [])]

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            self._history.pop(session_id, None)
            self._sessions.pop(session_id, None)


# Module-level singleton — adapters import this directly
_bus: EventBus | None = None


def get_bus() -> EventBus:
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus


def init_bus(jsonl_path: str | None = None) -> EventBus:
    global _bus
    _bus = EventBus(jsonl_path=jsonl_path)
    return _bus
