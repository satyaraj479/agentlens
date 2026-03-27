"""
FastAPI server — serves both the REST/SSE API and the embedded HTML visualizer.

Run directly:
    python -m agentlens.server

Or embed in an existing FastAPI app:
    from agentlens.server import create_app
    app = create_app()
"""
from __future__ import annotations
import asyncio
import json
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

from .bus import get_bus, init_bus
from .schema import AgentEvent, EventType


# ---------------------------------------------------------------------------
# HTML UI (inline — no separate static files needed)
# ---------------------------------------------------------------------------
_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AgentLens</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#0f1117;color:#e2e8f0;height:100vh;display:flex;flex-direction:column}

/* --- header --- */
#header{display:flex;align-items:center;gap:12px;padding:10px 18px;background:#1a1d27;border-bottom:1px solid #2d3148;flex-shrink:0}
#header h1{font-size:15px;font-weight:600;color:#a78bfa;letter-spacing:.3px}
#status-dot{width:8px;height:8px;border-radius:50%;background:#22c55e;flex-shrink:0}
#status-dot.disconnected{background:#ef4444}
#session-tabs{display:flex;gap:4px;overflow-x:auto;flex:1}
.stab{padding:4px 12px;border-radius:6px;font-size:12px;cursor:pointer;border:1px solid transparent;white-space:nowrap;color:#94a3b8}
.stab.active{background:#2d3148;border-color:#4c4f8a;color:#e2e8f0}
.stab:hover:not(.active){background:#1e2130}
#clear-btn{padding:4px 10px;font-size:12px;border-radius:6px;border:1px solid #3d4168;background:transparent;color:#94a3b8;cursor:pointer}
#clear-btn:hover{background:#2d3148;color:#e2e8f0}

/* --- main layout --- */
#main{display:flex;flex:1;overflow:hidden}
#graph-pane{flex:1;position:relative;overflow:hidden}
#sidebar{width:320px;flex-shrink:0;background:#141720;border-left:1px solid #2d3148;display:flex;flex-direction:column;overflow:hidden}

/* --- graph canvas --- */
#graph-svg{width:100%;height:100%}
.node-group{cursor:pointer}
.node-rect{rx:8;stroke-width:1.5}
.node-label{font-size:12px;font-weight:500;pointer-events:none}
.node-sub{font-size:10px;pointer-events:none;opacity:.7}
.link{fill:none;stroke:#4c4f8a;stroke-width:1.5;marker-end:url(#arrowhead)}
.link.tool{stroke:#f59e0b}
.link.llm{stroke:#a78bfa}

/* node fill by type */
.n-agent_start .node-rect,.n-agent_end .node-rect{fill:#1e2d3d;stroke:#3b82f6}
.n-tool_start .node-rect,.n-tool_end .node-rect{fill:#2d2212;stroke:#f59e0b}
.n-llm_start .node-rect,.n-llm_end .node-rect{fill:#251e36;stroke:#a78bfa}
.n-tool_error .node-rect,.n-agent_error .node-rect{fill:#2d1515;stroke:#ef4444}
.n-session_start .node-rect{fill:#0f1f2d;stroke:#0ea5e9;stroke-dasharray:4}

/* selected / hover */
.node-group:hover .node-rect{filter:brightness(1.3)}
.node-group.selected .node-rect{stroke-width:2.5}

/* --- sidebar tabs --- */
#sidebar-tabs{display:flex;border-bottom:1px solid #2d3148;flex-shrink:0}
.sptab{flex:1;padding:8px 4px;font-size:12px;text-align:center;cursor:pointer;color:#64748b;border-bottom:2px solid transparent}
.sptab.active{color:#e2e8f0;border-bottom-color:#a78bfa}

/* --- event log --- */
#event-log{flex:1;overflow-y:auto;padding:8px;font-size:11px;font-family:"JetBrains Mono","Fira Code",monospace}
.elog-row{padding:4px 6px;border-radius:4px;margin-bottom:2px;display:flex;gap:8px;cursor:pointer;border:1px solid transparent}
.elog-row:hover{background:#1e2130}
.elog-row.selected{border-color:#4c4f8a;background:#1e2130}
.elog-type{font-weight:600;min-width:100px}
.elog-type.et-agent_start,.elog-type.et-agent_end{color:#3b82f6}
.elog-type.et-tool_start,.elog-type.et-tool_end{color:#f59e0b}
.elog-type.et-llm_start,.elog-type.et-llm_end{color:#a78bfa}
.elog-type.et-agent_error,.elog-type.et-tool_error{color:#ef4444}
.elog-type.et-session_start{color:#0ea5e9}
.elog-ts{color:#475569;min-width:60px}
.elog-label{color:#94a3b8;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}

/* --- detail pane --- */
#detail-pane{flex:1;overflow-y:auto;padding:12px;font-size:12px}
.dp-field{margin-bottom:10px}
.dp-key{color:#64748b;font-size:10px;text-transform:uppercase;letter-spacing:.5px;margin-bottom:2px}
.dp-val{color:#e2e8f0;word-break:break-all;white-space:pre-wrap;font-family:"JetBrains Mono","Fira Code",monospace;font-size:11px}
.dp-badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600}
.dp-badge.agent_start,.dp-badge.agent_end{background:#1e2d3d;color:#3b82f6}
.dp-badge.tool_start,.dp-badge.tool_end{background:#2d2212;color:#f59e0b}
.dp-badge.llm_start,.dp-badge.llm_end{background:#251e36;color:#a78bfa}
.dp-badge.agent_error,.dp-badge.tool_error{background:#2d1515;color:#ef4444}

/* --- timeline --- */
#timeline-pane{flex:1;overflow-y:auto;padding:8px}
.tl-row{display:flex;align-items:center;gap:6px;margin-bottom:4px;font-size:11px}
.tl-label{width:110px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#94a3b8;text-align:right;flex-shrink:0}
.tl-bar-wrap{flex:1;background:#1e2130;border-radius:3px;height:14px;position:relative;overflow:hidden}
.tl-bar{height:100%;border-radius:3px;min-width:2px}
.tl-dur{position:absolute;right:4px;top:0;line-height:14px;font-size:10px;color:#475569}

/* --- scrollbar --- */
::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:#2d3148;border-radius:2px}

/* empty state */
#empty-state{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px;pointer-events:none;opacity:.4}
#empty-state p{font-size:13px;color:#94a3b8}
#empty-state code{font-size:11px;color:#64748b}
</style>
</head>
<body>

<div id="header">
  <div id="status-dot" class="disconnected"></div>
  <h1>⬡ AgentLens</h1>
  <div id="session-tabs"></div>
  <button id="clear-btn" onclick="clearSession()">Clear</button>
</div>

<div id="main">
  <div id="graph-pane">
    <div id="empty-state">
      <p>Waiting for agent events…</p>
      <code>from agentlens.adapters.langgraph import AgentFlowCallback</code>
    </div>
    <svg id="graph-svg"></svg>
  </div>

  <div id="sidebar">
    <div id="sidebar-tabs">
      <div class="sptab active" onclick="showTab('log')" id="tab-log">Events</div>
      <div class="sptab" onclick="showTab('detail')" id="tab-detail">Detail</div>
      <div class="sptab" onclick="showTab('timeline')" id="tab-timeline">Timeline</div>
    </div>
    <div id="event-log"></div>
    <div id="detail-pane" style="display:none"></div>
    <div id="timeline-pane" style="display:none"></div>
  </div>
</div>

<script>
// =====================================================================
// State
// =====================================================================
let sessions = {};          // session_id -> {events:[], nodes:{}, links:[]}
let activeSession = null;
let selectedEventId = null;
let evtSource = null;

// =====================================================================
// SSE connection
// =====================================================================
function connect() {
  if (evtSource) evtSource.close();
  evtSource = new EventSource('/stream');
  evtSource.onopen = () => {
    document.getElementById('status-dot').className = '';
  };
  evtSource.onmessage = (e) => {
    if (e.data === '__heartbeat__') return;
    try { handleEvent(JSON.parse(e.data)); } catch(_) {}
  };
  evtSource.onerror = () => {
    document.getElementById('status-dot').className = 'disconnected';
    setTimeout(connect, 3000);
  };
}

// =====================================================================
// Event handling
// =====================================================================
function handleEvent(evt) {
  const sid = evt.session_id;
  if (!sessions[sid]) sessions[sid] = {events:[], nodes:{}, links:[], label:''};
  const s = sessions[sid];
  s.events.push(evt);

  if (evt.type === 'session_start') {
    s.label = (evt.metadata && evt.metadata.label) || sid;
  }

  // Build graph nodes & links
  buildGraph(s, evt);

  if (!activeSession) setActiveSession(sid);
  if (activeSession === sid) {
    renderAll();
  }
  renderTabs();
}

function buildGraph(s, evt) {
  const type = evt.type;

  // Create or update node
  if (evt.node_id && evt.node_id !== evt.session_id) {
    if (!s.nodes[evt.node_id]) {
      s.nodes[evt.node_id] = {
        id: evt.node_id,
        label: evt.node_label || evt.tool_name || evt.node_id,
        type: type,
        events: [],
        duration: 0,
        tool_name: evt.tool_name || '',
        model: evt.model || '',
      };
    }
    const n = s.nodes[evt.node_id];
    n.events.push(evt);
    if (evt.duration_ms) n.duration = Math.max(n.duration, evt.duration_ms);
    // Keep most informative type
    if (type.endsWith('_end') || type.endsWith('_error')) n.type = type;

    // Add link from parent
    if (evt.parent_node_id && evt.parent_node_id !== evt.session_id && evt.parent_node_id !== evt.node_id) {
      const linkId = `${evt.parent_node_id}__${evt.node_id}`;
      if (!s.links.find(l => l.id === linkId)) {
        s.links.push({id: linkId, source: evt.parent_node_id, target: evt.node_id, type: type});
      }
    }
  }
}

// =====================================================================
// Render
// =====================================================================
function renderAll() {
  renderGraph();
  renderLog();
  renderTimeline();
}

// --- Graph (D3 force layout) ---
let simulation = null;
let zoom = null;
const svg = d3.select('#graph-svg');
let gMain = null;

function initSvg() {
  svg.selectAll('*').remove();
  svg.append('defs').html(`
    <marker id="arrowhead" viewBox="0 0 10 10" refX="9" refY="5"
      markerWidth="7" markerHeight="7" orient="auto">
      <path d="M2 1L9 5L2 9" fill="none" stroke="#4c4f8a" stroke-width="1.5"
            stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
    <marker id="arrowhead-tool" viewBox="0 0 10 10" refX="9" refY="5"
      markerWidth="7" markerHeight="7" orient="auto">
      <path d="M2 1L9 5L2 9" fill="none" stroke="#f59e0b" stroke-width="1.5"
            stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
    <marker id="arrowhead-llm" viewBox="0 0 10 10" refX="9" refY="5"
      markerWidth="7" markerHeight="7" orient="auto">
      <path d="M2 1L9 5L2 9" fill="none" stroke="#a78bfa" stroke-width="1.5"
            stroke-linecap="round" stroke-linejoin="round"/>
    </marker>`);
  zoom = d3.zoom().scaleExtent([0.2, 4]).on('zoom', e => gMain.attr('transform', e.transform));
  svg.call(zoom);
  gMain = svg.append('g');
}

function renderGraph() {
  if (!activeSession || !sessions[activeSession]) return;
  const s = sessions[activeSession];
  const nodeArr = Object.values(s.nodes);
  const linkArr = s.links.filter(l => s.nodes[l.source] && s.nodes[l.target]);

  const empty = document.getElementById('empty-state');
  empty.style.display = nodeArr.length ? 'none' : 'flex';

  initSvg();
  if (!nodeArr.length) return;

  const W = document.getElementById('graph-pane').clientWidth;
  const H = document.getElementById('graph-pane').clientHeight;
  const NW = 150, NH = 50;

  simulation = d3.forceSimulation(nodeArr)
    .force('link', d3.forceLink(linkArr).id(d => d.id).distance(120))
    .force('charge', d3.forceManyBody().strength(-400))
    .force('center', d3.forceCenter(W / 2, H / 2))
    .force('collision', d3.forceCollide(90));

  const link = gMain.append('g').selectAll('path')
    .data(linkArr).join('path')
    .attr('class', d => `link ${d.type.includes('tool') ? 'tool' : d.type.includes('llm') ? 'llm' : ''}`)
    .attr('marker-end', d => {
      if (d.type.includes('tool')) return 'url(#arrowhead-tool)';
      if (d.type.includes('llm')) return 'url(#arrowhead-llm)';
      return 'url(#arrowhead)';
    });

  const node = gMain.append('g').selectAll('g')
    .data(nodeArr).join('g')
    .attr('class', d => `node-group n-${d.type}`)
    .call(d3.drag()
      .on('start', (e, d) => { if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; })
      .on('drag',  (e, d) => { d.fx=e.x; d.fy=e.y; })
      .on('end',   (e, d) => { if (!e.active) simulation.alphaTarget(0); d.fx=null; d.fy=null; }))
    .on('click', (e, d) => { selectNode(d); e.stopPropagation(); });

  node.append('rect')
    .attr('class', 'node-rect')
    .attr('width', NW).attr('height', NH)
    .attr('x', -NW/2).attr('y', -NH/2);

  node.append('text')
    .attr('class', 'node-label')
    .attr('text-anchor', 'middle')
    .attr('dy', d => d.duration ? '-6px' : '4px')
    .text(d => truncate(d.label || d.id, 18));

  node.filter(d => !!d.duration)
    .append('text')
    .attr('class', 'node-sub')
    .attr('text-anchor', 'middle')
    .attr('dy', '12px')
    .attr('fill', '#94a3b8')
    .text(d => `${d.duration.toFixed(0)}ms`);

  // Type badge
  node.append('text')
    .attr('class', 'node-sub')
    .attr('text-anchor', 'middle')
    .attr('dy', '-18px')
    .text(d => typeEmoji(d.type));

  simulation.on('tick', () => {
    link.attr('d', d => {
      const dx = d.target.x - d.source.x;
      const dy = d.target.y - d.source.y;
      const dr = Math.sqrt(dx*dx + dy*dy);
      return `M${d.source.x},${d.source.y} A${dr},${dr} 0 0,1 ${d.target.x},${d.target.y}`;
    });
    node.attr('transform', d => `translate(${d.x},${d.y})`);
  });
}

function typeEmoji(t) {
  if (t.includes('tool')) return '⚙';
  if (t.includes('llm'))  return '◈';
  if (t.includes('agent')) return '◎';
  return '';
}

// --- Event log ---
function renderLog() {
  if (!activeSession || !sessions[activeSession]) return;
  const s = sessions[activeSession];
  const el = document.getElementById('event-log');
  const baseTime = s.events[0] ? s.events[0].timestamp : 0;
  el.innerHTML = s.events.map((evt, i) => {
    const relMs = ((evt.timestamp - baseTime) * 1000).toFixed(0);
    const label = evt.node_label || evt.tool_name || evt.model || evt.node_id || '';
    return `<div class="elog-row${selectedEventId === evt.event_id ? ' selected' : ''}"
               onclick="selectEvent(${i})">
      <span class="elog-ts">+${relMs}ms</span>
      <span class="elog-type et-${evt.type}">${evt.type}</span>
      <span class="elog-label">${escHtml(label)}</span>
    </div>`;
  }).join('');
  el.scrollTop = el.scrollHeight;
}

// --- Timeline ---
function renderTimeline() {
  if (!activeSession || !sessions[activeSession]) return;
  const s = sessions[activeSession];
  const pane = document.getElementById('timeline-pane');
  const nodes = Object.values(s.nodes).filter(n => n.duration > 0);
  if (!nodes.length) { pane.innerHTML = '<p style="color:#475569;padding:16px;font-size:12px">No duration data yet.</p>'; return; }
  const maxDur = Math.max(...nodes.map(n => n.duration));
  const colors = {agent:'#3b82f6', tool:'#f59e0b', llm:'#a78bfa'};
  pane.innerHTML = nodes.sort((a,b) => b.duration - a.duration).map(n => {
    const cat = n.type.includes('tool') ? 'tool' : n.type.includes('llm') ? 'llm' : 'agent';
    const pct = Math.max(2, (n.duration / maxDur) * 100);
    return `<div class="tl-row">
      <div class="tl-label" title="${escHtml(n.label)}">${escHtml(truncate(n.label,14))}</div>
      <div class="tl-bar-wrap">
        <div class="tl-bar" style="width:${pct}%;background:${colors[cat]}44;border-right:2px solid ${colors[cat]}"></div>
        <span class="tl-dur">${n.duration.toFixed(1)}ms</span>
      </div>
    </div>`;
  }).join('');
}

// =====================================================================
// Interaction
// =====================================================================
function selectEvent(idx) {
  if (!activeSession || !sessions[activeSession]) return;
  const s = sessions[activeSession];
  const evt = s.events[idx];
  if (!evt) return;
  selectedEventId = evt.event_id;
  showTab('detail');
  renderDetail(evt);
  renderLog();
}

function selectNode(node) {
  if (!activeSession || !sessions[activeSession]) return;
  const s = sessions[activeSession];
  // Find last event for this node
  const evt = [...s.events].reverse().find(e => e.node_id === node.id);
  if (evt) { selectedEventId = evt.event_id; renderDetail(evt); showTab('detail'); renderLog(); }
  // Highlight
  d3.selectAll('.node-group').classed('selected', d => d.id === node.id);
}

function renderDetail(evt) {
  const pane = document.getElementById('detail-pane');
  const fields = [
    ['type', `<span class="dp-badge ${evt.type}">${evt.type}</span>`],
    ['event id', evt.event_id],
    ['session', evt.session_id],
    ['node', evt.node_label || evt.node_id || '-'],
    ['duration', evt.duration_ms ? `${evt.duration_ms.toFixed(2)} ms` : '-'],
    ['timestamp', new Date(evt.timestamp * 1000).toLocaleTimeString()],
  ];
  if (evt.tool_name)    fields.push(['tool', evt.tool_name]);
  if (evt.tool_input)   fields.push(['input', JSON.stringify(evt.tool_input, null, 2)]);
  if (evt.tool_output)  fields.push(['output', JSON.stringify(evt.tool_output, null, 2)]);
  if (evt.model)        fields.push(['model', evt.model]);
  if (evt.prompt_tokens)      fields.push(['prompt tokens', evt.prompt_tokens]);
  if (evt.completion_tokens)  fields.push(['completion tokens', evt.completion_tokens]);
  if (evt.metadata && Object.keys(evt.metadata).length) {
    fields.push(['metadata', JSON.stringify(evt.metadata, null, 2)]);
  }
  pane.innerHTML = fields.map(([k,v]) =>
    `<div class="dp-field"><div class="dp-key">${k}</div><div class="dp-val">${escHtml(String(v))}</div></div>`
  ).join('');
}

// =====================================================================
// Session tabs
// =====================================================================
function renderTabs() {
  const el = document.getElementById('session-tabs');
  el.innerHTML = Object.entries(sessions).map(([sid, s]) =>
    `<div class="stab${activeSession === sid ? ' active' : ''}" onclick="setActiveSession('${sid}')">
      ${escHtml(s.label || sid)}
      <span style="color:#475569;margin-left:4px">${s.events.length}</span>
    </div>`
  ).join('');
}

function setActiveSession(sid) {
  activeSession = sid;
  selectedEventId = null;
  renderTabs();
  renderAll();
}

function clearSession() {
  if (!activeSession) return;
  fetch(`/sessions/${activeSession}`, {method:'DELETE'});
  delete sessions[activeSession];
  activeSession = Object.keys(sessions)[0] || null;
  renderTabs();
  renderAll();
}

// =====================================================================
// Sidebar tabs
// =====================================================================
function showTab(name) {
  ['log','detail','timeline'].forEach(t => {
    const pane = document.getElementById(t === 'log' ? 'event-log' : t + '-pane');
    const tab  = document.getElementById('tab-' + t);
    pane.style.display = t === name ? '' : 'none';
    tab.className = 'sptab' + (t === name ? ' active' : '');
  });
}

// =====================================================================
// Helpers
// =====================================================================
function truncate(s, n) { return s && s.length > n ? s.slice(0, n-1) + '…' : s; }
function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// =====================================================================
// Boot
// =====================================================================
initSvg();
svg.on('click', () => { d3.selectAll('.node-group').classed('selected', false); });
connect();

// Poll sessions list on startup to populate from history
fetch('/sessions').then(r => r.json()).then(sids => {
  sids.forEach(sid => {
    fetch(`/sessions/${sid}/events`).then(r => r.json()).then(evts => {
      evts.forEach(handleEvent);
    });
  });
});

window.addEventListener('resize', () => { if (activeSession) renderGraph(); });
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# FastAPI app factory
# ---------------------------------------------------------------------------
def create_app(jsonl_path: str | None = None) -> FastAPI:
    bus = init_bus(jsonl_path=jsonl_path) if jsonl_path else get_bus()

    app = FastAPI(title="AgentLens", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", response_class=HTMLResponse)
    async def ui():
        return _HTML

    @app.get("/stream")
    async def stream(session_id: str | None = None):
        async def event_generator():
            async for payload in bus.subscribe(session_id=session_id, replay=True):
                if payload == "__heartbeat__":
                    yield "data: __heartbeat__\n\n"
                else:
                    yield f"data: {payload}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/event")
    async def post_event(request: Request):
        """HTTP endpoint — lets external processes push events to the bus."""
        data = await request.json()
        try:
            evt_type = EventType(data.get("type", "custom"))
        except ValueError:
            evt_type = EventType.CUSTOM
        data["type"] = evt_type
        event = AgentEvent(**{k: v for k, v in data.items() if k in AgentEvent.__dataclass_fields__})
        bus.publish(event)
        return {"ok": True}

    @app.get("/sessions")
    async def list_sessions():
        return [s["session_id"] for s in bus.get_sessions()]

    @app.get("/sessions/{session_id}/events")
    async def session_events(session_id: str):
        return bus.get_history(session_id)

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str):
        bus.clear_session(session_id)
        return {"ok": True}

    return app


# ---------------------------------------------------------------------------
# __main__ entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="AgentLens visualizer server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7788)
    parser.add_argument("--jsonl", default=None, help="Path to JSONL event log file")
    parser.add_argument("--open", action="store_true", help="Auto-open browser")
    args = parser.parse_args()

    app = create_app(jsonl_path=args.jsonl)

    if args.open:
        import threading, webbrowser
        threading.Timer(1.0, lambda: webbrowser.open(f"http://{args.host}:{args.port}")).start()

    print(f"\n  ⬡  AgentLens  →  http://{args.host}:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
