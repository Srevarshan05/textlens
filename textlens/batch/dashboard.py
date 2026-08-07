"""
textlens.batch.dashboard
─────────────────────────
Live monitoring dashboard for BatchOCR.

Serves a real-time web dashboard at http://localhost:8765 (default) with:
- Live KPI metrics (speed, ETA, workers, CPU/VRAM)
- Task list with per-file status and duration
- Streaming live logs via Server-Sent Events (SSE)
- Interactive controls: pause, resume, cancel, retry failed
- Runtime reconfiguration: workers, format, retries

Uses Python's built-in `http.server` to keep zero extra dependencies.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import TYPE_CHECKING, Any, Dict
from urllib.parse import urlparse

if TYPE_CHECKING:
    from textlens.batch.engine import BatchOCR

logger = logging.getLogger("textlens.batch.dashboard")

# ── Embedded SPA HTML ────────────────────────────────────────────────────────
_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>TextLens BatchOCR Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet"/>
<style>
  :root {
    --bg: #0d0f1a; --surface: #131625; --surface2: #1a1e34;
    --border: rgba(99,179,237,0.15); --primary: #63b3ed;
    --accent: #9f7aea; --success: #68d391; --warn: #fbd38d;
    --danger: #fc8181; --text: #e2e8f0; --muted: #718096;
    --glass: rgba(19,22,37,0.8);
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }
  header {
    background: var(--glass); backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border); padding: 14px 28px;
    display: flex; align-items: center; justify-content: space-between;
    position: sticky; top: 0; z-index: 100;
  }
  .logo { display: flex; align-items: center; gap: 10px; }
  .logo span { font-size: 1.1rem; font-weight: 700; letter-spacing: -0.5px; }
  .logo-icon { width: 30px; height: 30px; background: linear-gradient(135deg,var(--primary),var(--accent)); border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 1rem; }
  .status-badge { padding: 4px 12px; border-radius: 20px; font-size: 0.72rem; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; }
  .badge-RUNNING { background: rgba(104,211,145,0.15); color: var(--success); border: 1px solid rgba(104,211,145,0.3); }
  .badge-PAUSED  { background: rgba(251,211,141,0.15); color: var(--warn);    border: 1px solid rgba(251,211,141,0.3); }
  .badge-IDLE    { background: rgba(113,128,150,0.15); color: var(--muted);   border: 1px solid rgba(113,128,150,0.3); }
  .badge-CANCELLED { background: rgba(252,129,129,0.15); color: var(--danger); border: 1px solid rgba(252,129,129,0.3); }
  .badge-COMPLETED { background: rgba(99,179,237,0.15); color: var(--primary); border: 1px solid rgba(99,179,237,0.3); }
  .badge-FAILED  { background: rgba(252,129,129,0.15); color: var(--danger); border: 1px solid rgba(252,129,129,0.3); }

  .main { max-width: 1400px; margin: 0 auto; padding: 24px 20px; }

  /* KPI Grid */
  .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px,1fr)); gap: 14px; margin-bottom: 22px; }
  .kpi-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 14px; padding: 18px 16px;
    transition: transform 0.2s, box-shadow 0.2s;
  }
  .kpi-card:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,0.3); }
  .kpi-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); margin-bottom: 6px; }
  .kpi-value { font-size: 1.8rem; font-weight: 700; line-height: 1; }
  .kpi-sub { font-size: 0.72rem; color: var(--muted); margin-top: 4px; }
  .kpi-total .kpi-value { color: var(--primary); }
  .kpi-done  .kpi-value { color: var(--success); }
  .kpi-failed .kpi-value { color: var(--danger); }
  .kpi-queue .kpi-value { color: var(--warn); }

  /* Progress Bar */
  .progress-section { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 20px; margin-bottom: 22px; }
  .progress-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
  .progress-title { font-size: 0.85rem; font-weight: 600; }
  .progress-pct { font-size: 1.1rem; font-weight: 700; color: var(--primary); }
  .progress-bar-bg { height: 8px; background: var(--surface2); border-radius: 6px; overflow: hidden; }
  .progress-bar-fill { height: 100%; background: linear-gradient(90deg,var(--primary),var(--accent)); border-radius: 6px; transition: width 0.4s ease; }
  .progress-meta { display: flex; gap: 24px; margin-top: 10px; }
  .progress-meta span { font-size: 0.78rem; color: var(--muted); }
  .progress-meta strong { color: var(--text); }

  /* Two column layout */
  .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 22px; }
  @media (max-width: 900px) { .cols { grid-template-columns: 1fr; } }

  /* Panels */
  .panel { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; overflow: hidden; }
  .panel-header { padding: 14px 18px; border-bottom: 1px solid var(--border); display: flex; align-items: center; justify-content: space-between; }
  .panel-title { font-size: 0.82rem; font-weight: 600; }
  .panel-body { padding: 12px; max-height: 260px; overflow-y: auto; }
  .panel-body::-webkit-scrollbar { width: 4px; }
  .panel-body::-webkit-scrollbar-track { background: transparent; }
  .panel-body::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

  /* System stats */
  .sys-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; padding: 14px; }
  .sys-item { display: flex; flex-direction: column; gap: 4px; }
  .sys-label { font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; }
  .sys-value { font-size: 1rem; font-weight: 600; }
  .sys-bar-bg { height: 5px; background: var(--surface2); border-radius: 4px; margin-top: 3px; }
  .sys-bar-fill { height: 100%; border-radius: 4px; transition: width 0.4s ease; }
  .sys-bar-cpu { background: linear-gradient(90deg,#63b3ed,#4299e1); }
  .sys-bar-ram { background: linear-gradient(90deg,#9f7aea,#805ad5); }
  .sys-bar-gpu { background: linear-gradient(90deg,#68d391,#38a169); }

  /* Tasks list */
  .task-item { display: flex; align-items: center; gap: 8px; padding: 7px 4px; border-bottom: 1px solid rgba(99,179,237,0.05); font-size: 0.77rem; }
  .task-item:last-child { border-bottom: none; }
  .task-name { flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .task-duration { color: var(--muted); min-width: 40px; text-align: right; }
  .task-status { font-size: 0.66rem; padding: 2px 7px; border-radius: 12px; font-weight: 600; white-space: nowrap; }
  .ts-QUEUED     { background: rgba(113,128,150,0.15); color: var(--muted); }
  .ts-PROCESSING { background: rgba(99,179,237,0.15);  color: var(--primary); }
  .ts-COMPLETED  { background: rgba(104,211,145,0.15); color: var(--success); }
  .ts-FAILED     { background: rgba(252,129,129,0.15); color: var(--danger); }
  .ts-RETRYING   { background: rgba(251,211,141,0.15); color: var(--warn); }

  /* Log console */
  .log-line { font-size: 0.72rem; line-height: 1.6; font-family: 'Courier New', monospace; color: var(--muted); padding: 1px 0; }
  .log-line.err { color: #fc8181; }
  .log-line.ok  { color: var(--success); }
  .log-line.info { color: var(--primary); }

  /* Controls */
  .controls { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 18px 20px; margin-bottom: 22px; }
  .controls-title { font-size: 0.82rem; font-weight: 600; margin-bottom: 14px; }
  .controls-row { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
  .btn {
    padding: 8px 18px; border-radius: 8px; border: none; cursor: pointer;
    font-size: 0.8rem; font-weight: 600; font-family: 'Inter', sans-serif;
    transition: transform 0.15s, opacity 0.15s; letter-spacing: 0.3px;
  }
  .btn:hover { transform: translateY(-1px); opacity: 0.9; }
  .btn:active { transform: translateY(0); }
  .btn-primary  { background: linear-gradient(135deg,var(--primary),#4299e1); color: #fff; }
  .btn-warn     { background: linear-gradient(135deg,var(--warn),#ed8936); color: #1a1e34; }
  .btn-danger   { background: linear-gradient(135deg,var(--danger),#e53e3e); color: #fff; }
  .btn-accent   { background: linear-gradient(135deg,var(--accent),#805ad5); color: #fff; }
  .btn-ghost    { background: var(--surface2); color: var(--text); border: 1px solid var(--border); }
  .divider { height: 26px; width: 1px; background: var(--border); margin: 0 4px; }
  .config-field { display: flex; align-items: center; gap: 8px; }
  .config-field label { font-size: 0.75rem; color: var(--muted); white-space: nowrap; }
  .config-field input, .config-field select {
    background: var(--surface2); border: 1px solid var(--border); color: var(--text);
    border-radius: 6px; padding: 5px 10px; font-size: 0.78rem; font-family: 'Inter', sans-serif;
    outline: none; width: 90px;
  }
  .config-field input:focus, .config-field select:focus { border-color: var(--primary); }

  .toast {
    position: fixed; bottom: 24px; right: 24px;
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 10px; padding: 10px 18px; font-size: 0.8rem;
    box-shadow: 0 8px 24px rgba(0,0,0,0.4); opacity: 0;
    transition: opacity 0.3s; pointer-events: none; z-index: 999;
  }
  .toast.show { opacity: 1; }
</style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-icon">⚡</div>
    <span>TextLens BatchOCR</span>
  </div>
  <div style="display:flex;align-items:center;gap:10px;">
    <span style="font-size:0.75rem;color:var(--muted);" id="hdr-model">model: —</span>
    <span class="status-badge badge-IDLE" id="hdr-status">IDLE</span>
  </div>
</header>

<div class="main">
  <!-- KPI Cards -->
  <div class="kpi-grid">
    <div class="kpi-card kpi-total"><div class="kpi-label">Total Files</div><div class="kpi-value" id="k-total">—</div></div>
    <div class="kpi-card kpi-done"><div class="kpi-label">Processed</div><div class="kpi-value" id="k-done">—</div></div>
    <div class="kpi-card kpi-failed"><div class="kpi-label">Failed</div><div class="kpi-value" id="k-fail">—</div></div>
    <div class="kpi-card kpi-queue"><div class="kpi-label">Queued</div><div class="kpi-value" id="k-queue">—</div></div>
    <div class="kpi-card"><div class="kpi-label">Workers</div><div class="kpi-value" id="k-workers" style="color:var(--accent);">—</div><div class="kpi-sub" id="k-workers-sub">—</div></div>
    <div class="kpi-card"><div class="kpi-label">Speed</div><div class="kpi-value" id="k-speed" style="color:var(--primary);">—</div><div class="kpi-sub">files / sec</div></div>
    <div class="kpi-card"><div class="kpi-label">Elapsed</div><div class="kpi-value" id="k-elapsed" style="color:var(--text);font-size:1.4rem;">—</div></div>
    <div class="kpi-card"><div class="kpi-label">ETA</div><div class="kpi-value" id="k-eta" style="color:var(--warn);font-size:1.4rem;">—</div></div>
  </div>

  <!-- Progress Bar -->
  <div class="progress-section">
    <div class="progress-header">
      <span class="progress-title">Batch Progress</span>
      <span class="progress-pct" id="prog-pct">0%</span>
    </div>
    <div class="progress-bar-bg"><div class="progress-bar-fill" id="prog-bar" style="width:0%"></div></div>
    <div class="progress-meta">
      <span><strong id="pm-done">0</strong> done</span>
      <span><strong id="pm-fail">0</strong> failed</span>
      <span><strong id="pm-queue">0</strong> queued</span>
      <span><strong id="pm-total">0</strong> total</span>
    </div>
  </div>

  <!-- Controls -->
  <div class="controls">
    <div class="controls-title">Job Controls</div>
    <div class="controls-row">
      <button class="btn btn-primary" onclick="apiAction('/api/resume')">▶ Resume</button>
      <button class="btn btn-warn"    onclick="apiAction('/api/pause')">⏸ Pause</button>
      <button class="btn btn-danger"  onclick="apiAction('/api/cancel')">⏹ Cancel</button>
      <button class="btn btn-accent"  onclick="apiAction('/api/retry-failed')">↻ Retry Failed</button>
      <div class="divider"></div>
      <div class="config-field">
        <label>Workers</label>
        <input type="number" id="cfg-workers" min="1" max="32" value="4"/>
      </div>
      <div class="config-field">
        <label>Format</label>
        <select id="cfg-format">
          <option value="json">JSON</option>
          <option value="markdown">Markdown</option>
          <option value="csv">CSV</option>
          <option value="txt">Plain Text</option>
        </select>
      </div>
      <button class="btn btn-ghost" onclick="reconfigure()">Apply</button>
    </div>
  </div>

  <!-- System + Tasks -->
  <div class="cols">
    <div class="panel">
      <div class="panel-header"><span class="panel-title">System Resources</span></div>
      <div class="sys-grid">
        <div class="sys-item">
          <span class="sys-label">CPU Usage</span>
          <span class="sys-value" id="s-cpu">—</span>
          <div class="sys-bar-bg"><div class="sys-bar-fill sys-bar-cpu" id="sb-cpu" style="width:0%"></div></div>
        </div>
        <div class="sys-item">
          <span class="sys-label">RAM</span>
          <span class="sys-value" id="s-ram">—</span>
          <div class="sys-bar-bg"><div class="sys-bar-fill sys-bar-ram" id="sb-ram" style="width:0%"></div></div>
        </div>
        <div class="sys-item" style="grid-column:1/-1;">
          <span class="sys-label" id="s-gpu-label">GPU / VRAM</span>
          <span class="sys-value" id="s-gpu">—</span>
          <div class="sys-bar-bg"><div class="sys-bar-fill sys-bar-gpu" id="sb-gpu" style="width:0%"></div></div>
        </div>
      </div>
    </div>

    <div class="panel">
      <div class="panel-header">
        <span class="panel-title">Tasks</span>
        <span style="font-size:0.72rem;color:var(--muted);">latest first</span>
      </div>
      <div class="panel-body" id="task-list">
        <div style="color:var(--muted);font-size:0.8rem;padding:8px;">No tasks yet.</div>
      </div>
    </div>
  </div>

  <!-- Logs -->
  <div class="panel">
    <div class="panel-header">
      <span class="panel-title">Live Logs</span>
      <div style="display:flex;gap:8px;">
        <button class="btn btn-ghost" style="font-size:0.7rem;padding:4px 10px;" onclick="clearLogs()">Clear</button>
        <label style="font-size:0.73rem;color:var(--muted);display:flex;align-items:center;gap:4px;">
          <input type="checkbox" id="autoscroll" checked/> Auto-scroll
        </label>
      </div>
    </div>
    <div class="panel-body" id="log-container" style="max-height:320px;"></div>
  </div>
</div>

<div class="toast" id="toast"></div>

<script>
let evtSource = null;

function fmtTime(sec) {
  if (sec <= 0 || isNaN(sec)) return "—";
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

function setText(id, val) {
  const el = document.getElementById(id);
  if (el && el.textContent !== String(val)) el.textContent = val;
}

function showToast(msg, ok=true) {
  const t = document.getElementById('toast');
  t.textContent = (ok ? '✓ ' : '⚠ ') + msg;
  t.className = 'toast show';
  setTimeout(() => t.className = 'toast', 2500);
}

function updateMetrics(d) {
  setText('k-total', d.total_files);
  setText('k-done', d.processed_files);
  setText('k-fail', d.failed_files);
  setText('k-queue', d.queued_files);
  setText('k-workers', d.active_workers + ' / ' + d.target_workers);
  setText('k-workers-sub', 'active / target');
  setText('k-speed', d.processing_speed_fps.toFixed(2));
  setText('k-elapsed', fmtTime(d.elapsed_time_sec));
  setText('k-eta', fmtTime(d.eta_sec));

  // Progress
  const pct = d.total_files > 0 ? Math.round((d.processed_files / d.total_files) * 100) : 0;
  setText('prog-pct', pct + '%');
  document.getElementById('prog-bar').style.width = pct + '%';
  setText('pm-done', d.processed_files);
  setText('pm-fail', d.failed_files);
  setText('pm-queue', d.queued_files);
  setText('pm-total', d.total_files);

  // Status badge
  const badge = document.getElementById('hdr-status');
  badge.textContent = d.status;
  badge.className = 'status-badge badge-' + d.status;

  setText('hdr-model', 'model: ' + d.model_id);

  // System
  setText('s-cpu', d.cpu_percent.toFixed(1) + '%');
  document.getElementById('sb-cpu').style.width = Math.min(d.cpu_percent, 100) + '%';

  const ramPct = d.ram_total_gb > 0 ? (d.ram_used_gb / d.ram_total_gb * 100) : 0;
  setText('s-ram', d.ram_used_gb.toFixed(1) + ' / ' + d.ram_total_gb.toFixed(1) + ' GB');
  document.getElementById('sb-ram').style.width = ramPct.toFixed(1) + '%';

  if (d.vram_total_gb > 0) {
    const vramPct = d.vram_used_gb / d.vram_total_gb * 100;
    const gpuLabel = d.gpu_name ? d.gpu_name : 'GPU / VRAM';
    setText('s-gpu-label', gpuLabel);
    setText('s-gpu', d.vram_used_gb.toFixed(1) + ' / ' + d.vram_total_gb.toFixed(1) + ' GB');
    document.getElementById('sb-gpu').style.width = vramPct.toFixed(1) + '%';
  } else {
    setText('s-gpu', 'No GPU detected');
  }
}

function updateTasks(tasks) {
  const el = document.getElementById('task-list');
  if (!tasks || tasks.length === 0) return;
  const sorted = tasks.slice().reverse();
  el.innerHTML = sorted.slice(0, 60).map(t => {
    const dur = t.duration_sec > 0 ? t.duration_sec.toFixed(1) + 's' : '—';
    const err = t.error ? ` title="${t.error.replace(/"/g,'')}"` : '';
    return `<div class="task-item"${err}>
      <span class="task-name">${t.file_name}</span>
      <span class="task-duration">${dur}</span>
      <span class="task-status ts-${t.status}">${t.status}</span>
    </div>`;
  }).join('');
}

function updateLogs(logs) {
  if (!logs || logs.length === 0) return;
  const container = document.getElementById('log-container');
  const autoScroll = document.getElementById('autoscroll').checked;
  logs.forEach(line => {
    const d = document.createElement('div');
    d.className = 'log-line' + (line.includes('ERROR') ? ' err' : line.includes('Done') ? ' ok' : line.includes('Dashboard') || line.includes('Started') ? ' info' : '');
    d.textContent = line;
    container.appendChild(d);
  });
  if (container.childNodes.length > 1000) {
    while (container.childNodes.length > 500) container.removeChild(container.firstChild);
  }
  if (autoScroll) container.scrollTop = container.scrollHeight;
}

function clearLogs() {
  document.getElementById('log-container').innerHTML = '';
}

let lastLogIdx = 0;

function connectStream() {
  if (evtSource) evtSource.close();
  evtSource = new EventSource('/api/stream');

  evtSource.addEventListener('metrics', e => {
    try { updateMetrics(JSON.parse(e.data)); } catch(err) {}
  });
  evtSource.addEventListener('tasks', e => {
    try { updateTasks(JSON.parse(e.data)); } catch(err) {}
  });
  evtSource.addEventListener('logs', e => {
    try { updateLogs(JSON.parse(e.data)); } catch(err) {}
  });

  evtSource.onerror = () => {
    setTimeout(connectStream, 2000);
  };
}

function apiAction(endpoint) {
  fetch(endpoint, { method: 'POST' })
    .then(r => r.json())
    .then(d => showToast(d.message || 'Done'))
    .catch(() => showToast('Request failed', false));
}

function reconfigure() {
  const workers = parseInt(document.getElementById('cfg-workers').value);
  const output_format = document.getElementById('cfg-format').value;
  fetch('/api/reconfigure', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ workers, output_format })
  })
  .then(r => r.json())
  .then(d => showToast(d.message || 'Reconfigured'))
  .catch(() => showToast('Reconfigure failed', false));
}

connectStream();
</script>
</body>
</html>
"""


class _DashboardHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the BatchOCR dashboard."""

    engine: "BatchOCR"

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress default HTTP server logging — BatchOCR has its own logger
        pass

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path in ("", "/"):
            self._serve_html()
        elif path == "/api/status":
            self._serve_json(self._metrics_payload())
        elif path == "/api/tasks":
            tasks = self.engine.get_tasks()
            self._serve_json([t.to_dict() for t in tasks])
        elif path == "/api/logs":
            self._serve_json(self.engine.get_logs(200))
        elif path == "/api/stream":
            self._serve_sse()
        else:
            self.send_error(404, "Not Found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/api/pause":
            self.engine.pause()
            self._serve_json({"ok": True, "message": "Job paused"})
        elif path == "/api/resume":
            self.engine.resume()
            self._serve_json({"ok": True, "message": "Job resumed"})
        elif path == "/api/cancel":
            self.engine.cancel()
            self._serve_json({"ok": True, "message": "Job cancelled"})
        elif path == "/api/retry-failed":
            n = self.engine.retry_failed()
            self._serve_json({"ok": True, "message": f"Re-queued {n} failed task(s)"})
        elif path == "/api/reconfigure":
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length).decode("utf-8")
                data: Dict = json.loads(body) if body else {}
                self.engine.reconfigure(
                    workers=data.get("workers"),
                    output_format=data.get("output_format"),
                    retries=data.get("retries"),
                )
                self._serve_json({"ok": True, "message": "Reconfigured successfully"})
            except Exception as exc:
                self._serve_json({"ok": False, "message": str(exc)}, 400)
        else:
            self.send_error(404, "Not Found")

    def _serve_html(self) -> None:
        body = _DASHBOARD_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _serve_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _serve_sse(self) -> None:
        """Server-Sent Events stream for real-time dashboard updates."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self._cors_headers()
        self.end_headers()

        last_log_count = 0
        try:
            while True:
                # ── Metrics ──────────────────────────────────────────
                metrics = self.engine.get_metrics()
                self._sse_event("metrics", metrics.to_dict())

                # ── Tasks ────────────────────────────────────────────
                tasks = self.engine.get_tasks()
                self._sse_event("tasks", [t.to_dict() for t in tasks])

                # ── Incremental Logs ─────────────────────────────────
                all_logs = self.engine.get_logs(500)
                if len(all_logs) > last_log_count:
                    new_lines = all_logs[last_log_count:]
                    last_log_count = len(all_logs)
                    self._sse_event("logs", new_lines)

                time.sleep(1.0)
        except Exception:
            pass  # Client disconnected

    def _sse_event(self, event: str, data: Any) -> None:
        """Write a single SSE event to the stream."""
        try:
            payload = json.dumps(data, ensure_ascii=False)
            msg = f"event: {event}\ndata: {payload}\n\n"
            self.wfile.write(msg.encode("utf-8"))
            self.wfile.flush()
        except Exception:
            raise  # Bubble up to break the SSE loop

    def _cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")


def start_dashboard_server(
    engine: "BatchOCR",
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    """Start the HTTP dashboard server and block (call from a daemon thread).

    Parameters
    ----------
    engine : BatchOCR
        The active BatchOCR instance to introspect.
    host : str
        Bind host. Defaults to ``"127.0.0.1"`` (localhost only).
    port : int
        HTTP port. Defaults to ``8765``.
    """

    # Inject engine reference into the handler class
    class _Handler(_DashboardHandler):
        pass

    _Handler.engine = engine

    try:
        server = HTTPServer((host, port), _Handler)
        logger.info("BatchOCR dashboard running at http://%s:%d", host, port)
        server.serve_forever()
    except OSError as exc:
        logger.warning("Dashboard server failed on port %d: %s", port, exc)
    except Exception as exc:
        logger.warning("Dashboard server error: %s", exc)
