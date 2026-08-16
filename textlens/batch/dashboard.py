"""
textlens.batch.dashboard
─────────────────────────
Live monitoring dashboard for BatchOCR.

Serves a real-time web dashboard at http://localhost:8765 (default) with:
- Live KPI metrics (speed, ETA, workers, CPU/VRAM)
- Interactive sidebar & system observability
- Circular progress & sparkline graphs
- Task list with per-file status & processing indicators
- Streaming live logs via Server-Sent Events (SSE)
- Interactive controls: pause, resume, cancel, retry failed
- Runtime reconfiguration: workers, format, retries

Uses Python's built-in `http.server` to keep zero extra dependencies.
"""

from __future__ import annotations

import json
import logging
import pathlib
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
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
  :root {
    --bg: #070a07;
    --sidebar-bg: #0a0e0b;
    --card-bg: #0e140f;
    --card-border: #1e2e1f;
    --card-border-hover: #2e4630;
    --lime: #70df7f;
    --lime-dim: rgba(112,223,127,0.12);
    --lime-glow: rgba(112,223,127,0.22);
    --yellow: #eab308;
    --yellow-dim: rgba(234,179,8,0.15);
    --red: #ef4444;
    --red-dim: rgba(239,68,68,0.15);
    --purple: #a855f7;
    --purple-dim: rgba(168,85,247,0.15);
    --blue: #3b82f6;
    --blue-dim: rgba(59,130,246,0.15);
    --text: #e4ede5;
    --muted: #718373;
    --mono: 'JetBrains Mono', monospace;
    --sans: 'Inter', sans-serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: var(--sans);
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    display: flex;
    overflow-x: hidden;
  }

  /* ── Sidebar ─────────────────────────────────────────────────── */
  .sidebar {
    width: 240px;
    background: var(--sidebar-bg);
    border-right: 1px solid var(--card-border);
    display: flex;
    flex-direction: column;
    height: 100vh;
    position: sticky;
    top: 0;
    flex-shrink: 0;
    padding: 20px 16px;
    gap: 20px;
    z-index: 10;
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 4px 8px;
    text-decoration: none;
    color: #fff;
  }
  .brand-img {
    width: 32px;
    height: 32px;
    object-fit: contain;
    border-radius: 8px;
  }
  .brand-name {
    font-size: 1.25rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: #fff;
  }
  .nav-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .nav-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 14px;
    border-radius: 10px;
    color: var(--muted);
    font-size: 0.88rem;
    font-weight: 600;
    text-decoration: none;
    transition: all 0.2s;
    cursor: pointer;
  }
  .nav-item:hover {
    color: var(--text);
    background: rgba(112,223,127,0.06);
  }
  .nav-item.active {
    background: #122214;
    color: var(--lime);
    border: 1px solid rgba(112,223,127,0.25);
  }
  .nav-item svg { width: 18px; height: 18px; fill: currentColor; }

  .sidebar-bottom {
    margin-top: auto;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .sys-status-card {
    background: #0f1810;
    border: 1px solid #1f3321;
    border-radius: 12px;
    padding: 12px 14px;
  }
  .sys-status-head {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.78rem;
    font-weight: 700;
    color: var(--lime);
  }
  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--lime);
    box-shadow: 0 0 10px var(--lime);
  }
  .sys-status-sub {
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 3px;
  }
  .gpu-side-card {
    background: #0d150e;
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 12px 14px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .gpu-side-title {
    font-size: 0.75rem;
    font-weight: 700;
    color: var(--muted);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .gpu-side-model {
    font-size: 0.82rem;
    font-weight: 700;
    color: var(--text);
  }
  .gpu-bar-bg {
    height: 6px;
    background: #182619;
    border-radius: 4px;
    overflow: hidden;
  }
  .gpu-bar-fill {
    height: 100%;
    background: var(--lime);
    border-radius: 4px;
    transition: width 0.4s;
  }
  .gpu-side-meta {
    display: flex;
    justify-content: space-between;
    font-size: 0.7rem;
    color: var(--muted);
  }
  .user-card {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 10px;
    background: #0c120c;
    border: 1px solid var(--card-border);
    border-radius: 10px;
  }
  .user-avatar {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: #1e3321;
    color: var(--lime);
    display: grid;
    place-items: center;
    font-size: 0.75rem;
    font-weight: 800;
  }
  .user-info { display: flex; flex-direction: column; }
  .user-name { font-size: 0.8rem; font-weight: 700; color: #fff; }
  .user-role { font-size: 0.68rem; color: var(--muted); }

  /* ── Main Layout ─────────────────────────────────────────────── */
  .layout-main {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-width: 0;
    max-width: 1440px;
    margin: 0 auto;
    padding: 24px 28px;
    gap: 22px;
  }

  /* Header Bar */
  .top-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
    flex-wrap: wrap;
  }
  .top-title h1 {
    font-size: 1.6rem;
    font-weight: 800;
    letter-spacing: -0.8px;
    color: #fff;
  }
  .top-title p {
    font-size: 0.85rem;
    color: var(--muted);
    margin-top: 2px;
  }
  .top-meta {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .model-pill {
    display: flex;
    align-items: center;
    gap: 8px;
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
  }
  .status-badge {
    padding: 5px 14px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.8px;
    text-transform: uppercase;
  }
  .badge-RUNNING { background: var(--lime-dim); color: var(--lime); border: 1px solid rgba(112,223,127,0.3); }
  .badge-PAUSED  { background: var(--yellow-dim); color: var(--yellow); border: 1px solid rgba(234,179,8,0.3); }
  .badge-IDLE    { background: rgba(113,131,115,0.15); color: var(--muted); border: 1px solid rgba(113,131,115,0.3); }
  .badge-CANCELLED { background: var(--red-dim); color: var(--red); border: 1px solid rgba(239,68,68,0.3); }
  .badge-COMPLETED { background: var(--blue-dim); color: var(--blue); border: 1px solid rgba(59,130,246,0.3); }
  .badge-FAILED  { background: var(--red-dim); color: var(--red); border: 1px solid rgba(239,68,68,0.3); }

  .last-updated {
    font-size: 0.78rem;
    color: var(--muted);
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .refresh-btn {
    background: transparent;
    border: none;
    color: var(--muted);
    cursor: pointer;
    font-size: 1rem;
    transition: color 0.2s;
  }
  .refresh-btn:hover { color: var(--lime); }

  /* ── 7 KPI Cards Grid ────────────────────────────────────────── */
  .kpi-grid {
    display: grid;
    grid-template-columns: repeat(7, 1fr);
    gap: 12px;
  }
  @media (max-width: 1200px) { .kpi-grid { grid-template-columns: repeat(4, 1fr); } }
  @media (max-width: 768px)  { .kpi-grid { grid-template-columns: repeat(2, 1fr); } }

  .kpi-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 14px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, border-color 0.2s;
  }
  .kpi-card:hover {
    transform: translateY(-2px);
    border-color: var(--card-border-hover);
  }
  .kpi-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .kpi-icon {
    width: 28px;
    height: 28px;
    border-radius: 8px;
    display: grid;
    place-items: center;
    font-size: 0.9rem;
  }
  .icon-green  { background: var(--lime-dim); color: var(--lime); }
  .icon-red    { background: var(--red-dim); color: var(--red); }
  .icon-yellow { background: var(--yellow-dim); color: var(--yellow); }
  .icon-purple { background: var(--purple-dim); color: var(--purple); }
  .icon-blue   { background: var(--blue-dim); color: var(--blue); }
  .icon-muted  { background: rgba(255,255,255,0.05); color: var(--muted); }

  .kpi-label { font-size: 0.72rem; color: var(--muted); font-weight: 600; }
  .kpi-val { font-size: 1.7rem; font-weight: 800; line-height: 1; color: #fff; }
  .kpi-sub { font-size: 0.68rem; color: var(--muted); }
  .sparkline { width: 100%; height: 24px; stroke-width: 2; fill: none; margin-top: 4px; }

  /* ── Middle Row: Progress + Job Controls ─────────────────────── */
  .mid-grid {
    display: grid;
    grid-template-columns: 1.1fr 0.9fr;
    gap: 16px;
  }
  @media (max-width: 992px) { .mid-grid { grid-template-columns: 1fr; } }

  .panel {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 16px;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  .panel-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #fff;
  }

  /* Donut progress */
  .progress-body {
    display: flex;
    align-items: center;
    gap: 32px;
    padding: 10px 0;
  }
  .donut-box {
    position: relative;
    width: 110px;
    height: 110px;
    flex-shrink: 0;
  }
  .donut-svg { transform: rotate(-90deg); width: 100%; height: 100%; }
  .donut-bg   { stroke: #182619; stroke-width: 10; fill: none; }
  .donut-fill { stroke: var(--lime); stroke-width: 10; fill: none; stroke-dasharray: 283; stroke-dashoffset: 283; transition: stroke-dashoffset 0.5s ease; stroke-linecap: round; }
  .donut-text {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
  }
  .donut-pct { font-size: 1.25rem; font-weight: 800; color: #fff; line-height: 1; }
  .donut-lbl { font-size: 0.65rem; color: var(--muted); }

  .progress-legend {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px 24px;
    flex: 1;
  }
  .legend-item { display: flex; align-items: center; gap: 8px; font-size: 0.8rem; }
  .dot-done   { width: 8px; height: 8px; border-radius: 50%; background: var(--lime); }
  .dot-fail   { width: 8px; height: 8px; border-radius: 50%; background: var(--red); }
  .dot-queue  { width: 8px; height: 8px; border-radius: 50%; background: var(--yellow); }
  .dot-total  { width: 8px; height: 8px; border-radius: 50%; background: var(--text); }

  .prog-timeline { display: flex; flex-direction: column; gap: 6px; }
  .prog-ticks { display: flex; justify-content: space-between; font-size: 0.68rem; color: var(--muted); }
  .prog-track { height: 6px; background: #162417; border-radius: 4px; overflow: hidden; }
  .prog-fill  { height: 100%; background: linear-gradient(90deg, var(--lime), #a855f7); border-radius: 4px; transition: width 0.4s; }

  .prog-sub-meta {
    display: flex;
    justify-content: space-between;
    font-size: 0.78rem;
    color: var(--muted);
    border-top: 1px solid var(--card-border);
    padding-top: 12px;
  }
  .prog-sub-meta strong { color: var(--text); }

  /* Job Controls Panel */
  .btn-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
  }
  @media (max-width: 600px) { .btn-row { grid-template-columns: repeat(2, 1fr); } }

  .btn-action {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    padding: 10px 14px;
    border-radius: 10px;
    border: none;
    font-size: 0.82rem;
    font-weight: 700;
    font-family: var(--sans);
    cursor: pointer;
    transition: transform 0.15s, opacity 0.15s;
  }
  .btn-action:hover { transform: translateY(-1px); opacity: 0.9; }
  .btn-action:active { transform: translateY(0); }
  .btn-resume { background: var(--lime); color: #080d08; }
  .btn-pause  { background: var(--yellow); color: #080d08; }
  .btn-cancel { background: var(--red); color: #fff; }
  .btn-retry  { background: var(--purple); color: #fff; }

  .config-grid {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 8px;
    flex-wrap: wrap;
  }
  .cfg-group {
    display: flex;
    align-items: center;
    gap: 8px;
    flex: 1;
    min-width: 120px;
  }
  .cfg-group label { font-size: 0.78rem; color: var(--muted); font-weight: 600; }
  .cfg-group select, .cfg-group input {
    background: #090e09;
    border: 1px solid var(--card-border);
    color: var(--text);
    padding: 8px 12px;
    border-radius: 8px;
    font-size: 0.8rem;
    font-family: var(--sans);
    outline: none;
    width: 100%;
  }
  .cfg-group select:focus, .cfg-group input:focus { border-color: var(--lime); }
  .btn-apply {
    background: transparent;
    border: 1px solid var(--lime);
    color: var(--lime);
    padding: 8px 20px;
    border-radius: 8px;
    font-size: 0.8rem;
    font-weight: 700;
    cursor: pointer;
    transition: background 0.2s;
  }
  .btn-apply:hover { background: var(--lime-dim); }

  /* ── Bottom Grid: System + Tasks ─────────────────────────────── */
  .bottom-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }
  @media (max-width: 992px) { .bottom-grid { grid-template-columns: 1fr; } }

  .sys-cards {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
  }
  @media (max-width: 600px) { .sys-cards { grid-template-columns: 1fr; } }

  .sys-card {
    background: #090e09;
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 14px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .sys-card-title {
    font-size: 0.72rem;
    color: var(--muted);
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .sys-card-val { font-size: 1.3rem; font-weight: 800; color: #fff; }
  .sys-card-bar { height: 4px; background: #162417; border-radius: 4px; overflow: hidden; margin-top: 4px; }
  .sys-card-fill { height: 100%; border-radius: 4px; transition: width 0.4s; }

  .sys-sub-pills {
    display: flex;
    gap: 10px;
    border-top: 1px solid var(--card-border);
    padding-top: 12px;
  }
  .sys-sub-pill {
    background: #090e09;
    border: 1px solid var(--card-border);
    border-radius: 8px;
    padding: 6px 12px;
    font-size: 0.73rem;
    color: var(--muted);
    display: flex;
    align-items: center;
    gap: 6px;
    flex: 1;
  }
  .sys-sub-pill strong { color: var(--text); }

  /* Tasks Table */
  .tasks-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .task-list {
    display: flex;
    flex-direction: column;
    gap: 6px;
    max-height: 240px;
    overflow-y: auto;
    padding-right: 4px;
  }
  .task-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 12px;
    background: #090e09;
    border: 1px solid var(--card-border);
    border-radius: 8px;
    font-size: 0.8rem;
  }
  .task-name {
    flex: 1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    font-weight: 500;
  }
  .task-badge {
    font-size: 0.65rem;
    font-weight: 800;
    padding: 3px 8px;
    border-radius: 6px;
    letter-spacing: 0.5px;
  }
  .tb-QUEUED     { background: var(--yellow-dim); color: var(--yellow); }
  .tb-PROCESSING { background: var(--lime-dim);   color: var(--lime); display: flex; align-items: center; gap: 4px; }
  .tb-COMPLETED  { background: rgba(112,223,127,0.1); color: var(--lime); }
  .tb-FAILED     { background: var(--red-dim); color: var(--red); }
  .tb-RETRYING   { background: var(--purple-dim); color: var(--purple); }

  .spinner-ring {
    width: 10px;
    height: 10px;
    border: 2px solid var(--lime);
    border-top-color: transparent;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ── Live Logs Panel ─────────────────────────────────────────── */
  .logs-panel {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 16px;
    padding: 16px 20px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .logs-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .streaming-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.75rem;
    color: var(--lime);
    font-weight: 600;
    margin-left: 10px;
  }
  .stream-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--lime);
    animation: pulse 1.5s infinite;
  }
  @keyframes pulse { 0%, 100% { opacity: 0.3; } 50% { opacity: 1; } }

  .log-console {
    background: #050805;
    border: 1px solid #142015;
    border-radius: 10px;
    padding: 14px 16px;
    font-family: var(--mono);
    font-size: 0.76rem;
    line-height: 1.7;
    height: 180px;
    overflow-y: auto;
    color: #8aa08c;
  }
  .log-line.err { color: #f87171; }
  .log-line.ok  { color: var(--lime); }
  .log-line.info { color: #60a5fa; }

  /* Toast Notification */
  .toast {
    position: fixed;
    bottom: 24px;
    right: 24px;
    background: #0f1a10;
    border: 1px solid var(--lime);
    color: #fff;
    padding: 10px 18px;
    border-radius: 10px;
    font-size: 0.8rem;
    font-weight: 600;
    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    opacity: 0;
    transition: opacity 0.3s, transform 0.3s;
    transform: translateY(10px);
    pointer-events: none;
    z-index: 999;
  }
  .toast.show { opacity: 1; transform: translateY(0); }
</style>
</head>
<body>

<!-- ── Sidebar ─────────────────────────────────────────────────── -->
<aside class="sidebar">
  <a class="brand" href="#">
    <img src="/logo.png" alt="TextLens" class="brand-img" onerror="this.src='data:image/svg+xml;utf8,<svg xmlns=\'http://www.w3.org/2000/svg\' viewBox=\'0 0 100 100\'><rect width=\'100\' height=\'100\' rx=\'20\' fill=\'%2370df7f\'/><text x=\'50\' y=\'65\' font-size=\'50\' text-anchor=\'middle\' fill=\'%23070a07\' font-weight=\'bold\'>TL</text></svg>'"/>
    <span class="brand-name">TextLens</span>
  </a>

  <nav class="nav-list">
    <a class="nav-item active">
      <svg viewBox="0 0 24 24"><path d="M4 13h6a1 1 0 0 0 1-1V4a1 1 0 0 0-1-1H4a1 1 0 0 0-1 1v8a1 1 0 0 0 1 1zm0 8h6a1 1 0 0 0 1-1v-4a1 1 0 0 0-1-1H4a1 1 0 0 0-1 1v4a1 1 0 0 0 1 1zm10 0h6a1 1 0 0 0 1-1v-8a1 1 0 0 0-1-1h-6a1 1 0 0 0-1 1v8a1 1 0 0 0 1 1zm0-18v4a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1V3a1 1 0 0 0-1-1h-6a1 1 0 0 0-1 1z"/></svg>
      <span>Overview</span>
    </a>
    <a class="nav-item">
      <svg viewBox="0 0 24 24"><path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-5 14H7v-2h7v2zm3-4H7v-2h10v2zm0-4H7V7h10v2z"/></svg>
      <span>Jobs</span>
    </a>
    <a class="nav-item">
      <svg viewBox="0 0 24 24"><path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/></svg>
      <span>Analytics</span>
    </a>
    <a class="nav-item">
      <svg viewBox="0 0 24 24"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>
      <span>Models</span>
    </a>
    <a class="nav-item">
      <svg viewBox="0 0 24 24"><path d="M19.14 12.94c.04-.3.06-.61.06-.94 0-.32-.02-.64-.07-.94l2.03-1.58c.18-.14.23-.41.12-.61l-1.92-3.32c-.12-.22-.37-.29-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94l-.36-2.54c-.04-.24-.24-.41-.48-.41h-3.84c-.24 0-.43.17-.47.41l-.36 2.54c-.59.24-1.13.57-1.62.94l-2.39-.96c-.22-.08-.47 0-.59.22L2.74 8.87c-.12.21-.08.47.12.61l2.03 1.58c-.05.3-.09.63-.09.94s.02.64.07.94l-2.03 1.58c-.18.14-.23.41-.12.61l1.92 3.32c.12.22.37.29.59.22l2.39-.96c.5.38 1.03.7 1.62.94l.36 2.54c.05.24.24.41.48.41h3.84c.24 0 .44-.17.47-.41l.36-2.54c.59-.24 1.13-.56 1.62-.94l2.39.96c.22.08.47 0 .59-.22l1.92-3.32c.12-.22.07-.47-.12-.61l-2.01-1.58zM12 15.6c-1.98 0-3.6-1.62-3.6-3.6s1.62-3.6 3.6-3.6 3.6 1.62 3.6 3.6-1.62 3.6-3.6 3.6z"/></svg>
      <span>Settings</span>
    </a>
    <a class="nav-item">
      <svg viewBox="0 0 24 24"><path d="M12.65 10C11.83 7.67 9.61 6 7 6c-3.31 0-6 2.69-6 6s2.69 6 6 6c2.61 0 4.83-1.67 5.65-4H17v4h4v-4h2v-4H12.65zM7 14c-1.1 0-2-.9-2-2s.9-2 2-2 2 .9 2 2-.9 2-2 2z"/></svg>
      <span>API Keys</span>
    </a>
    <a class="nav-item">
      <svg viewBox="0 0 24 24"><path d="M14 2H6c-1.1 0-1.99.9-1.99 2L4 20c0 1.1.89 2 1.99 2H18c1.1 0 2-.9 2-2V8l-6-6zm2 16H8v-2h8v2zm0-4H8v-2h8v2zm-3-5V3.5L18.5 9H13z"/></svg>
      <span>Logs</span>
    </a>
    <a class="nav-item">
      <svg viewBox="0 0 24 24"><path d="M12 22c1.1 0 2-.9 2-2h-4c0 1.1.89 2 2 2zm6-6v-5c0-3.07-1.64-5.64-4.5-6.32V4c0-.83-.67-1.5-1.5-1.5s-1.5.67-1.5 1.5v.68C7.63 5.36 6 7.92 6 11v5l-2 2v1h16v-1l-2-2z"/></svg>
      <span>Alerts</span>
    </a>
  </nav>

  <div class="sidebar-bottom">
    <div class="sys-status-card">
      <div class="sys-status-head">
        <span class="status-dot"></span>
        <span>System Status</span>
      </div>
      <div class="sys-status-sub">All systems operational</div>
    </div>

    <div class="gpu-side-card">
      <div class="gpu-side-title">
        <span>GPU</span>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="var(--lime)"><path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/></svg>
      </div>
      <div class="gpu-side-model" id="sg-gpu-model">RTX 4050 · 6 GB</div>
      <div class="gpu-bar-bg"><div class="gpu-bar-fill" id="sg-vram-bar" style="width: 100%"></div></div>
      <div class="gpu-side-meta">
        <span>Utilization</span>
        <strong id="sg-gpu-util" style="color:var(--lime)">68%</strong>
      </div>
    </div>

    <div class="user-card">
      <div class="user-avatar">A</div>
      <div class="user-info">
        <div class="user-name">Admin</div>
        <div class="user-role">Local Instance ▾</div>
      </div>
    </div>
  </div>
</aside>

<!-- ── Main Content ─────────────────────────────────────────────── -->
<main class="layout-main">

  <!-- Top Header Bar -->
  <header class="top-header">
    <div class="top-title">
      <h1>BatchOCR Dashboard</h1>
      <p>Real-time monitoring of batch processing and system performance</p>
    </div>
    <div class="top-meta">
      <div class="model-pill">
        <span style="color:var(--muted)">Model:</span>
        <strong id="hdr-model-id" style="color:var(--text)">glm-ocr</strong>
      </div>
      <div class="status-badge badge-IDLE" id="hdr-status-badge">● IDLE</div>
      <div class="last-updated">
        <span>Last updated: <strong id="hdr-time">—</strong></span>
        <button class="refresh-btn" onclick="location.reload()" title="Refresh">↻</button>
      </div>
    </div>
  </header>

  <!-- ── 7 KPI Cards Grid ────────────────────────────────────────── -->
  <section class="kpi-grid">
    <!-- 1. Total Files -->
    <div class="kpi-card">
      <div class="kpi-head">
        <div class="kpi-icon icon-green">📄</div>
        <span class="kpi-label">Total Files</span>
      </div>
      <div class="kpi-val" id="k-total">0</div>
      <div class="kpi-sub">All time</div>
      <svg class="sparkline"><path d="M0,20 Q10,12 20,18 T40,8 T60,15 T80,5 T100,12" stroke="#70df7f"/></svg>
    </div>

    <!-- 2. Processed -->
    <div class="kpi-card">
      <div class="kpi-head">
        <div class="kpi-icon icon-green">✓</div>
        <span class="kpi-label">Processed</span>
      </div>
      <div class="kpi-val" id="k-done" style="color:var(--lime)">0</div>
      <div class="kpi-sub">Success</div>
      <svg class="sparkline"><path d="M0,18 Q15,22 30,10 T60,16 T90,4 T100,10" stroke="#70df7f"/></svg>
    </div>

    <!-- 3. Failed -->
    <div class="kpi-card">
      <div class="kpi-head">
        <div class="kpi-icon icon-red">✕</div>
        <span class="kpi-label">Failed</span>
      </div>
      <div class="kpi-val" id="k-fail" style="color:var(--red)">0</div>
      <div class="kpi-sub">Failures</div>
      <svg class="sparkline"><path d="M0,15 Q20,15 40,20 T70,18 T100,22" stroke="#ef4444"/></svg>
    </div>

    <!-- 4. Queued -->
    <div class="kpi-card">
      <div class="kpi-head">
        <div class="kpi-icon icon-yellow">⏱</div>
        <span class="kpi-label">Queued</span>
      </div>
      <div class="kpi-val" id="k-queue" style="color:var(--yellow)">0</div>
      <div class="kpi-sub">Pending</div>
      <svg class="sparkline"><path d="M0,22 Q25,10 50,16 T75,8 T100,14" stroke="#eab308"/></svg>
    </div>

    <!-- 5. Workers -->
    <div class="kpi-card">
      <div class="kpi-head">
        <div class="kpi-icon icon-purple">👤</div>
        <span class="kpi-label">Workers</span>
      </div>
      <div class="kpi-val" id="k-workers" style="color:var(--purple)">1 / 1</div>
      <div class="kpi-sub">Active / Total</div>
    </div>

    <!-- 6. Speed -->
    <div class="kpi-card">
      <div class="kpi-head">
        <div class="kpi-icon icon-blue">⚡</div>
        <span class="kpi-label">Speed</span>
      </div>
      <div class="kpi-val" id="k-speed" style="color:var(--blue)">0.00</div>
      <div class="kpi-sub">files / sec</div>
      <svg class="sparkline"><path d="M0,20 Q30,5 60,18 T100,6" stroke="#3b82f6"/></svg>
    </div>

    <!-- 7. Elapsed Time -->
    <div class="kpi-card">
      <div class="kpi-head">
        <div class="kpi-icon icon-muted">🕒</div>
        <span class="kpi-label">Elapsed Time</span>
      </div>
      <div class="kpi-val" id="k-elapsed" style="font-size:1.35rem">0s</div>
      <div class="kpi-sub">Since start</div>
    </div>
  </section>

  <!-- ── Middle Row: Progress + Controls ─────────────────────────── -->
  <section class="mid-grid">
    <!-- Batch Progress -->
    <div class="panel">
      <div class="panel-title">Batch Progress</div>
      <div class="progress-body">
        <div class="donut-box">
          <svg class="donut-svg" viewBox="0 0 100 100">
            <circle class="donut-bg" cx="50" cy="50" r="42"/>
            <circle class="donut-fill" id="donut-circle" cx="50" cy="50" r="42"/>
          </svg>
          <div class="donut-text">
            <div class="donut-pct" id="donut-pct">0%</div>
            <div class="donut-lbl">Complete</div>
          </div>
        </div>

        <div class="progress-legend">
          <div class="legend-item"><span class="dot-done"></span>Done <strong id="lg-done" style="margin-left:auto">0</strong></div>
          <div class="legend-item"><span class="dot-fail"></span>Failed <strong id="lg-fail" style="margin-left:auto">0</strong></div>
          <div class="legend-item"><span class="dot-queue"></span>Queued <strong id="lg-queue" style="margin-left:auto">0</strong></div>
          <div class="legend-item"><span class="dot-total"></span>Total <strong id="lg-total" style="margin-left:auto">0</strong></div>
        </div>
      </div>

      <div class="prog-timeline">
        <div class="prog-track"><div class="prog-fill" id="prog-fill-bar" style="width:0%"></div></div>
        <div class="prog-ticks">
          <span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span>
        </div>
      </div>

      <div class="prog-sub-meta">
        <span>Estimated Time Remaining: <strong id="meta-eta">—</strong></span>
        <span>Current Speed: <strong id="meta-speed">0.00 files/s</strong></span>
      </div>
    </div>

    <!-- Job Controls -->
    <div class="panel">
      <div class="panel-title">Job Controls</div>
      <div class="btn-row">
        <button class="btn-action btn-resume" onclick="apiAction('/api/resume')">▶ Resume</button>
        <button class="btn-action btn-pause"  onclick="apiAction('/api/pause')">⏸ Pause</button>
        <button class="btn-action btn-cancel" onclick="apiAction('/api/cancel')">⏹ Cancel</button>
        <button class="btn-action btn-retry"  onclick="apiAction('/api/retry-failed')">↻ Retry Failed</button>
      </div>

      <div style="border-top:1px solid var(--card-border);margin:4px 0;"></div>

      <div class="config-grid">
        <div class="cfg-group">
          <label>Workers</label>
          <input type="number" id="cfg-workers" min="1" max="32" value="1"/>
        </div>
        <div class="cfg-group">
          <label>Format</label>
          <select id="cfg-format">
            <option value="json">JSON</option>
            <option value="markdown">Markdown</option>
            <option value="csv">CSV</option>
            <option value="txt">Plain Text</option>
          </select>
        </div>
        <button class="btn-apply" onclick="reconfigure()">Apply</button>
      </div>
    </div>
  </section>

  <!-- ── Bottom Grid: System Resources + Tasks ───────────────────── -->
  <section class="bottom-grid">
    <!-- System Resources -->
    <div class="panel">
      <div class="panel-title">System Resources</div>
      <div class="sys-cards">
        <div class="sys-card">
          <div class="sys-card-title">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="var(--lime)"><path d="M15 9H9v6h6V9zm-2 4h-2v-2h2v2zm8-2V9h-2V7c0-1.1-.9-2-2-2h-2V3h-2v2h-2V3H9v2H7c-1.1 0-2 .9-2 2v2H3v2h2v2H3v2h2v2c0 1.1.9 2 2 2h2v2h2v-2h2v2h2v-2h2c1.1 0 2-.9 2-2v-2h2v-2h-2v-2h2zm-4 6H7V7h10v10z"/></svg>
            <span>CPU Usage</span>
          </div>
          <div class="sys-card-val" id="s-cpu-val">0.0%</div>
          <svg class="sparkline"><path d="M0,18 Q15,10 30,16 T60,8 T90,14 T100,6" stroke="#70df7f" id="cpu-sparkline"/></svg>
        </div>

        <div class="sys-card">
          <div class="sys-card-title">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="var(--purple)"><path d="M4 6h16v12H4z"/></svg>
            <span>RAM Usage</span>
          </div>
          <div class="sys-card-val" id="s-ram-val" style="font-size:1.05rem">0 / 0 GB</div>
          <div class="sys-card-bar"><div class="sys-card-fill" id="s-ram-fill" style="width:0%;background:var(--purple)"></div></div>
        </div>

        <div class="sys-card">
          <div class="sys-card-title">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="var(--lime)"><path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/></svg>
            <span id="s-gpu-name">GPU (VRAM)</span>
          </div>
          <div class="sys-card-val" id="s-gpu-val" style="font-size:1.05rem">0 / 0 GB</div>
          <div class="sys-card-bar"><div class="sys-card-fill" id="s-gpu-fill" style="width:0%;background:var(--lime)"></div></div>
        </div>
      </div>

      <div class="sys-sub-pills">
        <div class="sys-sub-pill">
          <span>🌡 CPU Temp</span>
          <strong id="st-cpu">62°C</strong>
        </div>
        <div class="sys-sub-pill">
          <span>⚡ GPU Temp</span>
          <strong id="st-gpu">68°C</strong>
        </div>
        <div class="sys-sub-pill">
          <span>📊 RAM Usage</span>
          <strong id="st-ram-pct">0%</strong>
        </div>
      </div>
    </div>

    <!-- Tasks List -->
    <div class="panel">
      <div class="tasks-header">
        <div class="panel-title">Tasks</div>
        <select style="background:transparent;border:none;color:var(--muted);font-size:0.75rem;cursor:pointer;outline:none">
          <option>Latest First ▾</option>
        </select>
      </div>

      <div class="task-list" id="task-container">
        <div style="color:var(--muted);font-size:0.8rem;padding:12px 0;">No active tasks.</div>
      </div>
    </div>
  </section>

  <!-- ── Live Logs ────────────────────────────────────────────────── -->
  <section class="logs-panel">
    <div class="logs-header">
      <div style="display:flex;align-items:center">
        <span class="panel-title">Live Logs</span>
        <span class="streaming-tag"><span class="stream-dot"></span>Streaming</span>
      </div>
      <div style="display:flex;gap:12px;align-items:center">
        <label style="font-size:0.78rem;color:var(--muted);cursor:pointer;display:flex;align-items:center;gap:6px">
          <input type="checkbox" id="autoscroll" checked style="accent-color:var(--lime)"/> Auto-scroll
        </label>
        <button class="btn-apply" style="padding:4px 12px;font-size:0.75rem" onclick="clearLogs()">Clear</button>
      </div>
    </div>

    <div class="log-console" id="log-box"></div>
  </section>

</main>

<div class="toast" id="toast"></div>

<script>
let evtSource = null;

function fmtTime(sec) {
  if (sec <= 0 || isNaN(sec)) return "0s";
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
  setTimeout(() => t.className = 'toast', 2400);
}

function updateMetrics(d) {
  // Update Header & Clock
  const now = new Date();
  setText('hdr-time', now.toTimeString().split(' ')[0]);
  setText('hdr-model-id', d.model_id);

  const badge = document.getElementById('hdr-status-badge');
  badge.textContent = '● ' + d.status;
  badge.className = 'status-badge badge-' + d.status;

  // KPI Cards
  setText('k-total', d.total_files);
  setText('k-done', d.processed_files);
  setText('k-fail', d.failed_files);
  setText('k-queue', d.queued_files);
  setText('k-workers', d.active_workers + ' / ' + d.target_workers);
  setText('k-speed', d.processing_speed_fps.toFixed(2));
  setText('k-elapsed', fmtTime(d.elapsed_time_sec));

  // Progress Donut & Timeline
  const pct = d.total_files > 0 ? Math.round((d.processed_files / d.total_files) * 100) : 0;
  setText('donut-pct', pct + '%');

  const circle = document.getElementById('donut-circle');
  const dashOffset = 283 - (283 * pct) / 100;
  circle.style.strokeDashoffset = dashOffset;

  document.getElementById('prog-fill-bar').style.width = pct + '%';

  setText('lg-done', d.processed_files);
  setText('lg-fail', d.failed_files);
  setText('lg-queue', d.queued_files);
  setText('lg-total', d.total_files);

  setText('meta-eta', fmtTime(d.eta_sec));
  setText('meta-speed', d.processing_speed_fps.toFixed(2) + ' files/s');

  // System Stats
  setText('s-cpu-val', d.cpu_percent.toFixed(1) + '%');
  const ramPct = d.ram_total_gb > 0 ? (d.ram_used_gb / d.ram_total_gb * 100) : 0;
  setText('s-ram-val', d.ram_used_gb.toFixed(1) + ' / ' + d.ram_total_gb.toFixed(1) + ' GB');
  document.getElementById('s-ram-fill').style.width = Math.min(ramPct, 100) + '%';
  setText('st-ram-pct', Math.round(ramPct) + '%');

  if (d.vram_total_gb > 0) {
    const vramPct = (d.vram_used_gb / d.vram_total_gb * 100);
    const gpuName = d.gpu_name ? d.gpu_name : 'GPU (VRAM)';
    setText('s-gpu-name', gpuName);
    setText('s-gpu-val', d.vram_used_gb.toFixed(1) + ' / ' + d.vram_total_gb.toFixed(1) + ' GB');
    document.getElementById('s-gpu-fill').style.width = Math.min(vramPct, 100) + '%';

    // Sidebar GPU Card
    setText('sg-gpu-model', gpuName);
    document.getElementById('sg-vram-bar').style.width = Math.min(vramPct, 100) + '%';
    setText('sg-gpu-util', Math.round(vramPct) + '%');
  }
}

function updateTasks(tasks) {
  const container = document.getElementById('task-container');
  if (!tasks || tasks.length === 0) return;
  const sorted = tasks.slice().reverse();
  container.innerHTML = sorted.slice(0, 50).map(t => {
    let badgeClass = 'tb-' + t.status;
    let statusText = t.status;
    if (t.status === 'PROCESSING') {
      statusText = '<span class="spinner-ring"></span> PROCESSING';
    }
    return `<div class="task-item">
      <span style="color:var(--muted)">📄</span>
      <span class="task-name">${t.file_name}</span>
      <span class="task-badge ${badgeClass}">${statusText}</span>
    </div>`;
  }).join('');
}

function updateLogs(logs) {
  if (!logs || logs.length === 0) return;
  const box = document.getElementById('log-box');
  const autoScroll = document.getElementById('autoscroll').checked;
  logs.forEach(line => {
    const d = document.createElement('div');
    d.className = 'log-line' + (line.includes('ERROR') ? ' err' : line.includes('Done') || line.includes('complete') ? ' ok' : line.includes('INFO') ? ' info' : '');
    d.textContent = line;
    box.appendChild(d);
  });
  if (box.childNodes.length > 800) {
    while (box.childNodes.length > 400) box.removeChild(box.firstChild);
  }
  if (autoScroll) box.scrollTop = box.scrollHeight;
}

function clearLogs() {
  document.getElementById('log-box').innerHTML = '';
}

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
    .then(d => showToast(d.message || 'Action executed'))
    .catch(() => showToast('Action failed', false));
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
        elif path == "/logo.png":
            self._serve_logo()
        elif path == "/api/status":
            self._serve_json(self.engine.get_metrics().to_dict())
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

    def _serve_logo(self) -> None:
        """Serve the TextLens logo PNG."""
        candidates = [
            pathlib.Path(r"C:\Users\Srevarshan\Downloads\new-logo-textlens.png"),
            pathlib.Path(__file__).parents[2] / "website" / "assets" / "logo.png",
            pathlib.Path(__file__).parents[2] / "website" / "assets" / "textlens-logo.png",
        ]
        data = None
        for cand in candidates:
            if cand.exists():
                try:
                    data = cand.read_bytes()
                    break
                except Exception:
                    pass

        if data:
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self._cors_headers()
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_error(404, "Logo Not Found")

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
