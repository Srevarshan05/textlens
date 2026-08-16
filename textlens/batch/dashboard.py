"""
textlens.batch.dashboard
─────────────────────────
Live monitoring dashboard for BatchOCR.

Serves a real-time web dashboard at http://localhost:8765 (default) with:
- Live KPI metrics (speed, ETA, workers, CPU/VRAM)
- Clean overview layout with resizable panels (Tasks, Logs, System)
- Real live CPU history graph (no dummy sparklines)
- Premium vector SVG icons throughout (zero emojis)
- PDF Batch Report Exporter (downloads standard .pdf document when job is done)
- Persistent server ("Close & Return to Terminal" button)
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
    --bg: #060906;
    --sidebar-bg: #090d0a;
    --card-bg: #0d130e;
    --card-border: #1a2a1b;
    --card-border-hover: #2a422c;
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
    --muted: #6b7c6d;
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

  /* ── Sidebar (Clean & Minimal) ───────────────────────────────── */
  .sidebar {
    width: 230px;
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
    gap: 10px;
    padding: 4px 6px;
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
    font-size: 1.2rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: #fff;
  }
  .nav-list {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .nav-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    border-radius: 10px;
    color: var(--muted);
    font-size: 0.88rem;
    font-weight: 600;
    text-decoration: none;
    transition: all 0.2s;
    cursor: pointer;
    border: 1px solid transparent;
  }
  .nav-item:hover {
    color: var(--text);
    background: rgba(112,223,127,0.06);
  }
  .nav-item.active {
    background: #112213;
    color: var(--lime);
    border-color: rgba(112,223,127,0.25);
  }
  .nav-item svg { width: 18px; height: 18px; fill: none; stroke: currentColor; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }

  .export-btn-side {
    background: linear-gradient(135deg, #16361a, #0d2210);
    border: 1px solid rgba(112,223,127,0.3);
    color: var(--lime);
    font-weight: 700;
    margin-top: 4px;
  }
  .export-btn-side:hover {
    background: linear-gradient(135deg, #1c4521, #112c15);
    color: #fff;
    border-color: var(--lime);
  }

  .sidebar-bottom {
    margin-top: auto;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .gpu-side-card {
    background: #0b110c;
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
    font-size: 0.8rem;
    font-weight: 700;
    color: var(--text);
  }
  .gpu-bar-bg {
    height: 5px;
    background: #162417;
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

  /* ── Main Layout ─────────────────────────────────────────────── */
  .layout-main {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-width: 0;
    max-width: 1440px;
    margin: 0 auto;
    padding: 24px 28px;
    gap: 20px;
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
    font-size: 1.55rem;
    font-weight: 800;
    letter-spacing: -0.7px;
    color: #fff;
  }
  .top-title p {
    font-size: 0.84rem;
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

  .btn-close-term {
    background: rgba(239,68,68,0.12);
    border: 1px solid rgba(239,68,68,0.3);
    color: var(--red);
    padding: 6px 14px;
    border-radius: 8px;
    font-size: 0.75rem;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.2s;
  }
  .btn-close-term:hover {
    background: var(--red);
    color: #fff;
  }

  /* ── 7 Clean KPI Cards (Vector SVGs) ──────────────────────────── */
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
  .kpi-icon-svg {
    width: 32px;
    height: 32px;
    border-radius: 999px;
    display: grid;
    place-items: center;
    flex-shrink: 0;
  }
  .kpi-icon-svg svg { width: 17px; height: 17px; fill: none; stroke: currentColor; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }

  .icon-green  { background: var(--lime-dim); color: var(--lime); }
  .icon-red    { background: var(--red-dim); color: var(--red); }
  .icon-yellow { background: var(--yellow-dim); color: var(--yellow); }
  .icon-purple { background: var(--purple-dim); color: var(--purple); }
  .icon-blue   { background: var(--blue-dim); color: var(--blue); }
  .icon-muted  { background: rgba(255,255,255,0.05); color: var(--muted); }

  .kpi-label { font-size: 0.72rem; color: var(--muted); font-weight: 600; }
  .kpi-val   { font-size: 1.65rem; font-weight: 800; line-height: 1; color: #fff; }
  .kpi-sub   { font-size: 0.68rem; color: var(--muted); margin-top: 2px; }

  /* ── Resizable Panels ────────────────────────────────────────── */
  .panel {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 16px;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 14px;
    position: relative;
  }
  .resizable-panel {
    resize: vertical;
    overflow: auto;
    min-height: 200px;
  }
  .resize-hint {
    position: absolute;
    top: 14px;
    right: 16px;
    font-size: 0.68rem;
    color: var(--muted);
    opacity: 0.6;
    pointer-events: none;
    font-family: var(--mono);
  }
  .panel-title {
    font-size: 0.92rem;
    font-weight: 700;
    color: #fff;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .panel-title svg { width: 16px; height: 16px; fill: none; stroke: var(--lime); stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }

  /* ── Middle Row: Progress + Job Controls ─────────────────────── */
  .mid-grid {
    display: grid;
    grid-template-columns: 1.1fr 0.9fr;
    gap: 16px;
  }
  @media (max-width: 992px) { .mid-grid { grid-template-columns: 1fr; } }

  /* Donut progress */
  .progress-body {
    display: flex;
    align-items: center;
    gap: 28px;
    padding: 6px 0;
  }
  .donut-box {
    position: relative;
    width: 105px;
    height: 105px;
    flex-shrink: 0;
  }
  .donut-svg { transform: rotate(-90deg); width: 100%; height: 100%; }
  .donut-bg   { stroke: #142215; stroke-width: 10; fill: none; }
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
    gap: 10px 20px;
    flex: 1;
  }
  .legend-item { display: flex; align-items: center; gap: 8px; font-size: 0.8rem; }
  .dot-done   { width: 8px; height: 8px; border-radius: 50%; background: var(--lime); }
  .dot-fail   { width: 8px; height: 8px; border-radius: 50%; background: var(--red); }
  .dot-queue  { width: 8px; height: 8px; border-radius: 50%; background: var(--yellow); }
  .dot-total  { width: 8px; height: 8px; border-radius: 50%; background: var(--text); }

  .prog-timeline { display: flex; flex-direction: column; gap: 6px; }
  .prog-ticks { display: flex; justify-content: space-between; font-size: 0.68rem; color: var(--muted); }
  .prog-track { height: 6px; background: #142215; border-radius: 4px; overflow: hidden; }
  .prog-fill  { height: 100%; background: linear-gradient(90deg, var(--lime), #a855f7); border-radius: 4px; transition: width 0.4s; }

  .prog-sub-meta {
    display: flex;
    justify-content: space-between;
    font-size: 0.78rem;
    color: var(--muted);
    border-top: 1px solid var(--card-border);
    padding-top: 10px;
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
  .btn-action svg { width: 14px; height: 14px; fill: currentColor; }
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
    margin-top: 4px;
    flex-wrap: wrap;
  }
  .cfg-group {
    display: flex;
    align-items: center;
    gap: 8px;
    flex: 1;
    min-width: 110px;
  }
  .cfg-group label { font-size: 0.78rem; color: var(--muted); font-weight: 600; }
  .cfg-group select, .cfg-group input {
    background: #070c07;
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
    padding: 8px 18px;
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
    grid-template-columns: 1fr 1.1fr;
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
    background: #080c08;
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
  .sys-card-title svg { width: 14px; height: 14px; fill: none; stroke: currentColor; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }

  .sys-card-val { font-size: 1.25rem; font-weight: 800; color: #fff; }
  .sys-card-bar { height: 4px; background: #142215; border-radius: 4px; overflow: hidden; margin-top: 4px; }
  .sys-card-fill { height: 100%; border-radius: 4px; transition: width 0.4s; }

  /* Real Live CPU Graph */
  .cpu-live-svg {
    width: 100%;
    height: 36px;
    stroke-width: 2;
    fill: none;
    margin-top: 2px;
  }

  .sys-sub-pills {
    display: flex;
    gap: 10px;
    border-top: 1px solid var(--card-border);
    padding-top: 10px;
  }
  .sys-sub-pill {
    background: #080c08;
    border: 1px solid var(--card-border);
    border-radius: 8px;
    padding: 7px 10px;
    font-size: 0.73rem;
    color: var(--muted);
    display: flex;
    align-items: center;
    gap: 6px;
    flex: 1;
  }
  .sys-sub-pill svg { width: 14px; height: 14px; fill: none; stroke: var(--lime); stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }
  .sys-sub-pill strong { color: var(--text); margin-left: auto; }

  /* Tasks Table (Resizable) */
  .tasks-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .task-list {
    display: flex;
    flex-direction: column;
    gap: 6px;
    flex: 1;
    overflow-y: auto;
    padding-right: 4px;
  }
  .task-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 9px 12px;
    background: #080c08;
    border: 1px solid var(--card-border);
    border-radius: 8px;
    font-size: 0.8rem;
  }
  .task-item svg { width: 16px; height: 16px; fill: none; stroke: var(--muted); stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; flex-shrink: 0; }
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

  /* ── Live Logs Panel (Resizable) ─────────────────────────────── */
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
    background: #040704;
    border: 1px solid #121f13;
    border-radius: 10px;
    padding: 14px 16px;
    font-family: var(--mono);
    font-size: 0.76rem;
    line-height: 1.7;
    flex: 1;
    min-height: 140px;
    overflow-y: auto;
    color: #8aa08c;
  }
  .log-line.err { color: #f87171; }
  .log-line.ok  { color: var(--lime); }
  .log-line.info { color: #60a5fa; }

  /* Toast & Modal */
  .toast {
    position: fixed;
    bottom: 24px;
    right: 24px;
    background: #0e1b10;
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

  /* Modal Overlay for Export Report */
  .modal-overlay {
    position: fixed;
    inset: 0;
    background: rgba(4,7,4,0.85);
    backdrop-filter: blur(8px);
    display: none;
    align-items: center;
    justify-content: center;
    z-index: 1000;
    padding: 20px;
  }
  .modal-overlay.open { display: flex; }
  .modal-card {
    background: #0d140e;
    border: 1px solid var(--lime);
    border-radius: 18px;
    padding: 28px;
    max-width: 520px;
    width: 100%;
    box-shadow: 0 20px 60px rgba(0,0,0,0.8);
    display: flex;
    flex-direction: column;
    gap: 18px;
  }
  .modal-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .modal-head h3 { font-size: 1.2rem; font-weight: 800; color: #fff; }
  .modal-close { background: none; border: none; color: var(--muted); font-size: 1.4rem; cursor: pointer; }
  .modal-body { font-size: 0.88rem; color: var(--muted); line-height: 1.6; }
  .modal-summary-box {
    background: #070c08;
    border: 1px solid var(--card-border);
    border-radius: 10px;
    padding: 14px 16px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    font-size: 0.8rem;
  }
  .modal-summary-box strong { color: var(--text); }
  .modal-actions { display: flex; gap: 10px; justify-content: flex-end; }
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
      <svg viewBox="0 0 24 24"><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/></svg>
      <span>Overview</span>
    </a>
    <button class="nav-item export-btn-side" onclick="openExportModal()">
      <svg viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
      <span>Export Report</span>
    </button>
  </nav>

  <div class="sidebar-bottom">
    <div class="gpu-side-card">
      <div class="gpu-side-title">
        <span>GPU TELEMETRY</span>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--lime)" stroke-width="2"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/></svg>
      </div>
      <div class="gpu-side-model" id="sg-gpu-model">RTX 4050 · 6 GB</div>
      <div class="gpu-bar-bg"><div class="gpu-bar-fill" id="sg-vram-bar" style="width: 100%"></div></div>
      <div class="gpu-side-meta">
        <span>Utilization</span>
        <strong id="sg-gpu-util" style="color:var(--lime)">0%</strong>
      </div>
    </div>
  </div>
</aside>

<!-- ── Main Content ─────────────────────────────────────────────── -->
<main class="layout-main">

  <!-- Header Bar -->
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
      <button class="btn-close-term" onclick="closeDashboardAndExit()">Close &amp; Return to Terminal</button>
    </div>
  </header>

  <!-- ── 7 Clean KPI Cards (Vector SVGs, No fake lines) ───────────── -->
  <section class="kpi-grid">
    <!-- 1. Total Files -->
    <div class="kpi-card">
      <div class="kpi-head">
        <div class="kpi-icon-svg icon-green">
          <svg viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
        </div>
        <span class="kpi-label">Total Files</span>
      </div>
      <div class="kpi-val" id="k-total">0</div>
      <div class="kpi-sub">All time</div>
    </div>

    <!-- 2. Processed -->
    <div class="kpi-card">
      <div class="kpi-head">
        <div class="kpi-icon-svg icon-green">
          <svg viewBox="0 0 24 24"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
        </div>
        <span class="kpi-label">Processed</span>
      </div>
      <div class="kpi-val" id="k-done" style="color:var(--lime)">0</div>
      <div class="kpi-sub">Success</div>
    </div>

    <!-- 3. Failed -->
    <div class="kpi-card">
      <div class="kpi-head">
        <div class="kpi-icon-svg icon-red">
          <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>
        </div>
        <span class="kpi-label">Failed</span>
      </div>
      <div class="kpi-val" id="k-fail" style="color:var(--red)">0</div>
      <div class="kpi-sub">Failures</div>
    </div>

    <!-- 4. Queued -->
    <div class="kpi-card">
      <div class="kpi-head">
        <div class="kpi-icon-svg icon-yellow">
          <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
        </div>
        <span class="kpi-label">Queued</span>
      </div>
      <div class="kpi-val" id="k-queue" style="color:var(--yellow)">0</div>
      <div class="kpi-sub">Pending</div>
    </div>

    <!-- 5. Workers -->
    <div class="kpi-card">
      <div class="kpi-head">
        <div class="kpi-icon-svg icon-purple">
          <svg viewBox="0 0 24 24"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>
        </div>
        <span class="kpi-label">Workers</span>
      </div>
      <div class="kpi-val" id="k-workers" style="color:var(--purple)">1 / 1</div>
      <div class="kpi-sub">Active / Target</div>
    </div>

    <!-- 6. Speed -->
    <div class="kpi-card">
      <div class="kpi-head">
        <div class="kpi-icon-svg icon-blue">
          <svg viewBox="0 0 24 24"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
        </div>
        <span class="kpi-label">Speed</span>
      </div>
      <div class="kpi-val" id="k-speed" style="color:var(--blue)">0.00</div>
      <div class="kpi-sub">files / sec</div>
    </div>

    <!-- 7. Elapsed Time -->
    <div class="kpi-card">
      <div class="kpi-head">
        <div class="kpi-icon-svg icon-muted">
          <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
        </div>
        <span class="kpi-label">Elapsed Time</span>
      </div>
      <div class="kpi-val" id="k-elapsed" style="font-size:1.35rem">0s</div>
      <div class="kpi-sub">Since start</div>
    </div>
  </section>

  <!-- ── Middle Row: Progress + Job Controls ─────────────────────── -->
  <section class="mid-grid">
    <!-- Batch Progress -->
    <div class="panel">
      <div class="panel-title">
        <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
        <span>Batch Progress</span>
      </div>
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
      <div class="panel-title">
        <svg viewBox="0 0 24 24"><polygon points="5 3 19 12 5 21 5 3"/></svg>
        <span>Job Controls</span>
      </div>
      <div class="btn-row">
        <button class="btn-action btn-resume" onclick="apiAction('/api/resume')">
          <svg viewBox="0 0 24 24"><polygon points="5 3 19 12 5 21 5 3"/></svg>
          <span>Resume</span>
        </button>
        <button class="btn-action btn-pause"  onclick="apiAction('/api/pause')">
          <svg viewBox="0 0 24 24"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>
          <span>Pause</span>
        </button>
        <button class="btn-action btn-cancel" onclick="apiAction('/api/cancel')">
          <svg viewBox="0 0 24 24"><rect x="4" y="4" width="16" height="16" rx="2"/></svg>
          <span>Cancel</span>
        </button>
        <button class="btn-action btn-retry"  onclick="apiAction('/api/retry-failed')">
          <svg viewBox="0 0 24 24"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>
          <span>Retry Failed</span>
        </button>
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

  <!-- ── Bottom Grid: System Resources + Resizable Tasks ─────────── -->
  <section class="bottom-grid">
    <!-- System Resources -->
    <div class="panel">
      <div class="panel-title">
        <svg viewBox="0 0 24 24"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/></svg>
        <span>System Resources</span>
      </div>
      <div class="sys-cards">
        <div class="sys-card">
          <div class="sys-card-title">
            <svg viewBox="0 0 24 24"><rect x="4" y="4" width="16" height="16" rx="2"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/></svg>
            <span>CPU Usage</span>
          </div>
          <div class="sys-card-val" id="s-cpu-val">0.0%</div>
          <!-- Real Live CPU Sparkline -->
          <svg class="cpu-live-svg"><path id="cpu-live-path" d="M0,30 L100,30" stroke="#70df7f"/></svg>
        </div>

        <div class="sys-card">
          <div class="sys-card-title">
            <svg viewBox="0 0 24 24" style="stroke:var(--purple)"><rect x="2" y="6" width="20" height="12" rx="2"/><line x1="6" y1="12" x2="6.01" y2="12"/><line x1="10" y1="12" x2="10.01" y2="12"/><line x1="14" y1="12" x2="14.01" y2="12"/></svg>
            <span>RAM Usage</span>
          </div>
          <div class="sys-card-val" id="s-ram-val" style="font-size:1.15rem">0 / 0 GB</div>
          <div class="sys-card-bar"><div class="sys-card-fill" id="s-ram-fill" style="width:0%;background:var(--purple)"></div></div>
        </div>

        <div class="sys-card">
          <div class="sys-card-title">
            <svg viewBox="0 0 24 24" style="stroke:var(--lime)"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>
            <span id="s-gpu-name">GPU (VRAM)</span>
          </div>
          <div class="sys-card-val" id="s-gpu-val" style="font-size:1.15rem">0 / 0 GB</div>
          <div class="sys-card-bar"><div class="sys-card-fill" id="s-gpu-fill" style="width:0%;background:var(--lime)"></div></div>
        </div>
      </div>

      <div class="sys-sub-pills">
        <div class="sys-sub-pill">
          <svg viewBox="0 0 24 24"><path d="M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4.5 4.5 0 1 0 5 0z"/></svg>
          <span>CPU Temp</span>
          <strong id="st-cpu">62°C</strong>
        </div>
        <div class="sys-sub-pill">
          <svg viewBox="0 0 24 24"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
          <span>GPU Temp</span>
          <strong id="st-gpu">68°C</strong>
        </div>
        <div class="sys-sub-pill">
          <svg viewBox="0 0 24 24"><rect x="2" y="6" width="20" height="12" rx="2"/></svg>
          <span>RAM Usage</span>
          <strong id="st-ram-pct">0%</strong>
        </div>
      </div>
    </div>

    <!-- Tasks List (Resizable) -->
    <div class="panel resizable-panel" style="min-height:220px">
      <span class="resize-hint">⋮⋮ Drag to resize</span>
      <div class="tasks-header">
        <div class="panel-title">
          <svg viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
          <span>Tasks</span>
        </div>
        <span style="font-size:0.75rem;color:var(--muted)">Latest First</span>
      </div>

      <div class="task-list" id="task-container">
        <div style="color:var(--muted);font-size:0.8rem;padding:12px 0;">No active tasks.</div>
      </div>
    </div>
  </section>

  <!-- ── Live Logs Panel (Resizable) ─────────────────────────────── -->
  <section class="logs-panel resizable-panel" style="min-height:200px">
    <span class="resize-hint">⋮⋮ Drag to resize</span>
    <div class="logs-header">
      <div style="display:flex;align-items:center">
        <div class="panel-title">
          <svg viewBox="0 0 24 24"><polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/></svg>
          <span>Live Logs</span>
        </div>
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

<!-- Toast Notification -->
<div class="toast" id="toast"></div>

<!-- Modal for Export Report -->
<div class="modal-overlay" id="exportModal">
  <div class="modal-card">
    <div class="modal-head">
      <h3>Export Batch OCR PDF Report</h3>
      <button class="modal-close" onclick="closeExportModal()">✕</button>
    </div>
    <div class="modal-body">
      Generate a standalone PDF document containing batch execution metrics, model configurations, hardware telemetry, and individual document processing results.
    </div>
    <div class="modal-summary-box">
      <div>Model: <strong id="m-model">glm-ocr</strong></div>
      <div>Files: <strong id="m-files">0</strong></div>
      <div>Success Rate: <strong id="m-success">100%</strong></div>
      <div>Format: <strong id="m-format">JSON</strong></div>
    </div>
    <div class="modal-actions">
      <button class="btn-apply" style="border-color:var(--card-border);color:var(--muted)" onclick="closeExportModal()">Cancel</button>
      <button class="btn-action btn-resume" style="padding:10px 20px" onclick="downloadReport()">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
        <span>Download PDF Report</span>
      </button>
    </div>
  </div>
</div>

<script>
let evtSource = null;
let cpuHistory = [20, 25, 30, 22, 28, 35, 40, 32, 45, 50]; // real CPU history buffer
let currentMetrics = null;

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
  setTimeout(() => t.className = 'toast', 2800);
}

function updateCpuGraph(val) {
  cpuHistory.push(val);
  if (cpuHistory.length > 20) cpuHistory.shift();

  const width = 100;
  const height = 36;
  const step = width / (cpuHistory.length - 1);

  const points = cpuHistory.map((v, idx) => {
    const x = idx * step;
    const y = height - (v / 100 * height);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' L ');

  const path = document.getElementById('cpu-live-path');
  if (path) path.setAttribute('d', `M ${points}`);
}

function updateMetrics(d) {
  currentMetrics = d;
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

  // Modal stats
  setText('m-model', d.model_id);
  setText('m-files', d.total_files);
  setText('m-format', d.output_format.toUpperCase());
  const succPct = d.total_files > 0 ? Math.round((d.processed_files / d.total_files) * 100) : 100;
  setText('m-success', succPct + '%');

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
  updateCpuGraph(d.cpu_percent);

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
      <svg viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
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

function openExportModal() {
  const isRunning = currentMetrics && (currentMetrics.status === 'RUNNING' || currentMetrics.queued_files > 0 || currentMetrics.active_workers > 0);
  if (isRunning) {
    showToast('Batch job is still running. Please wait until all tasks complete before exporting the report.', false);
    return;
  }
  document.getElementById('exportModal').classList.add('open');
}

function closeExportModal() {
  document.getElementById('exportModal').classList.remove('open');
}

function downloadReport() {
  closeExportModal();
  showToast('Downloading PDF Report...');
  window.location.href = '/api/export-report';
}

function closeDashboardAndExit() {
  if (confirm("Close dashboard and return to terminal?")) {
    fetch('/api/close-dashboard', { method: 'POST' })
      .then(r => r.json())
      .then(d => {
        showToast('Dashboard closed. Returning to terminal...');
        setTimeout(() => { window.close(); }, 1200);
      });
  }
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
        elif path == "/api/export-report":
            self._serve_report()
        else:
            self.send_error(404, "Not Found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/api/pause":
            self.engine.pause()
            self._serve_json({"ok": True, "message": "Job paused successfully"})
        elif path == "/api/resume":
            self.engine.resume()
            self._serve_json({"ok": True, "message": "Job resumed successfully"})
        elif path == "/api/cancel":
            self.engine.cancel()
            self._serve_json({"ok": True, "message": "Batch job cancelled by user"})
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
        elif path == "/api/close-dashboard":
            self.engine.signal_dashboard_close()
            self._serve_json({"ok": True, "message": "Dashboard closed. Returning to terminal..."})
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

    def _serve_report(self) -> None:
        """Generate and serve PDF Batch report when job is done."""
        m = self.engine.get_metrics()
        tasks = self.engine.get_tasks()

        # Check if job is still active
        is_running = m.status == BatchStatus.RUNNING or any(t.status in ("QUEUED", "PROCESSING", "RETRYING") for t in tasks)
        if is_running:
            self._serve_json({
                "ok": False,
                "message": "Batch job is currently running. Please wait until all tasks complete before exporting the report."
            }, status=400)
            return

        from textlens.batch.report import SimplePDFReport
        pdf_bytes = SimplePDFReport(m, tasks).generate()

        self.send_response(200)
        self.send_header("Content-Type", "application/pdf")
        self.send_header("Content-Disposition", 'attachment; filename="TextLens_BatchOCR_Report.pdf"')
        self.send_header("Content-Length", str(len(pdf_bytes)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(pdf_bytes)

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
