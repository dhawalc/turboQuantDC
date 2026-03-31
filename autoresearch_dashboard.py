#!/usr/bin/env python3
"""Live dashboard for TurboQuantDC AutoResearch — watch the machine think."""

import http.server
import json
import os
import time

PORT = 8822
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "autoresearch_results.jsonl")

HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>TurboQuantDC AutoResearch</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #0a0a0f; color: #e0e0e0; font-family: 'SF Mono', 'Fira Code', monospace; }
.header { text-align: center; padding: 30px; border-bottom: 1px solid #1a1a2e; }
.header h1 { color: #00f0ff; font-size: 28px; margin-bottom: 8px; }
.header .sub { color: #666; font-size: 14px; }
.stats { display: flex; justify-content: center; gap: 40px; padding: 25px; border-bottom: 1px solid #1a1a2e; }
.stat { text-align: center; }
.stat .value { font-size: 36px; font-weight: bold; }
.stat .label { color: #888; font-size: 12px; margin-top: 4px; }
.stat.best .value { color: #00ff88; }
.stat.rounds .value { color: #00f0ff; }
.stat.compression .value { color: #ff6b35; }
.content { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; padding: 20px; max-width: 1400px; margin: 0 auto; }
.panel { background: #111118; border: 1px solid #1a1a2e; border-radius: 8px; padding: 20px; }
.panel h2 { color: #00f0ff; font-size: 16px; margin-bottom: 15px; border-bottom: 1px solid #1a1a2e; padding-bottom: 8px; }
.panel.full { grid-column: 1 / -1; }
canvas { width: 100%; height: 300px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { text-align: left; color: #00f0ff; padding: 8px 6px; border-bottom: 1px solid #1a1a2e; }
td { padding: 6px; border-bottom: 1px solid #0a0a12; }
tr:hover { background: #1a1a2e; }
.score-bar { height: 8px; border-radius: 4px; background: #1a1a2e; position: relative; }
.score-fill { height: 100%; border-radius: 4px; position: absolute; left: 0; top: 0; }
.good { background: #00ff88; }
.ok { background: #ffaa00; }
.bad { background: #ff4444; }
.live-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #00ff88; animation: pulse 1.5s infinite; margin-right: 8px; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }
.pareto { color: #00ff88; font-weight: bold; }
.log { font-size: 12px; color: #888; max-height: 200px; overflow-y: auto; white-space: pre; line-height: 1.6; }
.log .star { color: #ffaa00; }
.best-config { background: #0a1a0a; border: 1px solid #00ff88; border-radius: 8px; padding: 15px; margin-top: 10px; font-size: 14px; }
.best-config .key { color: #00f0ff; }
.best-config .val { color: #00ff88; font-weight: bold; }
</style>
</head>
<body>

<div class="header">
  <h1><span class="live-dot"></span>TurboQuantDC AutoResearch</h1>
  <div class="sub">Autonomous KV cache compression optimization -- Karpathy-style experimentation</div>
</div>

<div class="stats">
  <div class="stat rounds"><div class="value" id="rounds">-</div><div class="label">Rounds Complete</div></div>
  <div class="stat best"><div class="value" id="bestScore">-</div><div class="label">Best Score</div></div>
  <div class="stat compression"><div class="value" id="bestCompression">-</div><div class="label">Best Compression</div></div>
  <div class="stat"><div class="value" id="elapsed">-</div><div class="label">Time Elapsed</div></div>
</div>

<div class="content">
  <div class="panel">
    <h2>Pareto Frontier (Compression vs Quality)</h2>
    <canvas id="paretoChart"></canvas>
  </div>

  <div class="panel">
    <h2>Best Configuration Found</h2>
    <div id="bestConfig" class="best-config">Waiting for data...</div>
    <h2 style="margin-top:20px">Live Feed</h2>
    <div id="logFeed" class="log"></div>
  </div>

  <div class="panel full">
    <h2>All Results (sorted by score)</h2>
    <table id="resultsTable">
      <thead>
        <tr>
          <th>#</th><th>Score</th><th>Compression</th><th>K bits</th><th>V bits</th>
          <th>Anchors</th><th>Window</th><th>ResQ</th><th>Quality</th>
        </tr>
      </thead>
      <tbody id="resultsBody"></tbody>
    </table>
  </div>
</div>

<script>
let allResults = [];
let startTime = null;

async function fetchResults() {
  try {
    const resp = await fetch('/api/results');
    const data = await resp.json();
    allResults = data.results;
    if (allResults.length > 0 && !startTime) {
      startTime = Date.now() - (allResults.length * 30 * 1000); // estimate
    }
    updateDashboard();
  } catch(e) {}
}

function updateDashboard() {
  if (allResults.length === 0) return;

  // Stats
  document.getElementById('rounds').textContent = allResults.length;

  const best = allResults.reduce((a, b) => a.total_score > b.total_score ? a : b);
  document.getElementById('bestScore').textContent = (best.total_score * 100).toFixed(1) + '%';
  document.getElementById('bestCompression').textContent = (best.compression || 0).toFixed(1) + 'x';

  const mins = Math.floor((Date.now() - (startTime || Date.now())) / 60000);
  document.getElementById('elapsed').textContent = mins + 'min';

  // Best config
  const c = best.config || {};
  document.getElementById('bestConfig').innerHTML =
    `<div><span class="key">Key bits:</span> <span class="val">${c.key_bits || '?'}</span></div>` +
    `<div><span class="key">Val bits:</span> <span class="val">${c.val_bits || '?'}</span></div>` +
    `<div><span class="key">Anchor interval:</span> <span class="val">${c.anchor_interval || 'none'}</span></div>` +
    `<div><span class="key">FP16 window:</span> <span class="val">${c.fp16_window || 0}</span></div>` +
    `<div><span class="key">ResidualQuant:</span> <span class="val">${c.use_residual_quant ? 'YES' : 'no'}</span></div>` +
    `<div style="margin-top:10px"><span class="key">Score:</span> <span class="val">${(best.total_score * 100).toFixed(1)}%</span></div>` +
    `<div><span class="key">Compression:</span> <span class="val">${(best.compression || 0).toFixed(1)}x</span></div>`;

  // Log feed (last 10)
  const logLines = allResults.slice(-10).reverse().map(r => {
    const cfg = r.config || {};
    const star = r.total_score >= 0.8 ? '<span class="star">*</span>' : ' ';
    return `${star} [${r.round}] score=${(r.total_score*100).toFixed(0)}% ` +
           `comp=${(r.compression||0).toFixed(1)}x ` +
           `k=${cfg.key_bits||'?'}b v=${cfg.val_bits||'?'}b ` +
           `anc=${cfg.anchor_interval||0} resq=${cfg.use_residual_quant?'Y':'N'}`;
  }).join('\\n');
  document.getElementById('logFeed').innerHTML = logLines;

  // Results table
  const sorted = [...allResults].sort((a, b) => b.total_score - a.total_score);
  const tbody = document.getElementById('resultsBody');
  tbody.innerHTML = sorted.slice(0, 50).map((r, i) => {
    const c = r.config || {};
    const pct = Math.round(r.total_score * 100);
    const cls = pct >= 80 ? 'good' : pct >= 50 ? 'ok' : 'bad';
    const isPareto = isParetoOptimal(r, allResults);
    return `<tr>` +
      `<td>${r.round}</td>` +
      `<td>${isPareto ? '<span class="pareto">' : ''}${pct}%${isPareto ? ' P</span>' : ''}</td>` +
      `<td>${(r.compression||0).toFixed(1)}x</td>` +
      `<td>${c.key_bits||'?'}</td><td>${c.val_bits||'?'}</td>` +
      `<td>${c.anchor_interval||'-'}</td><td>${c.fp16_window||'-'}</td>` +
      `<td>${c.use_residual_quant?'Y':'N'}</td>` +
      `<td><div class="score-bar"><div class="score-fill ${cls}" style="width:${pct}%"></div></div></td>` +
      `</tr>`;
  }).join('');

  // Pareto chart
  drawPareto();
}

function isParetoOptimal(r, all) {
  return !all.some(o =>
    o.compression >= r.compression && o.total_score >= r.total_score &&
    (o.compression > r.compression || o.total_score > r.total_score)
  );
}

function drawPareto() {
  const canvas = document.getElementById('paretoChart');
  const ctx = canvas.getContext('2d');
  canvas.width = canvas.offsetWidth * 2;
  canvas.height = 600;
  const W = canvas.width, H = canvas.height;

  ctx.clearRect(0, 0, W, H);

  if (allResults.length === 0) return;

  const maxComp = Math.max(...allResults.map(r => r.compression || 1)) * 1.1;
  const pad = { l: 80, r: 30, t: 30, b: 50 };

  // Grid
  ctx.strokeStyle = '#1a1a2e';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const y = pad.t + (H - pad.t - pad.b) * i / 5;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
    ctx.fillStyle = '#666'; ctx.font = '20px monospace';
    ctx.fillText((100 - i * 20) + '%', 10, y + 6);
  }
  for (let x = 1; x <= maxComp; x++) {
    const px = pad.l + (W - pad.l - pad.r) * x / maxComp;
    ctx.beginPath(); ctx.moveTo(px, pad.t); ctx.lineTo(px, H - pad.b); ctx.stroke();
    ctx.fillText(x + 'x', px - 10, H - 15);
  }

  // Labels
  ctx.fillStyle = '#888'; ctx.font = '22px monospace';
  ctx.fillText('Quality', 5, pad.t - 10);
  ctx.fillText('Compression', W / 2 - 50, H - 2);

  // Points
  allResults.forEach(r => {
    const x = pad.l + (W - pad.l - pad.r) * (r.compression || 0) / maxComp;
    const y = pad.t + (H - pad.t - pad.b) * (1 - r.total_score);
    const pareto = isParetoOptimal(r, allResults);

    ctx.beginPath();
    ctx.arc(x, y, pareto ? 10 : 6, 0, Math.PI * 2);
    ctx.fillStyle = pareto ? '#00ff88' : r.total_score >= 0.8 ? '#00f0ff' : r.total_score >= 0.5 ? '#ffaa00' : '#ff4444';
    ctx.fill();
    if (pareto) {
      ctx.strokeStyle = '#00ff88'; ctx.lineWidth = 2; ctx.stroke();
    }
  });

  // Pareto frontier line
  const pareto = allResults.filter(r => isParetoOptimal(r, allResults))
    .sort((a, b) => a.compression - b.compression);
  if (pareto.length > 1) {
    ctx.beginPath();
    ctx.strokeStyle = '#00ff8866'; ctx.lineWidth = 3;
    pareto.forEach((r, i) => {
      const x = pad.l + (W - pad.l - pad.r) * r.compression / maxComp;
      const y = pad.t + (H - pad.t - pad.b) * (1 - r.total_score);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
}

// Poll every 5 seconds
fetchResults();
setInterval(fetchResults, 5000);
window.addEventListener('resize', drawPareto);
</script>
</body>
</html>"""


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML.encode())
        elif self.path == '/api/results':
            results = []
            if os.path.exists(RESULTS_FILE):
                with open(RESULTS_FILE) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                results.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
            body = json.dumps({"results": results}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    server = http.server.HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"\n  AutoResearch Dashboard: http://localhost:{PORT}\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
