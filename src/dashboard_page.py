"""Self-contained HTML status dashboard, served at GET /dashboard.

Inline string (not a static file) so .railwayignore can never exclude it from
the deploy image. Zero build step, zero CDN dependencies, free.

Resilience contract ("never fails to refresh"):
- polls every 30s with AbortController timeouts — a hung request can't wedge
  the refresh loop; every section fails independently
- failures flip a visible red OFFLINE banner and keep retrying forever
- deploy-drift check compares the running version against APP_VERSION in
  main.py on GitHub raw (CORS-open)
- polling pauses while the tab is hidden (visibilitychange) and resumes
  with an immediate refresh on return

v2 (2026-07-21): interactive equity chart (crosshair + tooltip + axes),
trade analytics (cumulative PnL, exit-reason breakdown), filterable +
sortable trades table, Kelly/sizing card, Integrity Sentinel card + checks,
section nav, refresh countdown.
"""

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>StockBot Status</title>
<style>
  :root {
    color-scheme: dark;
    --bg: #0d1117; --panel: #161b22; --border: #30363d; --border2: #21262d;
    --ink: #e6edf3; --ink2: #8b949e; --accent: #58a6ff;
    --ok: #3fb950; --warn: #d29922; --crit: #f85149;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html { scroll-behavior: smooth; scroll-padding-top: 64px; }
  body { font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
         background: var(--bg); color: var(--ink); max-width: 1280px;
         margin: 0 auto; padding: 0 20px 40px; }
  header { position: sticky; top: 0; z-index: 20; background: rgba(13,17,23,.92);
           backdrop-filter: blur(6px); border-bottom: 1px solid var(--border2);
           margin: 0 -20px 16px; padding: 10px 20px;
           display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
  header h1 { font-size: 17px; margin-right: 2px; }
  .pill { font-size: 12px; font-weight: 700; padding: 3px 10px; border-radius: 999px;
          border: 1px solid var(--border); }
  .pill.ok { color: var(--ok); border-color: rgba(63,185,80,.4); background: rgba(63,185,80,.08); }
  .pill.warn { color: var(--warn); border-color: rgba(210,153,34,.4); background: rgba(210,153,34,.08); }
  .pill.crit { color: var(--crit); border-color: rgba(248,81,73,.4); background: rgba(248,81,73,.08); }
  nav { display: flex; gap: 4px; flex-wrap: wrap; margin-left: auto; }
  nav a { color: var(--ink2); font-size: 12px; text-decoration: none;
          padding: 4px 9px; border-radius: 6px; }
  nav a:hover { color: var(--ink); background: var(--border2); }
  .hstat { color: var(--ink2); font-size: 12px; display: flex; gap: 10px; align-items: center; }
  .hstat button { background: var(--border2); color: var(--ink2); border: 1px solid var(--border);
          border-radius: 6px; padding: 3px 10px; font-size: 12px; cursor: pointer; }
  .hstat button:hover { color: var(--ink); border-color: var(--accent); }
  h2 { font-size: 13px; color: var(--ink2); margin: 24px 0 10px;
       text-transform: uppercase; letter-spacing: .6px; display: flex;
       align-items: center; gap: 8px; }
  h2 .spacer { flex: 1; }
  #banner { display: none; background: var(--crit); color: #fff; padding: 10px 14px;
            border-radius: 8px; margin-bottom: 14px; font-weight: 600; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(196px, 1fr));
          gap: 12px; }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
          padding: 13px 14px; transition: border-color .15s; }
  .card:hover { border-color: #3d444d; }
  .card h3 { font-size: 11px; text-transform: uppercase; letter-spacing: .5px;
             color: var(--ink2); margin-bottom: 7px; display: flex; gap: 6px;
             align-items: center; justify-content: space-between; }
  .big { font-size: 21px; font-weight: 700; font-variant-numeric: tabular-nums; }
  .ok { color: var(--ok); } .warn { color: var(--warn); } .crit { color: var(--crit); }
  .muted { color: var(--ink2); font-size: 12px; margin-top: 6px; line-height: 1.45; }
  table { width: 100%; border-collapse: collapse; font-size: 13px;
          font-variant-numeric: tabular-nums; }
  th { text-align: left; color: var(--ink2); font-weight: 500; padding: 6px 8px;
       border-bottom: 1px solid var(--border); white-space: nowrap; }
  th.sortable { cursor: pointer; user-select: none; }
  th.sortable:hover { color: var(--ink); }
  th .arrow { opacity: .8; font-size: 10px; }
  td { padding: 7px 8px; border-bottom: 1px solid var(--border2); }
  tbody tr:hover td { background: rgba(88,166,255,.045); }
  tbody tr:last-child td { border-bottom: none; }
  .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
           padding: 14px; }
  .dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%;
         margin-right: 7px; flex: none; }
  .dot.ok { background: var(--ok); } .dot.warn { background: var(--warn); }
  .dot.crit { background: var(--crit); }
  .checkrow { display: flex; justify-content: space-between; gap: 10px; align-items: baseline;
              padding: 6px 0; border-bottom: 1px solid var(--border2); font-size: 13px; }
  .checkrow:last-child { border-bottom: none; }
  .checkrow .d { color: var(--ink2); text-align: right; max-width: 62%; }
  .bar { background: var(--border2); border-radius: 4px; height: 7px; width: 90px;
         display: inline-block; vertical-align: middle; overflow: hidden; }
  .bar > i { display: block; height: 100%; border-radius: 4px; background: var(--ok); }
  .bar > i.crit { background: var(--crit); }
  .btns button, .chip { background: var(--border2); color: var(--ink2); border: 1px solid var(--border);
       border-radius: 6px; padding: 3px 10px; font-size: 12px; cursor: pointer; }
  .btns button.active, .chip.active { color: var(--ink); border-color: var(--accent); }
  .tag { display: inline-block; font-size: 11px; padding: 1px 7px; border-radius: 999px;
         border: 1px solid var(--border); color: var(--ink2); margin: 1px 2px 1px 0; }
  a { color: var(--accent); }
  details.strategy summary { cursor: pointer; color: var(--ink2); font-size: 13px;
         list-style: none; }
  details.strategy summary::before { content: "▸ "; }
  details.strategy[open] summary::before { content: "▾ "; }
  #tooltip { position: fixed; pointer-events: none; z-index: 50; display: none;
             background: #1c2128; border: 1px solid var(--border); border-radius: 8px;
             padding: 8px 10px; font-size: 12px; line-height: 1.5;
             box-shadow: 0 6px 18px rgba(0,0,0,.5); font-variant-numeric: tabular-nums; }
  .two { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  @media (max-width: 800px) { .two { grid-template-columns: 1fr; } nav { display: none; } }
  svg text { fill: var(--ink2); font-size: 10px; font-family: inherit; }
  .scroll { overflow-x: auto; }
  #posTable td, #posTable th, #tradeTable td, #tradeTable th { white-space: nowrap; }
</style>
</head>
<body>
<header>
  <h1>🤖 StockBot</h1>
  <span class="pill" id="hpill">…</span>
  <span class="pill" id="mktpill">market …</span>
  <nav>
    <a href="#overview">Overview</a><a href="#equity">Equity</a>
    <a href="#positions">Positions</a><a href="#signals">Signals</a>
    <a href="#trades">Trades</a><a href="#system">System</a>
  </nav>
  <span class="hstat">
    <span id="updated">–</span>
    <span id="countdown" title="next auto-refresh">30s</span>
    <button id="refreshNow" title="refresh now">↻</button>
  </span>
</header>

<div id="banner">⚠️ CANNOT REACH BOT — retrying every 30s…</div>
<div id="tooltip"></div>

<section id="overview">
<div class="grid">
  <div class="card"><h3>Bot</h3><div class="big" id="bot">–</div><div class="muted" id="botd"></div></div>
  <div class="card"><h3>Deployment</h3><div class="big" id="deploy">–</div><div class="muted" id="deployd"></div></div>
  <div class="card"><h3>Portfolio</h3><div class="big" id="pv">–</div><div class="muted" id="pvd"></div></div>
  <div class="card"><h3>Daily PnL</h3><div class="big" id="pnl">–</div><div class="muted" id="pnld"></div></div>
  <div class="card"><h3>Positions · Heat</h3><div class="big" id="pos">–</div>
    <div class="muted"><span class="bar" title="portfolio heat vs 75% ceiling"><i id="heatbar"></i></span>
    <span id="posd"></span></div></div>
  <div class="card"><h3>Kelly sizing</h3><div class="big" id="kelly">–</div><div class="muted" id="kellyd"></div></div>
  <div class="card"><h3>Ledger integrity</h3><div class="big" id="integ">–</div><div class="muted" id="integd"></div></div>
  <div class="card"><h3>News risk</h3><div class="big" id="news">–</div><div class="muted" id="newsd"></div></div>
  <div class="card"><h3>Trading halt</h3><div class="big" id="halt">–</div><div class="muted" id="haltd"></div></div>
  <div class="card"><h3>Last loop tick</h3><div class="big" id="tick">–</div><div class="muted" id="tickd"></div></div>
  <div class="card"><h3>Tick errors</h3><div class="big" id="errors">–</div><div class="muted" id="errorsd"></div></div>
</div>
</section>

<section id="equity">
<h2>Equity curve <span class="spacer"></span>
  <span class="btns" id="rangeBtns">
    <button data-r="1D-15Min">1D</button>
    <button data-r="1W-1H">1W</button>
    <button data-r="1M-1D" class="active">1M</button>
    <button data-r="3M-1D">3M</button>
  </span>
</h2>
<div class="panel">
  <svg id="chart" width="100%" height="240"></svg>
  <div class="muted" id="chartd">loading…</div>
</div>
</section>

<section id="positions">
<h2>Open positions</h2>
<div class="panel scroll">
  <table id="posTable"><thead></thead>
    <tbody id="positions"><tr><td colspan="9">loading…</td></tr></tbody></table>
</div>
</section>

<section id="signals">
<h2>Signal watchlist — bullish setups and what's gating them</h2>
<div class="panel scroll">
  <table><thead><tr>
    <th>Ticker</th><th>Pred return</th><th>P(up)</th><th>Ensemble</th><th>Status</th>
  </tr></thead><tbody id="watchlist"><tr><td colspan="5">loading…</td></tr></tbody></table>
</div>
</section>

<section id="trades">
<h2>Trade performance</h2>
<div class="grid" style="margin-bottom:12px">
  <div class="card"><h3>Win rate</h3><div class="big" id="tr_wr">–</div><div class="muted" id="tr_wrd"></div></div>
  <div class="card"><h3>Profit factor</h3><div class="big" id="tr_pf">–</div><div class="muted">gross win / gross loss</div></div>
  <div class="card"><h3>Expectancy</h3><div class="big" id="tr_exp">–</div><div class="muted">avg PnL per trade</div></div>
  <div class="card"><h3>Total PnL</h3><div class="big" id="tr_pnl">–</div><div class="muted" id="tr_pnld"></div></div>
  <div class="card"><h3>Avg hold</h3><div class="big" id="tr_hold">–</div><div class="muted" id="tr_holdd"></div></div>
</div>
<div class="two" style="margin-bottom:12px">
  <div class="panel">
    <h3 style="font-size:11px;text-transform:uppercase;color:var(--ink2);margin-bottom:8px">
      Cumulative realized PnL</h3>
    <svg id="cumchart" width="100%" height="140"></svg>
  </div>
  <div class="panel">
    <h3 style="font-size:11px;text-transform:uppercase;color:var(--ink2);margin-bottom:8px">
      Exits by reason — count and net PnL</h3>
    <div id="exitReasons" class="muted">loading…</div>
  </div>
</div>
<h2>Closed trades <span class="spacer"></span>
  <span class="btns" id="tradeFilters">
    <button data-f="all" class="active">All</button>
    <button data-f="win">Wins</button>
    <button data-f="loss">Losses</button>
  </span>
  <select id="reasonFilter" class="chip" style="appearance:auto">
    <option value="">every exit reason</option>
  </select>
</h2>
<div class="panel scroll">
  <table id="tradeTable"><thead></thead>
    <tbody id="tradesBody"><tr><td colspan="8">loading…</td></tr></tbody></table>
  <div class="muted" id="tradesFoot"></div>
</div>
</section>

<h2>The strategy (what you're watching)</h2>
<div class="panel">
<details class="strategy" open>
<summary>LightGBM signal → Kelly-governed sizing → volatility-scaled exits</summary>
<div class="muted" style="margin-top:8px">
  <b style="color:var(--ink)">Signal:</b> a LightGBM model predicts next-day returns
  from 40 engineered features (momentum, VWAP, options flow, volume-informed
  trading) refreshed every minute. Entries require top-8% conviction
  (self-calibrating threshold), P(up) ≥ 60%, and a positive live information
  coefficient on that specific ticker.<br>
  <b style="color:var(--ink)">Sizing:</b> conviction-proportional, volatility-normalized,
  Kelly-governed; max 15% per position, 6 positions, 75% portfolio heat ceiling,
  max 2 per sector.<br>
  <b style="color:var(--ink)">Exits:</b> daily-volatility-scaled stop (~1.1σ),
  trailing (~1.2σ), target (~3σ), 1-trading-day max hold; reversal exit only on
  a confirmed tradeable opposite signal sustained 45 minutes.<br>
  <b style="color:var(--ink)">Risk rails:</b> daily-loss halt, drawdown halt,
  25% position breaker, sector caps — never overridden by the model.<br>
  <span class="warn">⚠ Paper trading. Not investment advice. Past
  performance does not guarantee future results.</span>
</div>
</details>
</div>

<section id="system">
<h2>System checks — watchdog</h2>
<div class="panel"><div id="checks">loading…</div></div>
<h2>System checks — ledger integrity</h2>
<div class="panel"><div id="ichecks">loading…</div></div>
</section>

<div class="muted" style="margin-top:16px">
  Raw: <a href="/watchdog">/watchdog</a> · <a href="/integrity">/integrity</a>
  · <a href="/diagnostics">/diagnostics</a> · <a href="/portfolio/summary">/portfolio/summary</a>
  · <a href="/portfolio/history">/portfolio/history</a> · <a href="/track">/track</a>
</div>

<script>
"use strict";
const $ = id => document.getElementById(id);
const cls = s => s === "ok" ? "ok" : (s === "warn" ? "warn" : "crit");
const esc = s => String(s ?? "").replace(/[&<>"']/g,
  c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
function fmt(n, d = 2) { return Number(n).toLocaleString(undefined, { maximumFractionDigits: d }); }
function money(n, d = 0) { return (n < 0 ? "-$" : "$") + fmt(Math.abs(n), d); }
function signed(n, d = 0) { return (n >= 0 ? "+" : "-") + "$" + fmt(Math.abs(n), d); }
function ago(iso) {
  if (!iso) return "never";
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 90) return Math.round(s) + "s ago";
  if (s < 5400) return Math.round(s / 60) + "m ago";
  if (s < 172800) return (s / 3600).toFixed(1) + "h ago";
  return Math.round(s / 86400) + "d ago";
}
async function fetchJson(url, ms = 10000) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), ms);
  try {
    const r = await fetch(url, { signal: ctl.signal, cache: "no-store" });
    if (!r.ok) throw new Error(url + " -> " + r.status);
    return await r.json();
  } finally { clearTimeout(timer); }
}

/* ── Tooltip ─────────────────────────────────────────────────────────── */
const tip = $("tooltip");
function showTip(html, x, y) {
  tip.innerHTML = html; tip.style.display = "block";
  const r = tip.getBoundingClientRect();
  tip.style.left = Math.min(x + 14, innerWidth - r.width - 8) + "px";
  tip.style.top = Math.max(8, y - r.height - 12) + "px";
}
function hideTip() { tip.style.display = "none"; }

/* ── Equity chart: axes + gridlines + crosshair tooltip ───────────────── */
let range = "1M-1D", eqPoints = null;
function drawEquity() {
  const svg = $("chart"), points = eqPoints;
  const W = svg.clientWidth || 900, H = 240, L = 52, R = 10, T = 10, B = 20;
  svg.setAttribute("viewBox", "0 0 " + W + " " + H);
  if (!points || points.length < 2) {
    svg.innerHTML = ""; $("chartd").textContent = "not enough history yet"; return;
  }
  const eq = points.map(p => p.equity);
  const lo = Math.min(...eq), hi = Math.max(...eq), span = (hi - lo) || 1;
  const x = i => L + (W - L - R) * i / (points.length - 1);
  const y = v => T + (H - T - B) * (1 - (v - lo) / span);
  const up = eq[eq.length - 1] >= eq[0];
  const color = up ? "var(--ok)" : "var(--crit)";
  let g = "";
  for (let k = 0; k <= 3; k++) {                       // recessive grid + y labels
    const v = lo + span * k / 3, yy = y(v);
    g += '<line x1="' + L + '" x2="' + (W - R) + '" y1="' + yy + '" y2="' + yy +
         '" stroke="#21262d" stroke-width="1"></line>' +
         '<text x="' + (L - 6) + '" y="' + (yy + 3) + '" text-anchor="end">$' +
         fmt(v, 0) + "</text>";
  }
  const y0 = y(eq[0]);                                  // period-start reference
  g += '<line x1="' + L + '" x2="' + (W - R) + '" y1="' + y0 + '" y2="' + y0 +
       '" stroke="#8b949e" stroke-width="1" stroke-dasharray="3 4" opacity=".5"></line>';
  let d = "M " + x(0) + " " + y(eq[0]);
  for (let i = 1; i < eq.length; i++) d += " L " + x(i) + " " + y(eq[i]);
  const area = d + " L " + x(eq.length - 1) + " " + (H - B) + " L " + x(0) + " " + (H - B) + " Z";
  const t0 = new Date(points[0].t * 1000), t1 = new Date(points[points.length - 1].t * 1000);
  g += '<text x="' + L + '" y="' + (H - 5) + '">' + t0.toLocaleDateString() + "</text>" +
       '<text x="' + (W - R) + '" y="' + (H - 5) + '" text-anchor="end">' +
       t1.toLocaleDateString() + "</text>";
  svg.innerHTML = g +
    '<path d="' + area + '" fill="' + (up ? "#3fb950" : "#f85149") + '" opacity="0.10"></path>' +
    '<path d="' + d + '" fill="none" stroke="' + color + '" stroke-width="2"></path>' +
    '<line id="xh" y1="' + T + '" y2="' + (H - B) + '" stroke="#8b949e" stroke-width="1" ' +
      'stroke-dasharray="2 3" style="display:none"></line>' +
    '<circle id="xhc" r="4" fill="' + (up ? "#3fb950" : "#f85149") + '" stroke="#0d1117" ' +
      'stroke-width="2" style="display:none"></circle>' +
    '<rect x="' + L + '" y="' + T + '" width="' + (W - L - R) + '" height="' + (H - T - B) +
      '" fill="transparent" id="hover"></rect>';
  const first = eq[0], last = eq[eq.length - 1];
  const chg = ((last - first) / first * 100).toFixed(2);
  $("chartd").textContent = "$" + fmt(first, 0) + " → $" + fmt(last, 0) +
    " (" + (chg >= 0 ? "+" : "") + chg + "%) · low $" + fmt(lo, 0) + " · high $" + fmt(hi, 0) +
    " · " + points.length + " samples — hover for detail";
  const hov = svg.querySelector("#hover"), xh = svg.querySelector("#xh"),
        xhc = svg.querySelector("#xhc");
  hov.addEventListener("mousemove", ev => {
    const box = svg.getBoundingClientRect();
    const px = (ev.clientX - box.left) * (W / box.width);
    const i = Math.max(0, Math.min(points.length - 1,
      Math.round((px - L) / (W - L - R) * (points.length - 1))));
    const xi = x(i), yi = y(eq[i]);
    xh.setAttribute("x1", xi); xh.setAttribute("x2", xi); xh.style.display = "";
    xhc.setAttribute("cx", xi); xhc.setAttribute("cy", yi); xhc.style.display = "";
    const dt = new Date(points[i].t * 1000);
    const dchg = ((eq[i] - first) / first * 100).toFixed(2);
    showTip("<b>$" + fmt(eq[i], 0) + "</b> <span class='" + (dchg >= 0 ? "ok" : "crit") + "'>(" +
      (dchg >= 0 ? "+" : "") + dchg + "%)</span><br><span style='color:#8b949e'>" +
      dt.toLocaleDateString() + " " + dt.toLocaleTimeString([], {hour:"2-digit",minute:"2-digit"}) +
      "</span>", ev.clientX, ev.clientY);
  });
  hov.addEventListener("mouseleave", () => {
    xh.style.display = "none"; xhc.style.display = "none"; hideTip();
  });
}
async function refreshChart() {
  try {
    const [period, timeframe] = range.split("-");
    const h = await fetchJson("/portfolio/history?period=" + period + "&timeframe=" + timeframe, 15000);
    eqPoints = h.points; drawEquity();
  } catch (e) { $("chartd").textContent = "history unavailable: " + e.message; }
}
$("rangeBtns").querySelectorAll("button").forEach(b => {
  b.onclick = () => {
    $("rangeBtns").querySelectorAll("button").forEach(x => x.classList.remove("active"));
    b.classList.add("active"); range = b.dataset.r; refreshChart();
  };
});
addEventListener("resize", () => { drawEquity(); drawCum(); });

/* ── Sortable table helper ────────────────────────────────────────────── */
function makeTable(tableId, bodyId, cols, rows, renderRow, defaultSort) {
  const state = { key: defaultSort.key, dir: defaultSort.dir, rows };
  const thead = $(tableId).querySelector("thead");
  function head() {
    thead.innerHTML = "<tr>" + cols.map(c =>
      '<th class="' + (c.key ? "sortable" : "") + '" data-k="' + (c.key || "") + '">' + c.label +
      (c.key === state.key ? ' <span class="arrow">' + (state.dir > 0 ? "▲" : "▼") + "</span>" : "") +
      "</th>").join("") + "</tr>";
    thead.querySelectorAll("th.sortable").forEach(th => th.onclick = () => {
      const k = th.dataset.k;
      state.dir = (state.key === k) ? -state.dir : -1;
      state.key = k; render();
    });
  }
  function render() {
    head();
    const sorted = [...state.rows].sort((a, b) => {
      const va = a[state.key], vb = b[state.key];
      return (va < vb ? -1 : va > vb ? 1 : 0) * state.dir;
    });
    $(bodyId).innerHTML = sorted.map(renderRow).join("") ||
      '<tr><td colspan="' + cols.length + '" class="muted">nothing to show</td></tr>';
  }
  state.render = render;
  state.setRows = r => { state.rows = r; render(); };
  render();
  return state;
}

/* ── Cumulative PnL mini-chart ────────────────────────────────────────── */
let cumSeries = null;
function drawCum() {
  const svg = $("cumchart");
  if (!cumSeries || cumSeries.length < 2) { svg.innerHTML = ""; return; }
  const W = svg.clientWidth || 500, H = 140, L = 46, R = 6, T = 8, B = 6;
  svg.setAttribute("viewBox", "0 0 " + W + " " + H);
  const vals = cumSeries.map(p => p.v);
  const lo = Math.min(0, ...vals), hi = Math.max(0, ...vals), span = (hi - lo) || 1;
  const x = i => L + (W - L - R) * i / (vals.length - 1);
  const y = v => T + (H - T - B) * (1 - (v - lo) / span);
  const zero = y(0);
  const last = vals[vals.length - 1];
  const color = last >= 0 ? "var(--ok)" : "var(--crit)";
  let d = "M " + x(0) + " " + y(vals[0]);
  for (let i = 1; i < vals.length; i++) d += " L " + x(i) + " " + y(vals[i]);
  svg.innerHTML =
    '<line x1="' + L + '" x2="' + (W - R) + '" y1="' + zero + '" y2="' + zero +
      '" stroke="#30363d" stroke-width="1"></line>' +
    '<text x="' + (L - 6) + '" y="' + (zero + 3) + '" text-anchor="end">$0</text>' +
    '<text x="' + (L - 6) + '" y="' + (y(last) + 3) + '" text-anchor="end" class="' +
      (last >= 0 ? "ok" : "crit") + '">' + signed(last) + "</text>" +
    '<path d="' + d + '" fill="none" stroke="' + color + '" stroke-width="2"></path>' +
    '<rect x="' + L + '" y="' + T + '" width="' + (W - L - R) + '" height="' + (H - T - B) +
      '" fill="transparent" id="cumhov"></rect>';
  svg.querySelector("#cumhov").addEventListener("mousemove", ev => {
    const box = svg.getBoundingClientRect();
    const px = (ev.clientX - box.left) * (W / box.width);
    const i = Math.max(0, Math.min(vals.length - 1,
      Math.round((px - L) / (W - L - R) * (vals.length - 1))));
    const p = cumSeries[i];
    showTip("<b>" + signed(p.v) + "</b> cumulative<br><span style='color:#8b949e'>after " +
      esc(p.label) + "</span>", ev.clientX, ev.clientY);
  });
  svg.querySelector("#cumhov").addEventListener("mouseleave", hideTip);
}

/* ── Trades: analytics, filters, sortable table ───────────────────────── */
let allClosed = [], tradeState = null, pnlFilter = "all", reasonSel = "";
function tradeRow(t) {
  const c = t.pnl >= 0 ? "ok" : "crit";
  const pct = t.pnl_pct != null ? " (" + (t.pnl_pct >= 0 ? "+" : "") + (t.pnl_pct * 100).toFixed(2) + "%)" : "";
  return "<tr><td>" + esc(t.date) + "</td><td><b>" + esc(t.ticker) + "</b></td>" +
    "<td>$" + fmt(t.entry_price) + " → $" + fmt(t.exit_price) + "</td>" +
    "<td>" + esc(t.holdStr) + "</td>" +
    '<td class="' + c + '">' + signed(t.pnl) + pct + "</td>" +
    "<td><span class='tag'>" + esc(t.exit_reason || "?") + "</span></td>" +
    '<td class="muted">' + (t.ensemble_signal ?? 0).toFixed(2) + "</td></tr>";
}
function applyTradeFilters() {
  let rows = allClosed;
  if (pnlFilter === "win") rows = rows.filter(t => t.pnl > 0);
  if (pnlFilter === "loss") rows = rows.filter(t => t.pnl <= 0);
  if (reasonSel) rows = rows.filter(t => t.exit_reason === reasonSel);
  tradeState.setRows(rows);
  const tot = rows.reduce((s, t) => s + t.pnl, 0);
  $("tradesFoot").textContent = rows.length + " trade(s) shown · net " + signed(tot) +
    " · click a column header to sort";
}
$("tradeFilters").querySelectorAll("button").forEach(b => {
  b.onclick = () => {
    $("tradeFilters").querySelectorAll("button").forEach(x => x.classList.remove("active"));
    b.classList.add("active"); pnlFilter = b.dataset.f; applyTradeFilters();
  };
});
$("reasonFilter").onchange = e => { reasonSel = e.target.value; applyTradeFilters(); };

async function refreshTrades() {
  try {
    const td = await fetchJson("/trades?limit=200", 15000);
    const closed = (td.trades || []).filter(t => t.exit_time).map(t => {
      const hold = Math.round((new Date(t.exit_time) - new Date(t.entry_time)) / 60000);
      return { ...t,
        date: t.exit_time.slice(5, 10),
        holdMin: hold,
        holdStr: hold < 120 ? hold + "m" : hold < 2880 ? (hold / 60).toFixed(1) + "h"
                 : (hold / 1440).toFixed(1) + "d",
        exit_ts: new Date(t.exit_time).getTime(),
      };
    }).sort((a, b) => a.exit_ts - b.exit_ts);
    allClosed = closed;
    if (!closed.length) {
      $("tradesBody").innerHTML = '<tr><td colspan="7" class="muted">no closed trades yet</td></tr>';
      return;
    }
    const wins = closed.filter(t => t.pnl > 0);
    const gp = wins.reduce((s, t) => s + t.pnl, 0);
    const gl = -closed.filter(t => t.pnl < 0).reduce((s, t) => s + t.pnl, 0);
    const tot = closed.reduce((s, t) => s + t.pnl, 0);
    $("tr_wr").textContent = (wins.length / closed.length * 100).toFixed(0) + "%";
    $("tr_wrd").textContent = wins.length + " of " + closed.length + " trades";
    const pfv = gl > 0 ? gp / gl : Infinity;
    $("tr_pf").textContent = isFinite(pfv) ? pfv.toFixed(2) : "∞";
    $("tr_pf").className = "big " + (pfv >= 1 ? "ok" : "crit");
    $("tr_exp").textContent = signed(tot / closed.length);
    $("tr_exp").className = "big " + (tot >= 0 ? "ok" : "crit");
    $("tr_pnl").textContent = signed(tot);
    $("tr_pnl").className = "big " + (tot >= 0 ? "ok" : "crit");
    $("tr_pnld").textContent = "all " + closed.length + " recorded closed trades";
    const avgHold = closed.reduce((s, t) => s + t.holdMin, 0) / closed.length;
    $("tr_hold").textContent = avgHold < 120 ? avgHold.toFixed(0) + "m" : (avgHold / 60).toFixed(1) + "h";
    $("tr_holdd").textContent = "median " + (() => {
      const m = [...closed].sort((a, b) => a.holdMin - b.holdMin)[Math.floor(closed.length / 2)].holdMin;
      return m < 120 ? m + "m" : (m / 60).toFixed(1) + "h";
    })();

    let run = 0;
    cumSeries = closed.map(t => { run += t.pnl; return { v: run, label: t.ticker + " " + t.date }; });
    drawCum();

    const byReason = {};
    closed.forEach(t => {
      const r = t.exit_reason || "unknown";
      (byReason[r] = byReason[r] || { n: 0, pnl: 0 });
      byReason[r].n++; byReason[r].pnl += t.pnl;
    });
    const maxN = Math.max(...Object.values(byReason).map(v => v.n));
    $("exitReasons").innerHTML = Object.entries(byReason)
      .sort((a, b) => b[1].n - a[1].n)
      .map(([r, v]) =>
        '<div class="checkrow"><span style="min-width:130px">' + esc(r) + "</span>" +
        '<span class="bar" style="width:34%"><i class="' + (v.pnl < 0 ? "crit" : "") +
        '" style="width:' + (v.n / maxN * 100).toFixed(0) + '%"></i></span>' +
        '<span style="min-width:120px;text-align:right">' + v.n + " · " +
        '<span class="' + (v.pnl >= 0 ? "ok" : "crit") + '">' + signed(v.pnl) + "</span></span></div>"
      ).join("");

    const reasons = Object.keys(byReason).sort();
    const sel = $("reasonFilter"), cur = sel.value;
    sel.innerHTML = '<option value="">every exit reason</option>' +
      reasons.map(r => '<option' + (r === cur ? " selected" : "") + ">" + esc(r) + "</option>").join("");

    if (!tradeState) {
      tradeState = makeTable("tradeTable", "tradesBody", [
        { label: "Exit date", key: "exit_ts" }, { label: "Ticker", key: "ticker" },
        { label: "Entry → Exit" }, { label: "Hold", key: "holdMin" },
        { label: "PnL", key: "pnl" }, { label: "Exit reason", key: "exit_reason" },
        { label: "Ensemble", key: "ensemble_signal" },
      ], closed, tradeRow, { key: "exit_ts", dir: -1 });
    }
    applyTradeFilters();
  } catch (e) {
    $("tradesBody").innerHTML = '<tr><td colspan="7" class="muted">unavailable: ' + esc(e.message) + "</td></tr>";
  }
}

/* ── GitHub drift check ───────────────────────────────────────────────── */
async function ghMainVersion() {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), 10000);
  try {
    const r = await fetch(
      "https://raw.githubusercontent.com/Mithunprom/stockbot/main/main.py",
      { signal: ctl.signal, cache: "no-store" });
    if (!r.ok) return null;
    const m = (await r.text()).match(/^APP_VERSION = "([0-9.]+)"/m);
    return m ? m[1] : null;
  } catch (e) { return null; }
  finally { clearTimeout(timer); }
}

/* ── Main refresh ─────────────────────────────────────────────────────── */
function renderChecks(el, checks) {
  $(el).innerHTML = (checks || []).map(c =>
    '<div class="checkrow"><span style="display:flex;align-items:center">' +
    '<span class="dot ' + cls(c.status) + '"></span>' + esc(c.name) +
    (c.healed ? " 🔧" : "") + '</span><span class="d">' + esc(c.detail) + "</span></div>"
  ).join("") || "no checks yet";
}

async function refresh() {
  try {
    const [health, wd, pf] = await Promise.all([
      fetchJson("/health"), fetchJson("/watchdog"), fetchJson("/portfolio/summary"),
    ]);
    $("banner").style.display = "none";

    const status = (wd.status || "warn");
    const label = status === "ok" ? "HEALTHY" : status === "warn" ? "DEGRADED" : "BROKEN";
    $("bot").textContent = label;
    $("bot").className = "big " + cls(status);
    $("hpill").textContent = label;
    $("hpill").className = "pill " + cls(status);
    $("botd").textContent = "v" + health.version + " · loop " +
      (health.signal_loop_active ? "active" : "INACTIVE") + " · " + (health.mode || "");

    const mkt = health.market_open ?? null; marketOpen = mkt;
    const et = new Date().toLocaleTimeString("en-US", { timeZone: "America/New_York",
      hour: "2-digit", minute: "2-digit" });
    $("mktpill").textContent = (mkt === null ? "" : mkt ? "market open · " : "market closed · ") + et + " ET";
    $("mktpill").className = "pill " + (mkt ? "ok" : "");

    $("tick").textContent = ago(wd.last_tick_at);
    $("tick").className = "big " + (wd.last_tick_at &&
      (Date.now() - new Date(wd.last_tick_at)) < 5 * 60e3 ? "ok" : "warn");
    $("tickd").textContent = wd.last_exit_at ? ("last exit " + ago(wd.last_exit_at)) : "no exits yet";

    $("errors").textContent = wd.tick_error_count ?? "–";
    $("errors").className = "big " + ((wd.tick_error_count || 0) === 0 ? "ok" : "crit");
    $("errorsd").textContent = wd.last_tick_error || "no recent errors";

    let pv = pf.portfolio_value;
    try { pv = (await fetchJson("/account")).equity || pv; } catch (e) {}
    $("pv").textContent = "$" + fmt(pv, 0);
    $("pvd").textContent = "cash $" + fmt(pf.available_cash, 0) + " · live broker equity";

    const pnl = pf.daily_pnl_pct;
    $("pnl").textContent = (pnl >= 0 ? "+" : "") + pnl.toFixed(2) + "%";
    $("pnl").className = "big " + (pnl >= 0 ? "ok" : "crit");
    $("pnld").textContent = signed(pf.daily_pnl_dollar) + " today";

    const heat = pf.portfolio_heat || 0;
    $("pos").textContent = pf.n_open_positions + " · " + Math.round(heat * 100) + "%";
    $("heatbar").style.width = Math.min(100, heat / 0.75 * 100).toFixed(0) + "%";
    $("heatbar").className = heat > 0.75 ? "crit" : "";
    $("posd").textContent = " of 75% ceiling · " + pf.n_trades_today + " trades today";

    const kf = pf.kelly_fraction ?? 0;
    $("kelly").textContent = kf.toFixed(2);
    $("kelly").className = "big " + (kf > 0 ? "ok" : kf > -0.5 ? "warn" : "crit");
    $("kellyd").textContent = (pf.kelly_mode || "–") + " mode · sizes " +
      (kf > 0 ? "normal" : "throttled while expectancy is negative");

    $("halt").textContent = pf.halted ? "HALTED" : "armed";
    $("halt").className = "big " + (pf.halted ? "crit" : "ok");
    $("haltd").textContent = pf.halt_reason || "circuit breakers normal";

    renderChecks("checks", wd.checks);

    const ghv = await ghMainVersion();
    if (ghv === null) {
      $("deploy").textContent = "v" + health.version;
      $("deploy").className = "big";
      $("deployd").textContent = "GitHub unreachable — drift unknown";
    } else if (ghv === health.version) {
      $("deploy").textContent = "IN SYNC";
      $("deploy").className = "big ok";
      $("deployd").textContent = "prod v" + health.version + " == main v" + ghv;
    } else {
      $("deploy").textContent = "DRIFT";
      $("deploy").className = "big crit";
      $("deployd").textContent = "prod v" + health.version + " ≠ main v" + ghv +
        " — auto-heals within 30 min";
    }
    $("updated").textContent = "updated " + new Date().toLocaleTimeString();
  } catch (e) {
    $("banner").style.display = "block";
    $("bot").textContent = "UNREACHABLE";
    $("bot").className = "big crit";
    $("hpill").textContent = "OFFLINE"; $("hpill").className = "pill crit";
    $("updated").textContent = new Date().toLocaleTimeString() + " (failed)";
  }

  // Independent sections — each may fail without breaking the others
  try {
    const ig = await fetchJson("/integrity", 15000);
    const bad = (ig.checks || []).filter(c => c.status === "critical" && !c.healed);
    $("integ").textContent = ig.status === "ok" ? "CLEAN" :
      bad.length ? "ISSUES" : "HEALED";
    $("integ").className = "big " + (ig.status === "ok" ? "ok" : bad.length ? "crit" : "warn");
    $("integd").textContent = ig.timestamp ? ("audited " + ago(ig.timestamp)) : "";
    renderChecks("ichecks", ig.checks);
  } catch (e) {
    $("integ").textContent = "–"; $("integd").textContent = "unavailable";
    $("ichecks").textContent = "unavailable: " + e.message;
  }

  try {
    const nr = await fetchJson("/news-risk", 15000);
    const lv = nr.level_name || "none";
    $("news").textContent = lv.toUpperCase();
    $("news").className = "big " + (lv === "none" ? "ok" : lv === "elevated" ? "warn" : "crit");
    $("newsd").textContent = nr.headlines && nr.headlines.length
      ? nr.headlines[0].title.slice(0, 80)
      : "no macro shock headlines · scans every 15 min pre-market + market hours";
  } catch (e) { $("news").textContent = "–"; $("newsd").textContent = "unavailable"; }

  try {
    const pd = await fetchJson("/positions/detail");
    const rows = (pd.positions || []).map(p => {
      const dirn = p.side === "long" ? 1 : -1;
      const move = dirn * (p.last_price - p.avg_entry_price) / p.avg_entry_price;
      return { ...p, move, upnl: p.unrealized_pnl };
    });
    if (!$("posTable").dataset.init) {
      $("posTable").dataset.init = "1";
      window.posState = makeTable("posTable", "positions", [
        { label: "Ticker", key: "ticker" }, { label: "Side" },
        { label: "Qty", key: "qty" }, { label: "Entry", key: "avg_entry_price" },
        { label: "Last", key: "last_price" }, { label: "PnL", key: "upnl" },
        { label: "Held", key: "bars_held", }, { label: "→ Take profit", key: "move" },
      ], rows, p => {
        const pnlc = p.upnl >= 0 ? "ok" : "crit";
        const prog = Math.max(0, Math.min(1, p.move / (p.take_profit_pct || 0.02)));
        return "<tr><td><b>" + esc(p.ticker) + "</b></td><td>" + esc(p.side) + "</td>" +
          "<td>" + fmt(p.qty) + "</td><td>$" + fmt(p.avg_entry_price) + "</td>" +
          "<td>$" + fmt(p.last_price) + "</td>" +
          '<td class="' + pnlc + '">' + signed(p.upnl) + " (" +
          (p.move * 100).toFixed(2) + "%)</td>" +
          '<td title="minutes-in-market counter; resets to 0 on redeploy">' + p.bars_held + "/" + p.max_hold_bars + " bars</td>" +
          '<td><span class="bar"><i class="' + (p.move < 0 ? "crit" : "") + '" style="width:' +
          (prog * 100).toFixed(0) + '%"></i></span> <span class="muted">' +
          (p.move * 100).toFixed(2) + "% of " + ((p.take_profit_pct || 0) * 100).toFixed(1) +
          "%</span></td></tr>";
      }, { key: "upnl", dir: -1 });
    } else {
      window.posState.setRows(rows);
    }
    if (!rows.length)
      $("positions").innerHTML = '<tr><td colspan="8" class="muted">no open positions</td></tr>';
  } catch (e) {
    $("positions").innerHTML = '<tr><td colspan="8" class="muted">unavailable: ' + esc(e.message) + "</td></tr>";
  }

  try {
    const diag = await fetchJson("/diagnostics", 15000);
    const gates = (diag.pipeline_a && diag.pipeline_a.signal_gate_analysis) || [];
    const rows = gates
      .filter(g => g.lgbm_pred_return > 0)
      .sort((a, b) => b.lgbm_pred_return - a.lgbm_pred_return)
      .slice(0, 8)
      .map(g => {
        const status = g.would_trade
          ? '<span class="ok">✓ ready — waiting for capacity/cooldown</span>'
          : (g.blocked_by || []).map(b => '<span class="tag">' + esc(b) + "</span>").join("") ||
            '<span class="warn">gated</span>';
        return "<tr><td><b>" + esc(g.ticker) + "</b></td>" +
          "<td>" + (g.lgbm_pred_return * 100).toFixed(2) + "%</td>" +
          "<td>" + (g.lgbm_dir_prob * 100).toFixed(0) + "%</td>" +
          "<td>" + (g.ensemble_signal ?? 0).toFixed(2) + "</td>" +
          "<td>" + status + "</td></tr>";
      }).join("");
    $("watchlist").innerHTML = rows ||
      '<tr><td colspan="5" class="muted">' + (marketOpen === false
        ? "market closed — signal analysis repopulates within a minute of the 09:30 ET open"
        : "no bullish signals right now") + "</td></tr>";
  } catch (e) {
    $("watchlist").innerHTML = '<tr><td colspan="5" class="muted">unavailable: ' + esc(e.message) + "</td></tr>";
  }

  refreshTrades();
}

/* ── Refresh loop: countdown, pause when hidden, manual refresh ───────── */
let marketOpen = null;
let nextIn = 30;
function tickCountdown() {
  if (document.hidden) return;
  nextIn -= 1;
  if (nextIn <= 0) { nextIn = 30; refresh(); }
  $("countdown").textContent = nextIn + "s";
}
$("refreshNow").onclick = () => { nextIn = 30; refresh(); refreshChart(); };
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) { nextIn = 30; refresh(); refreshChart(); }
});
refresh();
refreshChart();
setInterval(tickCountdown, 1000);
setInterval(() => { if (!document.hidden) refreshChart(); }, 120000);
</script>
</body>
</html>
"""
