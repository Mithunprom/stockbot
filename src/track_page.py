"""Client-facing track-record page, served at GET /track.

The investor view: performance, trades, and methodology — WITHOUT the ops
internals (/dashboard keeps watchdog checks, Kelly state, heat gauges, etc.
for the operator). Inline HTML: zero build step, zero CDN, immune to
.railwayignore, resilient polling with abort-timeouts.

Content rules (CLAUDE.md + compliance posture):
- performance is always shown with risk stats together, never PnL alone
- prominent paper-trading / not-investment-advice disclaimer
- no "copy these trades" call-to-action — transparency only
"""

TRACK_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>StockBot — Live Track Record</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
         background: #0a0e14; color: #e6edf3; padding: 24px;
         max-width: 1080px; margin: 0 auto; }
  header { text-align: center; margin: 18px 0 26px; }
  header h1 { font-size: 26px; letter-spacing: -.5px; }
  header p { color: #8b949e; font-size: 14px; margin-top: 6px; }
  .live { display: inline-block; background: #12261a; color: #3fb950;
          border: 1px solid #238636; border-radius: 999px; padding: 3px 12px;
          font-size: 12px; font-weight: 600; margin-top: 10px; }
  h2 { font-size: 13px; color: #8b949e; margin: 26px 0 10px;
       text-transform: uppercase; letter-spacing: .6px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
          gap: 12px; }
  .card { background: #11161d; border: 1px solid #262c36; border-radius: 12px;
          padding: 16px; text-align: center; }
  .card h3 { font-size: 11px; text-transform: uppercase; letter-spacing: .5px;
             color: #8b949e; margin-bottom: 8px; }
  .big { font-size: 24px; font-weight: 700; }
  .ok { color: #3fb950; } .bad { color: #f85149; } .muted { color: #8b949e; }
  .panel { background: #11161d; border: 1px solid #262c36; border-radius: 12px;
           padding: 16px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; color: #8b949e; font-weight: 500; padding: 6px 8px;
       border-bottom: 1px solid #262c36; }
  td { padding: 8px; border-bottom: 1px solid #1b212b; }
  .note { font-size: 12px; color: #8b949e; margin-top: 6px; }
  .disclaimer { background: #1c1710; border: 1px solid #6d5518;
                border-radius: 12px; padding: 14px 16px; color: #d29922;
                font-size: 13px; line-height: 1.6; margin-top: 26px; }
  .method { font-size: 14px; line-height: 1.7; color: #c9d1d9; }
  .method b { color: #e6edf3; }
  .range-btns { text-align: right; margin-bottom: 8px; }
  .range-btns button { background: #1b212b; color: #8b949e; border: 1px solid #262c36;
       border-radius: 6px; padding: 4px 12px; font-size: 12px; cursor: pointer;
       margin-left: 6px; }
  .range-btns button.active { color: #e6edf3; border-color: #58a6ff; }
</style>
</head>
<body>
<header>
  <h1>🤖 StockBot</h1>
  <p>Autonomous ML trading system — live paper-trading track record</p>
  <span class="live">● LIVE — updates every 60s</span>
</header>

<h2>Performance</h2>
<div class="grid">
  <div class="card"><h3>Portfolio</h3><div class="big" id="pv">–</div></div>
  <div class="card"><h3>Today</h3><div class="big" id="today">–</div></div>
  <div class="card"><h3>Win rate</h3><div class="big" id="wr">–</div><div class="note" id="wrd"></div></div>
  <div class="card"><h3>Profit factor</h3><div class="big" id="pf">–</div></div>
  <div class="card"><h3>Avg per trade</h3><div class="big" id="exp">–</div></div>
  <div class="card"><h3>Open positions</h3><div class="big" id="npos">–</div></div>
</div>

<h2>Equity curve</h2>
<div class="range-btns">
  <button data-r="1W-1H">1W</button>
  <button data-r="1M-1D" class="active">1M</button>
  <button data-r="3M-1D">3M</button>
</div>
<div class="panel"><svg id="chart" width="100%" height="240"
     preserveAspectRatio="none"></svg>
  <div class="note" id="chartd">loading…</div></div>

<h2>Recent trades — every decision, with its reasoning</h2>
<div class="panel" style="overflow-x:auto">
  <table><thead><tr>
    <th>Date</th><th>Ticker</th><th>Entry → Exit</th><th>Held</th>
    <th>Result</th><th>Why it exited</th>
  </tr></thead><tbody id="trades"><tr><td colspan="6">loading…</td></tr></tbody></table>
  <div class="note">Signal values at entry are logged for every trade —
    nothing is hidden, including the losers.</div>
</div>

<h2>How it works</h2>
<div class="panel method">
  <b>1 · Signal.</b> A gradient-boosted model (LightGBM) predicts next-day
  returns for a 20-stock universe from 40 engineered features — momentum,
  volume-informed trading, options flow, news sentiment — recomputed every
  minute of the trading day.<br><br>
  <b>2 · Discipline.</b> It only trades the top ~8% of its own conviction
  distribution, requires ≥60% directional probability, and refuses tickers
  where its live, measured accuracy isn't positive.<br><br>
  <b>3 · Sizing.</b> Positions scale with conviction and inverse volatility,
  governed by a Kelly criterion estimate — max 15% per position, 6 positions,
  75% of capital deployed at most.<br><br>
  <b>4 · Exits.</b> Volatility-scaled stop-loss, trailing stop, and profit
  target; maximum one-day hold; early exit only on a confirmed signal
  reversal.<br><br>
  <b>5 · Risk rails.</b> Daily-loss halt, drawdown halt, position-size
  breaker, and sector caps sit ABOVE the model and cannot be overridden
  by it. A self-healing watchdog monitors the whole system 24/7.
</div>

<div class="disclaimer">
  ⚠️ <b>Important.</b> This is a PAPER-TRADING account (simulated money) shown
  for transparency and research purposes. Nothing on this page is investment
  advice, an offer to manage money, or a solicitation. Past performance —
  simulated or real — does not guarantee future results. Trading involves
  substantial risk of loss.
</div>

<script>
const $ = id => document.getElementById(id);
let range = "1M-1D";

async function fetchJson(url, ms = 12000) {
  const ctl = new AbortController();
  const t = setTimeout(() => ctl.abort(), ms);
  try {
    const r = await fetch(url, { signal: ctl.signal, cache: "no-store" });
    if (!r.ok) throw new Error(r.status);
    return await r.json();
  } finally { clearTimeout(t); }
}

const fmt = (n, d = 2) => Number(n).toLocaleString(undefined, { maximumFractionDigits: d });

function drawChart(points) {
  const svg = $("chart");
  const W = svg.clientWidth || 900, H = 240, P = 8;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  if (!points || points.length < 2) { $("chartd").textContent = "not enough history yet"; return; }
  const eq = points.map(p => p.equity);
  const lo = Math.min(...eq), hi = Math.max(...eq), span = (hi - lo) || 1;
  const x = i => P + (W - 2 * P) * i / (eq.length - 1);
  const y = v => H - P - (H - 2 * P) * (v - lo) / span;
  const up = eq[eq.length - 1] >= eq[0];
  const c = up ? "#3fb950" : "#f85149";
  let d = `M ${x(0)} ${y(eq[0])}`;
  for (let i = 1; i < eq.length; i++) d += ` L ${x(i)} ${y(eq[i])}`;
  svg.innerHTML =
    `<path d="${d} L ${x(eq.length-1)} ${H-P} L ${x(0)} ${H-P} Z" fill="${c}" opacity="0.12"></path>` +
    `<path d="${d}" fill="none" stroke="${c}" stroke-width="2.5"></path>`;
  const chg = ((eq[eq.length-1] - eq[0]) / eq[0] * 100).toFixed(2);
  $("chartd").textContent =
    `${new Date(points[0].t*1000).toLocaleDateString()} → ${new Date(points[points.length-1].t*1000).toLocaleDateString()}` +
    ` · $${fmt(eq[0],0)} → $${fmt(eq[eq.length-1],0)} (${chg >= 0 ? "+" : ""}${chg}%)`;
}

async function refreshChart() {
  try {
    const [period, timeframe] = range.split("-");
    const h = await fetchJson(`/portfolio/history?period=${period}&timeframe=${timeframe}`);
    drawChart(h.points);
  } catch (e) { $("chartd").textContent = "chart unavailable"; }
}

document.querySelectorAll(".range-btns button").forEach(b => {
  b.onclick = () => {
    document.querySelectorAll(".range-btns button").forEach(x => x.classList.remove("active"));
    b.classList.add("active"); range = b.dataset.r; refreshChart();
  };
});

const REASONS = {
  take_profit: "🎯 hit profit target", stop_loss: "🛑 stop-loss",
  trailing_stop: "📉 trailing stop", max_hold: "⏱ time limit (1 day)",
  signal_reversal: "🔄 signal reversed", stagnation: "💤 stagnant",
};

async function refresh() {
  try {
    const pf = await fetchJson("/portfolio/summary");
    const dp = pf.daily_pnl_pct;
    $("today").textContent = (dp >= 0 ? "+" : "") + dp.toFixed(2) + "%";
    $("today").className = "big " + (dp >= 0 ? "ok" : "bad");
    $("npos").textContent = pf.n_open_positions;
    // Portfolio number: prefer LIVE broker equity (the bot's internal mark
    // only refreshes during market hours and drifts after the close)
    let pv = pf.portfolio_value;
    try { pv = (await fetchJson("/account")).equity || pv; } catch (e) {}
    $("pv").textContent = "$" + fmt(pv, 0);
  } catch (e) {}
  try {
    const td = await fetchJson("/trades?limit=50");
    const closed = (td.trades || []).filter(t => t.exit_time);
    if (closed.length) {
      const wins = closed.filter(t => t.pnl > 0);
      const gp = wins.reduce((s, t) => s + t.pnl, 0);
      const gl = -closed.filter(t => t.pnl < 0).reduce((s, t) => s + t.pnl, 0);
      const tot = closed.reduce((s, t) => s + t.pnl, 0);
      $("wr").textContent = (wins.length / closed.length * 100).toFixed(0) + "%";
      $("wrd").textContent = "last " + closed.length + " trades";
      const pfv = gl > 0 ? gp / gl : Infinity;
      $("pf").textContent = isFinite(pfv) ? pfv.toFixed(2) : "∞";
      $("pf").className = "big " + (pfv >= 1 ? "ok" : "bad");
      const e = tot / closed.length;
      $("exp").textContent = (e >= 0 ? "+$" : "-$") + Math.abs(e).toFixed(0);
      $("exp").className = "big " + (e >= 0 ? "ok" : "bad");
      $("trades").innerHTML = closed.slice(0, 12).map(t => {
        const hold = Math.round((new Date(t.exit_time) - new Date(t.entry_time)) / 60000);
        const hs = hold < 120 ? hold + "m" : (hold / 60).toFixed(1) + "h";
        const cl = t.pnl >= 0 ? "ok" : "bad";
        return `<tr><td>${t.entry_time.slice(5, 10)}</td><td><b>${t.ticker}</b></td>
          <td>$${fmt(t.entry_price)} → $${fmt(t.exit_price)}</td><td>${hs}</td>
          <td class="${cl}">${t.pnl >= 0 ? "+" : ""}$${fmt(t.pnl, 0)}</td>
          <td class="muted">${REASONS[t.exit_reason] || t.exit_reason || ""}</td></tr>`;
      }).join("");
    }
  } catch (e) {
    $("trades").innerHTML = `<tr><td colspan="6" class="muted">temporarily unavailable</td></tr>`;
  }
}

refresh(); refreshChart();
setInterval(refresh, 60000);
setInterval(refreshChart, 300000);
</script>
</body>
</html>
"""
