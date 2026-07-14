"""Self-contained HTML status dashboard, served at GET /dashboard.

Inline string (not a static file) so .railwayignore can never exclude it from
the deploy image. Zero build step, zero CDN dependencies, free.

Resilience contract ("never fails to refresh"):
- polls every 30s with AbortController timeouts — a hung request can't wedge
  the refresh loop; every section fails independently
- failures flip a visible red OFFLINE banner and keep retrying forever
- deploy-drift check compares the running version against APP_VERSION in
  main.py on GitHub raw (CORS-open)

Sections: health cards, equity time-series (hand-rolled SVG — no chart lib),
open positions table with take-profit progress, and a "promising / needs
time" watchlist derived from live signal gate analysis.
"""

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>StockBot Status</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
         background: #0d1117; color: #e6edf3; padding: 20px; max-width: 1200px;
         margin: 0 auto; }
  h1 { font-size: 20px; margin-bottom: 4px; }
  h2 { font-size: 14px; color: #8b949e; margin: 22px 0 10px;
       text-transform: uppercase; letter-spacing: .5px; }
  .sub { color: #8b949e; font-size: 12px; margin-bottom: 16px; }
  #banner { display: none; background: #da3633; color: #fff; padding: 10px 14px;
            border-radius: 8px; margin-bottom: 14px; font-weight: 600; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
          gap: 12px; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 10px;
          padding: 14px; }
  .card h3 { font-size: 11px; text-transform: uppercase; letter-spacing: .5px;
             color: #8b949e; margin-bottom: 8px; }
  .big { font-size: 21px; font-weight: 700; }
  .ok { color: #3fb950; } .warn { color: #d29922; } .crit { color: #f85149; }
  .muted { color: #8b949e; font-size: 12px; margin-top: 6px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; color: #8b949e; font-weight: 500; padding: 6px 8px;
       border-bottom: 1px solid #30363d; }
  td { padding: 7px 8px; border-bottom: 1px solid #21262d; }
  .panel { background: #161b22; border: 1px solid #30363d; border-radius: 10px;
           padding: 14px; }
  .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%;
         margin-right: 6px; }
  .dot.ok { background: #3fb950; } .dot.warn { background: #d29922; }
  .dot.crit { background: #f85149; }
  #checks .row { display: flex; justify-content: space-between; gap: 8px;
                 padding: 6px 0; border-bottom: 1px solid #21262d; font-size: 13px; }
  #checks .row:last-child { border-bottom: none; }
  .bar { background: #21262d; border-radius: 4px; height: 8px; width: 100px;
         display: inline-block; vertical-align: middle; }
  .bar > i { display: block; height: 100%; border-radius: 4px; background: #3fb950; }
  .range-btns button { background: #21262d; color: #8b949e; border: 1px solid #30363d;
       border-radius: 6px; padding: 3px 10px; font-size: 12px; cursor: pointer; }
  .range-btns button.active { color: #e6edf3; border-color: #58a6ff; }
  a { color: #58a6ff; }
</style>
</head>
<body>
<h1>🤖 StockBot Status</h1>
<div class="sub">auto-refreshes every 30s &middot; last update: <span id="updated">–</span></div>
<div id="banner">⚠️ CANNOT REACH BOT — retrying every 30s…</div>

<div class="grid">
  <div class="card"><h3>Bot</h3><div class="big" id="bot">–</div><div class="muted" id="botd"></div></div>
  <div class="card"><h3>Deployment</h3><div class="big" id="deploy">–</div><div class="muted" id="deployd"></div></div>
  <div class="card"><h3>Portfolio</h3><div class="big" id="pv">–</div><div class="muted" id="pvd"></div></div>
  <div class="card"><h3>Daily PnL</h3><div class="big" id="pnl">–</div><div class="muted" id="pnld"></div></div>
  <div class="card"><h3>Positions / Heat</h3><div class="big" id="pos">–</div><div class="muted" id="posd"></div></div>
  <div class="card"><h3>Trading Halt</h3><div class="big" id="halt">–</div><div class="muted" id="haltd"></div></div>
  <div class="card"><h3>Last Loop Tick</h3><div class="big" id="tick">–</div><div class="muted" id="tickd"></div></div>
  <div class="card"><h3>Tick Errors</h3><div class="big" id="errors">–</div><div class="muted" id="errorsd"></div></div>
</div>

<h2>Equity curve
  <span class="range-btns" style="float:right">
    <button data-r="1D-15Min">1D</button>
    <button data-r="1W-1H">1W</button>
    <button data-r="1M-1D" class="active">1M</button>
    <button data-r="3M-1D">3M</button>
  </span>
</h2>
<div class="panel"><svg id="chart" width="100%" height="220"
     preserveAspectRatio="none"></svg>
  <div class="muted" id="chartd">loading…</div></div>

<h2>Open positions</h2>
<div class="panel" style="overflow-x:auto">
  <table><thead><tr>
    <th>Ticker</th><th>Side</th><th>Qty</th><th>Entry</th><th>Last</th>
    <th>PnL</th><th>Held</th><th>→ Take profit</th>
  </tr></thead><tbody id="positions"><tr><td colspan="8">loading…</td></tr></tbody></table>
</div>

<h2>Promising — needs a bit more time</h2>
<div class="panel" style="overflow-x:auto">
  <table><thead><tr>
    <th>Ticker</th><th>Pred return</th><th>P(up)</th><th>Ensemble</th><th>Status</th>
  </tr></thead><tbody id="watchlist"><tr><td colspan="5">loading…</td></tr></tbody></table>
</div>

<h2>Track record — recent closed trades</h2>
<div class="grid" style="margin-bottom:12px">
  <div class="card"><h3>Win rate</h3><div class="big" id="tr_wr">–</div><div class="muted" id="tr_wrd"></div></div>
  <div class="card"><h3>Profit factor</h3><div class="big" id="tr_pf">–</div><div class="muted">gross win / gross loss</div></div>
  <div class="card"><h3>Expectancy</h3><div class="big" id="tr_exp">–</div><div class="muted">avg PnL per trade</div></div>
  <div class="card"><h3>Total PnL</h3><div class="big" id="tr_pnl">–</div><div class="muted" id="tr_pnld"></div></div>
</div>
<div class="panel" style="overflow-x:auto">
  <table><thead><tr>
    <th>Date</th><th>Ticker</th><th>Entry → Exit</th><th>Hold</th>
    <th>PnL</th><th>Exit reason</th><th>Signal at entry</th>
  </tr></thead><tbody id="trades"><tr><td colspan="7">loading…</td></tr></tbody></table>
</div>

<h2>The strategy (what you're watching)</h2>
<div class="panel muted" style="font-size:13px; line-height:1.6">
  <b style="color:#e6edf3">Signal:</b> a LightGBM model predicts next-day returns
  from 40 engineered features (momentum, VWAP, options flow, volume-informed
  trading) refreshed every minute. Entries require top-8% conviction
  (self-calibrating threshold), P(up) ≥ 60%, and a positive live information
  coefficient on that specific ticker.<br>
  <b style="color:#e6edf3">Sizing:</b> conviction-proportional, volatility-normalized,
  Kelly-governed; max 15% per position, 6 positions, 75% portfolio heat ceiling,
  max 2 per sector.<br>
  <b style="color:#e6edf3">Exits:</b> daily-volatility-scaled stop (~1.1σ),
  trailing (~1.2σ), target (~3σ), 1-trading-day max hold; reversal exit only on
  a confirmed tradeable opposite signal sustained 45 minutes.<br>
  <b style="color:#e6edf3">Risk rails:</b> daily-loss halt, drawdown halt,
  25% position breaker, sector caps — never overridden by the model.<br>
  <span style="color:#d29922">⚠ Paper trading. Not investment advice. Past
  performance does not guarantee future results.</span>
</div>

<h2>Watchdog checks</h2>
<div class="panel"><div id="checks">loading…</div></div>

<div class="sub" style="margin-top:14px">
  Raw: <a href="/watchdog">/watchdog</a> &middot; <a href="/diagnostics">/diagnostics</a>
  &middot; <a href="/portfolio/summary">/portfolio/summary</a>
  &middot; <a href="/portfolio/history">/portfolio/history</a>
</div>

<script>
const $ = id => document.getElementById(id);
const cls = s => s === "ok" ? "ok" : (s === "warn" ? "warn" : "crit");
let range = "1M-1D";

async function fetchJson(url, ms = 10000) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), ms);
  try {
    const r = await fetch(url, { signal: ctl.signal, cache: "no-store" });
    if (!r.ok) throw new Error(url + " -> " + r.status);
    return await r.json();
  } finally { clearTimeout(timer); }
}

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

function ago(iso) {
  if (!iso) return "never";
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 90) return Math.round(s) + "s ago";
  if (s < 5400) return Math.round(s / 60) + "m ago";
  return (s / 3600).toFixed(1) + "h ago";
}

function drawChart(points) {
  const svg = $("chart");
  const W = svg.clientWidth || 800, H = 220, PAD = 6;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  if (!points || points.length < 2) {
    svg.innerHTML = ""; $("chartd").textContent = "not enough history yet"; return;
  }
  const eq = points.map(p => p.equity);
  const lo = Math.min(...eq), hi = Math.max(...eq), span = (hi - lo) || 1;
  const x = i => PAD + (W - 2 * PAD) * i / (points.length - 1);
  const y = v => H - PAD - (H - 2 * PAD) * (v - lo) / span;
  const up = eq[eq.length - 1] >= eq[0];
  const color = up ? "#3fb950" : "#f85149";
  let d = `M ${x(0)} ${y(eq[0])}`;
  for (let i = 1; i < eq.length; i++) d += ` L ${x(i)} ${y(eq[i])}`;
  const area = d + ` L ${x(eq.length - 1)} ${H - PAD} L ${x(0)} ${H - PAD} Z`;
  svg.innerHTML =
    `<path d="${area}" fill="${color}" opacity="0.12"></path>` +
    `<path d="${d}" fill="none" stroke="${color}" stroke-width="2"></path>`;
  const first = eq[0], last = eq[eq.length - 1];
  const chg = ((last - first) / first * 100).toFixed(2);
  const t0 = new Date(points[0].t * 1000).toLocaleDateString();
  const t1 = new Date(points[points.length - 1].t * 1000).toLocaleDateString();
  $("chartd").textContent =
    `${t0} → ${t1} · $${first.toLocaleString()} → $${last.toLocaleString()} (${chg >= 0 ? "+" : ""}${chg}%) · low $${lo.toLocaleString()} · high $${hi.toLocaleString()}`;
}

async function refreshChart() {
  try {
    const [period, timeframe] = range.split("-");
    const h = await fetchJson(`/portfolio/history?period=${period}&timeframe=${timeframe}`, 15000);
    drawChart(h.points);
  } catch (e) { $("chartd").textContent = "history unavailable: " + e.message; }
}

document.querySelectorAll(".range-btns button").forEach(b => {
  b.onclick = () => {
    document.querySelectorAll(".range-btns button").forEach(x => x.classList.remove("active"));
    b.classList.add("active");
    range = b.dataset.r;
    refreshChart();
  };
});

function fmt(n, d = 2) { return Number(n).toLocaleString(undefined, { maximumFractionDigits: d }); }

async function refresh() {
  try {
    const [health, wd, pf] = await Promise.all([
      fetchJson("/health"), fetchJson("/watchdog"), fetchJson("/portfolio/summary"),
    ]);
    $("banner").style.display = "none";

    const status = (wd.status || "warn");
    $("bot").textContent = status === "ok" ? "HEALTHY" :
                           status === "warn" ? "DEGRADED" : "BROKEN";
    $("bot").className = "big " + cls(status);
    $("botd").textContent = "v" + health.version + " · loop " +
                            (health.signal_loop_active ? "active" : "INACTIVE");

    $("tick").textContent = ago(wd.last_tick_at);
    $("tick").className = "big " + (wd.last_tick_at &&
      (Date.now() - new Date(wd.last_tick_at)) < 5 * 60e3 ? "ok" : "warn");
    $("tickd").textContent = wd.last_exit_at ? ("last exit " + ago(wd.last_exit_at)) : "no exits yet";

    $("errors").textContent = wd.tick_error_count ?? "–";
    $("errors").className = "big " + ((wd.tick_error_count || 0) === 0 ? "ok" : "crit");
    $("errorsd").textContent = wd.last_tick_error || "";

    // Prefer LIVE broker equity — the internal mark only refreshes during
    // market hours and drifts after the close (owner saw a $440 gap vs Alpaca)
    let pv = pf.portfolio_value;
    try { pv = (await fetchJson("/account")).equity || pv; } catch (e) {}
    $("pv").textContent = "$" + fmt(pv, 0);
    $("pvd").textContent = "cash $" + fmt(pf.available_cash, 0);

    const pnl = pf.daily_pnl_pct;
    $("pnl").textContent = (pnl >= 0 ? "+" : "") + pnl.toFixed(2) + "%";
    $("pnl").className = "big " + (pnl >= 0 ? "ok" : "crit");
    $("pnld").textContent = "$" + fmt(pf.daily_pnl_dollar, 0) + " today";

    $("pos").textContent = pf.n_open_positions + " / " + Math.round(pf.portfolio_heat * 100) + "%";
    $("posd").textContent = "trades today " + pf.n_trades_today +
      " · kelly " + (pf.kelly_fraction ?? 0).toFixed(2) + " " + (pf.kelly_mode || "");

    $("halt").textContent = pf.halted ? "HALTED" : "armed";
    $("halt").className = "big " + (pf.halted ? "crit" : "ok");
    $("haltd").textContent = pf.halt_reason || "circuit breakers normal";

    $("checks").innerHTML = (wd.checks || []).map(c =>
      `<div class="row"><span><span class="dot ${cls(c.status)}"></span>${c.name}` +
      (c.healed ? " 🔧" : "") + `</span><span style="color:#8b949e">${c.detail}</span></div>`
    ).join("") || "no checks yet";

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
        " — deployment not moved to prod";
    }
    $("updated").textContent = new Date().toLocaleTimeString();
  } catch (e) {
    $("banner").style.display = "block";
    $("bot").textContent = "UNREACHABLE";
    $("bot").className = "big crit";
    $("updated").textContent = new Date().toLocaleTimeString() + " (failed)";
  }

  // Independent sections — each may fail without breaking the others
  try {
    const pd = await fetchJson("/positions/detail");
    const rows = (pd.positions || []).map(p => {
      const pnlc = p.unrealized_pnl >= 0 ? "ok" : "crit";
      // progress toward take profit (longs: price up; shorts: price down)
      const dirn = p.side === "long" ? 1 : -1;
      const move = dirn * (p.last_price - p.avg_entry_price) / p.avg_entry_price;
      const prog = Math.max(0, Math.min(1, move / (p.take_profit_pct || 0.02)));
      return `<tr><td><b>${p.ticker}</b></td><td>${p.side}</td>
        <td>${fmt(p.qty)}</td><td>$${fmt(p.avg_entry_price)}</td>
        <td>$${fmt(p.last_price)}</td>
        <td class="${pnlc}">${p.unrealized_pnl >= 0 ? "+" : ""}$${fmt(p.unrealized_pnl, 0)}</td>
        <td>${p.bars_held}/${p.max_hold_bars} bars</td>
        <td><span class="bar"><i style="width:${(prog * 100).toFixed(0)}%"></i></span>
            <span class="muted">${(move * 100).toFixed(2)}% of ${(p.take_profit_pct * 100).toFixed(1)}%</span></td>
      </tr>`;
    }).join("");
    $("positions").innerHTML = rows || `<tr><td colspan="8" class="muted">no open positions</td></tr>`;
  } catch (e) {
    $("positions").innerHTML = `<tr><td colspan="8" class="muted">unavailable: ${e.message}</td></tr>`;
  }

  try {
    const td = await fetchJson("/trades?limit=50", 15000);
    const closed = (td.trades || []).filter(t => t.exit_time);
    if (closed.length) {
      const wins = closed.filter(t => t.pnl > 0);
      const gp = wins.reduce((s, t) => s + t.pnl, 0);
      const gl = -closed.filter(t => t.pnl < 0).reduce((s, t) => s + t.pnl, 0);
      const tot = closed.reduce((s, t) => s + t.pnl, 0);
      $("tr_wr").textContent = (wins.length / closed.length * 100).toFixed(0) + "%";
      $("tr_wrd").textContent = wins.length + " of " + closed.length + " trades";
      const pf = gl > 0 ? (gp / gl) : Infinity;
      $("tr_pf").textContent = isFinite(pf) ? pf.toFixed(2) : "∞";
      $("tr_pf").className = "big " + (pf >= 1 ? "ok" : "crit");
      $("tr_exp").textContent = (tot / closed.length >= 0 ? "+$" : "-$") +
        Math.abs(tot / closed.length).toFixed(0);
      $("tr_pnl").textContent = (tot >= 0 ? "+$" : "-$") + Math.abs(tot).toFixed(0);
      $("tr_pnl").className = "big " + (tot >= 0 ? "ok" : "crit");
      $("tr_pnld").textContent = "last " + closed.length + " closed trades";
      $("trades").innerHTML = closed.slice(0, 15).map(t => {
        const hold = Math.round((new Date(t.exit_time) - new Date(t.entry_time)) / 60000);
        const holdStr = hold < 120 ? hold + "m" : (hold / 60).toFixed(1) + "h";
        const c = t.pnl >= 0 ? "ok" : "crit";
        return `<tr><td>${t.entry_time.slice(5, 10)}</td><td><b>${t.ticker}</b></td>
          <td>$${fmt(t.entry_price)} → $${fmt(t.exit_price)}</td><td>${holdStr}</td>
          <td class="${c}">${t.pnl >= 0 ? "+" : ""}$${fmt(t.pnl, 0)}</td>
          <td>${t.exit_reason || ""}</td>
          <td class="muted">ens ${(t.ensemble_signal ?? 0).toFixed(2)}${t.sentiment_index ? " · sent " + t.sentiment_index.toFixed(2) : ""}</td></tr>`;
      }).join("");
    } else {
      $("trades").innerHTML = `<tr><td colspan="7" class="muted">no closed trades yet</td></tr>`;
    }
  } catch (e) {
    $("trades").innerHTML = `<tr><td colspan="7" class="muted">unavailable: ${e.message}</td></tr>`;
  }

  try {
    const diag = await fetchJson("/diagnostics", 15000);
    const gates = (diag.pipeline_a && diag.pipeline_a.signal_gate_analysis) || [];
    const rows = gates
      .filter(g => g.lgbm_pred_return > 0)
      .sort((a, b) => b.lgbm_pred_return - a.lgbm_pred_return)
      .slice(0, 8)
      .map(g => {
        const ready = g.would_trade;
        const status = ready
          ? `<span class="ok">✓ ready — waiting for capacity/cooldown</span>`
          : `<span class="warn">${(g.blocked_by || []).join(", ") || "gated"}</span>`;
        return `<tr><td><b>${g.ticker}</b></td>
          <td>${(g.lgbm_pred_return * 100).toFixed(2)}%</td>
          <td>${(g.lgbm_dir_prob * 100).toFixed(0)}%</td>
          <td>${(g.ensemble_signal ?? 0).toFixed(2)}</td>
          <td>${status}</td></tr>`;
      }).join("");
    $("watchlist").innerHTML = rows ||
      `<tr><td colspan="5" class="muted">no bullish signals right now</td></tr>`;
  } catch (e) {
    $("watchlist").innerHTML = `<tr><td colspan="5" class="muted">unavailable: ${e.message}</td></tr>`;
  }
}

refresh();
refreshChart();
setInterval(refresh, 30000);
setInterval(refreshChart, 120000);
</script>
</body>
</html>
"""
