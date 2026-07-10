"""Self-contained HTML status dashboard, served at GET /dashboard.

Inline string (not a static file) so .railwayignore can never exclude it from
the deploy image. Zero build step, zero dependencies, free.

Resilience contract ("never fails to refresh"):
- polls every 30s with an AbortController timeout — a hung request can't
  wedge the refresh loop
- failures flip a visible red OFFLINE banner and keep retrying forever
- deploy-drift check compares the running version against main.py on GitHub
  raw (CORS-open), so "code pushed but prod not updated" is visible at a
  glance
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
         background: #0d1117; color: #e6edf3; padding: 20px; }
  h1 { font-size: 20px; margin-bottom: 4px; }
  .sub { color: #8b949e; font-size: 12px; margin-bottom: 16px; }
  #banner { display: none; background: #da3633; color: #fff; padding: 10px 14px;
            border-radius: 8px; margin-bottom: 14px; font-weight: 600; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
          gap: 12px; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 10px;
          padding: 14px; }
  .card h3 { font-size: 11px; text-transform: uppercase; letter-spacing: .5px;
             color: #8b949e; margin-bottom: 8px; }
  .big { font-size: 22px; font-weight: 700; }
  .ok    { color: #3fb950; }
  .warn  { color: #d29922; }
  .crit  { color: #f85149; }
  .muted { color: #8b949e; font-size: 12px; margin-top: 6px; }
  #checks .row { display: flex; justify-content: space-between; gap: 8px;
                 padding: 6px 0; border-bottom: 1px solid #21262d; font-size: 13px; }
  #checks .row:last-child { border-bottom: none; }
  .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%;
         margin-right: 6px; }
  .dot.ok { background: #3fb950; } .dot.warn { background: #d29922; }
  .dot.crit { background: #f85149; }
  .wide { grid-column: 1 / -1; }
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
  <div class="card"><h3>Last Loop Tick</h3><div class="big" id="tick">–</div><div class="muted" id="tickd"></div></div>
  <div class="card"><h3>Tick Errors</h3><div class="big" id="errors">–</div><div class="muted" id="errorsd"></div></div>
  <div class="card"><h3>Portfolio</h3><div class="big" id="pv">–</div><div class="muted" id="pvd"></div></div>
  <div class="card"><h3>Daily PnL</h3><div class="big" id="pnl">–</div><div class="muted" id="pnld"></div></div>
  <div class="card"><h3>Positions / Heat</h3><div class="big" id="pos">–</div><div class="muted" id="posd"></div></div>
  <div class="card"><h3>Trading Halt</h3><div class="big" id="halt">–</div><div class="muted" id="haltd"></div></div>
  <div class="card wide"><h3>Watchdog Checks</h3><div id="checks">loading…</div></div>
</div>
<div class="sub" style="margin-top:14px">
  Raw: <a href="/watchdog">/watchdog</a> &middot; <a href="/diagnostics">/diagnostics</a>
  &middot; <a href="/portfolio/summary">/portfolio/summary</a>
</div>

<script>
const $ = id => document.getElementById(id);
const cls = s => s === "ok" ? "ok" : (s === "warn" ? "warn" : "crit");

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
    const text = await r.text();
    const m = text.match(/^APP_VERSION = "([0-9.]+)"/m);
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

    $("pv").textContent = "$" + Number(pf.portfolio_value).toLocaleString(
      undefined, { maximumFractionDigits: 0 });
    $("pvd").textContent = "cash $" + Number(pf.available_cash).toLocaleString(
      undefined, { maximumFractionDigits: 0 });

    const pnl = pf.daily_pnl_pct;
    $("pnl").textContent = (pnl >= 0 ? "+" : "") + pnl.toFixed(2) + "%";
    $("pnl").className = "big " + (pnl >= 0 ? "ok" : "crit");
    $("pnld").textContent = "$" + pf.daily_pnl_dollar.toFixed(0) + " today";

    $("pos").textContent = pf.n_open_positions + " / " +
      Math.round(pf.portfolio_heat * 100) + "%";
    $("posd").textContent = "trades today " + pf.n_trades_today +
      " · kelly " + (pf.kelly_fraction ?? 0).toFixed(2) + " " + (pf.kelly_mode || "");

    $("halt").textContent = pf.halted ? "HALTED" : "armed";
    $("halt").className = "big " + (pf.halted ? "crit" : "ok");
    $("haltd").textContent = pf.halt_reason || "circuit breakers normal";

    const rows = (wd.checks || []).map(c =>
      `<div class="row"><span><span class="dot ${cls(c.status)}"></span>${c.name}` +
      (c.healed ? " 🔧" : "") + `</span><span style="color:#8b949e">${c.detail}</span></div>`
    ).join("");
    $("checks").innerHTML = rows || "no checks yet";

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
}

refresh();
setInterval(refresh, 30000);
</script>
</body>
</html>
"""
