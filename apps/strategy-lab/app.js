import { SEARCH, SEEDS } from "./data.js";
import { runGraph, DEFAULT_ROW, SCENARIOS } from "./engine.js";

const $ = (sel) => document.querySelector(sel);

const state = {
  tab: "overview",
  seedId: "alpaca_five",
  scenario: "trend",
  row: { ...DEFAULT_ROW },
  ctx: { ...SCENARIOS.trend.ctx },
};

function fmtPct(v, digits = 2) {
  if (v == null || Number.isNaN(v)) return "—";
  const n = Number(v);
  const sign = n > 0 ? "+" : "";
  return `${sign}${n.toFixed(digits)}%`;
}
function fmtN(v, digits = 2) {
  if (v == null || Number.isNaN(v)) return "—";
  return Number(v).toFixed(digits);
}
function clsNum(v) {
  if (v == null) return "";
  return v > 0 ? "pos" : v < 0 ? "neg" : "";
}

function seed() {
  return SEEDS.find((s) => s.id === state.seedId) || SEEDS[0];
}

function nav() {
  const tabs = [
    ["overview", "Overview"],
    ["graph", "How the graph works"],
    ["play", "Live decision"],
    ["results", "Walk-forward results"],
    ["safety", "Risk controls"],
  ];
  return `<nav class="tabs">${tabs
    .map(([id, label]) => `<button data-tab="${id}" class="${state.tab === id ? "active" : ""}">${label}</button>`)
    .join("")}</nav>`;
}

function overview() {
  const w = SEARCH.winner;
  return `
    <div class="warn-box">
      Paper / research only. This is not a live trading console, does not place orders,
      and the search winner is <strong>not</strong> ready for paper promotion
      (Sharpe far below 1.5). <code>config/live.yaml</code> is untouched.
    </div>
    <div class="kpis">
      <div class="kpi"><div class="lbl">Winner</div><div class="val">${w.spec.name}</div><div class="sub">${w.spec.source}</div></div>
      <div class="kpi"><div class="lbl">Val 2025 Sharpe</div><div class="val ${clsNum(w.validation.sharpe)}">${fmtN(w.validation.sharpe, 2)}</div><div class="sub">target ≥ 1.50</div></div>
      <div class="kpi"><div class="lbl">Val max drawdown</div><div class="val">${fmtN(w.validation.max_dd_pct, 2)}%</div><div class="sub">live cap 8%</div></div>
      <div class="kpi"><div class="lbl">Ready for paper</div><div class="val neg">No</div><div class="sub">staging proposal only</div></div>
    </div>
    <div class="grid-2">
      <div class="panel">
        <h2>What you are looking at</h2>
        <p class="lead">Five public multi-agent trading repos were fused into one StockBot-native DAG. An Opportunity Agent searches topologies, weights, and risk geometry with a walk-forward backtest, then writes a staging proposal.</p>
        <p>LLM calls stay off on this path. Each analyst is a deterministic score in [−1, +1] so thousands of graphs can be ranked without look-ahead or token cost. Features are lagged one bar; fills are at that bar’s open.</p>
        <h3>Search windows</h3>
        <table>
          <tr><th>Leg</th><th>Dates</th><th>Role</th></tr>
          <tr><td>Tune</td><td>2022-01 → 2024-12</td><td>Evolve population</td></tr>
          <tr><td>Validation</td><td>2025</td><td>Untouched rank</td></tr>
          <tr><td>Holdout</td><td>2026 YTD</td><td>Final stress</td></tr>
        </table>
        <p class="note">Universe: ${SEARCH.universe.join(", ")} plus SPY as the macro tape. ${SEARCH.n_tickers} names, ${SEARCH.generations} generations.</p>
      </div>
      <div class="panel">
        <h2>Where each repo landed</h2>
        <div class="sources">
          <div class="source"><strong>TradingAgents</strong><span>Technical, fundamental, sentiment, news → Bull/Bear debate → trader → risk.</span></div>
          <div class="source"><strong>NexusTrade</strong><span>Research judge and a five-role risk committee (median CRO).</span></div>
          <div class="source"><strong>stock_bot_langgraph</strong><span>40+ daily indicators, causal HMM regime, sequence predictor, swing stops.</span></div>
          <div class="source"><strong>AlpacaTradingAgent</strong><span>Fifth macro analyst, LONG/FLAT, 20% notional cap, 3%/8% breakers. Walk-forward winner.</span></div>
          <div class="source"><strong>PrimoAgent</strong><span>Linear Data → TA → 7-feature news NLP → portfolio manager.</span></div>
          <div class="source"><strong>StockBot hybrid</strong><span>All of the above in one graph, scored by the same composite as live KPIs.</span></div>
        </div>
      </div>
    </div>
  `;
}

function dagHtml(s) {
  const nodes = [
    ["technical", "Technical"],
    ["fundamental", "Fundamental"],
    ["sentiment", "Sentiment"],
    ["news", "News"],
    ["macro", "Macro"],
    ["quant", "Quant"],
  ];
  return `
    <div class="dag">
      <div class="dag-row">
        ${nodes
          .map(([id, label]) => {
            const on = s.analysts.includes(id);
            return `<div class="node ${on ? "on" : "off"}">${label}<small>${on ? `${Math.round((s.weights[id] || 0) * 100)}%` : "off"}</small></div>`;
          })
          .join("")}
      </div>
      <div class="arrow">↓ ${s.debate_weight > 0 ? `Bull / Bear debate (weight ${s.debate_weight})` : "no debate — weighted mean only"}</div>
      <div class="dag-row"><div class="node on">Trader<small>threshold ${s.entry_threshold} · stop ${s.stop_atr}×ATR · hold ${s.holding_days}d</small></div></div>
      <div class="arrow">↓ risk mode: ${s.risk_mode}${s.committee ? " (committee)" : ""}</div>
      <div class="dag-row"><div class="node on">Risk committee<small>heat ${Math.round(s.heat_ceiling * 100)}% · daily −3% · DD −8%${s.trend_filter ? " · SPY>200d" : ""}${s.hmm_filter ? " · skip high-vol" : ""}</small></div></div>
      <div class="arrow">↓</div>
      <div class="dag-row"><div class="node on">LONG or FLAT<small>long-only · fill next open</small></div></div>
    </div>
  `;
}

function graphTab() {
  const s = seed();
  return `
    <div class="topo-pills">
      ${SEEDS.map((x) => `<button data-seed="${x.id}" class="${x.id === s.id ? "on" : ""}">${x.label}${x.winner ? " · winner" : ""}</button>`).join("")}
    </div>
    <div class="grid-2">
      <div class="panel">
        <h2>${s.label}</h2>
        <p class="lead">${s.blurb}</p>
        <p>${s.source}</p>
        ${dagHtml(s)}
      </div>
      <div class="panel">
        <h2>Signal contract</h2>
        <ol class="steps">
          <li>Indicators and HMM/SPY overlays are computed, then shifted one bar.</li>
          <li>Each active analyst emits a score in [−1, +1] and a confidence.</li>
          <li>If debate is on, a Bull/Bear judge blends with the weighted mean.</li>
          <li>Trader maps ensemble ≥ threshold to LONG with ATR stop/target.</li>
          <li>Risk can reject (breaker, heat, regime) or resize, never disable breakers.</li>
          <li>The backtester fills at that bar’s open and can stop out the same day.</li>
        </ol>
        <p class="note">Open <strong>Live decision</strong> to drive the same nodes with sliders. Numbers update immediately — no broker, no LLM.</p>
      </div>
    </div>
  `;
}

function slider(key, label, min, max, step, value, fmt) {
  return `<label class="slider"><span>${label}</span><input type="range" data-key="${key}" min="${min}" max="${max}" step="${step}" value="${value}"><span class="n">${fmt(value)}</span></label>`;
}

function sliderFmt(key, value) {
  if (key === "hmm_regime") return Number(value) === 0 ? "trend" : Number(value) === 2 ? "high-vol" : "choppy";
  if (["spy_vs_sma200", "spy_rvol_20", "ret_5", "daily_pnl_pct", "drawdown_pct", "portfolio_heat"].includes(key)) {
    return fmtPct(Number(value) * 100, key === "spy_rvol_20" || key === "portfolio_heat" ? 0 : 1);
  }
  if (key === "seq_pred") return fmtN(value, 3);
  if (key === "vol_z") return fmtN(value, 1);
  return fmtN(value, 0);
}

function playOutHtml() {
  const s = seed();
  const out = runGraph(state.row, s, state.ctx);
  const reports = Object.values(out.reports);
  const size = out.action === "LONG" ? out.risk.sizePct : 0;
  return `
        <div class="action ${out.action}">${out.action}</div>
        <div class="kpis" style="grid-template-columns:1fr 1fr; margin-top:12px">
          <div class="kpi"><div class="lbl">Ensemble</div><div class="val ${clsNum(out.ensemble)}">${fmtN(out.ensemble, 3)}</div><div class="sub">threshold ${s.entry_threshold}</div></div>
          <div class="kpi"><div class="lbl">Size</div><div class="val">${fmtPct(size * 100, 1)}</div><div class="sub">${out.risk.decision}${out.risk.reason ? " · " + out.risk.reason : ""}</div></div>
        </div>
        <div class="panel" style="margin-top:12px">
          <h2>Debate</h2>
          <p>Judge <strong>${out.debate.judgeVerdict}</strong> · confidence ${fmtN(out.debate.confidence, 2)} · bull ${fmtN(out.debate.bullScore, 2)} vs bear ${fmtN(out.debate.bearScore, 2)}</p>
          ${out.filterReason ? `<p>Filter: ${out.filterReason}</p>` : ""}
          <p>Stop ${(out.risk.stopPct * 100).toFixed(1)}% · target ${(out.risk.targetPct * 100).toFixed(1)}% · hold ${s.holding_days}d</p>
        </div>
        <div class="panel" style="margin-top:12px">
          <h2>Analyst scores</h2>
          ${reports
            .map((r) => {
              const left = 50 + r.score * 50;
              return `<div class="bar-row"><span>${r.name}</span><div class="score-bar"><i style="left:${left}%"></i></div><span class="${clsNum(r.score)}">${fmtN(r.score, 2)}</span></div>`;
            })
            .join("")}
          <p class="note">${reports.map((r) => (r.findings[0] ? r.name + ": " + r.findings[0] : "")).filter(Boolean).join(" · ") || "No standout findings on this bar."}</p>
        </div>
  `;
}

function playTab() {
  const s = seed();
  const scen = SCENARIOS[state.scenario] || { note: "Custom features — drag sliders to see the graph react." };
  return `
    <div class="topo-pills">
      ${SEEDS.map((x) => `<button data-seed="${x.id}" class="${x.id === s.id ? "on" : ""}">${x.label}</button>`).join("")}
    </div>
    <div class="topo-pills">
      ${Object.entries(SCENARIOS)
        .map(([id, sc]) => `<button data-scen="${id}" class="${state.scenario === id ? "on" : ""}">${sc.label}</button>`)
        .join("")}
    </div>
    <p class="note" style="margin-top:0">${scen.note}</p>
    <div class="grid-2">
      <div class="panel">
        <h2>Market features (lagged)</h2>
        <div class="sliders">
          ${slider("rsi_14", "RSI 14", 15, 85, 1, state.row.rsi_14, (v) => sliderFmt("rsi_14", v))}
          ${slider("spy_vs_sma200", "SPY vs SMA200", -0.12, 0.12, 0.005, state.row.spy_vs_sma200, (v) => sliderFmt("spy_vs_sma200", v))}
          ${slider("spy_rvol_20", "SPY rvol", 0.08, 0.45, 0.01, state.row.spy_rvol_20, (v) => sliderFmt("spy_rvol_20", v))}
          ${slider("hmm_regime", "HMM regime", 0, 2, 1, state.row.hmm_regime, (v) => sliderFmt("hmm_regime", v))}
          ${slider("seq_pred", "Seq. predictor", -0.06, 0.06, 0.002, state.row.seq_pred, (v) => sliderFmt("seq_pred", v))}
          ${slider("ret_5", "5-day return", -0.12, 0.12, 0.002, state.row.ret_5, (v) => sliderFmt("ret_5", v))}
          ${slider("vol_z", "Volume z", -1, 4, 0.1, state.row.vol_z, (v) => sliderFmt("vol_z", v))}
          ${slider("adx", "ADX", 8, 45, 1, state.row.adx, (v) => sliderFmt("adx", v))}
        </div>
        <h3>Book state (risk inputs)</h3>
        <div class="sliders">
          ${slider("daily_pnl_pct", "Daily PnL", -0.05, 0.02, 0.002, state.ctx.daily_pnl_pct, (v) => sliderFmt("daily_pnl_pct", v))}
          ${slider("drawdown_pct", "Drawdown", -0.12, 0, 0.005, state.ctx.drawdown_pct, (v) => sliderFmt("drawdown_pct", v))}
          ${slider("portfolio_heat", "Portfolio heat", 0, 0.9, 0.05, state.ctx.portfolio_heat, (v) => sliderFmt("portfolio_heat", v))}
        </div>
      </div>
      <div id="play-out">${playOutHtml()}</div>
    </div>
  `;
}

function metricRow(label, m, extra = "") {
  return `<tr>
    <td>${label}${extra}</td>
    <td class="num ${clsNum(m.total_return_pct)}">${fmtPct(m.total_return_pct)}</td>
    <td class="num ${clsNum(m.sharpe)}">${fmtN(m.sharpe, 2)}</td>
    <td class="num">${fmtN(m.max_dd_pct, 2)}%</td>
    <td class="num">${fmtN(m.profit_factor, 2)}</td>
    <td class="num">${fmtPct(m.win_rate * 100, 1)}</td>
    <td class="num">${m.n_trades}</td>
  </tr>`;
}

function shortSpecName(name) {
  const parts = String(name).split("~");
  if (parts.length >= 2) return `${parts[0]}~${parts[1]}`;
  return name;
}

function resultsTab() {
  const w = SEARCH.winner;
  const bh = SEARCH.buy_hold_validation;
  const board = SEARCH.validation_leaderboard;
  const maxAbs = Math.max(...board.map((r) => Math.abs(r.metrics.total_return_pct)), 1);
  return `
    <div class="warn-box">Composite score is Sharpe-led with penalties for DD &gt; 8%, profit factor &lt; 1.4, and win rate &lt; 52%. That is why buy-and-hold loses the rank despite a higher 2025 return.</div>
    <div class="panel">
      <h2>Winner — ${w.spec.name} — three legs</h2>
      <table>
        <thead><tr><th>Window</th><th class="num">Return</th><th class="num">Sharpe</th><th class="num">Max DD</th><th class="num">PF</th><th class="num">Win rate</th><th class="num">Trades</th></tr></thead>
        <tbody>
          ${metricRow("Tune 2022–2024", w.tune)}
          ${metricRow("Validation 2025", w.validation)}
          ${metricRow("Holdout 2026 YTD", w.holdout)}
        </tbody>
      </table>
    </div>
    <div class="grid-2" style="margin-top:16px">
      <div class="panel">
        <h2>2025 validation return vs 8% DD cap</h2>
        <div class="bars">
          ${board
            .map((r) => {
              const v = r.metrics.total_return_pct;
              const wdt = Math.min(100, (Math.abs(v) / maxAbs) * 100);
              const ddFail = r.metrics.max_dd_pct > 8;
              return `<div class="bar-row"><span>${shortSpecName(r.name)}</span><div class="bar-track"><div class="bar-fill ${v < 0 ? "neg" : ddFail ? "" : "good"}" style="width:${wdt}%"></div></div><span class="${clsNum(v)}">${fmtPct(v, 1)}</span></div>
                <div class="note" style="margin:0 0 8px 128px">Sharpe ${fmtN(r.metrics.sharpe, 2)} · DD ${fmtN(r.metrics.max_dd_pct, 1)}%${ddFail ? " · fails 8% cap" : ""} · score ${fmtN(r.metrics.score, 2)}</div>`;
            })
            .join("")}
        </div>
        <p class="note">Buy-and-hold 2025: ${fmtPct(bh.total_return_pct)} return, Sharpe ${fmtN(bh.sharpe, 2)}, drawdown ${fmtN(bh.max_dd_pct, 1)}%. The live DD cap rejects it even though raw PnL wins.</p>
      </div>
      <div class="panel">
        <h2>Validation leaderboard</h2>
        <table>
          <thead><tr><th>Spec</th><th class="num">Sharpe</th><th class="num">DD</th><th class="num">Score</th></tr></thead>
          <tbody>
            ${board
              .map(
                (r, i) => `<tr class="${i === 0 ? "winner" : ""}">
                <td>${r.name}<div class="note" style="margin:0">${r.topology} · ${r.risk_mode}</div></td>
                <td class="num ${clsNum(r.metrics.sharpe)}">${fmtN(r.metrics.sharpe, 2)}</td>
                <td class="num">${fmtN(r.metrics.max_dd_pct, 1)}%</td>
                <td class="num">${fmtN(r.metrics.score, 2)}</td>
              </tr>`
              )
              .join("")}
          </tbody>
        </table>
        <h3>Tune-leg evolution</h3>
        <table>
          <thead><tr><th>Gen</th><th>Best on tune</th><th class="num">Score</th></tr></thead>
          <tbody>
            ${SEARCH.history.map((g) => `<tr><td>${g.gen}</td><td>${g.best}</td><td class="num">${fmtN(g.best_score, 3)}</td></tr>`).join("")}
          </tbody>
        </table>
        <p class="note">Tune-best ${SEARCH.history[SEARCH.history.length - 1].best} did not survive validation as well as seed alpaca_five, so the agent kept alpaca_five.</p>
      </div>
    </div>
  `;
}

function safetyTab() {
  return `
    <div class="grid-2">
      <div class="panel">
        <h2>Circuit breakers — never disabled</h2>
        <table>
          <tr><th>Control</th><th>Threshold</th><th>Effect</th></tr>
          <tr><td>Daily loss</td><td>−3%</td><td>Halt new entries for the session</td></tr>
          <tr><td>Drawdown</td><td>−8%</td><td>Flatten book, pause 21 sessions, resume at half size (max 3)</td></tr>
          <tr><td>Vol spike</td><td>SPY rvol &gt; 35%</td><td>Reject the plan</td></tr>
          <tr><td>Heat</td><td>spec ceiling (winner 80%)</td><td>No new risk</td></tr>
          <tr><td>Notional cap</td><td>20% per name</td><td>Alpaca-style size clip</td></tr>
          <tr><td>Long only</td><td>on</td><td>No shorts (paper still needs margin)</td></tr>
        </table>
        <p class="note">On the research sim, an 8% halt models a human paper resume — it does not silently re-enable live trading.</p>
      </div>
      <div class="panel">
        <h2>Paper promotion gates (not met)</h2>
        <p>PROJECT.md / CLAUDE.md: Sharpe ≥ 1.5, drawdown ≤ 8%, profit factor ≥ 1.4, win rate ≥ 52%, plus paper-trading consistency. Backtest alone is never “ready for live.”</p>
        <p>Winner val Sharpe 0.36, PF 1.12, win rate 47.5%. Holdout Sharpe is negative. Staging file is marked <code>do_not_apply_to_live</code>.</p>
        <h3>Try the halt yourself</h3>
        <p>Open Live decision → “Daily −3% halt”. Same bullish features, rejected by the breaker.</p>
      </div>
    </div>
  `;
}

function render() {
  const body =
    state.tab === "overview"
      ? overview()
      : state.tab === "graph"
        ? graphTab()
        : state.tab === "play"
          ? playTab()
          : state.tab === "results"
            ? resultsTab()
            : safetyTab();
  $("#app").innerHTML = `
    <div class="wrap">
      <header class="top">
        <div>
          <h1>Hybrid strategy lab</h1>
          <p>Review how the fused multi-agent graph decides LONG vs FLAT, and how walk-forward search ranked alpaca_five — without touching live trading.</p>
        </div>
        <span class="badge">Research · not live</span>
      </header>
      ${nav()}
      ${body}
    </div>
  `;
}

function applyScenario(id) {
  const sc = SCENARIOS[id];
  if (!sc) return;
  state.scenario = id;
  state.row = { ...sc.row };
  state.ctx = { ...sc.ctx };
}

document.addEventListener("click", (e) => {
  const tab = e.target.closest("[data-tab]");
  if (tab) {
    state.tab = tab.dataset.tab;
    render();
    return;
  }
  const seedBtn = e.target.closest("[data-seed]");
  if (seedBtn) {
    state.seedId = seedBtn.dataset.seed;
    render();
    return;
  }
  const scen = e.target.closest("[data-scen]");
  if (scen) {
    applyScenario(scen.dataset.scen);
    render();
  }
});

document.addEventListener("input", (e) => {
  const key = e.target.dataset.key;
  if (!key) return;
  const v = Number(e.target.value);
  if (key in state.ctx) state.ctx[key] = v;
  else state.row[key] = v;
  state.scenario = "custom";
  const label = e.target.parentElement.querySelector(".n");
  if (label) label.textContent = sliderFmt(key, v);
  const out = document.getElementById("play-out");
  if (out) out.innerHTML = playOutHtml();
});

render();
