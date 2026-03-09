/**
 * StockBot Dashboard — Phase 6
 *
 * Connects to FastAPI backend:
 *   GET /api/signals       → latest ensemble signals
 *   GET /api/health        → system health
 *   GET /api/trades        → recent trade history
 *   GET /api/reports/risk  → live risk report
 *   WS  /ws/dashboard     → real-time push updates
 */

import React, { useState, useEffect, useRef, useCallback } from 'react'
import {
  AreaChart, Area, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine,
} from 'recharts'

// ── Color palette ─────────────────────────────────────────────────────────────

const C = {
  bg:       '#050510',
  surface:  '#0a0a1a',
  panel:    '#0f0f22',
  border:   '#1a1a3a',
  accent:   '#00d4ff',
  green:    '#00ff88',
  red:      '#ff3366',
  yellow:   '#ffd700',
  purple:   '#a855f7',
  text:     '#e2e8f0',
  textDim:  '#64748b',
  textBright:'#ffffff',
}

// ── Global styles ─────────────────────────────────────────────────────────────

const S = {
  app: {
    background: C.bg, minHeight: '100vh',
    fontFamily: "'IBM Plex Mono','Courier New',monospace",
    color: C.text, display: 'flex', flexDirection: 'column',
  },
  header: {
    background: `linear-gradient(90deg,${C.surface},#0a0a2a)`,
    borderBottom: `1px solid ${C.border}`,
    padding: '0 20px', height: 52,
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    position: 'sticky', top: 0, zIndex: 100,
  },
  logo: {
    fontSize: 16, fontWeight: 700, letterSpacing: 4,
    color: C.accent, textShadow: `0 0 20px ${C.accent}`,
  },
  main: { flex: 1, padding: 16, maxWidth: 1400, margin: '0 auto', width: '100%' },
  panel: {
    background: C.panel, border: `1px solid ${C.border}`,
    borderRadius: 8, padding: 16, marginBottom: 16,
  },
  panelTitle: {
    fontSize: 9, letterSpacing: 3, color: C.textDim,
    marginBottom: 12, textTransform: 'uppercase',
  },
  table: { width: '100%', borderCollapse: 'collapse', fontSize: 11 },
  th: {
    textAlign: 'left', padding: '6px 10px',
    borderBottom: `1px solid ${C.border}`,
    color: C.textDim, fontSize: 9, letterSpacing: 1, fontWeight: 400,
  },
  td: { padding: '7px 10px', borderBottom: `1px solid ${C.border}40` },
  grid2: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 },
  grid3: { display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16 },
}

// ── Formatters ────────────────────────────────────────────────────────────────

const fmt = {
  price: v => !v ? '—' : v >= 1000 ? `$${(v / 1000).toFixed(2)}K` : `$${Number(v).toFixed(2)}`,
  pct:   v => v == null ? '—' : `${v >= 0 ? '+' : ''}${(v * 100).toFixed(2)}%`,
  sig:   v => v == null ? '—' : v.toFixed(3),
  conf:  v => v == null ? '—' : `${(v * 100).toFixed(0)}%`,
  pnl:   v => v == null ? '—' : `${v >= 0 ? '+' : ''}$${Math.abs(v).toFixed(2)}`,
  ts:    s => s ? new Date(s).toLocaleTimeString() : '—',
  date:  s => s ? new Date(s).toLocaleDateString() : '—',
}

// ── Signal direction badge ────────────────────────────────────────────────────

function SigBadge({ signal }) {
  const s = signal >= 0.4 ? 'LONG' : signal <= -0.4 ? 'SHORT' : 'FLAT'
  const col = s === 'LONG' ? C.green : s === 'SHORT' ? C.red : C.textDim
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 3,
      fontSize: 9, fontWeight: 700, letterSpacing: 1,
      border: `1px solid ${col}`, color: col,
      background: `${col}18`,
    }}>{s}</span>
  )
}

// ── Status dot ────────────────────────────────────────────────────────────────

function Dot({ active }) {
  return (
    <span style={{
      display: 'inline-block', width: 8, height: 8, borderRadius: '50%',
      background: active ? C.green : C.red, marginRight: 6,
      boxShadow: active ? `0 0 6px ${C.green}` : 'none',
    }} />
  )
}

// ── Stat card ─────────────────────────────────────────────────────────────────

function StatCard({ label, value, color }) {
  return (
    <div style={{ ...S.panel, textAlign: 'center' }}>
      <div style={{ fontSize: 9, color: C.textDim, letterSpacing: 2, marginBottom: 8 }}>
        {label}
      </div>
      <div style={{ fontSize: 22, color: color || C.accent, fontWeight: 700 }}>
        {value}
      </div>
    </div>
  )
}

// ── Signals panel ─────────────────────────────────────────────────────────────

function SignalsPanel({ signals }) {
  if (!signals.length) {
    return (
      <div style={S.panel}>
        <div style={S.panelTitle}>Live Signals</div>
        <div style={{ color: C.textDim, fontSize: 11, padding: '20px 0', textAlign: 'center' }}>
          No signals yet — market may be closed or models warming up
        </div>
      </div>
    )
  }

  return (
    <div style={S.panel}>
      <div style={S.panelTitle}>Live Signals ({signals.length})</div>
      <table style={S.table}>
        <thead>
          <tr>
            <th style={S.th}>Ticker</th>
            <th style={S.th}>Signal</th>
            <th style={S.th}>Direction</th>
            <th style={S.th}>Transformer</th>
            <th style={S.th}>TCN</th>
            <th style={S.th}>Sentiment</th>
            <th style={S.th}>Strength</th>
            <th style={S.th}>Time</th>
          </tr>
        </thead>
        <tbody>
          {signals.map(s => (
            <tr key={s.ticker}>
              <td style={{ ...S.td, color: C.accent, fontWeight: 700 }}>{s.ticker}</td>
              <td style={{ ...S.td, color: s.ensemble_signal >= 0 ? C.green : C.red }}>
                {fmt.sig(s.ensemble_signal)}
              </td>
              <td style={S.td}><SigBadge signal={s.ensemble_signal} /></td>
              <td style={S.td}>{fmt.conf(s.transformer_confidence)}</td>
              <td style={S.td}>{fmt.conf(s.tcn_confidence)}</td>
              <td style={S.td}>{fmt.sig(s.sentiment_index)}</td>
              <td style={{ ...S.td, color: C.textDim, fontSize: 10 }}>{s.strength}</td>
              <td style={{ ...S.td, color: C.textDim, fontSize: 10 }}>
                {fmt.ts(s.timestamp)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ── Positions panel ───────────────────────────────────────────────────────────

function PositionsPanel({ positions, portfolioValue, heat, halted, haltReason }) {
  const posEntries = Object.entries(positions || {})
  return (
    <div style={S.panel}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <div style={S.panelTitle}>Positions & Portfolio</div>
        {halted && (
          <span style={{
            fontSize: 9, padding: '3px 8px', borderRadius: 3,
            background: '#ff336622', border: `1px solid ${C.red}`,
            color: C.red, letterSpacing: 1,
          }}>
            HALTED: {haltReason || 'circuit breaker'}
          </span>
        )}
      </div>

      <div style={{ display: 'flex', gap: 16, marginBottom: 16 }}>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 9, color: C.textDim, marginBottom: 4 }}>Portfolio Value</div>
          <div style={{ fontSize: 20, color: C.accent, fontWeight: 700 }}>
            {portfolioValue ? `$${portfolioValue.toLocaleString()}` : '—'}
          </div>
        </div>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 9, color: C.textDim, marginBottom: 4 }}>Portfolio Heat</div>
          <div style={{
            fontSize: 20, fontWeight: 700,
            color: (heat || 0) > 0.7 ? C.red : (heat || 0) > 0.4 ? C.yellow : C.green,
          }}>
            {heat != null ? `${(heat * 100).toFixed(1)}%` : '—'}
          </div>
        </div>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 9, color: C.textDim, marginBottom: 4 }}>Open Positions</div>
          <div style={{ fontSize: 20, color: C.text, fontWeight: 700 }}>
            {posEntries.length}
          </div>
        </div>
      </div>

      {posEntries.length > 0 ? (
        <table style={S.table}>
          <thead>
            <tr>
              <th style={S.th}>Ticker</th>
              <th style={S.th}>Side</th>
              <th style={S.th}>Qty</th>
              <th style={S.th}>Entry</th>
              <th style={S.th}>Current</th>
              <th style={S.th}>Unrealized PnL</th>
            </tr>
          </thead>
          <tbody>
            {posEntries.map(([ticker, pos]) => {
              const unreal = pos.unrealized_pnl
              return (
                <tr key={ticker}>
                  <td style={{ ...S.td, color: C.accent, fontWeight: 700 }}>{ticker}</td>
                  <td style={{ ...S.td, color: pos.side === 'long' ? C.green : C.red }}>
                    {pos.side?.toUpperCase()}
                  </td>
                  <td style={S.td}>{pos.qty}</td>
                  <td style={S.td}>{fmt.price(pos.entry_price)}</td>
                  <td style={S.td}>{fmt.price(pos.current_price)}</td>
                  <td style={{ ...S.td, color: (unreal || 0) >= 0 ? C.green : C.red }}>
                    {fmt.pnl(unreal)}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      ) : (
        <div style={{ color: C.textDim, fontSize: 11, textAlign: 'center', padding: '12px 0' }}>
          No open positions
        </div>
      )}
    </div>
  )
}

// ── Trades panel ──────────────────────────────────────────────────────────────

function TradesPanel({ trades }) {
  if (!trades.length) {
    return (
      <div style={S.panel}>
        <div style={S.panelTitle}>Recent Trades</div>
        <div style={{ color: C.textDim, fontSize: 11, padding: '20px 0', textAlign: 'center' }}>
          No trades yet
        </div>
      </div>
    )
  }

  return (
    <div style={S.panel}>
      <div style={S.panelTitle}>Recent Trades ({trades.length})</div>
      <table style={S.table}>
        <thead>
          <tr>
            <th style={S.th}>Ticker</th>
            <th style={S.th}>Entry</th>
            <th style={S.th}>Exit</th>
            <th style={S.th}>Shares</th>
            <th style={S.th}>Entry $</th>
            <th style={S.th}>Exit $</th>
            <th style={S.th}>PnL</th>
            <th style={S.th}>Signal</th>
            <th style={S.th}>Reason</th>
          </tr>
        </thead>
        <tbody>
          {trades.map(t => (
            <tr key={t.id}>
              <td style={{ ...S.td, color: C.accent, fontWeight: 700 }}>{t.ticker}</td>
              <td style={{ ...S.td, fontSize: 10 }}>{fmt.ts(t.entry_time)}</td>
              <td style={{ ...S.td, fontSize: 10, color: t.exit_time ? C.text : C.textDim }}>
                {t.exit_time ? fmt.ts(t.exit_time) : 'open'}
              </td>
              <td style={S.td}>{t.shares}</td>
              <td style={S.td}>{fmt.price(t.entry_price)}</td>
              <td style={S.td}>{fmt.price(t.exit_price)}</td>
              <td style={{ ...S.td, color: (t.pnl || 0) >= 0 ? C.green : C.red }}>
                {fmt.pnl(t.pnl)}
              </td>
              <td style={S.td}>{fmt.sig(t.ensemble_signal)}</td>
              <td style={{ ...S.td, fontSize: 10, color: C.textDim }}>
                {t.exit_reason || '—'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ── Risk panel ────────────────────────────────────────────────────────────────

function RiskPanel({ risk }) {
  if (!risk) {
    return (
      <div style={S.panel}>
        <div style={S.panelTitle}>Risk Metrics</div>
        <div style={{ color: C.textDim, fontSize: 11, textAlign: 'center', padding: '20px 0' }}>
          No risk report yet — Risk Agent runs every 15 min during market hours
        </div>
      </div>
    )
  }

  const fields = [
    { label: 'Portfolio Value',   val: risk.portfolio_value ? `$${risk.portfolio_value.toLocaleString()}` : '—' },
    { label: 'Daily PnL',         val: risk.daily_pnl != null ? fmt.pnl(risk.daily_pnl) : '—', colorFn: v => risk.daily_pnl >= 0 ? C.green : C.red },
    { label: 'Drawdown',          val: risk.drawdown_pct != null ? `${(risk.drawdown_pct * 100).toFixed(2)}%` : '—', colorFn: () => risk.drawdown_pct > 0.05 ? C.red : C.text },
    { label: 'VIX',               val: risk.vix != null ? risk.vix.toFixed(1) : '—', colorFn: () => risk.vix > 35 ? C.red : risk.vix > 25 ? C.yellow : C.green },
    { label: 'Consecutive Losses',val: risk.consecutive_losses ?? '—', colorFn: () => risk.consecutive_losses >= 3 ? C.red : C.text },
    { label: 'Trading Halted',    val: risk.halted ? 'YES' : 'No', colorFn: () => risk.halted ? C.red : C.green },
    { label: 'Generated',         val: risk.generated_at ? fmt.ts(risk.generated_at) : '—' },
  ]

  return (
    <div style={S.panel}>
      <div style={S.panelTitle}>Risk Metrics (last report)</div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px 24px' }}>
        {fields.map(({ label, val, colorFn }) => (
          <div key={label} style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: `1px solid ${C.border}40` }}>
            <span style={{ fontSize: 10, color: C.textDim }}>{label}</span>
            <span style={{ fontSize: 10, color: colorFn ? colorFn() : C.text, fontWeight: 700 }}>{val}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Agent status panel ────────────────────────────────────────────────────────

function AgentPanel({ health }) {
  const agents = [
    { name: 'Signal Loop',   active: health?.signal_loop_active, freq: '1 min' },
    { name: 'Risk Agent',    active: true,                       freq: '15 min' },
    { name: 'Latency Agent', active: true,                       freq: '1 hour' },
    { name: 'Profit Agent',  active: true,                       freq: 'daily 16:30' },
  ]

  return (
    <div style={S.panel}>
      <div style={S.panelTitle}>Sub-Agent Status</div>
      {agents.map(a => (
        <div key={a.name} style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          padding: '8px 0', borderBottom: `1px solid ${C.border}40`,
        }}>
          <div style={{ display: 'flex', alignItems: 'center' }}>
            <Dot active={a.active} />
            <span style={{ fontSize: 11 }}>{a.name}</span>
          </div>
          <span style={{ fontSize: 9, color: C.textDim, letterSpacing: 1 }}>
            {a.freq}
          </span>
        </div>
      ))}
    </div>
  )
}

// ── PnL mini-chart ────────────────────────────────────────────────────────────

function PnLChart({ trades }) {
  if (!trades.length) return null

  const completed = trades
    .filter(t => t.pnl != null && t.exit_time)
    .sort((a, b) => new Date(a.exit_time) - new Date(b.exit_time))

  if (!completed.length) return null

  let cum = 0
  const data = completed.map(t => {
    cum += t.pnl
    return { ts: fmt.ts(t.exit_time), cum: Math.round(cum * 100) / 100, pnl: Math.round(t.pnl * 100) / 100 }
  })

  return (
    <div style={S.panel}>
      <div style={S.panelTitle}>Cumulative PnL — {completed.length} closed trades</div>
      <ResponsiveContainer width="100%" height={160}>
        <AreaChart data={data} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="pnlGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={C.accent} stopOpacity={0.3} />
              <stop offset="95%" stopColor={C.accent} stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
          <XAxis dataKey="ts" stroke={C.textDim} tick={{ fontSize: 9 }} />
          <YAxis stroke={C.textDim} tick={{ fontSize: 9 }} tickFormatter={v => `$${v}`} />
          <Tooltip
            contentStyle={{ background: C.panel, border: `1px solid ${C.border}`, fontSize: 10 }}
            formatter={v => [`$${v}`, 'Cumulative PnL']}
          />
          <ReferenceLine y={0} stroke={C.border} />
          <Area type="monotone" dataKey="cum" stroke={C.accent} fill="url(#pnlGrad)" dot={false} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}

// ── Main App ──────────────────────────────────────────────────────────────────

export default function App() {
  const [health,         setHealth]         = useState(null)
  const [signals,        setSignals]        = useState([])
  const [trades,         setTrades]         = useState([])
  const [riskReport,     setRiskReport]     = useState(null)
  const [wsConnected,    setWsConnected]    = useState(false)
  const [lastUpdate,     setLastUpdate]     = useState(null)

  // Live state pushed over WebSocket
  const [livePositions,  setLivePositions]  = useState({})
  const [livePortfolio,  setLivePortfolio]  = useState(null)
  const [liveHeat,       setLiveHeat]       = useState(null)
  const [liveHalted,     setLiveHalted]     = useState(false)
  const [liveHaltReason, setLiveHaltReason] = useState(null)

  const wsRef = useRef(null)

  // ── WebSocket (live push) ───────────────────────────────────────────────────

  const connectWs = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    const ws = new WebSocket(`${proto}://${location.host}/ws/dashboard`)
    wsRef.current = ws

    ws.onopen = () => {
      setWsConnected(true)
    }

    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data)
        if (msg.type === 'signals') {
          if (msg.signals?.length) setSignals(msg.signals)
          if (msg.positions)       setLivePositions(msg.positions)
          if (msg.portfolio_value) setLivePortfolio(msg.portfolio_value)
          if (msg.portfolio_heat != null) setLiveHeat(msg.portfolio_heat)
          if (msg.halted != null)  setLiveHalted(msg.halted)
          if (msg.halt_reason)     setLiveHaltReason(msg.halt_reason)
          setLastUpdate(new Date().toLocaleTimeString())
        }
      } catch (_) { /* ignore malformed */ }
    }

    ws.onclose = () => {
      setWsConnected(false)
      // Reconnect after 5s
      setTimeout(connectWs, 5000)
    }

    ws.onerror = () => ws.close()
  }, [])

  // ── REST polling fallbacks ───────────────────────────────────────────────────

  const fetchSignals = useCallback(async () => {
    try {
      const r = await fetch('/api/signals')
      if (!r.ok) return
      const d = await r.json()
      if (d.signals?.length) {
        setSignals(d.signals)
        setLastUpdate(new Date().toLocaleTimeString())
      }
    } catch (_) { /* backend not ready */ }
  }, [])

  const fetchHealth = useCallback(async () => {
    try {
      const r = await fetch('/api/health')
      if (!r.ok) return
      setHealth(await r.json())
    } catch (_) { }
  }, [])

  const fetchTrades = useCallback(async () => {
    try {
      const r = await fetch('/api/trades?limit=50')
      if (!r.ok) return
      const d = await r.json()
      setTrades(d.trades || [])
    } catch (_) { }
  }, [])

  const fetchRisk = useCallback(async () => {
    try {
      const r = await fetch('/api/reports/risk')
      if (!r.ok) return
      const d = await r.json()
      setRiskReport(d.data || null)
    } catch (_) { }
  }, [])

  // ── Bootstrap ────────────────────────────────────────────────────────────────

  useEffect(() => {
    connectWs()
    fetchHealth()
    fetchSignals()
    fetchTrades()
    fetchRisk()

    const healthTimer  = setInterval(fetchHealth,  15_000)
    const signalTimer  = setInterval(fetchSignals, 10_000)
    const tradeTimer   = setInterval(fetchTrades,  30_000)
    const riskTimer    = setInterval(fetchRisk,    60_000)

    return () => {
      clearInterval(healthTimer)
      clearInterval(signalTimer)
      clearInterval(tradeTimer)
      clearInterval(riskTimer)
      wsRef.current?.close()
    }
  }, [])  // eslint-disable-line

  // ── Render ───────────────────────────────────────────────────────────────────

  const mode = health?.mode || 'paper'
  const uptime = health?.signal_loop_active

  return (
    <div style={S.app}>
      {/* Header */}
      <div style={S.header}>
        <div style={S.logo}>STOCKBOT</div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 20, fontSize: 10 }}>
          <span style={{ color: C.textDim }}>
            MODE: <span style={{ color: mode === 'live' ? C.red : C.yellow }}>{mode.toUpperCase()}</span>
          </span>
          <span style={{ display: 'flex', alignItems: 'center' }}>
            <Dot active={wsConnected} />
            <span style={{ color: C.textDim }}>WS {wsConnected ? 'LIVE' : 'RECONNECTING'}</span>
          </span>
          <span style={{ display: 'flex', alignItems: 'center' }}>
            <Dot active={uptime} />
            <span style={{ color: C.textDim }}>SIGNAL LOOP</span>
          </span>
          {lastUpdate && (
            <span style={{ color: C.textDim }}>updated {lastUpdate}</span>
          )}
        </div>
      </div>

      {/* Main content */}
      <div style={S.main}>
        {/* Top stat row */}
        <div style={S.grid3}>
          <StatCard
            label="Portfolio Value"
            value={livePortfolio ? `$${livePortfolio.toLocaleString()}` : '—'}
          />
          <StatCard
            label="Open Signals"
            value={signals.filter(s => Math.abs(s.ensemble_signal) >= 0.4).length}
            color={C.green}
          />
          <StatCard
            label="Closed Trades"
            value={trades.filter(t => t.pnl != null).length}
            color={C.purple}
          />
        </div>

        {/* Signals table */}
        <SignalsPanel signals={signals} />

        {/* Positions + Risk side-by-side */}
        <div style={S.grid2}>
          <PositionsPanel
            positions={livePositions}
            portfolioValue={livePortfolio}
            heat={liveHeat}
            halted={liveHalted}
            haltReason={liveHaltReason}
          />
          <div>
            <RiskPanel risk={riskReport} />
            <AgentPanel health={health} />
          </div>
        </div>

        {/* PnL chart */}
        <PnLChart trades={trades} />

        {/* Trades table */}
        <TradesPanel trades={trades.slice(0, 25)} />
      </div>
    </div>
  )
}
