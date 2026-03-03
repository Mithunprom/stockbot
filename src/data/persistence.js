/**
 * persistence.js — Bulletproof localStorage persistence
 *
 * Key design: save SYNCHRONOUSLY inside state mutations, not in useEffect.
 * useEffect fires after render — if the tab closes between mutation and effect,
 * data is lost. We save directly inside every write operation instead.
 */

export const KEYS = {
  portfolio: 'stockbot_portfolio_v3',   // bumped to v3 — clean slate
  tradeLog:  'stockbot_tradelog_v3',
  watchlist: 'stockbot_watchlist_v2',
  settings:  'stockbot_settings_v2',
}

export function save(key, data) {
  try {
    localStorage.setItem(key, JSON.stringify(data))
    return true
  } catch(e) {
    console.error('[Persist] Save failed:', key, e)
    return false
  }
}

export function load(key, fallback) {
  try {
    const raw = localStorage.getItem(key)
    if (raw === null || raw === undefined) return fallback
    const parsed = JSON.parse(raw)
    return parsed ?? fallback
  } catch(e) {
    console.error('[Persist] Load failed:', key, e)
    return fallback
  }
}

// ── Portfolio ──────────────────────────────────────────────────────────────
export function makeDefaultPortfolio() {
  return { cash: 100000, positions: {}, history: [{ value: 100000, t: Date.now() }], createdAt: Date.now(), version: 3 }
}

export function loadPortfolio() {
  const p = load(KEYS.portfolio, null)
  if (!p || typeof p.cash !== 'number' || typeof p.positions !== 'object') {
    console.log('[Persist] No valid portfolio found, creating default')
    return makeDefaultPortfolio()
  }
  // Ensure positions is always an object
  if (!p.positions) p.positions = {}
  if (!p.history) p.history = [{ value: p.cash, t: Date.now() }]
  console.log('[Persist] Loaded portfolio: cash=$' + p.cash.toFixed(0) + ', positions=' + Object.keys(p.positions).length)
  return p
}

export function savePortfolio(p) {
  const ok = save(KEYS.portfolio, p)
  if (ok) console.log('[Persist] Saved portfolio: cash=$' + p.cash.toFixed(0) + ', positions=' + Object.keys(p.positions).length)
  return ok
}

export function resetPortfolio() {
  const fresh = makeDefaultPortfolio()
  save(KEYS.portfolio, fresh)
  return fresh
}

// ── Trade Log ─────────────────────────────────────────────────────────────
export function loadTradeLog() { return load(KEYS.tradeLog, []) }
export function saveTradeLog(log) { return save(KEYS.tradeLog, log.slice(0, 500)) }

// ── Watchlist ─────────────────────────────────────────────────────────────
const DEFAULT_WATCHLIST = ['NVDA','TSLA','AAPL','BTC','ETH','SOL']
export function loadWatchlist() { return load(KEYS.watchlist, DEFAULT_WATCHLIST) }
export function saveWatchlist(list) { return save(KEYS.watchlist, list) }

// ── Settings ──────────────────────────────────────────────────────────────
const DEFAULT_SETTINGS = { tradeSize: 2000, screenProfile: 'momentum' }
export function loadSettings() { return load(KEYS.settings, DEFAULT_SETTINGS) }
export function saveSettings(s) { return save(KEYS.settings, s) }

// ── Ticker helpers ────────────────────────────────────────────────────────
const TICKER_MAP = { 'BTC':'X:BTCUSD','ETH':'X:ETHUSD','SOL':'X:SOLUSD','BTCUSD':'X:BTCUSD','ETHUSD':'X:ETHUSD','SOLUSD':'X:SOLUSD' }
const DISPLAY_MAP = { 'X:BTCUSD':'BTC','X:ETHUSD':'ETH','X:SOLUSD':'SOL' }

export function normalizeTicket(input) {
  if (!input) return ''
  const upper = input.toUpperCase().trim()
  return TICKER_MAP[upper] || upper
}

export function displayTicker(ticker) {
  return DISPLAY_MAP[ticker] || ticker
}

// ── Debug helper ──────────────────────────────────────────────────────────
export function debugStorage() {
  const p = loadPortfolio()
  const t = loadTradeLog()
  const w = loadWatchlist()
  console.log('[Persist DEBUG]', {
    portfolio: { cash: p.cash, positionCount: Object.keys(p.positions).length, positions: Object.keys(p.positions) },
    tradeCount: t.length,
    watchlist: w,
    rawPortfolio: localStorage.getItem(KEYS.portfolio)?.slice(0, 200)
  })
  return { p, t, w }
}
