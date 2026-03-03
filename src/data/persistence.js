/**
 * persistence.js
 * localStorage-backed persistence for portfolio, watchlist, and settings.
 * All data survives page refresh and browser restarts.
 */

const KEYS = {
  portfolio: 'stockbot_portfolio_v2',
  watchlist: 'stockbot_watchlist_v2',
  tradeLog: 'stockbot_tradelog_v2',
  settings: 'stockbot_settings_v1',
  rlAgent: 'stockbot_rl_agent',
}

function save(key, data) {
  try { localStorage.setItem(key, JSON.stringify(data)) } catch(e) { console.warn('[Persist] Save failed:', e) }
}

function load(key, fallback) {
  try {
    const raw = localStorage.getItem(key)
    if (!raw) return fallback
    return JSON.parse(raw)
  } catch(e) { return fallback }
}

// ── Portfolio ──────────────────────────────────────────────────────────────
const DEFAULT_PORTFOLIO = {
  cash: 100000,
  positions: {},
  history: [{ value: 100000, t: Date.now() }],
  createdAt: Date.now(),
}

export function loadPortfolio() { return load(KEYS.portfolio, DEFAULT_PORTFOLIO) }
export function savePortfolio(p) { save(KEYS.portfolio, p) }
export function resetPortfolio() { save(KEYS.portfolio, { ...DEFAULT_PORTFOLIO, createdAt: Date.now(), history: [{ value: 100000, t: Date.now() }] }) }

// ── Trade Log ─────────────────────────────────────────────────────────────
export function loadTradeLog() { return load(KEYS.tradeLog, []) }
export function saveTradeLog(log) { save(KEYS.tradeLog, log.slice(0, 500)) } // keep last 500 trades

// ── Watchlist ─────────────────────────────────────────────────────────────
const DEFAULT_WATCHLIST = ['NVDA', 'TSLA', 'AAPL', 'BTC', 'ETH']

export function loadWatchlist() { return load(KEYS.watchlist, DEFAULT_WATCHLIST) }
export function saveWatchlist(list) { save(KEYS.watchlist, list) }

// ── Settings ──────────────────────────────────────────────────────────────
const DEFAULT_SETTINGS = {
  tradeSize: 2000,
  screenProfile: 'momentum',
  autoScreenEnabled: true,
}

export function loadSettings() { return load(KEYS.settings, DEFAULT_SETTINGS) }
export function saveSettings(s) { save(KEYS.settings, s) }

// ── Ticker normalization ───────────────────────────────────────────────────
// Maps display names to API tickers
const TICKER_MAP = {
  'BTC': 'X:BTCUSD', 'ETH': 'X:ETHUSD', 'SOL': 'X:SOLUSD',
  'BTCUSD': 'X:BTCUSD', 'ETHUSD': 'X:ETHUSD', 'SOLUSD': 'X:SOLUSD',
}

export function normalizeTicket(input) {
  const upper = input.toUpperCase().trim()
  return TICKER_MAP[upper] || upper
}

export function displayTicker(ticker) {
  const DISPLAY = { 'X:BTCUSD':'BTC', 'X:ETHUSD':'ETH', 'X:SOLUSD':'SOL' }
  return DISPLAY[ticker] || ticker
}
