/**
 * polygonClient.js
 * Polygon.io API client - free tier compatible.
 * Uses only /v2/aggs/ticker/{ticker}/range which is confirmed free.
 */

const BASE_URL = 'https://api.polygon.io'
const API_KEY = import.meta.env.VITE_POLYGON_API_KEY || ''

// ── Rate Limiter (5 req/min free tier) ────────────────────────────────────
const rateLimiter = {
  tokens: 5,
  maxTokens: 5,
  refillRate: 5 / 60,
  lastRefill: Date.now(),
  queue: [],

  async acquire() {
    this._refill()
    if (this.tokens >= 1) { this.tokens -= 1; return }
    await new Promise(resolve => {
      this.queue.push(resolve)
      setTimeout(() => this._drain(), (1 / this.refillRate) * 1000)
    })
  },

  _refill() {
    const now = Date.now()
    const elapsed = (now - this.lastRefill) / 1000
    this.tokens = Math.min(this.maxTokens, this.tokens + elapsed * this.refillRate)
    this.lastRefill = now
  },

  _drain() {
    this._refill()
    while (this.queue.length > 0 && this.tokens >= 1) {
      this.tokens -= 1
      this.queue.shift()()
    }
  }
}

// ── Cache ──────────────────────────────────────────────────────────────────
const cache = new Map()
const CACHE_TTL = 15 * 60 * 1000

function getCached(key) {
  const entry = cache.get(key)
  if (!entry) return null
  if (Date.now() - entry.ts > CACHE_TTL) { cache.delete(key); return null }
  return entry.data
}
function setCached(key, data) {
  cache.set(key, { data, ts: Date.now() })
}

// ── Core Fetch ─────────────────────────────────────────────────────────────
async function apiFetch(path, params = {}) {
  const cacheKey = path + JSON.stringify(params)
  const cached = getCached(cacheKey)
  if (cached) return cached

  await rateLimiter.acquire()

  const url = new URL(BASE_URL + path)
  url.searchParams.set('apiKey', API_KEY)
  Object.entries(params).forEach(([k, v]) => url.searchParams.set(k, v))

  console.log('[Polygon] Fetching:', url.pathname)
  const res = await fetch(url.toString())

  if (res.status === 403) {
    const body = await res.json().catch(() => ({}))
    console.error('[Polygon] 403 Forbidden:', body)
    throw new Error(`FORBIDDEN: ${body.message || 'Not authorized'}`)
  }
  if (res.status === 429) throw new Error('RATE_LIMITED')
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(`API_ERROR_${res.status}: ${body.message || ''}`)
  }

  const data = await res.json()
  console.log('[Polygon] Got response for', url.pathname, '— results:', data.resultsCount || data.count || '?')
  setCached(cacheKey, data)
  return data
}

// ── Date helpers ───────────────────────────────────────────────────────────
function toDateStr(date) {
  return date.toISOString().split('T')[0]
}

function getPreviousWeekday(daysBack = 2) {
  const d = new Date()
  d.setDate(d.getDate() - daysBack)
  // Skip weekends
  while (d.getDay() === 0 || d.getDay() === 6) d.setDate(d.getDate() - 1)
  return toDateStr(d)
}

// ── Public API ─────────────────────────────────────────────────────────────

/**
 * Get recent daily bars for a ticker — confirmed free tier endpoint.
 * Uses /v2/aggs/ticker/{ticker}/range/1/day/{from}/{to}
 * Returns last 5 days of OHLCV data.
 */
async function getRecentBars(ticker, days = 5) {
  const to = toDateStr(new Date())
  const from = toDateStr(new Date(Date.now() - (days + 5) * 86400000)) // extra buffer for weekends
  const data = await apiFetch(
    `/v2/aggs/ticker/${ticker}/range/1/day/${from}/${to}`,
    { adjusted: 'true', sort: 'asc', limit: 10 }
  )
  return data.results || []
}

/**
 * Get latest price snapshot for multiple stock tickers.
 * Uses daily aggregate range — free tier compatible.
 * Returns map: { TICKER: { price, open, high, low, volume, change, changePct } }
 */
export async function getSnapshots(tickers) {
  const result = {}
  for (const ticker of tickers) {
    try {
      const bars = await getRecentBars(ticker, 5)
      if (bars.length === 0) {
        console.warn(`[Polygon] No bars returned for ${ticker}`)
        continue
      }
      const last = bars[bars.length - 1]
      const prev = bars.length >= 2 ? bars[bars.length - 2] : last
      result[ticker] = {
        price: last.c,
        open: last.o,
        high: last.h,
        low: last.l,
        volume: last.v,
        vwap: last.vw || last.c,
        change: last.c - prev.c,
        changePct: prev.c ? ((last.c - prev.c) / prev.c) * 100 : 0,
      }
      console.log(`[Polygon] ${ticker} = $${last.c.toFixed(2)}`)
    } catch (e) {
      console.error(`[Polygon] Failed ${ticker}:`, e.message)
    }
  }
  return result
}

/**
 * Get crypto prices.
 * Returns map: { 'X:BTCUSD': { price, ... } }
 */
export async function getCryptoSnapshots(pairs) {
  const result = {}
  for (const pair of pairs) {
    try {
      const bars = await getRecentBars(pair, 5)
      if (bars.length === 0) continue
      const last = bars[bars.length - 1]
      const prev = bars.length >= 2 ? bars[bars.length - 2] : last
      result[pair] = {
        price: last.c,
        open: last.o,
        high: last.h,
        low: last.l,
        volume: last.v,
        change: last.c - prev.c,
        changePct: prev.c ? ((last.c - prev.c) / prev.c) * 100 : 0,
      }
    } catch (e) {
      console.error(`[Polygon] Failed ${pair}:`, e.message)
    }
  }
  return result
}

/**
 * Get historical OHLCV bars for backtesting.
 * @param {string} ticker
 * @param {string} from  - 'YYYY-MM-DD'
 * @param {string} to    - 'YYYY-MM-DD'
 * @param {string} timespan - 'day' | 'hour'
 */
export async function getHistoricalBars(ticker, from, to, timespan = 'day') {
  const data = await apiFetch(
    `/v2/aggs/ticker/${ticker}/range/1/${timespan}/${from}/${to}`,
    { adjusted: 'true', sort: 'asc', limit: 5000 }
  )
  return (data.results || []).map(r => ({
    t: r.t, o: r.o, h: r.h, l: r.l, c: r.c, v: r.v, vw: r.vw || r.c
  }))
}

export async function getTickerDetails(ticker) {
  const data = await apiFetch(`/v3/reference/tickers/${ticker}`)
  return data.results || {}
}

// ── Asset Universe ─────────────────────────────────────────────────────────
export const STOCK_TICKERS = ['NVDA', 'AMD', 'TSLA', 'MSTR', 'PLTR', 'COIN', 'MARA', 'IONQ', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'SMCI', 'ARM', 'AVGO']
export const CRYPTO_TICKERS = ['X:BTCUSD', 'X:ETHUSD', 'X:SOLUSD']
export const ALL_TICKERS = [...STOCK_TICKERS, ...CRYPTO_TICKERS]

export const REFRESH_INTERVAL = parseInt(import.meta.env.VITE_REFRESH_INTERVAL || '') || 15 * 60 * 1000
