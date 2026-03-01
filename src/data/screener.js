/**
 * screener.js
 * Dynamic stock screener — discovers stocks based on real performance criteria.
 *
 * Screening criteria (configurable):
 *   1. Momentum     — top % gainers over N days
 *   2. Volatility   — stocks with unusual price movement (ATR-based)
 *   3. Volume surge — unusual volume vs 20-day average
 *   4. Trend break  — price crossing above key moving averages
 *   5. RSI oversold — potential mean-reversion candidates
 *
 * Data source: Polygon.io free tier (/v2/aggs/grouped/locale/us/market/stocks/{date})
 * This endpoint returns ALL stocks for a given day in one call — very efficient.
 */

const BASE_URL = 'https://api.polygon.io'
const API_KEY = import.meta.env.VITE_POLYGON_API_KEY || ''

// ── Candidate universe (broad market, not cherry-picked) ──────────────────
// These are just seeds for the screener — the screener will rank and filter them
// In a real system you'd pull from a full market index
const BROAD_UNIVERSE = [
  // Mega cap tech
  'AAPL','MSFT','NVDA','AMZN','GOOGL','META','TSLA','AVGO','AMD','ORCL',
  // Growth / momentum
  'PLTR','ARM','SMCI','MSTR','COIN','MARA','IONQ','HOOD','RKLB','ACHR',
  // Finance
  'JPM','GS','V','MA','BRK.B','BAC','C','WFC',
  // Healthcare
  'LLY','NVO','UNH','JNJ','ABBV','MRK',
  // Energy
  'XOM','CVX','OXY','SLB',
  // Consumer
  'COST','WMT','HD','NKE','SBUX',
  // ETFs as market proxies
  'SPY','QQQ','IWM','ARKK','SOXL',
  // Crypto-adjacent
  'RIOT','HUT','CLSK','BTBT',
]

// ── Cache ──────────────────────────────────────────────────────────────────
const cache = new Map()
function getCached(key) {
  const e = cache.get(key)
  if (!e || Date.now() - e.ts > 15 * 60 * 1000) { cache.delete(key); return null }
  return e.data
}
function setCached(key, data) { cache.set(key, { data, ts: Date.now() }) }

async function apiFetch(path, params = {}) {
  const key = path + JSON.stringify(params)
  const cached = getCached(key)
  if (cached) return cached

  const url = new URL(BASE_URL + path)
  url.searchParams.set('apiKey', API_KEY)
  Object.entries(params).forEach(([k, v]) => url.searchParams.set(k, v))

  const res = await fetch(url.toString())
  if (!res.ok) throw new Error(`API_${res.status}`)
  const data = await res.json()
  setCached(key, data)
  return data
}

// ── Get grouped daily bars (1 API call for entire market) ──────────────────
async function getGroupedDaily(date) {
  const data = await apiFetch(
    `/v2/aggs/grouped/locale/us/market/stocks/${date}`,
    { adjusted: 'true' }
  )
  // Returns { results: [{ T: ticker, o, h, l, c, v, vw, n }] }
  const map = {}
  for (const r of (data.results || [])) {
    if (r.T) map[r.T] = { ticker: r.T, open: r.o, high: r.h, low: r.l, close: r.c, volume: r.v, vwap: r.vw, trades: r.n }
  }
  return map
}

// ── Get recent daily bars for a ticker (for multi-day calculations) ────────
async function getDailyBars(ticker, days = 30) {
  const to = new Date()
  const from = new Date(Date.now() - (days + 10) * 86400000)
  const fmtD = d => d.toISOString().split('T')[0]
  const data = await apiFetch(
    `/v2/aggs/ticker/${ticker}/range/1/day/${fmtD(from)}/${fmtD(to)}`,
    { adjusted: 'true', sort: 'asc', limit: days + 10 }
  )
  return data.results || []
}

// ── Previous trading day ───────────────────────────────────────────────────
function getPrevTradingDay(daysBack = 1) {
  const d = new Date()
  d.setDate(d.getDate() - daysBack)
  while (d.getDay() === 0 || d.getDay() === 6) d.setDate(d.getDate() - 1)
  return d.toISOString().split('T')[0]
}

// ── Scoring functions ──────────────────────────────────────────────────────

function calcMomentumScore(bars) {
  if (bars.length < 5) return 0
  const current = bars[bars.length - 1].c
  const weekAgo = bars[Math.max(0, bars.length - 5)].c
  const monthAgo = bars[Math.max(0, bars.length - 20)].c
  const weekReturn = (current - weekAgo) / weekAgo
  const monthReturn = (current - monthAgo) / monthAgo
  return weekReturn * 0.6 + monthReturn * 0.4
}

function calcVolatilityScore(bars) {
  if (bars.length < 10) return 0
  const returns = bars.slice(-10).map((b, i, arr) =>
    i === 0 ? 0 : Math.abs((b.c - arr[i-1].c) / arr[i-1].c)
  )
  return returns.reduce((a, b) => a + b, 0) / returns.length
}

function calcVolumeSurgeScore(bars) {
  if (bars.length < 21) return 0
  const avgVol = bars.slice(-21, -1).reduce((a, b) => a + b.v, 0) / 20
  const todayVol = bars[bars.length - 1].v
  return avgVol > 0 ? (todayVol / avgVol) - 1 : 0
}

function calcTrendBreakScore(bars) {
  if (bars.length < 20) return 0
  const closes = bars.map(b => b.c)
  const ma20 = closes.slice(-20).reduce((a, b) => a + b, 0) / 20
  const ma5 = closes.slice(-5).reduce((a, b) => a + b, 0) / 5
  const current = closes[closes.length - 1]
  // Positive when price > both MAs and short MA > long MA (golden cross-like)
  const aboveMa20 = (current - ma20) / ma20
  const shortAboveLong = (ma5 - ma20) / ma20
  return aboveMa20 * 0.5 + shortAboveLong * 0.5
}

function calcRSIOversoldScore(bars) {
  if (bars.length < 15) return 0
  const closes = bars.map(b => b.c)
  let gains = 0, losses = 0
  for (let i = closes.length - 14; i < closes.length; i++) {
    const diff = closes[i] - closes[i-1]
    if (diff > 0) gains += diff; else losses -= diff
  }
  const rs = losses === 0 ? 100 : gains / losses
  const rsi = 100 - 100 / (1 + rs)
  // Score is higher when RSI is oversold (< 35) — contrarian bounce signal
  return rsi < 35 ? (35 - rsi) / 35 : rsi > 65 ? -(rsi - 65) / 35 : 0
}

// ── Main Screener ──────────────────────────────────────────────────────────

/**
 * Screen stocks based on criteria weights.
 * @param {Object} criteria - weights for each criterion (0 to 1)
 * @param {number} topN     - how many stocks to return
 * @returns {Array} ranked stocks with scores
 */
export async function screenStocks(criteria = DEFAULT_CRITERIA, topN = 20) {
  const {
    momentum = 0.3,
    volatility = 0.2,
    volumeSurge = 0.2,
    trendBreak = 0.2,
    rsiOversold = 0.1,
    minPrice = 5,
    maxPrice = 10000,
    minVolume = 500000,
  } = criteria

  // Use a sample of the universe to stay within rate limits
  // In a real system this would use the grouped daily endpoint
  const universe = BROAD_UNIVERSE.filter(t => !t.includes('.')) // skip BRK.B for simplicity

  const results = []
  const errors = []

  // Rate-limited sequential fetch
  for (const ticker of universe.slice(0, 25)) { // limit to 25 for free tier
    try {
      const bars = await getDailyBars(ticker, 30)
      if (bars.length < 10) continue

      const last = bars[bars.length - 1]

      // Apply filters
      if (last.c < minPrice || last.c > maxPrice) continue
      if (last.v < minVolume) continue

      // Score each criterion
      const scores = {
        momentum: calcMomentumScore(bars),
        volatility: calcVolatilityScore(bars),
        volumeSurge: calcVolumeSurgeScore(bars),
        trendBreak: calcTrendBreakScore(bars),
        rsiOversold: calcRSIOversoldScore(bars),
      }

      // Composite weighted score
      const composite =
        scores.momentum * momentum +
        scores.volatility * volatility +
        scores.volumeSurge * volumeSurge +
        scores.trendBreak * trendBreak +
        scores.rsiOversold * rsiOversold

      const dayChange = bars.length >= 2
        ? (last.c - bars[bars.length - 2].c) / bars[bars.length - 2].c * 100
        : 0

      const weekChange = bars.length >= 5
        ? (last.c - bars[bars.length - 5].c) / bars[bars.length - 5].c * 100
        : 0

      results.push({
        ticker,
        price: last.c,
        change1d: dayChange,
        change5d: weekChange,
        volume: last.v,
        composite,
        scores,
        bars,
      })

    } catch (e) {
      errors.push({ ticker, error: e.message })
    }

    // Small delay to be gentle on rate limits
    await new Promise(r => setTimeout(r, 200))
  }

  // Sort by composite score descending
  results.sort((a, b) => b.composite - a.composite)

  console.log(`[Screener] Screened ${results.length} stocks, ${errors.length} errors`)

  return {
    stocks: results.slice(0, topN),
    all: results,
    errors,
    criteria,
    timestamp: new Date(),
  }
}

// ── Preset Criteria Profiles ───────────────────────────────────────────────

export const SCREENING_PROFILES = {
  momentum: {
    label: 'MOMENTUM',
    description: 'Top gainers with strong trend',
    icon: '🚀',
    criteria: { momentum: 0.5, volatility: 0.1, volumeSurge: 0.2, trendBreak: 0.2, rsiOversold: 0.0 }
  },
  breakout: {
    label: 'BREAKOUT',
    description: 'Volume surges + trend breaks',
    icon: '⚡',
    criteria: { momentum: 0.2, volatility: 0.1, volumeSurge: 0.4, trendBreak: 0.3, rsiOversold: 0.0 }
  },
  meanReversion: {
    label: 'MEAN REVERSION',
    description: 'Oversold bounce candidates',
    icon: '🔄',
    criteria: { momentum: 0.0, volatility: 0.2, volumeSurge: 0.1, trendBreak: 0.1, rsiOversold: 0.6 }
  },
  volatile: {
    label: 'HIGH VOLATILITY',
    description: 'Most volatile for options/momentum',
    icon: '⚡',
    criteria: { momentum: 0.2, volatility: 0.5, volumeSurge: 0.2, trendBreak: 0.1, rsiOversold: 0.0 }
  },
  balanced: {
    label: 'BALANCED',
    description: 'Equal weight across all factors',
    icon: '⚖️',
    criteria: { momentum: 0.25, volatility: 0.2, volumeSurge: 0.2, trendBreak: 0.2, rsiOversold: 0.15 }
  },
}

export const DEFAULT_CRITERIA = SCREENING_PROFILES.momentum.criteria
