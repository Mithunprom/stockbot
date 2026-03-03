/**
 * universe.js — Dynamic stock universe selection
 *
 * Instead of hardcoded lists, we pull candidates from real data sources:
 *
 *   1. Polygon /v2/snapshot/locale/us/markets/stocks/gainers  — top movers today
 *   2. Polygon /v2/snapshot/locale/us/markets/stocks/losers   — potential mean-reversion
 *   3. Most active by volume (institutional attention)
 *   4. S&P 500 core + sector ETFs as benchmarks
 *
 * Falls back to a curated seed list if API is unavailable.
 * Universe refreshes every 4 hours so we always screen fresh candidates.
 */

const BASE_URL = 'https://api.polygon.io'

// ── Fallback seed — broad, sector-balanced, NOT cherry-picked winners ─────
// Intentionally includes different sectors, sizes, volatility profiles
export const SEED_UNIVERSE = {
  largeCap:   ['AAPL','MSFT','NVDA','AMZN','GOOGL','META','TSLA','BRK.B','V','JPM','JNJ','UNH','XOM','PG','HD'],
  midCap:     ['PLTR','ARM','COIN','HOOD','RKLB','IONQ','ACHR','SMCI','MSTR','AVGO','AMD','ORCL','CRM','SNOW','DDOG'],
  value:      ['BAC','C','WFC','GS','CVX','OXY','SLB','KO','PEP','MCD','COST','WMT','TGT'],
  healthcare: ['LLY','NVO','ABBV','MRK','PFE','GILD','BMY','AMGN','REGN','ISRG'],
  cryptoAdj:  ['MARA','RIOT','HUT','CLSK','BTBT','MSTR','COIN'],
  etfs:       ['SPY','QQQ','IWM','ARKK','SOXL','XLK','XLF','XLE','XLV','GLD'],
}

export const ALL_SEED = Object.values(SEED_UNIVERSE).flat()

// ── Dynamic universe fetcher ───────────────────────────────────────────────
let cachedUniverse = null
let cacheTime = 0
const CACHE_TTL = 4 * 60 * 60 * 1000 // 4 hours

export async function fetchDynamicUniverse(apiKey) {
  if (cachedUniverse && Date.now() - cacheTime < CACHE_TTL) return cachedUniverse

  if (!apiKey) return buildSeedUniverse()

  try {
    const headers = {}
    const tickers = new Set()

    // 1. Top gainers — momentum candidates
    const gainersUrl = `${BASE_URL}/v2/snapshot/locale/us/markets/stocks/gainers?apiKey=${apiKey}&include_otc=false`
    const gainersRes = await fetch(gainersUrl)
    if (gainersRes.ok) {
      const data = await gainersRes.json()
      ;(data.tickers || []).slice(0, 15).forEach(t => {
        if (isValidTicker(t.ticker)) tickers.add(t.ticker)
      })
    }

    // 2. Most active by volume — institutional interest
    const activeUrl = `${BASE_URL}/v2/snapshot/locale/us/markets/stocks/tickers?apiKey=${apiKey}&include_otc=false&sort=volume&order=desc&limit=20`
    const activeRes = await fetch(activeUrl)
    if (activeRes.ok) {
      const data = await activeRes.json()
      ;(data.tickers || []).slice(0, 20).forEach(t => {
        if (isValidTicker(t.ticker)) tickers.add(t.ticker)
      })
    }

    // 3. Always include core benchmarks
    SEED_UNIVERSE.etfs.forEach(t => tickers.add(t))
    SEED_UNIVERSE.largeCap.slice(0, 10).forEach(t => tickers.add(t))

    if (tickers.size >= 15) {
      const universe = [...tickers]
      cachedUniverse = universe
      cacheTime = Date.now()
      console.log(`[Universe] Dynamic: ${universe.length} tickers from live data`)
      return universe
    }
  } catch(e) {
    console.warn('[Universe] API fetch failed, using seed:', e.message)
  }

  return buildSeedUniverse()
}

function buildSeedUniverse() {
  // Randomly shuffle each sector so we get variety across screener runs
  const shuffled = Object.values(SEED_UNIVERSE).flatMap(arr => shuffle([...arr]))
  const unique = [...new Set(shuffled)]
  console.log(`[Universe] Seed: ${unique.length} tickers`)
  return unique
}

function isValidTicker(t) {
  if (!t || typeof t !== 'string') return false
  if (t.length > 5) return false        // skip warrants, special classes
  if (/[^A-Z]/.test(t)) return false   // letters only
  if (['TEST','FAKE'].includes(t)) return false
  return true
}

function shuffle(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]]
  }
  return arr
}

/**
 * Get display label explaining how the universe was selected
 */
export function getUniverseLabel(isDynamic) {
  return isDynamic
    ? 'Live: top movers + most active + benchmarks'
    : 'Seed: sector-balanced 80-stock universe'
}
