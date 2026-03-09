/**
 * universe.js — Dynamic stock universe
 *
 * Priority order for candidate selection:
 *   1. News-mentioned tickers from today's headlines (highest relevance)
 *   2. Polygon top gainers + most active (momentum candidates)  
 *   3. User's watchlist tickers
 *   4. Sector-balanced seed (fallback)
 *
 * Universe refreshes every 4 hours.
 */

export const SEED_UNIVERSE = {
  tech:      ['AAPL','MSFT','NVDA','AMZN','GOOGL','META','TSLA','AMD','ORCL','CRM','SNOW','DDOG','ARM','AVGO'],
  growth:    ['PLTR','COIN','HOOD','RKLB','IONQ','ACHR','SMCI','MSTR','SNOW','UBER','ABNB','RIVN'],
  defense:   ['LMT','NOC','RTX','BA','GD','LHX','KTOS','HII','LDOS'],
  finance:   ['JPM','GS','V','MA','BAC','BLK','C','WFC','SCHW'],
  health:    ['LLY','NVO','ABBV','MRK','PFE','GILD','AMGN','REGN','ISRG','MRNA'],
  energy:    ['XOM','CVX','OXY','SLB','COP','PSX'],
  consumer:  ['COST','WMT','HD','NKE','MCD','SBUX','TGT'],
  crypto:    ['MARA','RIOT','HUT','CLSK','MSTR','COIN'],
  etfs:      ['SPY','QQQ','IWM','XLK','XLE','XLF','XLV','GLD','TLT'],
}

export const ALL_SEED = [...new Set(Object.values(SEED_UNIVERSE).flat())]

let cachedUniverse = null
let cacheTime = 0
const CACHE_TTL = 4 * 60 * 60 * 1000

export async function fetchDynamicUniverse(apiKey, newsStocks = [], watchlist = []) {
  const now = Date.now()
  // Reuse cache but always prepend fresh news/watchlist tickers
  const base = (cachedUniverse && now - cacheTime < CACHE_TTL) ? cachedUniverse : await _fetchBase(apiKey)

  // News tickers first (most timely), then watchlist, then base universe
  const newsTickers = newsStocks.map(n => n.ticker).filter(t => isValid(t))
  const watchTickers = (watchlist || []).map(t => t.replace('X:','').replace('USD','')).filter(t => isValid(t))

  const combined = [...new Set([...newsTickers, ...watchTickers, ...base])]
  console.log(`[Universe] ${combined.length} tickers: ${newsTickers.length} from news, ${watchTickers.length} from watchlist, ${base.length} base`)
  return combined
}

async function _fetchBase(apiKey) {
  if (!apiKey) {
    const u = buildSeed()
    cachedUniverse = u; cacheTime = Date.now()
    return u
  }
  try {
    const tickers = new Set()

    const gainers = await safeFetch(`https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/gainers?apiKey=${apiKey}&include_otc=false`)
    ;(gainers?.tickers || []).slice(0,15).forEach(t => isValid(t.ticker) && tickers.add(t.ticker))

    const active = await safeFetch(`https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers?apiKey=${apiKey}&include_otc=false&sort=volume&order=desc&limit=20`)
    ;(active?.tickers || []).slice(0,20).forEach(t => isValid(t.ticker) && tickers.add(t.ticker))

    SEED_UNIVERSE.etfs.forEach(t => tickers.add(t))
    SEED_UNIVERSE.tech.slice(0,8).forEach(t => tickers.add(t))
    SEED_UNIVERSE.defense.forEach(t => tickers.add(t))

    if (tickers.size >= 15) {
      const u = [...tickers]
      cachedUniverse = u; cacheTime = Date.now()
      return u
    }
  } catch(e) { console.warn('[Universe]', e.message) }
  return buildSeed()
}

async function safeFetch(url) {
  try {
    const r = await fetch(url, { signal: AbortSignal.timeout(4000) })
    return r.ok ? r.json() : null
  } catch { return null }
}

function buildSeed() {
  const hour = new Date().getHours()
  // Shuffle sectors so different sectors lead at different times
  const sectors = Object.values(SEED_UNIVERSE)
  const rotated = [...sectors.slice(hour % sectors.length), ...sectors.slice(0, hour % sectors.length)]
  return [...new Set(rotated.flat())]
}

function isValid(t) {
  return t && typeof t === 'string' && t.length >= 1 && t.length <= 5 && /^[A-Z]+$/.test(t)
}

export function getUniverseLabel(isDynamic, newsCount = 0) {
  if (newsCount > 0) return `News-driven (${newsCount} from today's headlines) + live movers`
  return isDynamic ? 'Live: top movers + most active' : 'Sector-balanced seed universe'
}
