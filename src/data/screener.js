/**
 * screener.js — S&P 500 screener via grouped daily bars
 *
 * Strategy:
 *   1. Fetch last 5 trading days using the grouped bars endpoint
 *      GET /v2/aggs/grouped/locale/us/market/stocks/{date}
 *      → 1 API call returns ALL US stocks for that date
 *   2. Filter to S&P 500 universe (~250 major components)
 *   3. Score by 5-day momentum + volume surge + trend + news sentiment
 *   4. Return top 20 with their real 5-day OHLCV bars
 *
 * API budget: 5 calls on first screen (one per date, 13s apart).
 * groupedDayCache is in-memory with NO TTL — daily bars are historical
 * and immutable, so subsequent hourly re-screens are instant (0 API calls).
 */

const BASE_URL = 'https://api.polygon.io'
const API_KEY  = import.meta.env.VITE_POLYGON_API_KEY || ''

// ── S&P 500 Universe (~250 major components) ──────────────────────────────
// Covers all 11 GICS sectors. Updated for 2025/2026 composition.
export const SP500_UNIVERSE = [
  // ── Mega-cap Tech ──────────────────────────────────────────────────────
  'AAPL','MSFT','NVDA','AMZN','GOOGL','GOOG','META','TSLA','AVGO','ORCL',
  // ── Large-cap Tech / Semis ─────────────────────────────────────────────
  'AMD','INTC','QCOM','TXN','AMAT','LRCX','KLAC','MRVL','MU','NXPI',
  'ADI','KEYS','MPWR','MCHP','SWKS','QRVO','ENPH','ON','GFS',
  // ── Software / Cloud ───────────────────────────────────────────────────
  'CRM','PANW','SNPS','CDNS','FTNT','CSCO','IBM','DELL','MSI','CDW',
  'NOW','INTU','ADSK','ANSS','CTSH','WDAY','TEAM','HUBS','DDOG','SNOW',
  'CRWD','ZS','OKTA','NET','MDB','ESTC','SPLK','PAYC','PCTY',
  // ── Cyber / Defense Tech ───────────────────────────────────────────────
  'LDOS','BAH','SAIC','CACI','EPAM','GEN',
  // ── Financials ─────────────────────────────────────────────────────────
  'JPM','BAC','WFC','GS','MS','C','USB','PNC','TFC','COF',
  'AXP','BLK','SCHW','CME','ICE','MSCI','SPGI','MCO','FIS','FISV',
  'GPN','MA','V','PYPL','SYF','DFS','KEY','RF','HBAN','CFG','FHN',
  // ── Healthcare ─────────────────────────────────────────────────────────
  'LLY','UNH','JNJ','ABT','ABBV','MRK','PFE','TMO','DHR','SYK',
  'MDT','BSX','EW','ISRG','REGN','VRTX','AMGN','BIIB','BMY','GILD',
  'MRNA','CI','HUM','ELV','CVS','MCK','CAH','ABC','HOLX','ZBH','IQV',
  // ── Consumer Discretionary ─────────────────────────────────────────────
  'HD','MCD','NKE','LOW','SBUX','TJX','BKNG','CMG','YUM','ROST',
  'ORLY','AZO','DHI','LEN','PHM','TOL','NVR','GRMN','ABNB','UBER',
  'LYFT','DASH','RCL','CCL','MAR','HLT','NCLH','MGM','WYNN','LVS',
  'F','GM','RIVN','LCID','STLA',
  // ── Consumer Staples ───────────────────────────────────────────────────
  'WMT','PG','KO','PEP','COST','MDLZ','PM','MO','KMB','CL',
  'GIS','K','HSY','SYY','KR','ADM','EL','CHD','CLX','COTY',
  // ── Energy ─────────────────────────────────────────────────────────────
  'XOM','CVX','COP','EOG','PSX','VLO','MPC','SLB','BKR','HAL',
  'OXY','DVN','FANG','MRO','APA','CTRA','EQT','RRC','AR','SM',
  // ── Industrials ────────────────────────────────────────────────────────
  'HON','CAT','DE','GE','MMM','RTX','LMT','NOC','GD','LHX',
  'UNP','UPS','FDX','DAL','UAL','AAL','LUV','CSX','NSC','CP',
  'EMR','ETN','ROK','PH','ITW','TT','DOV','FTV','XYL','AME','GNRC',
  'KTOS','HII','GD','AXON','RGEN',
  // ── Materials ──────────────────────────────────────────────────────────
  'LIN','APD','ECL','SHW','PPG','IFF','DD','DOW','NUE','STLD',
  'FCX','NEM','ALB','RPM','SEE','PKG','IP','CF','MOS','CE',
  // ── Utilities ──────────────────────────────────────────────────────────
  'NEE','DUK','SO','D','AEP','EXC','SRE','XEL','ES','AWK',
  // ── Real Estate ────────────────────────────────────────────────────────
  'AMT','PLD','CCI','EQIX','SBAC','O','AVB','EQR','PSA','EXR',
  // ── Communication ──────────────────────────────────────────────────────
  'NFLX','DIS','CMCSA','VZ','T','CHTR','TMUS','EA','TTWO','RBLX',
  'SNAP','PINS','ZM','MTCH','PARA','WBD',
  // ── Growth / Momentum (S&P 500 eligible or near-S&P) ──────────────────
  'PLTR','ARM','SMCI','COIN','MSTR','MARA','RIOT','IONQ','HOOD','RKLB',
  'ACHR','JOBY','LUNR','RDDT','CAVA','BIRK',
]

// Deduplicate
const SP500 = [...new Set(SP500_UNIVERSE)]

// ── Date Helpers ──────────────────────────────────────────────────────────
function getRecentTradingDates(n) {
  const dates = []
  const d = new Date()
  // Step back 1 day first — today's grouped bars aren't complete until market close
  d.setDate(d.getDate() - 1)
  while (dates.length < n) {
    if (d.getDay() !== 0 && d.getDay() !== 6) dates.unshift(d.toISOString().split('T')[0])
    d.setDate(d.getDate() - 1)
  }
  return dates
}

// ── Grouped Day Cache ─────────────────────────────────────────────────────
// No TTL: daily bars are historical and immutable within a session.
// First screen fetches 5 days (5 API calls). Every subsequent re-screen
// reuses this cache → 0 additional API calls, instant re-scoring.
const groupedDayCache = new Map()

async function fetchGroupedDay(date) {
  if (groupedDayCache.has(date)) {
    console.log(`[Screener] Cache hit: ${date}`)
    return groupedDayCache.get(date)
  }
  if (!API_KEY) return {}
  try {
    const res = await fetch(
      `${BASE_URL}/v2/aggs/grouped/locale/us/market/stocks/${date}?adjusted=true&include_otc=false&apiKey=${API_KEY}`,
      { signal: AbortSignal.timeout(20000) }
    )
    if (res.status === 429) { console.warn('[Screener] Rate limited on', date); return {} }
    if (!res.ok) { console.warn('[Screener] HTTP', res.status, 'for', date); return {} }
    const data = await res.json()
    const map = {}
    for (const r of (data.results || [])) if (r.T) map[r.T] = r
    groupedDayCache.set(date, map)
    console.log(`[Screener] Grouped ${date}: ${Object.keys(map).length} stocks`)
    return map
  } catch(e) {
    console.warn('[Screener] fetchGroupedDay failed:', date, e.message)
    return {}
  }
}

// ── Main Screener ──────────────────────────────────────────────────────────
/**
 * Screen S&P 500 using grouped daily bars.
 *
 * @param {Object} criteria   - Profile weights (momentum, volumeSurge, trendBreak, ...)
 * @param {number} topN       - Max results (default 20)
 * @param {Array}  newsStocks - [{ticker, score}] from news engine for sentiment boost
 * @returns {Promise<{stocks, all, errors, criteria, timestamp}>}
 */
export async function screenStocks(criteria = DEFAULT_CRITERIA, topN = 20, newsStocks = []) {
  const {
    momentum    = 0.35,
    volumeSurge = 0.25,
    trendBreak  = 0.20,
    rsiOversold = 0.00,
    minPrice  = 5,
    maxPrice  = 15000,
    minVolume = 200000,
  } = criteria

  const NEWS_WEIGHT = 0.20

  // News sentiment boost: ticker → normalized score [0, 1]
  const newsBoost = {}
  for (const n of (newsStocks || [])) {
    if (n.ticker) newsBoost[n.ticker] = Math.min((newsBoost[n.ticker] || 0) + Math.abs(n.score || 0.3), 1)
  }

  // Fetch last 5 completed trading days.
  // Rate limit: 5 req/min → 13s between calls.
  // Cache makes repeat calls instant — only new dates trigger real fetches.
  const dates = getRecentTradingDates(5)
  const dayMaps = []
  for (let i = 0; i < dates.length; i++) {
    dayMaps.push(await fetchGroupedDay(dates[i]))
    // Only pause between calls, and only when a real fetch happened (not cached)
    if (i < dates.length - 1 && !groupedDayCache.has(dates[i + 1])) {
      await new Promise(r => setTimeout(r, 13000))
    }
  }

  // Score every S&P 500 ticker
  const results = []
  for (const ticker of SP500) {
    // Build bar array from the fetched days
    const bars = []
    for (let i = 0; i < dates.length; i++) {
      const r = dayMaps[i][ticker]
      if (r) bars.push({
        t:  new Date(dates[i] + 'T20:00:00Z').getTime(), // ~4pm ET close
        o: r.o, h: r.h, l: r.l, c: r.c, v: r.v, vw: r.vw || r.c,
      })
    }
    if (bars.length < 2) continue  // need at least 2 days to compute returns

    const today     = bars[bars.length - 1]
    const yesterday = bars[bars.length - 2]
    const oldest    = bars[0]

    // Basic filters
    if (today.c < minPrice || today.c > maxPrice) continue
    if (today.v < minVolume) continue

    // ── Score factors ────────────────────────────────────────────────────
    const dayReturn  = (today.c - yesterday.c) / yesterday.c
    const weekReturn = bars.length >= 5
      ? (today.c - oldest.c) / oldest.c
      : dayReturn

    const avgVol = bars.slice(0, -1).reduce((s, b) => s + b.v, 0) / Math.max(1, bars.length - 1)
    const volSurge = avgVol > 0 ? Math.min((today.v / avgVol) - 1, 3) : 0

    const avg5dPrice = bars.reduce((s, b) => s + b.c, 0) / bars.length
    const trendScore = (today.c - avg5dPrice) / avg5dPrice // > 0 = above 5-day avg

    const newsScore  = newsBoost[ticker] || 0

    // Composite: news boosts the momentum component when sentiment is strong
    const adjMomentum = momentum * (1 + newsScore * 0.5)
    const composite =
      dayReturn  * 0.6 * adjMomentum +
      weekReturn * 0.4 * adjMomentum +
      volSurge   * volumeSurge +
      trendScore * trendBreak +
      newsScore  * NEWS_WEIGHT

    const scores = {
      momentum:    dayReturn * 10,
      volatility:  Math.abs(dayReturn) * 5,
      volumeSurge: volSurge,
      trendBreak:  trendScore * 5,
      rsiOversold: 0,
    }

    results.push({
      ticker,
      price:    today.c,
      change1d: dayReturn  * 100,
      change5d: weekReturn * 100,
      volume:   today.v,
      composite,
      scores,
      bars,   // 5 real OHLCV bars — App.jsx will prepend scaled mockBars for full history
    })
  }

  results.sort((a, b) => b.composite - a.composite)

  const top = results.slice(0, topN)
  console.log(`[Screener] S&P 500: scored ${results.length}, top 5: ${top.slice(0,5).map(r=>r.ticker+' '+r.change1d.toFixed(1)+'%').join(' | ')}`)

  return { stocks: top, all: results, errors: [], criteria, timestamp: new Date() }
}

// ── Screening Profiles ─────────────────────────────────────────────────────
export const SCREENING_PROFILES = {
  momentum: {
    label: 'MOMENTUM', description: 'Top gainers with strong trend', icon: '🚀',
    criteria: { momentum: 0.50, volumeSurge: 0.20, trendBreak: 0.20, rsiOversold: 0.00 }
  },
  breakout: {
    label: 'BREAKOUT', description: 'Volume surges + trend breaks', icon: '⚡',
    criteria: { momentum: 0.20, volumeSurge: 0.40, trendBreak: 0.30, rsiOversold: 0.00 }
  },
  meanReversion: {
    label: 'MEAN REVERSION', description: 'Oversold bounce candidates', icon: '🔄',
    criteria: { momentum: 0.00, volumeSurge: 0.10, trendBreak: -0.30, rsiOversold: 0.60 }
  },
  volatile: {
    label: 'HIGH VOLATILITY', description: 'Most volatile for options/momentum', icon: '⚡',
    criteria: { momentum: 0.20, volumeSurge: 0.30, trendBreak: 0.10, rsiOversold: 0.00 }
  },
  balanced: {
    label: 'BALANCED', description: 'Equal weight across all factors', icon: '⚖️',
    criteria: { momentum: 0.25, volumeSurge: 0.25, trendBreak: 0.25, rsiOversold: 0.00 }
  },
}

export const DEFAULT_CRITERIA = SCREENING_PROFILES.momentum.criteria
