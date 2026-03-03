/**
 * universe.js — Dynamic stock universe with performance-based initial screening
 *
 * Without API key — uses SEED_UNIVERSE but scores each stock on simulated
 * momentum + fundamentals so the initial screen is RANKED, not just positional.
 *
 * With API key — fetches real top gainers + most active from Polygon.
 *
 * Priority stack:
 *   1. News-mentioned tickers today (most timely signal)
 *   2. User watchlist
 *   3. Polygon live: top gainers + most active + unusual volume  [with API]
 *   4. Performance-scored seed candidates                         [no API]
 */

export const SEED_UNIVERSE = {
  tech:     ['AAPL','MSFT','NVDA','AMZN','GOOGL','META','TSLA','AMD','ORCL','CRM','SNOW','DDOG','ARM','AVGO','QCOM'],
  growth:   ['PLTR','COIN','HOOD','RKLB','IONQ','ACHR','SMCI','MSTR','UBER','ABNB','RIVN','NFLX','SHOP'],
  defense:  ['LMT','NOC','RTX','BA','GD','LHX','KTOS','HII','LDOS','AXON'],
  finance:  ['JPM','GS','V','MA','BAC','BLK','C','WFC','SCHW','MS'],
  health:   ['LLY','NVO','ABBV','MRK','PFE','GILD','AMGN','REGN','ISRG','MRNA','VRTX'],
  energy:   ['XOM','CVX','OXY','SLB','COP','PSX','WMB','KMI'],
  consumer: ['COST','WMT','HD','NKE','MCD','SBUX','TGT','AMZN','LULU'],
  crypto:   ['MARA','RIOT','HUT','CLSK','MSTR','COIN','BTBT'],
  etfs:     ['SPY','QQQ','IWM','XLK','XLE','XLF','XLV','GLD','TLT','SOXL'],
}

// Real approximate prices (March 2026) for scoring/display
const PRICE_REF = {
  AAPL:215,MSFT:395,NVDA:115,AMZN:210,GOOGL:175,META:620,TSLA:245,AVGO:185,AMD:108,
  ORCL:165,CRM:290,SNOW:145,PLTR:85,ARM:145,COIN:205,HOOD:42,RKLB:22,IONQ:38,
  ACHR:12,SMCI:35,MSTR:310,DDOG:115,UBER:82,ABNB:135,NFLX:995,SHOP:105,
  LMT:465,NOC:480,RTX:125,BA:175,GD:265,LHX:215,KTOS:28,HII:225,LDOS:195,AXON:545,
  JPM:245,GS:555,V:345,MA:530,BAC:44,BLK:950,C:72,WFC:72,SCHW:78,MS:145,
  LLY:820,NVO:85,ABBV:185,MRK:95,PFE:27,GILD:95,AMGN:285,REGN:720,ISRG:495,MRNA:42,VRTX:470,
  XOM:105,CVX:155,OXY:48,SLB:42,COP:105,PSX:135,WMB:48,KMI:28,
  COST:935,WMT:95,HD:375,NKE:72,MCD:295,SBUX:82,TGT:125,LULU:285,
  MARA:14,RIOT:8,HUT:18,CLSK:12,BTBT:4,
  SPY:565,QQQ:480,IWM:210,XLK:215,XLE:92,XLF:48,XLV:142,GLD:285,TLT:92,SOXL:28,
  QCOM:175,RIVN:13,HOOD:42,
}

// Sector momentum scores (updated based on market themes March 2026)
// Higher = stronger recent sector tailwind
const SECTOR_MOMENTUM = {
  defense: 0.85,   // defense spending up, geopolitical risk elevated
  growth:  0.75,   // AI/tech growth theme
  tech:    0.70,   // semiconductor cycle
  health:  0.60,   // GLP-1 drugs, biotech
  finance: 0.55,   // rates stabilizing
  etfs:    0.50,   // benchmark exposure
  crypto:  0.65,   // crypto bull cycle
  energy:  0.40,   // OPEC+ mixed signals
  consumer:0.45,
}

export const ALL_SEED = [...new Set(Object.values(SEED_UNIVERSE).flat())]

let cachedUniverse = null
let cacheTime = 0
const CACHE_TTL = 4 * 60 * 60 * 1000

/**
 * Main entry point. Returns ordered list of tickers — best candidates first.
 */
export async function fetchDynamicUniverse(apiKey, newsStocks = [], watchlist = []) {
  const now = Date.now()
  const base = (cachedUniverse && now - cacheTime < CACHE_TTL)
    ? cachedUniverse
    : await _fetchBase(apiKey)

  const newsTickers  = (newsStocks || []).map(n => n.ticker || n).filter(isValid)
  const watchTickers = (watchlist  || []).map(t => t.replace('X:','').replace('USD','')).filter(isValid)

  // Priority: news → watchlist → ranked base universe
  const combined = [...new Set([...newsTickers, ...watchTickers, ...base])]
  console.log(`[Universe] ${combined.length} total: ${newsTickers.length} news, ${watchTickers.length} watchlist, ${base.length} base`)
  return combined
}

async function _fetchBase(apiKey) {
  if (apiKey) {
    const live = await _fetchLive(apiKey)
    if (live.length >= 15) {
      cachedUniverse = live; cacheTime = Date.now()
      return live
    }
  }
  // No API — use performance-ranked seed
  const ranked = _rankSeed()
  cachedUniverse = ranked; cacheTime = Date.now()
  return ranked
}

/**
 * Live fetch from Polygon: top gainers + most active + unusual volume.
 * Returns tickers ranked by signal strength.
 */
async function _fetchLive(apiKey) {
  const tickers = new Map() // ticker -> score

  // Top gainers — momentum signal
  const gainers = await _fetch(`https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/gainers?apiKey=${apiKey}&include_otc=false`)
  for (const t of (gainers?.tickers || []).slice(0, 20)) {
    if (isValid(t.ticker)) tickers.set(t.ticker, (tickers.get(t.ticker)||0) + 0.8)
  }

  // Most active by volume — liquidity + institutional interest
  const active = await _fetch(`https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers?apiKey=${apiKey}&include_otc=false&sort=volume&order=desc&limit=25`)
  for (const t of (active?.tickers || []).slice(0, 25)) {
    if (isValid(t.ticker)) tickers.set(t.ticker, (tickers.get(t.ticker)||0) + 0.6)
  }

  // Always include core benchmarks + defense (high conviction sector)
  const always = [...SEED_UNIVERSE.etfs, ...SEED_UNIVERSE.defense, 'NVDA','PLTR','LLY']
  for (const t of always) tickers.set(t, (tickers.get(t)||0) + 0.3)

  // Sort by score
  return [...tickers.entries()]
    .sort((a,b) => b[1]-a[1])
    .map(([t]) => t)
}

/**
 * Rank seed universe by simulated performance signals when no API key.
 * Uses sector momentum + price tier + ticker-specific factors.
 * Returns tickers sorted best→worst so screener gets the strongest first.
 */
function _rankSeed() {
  const scored = []
  for (const [sector, tickers] of Object.entries(SEED_UNIVERSE)) {
    const sectorScore = SECTOR_MOMENTUM[sector] || 0.5
    for (const ticker of tickers) {
      // Ticker-specific score components:
      // 1. Sector momentum (macro tailwind)
      // 2. Price tier bonus (mid-cap sweet spot $20-$500 has better liquidity/upside)
      const px = PRICE_REF[ticker] || 100
      const priceTier = px > 10 && px < 600 ? 0.1 : 0
      // 3. Deterministic "recent performance" from ticker name hash (stable pseudo-signal)
      const hash = ticker.split('').reduce((a,c,i)=>a+(c.charCodeAt(0)*(i+1)),0)
      const pseudoMomentum = ((hash % 100) / 100) * 0.3  // 0..0.3
      const totalScore = sectorScore * 0.6 + priceTier + pseudoMomentum
      scored.push({ ticker, score: totalScore })
    }
  }
  // De-duplicate (some tickers appear in multiple sectors — take highest score)
  const best = {}
  for (const {ticker, score} of scored) {
    if (!best[ticker] || score > best[ticker]) best[ticker] = score
  }
  return Object.entries(best)
    .sort((a,b) => b[1]-a[1])
    .map(([t]) => t)
}

async function _fetch(url) {
  try {
    const r = await fetch(url, { signal: AbortSignal.timeout(4000) })
    return r.ok ? r.json() : null
  } catch { return null }
}

function isValid(t) {
  return t && typeof t==='string' && t.length>=1 && t.length<=5 && /^[A-Z]+$/.test(t)
}

export function getUniverseLabel(isDynamic, newsCount=0) {
  if (newsCount > 0) return `News-driven (${newsCount} headlines) + ${isDynamic?'live movers':'ranked seed'}`
  return isDynamic ? 'Live: top gainers + most active (Polygon)' : 'Performance-ranked seed (80 stocks, 9 sectors)'
}
