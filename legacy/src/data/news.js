/**
 * news.js — Live news with GNews free API + sentiment + stock extraction
 *
 * Sources:
 *   1. GNews API (free tier: 100 req/day) — real today's headlines
 *   2. Polygon.io /v2/reference/news (if API key present)
 *   3. Mock headlines (last resort, always fresh timestamps)
 *
 * Key features:
 *   - Cache bypass on manual refresh (force=true)
 *   - Extract mentioned tickers from headlines
 *   - Score sentiment with keywords + optional Claude AI
 *   - Return stocks to ADD to universe based on news
 */

const POLYGON_KEY = import.meta.env.VITE_POLYGON_API_KEY || ''
const GNEWS_KEY   = import.meta.env.VITE_GNEWS_API_KEY || ''

// ── Cache — force=true bypasses it ────────────────────────────────────────
const cache = new Map()
function getCached(key) {
  const e = cache.get(key)
  if (!e || Date.now() - e.ts > 10 * 60 * 1000) { cache.delete(key); return null }
  return e.data
}
function setCached(key, data) { cache.set(key, { data, ts: Date.now() }) }
export function clearNewsCache() { cache.clear() }

// ── Ticker extraction from headline text ──────────────────────────────────
const KNOWN_TICKERS = {
  'nvidia': 'NVDA', 'nvda': 'NVDA', 'amd': 'AMD', 'intel': 'INTC',
  'apple': 'AAPL', 'microsoft': 'MSFT', 'google': 'GOOGL', 'alphabet': 'GOOGL',
  'amazon': 'AMZN', 'meta': 'META', 'tesla': 'TSLA', 'spacex': 'TSLA',
  'palantir': 'PLTR', 'coinbase': 'COIN', 'bitcoin': 'MSTR',
  'boeing': 'BA', 'lockheed': 'LMT', 'northrop': 'NOC', 'raytheon': 'RTX',
  'general dynamics': 'GD', 'l3harris': 'LHX',
  'exxon': 'XOM', 'chevron': 'CVX', 'bp': 'BP',
  'jpmorgan': 'JPM', 'goldman': 'GS', 'blackrock': 'BLK',
  'eli lilly': 'LLY', 'pfizer': 'PFE', 'moderna': 'MRNA',
  'solana': 'SOL', 'ethereum': 'ETH', 'crypto': 'COIN',
  'openai': 'MSFT', 'anthropic': 'GOOGL', 'deepseek': 'NVDA',
  'arm': 'ARM', 'broadcom': 'AVGO', 'qualcomm': 'QCOM',
  'walmart': 'WMT', 'target': 'TGT', 'costco': 'COST',
  'twitter': 'X', 'uber': 'UBER', 'lyft': 'LYFT', 'airbnb': 'ABNB',
  'netflix': 'NFLX', 'disney': 'DIS', 'warner': 'WBD',
  'ford': 'F', 'gm': 'GM', 'rivian': 'RIVN',
  'taiwan': 'TSM', 'tsmc': 'TSM', 'samsung': 'SSNLF',
  'rocketlab': 'RKLB', 'rocket lab': 'RKLB',
  'pentagon': 'LMT', 'defense': 'LMT',
  'fed': 'SPY', 'federal reserve': 'SPY', 'rate': 'TLT',
  'oil': 'XOM', 'energy': 'XLE', 'gold': 'GLD',
  'semiconductor': 'NVDA', 'chip': 'NVDA', 'ai chip': 'NVDA',
}

export function extractTickersFromText(text) {
  if (!text) return []
  const lower = text.toLowerCase()
  const found = new Set()

  // Check known name mappings
  for (const [name, ticker] of Object.entries(KNOWN_TICKERS)) {
    if (lower.includes(name)) found.add(ticker)
  }

  // Check direct ticker mentions ($NVDA or standalone NVDA)
  const tickerPattern = /\$([A-Z]{2,5})\b|\b([A-Z]{2,5})\b/g
  const words = text.match(/\b[A-Z]{2,5}\b/g) || []
  const stockWords = new Set(['AI','US','UK','EU','CEO','IPO','ETF','GDP','CPI','FED','SEC','FDA','DOJ','DOD','NATO','FBI','CIA','OIL','GAS','ESG','SPY','QQQ','ETF'])
  words.forEach(w => { if (w.length >= 2 && w.length <= 5 && !stockWords.has(w)) found.add(w) })

  return [...found].slice(0, 5)
}

// ── Sentiment scoring ──────────────────────────────────────────────────────
const BULLISH = ['surge','rally','beat','growth','profit','upgrade','partnership','breakthrough','record','soar','jump','win','contract','buy','strong','boost','rise','gain','bull','optimis']
const BEARISH  = ['crash','fall','miss','loss','downgrade','warning','lawsuit','bankrupt','decline','drop','sell','weak','recession','layoff','fired','probe','fraud','sanction','tariff','slump']
const HIGH_IMPACT = ['fed','rate','war','invasion','earnings','bankruptcy','merger','acquisition','fda','ban','sanction','nuclear','default','crisis','collapse','bailout']

export function scoreSentiment(text) {
  if (!text) return { score: 0, label: 'NEUTRAL', impact: 'low' }
  const lower = text.toLowerCase()
  let score = 0
  BULLISH.forEach(w => { if (lower.includes(w)) score += 0.15 })
  BEARISH.forEach(w => { if (lower.includes(w)) score -= 0.15 })
  const impact = HIGH_IMPACT.some(w => lower.includes(w)) ? 'high' : 'low'
  score = Math.max(-1, Math.min(1, score))
  return { score, label: score > 0.1 ? 'BULLISH' : score < -0.1 ? 'BEARISH' : 'NEUTRAL', impact }
}

// ── GNews fetch (real today's headlines, free) ────────────────────────────
async function fetchGNews(query = 'stock market', force = false) {
  const key = `gnews_${query}`
  if (!force) { const c = getCached(key); if (c) return c }

  if (!GNEWS_KEY) return null

  try {
    const url = `https://gnews.io/api/v4/search?q=${encodeURIComponent(query)}&lang=en&country=us&max=10&apikey=${GNEWS_KEY}&sortby=publishedAt`
    const res = await fetch(url, { signal: AbortSignal.timeout(5000) })
    if (!res.ok) return null
    const data = await res.json()
    const articles = (data.articles || []).map(a => ({
      title: a.title,
      summary: a.description || '',
      url: a.url,
      published: a.publishedAt,
      source: a.source?.name || 'GNews',
      tickers: extractTickersFromText(a.title + ' ' + (a.description || '')),
      sentiment: scoreSentiment(a.title + ' ' + (a.description || '')),
      isLive: true,
    }))
    if (articles.length > 0) { setCached(key, articles); return articles }
  } catch(e) { console.warn('[News] GNews failed:', e.message) }
  return null
}

// ── Polygon news fetch ────────────────────────────────────────────────────
async function fetchPolygonNews(ticker = '', limit = 10, force = false) {
  const key = `polygon_news_${ticker}_${limit}`
  if (!force) { const c = getCached(key); if (c) return c }
  if (!POLYGON_KEY) return null

  try {
    const url = new URL('https://api.polygon.io/v2/reference/news')
    url.searchParams.set('limit', limit)
    url.searchParams.set('order', 'desc')
    url.searchParams.set('sort', 'published_utc')
    url.searchParams.set('apiKey', POLYGON_KEY)
    if (ticker) url.searchParams.set('ticker', ticker)

    const res = await fetch(url.toString(), { signal: AbortSignal.timeout(5000) })
    if (!res.ok) return null
    const data = await res.json()
    const articles = (data.results || []).map(a => ({
      title: a.title,
      summary: a.description || '',
      url: a.article_url,
      published: a.published_utc,
      source: a.publisher?.name || 'News',
      tickers: a.tickers?.length ? a.tickers : extractTickersFromText(a.title),
      sentiment: scoreSentiment(a.title + ' ' + (a.description || '')),
      isLive: true,
    }))
    if (articles.length > 0) { setCached(key, articles); return articles }
  } catch(e) { console.warn('[News] Polygon failed:', e.message) }
  return null
}

// ── Mock news — always fresh timestamps, rotation so it changes ───────────
const MOCK_POOL = [
  { title: 'NVIDIA H200 Demand Surges as AI Training Clusters Expand Globally', tickers: ['NVDA','AMD'], sentiment: { score:0.85, label:'BULLISH', impact:'high' } },
  { title: 'Federal Reserve Holds Rates, Signals 2 Cuts Possible in 2026', tickers: ['SPY','TLT','GLD'], sentiment: { score:0.5, label:'BULLISH', impact:'high' } },
  { title: 'Pentagon Awards $2.5B AI Contract to Palantir and Microsoft', tickers: ['PLTR','MSFT','LMT'], sentiment: { score:0.9, label:'BULLISH', impact:'high' } },
  { title: 'Boeing 737 MAX Production Resumes After FAA Safety Clearance', tickers: ['BA'], sentiment: { score:0.7, label:'BULLISH', impact:'high' } },
  { title: 'Lockheed Martin Wins F-47 Next-Gen Fighter Contract', tickers: ['LMT','NOC','RTX'], sentiment: { score:0.85, label:'BULLISH', impact:'high' } },
  { title: 'Tesla Deliveries Miss Q1 Estimates by 18%, Shares Fall Pre-Market', tickers: ['TSLA'], sentiment: { score:-0.7, label:'BEARISH', impact:'high' } },
  { title: 'Solana Network Hits 100K TPS Milestone, SOL Rallies 12%', tickers: ['SOL','COIN'], sentiment: { score:0.8, label:'BULLISH', impact:'medium' } },
  { title: 'China Tariffs Expanded to Include Semiconductors, TSMC Warns of Impact', tickers: ['NVDA','AMD','AVGO','TSM'], sentiment: { score:-0.65, label:'BEARISH', impact:'high' } },
  { title: 'Northrop Grumman B-21 Raider Production Accelerates on Pentagon Order', tickers: ['NOC','LMT','GD'], sentiment: { score:0.8, label:'BULLISH', impact:'high' } },
  { title: 'Eli Lilly Obesity Drug Gets EU Approval, Stock Up 8%', tickers: ['LLY','NVO'], sentiment: { score:0.75, label:'BULLISH', impact:'high' } },
  { title: 'Bitcoin ETF Inflows Hit $1.2B in Single Day as Institutions Buy', tickers: ['MSTR','COIN','MARA'], sentiment: { score:0.85, label:'BULLISH', impact:'high' } },
  { title: 'Apple Vision Pro 2 Unveiled with M4 Chip, Pre-orders Sell Out', tickers: ['AAPL'], sentiment: { score:0.7, label:'BULLISH', impact:'medium' } },
  { title: 'Palantir AIP Deployed Across 40 NATO Allies for Battlefield AI', tickers: ['PLTR','LMT','RTX'], sentiment: { score:0.9, label:'BULLISH', impact:'high' } },
  { title: 'Oil Prices Slip as OPEC+ Raises Production Quota Unexpectedly', tickers: ['XOM','CVX','OXY'], sentiment: { score:-0.4, label:'BEARISH', impact:'medium' } },
  { title: 'Goldman Sachs Upgrades AMD to Buy, Sets $250 Target', tickers: ['AMD','NVDA'], sentiment: { score:0.7, label:'BULLISH', impact:'medium' } },
  { title: 'DeepSeek R2 Launch Sparks AI Spending Debate, Nvidia Dips 5%', tickers: ['NVDA','MSFT','GOOGL'], sentiment: { score:-0.3, label:'BEARISH', impact:'high' } },
  { title: 'Rocket Lab Wins NASA Artemis Payload Contract Worth $450M', tickers: ['RKLB'], sentiment: { score:0.85, label:'BULLISH', impact:'high' } },
  { title: 'Meta AI Assistant Hits 1B Users, Ad Revenue Beats Estimates', tickers: ['META'], sentiment: { score:0.8, label:'BULLISH', impact:'medium' } },
]

function getMockMarketNews() {
  // Shuffle deterministically by hour so news "rotates" through the day
  const hour = new Date().getHours()
  const shuffled = [...MOCK_POOL].sort((a,b) => {
    const hashA = (a.title.charCodeAt(0) + hour) % MOCK_POOL.length
    const hashB = (b.title.charCodeAt(0) + hour) % MOCK_POOL.length
    return hashA - hashB
  })
  return shuffled.slice(0, 12).map((n, i) => ({
    ...n,
    source: ['Reuters','Bloomberg','CNBC','WSJ','Barron\'s','MarketWatch'][i % 6],
    published: new Date(Date.now() - i * 28 * 60000).toISOString(), // staggered by 28 min
    url: '#',
    isLive: false,
  }))
}

function getMockTickerNews(ticker) {
  const relevant = MOCK_POOL.filter(n => n.tickers.includes(ticker))
  const generic = MOCK_POOL.slice(0, 3)
  return [...relevant, ...generic].slice(0, 5).map((n, i) => ({
    ...n,
    published: new Date(Date.now() - i * 2 * 3600000).toISOString(),
    url: '#', isLive: false,
  }))
}

// ── Public API ─────────────────────────────────────────────────────────────

export async function fetchMarketNews(limit = 12, force = false) {
  if (force) { cache.delete('gnews_stock market'); cache.delete(`polygon_news__${limit}`) }

  // Try live sources
  const polygon = await fetchPolygonNews('', limit, force)
  if (polygon?.length) return polygon

  const gnews = await fetchGNews('stock market investing', force)
  if (gnews?.length) return gnews.slice(0, limit)

  // Mock with fresh timestamps
  return getMockMarketNews().slice(0, limit)
}

export async function fetchTickerNews(ticker, limit = 5, force = false) {
  if (force) cache.delete(`polygon_news_${ticker}_${limit}`)
  const polygon = await fetchPolygonNews(ticker, limit, force)
  if (polygon?.length) return polygon
  return getMockTickerNews(ticker).slice(0, limit)
}

/**
 * Extract new stock candidates from news headlines.
 * Returns tickers that appear in bullish high-impact news.
 */
export function extractNewsStocks(articles) {
  const candidates = {}
  for (const a of articles) {
    const tickers = a.tickers?.length ? a.tickers : extractTickersFromText(a.title)
    const score = a.sentiment?.score || 0
    const weight = a.sentiment?.impact === 'high' ? 2 : 1
    for (const t of tickers) {
      if (!candidates[t]) candidates[t] = { ticker: t, score: 0, count: 0, headlines: [] }
      candidates[t].score += score * weight
      candidates[t].count += 1
      candidates[t].headlines.push(a.title)
    }
  }
  // Return tickers with net positive mention
  return Object.values(candidates)
    .filter(c => c.count >= 1 && Math.abs(c.score) > 0.2)
    .sort((a,b) => Math.abs(b.score) - Math.abs(a.score))
    .slice(0, 10)
}

/**
 * Claude AI market sentiment analysis
 */
export async function scoreWithClaude(articles) {
  const key = import.meta.env.VITE_ANTHROPIC_API_KEY
  if (!key || !articles?.length) return null
  try {
    const headlines = articles.slice(0, 8).map(a => a.title).join('\n')
    const res = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: { 'x-api-key': key, 'anthropic-version': '2023-06-01', 'content-type': 'application/json' },
      body: JSON.stringify({
        model: 'claude-haiku-4-5-20251001', max_tokens: 200,
        messages: [{ role: 'user', content: `Rate overall stock market sentiment from these headlines. Reply JSON only: {"sentiment":"BULLISH"|"BEARISH"|"NEUTRAL","score":-1to1,"reason":"one sentence","tradeable":true|false}\n\n${headlines}` }]
      })
    })
    if (!res.ok) return null
    const d = await res.json()
    const text = d.content?.[0]?.text || ''
    return JSON.parse(text.replace(/```json|```/g, '').trim())
  } catch(e) { return null }
}
