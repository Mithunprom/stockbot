/**
 * news.js
 * Fetches market news from Polygon.io (free tier) and scores sentiment.
 * Falls back to curated mock headlines when API unavailable.
 */

const BASE_URL = 'https://api.polygon.io'
const API_KEY = import.meta.env.VITE_POLYGON_API_KEY || ''
const ANTHROPIC_KEY = import.meta.env.VITE_ANTHROPIC_API_KEY || ''

const cache = new Map()
function getCached(key) {
  const e = cache.get(key)
  if (!e || Date.now() - e.ts > 15 * 60 * 1000) { cache.delete(key); return null }
  return e.data
}
function setCached(key, data) { cache.set(key, { data, ts: Date.now() }) }

/**
 * Fetch latest news for a ticker from Polygon.io
 * Endpoint: /v2/reference/news — free tier compatible
 */
export async function fetchTickerNews(ticker, limit = 5) {
  const cacheKey = `news_${ticker}`
  const cached = getCached(cacheKey)
  if (cached) return cached

  if (!API_KEY) return getMockNews(ticker)

  try {
    const url = new URL(`${BASE_URL}/v2/reference/news`)
    url.searchParams.set('ticker', ticker)
    url.searchParams.set('limit', limit)
    url.searchParams.set('order', 'desc')
    url.searchParams.set('sort', 'published_utc')
    url.searchParams.set('apiKey', API_KEY)

    const res = await fetch(url.toString())
    if (!res.ok) return getMockNews(ticker)

    const data = await res.json()
    const articles = (data.results || []).map(a => ({
      title: a.title,
      summary: a.description || '',
      url: a.article_url,
      published: a.published_utc,
      source: a.publisher?.name || 'Unknown',
      tickers: a.tickers || [],
      sentiment: scoreSentimentKeywords(a.title + ' ' + (a.description || '')),
    }))

    setCached(cacheKey, articles)
    return articles
  } catch(e) {
    return getMockNews(ticker)
  }
}

/**
 * Fetch broad market news (no ticker filter)
 */
export async function fetchMarketNews(limit = 10) {
  const cacheKey = 'market_news'
  const cached = getCached(cacheKey)
  if (cached) return cached

  if (!API_KEY) return getMarketMockNews()

  try {
    const url = new URL(`${BASE_URL}/v2/reference/news`)
    url.searchParams.set('limit', limit)
    url.searchParams.set('order', 'desc')
    url.searchParams.set('apiKey', API_KEY)

    const res = await fetch(url.toString())
    if (!res.ok) return getMarketMockNews()

    const data = await res.json()
    const articles = (data.results || []).map(a => ({
      title: a.title,
      summary: a.description || '',
      url: a.article_url,
      published: a.published_utc,
      source: a.publisher?.name || 'Unknown',
      tickers: a.tickers || [],
      sentiment: scoreSentimentKeywords(a.title + ' ' + (a.description || '')),
    }))

    setCached(cacheKey, articles)
    return articles
  } catch(e) {
    return getMarketMockNews()
  }
}

/**
 * Keyword-based sentiment scoring (-1 to +1)
 * No API needed — fast, runs locally
 */
export function scoreSentimentKeywords(text) {
  const t = text.toLowerCase()
  const bullish = ['surge', 'soar', 'rally', 'beat', 'record', 'growth', 'profit', 'gain', 'rise', 'bullish', 'upgrade', 'buy', 'strong', 'breakthrough', 'partnership', 'deal', 'positive', 'exceed', 'outperform', 'boom', 'expansion', 'optimistic']
  const bearish = ['crash', 'fall', 'drop', 'miss', 'loss', 'decline', 'bearish', 'downgrade', 'sell', 'weak', 'warning', 'risk', 'concern', 'fear', 'war', 'sanction', 'tariff', 'recession', 'bankruptcy', 'layoff', 'cut', 'investigation', 'lawsuit', 'negative']
  const highImpact = ['war', 'invasion', 'sanctions', 'fed', 'rate hike', 'rate cut', 'earnings', 'merger', 'acquisition', 'bankrupt', 'default', 'nuclear']

  let score = 0
  let impact = 'low'

  bullish.forEach(w => { if (t.includes(w)) score += 0.15 })
  bearish.forEach(w => { if (t.includes(w)) score -= 0.15 })
  highImpact.forEach(w => { if (t.includes(w)) impact = 'high' })

  return {
    score: Math.max(-1, Math.min(1, score)),
    label: score > 0.2 ? 'BULLISH' : score < -0.2 ? 'BEARISH' : 'NEUTRAL',
    impact,
  }
}

/**
 * Use Claude AI to score sentiment (requires VITE_ANTHROPIC_API_KEY)
 */
export async function scoreWithClaude(headlines) {
  if (!ANTHROPIC_KEY) return null
  try {
    const res = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'x-api-key': ANTHROPIC_KEY, 'anthropic-version': '2023-06-01' },
      body: JSON.stringify({
        model: 'claude-haiku-4-5-20251001',
        max_tokens: 200,
        messages: [{
          role: 'user',
          content: `Rate these news headlines for stock market impact. Reply ONLY with JSON: {"sentiment": "BULLISH"|"BEARISH"|"NEUTRAL", "score": -1 to 1, "reason": "one sentence", "tradeable": true|false}

Headlines:
${headlines.slice(0, 5).map(h => '- ' + h.title).join('\n')}`
        }]
      })
    })
    const data = await res.json()
    const text = data.content?.[0]?.text || '{}'
    const clean = text.replace(/```json|```/g, '').trim()
    return JSON.parse(clean)
  } catch(e) {
    return null
  }
}

// ── Mock data ──────────────────────────────────────────────────────────────
function getMockNews(ticker) {
  const templates = [
    { title: `${ticker} Reports Strong Q4 Earnings, Beats Estimates by 12%`, sentiment: { score: 0.8, label: 'BULLISH', impact: 'high' } },
    { title: `Analysts Upgrade ${ticker} to Buy, Raise Price Target`, sentiment: { score: 0.6, label: 'BULLISH', impact: 'medium' } },
    { title: `${ticker} Announces Strategic Partnership with Major Tech Firm`, sentiment: { score: 0.5, label: 'BULLISH', impact: 'medium' } },
    { title: `Market Volatility Weighs on ${ticker} Shares`, sentiment: { score: -0.3, label: 'BEARISH', impact: 'low' } },
    { title: `${ticker} CEO Discusses AI Strategy at Investor Conference`, sentiment: { score: 0.2, label: 'NEUTRAL', impact: 'low' } },
  ]
  return templates.map(t => ({ ...t, source: 'MarketWatch', published: new Date(Date.now() - Math.random() * 86400000 * 3).toISOString(), url: '#', tickers: [ticker] }))
}

function getMarketMockNews() {
  return [
    { title: 'Fed Signals Potential Rate Cut as Inflation Cools to 2.3%', sentiment: { score: 0.7, label: 'BULLISH', impact: 'high' }, source: 'Reuters', tickers: ['SPY', 'QQQ'], published: new Date(Date.now() - 3600000).toISOString() },
    { title: 'Iran-Israel Tensions Spike Oil Prices 8%, Energy Stocks Surge', sentiment: { score: 0.4, label: 'BULLISH', impact: 'high' }, source: 'Bloomberg', tickers: ['XOM', 'CVX', 'OXY'], published: new Date(Date.now() - 7200000).toISOString() },
    { title: 'NVIDIA Breaks Record as AI Chip Demand Shows No Signs of Slowing', sentiment: { score: 0.9, label: 'BULLISH', impact: 'high' }, source: 'CNBC', tickers: ['NVDA', 'AMD'], published: new Date(Date.now() - 10800000).toISOString() },
    { title: 'Bitcoin Surges Past $100K as Institutional Buying Accelerates', sentiment: { score: 0.85, label: 'BULLISH', impact: 'high' }, source: 'CoinDesk', tickers: ['MSTR', 'COIN', 'MARA'], published: new Date(Date.now() - 14400000).toISOString() },
    { title: 'New China Tariffs Risk Disrupting Global Supply Chains', sentiment: { score: -0.6, label: 'BEARISH', impact: 'high' }, source: 'WSJ', tickers: ['AAPL', 'TSLA'], published: new Date(Date.now() - 18000000).toISOString() },
    { title: 'Tesla Sales Miss Q1 Estimates, Stock Falls Pre-Market', sentiment: { score: -0.7, label: 'BEARISH', impact: 'high' }, source: 'Bloomberg', tickers: ['TSLA'], published: new Date(Date.now() - 21600000).toISOString() },
    { title: 'Palantir Wins $500M Pentagon Contract, Shares Jump 15%', sentiment: { score: 0.9, label: 'BULLISH', impact: 'high' }, source: 'Defense News', tickers: ['PLTR'], published: new Date(Date.now() - 25200000).toISOString() },
    { title: 'Semiconductor Stocks Rally on Strong Taiwan Manufacturing Data', sentiment: { score: 0.5, label: 'BULLISH', impact: 'medium' }, source: 'Reuters', tickers: ['NVDA', 'AMD', 'AVGO', 'ARM'], published: new Date(Date.now() - 28800000).toISOString() },
  ]
}
