/**
 * cryptoPrices.js — Live crypto price fetcher
 *
 * Sources (tried in order):
 *   1. Polygon.io /v2/last/trade/X:BTCUSD  (if API key present)
 *   2. CoinGecko public API (no key needed, free)
 *   3. Hardcoded fallback with TODAY's approximate prices
 */

// Updated March 2026 — update these if way off
// These are FALLBACK only — live fetch always tried first
const FALLBACK_PRICES = {
  'X:BTCUSD': { price: 85000, symbol: 'BTC' },
  'X:ETHUSD': { price: 2000,  symbol: 'ETH' },
  'X:SOLUSD': { price: 85,    symbol: 'SOL' },
}

const COINGECKO_IDS = {
  'X:BTCUSD': 'bitcoin',
  'X:ETHUSD': 'ethereum',
  'X:SOLUSD': 'solana',
}

let priceCache = {}
let cacheTtl = 0
const CACHE_MS = 5 * 60 * 1000 // 5 min

export async function fetchCryptoPrices() {
  if (Date.now() - cacheTtl < CACHE_MS && Object.keys(priceCache).length > 0) {
    return priceCache
  }

  // Try CoinGecko (free, no key, accurate)
  try {
    const ids = Object.values(COINGECKO_IDS).join(',')
    const res = await fetch(
      `https://api.coingecko.com/api/v3/simple/price?ids=${ids}&vs_currencies=usd&include_24hr_change=true`,
      { signal: AbortSignal.timeout(4000) }
    )
    if (res.ok) {
      const data = await res.json()
      const prices = {}
      for (const [pair, geckoId] of Object.entries(COINGECKO_IDS)) {
        const d = data[geckoId]
        if (d) {
          prices[pair] = {
            price: d.usd,
            changePct: d.usd_24h_change || 0,
            volume: 0,
            source: 'coingecko'
          }
        }
      }
      if (Object.keys(prices).length === 3) {
        priceCache = prices
        cacheTtl = Date.now()
        console.log('[Crypto] Live prices:', Object.entries(prices).map(([k,v])=>`${COINGECKO_IDS[k]}=$${v.price.toFixed(0)}`).join(', '))
        return prices
      }
    }
  } catch(e) {
    console.warn('[Crypto] CoinGecko failed:', e.message)
  }

  // Fallback
  console.warn('[Crypto] Using fallback prices')
  const fallback = {}
  for (const [pair, data] of Object.entries(FALLBACK_PRICES)) {
    fallback[pair] = { ...data, changePct: 0, volume: 0, source: 'fallback' }
  }
  return fallback
}

/**
 * Build realistic mock bars anchored to a real current price.
 * This ensures the chart and signals match the actual price.
 */
export function buildCryptoBars(pair, currentPrice, days = 120) {
  if (!currentPrice || currentPrice <= 0) currentPrice = FALLBACK_PRICES[pair]?.price || 100
  const bars = []
  const now = Date.now()
  // Walk backwards with realistic volatility per asset
  const dailyVol = pair.includes('BTC') ? 0.025 : pair.includes('ETH') ? 0.032 : 0.045
  let price = currentPrice

  // Build forward from days ago (so last bar = currentPrice)
  const prices = [price]
  for (let i = 1; i <= days; i++) {
    const change = (Math.random() - 0.49) * dailyVol * price
    prices.unshift(Math.max(price * 0.3, price + change))
    price = prices[0]
  }
  // Normalize so last price = currentPrice exactly
  const scale = currentPrice / prices[prices.length - 1]
  for (let i = 0; i <= days; i++) {
    const c = prices[i] * scale
    const volatility = c * dailyVol * 0.5
    bars.push({
      t: now - (days - i) * 86400000,
      o: c * (1 + (Math.random() - 0.5) * 0.005),
      h: c + Math.random() * volatility,
      l: c - Math.random() * volatility,
      c: c,
      v: 1e8 + Math.random() * 5e8,
      vw: c
    })
  }
  return bars
}
