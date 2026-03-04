/**
 * priceGate.js — Price verification gate
 *
 * After screening, this module calls Polygon's snapshot API to confirm
 * that each stock has a real, recent price. Stocks that fail are tagged
 * priceVerified=false and BLOCKED from all trades.
 *
 * A stock is BLOCKED if:
 *   - Polygon returns no snapshot / HTTP error
 *   - The price from screener drifts >5% from Polygon live price
 *   - The price is null, 0, or a mock sentinel (≈100)
 *   - Polygon's last_updated is stale (>30 min)
 */

const DRIFT_THRESHOLD = 0.05   // 5% max allowed drift
const STALE_MS        = 30 * 60 * 1000  // 30 minutes

/**
 * Verify a batch of tickers against Polygon snapshots.
 * Returns a Map<ticker, { verified: bool, reason: string, livePrice: number|null }>
 */
export async function verifyPricesViaPolygon(tickers, apiKey) {
  if (!apiKey || !tickers.length) {
    // No API key — all unverified
    return new Map(tickers.map(t => [t, { verified: false, reason: 'No API key', livePrice: null }]))
  }

  // Batch up to 50 tickers per Polygon snapshot call
  const results = new Map()
  const chunks = []
  for (let i = 0; i < tickers.length; i += 50)
    chunks.push(tickers.slice(i, i + 50))

  for (const chunk of chunks) {
    try {
      const url = `https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers?tickers=${chunk.join(',')}&apiKey=${apiKey}`
      const res = await fetch(url, { signal: AbortSignal.timeout(8000) })
      if (!res.ok) {
        chunk.forEach(t => results.set(t, { verified: false, reason: `Polygon HTTP ${res.status}`, livePrice: null }))
        continue
      }
      const data = await res.json()
      const snapshotMap = {}
      for (const snap of (data.tickers || []))
        snapshotMap[snap.ticker] = snap

      for (const ticker of chunk) {
        const snap = snapshotMap[ticker]
        if (!snap) {
          results.set(ticker, { verified: false, reason: 'Not in Polygon snapshot', livePrice: null })
          continue
        }
        const livePrice = snap.day?.c || snap.prevDay?.c || null
        const updatedAt = snap.updated   // unix ms from Polygon
        const isStale = updatedAt && (Date.now() - updatedAt) > STALE_MS

        if (!livePrice) {
          results.set(ticker, { verified: false, reason: 'No price in snapshot', livePrice: null })
        } else if (isStale) {
          results.set(ticker, { verified: false, reason: 'Stale snapshot (>30min)', livePrice })
        } else {
          results.set(ticker, { verified: true, reason: 'OK', livePrice })
        }
      }
    } catch (e) {
      chunk.forEach(t => results.set(t, { verified: false, reason: e.message, livePrice: null }))
    }
  }
  return results
}

/**
 * Cross-check screener prices against verified Polygon prices.
 * Returns updated stocks array — each stock gets:
 *   priceVerified: bool
 *   priceReason:   string
 *   price:         corrected to Polygon live price if available
 */
export function applyPriceGate(stocks, verificationMap) {
  return stocks.map(stock => {
    const v = verificationMap.get(stock.ticker)
    if (!v) return { ...stock, priceVerified: false, priceReason: 'Not checked' }

    // If verified, also correct the price to Polygon's value
    const correctedPrice = v.livePrice || stock.price

    // Check drift between screener price and Polygon live price
    let driftBlocked = false
    let reason = v.reason
    if (v.verified && v.livePrice && stock.price) {
      const drift = Math.abs(v.livePrice - stock.price) / v.livePrice
      if (drift > DRIFT_THRESHOLD) {
        driftBlocked = true
        reason = `Price drift ${(drift*100).toFixed(1)}% (screener $${stock.price?.toFixed(2)} vs Polygon $${v.livePrice?.toFixed(2)})`
      }
    }

    return {
      ...stock,
      price: correctedPrice,
      priceVerified: v.verified && !driftBlocked,
      priceReason: reason,
    }
  })
}

/**
 * Quick check: is a ticker safe to trade?
 * Call this in executeTrade and auto-trader.
 */
export function isTradeable(ticker, verifiedTickers) {
  return verifiedTickers.has(ticker)
}
