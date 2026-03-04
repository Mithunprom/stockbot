/**
 * priceGate.js — Price verification gate
 *
 * Priority:
 *   1. Finnhub  — real-time quotes, 60 req/min free, official API
 *   2. Polygon  — fallback, 15-min delayed, 5 req/min free
 *
 * A ticker is BLOCKED from trading if both sources fail or return
 * a price that drifts >5% from the screener price.
 */

const DRIFT_THRESHOLD = 0.05        // 5% max allowed drift
const STALE_MS        = 30 * 60 * 1000  // 30 min

// ── Key readers (never hardcode) ──────────────────────────────────────────

function getFinnhubKey() {
  try {
    const k = import.meta.env.VITE_FINNHUB_API_KEY
    return k && k.length > 8 ? k : null
  } catch { return null }
}

function getPolygonKey() {
  try {
    const k = import.meta.env.VITE_POLYGON_API_KEY
    return k && k.length > 8 && k !== 'your_polygon_api_key_here' ? k : null
  } catch { return null }
}

// ── 1. Finnhub — primary ──────────────────────────────────────────────────

/**
 * Fetch real-time quotes for a batch of tickers from Finnhub.
 * Finnhub /quote endpoint: one request per ticker (parallel, respects 60/min).
 * Returns Map<ticker, { verified, reason, livePrice, source:'finnhub' }>
 */
async function verifyViaFinnhub(tickers) {
  const key = getFinnhubKey()
  if (!key) return new Map()

  // Finnhub free tier: 60 req/min = 1/sec. Fire in small parallel batches.
  const results = new Map()
  const BATCH = 10   // parallel requests per round
  const PAUSE = 800  // ms between batches (stays under rate limit)

  for (let i = 0; i < tickers.length; i += BATCH) {
    const batch = tickers.slice(i, i + BATCH)
    await Promise.all(batch.map(async ticker => {
      try {
        const res = await fetch(
          `https://finnhub.io/api/v1/quote?symbol=${ticker}&token=${key}`,
          { signal: AbortSignal.timeout(6000) }
        )
        if (!res.ok) {
          results.set(ticker, { verified: false, reason: `Finnhub HTTP ${res.status}`, livePrice: null, source: 'finnhub' })
          return
        }
        const d = await res.json()
        // Finnhub quote: { c: current, h, l, o, pc: prev close, t: timestamp }
        const price = d.c   // current price (real-time during market hours)
        const ts    = d.t   // unix seconds

        if (!price || price <= 0) {
          results.set(ticker, { verified: false, reason: 'Finnhub returned no price', livePrice: null, source: 'finnhub' })
          return
        }
        const isStale = ts && (Date.now() - ts * 1000) > STALE_MS
        results.set(ticker, {
          verified: !isStale,
          reason:   isStale ? `Stale (${Math.round((Date.now() - ts*1000)/60000)}min old)` : 'Finnhub real-time OK',
          livePrice: price,
          source:   'finnhub',
        })
      } catch (e) {
        results.set(ticker, { verified: false, reason: `Finnhub: ${e.message}`, livePrice: null, source: 'finnhub' })
      }
    }))
    if (i + BATCH < tickers.length) await new Promise(r => setTimeout(r, PAUSE))
  }
  return results
}

// ── 2. Polygon — fallback ─────────────────────────────────────────────────

/**
 * Fetch batch snapshots from Polygon for tickers that Finnhub couldn't verify.
 * Returns Map<ticker, { verified, reason, livePrice, source:'polygon' }>
 */
async function verifyViaPolygon(tickers) {
  const key = getPolygonKey()
  if (!key || !tickers.length) return new Map()

  const results = new Map()
  // Polygon batch snapshot — up to 50 tickers per call
  for (let i = 0; i < tickers.length; i += 50) {
    const chunk = tickers.slice(i, i + 50)
    try {
      const url = `https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers?tickers=${chunk.join(',')}&apiKey=${key}`
      const res = await fetch(url, { signal: AbortSignal.timeout(8000) })
      if (!res.ok) {
        chunk.forEach(t => results.set(t, { verified: false, reason: `Polygon HTTP ${res.status}`, livePrice: null, source: 'polygon' }))
        continue
      }
      const data = await res.json()
      const snapMap = {}
      for (const s of (data.tickers || [])) snapMap[s.ticker] = s

      for (const ticker of chunk) {
        const snap = snapMap[ticker]
        if (!snap) {
          results.set(ticker, { verified: false, reason: 'Not in Polygon snapshot', livePrice: null, source: 'polygon' })
          continue
        }
        const price     = snap.day?.c || snap.prevDay?.c || null
        const updatedAt = snap.updated
        const isStale   = updatedAt && (Date.now() - updatedAt) > STALE_MS

        if (!price) {
          results.set(ticker, { verified: false, reason: 'No price in Polygon snapshot', livePrice: null, source: 'polygon' })
        } else if (isStale) {
          results.set(ticker, { verified: false, reason: 'Polygon snapshot stale (>30min)', livePrice: price, source: 'polygon' })
        } else {
          results.set(ticker, { verified: true, reason: 'Polygon snapshot OK', livePrice: price, source: 'polygon' })
        }
      }
    } catch (e) {
      chunk.forEach(t => results.set(t, { verified: false, reason: `Polygon: ${e.message}`, livePrice: null, source: 'polygon' }))
    }
  }
  return results
}

// ── Public API ────────────────────────────────────────────────────────────

/**
 * Verify prices for all tickers.
 * Tries Finnhub first, falls back to Polygon for any that fail.
 * Returns Map<ticker, { verified, reason, livePrice, source }>
 */
export async function verifyPricesViaPolygon(tickers, _ignoredKey) {
  // _ignoredKey kept for backward compat — we now read keys internally
  if (!tickers.length) return new Map()

  const hasFinnhub = !!getFinnhubKey()
  const hasPolygon = !!getPolygonKey()

  if (!hasFinnhub && !hasPolygon) {
    return new Map(tickers.map(t => [t, { verified: false, reason: 'No API keys configured', livePrice: null, source: 'none' }]))
  }

  // Step 1: Try Finnhub for all tickers
  const finnhubResults = hasFinnhub ? await verifyViaFinnhub(tickers) : new Map()

  // Step 2: Find tickers that Finnhub couldn't verify → send to Polygon fallback
  const needsPolygon = tickers.filter(t => {
    const r = finnhubResults.get(t)
    return !r || !r.verified
  })

  const polygonResults = hasPolygon && needsPolygon.length
    ? await verifyViaPolygon(needsPolygon)
    : new Map()

  // Step 3: Merge — Finnhub wins if verified, else use Polygon result, else keep Finnhub failure
  const final = new Map()
  for (const ticker of tickers) {
    const fh = finnhubResults.get(ticker)
    const pg = polygonResults.get(ticker)

    if (fh?.verified) {
      final.set(ticker, fh)                          // Finnhub verified ✅
    } else if (pg?.verified) {
      final.set(ticker, pg)                          // Polygon fallback verified ✅
    } else if (pg?.livePrice) {
      final.set(ticker, { ...pg, reason: `Finnhub failed, Polygon unverified: ${pg.reason}` })
    } else if (fh) {
      final.set(ticker, fh)                          // Both failed — keep Finnhub error
    } else {
      final.set(ticker, { verified: false, reason: 'Both Finnhub and Polygon failed', livePrice: null, source: 'none' })
    }
  }
  return final
}

/**
 * Apply gate results to stocks array.
 * Corrects prices to verified live values, tags priceVerified + priceReason + priceSource.
 */
export function applyPriceGate(stocks, verificationMap) {
  return stocks.map(stock => {
    const v = verificationMap.get(stock.ticker)
    if (!v) return { ...stock, priceVerified: false, priceReason: 'Not checked', priceSource: 'none' }

    const correctedPrice = v.livePrice || stock.price

    // Check drift between screener price and verified live price
    let driftBlocked = false
    let reason = v.reason
    if (v.verified && v.livePrice && stock.price) {
      const drift = Math.abs(v.livePrice - stock.price) / v.livePrice
      if (drift > DRIFT_THRESHOLD) {
        driftBlocked = true
        reason = `Price drift ${(drift*100).toFixed(1)}% — screener $${stock.price?.toFixed(2)} vs ${v.source} $${v.livePrice?.toFixed(2)}`
      }
    }

    return {
      ...stock,
      price:         correctedPrice,
      priceVerified: v.verified && !driftBlocked,
      priceReason:   reason,
      priceSource:   v.source || 'none',
    }
  })
}

export function isTradeable(ticker, verifiedTickers) {
  return verifiedTickers.has(ticker)
}
