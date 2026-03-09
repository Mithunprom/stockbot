/**
 * priceGate.js — Price verification gate
 *
 * Priority order:
 *   1. Finnhub  — real-time quotes, official API (VITE_FINNHUB_API_KEY)
 *   2. Polygon  — 15-min delayed fallback   (VITE_POLYGON_API_KEY)
 *   3. Sanity   — if both APIs fail/rate-limit, accept screener price
 *                 if it passes basic sanity checks (not mock sentinel)
 *
 * A ticker is BLOCKED only if the price is clearly wrong (mock sentinel,
 * zero, or huge drift). API failures alone do NOT block trading.
 */

const DRIFT_THRESHOLD = 0.10        // 10% drift = suspicious
const STALE_MS        = 30 * 60 * 1000

function getFinnhubKey() {
  try { const k = import.meta.env.VITE_FINNHUB_API_KEY; return k?.length > 8 ? k : null } catch { return null }
}
function getPolygonKey() {
  try { const k = import.meta.env.VITE_POLYGON_API_KEY; return k?.length > 8 && k !== 'your_polygon_api_key_here' ? k : null } catch { return null }
}

// ── Tier 3: sanity-check the screener price itself ────────────────────────
// Used when both APIs fail (rate limit, network, etc.)
// Rejects obvious mock sentinels but accepts real-looking prices.
function sanitiseScreenerPrice(price) {
  if (!price || price <= 0)         return { ok: false, reason: 'Zero/null price' }
  if (price === 100)                return { ok: false, reason: 'Mock sentinel $100' }
  if (price === 1 || price === 0.01) return { ok: false, reason: 'Mock sentinel price' }
  if (price > 0.50 && price < 100000) return { ok: true,  reason: 'Screener price passed sanity check' }
  return { ok: false, reason: `Price $${price} out of sane range` }
}

// ── Tier 1: Finnhub real-time quotes ─────────────────────────────────────
async function verifyViaFinnhub(tickers) {
  const key = getFinnhubKey()
  if (!key) return new Map()

  const results = new Map()
  const BATCH = 10
  const PAUSE = 800   // stay under 60/min

  for (let i = 0; i < tickers.length; i += BATCH) {
    const batch = tickers.slice(i, i + BATCH)
    await Promise.all(batch.map(async ticker => {
      try {
        const res = await fetch(
          `https://finnhub.io/api/v1/quote?symbol=${ticker}&token=${key}`,
          { signal: AbortSignal.timeout(6000) }
        )
        if (res.status === 429) {
          // Rate limited — do NOT block, mark as api_limit
          results.set(ticker, { verified: false, reason: 'Finnhub rate limit', livePrice: null, source: 'finnhub', rateLimited: true })
          return
        }
        if (!res.ok) {
          results.set(ticker, { verified: false, reason: `Finnhub HTTP ${res.status}`, livePrice: null, source: 'finnhub' })
          return
        }
        const d = await res.json()
        const price = d.c
        const ts    = d.t
        if (!price || price <= 0) {
          results.set(ticker, { verified: false, reason: 'Finnhub: no price in response', livePrice: null, source: 'finnhub' })
          return
        }
        const isStale = ts && (Date.now() - ts * 1000) > STALE_MS
        results.set(ticker, {
          verified:  !isStale,
          reason:    isStale ? `Finnhub stale (${Math.round((Date.now()-ts*1000)/60000)}min)` : 'Finnhub OK',
          livePrice: price,
          source:    'finnhub',
        })
      } catch (e) {
        results.set(ticker, { verified: false, reason: `Finnhub: ${e.message}`, livePrice: null, source: 'finnhub' })
      }
    }))
    if (i + BATCH < tickers.length) await new Promise(r => setTimeout(r, PAUSE))
  }
  return results
}

// ── Tier 2: Polygon batch snapshot ───────────────────────────────────────
async function verifyViaPolygon(tickers) {
  const key = getPolygonKey()
  if (!key || !tickers.length) return new Map()

  const results = new Map()
  for (let i = 0; i < tickers.length; i += 50) {
    const chunk = tickers.slice(i, i + 50)
    try {
      const res = await fetch(
        `https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers?tickers=${chunk.join(',')}&apiKey=${key}`,
        { signal: AbortSignal.timeout(8000) }
      )
      if (res.status === 429) {
        chunk.forEach(t => results.set(t, { verified: false, reason: 'Polygon rate limit', livePrice: null, source: 'polygon', rateLimited: true }))
        continue
      }
      if (!res.ok) {
        chunk.forEach(t => results.set(t, { verified: false, reason: `Polygon HTTP ${res.status}`, livePrice: null, source: 'polygon' }))
        continue
      }
      const data = await res.json()
      const snapMap = {}
      for (const s of (data.tickers || [])) snapMap[s.ticker] = s

      for (const ticker of chunk) {
        const snap = snapMap[ticker]
        if (!snap) { results.set(ticker, { verified: false, reason: 'Not in Polygon snapshot', livePrice: null, source: 'polygon' }); continue }
        const price     = snap.day?.c || snap.prevDay?.c || null
        const updatedAt = snap.updated
        const isStale   = updatedAt && (Date.now() - updatedAt) > STALE_MS
        if (!price)        results.set(ticker, { verified: false, reason: 'No price in Polygon snapshot', livePrice: null, source: 'polygon' })
        else if (isStale)  results.set(ticker, { verified: false, reason: 'Polygon snapshot stale', livePrice: price, source: 'polygon' })
        else               results.set(ticker, { verified: true,  reason: 'Polygon OK', livePrice: price, source: 'polygon' })
      }
    } catch (e) {
      chunk.forEach(t => results.set(t, { verified: false, reason: `Polygon: ${e.message}`, livePrice: null, source: 'polygon' }))
    }
  }
  return results
}

// ── Public API ────────────────────────────────────────────────────────────

export async function verifyPricesViaPolygon(tickers, _ignored) {
  if (!tickers.length) return new Map()

  // Tier 1: Finnhub
  const fhResults  = getFinnhubKey() ? await verifyViaFinnhub(tickers) : new Map()

  // Tier 2: Polygon — only for tickers Finnhub couldn't verify
  const needsPoly  = tickers.filter(t => { const r = fhResults.get(t); return !r?.verified })
  const pgResults  = getPolygonKey() && needsPoly.length ? await verifyViaPolygon(needsPoly) : new Map()

  // Merge with Tier 3 sanity fallback
  const final = new Map()
  for (const ticker of tickers) {
    const fh = fhResults.get(ticker)
    const pg = pgResults.get(ticker)

    if (fh?.verified) {
      final.set(ticker, fh)
    } else if (pg?.verified) {
      final.set(ticker, pg)
    } else {
      // Both APIs failed — ticker will be sanity-checked against screener price in applyPriceGate
      final.set(ticker, {
        verified:    false,
        reason:      fh?.reason || pg?.reason || 'No API response',
        livePrice:   fh?.livePrice || pg?.livePrice || null,
        source:      'none',
        apiFailed:   true,   // flag: failure was API, not bad price
        rateLimited: fh?.rateLimited || pg?.rateLimited || false,
      })
    }
  }
  return final
}

export function applyPriceGate(stocks, verificationMap) {
  return stocks.map(stock => {
    const v = verificationMap.get(stock.ticker)
    if (!v) return { ...stock, priceVerified: false, priceReason: 'Not checked', priceSource: 'none' }

    // Already verified by Finnhub or Polygon
    if (v.verified) {
      const corrected = v.livePrice || stock.price
      // Check drift
      if (v.livePrice && stock.price) {
        const drift = Math.abs(v.livePrice - stock.price) / v.livePrice
        if (drift > DRIFT_THRESHOLD) {
          return { ...stock, price: v.livePrice, priceVerified: false, priceSource: v.source,
            priceReason: `Drift ${(drift*100).toFixed(1)}% — corrected to ${v.source} $${v.livePrice.toFixed(2)}` }
        }
      }
      return { ...stock, price: corrected, priceVerified: true, priceReason: v.reason, priceSource: v.source }
    }

    // API failed (rate limit / network) — fall back to sanity check on screener price
    if (v.apiFailed || v.rateLimited) {
      const sanity = sanitiseScreenerPrice(stock.price)
      return {
        ...stock,
        priceVerified: sanity.ok,
        priceReason:   sanity.ok
          ? `${v.reason} — screener price $${stock.price?.toFixed(2)} accepted`
          : `${v.reason} — ${sanity.reason}`,
        priceSource:   'screener',
      }
    }

    // Explicit bad price (not in snapshot, wrong ticker, etc.)
    return { ...stock, priceVerified: false, priceReason: v.reason, priceSource: v.source }
  })
}

export function isTradeable(ticker, verifiedTickers) {
  return verifiedTickers.has(ticker)
}
