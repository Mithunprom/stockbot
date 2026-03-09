/**
 * alpaca.js — Alpaca Markets broker bridge (stub)
 *
 * Full integration requires:
 *   VITE_ALPACA_KEY_ID + VITE_ALPACA_SECRET_KEY in Vercel env vars.
 * Until configured, all functions return safe no-op values.
 */

const ALPACA_KEY    = import.meta.env?.VITE_ALPACA_KEY_ID    || ''
const ALPACA_SECRET = import.meta.env?.VITE_ALPACA_SECRET_KEY || ''
const BASE_URL      = import.meta.env?.VITE_ALPACA_PAPER === 'false'
  ? 'https://api.alpaca.markets'
  : 'https://paper-api.alpaca.markets'

/** True if credentials are configured */
export function isConfigured() {
  return !!(ALPACA_KEY && ALPACA_SECRET)
}

/** 'paper' | 'live' based on env var */
export function getMode() {
  return import.meta.env?.VITE_ALPACA_PAPER === 'false' ? 'live' : 'paper'
}

/**
 * Sync Alpaca portfolio positions back into the app's portfolio state.
 * Returns null if not configured or on error.
 */
export async function syncPortfolio() {
  if (!isConfigured()) return null
  try {
    const headers = {
      'APCA-API-KEY-ID':     ALPACA_KEY,
      'APCA-API-SECRET-KEY': ALPACA_SECRET,
    }
    const [posRes, acctRes] = await Promise.all([
      fetch(`${BASE_URL}/v2/positions`, { headers, signal: AbortSignal.timeout(5000) }),
      fetch(`${BASE_URL}/v2/account`,   { headers, signal: AbortSignal.timeout(5000) }),
    ])
    if (!posRes.ok || !acctRes.ok) return null
    const positions = await posRes.json()
    const account   = await acctRes.json()

    // Convert Alpaca positions format → app positions format
    const appPositions = {}
    for (const p of positions) {
      appPositions[p.symbol] = {
        shares:   Math.abs(parseFloat(p.qty)),
        avgPrice: parseFloat(p.avg_entry_price),
        side:     parseFloat(p.qty) >= 0 ? 'LONG' : 'SHORT',
        highWater: parseFloat(p.current_price),
      }
    }
    return {
      cash:      parseFloat(account.cash),
      positions: appPositions,
      equity:    parseFloat(account.equity),
    }
  } catch (e) {
    console.warn('[Alpaca] syncPortfolio error:', e.message)
    return null
  }
}

/**
 * Place a market order.
 * Returns order object or null on failure.
 */
export async function placeOrder({ symbol, qty, side }) {
  if (!isConfigured()) return null
  try {
    const r = await fetch(`${BASE_URL}/v2/orders`, {
      method: 'POST',
      headers: {
        'APCA-API-KEY-ID':     ALPACA_KEY,
        'APCA-API-SECRET-KEY': ALPACA_SECRET,
        'Content-Type':        'application/json',
      },
      body: JSON.stringify({ symbol, qty, side, type: 'market', time_in_force: 'day' }),
      signal: AbortSignal.timeout(8000),
    })
    if (!r.ok) return null
    return await r.json()
  } catch (e) {
    console.warn('[Alpaca] placeOrder error:', e.message)
    return null
  }
}
