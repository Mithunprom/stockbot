/**
 * kelly.js
 * Kelly criterion for position sizing.
 * Uses fractional Kelly (0.5x) for practical risk management.
 *
 * Full Kelly: f = (bp - q) / b
 * where b = odds, p = win prob, q = 1 - p
 *
 * We map the signal score (-1 to +1) to Kelly fraction.
 */

const KELLY_FRACTION = 0.5  // Half-Kelly for safety
const MAX_POSITION = 0.25    // Never more than 25% in one trade

/**
 * Compute position size as fraction of portfolio.
 * @param {number} signalScore  - -1 to +1
 * @param {number} portfolioValue - total portfolio value
 * @param {Object} options
 * @returns {number} fraction of portfolio to allocate (0 to MAX_POSITION)
 */
export function kellySize(signalScore, portfolioValue, options = {}) {
  const {
    winRate = 0.55,      // historical win rate estimate
    avgWin = 0.03,       // avg win return estimate
    avgLoss = 0.02,      // avg loss return estimate
  } = options

  // Classic Kelly
  const b = avgWin / avgLoss  // odds ratio
  const p = winRate
  const q = 1 - p
  const fullKelly = (b * p - q) / b

  // Scale by signal strength
  const strength = Math.abs(signalScore)
  const fraction = fullKelly * KELLY_FRACTION * strength

  return Math.max(0, Math.min(MAX_POSITION, fraction))
}

/**
 * Compute dynamic stop loss level.
 * Uses ATR-based stops (2x ATR from entry).
 */
export function atrStop(bars, entryPrice, side = 1) {
  if (bars.length < 14) return entryPrice * (1 - 0.07 * side)
  const atrs = []
  for (let i = 1; i < bars.length; i++) {
    const tr = Math.max(
      bars[i].h - bars[i].l,
      Math.abs(bars[i].h - bars[i - 1].c),
      Math.abs(bars[i].l - bars[i - 1].c)
    )
    atrs.push(tr)
  }
  const atr = atrs.slice(-14).reduce((a, b) => a + b, 0) / 14
  return entryPrice - side * 2 * atr
}
