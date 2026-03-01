/**
 * signals.js
 * Multi-factor signal generation engine.
 *
 * Factors implemented:
 *   1. Momentum (12-1 month, classic Jegadeesh-Titman)
 *   2. Mean-reversion (short-term, 5-day RSI-based)
 *   3. Volume anomaly (unusual volume = institutional interest)
 *   4. Volatility regime (low-vol = risk-on bias)
 *   5. Trend strength (price vs VWAP, 20-day MA)
 *
 * Weights are managed by the RL agent in /src/rl/agent.js
 */

// ── Utility ────────────────────────────────────────────────────────────────

function mean(arr) {
  return arr.reduce((a, b) => a + b, 0) / arr.length
}

function stddev(arr) {
  const m = mean(arr)
  return Math.sqrt(arr.reduce((a, b) => a + (b - m) ** 2, 0) / arr.length)
}

function zscore(value, arr) {
  const m = mean(arr)
  const s = stddev(arr)
  return s === 0 ? 0 : (value - m) / s
}

function clamp(v, min = -3, max = 3) {
  return Math.max(min, Math.min(max, v))
}

// ── RSI ────────────────────────────────────────────────────────────────────
export function calcRSI(closes, period = 14) {
  if (closes.length < period + 1) return 50
  let gains = 0, losses = 0
  for (let i = closes.length - period; i < closes.length; i++) {
    const diff = closes[i] - closes[i - 1]
    if (diff > 0) gains += diff
    else losses -= diff
  }
  const rs = losses === 0 ? 100 : gains / losses
  return 100 - 100 / (1 + rs)
}

// ── EMA ────────────────────────────────────────────────────────────────────
export function calcEMA(closes, period) {
  if (closes.length < period) return closes[closes.length - 1]
  const k = 2 / (period + 1)
  let ema = mean(closes.slice(0, period))
  for (let i = period; i < closes.length; i++) {
    ema = closes[i] * k + ema * (1 - k)
  }
  return ema
}

// ── Bollinger Bands ────────────────────────────────────────────────────────
export function calcBollingerBands(closes, period = 20, stdMult = 2) {
  if (closes.length < period) return { upper: 0, middle: 0, lower: 0, pct: 0.5 }
  const slice = closes.slice(-period)
  const middle = mean(slice)
  const sd = stddev(slice)
  const upper = middle + stdMult * sd
  const lower = middle - stdMult * sd
  const current = closes[closes.length - 1]
  const pct = upper === lower ? 0.5 : (current - lower) / (upper - lower)
  return { upper, middle, lower, pct }
}

// ── Individual Factors ─────────────────────────────────────────────────────

/**
 * Momentum factor: 12-1 month return z-scored.
 * Positive = strong uptrend. Range: -3 to +3.
 */
export function momentumFactor(bars) {
  if (bars.length < 60) return 0
  const current = bars[bars.length - 1].c
  const oneMonth = bars[bars.length - 20].c
  const twelveMonth = bars[bars.length - 60].c  // use 60 days as proxy for free tier
  const momentum = (oneMonth - twelveMonth) / twelveMonth
  // Cross-sectional z-score requires peer data; use absolute threshold instead
  return clamp(momentum * 10) // scale: 10% move = z~1
}

/**
 * Mean-reversion factor: RSI-based contrarian signal.
 * Negative when overbought (sell), positive when oversold (buy).
 */
export function meanReversionFactor(bars) {
  const closes = bars.map(b => b.c)
  const rsi = calcRSI(closes, 14)
  // Convert RSI → signal: RSI<30 = strong buy (+3), RSI>70 = strong sell (-3)
  return clamp(((50 - rsi) / 50) * 3)
}

/**
 * Volume anomaly factor: unusual volume = institutional activity.
 * Positive volume spike on up day = buy signal.
 */
export function volumeFactor(bars) {
  if (bars.length < 20) return 0
  const recentBars = bars.slice(-20)
  const avgVol = mean(recentBars.map(b => b.v))
  const lastBar = bars[bars.length - 1]
  const volRatio = lastBar.v / (avgVol || 1)
  const priceDir = lastBar.c > lastBar.o ? 1 : -1
  return clamp((volRatio - 1) * priceDir * 1.5)
}

/**
 * Volatility regime factor: low volatility favors momentum continuation.
 * Positive in calm markets, negative in choppy/fearful markets.
 */
export function volatilityFactor(bars) {
  if (bars.length < 20) return 0
  const returns = []
  for (let i = 1; i < bars.length; i++) {
    returns.push((bars[i].c - bars[i - 1].c) / bars[i - 1].c)
  }
  const vol20 = stddev(returns.slice(-20))
  const vol5 = stddev(returns.slice(-5))
  const volRatio = vol5 / (vol20 || 0.001)
  // Low recent vol vs historical = risk-on signal (+)
  return clamp((1 - volRatio) * 2)
}

/**
 * Trend factor: price vs VWAP and 20-day MA.
 * Positive = trading above key levels.
 */
export function trendFactor(bars) {
  if (bars.length < 20) return 0
  const current = bars[bars.length - 1].c
  const vwap = bars[bars.length - 1].vw || current
  const ma20 = mean(bars.slice(-20).map(b => b.c))
  const vsVwap = (current - vwap) / (vwap || 1)
  const vsMa = (current - ma20) / (ma20 || 1)
  return clamp((vsVwap + vsMa) * 10)
}

// ── Composite Signal ───────────────────────────────────────────────────────

/**
 * Generate composite signal for a single asset.
 *
 * @param {Array}  bars    - OHLCV bars array
 * @param {Object} weights - factor weights from RL agent
 * @returns {Object} { score, signal, factors, confidence }
 */
export function generateSignal(bars, weights = DEFAULT_WEIGHTS) {
  if (!bars || bars.length < 20) {
    return { score: 0, signal: 'HOLD', factors: {}, confidence: 0 }
  }

  const factors = {
    momentum: momentumFactor(bars),
    meanReversion: meanReversionFactor(bars),
    volume: volumeFactor(bars),
    volatility: volatilityFactor(bars),
    trend: trendFactor(bars),
  }

  // Weighted sum
  const weightSum = Object.values(weights).reduce((a, b) => a + Math.abs(b), 0)
  let score = 0
  for (const [key, val] of Object.entries(factors)) {
    score += (weights[key] || 0) * val
  }
  score = weightSum > 0 ? score / weightSum : 0  // normalize
  score = clamp(score, -1, 1)

  // Confidence = agreement between factors (inverse of variance)
  const factorVals = Object.values(factors)
  const factorStd = stddev(factorVals)
  const confidence = Math.max(0, Math.min(1, 1 - factorStd / 3))

  const signal = score > 0.15 ? 'BUY' : score < -0.15 ? 'SELL' : 'HOLD'

  return { score, signal, factors, confidence }
}

export const DEFAULT_WEIGHTS = {
  momentum: 0.3,
  meanReversion: 0.15,
  volume: 0.2,
  volatility: 0.15,
  trend: 0.2,
}
