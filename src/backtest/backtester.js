/**
 * backtester.js
 * Vectorized backtester for signal evaluation on real OHLCV data.
 *
 * Design principles:
 * - Zero lookahead bias: signals computed only on data available at bar T,
 *   executed at open of bar T+1
 * - Realistic transaction costs: 0.05% per side (institutional estimate)
 * - Position sizing via Kelly criterion (from risk module)
 * - Tracks: equity curve, drawdowns, trade log, Sharpe, Calmar
 */

import { generateSignal } from '../signals/signals.js'
import { kellySize } from '../risk/kelly.js'

const TRANSACTION_COST = 0.0005  // 0.05% per side
const INITIAL_CAPITAL = 100_000

// ── Core Backtest ──────────────────────────────────────────────────────────

/**
 * Run a backtest over historical bars for a single asset.
 *
 * @param {Array}  bars      - OHLCV bars [{t,o,h,l,c,v}]
 * @param {Object} weights   - factor weights from RL agent
 * @param {Object} options
 * @returns {Object} BacktestResult
 */
export function backtest(bars, weights, options = {}) {
  const {
    capital = INITIAL_CAPITAL,
    warmup = 60,        // bars before first trade
    maxPosition = 0.25  // max 25% in any single position
  } = options

  if (bars.length < warmup + 2) {
    return emptyResult(capital)
  }

  const equity = [capital]
  const trades = []
  const returns = []
  let cash = capital
  let position = 0    // shares held
  let entryPrice = 0

  for (let i = warmup; i < bars.length - 1; i++) {
    const historyBars = bars.slice(0, i + 1)  // no lookahead
    const { score, signal } = generateSignal(historyBars, weights)

    const currentClose = bars[i].c
    const nextOpen = bars[i + 1].o
    const portfolioValue = cash + position * currentClose

    // Kelly-sized position
    const targetAlloc = kellySize(score, portfolioValue) * Math.sign(score)
    const targetShares = Math.floor((portfolioValue * Math.min(Math.abs(targetAlloc), maxPosition)) / nextOpen)

    // Execute at next open
    if (signal === 'BUY' && position <= 0) {
      // Close short if any
      if (position < 0) {
        const closeProfit = -position * nextOpen
        cash += closeProfit * (1 - TRANSACTION_COST)
        trades.push({ t: bars[i + 1].t, type: 'COVER', price: nextOpen, shares: -position, pnl: closeProfit - entryPrice * -position })
        position = 0
      }
      // Open long
      const shares = targetShares
      const cost = shares * nextOpen * (1 + TRANSACTION_COST)
      if (cost <= cash && shares > 0) {
        cash -= cost
        position = shares
        entryPrice = nextOpen
        trades.push({ t: bars[i + 1].t, type: 'BUY', price: nextOpen, shares })
      }
    } else if (signal === 'SELL' && position >= 0) {
      // Close long if any
      if (position > 0) {
        const proceeds = position * nextOpen * (1 - TRANSACTION_COST)
        const pnl = proceeds - entryPrice * position
        cash += proceeds
        trades.push({ t: bars[i + 1].t, type: 'SELL', price: nextOpen, shares: position, pnl })
        position = 0
        entryPrice = 0
      }
    } else if (signal === 'HOLD' && position !== 0) {
      // Dynamic stop: exit if position down >7%
      const unrealized = position > 0
        ? (currentClose - entryPrice) / entryPrice
        : (entryPrice - currentClose) / entryPrice
      if (unrealized < -0.07) {
        const proceeds = Math.abs(position) * nextOpen * (1 - TRANSACTION_COST)
        const pnl = position > 0
          ? proceeds - entryPrice * position
          : entryPrice * -position - proceeds
        cash += position > 0 ? proceeds : -position * nextOpen
        trades.push({ t: bars[i + 1].t, type: 'STOP', price: nextOpen, shares: Math.abs(position), pnl })
        position = 0
        entryPrice = 0
      }
    }

    const newValue = cash + position * bars[i + 1].c
    returns.push((newValue - equity[equity.length - 1]) / equity[equity.length - 1])
    equity.push(newValue)
  }

  // Close any open position at last bar
  if (position !== 0) {
    const lastPrice = bars[bars.length - 1].c
    cash += position * lastPrice * (1 - TRANSACTION_COST)
    position = 0
  }

  const finalValue = cash
  return computeMetrics(equity, returns, trades, capital, finalValue)
}

// ── Multi-Asset Portfolio Backtest ─────────────────────────────────────────

/**
 * Backtest across multiple assets, equal-weight allocation.
 * @param {Object} barsMap - { TICKER: bars[] }
 * @param {Object} weights - factor weights
 * @returns {Object} AggregatedBacktestResult
 */
export function backtestPortfolio(barsMap, weights) {
  const tickers = Object.keys(barsMap)
  if (tickers.length === 0) return emptyResult(INITIAL_CAPITAL)

  const alloc = INITIAL_CAPITAL / tickers.length
  const results = {}
  let totalFinal = 0

  for (const ticker of tickers) {
    results[ticker] = backtest(barsMap[ticker], weights, { capital: alloc })
    totalFinal += results[ticker].finalValue
  }

  // Aggregate equity curves (pad shorter ones)
  const maxLen = Math.max(...Object.values(results).map(r => r.equity.length))
  const combined = Array(maxLen).fill(0)
  for (const r of Object.values(results)) {
    for (let i = 0; i < maxLen; i++) {
      combined[i] += r.equity[Math.min(i, r.equity.length - 1)]
    }
  }

  const combinedReturns = combined.slice(1).map((v, i) => (v - combined[i]) / combined[i])
  const allTrades = Object.values(results).flatMap(r => r.trades)

  return {
    ...computeMetrics(combined, combinedReturns, allTrades, INITIAL_CAPITAL, totalFinal),
    perAsset: results,
    tickers
  }
}

// ── Metrics ────────────────────────────────────────────────────────────────

function computeMetrics(equity, returns, trades, initialCapital, finalValue) {
  const totalReturn = (finalValue - initialCapital) / initialCapital
  const annualReturn = Math.pow(1 + totalReturn, 252 / Math.max(returns.length, 1)) - 1

  // Sharpe ratio (annualized, risk-free = 0 for simplicity)
  const meanRet = returns.reduce((a, b) => a + b, 0) / (returns.length || 1)
  const retStd = Math.sqrt(returns.reduce((a, b) => a + (b - meanRet) ** 2, 0) / (returns.length || 1))
  const sharpe = retStd === 0 ? 0 : (meanRet / retStd) * Math.sqrt(252)

  // Max drawdown
  let peak = -Infinity, maxDD = 0
  for (const v of equity) {
    if (v > peak) peak = v
    const dd = (peak - v) / peak
    if (dd > maxDD) maxDD = dd
  }

  // Calmar ratio
  const calmar = maxDD === 0 ? 0 : annualReturn / maxDD

  // Risk-adjusted growth score (optimization target for RL)
  // Combines Sharpe and Calmar, penalizes drawdown
  const ragScore = 0.6 * Math.max(-3, Math.min(3, sharpe)) + 0.4 * Math.max(-3, Math.min(3, calmar))

  // Win rate
  const closedTrades = trades.filter(t => t.pnl !== undefined)
  const winRate = closedTrades.length === 0 ? 0
    : closedTrades.filter(t => t.pnl > 0).length / closedTrades.length

  return {
    equity,
    trades,
    totalReturn,
    annualReturn,
    sharpe,
    maxDrawdown: maxDD,
    calmar,
    ragScore,    // RL reward signal
    winRate,
    tradeCount: closedTrades.length,
    finalValue
  }
}

function emptyResult(capital) {
  return {
    equity: [capital],
    trades: [],
    totalReturn: 0,
    annualReturn: 0,
    sharpe: 0,
    maxDrawdown: 0,
    calmar: 0,
    ragScore: 0,
    winRate: 0,
    tradeCount: 0,
    finalValue: capital
  }
}
