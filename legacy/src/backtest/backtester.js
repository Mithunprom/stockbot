/**
 * backtester.js — Vectorized backtester
 *
 * KEY FIX: signals are pre-computed with a fixed lookback window (O(n)),
 * NOT recomputed from scratch on a growing slice each bar (O(n²/n³)).
 * This makes each episode ~100x faster, preventing the black screen.
 */

import { vectorizeSignals, generateSignal } from '../signals/signals.js'
import { kellySize } from '../risk/kelly.js'

const TRANSACTION_COST = 0.0005
const INITIAL_CAPITAL  = 100_000

// ── Single-asset backtest (vectorized) ───────────────────────────────────

export function backtest(bars, weights, options = {}) {
  const {
    capital    = INITIAL_CAPITAL,
    warmup     = 60,
    maxPosition = 0.25,
  } = options

  if (!bars || bars.length < warmup + 2) return emptyResult(capital)

  // PRE-COMPUTE all signals once — O(n), not O(n²)
  const signalArr = vectorizeSignals(bars, weights)

  const equity  = [capital]
  const trades  = []
  const returns = []
  let cash      = capital
  let position  = 0
  let entryPrice = 0

  for (let i = warmup; i < bars.length - 1; i++) {
    const { score, signal } = signalArr[i]   // O(1) lookup

    const currentClose = bars[i].c
    const nextOpen     = bars[i + 1].o
    const portfolioVal = cash + position * currentClose
    const targetShares = Math.floor(
      (portfolioVal * Math.min(Math.abs(kellySize(score, portfolioVal)), maxPosition)) / nextOpen
    )

    if (signal === 'BUY' && position <= 0) {
      if (position < 0) {
        cash += (-position) * nextOpen * (1 - TRANSACTION_COST)
        trades.push({ t: bars[i+1].t, type:'COVER', price:nextOpen, shares:-position,
          pnl: (-position) * (nextOpen - entryPrice) * -1 })
        position = 0
      }
      if (targetShares > 0) {
        const cost = targetShares * nextOpen * (1 + TRANSACTION_COST)
        if (cost <= cash) {
          cash -= cost; position = targetShares; entryPrice = nextOpen
          trades.push({ t:bars[i+1].t, type:'BUY', price:nextOpen, shares:targetShares })
        }
      }
    } else if (signal === 'SELL' && position > 0) {
      const proceeds = position * nextOpen * (1 - TRANSACTION_COST)
      trades.push({ t:bars[i+1].t, type:'SELL', price:nextOpen, shares:position,
        pnl: proceeds - entryPrice * position })
      cash += proceeds; position = 0; entryPrice = 0
    } else if (position !== 0) {
      // Dynamic stop-loss: exit if down >7%
      const unreal = position > 0
        ? (currentClose - entryPrice) / entryPrice
        : (entryPrice - currentClose) / entryPrice
      if (unreal < -0.07) {
        const proceeds = Math.abs(position) * nextOpen * (1 - TRANSACTION_COST)
        const pnl = position > 0 ? proceeds - entryPrice * position
          : entryPrice * (-position) - (-position) * nextOpen
        cash += proceeds
        trades.push({ t:bars[i+1].t, type:'STOP', price:nextOpen, shares:Math.abs(position), pnl })
        position = 0; entryPrice = 0
      }
    }

    const newVal = cash + position * bars[i + 1].c
    returns.push((newVal - equity[equity.length - 1]) / equity[equity.length - 1])
    equity.push(newVal)
  }

  if (position !== 0) {
    cash += position * bars[bars.length - 1].c * (1 - TRANSACTION_COST)
  }

  return computeMetrics(equity, returns, trades, capital, cash)
}

// ── Multi-asset portfolio backtest ────────────────────────────────────────

export function backtestPortfolio(barsMap, weights) {
  const tickers = Object.keys(barsMap)
  if (!tickers.length) return emptyResult(INITIAL_CAPITAL)

  const alloc = INITIAL_CAPITAL / tickers.length
  const results = {}
  let totalFinal = 0

  for (const ticker of tickers) {
    results[ticker] = backtest(barsMap[ticker], weights, { capital: alloc })
    totalFinal += results[ticker].finalValue
  }

  const maxLen  = Math.max(...Object.values(results).map(r => r.equity.length))
  const combined = Array(maxLen).fill(0)
  for (const r of Object.values(results)) {
    for (let i = 0; i < maxLen; i++)
      combined[i] += r.equity[Math.min(i, r.equity.length - 1)]
  }

  const combinedReturns = combined.slice(1).map((v, i) => (v - combined[i]) / combined[i])
  const allTrades = Object.values(results).flatMap(r => r.trades)

  return {
    ...computeMetrics(combined, combinedReturns, allTrades, INITIAL_CAPITAL, totalFinal),
    perAsset: results,
    tickers,
  }
}

// ── Metrics ───────────────────────────────────────────────────────────────

function computeMetrics(equity, returns, trades, initialCapital, finalValue) {
  const totalReturn  = (finalValue - initialCapital) / initialCapital
  const annualReturn = Math.pow(1 + totalReturn, 252 / Math.max(returns.length, 1)) - 1

  const meanRet = returns.reduce((a, b) => a + b, 0) / (returns.length || 1)
  const retStd  = Math.sqrt(returns.reduce((a, b) => a + (b - meanRet) ** 2, 0) / (returns.length || 1))
  const sharpe  = retStd === 0 ? 0 : (meanRet / retStd) * Math.sqrt(252)

  let peak = -Infinity, maxDD = 0
  for (const v of equity) {
    if (v > peak) peak = v
    const dd = (peak - v) / peak
    if (dd > maxDD) maxDD = dd
  }

  const calmar   = maxDD === 0 ? 0 : annualReturn / maxDD
  const ragScore = 0.6 * Math.max(-3, Math.min(3, sharpe))
                 + 0.4 * Math.max(-3, Math.min(3, calmar))

  const closedTrades = trades.filter(t => t.pnl != null)
  const winRate = closedTrades.length === 0 ? 0
    : closedTrades.filter(t => t.pnl > 0).length / closedTrades.length

  return {
    equity, trades, totalReturn, annualReturn,
    sharpe, maxDrawdown: maxDD, calmar, ragScore,
    winRate, tradeCount: closedTrades.length, finalValue,
  }
}

function emptyResult(capital) {
  return {
    equity: [capital], trades: [],
    totalReturn: 0, annualReturn: 0,
    sharpe: 0, maxDrawdown: 0, calmar: 0, ragScore: 0,
    winRate: 0, tradeCount: 0, finalValue: capital,
  }
}

// ── CSV export ────────────────────────────────────────────────────────────

export function equityCurveToCSV(equityCurve) {
  const rows = ['date,equity,cash,positions,observationMode']
  for (const pt of (equityCurve || [])) {
    rows.push([
      pt.date || '',
      (pt.equity  || 0).toFixed(2),
      (pt.cash    || 0).toFixed(2),
      pt.positions != null ? pt.positions : '',
      pt.observationMode ? 'true' : 'false',
    ].join(','))
  }
  return rows.join('\n')
}
