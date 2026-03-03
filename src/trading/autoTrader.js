/**
 * autoTrader.js — Automated signal-based execution engine
 * Runs on a timer, checks signals, executes trades, monitors exits.
 */

import { checkExitSignal, checkPortfolioHeat, EXIT_PRESETS } from './exitStrategy.js'

export const AUTO_TRADE_DEFAULTS = {
  enabled: false,
  minScore: 0.25,         // min signal score to auto-buy
  minConfidence: 0.55,    // min confidence
  maxPositions: 8,        // max concurrent positions
  positionSizePct: 0.08,  // 8% of portfolio per trade
  exitPreset: 'moderate',
  requireMultiModel: true, // require 2+ models to agree
  checkIntervalMs: 60000,  // check every 60s
}

/**
 * Evaluate whether to auto-trade a given asset.
 * Returns { action, reason } or null.
 */
export function evaluateAutoTrade(ticker, signal, bars, portfolio, prices, settings) {
  const s = { ...AUTO_TRADE_DEFAULTS, ...settings }
  const px = prices[ticker]?.price
  if (!px || px <= 0) return null

  const pos = portfolio.positions[ticker]
  const numPositions = Object.keys(portfolio.positions).length

  // ── Check exits first ─────────────────────────────────────────────────
  if (pos) {
    const preset = EXIT_PRESETS[s.exitPreset] || EXIT_PRESETS.moderate
    const exit = checkExitSignal(pos, px, signal, bars, preset)
    if (exit?.shouldExit) {
      return { action: 'CLOSE', reason: exit.reason, urgency: exit.urgency, pnlPct: exit.pnlPct }
    }
  }

  // ── Portfolio heat check ───────────────────────────────────────────────
  const heat = checkPortfolioHeat(portfolio, prices, 0.07)
  if (heat?.shouldReduce && pos) {
    return { action: 'CLOSE', reason: heat.reason, urgency: 'HIGH', pnlPct: null }
  }

  // ── Entry logic ───────────────────────────────────────────────────────
  if (!pos && signal.score > s.minScore && signal.confidence > s.minConfidence) {
    // Don't add more positions if at max
    if (numPositions >= s.maxPositions) return null

    // Check cash available
    const tradeValue = (portfolio.cash + Object.values(portfolio.positions).reduce((a,p)=>a+p.shares*(prices[p.ticker]?.price||p.avgPrice),0)) * s.positionSizePct
    if (portfolio.cash < tradeValue * 0.8) return null

    return {
      action: 'BUY',
      reason: `Auto: score=${signal.score.toFixed(3)}, conf=${(signal.confidence*100).toFixed(0)}%`,
      tradeValue,
      urgency: 'AUTO',
    }
  }

  // ── Auto-sell on strong SELL signal ───────────────────────────────────
  if (pos && signal.signal === 'SELL' && signal.score < -s.minScore && signal.confidence > s.minConfidence) {
    return { action: 'CLOSE', reason: `Auto SELL signal: score=${signal.score.toFixed(3)}`, urgency: 'AUTO', pnlPct: null }
  }

  return null
}
