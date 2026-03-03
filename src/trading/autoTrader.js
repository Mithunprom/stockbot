/**
 * autoTrader.js — Automated trading engine
 *
 * Defaults: always ON, 10% take-profit, 7% stop-loss
 * All thresholds are adjustable in the app UI.
 */

import { checkExitSignal, checkPortfolioHeat, EXIT_PRESETS } from './exitStrategy.js'

export const AUTO_TRADE_DEFAULTS = {
  enabled: true,              // ← always on by default
  minScore: 0.25,
  minConfidence: 0.55,
  maxPositions: 8,
  positionSizePct: 0.08,      // 8% of portfolio per position
  takeProfitPct: 0.10,        // 10% take-profit  (user-adjustable)
  stopLossPct: 0.07,          // 7%  stop-loss    (user-adjustable)
  trailingStopPct: 0.05,      // 5%  trailing stop from peak
  requireMultiModel: true,
  checkIntervalMs: 60000,
}

/**
 * Evaluate whether to auto-trade a given asset.
 * Uses inline take-profit / stop-loss from settings (not EXIT_PRESETS)
 * so UI sliders directly control live exit behavior.
 */
export function evaluateAutoTrade(ticker, signal, ensemble, bars, portfolio, prices, settings) {
  const s = { ...AUTO_TRADE_DEFAULTS, ...settings }
  const px = prices[ticker]?.price
  if (!px || px <= 0) return null

  const pos = portfolio.positions[ticker]
  const numPositions = Object.keys(portfolio.positions).length

  // ── 1. Exit checks (highest priority) ─────────────────────────────────────
  if (pos) {
    const curPx = px
    const entryPx = pos.avgPrice || curPx
    const pnlPct = pos.side === 'LONG'
      ? (curPx - entryPx) / entryPx
      : (entryPx - curPx) / entryPx

    // Take-profit hit
    if (pnlPct >= s.takeProfitPct) {
      return {
        action: 'CLOSE',
        reason: `✅ Take-profit ${(s.takeProfitPct*100).toFixed(0)}% reached (+${(pnlPct*100).toFixed(1)}%)`,
        urgency: 'TAKE_PROFIT',
        pnlPct
      }
    }

    // Hard stop-loss hit
    if (pnlPct <= -s.stopLossPct) {
      return {
        action: 'CLOSE',
        reason: `🛑 Stop-loss ${(s.stopLossPct*100).toFixed(0)}% hit (${(pnlPct*100).toFixed(1)}%)`,
        urgency: 'STOP_LOSS',
        pnlPct
      }
    }

    // Trailing stop: if peak existed and price fell trailing% below it
    const peakPx = pos.highWater || entryPx
    const drawFromPeak = (peakPx - curPx) / peakPx
    if (drawFromPeak >= s.trailingStopPct) {
      return {
        action: 'CLOSE',
        reason: `📉 Trailing stop ${(s.trailingStopPct*100).toFixed(0)}% from peak (${(pnlPct*100).toFixed(1)}%)`,
        urgency: 'TRAIL_STOP',
        pnlPct
      }
    }

    // Strong SELL signal override
    if (signal.signal === 'SELL' && signal.score < -s.minScore && signal.confidence > s.minConfidence) {
      return {
        action: 'CLOSE',
        reason: `📡 Strong SELL signal (score ${signal.score.toFixed(3)}, conf ${(signal.confidence*100).toFixed(0)}%)`,
        urgency: 'SIGNAL',
        pnlPct
      }
    }
  }

  // ── 2. Portfolio heat — reduce if portfolio down 7% ───────────────────────
  const heat = checkPortfolioHeat(portfolio, prices, s.stopLossPct)
  if (heat?.shouldReduce && pos) {
    return { action: 'CLOSE', reason: `🌡️ Portfolio heat: ${heat.reason}`, urgency: 'HIGH', pnlPct: null }
  }

  // ── 3. Entry ───────────────────────────────────────────────────────────────
  if (pos) return null
  if (signal.score <= s.minScore || signal.confidence <= s.minConfidence) return null
  if (numPositions >= s.maxPositions) return null

  // requireMultiModel: at least 2 of ensemble models must agree BUY
  if (s.requireMultiModel && ensemble) {
    const buyVotes = Object.values(ensemble.models || {}).filter(m => m?.signal === 'BUY').length
    if (buyVotes < 2) return null
  }

  // Cash check
  let positionsValue = 0
  for (const [pt, pd] of Object.entries(portfolio.positions)) {
    positionsValue += pd.shares * (prices[pt]?.price || pd.avgPrice)
  }
  const totalValue = portfolio.cash + positionsValue
  const tradeValue = totalValue * s.positionSizePct
  if (portfolio.cash < tradeValue * 0.8) return null

  const modelAgreement = ensemble
    ? Object.values(ensemble.models || {}).filter(m => m?.signal === 'BUY').length
    : null

  return {
    action: 'BUY',
    reason: `Score ${signal.score.toFixed(3)} · Conf ${(signal.confidence*100).toFixed(0)}%${modelAgreement ? ` · ${modelAgreement}/4 models` : ''}`,
    urgency: 'AUTO',
    tradeValue,
  }
}
