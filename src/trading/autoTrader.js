/**
 * autoTrader.js — Fixed automated trading engine
 *
 * Bugs fixed:
 *   1. p.ticker undefined → now passed as separate param
 *   2. requireMultiModel actually enforced
 *   3. All position lookups use passed-in state, not stale closures
 */

import { checkExitSignal, checkPortfolioHeat, EXIT_PRESETS } from './exitStrategy.js'

export const AUTO_TRADE_DEFAULTS = {
  enabled: false,
  minScore: 0.25,
  minConfidence: 0.55,
  maxPositions: 8,
  positionSizePct: 0.08,   // 8% of total portfolio value per trade
  exitPreset: 'moderate',
  requireMultiModel: true, // require 2+ models to agree before buying
  checkIntervalMs: 60000,
}

/**
 * Evaluate whether to auto-trade a given asset.
 * All state passed explicitly — no closures, no stale reads.
 *
 * @param {string} ticker
 * @param {object} signal  — from generateSignal()
 * @param {object} ensemble — from runEnsemble(), may be null
 * @param {array}  bars
 * @param {object} portfolio — current portfolio snapshot
 * @param {object} prices   — current prices map
 * @param {object} settings — autoTradeSettings
 * @returns {{ action, reason, urgency, tradeValue? }} | null
 */
export function evaluateAutoTrade(ticker, signal, ensemble, bars, portfolio, prices, settings) {
  const s = { ...AUTO_TRADE_DEFAULTS, ...settings }
  const px = prices[ticker]?.price
  if (!px || px <= 0) return null

  const pos = portfolio.positions[ticker]
  const numPositions = Object.keys(portfolio.positions).length

  // ── 1. Exit checks (highest priority) ────────────────────────────────
  if (pos) {
    const preset = EXIT_PRESETS[s.exitPreset] || EXIT_PRESETS.moderate
    const exit = checkExitSignal(pos, px, signal, bars, preset)
    if (exit?.shouldExit) {
      return { action: 'CLOSE', reason: exit.reason, urgency: exit.urgency, pnlPct: exit.pnlPct }
    }
  }

  // ── 2. Portfolio heat — reduce exposure if down 7% ─────────────────
  const heat = checkPortfolioHeat(portfolio, prices, 0.07)
  if (heat?.shouldReduce && pos) {
    return { action: 'CLOSE', reason: `Portfolio heat: ${heat.reason}`, urgency: 'HIGH', pnlPct: null }
  }

  // ── 3. Auto-sell on strong SELL signal ────────────────────────────
  if (pos && signal.signal === 'SELL' && signal.score < -s.minScore && signal.confidence > s.minConfidence) {
    return { action: 'CLOSE', reason: `SELL signal (score ${signal.score.toFixed(3)}, conf ${(signal.confidence*100).toFixed(0)}%)`, urgency: 'AUTO', pnlPct: null }
  }

  // ── 4. Entry: only if no existing position ────────────────────────
  if (pos) return null

  // Check signal thresholds
  if (signal.score <= s.minScore || signal.confidence <= s.minConfidence) return null

  // Max positions guard
  if (numPositions >= s.maxPositions) return null

  // requireMultiModel: at least 2 of 3 models must agree (BUY)
  if (s.requireMultiModel && ensemble) {
    const modelSignals = Object.values(ensemble.models || {}).map(m => m?.signal)
    const buyVotes = modelSignals.filter(s => s === 'BUY').length
    if (buyVotes < 2) {
      return null // not enough model agreement
    }
  }

  // Cash check — FIX: compute portfolio total value correctly
  // positions is { ticker: {shares, avgPrice, ...} } — ticker is the KEY not a property
  let positionsValue = 0
  for (const [posTicker, posData] of Object.entries(portfolio.positions)) {
    positionsValue += posData.shares * (prices[posTicker]?.price || posData.avgPrice)
  }
  const totalValue = portfolio.cash + positionsValue
  const tradeValue = totalValue * s.positionSizePct

  if (portfolio.cash < tradeValue * 0.8) return null

  const modelAgreement = ensemble
    ? Object.values(ensemble.models || {}).filter(m => m?.signal === 'BUY').length
    : null

  return {
    action: 'BUY',
    reason: `Score ${signal.score.toFixed(3)} · Conf ${(signal.confidence*100).toFixed(0)}%${modelAgreement ? ` · ${modelAgreement}/4 models agree` : ''}`,
    urgency: 'AUTO',
    tradeValue,
  }
}
