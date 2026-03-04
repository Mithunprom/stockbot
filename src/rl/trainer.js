/**
 * trainer.js — Enhanced RL training loop
 *
 * Now uses ALL models in the reward function:
 *   - Technical signals (7 factors incl. FF Alpha + TCN alignment)
 *   - Fama-French 5F alpha/beta/rmw
 *   - Temporal CNN pattern score
 *   - Ensemble agreement bonus
 *
 * The reward (RAG score) is a composite of:
 *   - Backtest Sharpe ratio (50%)
 *   - Backtest Calmar ratio (20%)
 *   - Model agreement score (20%) — how often all models agreed with trades
 *   - FF5 quality score (10%) — did we pick high-alpha, high-RMW stocks?
 */

import { agent } from './agent.js'
import { backtestPortfolio } from '../backtest/backtester.js'
import { computeFF5 } from '../models/famaFrench.js'
import { computeTCN } from '../models/temporalCNN.js'
import { generateSignal } from '../signals/signals.js'

let running = false
let episodeCount = 0

/**
 * Compute a multi-model quality score for a set of bars.
 * Returns 0..1 representing how much models agree and quality of picks.
 */
function computeModelQuality(barsMap, weights) {
  let totalScore = 0, count = 0

  for (const [ticker, bars] of Object.entries(barsMap)) {
    if (!bars || bars.length < 60) continue

    const sig = generateSignal(bars, weights)
    const ff5 = computeFF5(bars, null)
    const tcn = computeTCN(bars)

    if (!sig || !ff5 || !tcn) continue

    // Model agreement: do all three point same direction?
    const signals = [sig.signal, ff5.signal, tcn.signal]
    const buyVotes = signals.filter(s => s === 'BUY').length
    const sellVotes = signals.filter(s => s === 'SELL').length
    const agreementScore = Math.max(buyVotes, sellVotes) / 3

    // FF5 quality: prefer high alpha + high RMW (profitability)
    const ff5Quality = Math.max(0, Math.min(1,
      (ff5.alpha * 50 + 0.5) * 0.5 +  // alpha contribution
      (ff5.rmw * 0.5 + 0.5) * 0.3 +   // profitability
      (ff5.beta > 0.5 && ff5.beta < 1.5 ? 0.2 : 0) // reasonable beta
    ))

    // TCN pattern quality: prefer strong trends
    const tcnQuality = Math.abs(tcn.maAlignment || 0)

    // Ensemble quality per ticker
    const tickerScore = agreementScore * 0.5 + ff5Quality * 0.3 + tcnQuality * 0.2
    totalScore += tickerScore
    count++
  }

  return count > 0 ? totalScore / count : 0
}

/**
 * Enhanced RAG score incorporating all models.
 */
function computeEnhancedRAG(backtestResult, modelQuality) {
  const { sharpe, calmar, totalReturn } = backtestResult

  // Clamp individual components
  const sharpeClamped = Math.max(-2, Math.min(3, sharpe || 0))
  const calmarClamped = Math.max(-1, Math.min(2, calmar || 0))
  const returnClamped = Math.max(-0.5, Math.min(1, totalReturn || 0))
  const qualityClamped = Math.max(0, Math.min(1, modelQuality))

  // Weighted composite
  const ragScore = (
    sharpeClamped * 0.45 +
    calmarClamped * 0.20 +
    returnClamped * 0.15 +
    qualityClamped * 0.20  // model agreement reward
  )

  return ragScore
}

export async function trainEpisodes(barsMap, episodes, onEpisode, onDone) {
  if (running) return
  running = true
  episodeCount = 0

  // Pre-compute model quality for reference (expensive, do once)
  // Then compare per episode

  for (let i = 0; i < episodes && running; i++) {
    // 1. Sample weights — regime-aware
    const firstBars = Object.values(barsMap)[0] || []
    const regime = agent.detectRegime(firstBars)
    const weights = agent.sampleRegimeWeights(regime)

    // 2. Backtest with these weights
    const backtestResult = backtestPortfolio(barsMap, weights)

    // 3. Compute multi-model quality with these weights
    const modelQuality = computeModelQuality(barsMap, weights)

    // 4. Enhanced RAG score using ALL models
    const enhancedRAG = computeEnhancedRAG(backtestResult, modelQuality)

    // 5. Record with model scores
    agent.recordEpisode(weights, enhancedRAG, regime, {
      sharpe: backtestResult.sharpe,
      calmar: backtestResult.calmar,
      modelQuality,
    })

    episodeCount++

    if (onEpisode) {
      onEpisode({
        episode: episodeCount,
        weights,
        result: { ...backtestResult, ragScore: enhancedRAG, modelQuality },
        regime,
        agentState: agent.getProgress(),
        currentWeights: agent.weights,
        bestWeights: agent.bestWeights,
        bestScore: agent.bestScore,
      })
    }

    // Yield to browser every 3 episodes
    if (i % 3 === 0) await new Promise(r => setTimeout(r, 0))
  }

  running = false

  if (onDone) {
    onDone({
      bestWeights: agent.bestWeights,
      bestScore: agent.bestScore,
      episodes: episodeCount,
    })
  }
}

export function stopTraining() { running = false }
