/**
 * trainer.js — RL training loop
 *
 * Reward (RAG score) composite:
 *   - Backtest Sharpe ratio  (60%)
 *   - Backtest Calmar ratio  (25%)
 *   - Total return           (15%)
 *
 * Model quality scoring is computed ONCE before training starts
 * (not per-episode) to avoid blocking the main thread.
 */

import { agent } from './agent.js'
import { backtestPortfolio } from '../backtest/backtester.js'

let running = false
let episodeCount = 0

/**
 * Lightweight RAG score — only uses backtest metrics.
 * Heavy model quality is precomputed and passed in as a constant bonus.
 */
function computeRAG(backtestResult, modelQualityBonus = 0) {
  const sharpe  = Math.max(-2, Math.min(3,  backtestResult.sharpe      || 0))
  const calmar  = Math.max(-1, Math.min(2,  backtestResult.calmar      || 0))
  const ret     = Math.max(-0.5, Math.min(1, backtestResult.totalReturn || 0))
  return sharpe * 0.60 + calmar * 0.25 + ret * 0.15 + modelQualityBonus * 0.10
}

/**
 * Detect current market regime from bars (lightweight — no model calls).
 * Returns algo mode: 'CEM' | 'PPO' | 'SAC'
 */
function pickAlgo(backtestResult) {
  const { sharpe = 0, maxDrawdown = 0 } = backtestResult
  if (sharpe > 1.0 && maxDrawdown < 0.10) return 'PPO'   // strong trend, low dd
  if (sharpe > 0.5 || maxDrawdown < 0.15) return 'SAC'   // moderate
  return 'CEM'                                            // high uncertainty
}

export async function trainEpisodes(barsMap, episodes, onEpisode, onDone) {
  if (running) return
  running = true
  episodeCount = 0

  const tickers = Object.keys(barsMap)
  if (tickers.length === 0) {
    running = false
    if (onDone) onDone({ bestWeights: agent.bestWeights, bestScore: agent.bestScore, episodes: 0, currentAlgo: 'CEM' })
    return
  }

  try {
    for (let i = 0; i < episodes && running; i++) {
      // 1. Detect regime from first ticker bars (lightweight)
      const firstBars = barsMap[tickers[0]] || []
      const regime = agent.detectRegime(firstBars)

      // 2. Sample weights
      const weights = agent.sampleRegimeWeights(regime)

      // 3. Backtest — the only heavy computation per episode
      let backtestResult
      try {
        backtestResult = backtestPortfolio(barsMap, weights)
      } catch (e) {
        console.warn('[Trainer] backtestPortfolio error:', e.message)
        backtestResult = { sharpe: 0, calmar: 0, totalReturn: 0, maxDrawdown: 0, ragScore: 0, winRate: 0, tradeCount: 0 }
      }

      // 4. RAG score (lightweight, no per-episode model calls)
      const ragScore = computeRAG(backtestResult)

      // 5. Record episode in agent
      agent.recordEpisode(weights, ragScore, regime, {
        sharpe: backtestResult.sharpe,
        calmar: backtestResult.calmar,
      })

      episodeCount++
      const algo = pickAlgo(backtestResult)

      // 6. Notify UI — null-safe
      if (onEpisode) {
        try {
          onEpisode({
            episode:        episodeCount,
            weights,
            result: {
              ...backtestResult,
              ragScore,
              algo,
              newsCount:    0,
              modelQuality: 0,
            },
            regime,
            agentState:    agent.getProgress(),
            currentWeights: agent.weights,
            bestWeights:   agent.bestWeights,
            bestScore:     agent.bestScore,
          })
        } catch (e) {
          console.warn('[Trainer] onEpisode callback error:', e.message)
        }
      }

      // Yield to browser every episode so UI stays responsive
      await new Promise(r => setTimeout(r, 0))
    }
  } catch (e) {
    console.error('[Trainer] training loop error:', e)
  }

  running = false

  if (onDone) {
    try {
      onDone({
        bestWeights:  agent.bestWeights,
        bestScore:    agent.bestScore,
        episodes:     episodeCount,
        currentAlgo:  pickAlgo({ sharpe: agent.bestScore }),
      })
    } catch (e) {
      console.warn('[Trainer] onDone callback error:', e.message)
    }
  }
}

export function stopTraining() { running = false }
