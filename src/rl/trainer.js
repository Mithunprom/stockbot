/**
 * trainer.js
 * Orchestrates the RL training loop:
 *   1. Agent samples candidate weights
 *   2. Backtest evaluates weights on historical data
 *   3. Agent records reward and updates
 *   4. Repeat
 *
 * Designed to run in the browser on a Web Worker in the future,
 * but currently runs synchronously with callbacks for UI updates.
 */

import { agent } from './agent.js'
import { backtestPortfolio } from '../backtest/backtester.js'

let running = false
let episodeCount = 0

/**
 * Run N training episodes.
 * @param {Object} barsMap   - { TICKER: bars[] }
 * @param {number} episodes  - number of episodes to run
 * @param {Function} onEpisode - callback({ episode, weights, result, agentState })
 * @param {Function} onDone   - callback({ bestWeights, bestScore })
 */
export async function trainEpisodes(barsMap, episodes, onEpisode, onDone) {
  if (running) return
  running = true

  for (let i = 0; i < episodes && running; i++) {
    // 1. Sample weights
    const weights = agent.sampleWeights()

    // 2. Backtest
    const result = backtestPortfolio(barsMap, weights)

    // 3. Detect regime (use first ticker as proxy)
    const firstBars = Object.values(barsMap)[0] || []
    const regime = agent.detectRegime(firstBars)

    // 4. Record reward
    agent.recordEpisode(weights, result.ragScore, regime)

    episodeCount++

    // 5. Notify UI
    if (onEpisode) {
      onEpisode({
        episode: episodeCount,
        weights,
        result,
        regime,
        agentState: agent.getProgress(),
        currentWeights: agent.weights,
        bestWeights: agent.bestWeights,
        bestScore: agent.bestScore,
      })
    }

    // Yield to browser event loop
    await new Promise(r => setTimeout(r, 0))
  }

  running = false
  if (onDone) {
    onDone({ bestWeights: agent.bestWeights, bestScore: agent.bestScore })
  }
}

export function stopTraining() {
  running = false
}

export function isTraining() {
  return running
}

export function getEpisodeCount() {
  return episodeCount
}
