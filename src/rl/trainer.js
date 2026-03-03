/**
 * trainer.js — RL training with ALL models + LLM news sentiment as RL weight
 *
 * Reward = Sharpe(40%) + Calmar(20%) + ModelAgreement+NewsAlignment(30%) + Return(10%)
 * News sentiment map { TICKER: score -1..1 } passed in from live headlines.
 */
import { agent } from './agent.js'
import { backtestPortfolio } from '../backtest/backtester.js'
import { computeFF5 } from '../models/famaFrench.js'
import { computeTCN } from '../models/temporalCNN.js'
import { generateSignal } from '../signals/signals.js'

let running = false, episodeCount = 0

function computeModelQuality(barsMap, weights, newsSentimentMap = {}) {
  let totalScore = 0, count = 0
  for (const [ticker, bars] of Object.entries(barsMap)) {
    if (!bars || bars.length < 60) continue
    const sig = generateSignal(bars, weights)
    const ff5 = computeFF5(bars, null)
    const tcn = computeTCN(bars)
    if (!sig || !ff5 || !tcn) continue
    const allSigs = [sig.signal, ff5.signal, tcn.signal]
    const buyVotes = allSigs.filter(s => s === 'BUY').length
    const sellVotes = allSigs.filter(s => s === 'SELL').length
    const agreementScore = Math.max(buyVotes, sellVotes) / 3
    const ff5Quality = Math.max(0, Math.min(1,
      (ff5.alpha * 50 + 0.5) * 0.5 + (ff5.rmw * 0.5 + 0.5) * 0.3 + (ff5.beta > 0.5 && ff5.beta < 1.5 ? 0.2 : 0)
    ))
    const tcnQuality = Math.abs(tcn.maAlignment || 0)
    // NEWS ALIGNMENT: reward when RL signal agrees with today's news sentiment
    let newsAlignment = 0
    const ns = newsSentimentMap[ticker]
    if (ns !== undefined) {
      const sigDir = sig.signal === 'BUY' ? 1 : sig.signal === 'SELL' ? -1 : 0
      const newsDir = Math.sign(ns)
      if (sigDir !== 0 && newsDir !== 0)
        newsAlignment = sigDir === newsDir ? Math.abs(ns) * 0.3 : -Math.abs(ns) * 0.15
    }
    totalScore += agreementScore * 0.40 + ff5Quality * 0.25 + tcnQuality * 0.20 + Math.max(-0.25, Math.min(0.25, newsAlignment)) * 0.15
    count++
  }
  return count > 0 ? totalScore / count : 0
}

function computeEnhancedRAG(backtestResult, modelQuality) {
  return (
    Math.max(-2, Math.min(3, backtestResult.sharpe || 0)) * 0.40 +
    Math.max(-1, Math.min(2, backtestResult.calmar || 0)) * 0.20 +
    Math.max(0, Math.min(1, modelQuality)) * 0.30 +
    Math.max(-0.5, Math.min(1, backtestResult.totalReturn || 0)) * 0.10
  )
}

export async function trainEpisodes(barsMap, episodes, onEpisode, onDone, newsSentimentMap = {}) {
  if (running) return
  running = true; episodeCount = 0
  const newsCount = Object.keys(newsSentimentMap).length
  if (newsCount > 0) console.log('[Trainer] News sentiment wired for', newsCount, 'tickers')
  for (let i = 0; i < episodes && running; i++) {
    const firstBars = Object.values(barsMap)[0] || []
    const regime = agent.detectRegime(firstBars)
    const weights = agent.sampleRegimeWeights(regime)
    const backtestResult = backtestPortfolio(barsMap, weights)
    const modelQuality = computeModelQuality(barsMap, weights, newsSentimentMap)
    const enhancedRAG = computeEnhancedRAG(backtestResult, modelQuality)
    agent.recordEpisode(weights, enhancedRAG, regime, { sharpe: backtestResult.sharpe, calmar: backtestResult.calmar, modelQuality, newsCount })
    episodeCount++
    if (onEpisode) onEpisode({
      episode: episodeCount, weights,
      result: { ...backtestResult, ragScore: enhancedRAG, modelQuality, newsCount },
      regime, agentState: agent.getProgress(),
      currentWeights: agent.weights, bestWeights: agent.bestWeights, bestScore: agent.bestScore,
    })
    if (i % 3 === 0) await new Promise(r => setTimeout(r, 0))
  }
  running = false
  if (onDone) onDone({ bestWeights: agent.bestWeights, bestScore: agent.bestScore, episodes: episodeCount })
}
export function stopTraining() { running = false }
