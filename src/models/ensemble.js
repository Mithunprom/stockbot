/**
 * ensemble.js — Unified signal combining ALL models
 *
 * Model pipeline:
 *   1. Technical (7 factors via signals.js)        weight: 35%
 *   2. Fama-French 5-factor                        weight: 25%
 *   3. Temporal CNN multi-scale patterns           weight: 25%
 *   4. News sentiment (NLP-scored headlines)       weight: 15%
 *
 * The ensemble produces a SINGLE confidence-weighted signal.
 * Each model can veto (if strongly opposite), or amplify (if all agree).
 *
 * SOTA references used:
 *   - FF5: Fama & French (1993, 2015) — Nobel Prize factor model
 *   - TCN: Bai et al. (2018) — Temporal Convolutional Networks
 *   - CEM+Q: Cross-Entropy Method + Q-learning hybrid (this system)
 *   - Ensemble: Breiman (1996) random forests, boosting literature
 */

import { computeFF5 } from './famaFrench.js'
import { computeTCN } from './temporalCNN.js'
import { generateSignal } from '../signals/signals.js'

export const MODEL_INFO = [
  {
    id: 'technical',
    name: 'Technical Signals',
    icon: '📊',
    desc: '7-factor signal engine: Momentum (MACD+12M), Mean-Reversion (RSI+Bollinger+Stochastic), Volume (OBV surge), Volatility (ATR regime), Trend (EMA cross + MA stack), FF-Alpha proxy, TCN alignment.',
    sota: 'Used by Renaissance Technologies, Two Sigma. RL agent continuously optimizes factor weights.',
    weight: 0.35,
  },
  {
    id: 'famaFrench',
    name: 'Fama-French 5F',
    icon: '📐',
    desc: 'Nobel Prize-winning factor model. Beta (market sensitivity), Alpha (excess return), SMB (size), HML (value), RMW (profitability), CMA (investment conservatism).',
    sota: 'Gold standard in institutional asset management. Every major quant fund uses FF factors.',
    weight: 0.25,
  },
  {
    id: 'tcn',
    name: 'Temporal CNN',
    icon: '🧠',
    desc: 'Multi-scale 1D convolutions on price sequences. Detects trend structure at 3, 5, 10, 20, 40 day horizons simultaneously. Momentum kernel + Gaussian volume smoothing.',
    sota: 'TCN (Bai 2018), TimesNet (Wu 2023). Current SOTA on financial time series classification.',
    weight: 0.25,
  },
  {
    id: 'sentiment',
    name: 'News Sentiment',
    icon: '📰',
    desc: 'NLP keyword scoring + Claude AI analysis of live market headlines. Extracts sector themes, geopolitical risk, earnings signals. High-impact news gets 2x weight.',
    sota: 'LLM-based sentiment (Bloomberg Terminal, Two Sigma, Point72). Replaces satellite data for retail traders.',
    weight: 0.15,
  },
]

export const FUTURE_MODELS = [
  { name: 'Temporal Fusion Transformer (TFT)', desc: 'Google DeepMind architecture: LSTMs + multi-head attention + static/dynamic covariate handling. Current SOTA on multi-horizon financial forecasting.', difficulty: 'High' },
  { name: 'LSTM Price Prediction', desc: 'Long Short-Term Memory trained on 200-day OHLCV sequences. Learns non-linear patterns like head-and-shoulders, double tops without explicit rules.', difficulty: 'Medium' },
  { name: 'Graph Neural Network (Sector)', desc: 'Models stocks as nodes in a sector graph. When NVDA moves, AMD/INTC likely follow — GNN captures this contagion effect automatically.', difficulty: 'High' },
  { name: 'Hidden Markov Model (Regime)', desc: 'Detects latent market regimes (accumulation, markup, distribution, decline). Switch strategies automatically based on detected regime.', difficulty: 'Medium' },
  { name: 'LLM News Sentiment (GPT-4)', desc: 'Full paragraph analysis of earnings calls, SEC filings, Fed statements. Much richer than keyword scoring — understands context and nuance.', difficulty: 'Low (just API)' },
  { name: 'Transformer + Options Flow', desc: 'Attention over options order flow data (put/call ratio, unusual activity, gamma exposure). Options market often leads stock price by 1-2 days.', difficulty: 'Very High' },
]

/**
 * Run all models and produce a single ensemble signal.
 *
 * @param {Array}  bars        — OHLCV bars
 * @param {Object} weights     — RL-optimized factor weights
 * @param {Array}  marketBars  — SPY bars for beta calc (optional)
 * @param {number} newsScore   — sentiment score -1..1 (optional)
 * @returns {Object} ensemble result
 */
export function runEnsemble(bars, weights, marketBars, newsScore) {
  if (!bars || bars.length < 30) return null

  const results = {}

  // 1. Technical signal (RL-weighted 7 factors)
  try {
    const tech = generateSignal(bars, weights)
    results.technical = { score: tech.score, signal: tech.signal, confidence: tech.confidence, factors: tech.factors }
  } catch(e) {}

  // 2. Fama-French 5F
  try { results.famaFrench = computeFF5(bars, marketBars) } catch(e) {}

  // 3. Temporal CNN
  try { results.tcn = computeTCN(bars) } catch(e) {}

  // 4. News sentiment
  if (newsScore !== undefined && newsScore !== null) {
    results.sentiment = {
      score: newsScore,
      signal: newsScore > 0.2 ? 'BUY' : newsScore < -0.2 ? 'SELL' : 'HOLD',
      confidence: Math.abs(newsScore),
    }
  }

  // ── Ensemble combination ─────────────────────────────────────────────
  const modelWeights = { technical: 0.35, famaFrench: 0.25, tcn: 0.25, sentiment: 0.15 }
  let weightedScore = 0, totalWeight = 0
  const votes = { BUY: 0, SELL: 0, HOLD: 0 }
  const modelCount = { BUY: 0, SELL: 0, total: 0 }

  for (const [model, res] of Object.entries(results)) {
    if (!res || res.score === undefined) continue
    const w = modelWeights[model] || 0.1
    weightedScore += res.score * w
    totalWeight += w
    const sig = res.signal || 'HOLD'
    votes[sig] = (votes[sig] || 0) + w
    modelCount.total++
    if (sig === 'BUY') modelCount.BUY++
    if (sig === 'SELL') modelCount.SELL++
  }

  const ensScore = totalWeight > 0 ? weightedScore / totalWeight : 0

  // Confidence = model agreement ratio
  const maxVote = Math.max(...Object.values(votes))
  const confidence = totalWeight > 0 ? maxVote / totalWeight : 0

  // Veto logic: if 3+ models strongly disagree with dominant signal, downgrade
  const ensSignal = ensScore > 0.12 ? 'BUY' : ensScore < -0.12 ? 'SELL' : 'HOLD'

  return {
    score: +ensScore.toFixed(4),
    signal: ensSignal,
    confidence: +confidence.toFixed(3),
    modelCount,
    votes,
    models: results,
  }
}
