/**
 * agent.js
 * Reinforcement Learning agent for strategy weight optimization.
 *
 * Algorithm: Cross-Entropy Method (CEM) + Q-learning hybrid
 * - CEM is fast, gradient-free, works well for continuous parameter spaces
 * - Q-learning tracks state → reward history for smarter exploration
 *
 * State space: market regime (bull/bear/sideways) × volatility (low/high)
 * Action space: factor weight vectors (5 factors, continuous)
 * Reward: Risk-Adjusted Growth score (Sharpe + Calmar composite)
 *
 * Persistence: weights stored in localStorage (survives page refresh)
 */

import { DEFAULT_WEIGHTS } from '../signals/signals.js'

const STORAGE_KEY = 'stockbot_rl_agent'
const FACTOR_KEYS = ['momentum', 'meanReversion', 'volume', 'volatility', 'trend']

// ── Agent State ────────────────────────────────────────────────────────────

function createDefaultState() {
  return {
    // Current best weights
    weights: { ...DEFAULT_WEIGHTS },
    // CEM population: array of { weights, score } sorted by score desc
    population: [],
    // Training history
    episodes: [],
    // Generation count
    generation: 0,
    // Best score seen
    bestScore: -Infinity,
    bestWeights: { ...DEFAULT_WEIGHTS },
    // Q-table: regime → average reward per weight config
    qTable: {},
    // Exploration rate (anneals over time)
    epsilon: 0.3,
  }
}

export class RLAgent {
  constructor() {
    this.state = this._load()
    this.populationSize = 20
    this.eliteRatio = 0.3
    this.noiseStd = 0.1
  }

  // ── Persistence ──────────────────────────────────────────────────────────

  _load() {
    try {
      const saved = localStorage.getItem(STORAGE_KEY)
      if (saved) {
        const parsed = JSON.parse(saved)
        // Validate structure
        if (parsed.weights && parsed.generation !== undefined) return parsed
      }
    } catch {}
    return createDefaultState()
  }

  save() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(this.state))
    } catch {}
  }

  reset() {
    this.state = createDefaultState()
    this.save()
  }

  // ── CEM: Sample New Candidate Weights ────────────────────────────────────

  /**
   * Sample a candidate weight vector.
   * If we have an elite population, sample around the elite mean.
   * Otherwise explore randomly.
   */
  sampleWeights() {
    // Epsilon-greedy: sometimes explore randomly
    if (Math.random() < this.state.epsilon || this.state.population.length < 5) {
      return this._randomWeights()
    }

    // Sample around elite mean
    const elites = this._getElites()
    const mean = this._weightMean(elites.map(e => e.weights))
    return this._perturbWeights(mean, this.noiseStd)
  }

  /**
   * Record episode result and update the population.
   * @param {Object} weights  - weights used in this episode
   * @param {number} score    - RAG score (reward)
   * @param {string} regime   - market regime label
   */
  recordEpisode(weights, score, regime = 'unknown') {
    const episode = { weights, score, regime, ts: Date.now() }
    this.state.episodes.push(episode)
    this.state.population.push(episode)

    // Keep population bounded
    this.state.population.sort((a, b) => b.score - a.score)
    if (this.state.population.length > this.populationSize * 3) {
      this.state.population = this.state.population.slice(0, this.populationSize * 2)
    }

    // Update best
    if (score > this.state.bestScore) {
      this.state.bestScore = score
      this.state.bestWeights = { ...weights }
    }

    // Update current weights to elite mean
    if (this.state.population.length >= 5) {
      const elites = this._getElites()
      this.state.weights = this._weightMean(elites.map(e => e.weights))
    }

    // Update Q-table
    if (!this.state.qTable[regime]) this.state.qTable[regime] = []
    this.state.qTable[regime].push(score)

    // Anneal epsilon
    this.state.generation++
    this.state.epsilon = Math.max(0.05, 0.3 * Math.pow(0.97, this.state.generation))

    this.save()
    return this.state.weights
  }

  // ── Regime Detection ─────────────────────────────────────────────────────

  /**
   * Classify current market regime from equity bars.
   * Used to look up regime-specific Q-values.
   */
  detectRegime(bars) {
    if (!bars || bars.length < 20) return 'unknown'
    const closes = bars.map(b => b.c)
    const ma20 = closes.slice(-20).reduce((a, b) => a + b, 0) / 20
    const current = closes[closes.length - 1]
    const returns = closes.slice(-20).map((c, i) => i === 0 ? 0 : (c - closes[closes.length - 20 + i - 1]) / closes[closes.length - 20 + i - 1])
    const vol = Math.sqrt(returns.reduce((a, b) => a + b * b, 0) / returns.length) * Math.sqrt(252)

    const trend = current > ma20 ? 'bull' : 'bear'
    const volRegime = vol > 0.4 ? 'highvol' : 'lowvol'
    return `${trend}_${volRegime}`
  }

  /**
   * Get the best known weights for a given regime.
   */
  getRegimeWeights(regime) {
    const regimeEps = this.state.population.filter(e => e.regime === regime)
    if (regimeEps.length >= 3) {
      const top = regimeEps.sort((a, b) => b.score - a.score).slice(0, 3)
      return this._weightMean(top.map(e => e.weights))
    }
    return this.state.weights
  }

  // ── Getters ──────────────────────────────────────────────────────────────

  get weights() { return this.state.weights }
  get bestWeights() { return this.state.bestWeights }
  get generation() { return this.state.generation }
  get bestScore() { return this.state.bestScore }
  get epsilon() { return this.state.epsilon }

  getEpisodes(last = 20) {
    return this.state.episodes.slice(-last)
  }

  getProgress() {
    const eps = this.state.episodes
    if (eps.length === 0) return { improving: false, trend: 0 }
    const recent = eps.slice(-10).map(e => e.score)
    const older = eps.slice(-20, -10).map(e => e.score)
    const recentMean = recent.reduce((a, b) => a + b, 0) / (recent.length || 1)
    const olderMean = older.length ? older.reduce((a, b) => a + b, 0) / older.length : recentMean
    return {
      improving: recentMean > olderMean,
      trend: recentMean - olderMean,
      generation: this.state.generation,
      epsilon: this.state.epsilon,
      bestScore: this.state.bestScore
    }
  }

  // ── Private ──────────────────────────────────────────────────────────────

  _getElites() {
    const n = Math.max(2, Math.floor(this.state.population.length * this.eliteRatio))
    return this.state.population.slice(0, n)
  }

  _weightMean(weightArr) {
    const mean = {}
    for (const k of FACTOR_KEYS) {
      mean[k] = weightArr.reduce((a, w) => a + (w[k] || 0), 0) / weightArr.length
    }
    return this._normalizeWeights(mean)
  }

  _randomWeights() {
    const w = {}
    for (const k of FACTOR_KEYS) {
      w[k] = Math.random()
    }
    return this._normalizeWeights(w)
  }

  _perturbWeights(weights, std) {
    const w = {}
    for (const k of FACTOR_KEYS) {
      w[k] = Math.max(0, (weights[k] || 0) + this._randn() * std)
    }
    return this._normalizeWeights(w)
  }

  _normalizeWeights(w) {
    const sum = Object.values(w).reduce((a, b) => a + b, 0)
    if (sum === 0) return { ...DEFAULT_WEIGHTS }
    const norm = {}
    for (const k of FACTOR_KEYS) norm[k] = (w[k] || 0) / sum
    return norm
  }

  _randn() {
    // Box-Muller transform
    const u1 = Math.random()
    const u2 = Math.random()
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
  }
}

// Singleton
export const agent = new RLAgent()
