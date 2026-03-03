/**
 * agent.js — Enhanced RL Agent
 * Algorithm: Cross-Entropy Method + Q-learning + Regime-aware optimization
 * Now optimizes 7 factors including FF Alpha and TCN Alignment.
 */

import { DEFAULT_WEIGHTS } from '../signals/signals.js'

const STORAGE_KEY = 'stockbot_rl_v3'
const FACTOR_KEYS = ['momentum','meanReversion','volume','volatility','trend','ffAlpha','tcnAlign']

function createDefaultState() {
  return {
    weights: { ...DEFAULT_WEIGHTS },
    population: [],
    episodes: [],
    generation: 0,
    bestScore: -Infinity,
    bestWeights: { ...DEFAULT_WEIGHTS },
    qTable: {},
    epsilon: 0.35,
    regimePerformance: {},
  }
}

export class RLAgent {
  constructor() {
    this.state = this._load()
    this.populationSize = 30
    this.eliteRatio = 0.25
    this.noiseStd = 0.12
  }

  _load() {
    try {
      const s=localStorage.getItem(STORAGE_KEY)
      if(s) { const p=JSON.parse(s); if(p.weights&&p.generation!==undefined) return p }
    } catch{}
    return createDefaultState()
  }

  save() {
    try { localStorage.setItem(STORAGE_KEY,JSON.stringify(this.state)) } catch{}
  }

  reset() { this.state=createDefaultState(); this.save() }

  sampleWeights() {
    if(Math.random()<this.state.epsilon||this.state.population.length<8) return this._randomWeights()
    const elites=this._getElites()
    const mean=this._weightMean(elites.map(e=>e.weights))
    return this._perturbWeights(mean, this.noiseStd)
  }

  // Sample weights biased toward current regime's best performers
  sampleRegimeWeights(regime) {
    const regimePop=this.state.population.filter(e=>e.regime===regime)
    if(regimePop.length>=5 && Math.random()>this.state.epsilon) {
      const top=regimePop.sort((a,b)=>b.score-a.score).slice(0,5)
      return this._perturbWeights(this._weightMean(top.map(e=>e.weights)), this.noiseStd*0.5)
    }
    return this.sampleWeights()
  }

  recordEpisode(weights, score, regime='unknown', modelScores={}) {
    const ep={ weights, score, regime, modelScores, ts:Date.now() }
    this.state.episodes.push(ep)
    this.state.population.push(ep)
    this.state.population.sort((a,b)=>b.score-a.score)
    if(this.state.population.length>this.populationSize*3)
      this.state.population=this.state.population.slice(0,this.populationSize*2)

    if(score>this.state.bestScore) {
      this.state.bestScore=score
      this.state.bestWeights={...weights}
    }

    if(this.state.population.length>=5) {
      const elites=this._getElites()
      this.state.weights=this._weightMean(elites.map(e=>e.weights))
    }

    // Track regime performance
    if(!this.state.regimePerformance[regime]) this.state.regimePerformance[regime]={ scores:[], bestWeights:null, bestScore:-Infinity }
    this.state.regimePerformance[regime].scores.push(score)
    if(score>this.state.regimePerformance[regime].bestScore) {
      this.state.regimePerformance[regime].bestScore=score
      this.state.regimePerformance[regime].bestWeights={...weights}
    }

    // Q-table
    if(!this.state.qTable[regime]) this.state.qTable[regime]=[]
    this.state.qTable[regime].push(score)

    // Adaptive epsilon: decay faster when improving
    const recentScores=this.state.episodes.slice(-10).map(e=>e.score)
    const improving=recentScores.length>=5&&mean(recentScores.slice(-5))>mean(recentScores.slice(0,5))
    this.state.epsilon=Math.max(0.05,this.state.epsilon*(improving?0.96:0.985))
    this.state.generation++
    this.save()
    return this.state.weights
  }

  detectRegime(bars) {
    if(!bars||bars.length<20) return 'unknown'
    const closes=bars.map(b=>b.c)
    const ma20=mean(closes.slice(-20))
    const current=closes[closes.length-1]
    const returns=closes.slice(1).map((c,i)=>(c-closes[i])/closes[i])
    const vol=stddev(returns.slice(-20))*Math.sqrt(252)
    const trend=current>ma20?'bull':'bear'
    const volR=vol>0.4?'highvol':vol>0.2?'medvol':'lowvol'
    // Momentum regime
    const mom=(current-closes[closes.length-20])/closes[closes.length-20]
    const momR=mom>0.05?'trending':mom<-0.05?'declining':'ranging'
    return `${trend}_${volR}_${momR}`
  }

  getRegimeWeights(regime) {
    const rp=this.state.regimePerformance[regime]
    if(rp&&rp.bestWeights&&rp.scores.length>=3) return rp.bestWeights
    const regimePop=this.state.population.filter(e=>e.regime===regime)
    if(regimePop.length>=3) return this._weightMean(regimePop.slice(0,3).map(e=>e.weights))
    return this.state.weights
  }

  get weights() { return this.state.weights }
  get bestWeights() { return this.state.bestWeights }
  get generation() { return this.state.generation }
  get bestScore() { return this.state.bestScore }
  get epsilon() { return this.state.epsilon }

  getProgress() {
    const eps=this.state.episodes
    if(!eps.length) return { improving:false, trend:0 }
    const recent=eps.slice(-10).map(e=>e.score)
    const older=eps.slice(-20,-10).map(e=>e.score)
    const rm=mean(recent), om=older.length?mean(older):rm
    return { improving:rm>om, trend:rm-om, generation:this.state.generation, epsilon:this.state.epsilon, bestScore:this.state.bestScore, regimes:Object.keys(this.state.regimePerformance) }
  }

  _getElites() {
    const n=Math.max(3,Math.floor(this.state.population.length*this.eliteRatio))
    return this.state.population.slice(0,n)
  }
  _weightMean(wArr) {
    const m={}
    for(const k of FACTOR_KEYS) m[k]=wArr.reduce((a,w)=>a+(w[k]||0),0)/wArr.length
    return this._normalizeWeights(m)
  }
  _randomWeights() {
    const w={}
    for(const k of FACTOR_KEYS) w[k]=Math.random()
    return this._normalizeWeights(w)
  }
  _perturbWeights(weights,std) {
    const w={}
    for(const k of FACTOR_KEYS) w[k]=Math.max(0,(weights[k]||0)+this._randn()*std)
    return this._normalizeWeights(w)
  }
  _normalizeWeights(w) {
    const sum=Object.values(w).reduce((a,b)=>a+b,0)
    if(sum===0) return {...DEFAULT_WEIGHTS}
    const n={}
    for(const k of FACTOR_KEYS) n[k]=(w[k]||0)/sum
    return n
  }
  _randn() {
    return Math.sqrt(-2*Math.log(Math.random()))*Math.cos(2*Math.PI*Math.random())
  }
}

function mean(arr) { return arr.reduce((a,b)=>a+b,0)/arr.length }
function stddev(arr) { const m=mean(arr); return Math.sqrt(arr.reduce((a,b)=>a+(b-m)**2,0)/arr.length) }

export const agent = new RLAgent()
