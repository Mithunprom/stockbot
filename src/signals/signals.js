/**
 * signals.js — Enhanced multi-factor signal engine
 * Incorporates: Momentum, Mean-Reversion, Volume, Volatility, Trend,
 *               MACD, Bollinger Bands, Stochastic, ATR, OBV
 * All factors feed the RL agent for weight optimization.
 */

function mean(arr) { return arr.reduce((a,b)=>a+b,0)/arr.length }
function stddev(arr) { const m=mean(arr); return Math.sqrt(arr.reduce((a,b)=>a+(b-m)**2,0)/arr.length) }
function clamp(v,min=-3,max=3) { return Math.max(min,Math.min(max,v)) }

// ── Technical Indicators ───────────────────────────────────────────────────

export function calcRSI(closes, period=14) {
  if (closes.length < period+1) return 50
  let gains=0, losses=0
  for (let i=closes.length-period; i<closes.length; i++) {
    const d = closes[i]-closes[i-1]
    if (d>0) gains+=d; else losses-=d
  }
  const rs = losses===0?100:gains/losses
  return 100-100/(1+rs)
}

export function calcEMA(closes, period) {
  if (closes.length<period) return closes[closes.length-1]
  const k=2/(period+1)
  let ema=mean(closes.slice(0,period))
  for (let i=period; i<closes.length; i++) ema=closes[i]*k+ema*(1-k)
  return ema
}

export function calcMACD(closes) {
  if (closes.length < 26) return { macd:0, signal:0, hist:0 }
  const ema12=calcEMA(closes,12), ema26=calcEMA(closes,26)
  const macdLine=ema12-ema26
  // Signal: 9-period EMA of macd (approximate)
  const macdHistory = []
  for (let i=26; i<=closes.length; i++) {
    const e12=calcEMA(closes.slice(0,i),12)
    const e26=calcEMA(closes.slice(0,i),26)
    macdHistory.push(e12-e26)
  }
  const signalLine = calcEMA(macdHistory, 9)
  return { macd:macdLine, signal:signalLine, hist:macdLine-signalLine }
}

export function calcBollingerBands(closes, period=20, mult=2) {
  if (closes.length<period) return { upper:0, middle:0, lower:0, pct:0.5, bandwidth:0 }
  const slice=closes.slice(-period)
  const middle=mean(slice), sd=stddev(slice)
  const upper=middle+mult*sd, lower=middle-mult*sd
  const current=closes[closes.length-1]
  const pct=upper===lower?0.5:(current-lower)/(upper-lower)
  return { upper, middle, lower, pct, bandwidth:(upper-lower)/middle }
}

export function calcStochastic(bars, kPeriod=14, dPeriod=3) {
  if (bars.length<kPeriod) return { k:50, d:50 }
  const slice=bars.slice(-kPeriod)
  const highH=Math.max(...slice.map(b=>b.h))
  const lowL=Math.min(...slice.map(b=>b.l))
  const current=bars[bars.length-1].c
  const k=highH===lowL?50:((current-lowL)/(highH-lowL))*100
  // D = 3-period SMA of K (simplified)
  const kValues=[k]
  return { k, d:kValues.reduce((a,b)=>a+b,0)/kValues.length }
}

export function calcOBV(bars) {
  if (bars.length<2) return 0
  let obv=0
  for (let i=1; i<bars.length; i++) {
    if (bars[i].c>bars[i-1].c) obv+=bars[i].v
    else if (bars[i].c<bars[i-1].c) obv-=bars[i].v
  }
  return obv
}

export function calcATR(bars, period=14) {
  if (bars.length<period+1) return 0
  const trs=bars.slice(-(period+1)).map((b,i,a)=>{
    if(i===0) return b.h-b.l
    return Math.max(b.h-b.l,Math.abs(b.h-a[i-1].c),Math.abs(b.l-a[i-1].c))
  })
  return trs.reduce((a,b)=>a+b,0)/trs.length
}

// ── Enhanced Factors ───────────────────────────────────────────────────────

export function momentumFactor(bars) {
  if (bars.length<60) return 0
  const c=bars.map(b=>b.c), n=c.length
  const r1m=(c[n-1]-c[n-20])/c[n-20]
  const r3m=(c[n-1]-c[n-Math.min(60,n-1)])/c[n-Math.min(60,n-1)]
  // MACD confirmation
  const macd=calcMACD(c)
  const macdSignal=macd.hist>0?0.3:macd.hist<0?-0.3:0
  return clamp((r1m*0.5+r3m*0.3)*10+macdSignal*2)
}

export function meanReversionFactor(bars) {
  const c=bars.map(b=>b.c)
  const rsi=calcRSI(c,14)
  const bb=calcBollingerBands(c)
  const stoch=calcStochastic(bars)
  // RSI oversold/overbought
  const rsiSig=clamp(((50-rsi)/50)*3)
  // Bollinger: below lower band = buy, above upper = sell
  const bbSig=clamp((0.5-bb.pct)*4)
  // Stochastic
  const stochSig=clamp(((50-stoch.k)/50)*2)
  return clamp(rsiSig*0.4+bbSig*0.4+stochSig*0.2)
}

export function volumeFactor(bars) {
  if (bars.length<20) return 0
  const avgVol=mean(bars.slice(-20).map(b=>b.v))
  const last=bars[bars.length-1]
  const volRatio=last.v/(avgVol||1)
  const priceDir=last.c>last.o?1:-1
  // OBV trend
  const obvRecent=calcOBV(bars.slice(-10))
  const obvOlder=calcOBV(bars.slice(-20,-10))
  const obvTrend=obvRecent>obvOlder?0.5:-0.5
  return clamp((volRatio-1)*priceDir*1.5+obvTrend)
}

export function volatilityFactor(bars) {
  if (bars.length<20) return 0
  const returns=bars.slice(1).map((b,i)=>(b.c-bars[i].c)/bars[i].c)
  const vol20=stddev(returns.slice(-20))
  const vol5=stddev(returns.slice(-5))
  const atr=calcATR(bars)
  const atrNorm=bars[bars.length-1].c>0?atr/bars[bars.length-1].c:0
  const volRatio=vol5/(vol20||0.001)
  // Low ATR relative to price = calm = good for momentum
  const atrSig=atrNorm<0.015?0.5:atrNorm>0.04?-0.5:0
  return clamp((1-volRatio)*2+atrSig)
}

export function trendFactor(bars) {
  if (bars.length<50) return 0
  const c=bars.map(b=>b.c), n=c.length
  const current=c[n-1]
  const vwap=bars[n-1].vw||current
  const ma20=mean(c.slice(-20))
  const ma50=mean(c.slice(-50))
  const ema12=calcEMA(c,12)
  const ema26=calcEMA(c,26)
  // Golden cross: EMA12 > EMA26 = bullish
  const emaCross=ema12>ema26?0.5:-0.5
  // Price above all MAs = strong trend
  const aboveMas=[current>vwap,current>ma20,current>ma50,ema12>ema26].filter(Boolean).length
  return clamp((aboveMas/4-0.5)*4+emaCross)
}

// ── NEW: Fama-French Alpha proxy ───────────────────────────────────────────
export function ffAlphaFactor(bars) {
  if (bars.length<60) return 0
  const c=bars.map(b=>b.c), n=c.length
  const returns=c.slice(1).map((v,i)=>(v-c[i])/c[i])
  // Proxy alpha as return above what momentum predicts
  const mktProxy=mean(returns.slice(-20))
  const stockReturn=(c[n-1]-c[n-20])/c[n-20]
  const alpha=stockReturn-mktProxy*20
  // RMW proxy: consistency of positive weeks
  const weeklyR=[]
  for(let i=n-20;i<n;i+=5) if(i>0) weeklyR.push((c[i]-c[i-5])/c[i-5])
  const rmw=weeklyR.length>0?weeklyR.filter(r=>r>0).length/weeklyR.length*2-1:0
  return clamp(alpha*5+rmw*1.5)
}

// ── NEW: TCN multi-scale alignment ────────────────────────────────────────
export function tcnAlignmentFactor(bars) {
  if (bars.length<50) return 0
  const c=bars.map(b=>b.c), n=c.length
  const scales=[5,10,20,50]
  let aligned=0
  for(const s of scales) {
    if(n>s) {
      const ma=mean(c.slice(-s))
      aligned+=(c[n-1]>ma?1:-1)/scales.length
    }
  }
  return clamp(aligned*3)
}

// ── Composite Signal ───────────────────────────────────────────────────────

export function generateSignal(bars, weights=DEFAULT_WEIGHTS) {
  if (!bars||bars.length<20) return { score:0, signal:'HOLD', factors:{}, confidence:0, indicators:{} }

  const factors = {
    momentum:      momentumFactor(bars),
    meanReversion: meanReversionFactor(bars),
    volume:        volumeFactor(bars),
    volatility:    volatilityFactor(bars),
    trend:         trendFactor(bars),
    ffAlpha:       ffAlphaFactor(bars),
    tcnAlign:      tcnAlignmentFactor(bars),
  }

  // Use provided weights for core factors, fixed small weight for new factors
  const w = { ...weights, ffAlpha:0.08, tcnAlign:0.08 }
  const weightSum=Object.values(w).reduce((a,b)=>a+Math.abs(b),0)
  let score=0
  for (const [k,v] of Object.entries(factors)) score+=(w[k]||0)*v
  score=weightSum>0?score/weightSum:0
  score=clamp(score,-1,1)

  const confidence=Math.max(0,Math.min(1,1-stddev(Object.values(factors))/3))
  const signal=score>0.15?'BUY':score<-0.15?'SELL':'HOLD'

  // Raw indicator values for display
  const c=bars.map(b=>b.c)
  const indicators={
    rsi:+calcRSI(c,14).toFixed(1),
    macd:calcMACD(c),
    bb:calcBollingerBands(c),
    stoch:calcStochastic(bars),
    atr:+calcATR(bars).toFixed(2),
  }

  return { score, signal, factors, confidence, indicators }
}

export const DEFAULT_WEIGHTS = {
  momentum:0.28, meanReversion:0.12, volume:0.18,
  volatility:0.12, trend:0.20, ffAlpha:0.05, tcnAlign:0.05,
}
