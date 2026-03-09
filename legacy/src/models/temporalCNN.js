/**
 * temporalCNN.js
 * Temporal CNN pattern recognition (simplified, runs in-browser).
 *
 * Real TCNs use dilated causal convolutions on price sequences.
 * This implementation approximates TCN feature extraction using
 * multi-scale moving average crossovers (the core signal TCNs learn).
 *
 * SOTA references:
 *   - Temporal Convolutional Networks (Bai et al. 2018)
 *   - TFT: Temporal Fusion Transformer (Lim et al. 2021)
 *   - TimesNet (Wu et al. 2023) — current SOTA on time series
 */

// Multi-scale convolution kernels (window sizes)
const SCALES = [3, 5, 10, 20, 40]

function movingAvg(arr, window) {
  const result = []
  for (let i = 0; i < arr.length; i++) {
    if (i < window-1) { result.push(null); continue }
    result.push(arr.slice(i-window+1, i+1).reduce((a,b)=>a+b,0)/window)
  }
  return result
}

function conv1d(signal, kernel) {
  // Simple 1D convolution with edge handling
  const k = kernel.length, half = Math.floor(k/2), out = []
  for (let i = 0; i < signal.length; i++) {
    let sum = 0, norm = 0
    for (let j = 0; j < k; j++) {
      const idx = i - half + j
      if (idx >= 0 && idx < signal.length) { sum += signal[idx]*kernel[j]; norm += Math.abs(kernel[j]) }
    }
    out.push(norm>0?sum/norm:0)
  }
  return out
}

// Approximate dilated convolutions at multiple scales
export function computeTCN(bars) {
  if (!bars || bars.length < 50) return null

  const closes = bars.map(b=>b.c)
  const volumes = bars.map(b=>b.v)
  const n = closes.length

  // Multi-scale price features (mimics dilated conv layers)
  const features = {}
  for (const scale of SCALES) {
    const ma = movingAvg(closes, scale)
    const valid = ma.filter(x=>x!==null)
    if (valid.length < 5) continue

    const last = valid[valid.length-1]
    const prev = valid[valid.length-2]
    features[`ma${scale}_trend`] = (last-prev)/prev
    features[`price_vs_ma${scale}`] = (closes[n-1]-last)/last
  }

  // Momentum convolution (derivative kernel)
  const momentumKernel = [-1,-2,0,2,1] // Sobel-like gradient
  const momentumConv = conv1d(closes.slice(-20), momentumKernel)
  const momentumSignal = momentumConv[momentumConv.length-1]

  // Volume convolution
  const volKernel = [0.1,0.2,0.4,0.2,0.1] // Gaussian
  const volConv = conv1d(volumes.slice(-20), volKernel)
  const volSignal = volConv[volConv.length-1]
  const volNorm = volumes.slice(-20).reduce((a,b)=>a+b,0)/20

  // Multi-scale alignment score (when all MAs agree = strong trend)
  const maAlignment = SCALES.filter(s=>features[`price_vs_ma${s}`]!==undefined).reduce((agree,s,_,arr)=>{
    return agree + (features[`price_vs_ma${s}`]>0?1:-1)/arr.length
  },0)

  // Pattern detection
  const recentReturns = closes.slice(-10).map((c,i,a)=>i===0?0:(c-a[i-1])/a[i-1])
  const trend = recentReturns.reduce((a,b)=>a+b,0)
  const volatility = Math.sqrt(recentReturns.reduce((a,b)=>a+b**2,0)/recentReturns.length)
  const sharpeProxy = volatility>0?trend/volatility:0

  // TCN composite score
  const score = (
    maAlignment * 0.35 +
    Math.tanh(momentumSignal / (closes[n-1]*0.01)) * 0.25 +
    Math.tanh(sharpeProxy) * 0.25 +
    (volSignal > volNorm*1.5 ? Math.sign(trend)*0.15 : 0)
  )

  return {
    score: +score.toFixed(4),
    maAlignment: +maAlignment.toFixed(3),
    momentum: +momentumSignal.toFixed(2),
    sharpeProxy: +sharpeProxy.toFixed(3),
    volSurge: +(volSignal/Math.max(volNorm,1)).toFixed(2),
    pattern: maAlignment>0.6?'STRONG_UPTREND':maAlignment<-0.6?'STRONG_DOWNTREND':maAlignment>0.2?'UPTREND':maAlignment<-0.2?'DOWNTREND':'SIDEWAYS',
    signal: score>0.15?'BUY':score<-0.15?'SELL':'HOLD',
    scales: SCALES.reduce((acc,s)=>({ ...acc, [`${s}d`]:features[`price_vs_ma${s}`]!==undefined?+(features[`price_vs_ma${s}`]*100).toFixed(1):null }),{}),
  }
}
