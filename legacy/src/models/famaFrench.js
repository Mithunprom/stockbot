/**
 * famaFrench.js — Fama-French 5-Factor Model (proxied from price data)
 *
 * FF5 factors:
 *   Beta/Mkt  Market sensitivity
 *   SMB       Size (small minus big)
 *   HML       Value (high minus low book/price)
 *   RMW       Profitability (robust minus weak)
 *   CMA       Investment (conservative minus aggressive)
 */

function stdDev(arr) {
  const mean = arr.reduce((a,b)=>a+b,0)/arr.length
  return Math.sqrt(arr.reduce((a,b)=>a+(b-mean)**2,0)/arr.length)
}

function computeBeta(assetR, mktR) {
  const n = Math.min(assetR.length, mktR.length)
  if (n < 10) return 1
  const am = assetR.slice(0,n).reduce((a,b)=>a+b,0)/n
  const mm = mktR.slice(0,n).reduce((a,b)=>a+b,0)/n
  let cov=0, mv=0
  for(let i=0;i<n;i++){ cov+=(assetR[i]-am)*(mktR[i]-mm); mv+=(mktR[i]-mm)**2 }
  return mv>0?cov/mv:1
}

export function computeFF5(bars, marketBars) {
  if (!bars || bars.length < 60) return null
  const closes = bars.map(b=>b.c), n=closes.length
  const stockR = closes.slice(-60).map((c,i,a)=>i===0?0:(c-a[i-1])/a[i-1])
  const mktR = marketBars&&marketBars.length>60
    ? marketBars.slice(-60).map((b,i,a)=>i===0?0:(b.c-a[i-1].c)/a[i-1].c)
    : stockR.map(()=>0.0003)

  const beta = computeBeta(stockR, mktR)
  const price = closes[n-1]
  const high52 = Math.max(...closes.slice(-Math.min(252,n)))
  const mktRet = mktR.slice(-20).reduce((a,b)=>a+b,0)

  // SMB: price as size proxy
  const smb = price<50?0.8:price<200?0.3:price<500?-0.2:-0.6

  // HML: value factor — how far from 52w high
  const hml = (high52 - price) / high52

  // RMW: profitability — % of positive weekly returns
  const weeklyR = []
  for(let i=closes.length-20;i<closes.length;i+=5) if(i>0) weeklyR.push((closes[i]-closes[i-5])/closes[i-5])
  const rmw = weeklyR.length>0?(weeklyR.filter(r=>r>0).length/weeklyR.length)*2-1:0

  // CMA: low recent vol vs historical = conservative
  const cma = -(stdDev(stockR.slice(-10))/Math.max(stdDev(stockR.slice(-60)),0.001)-1)

  // Alpha: excess return vs beta-predicted
  const actual = (closes[n-1]-closes[n-21])/closes[n-21]
  const expected = beta * mktRet
  const alpha = actual - expected

  const score = alpha*0.3 + rmw*0.25 + cma*0.15 + (beta>0.5&&beta<1.8?0.1:-0.1) + (-hml*0.1) + smb*0.1

  return {
    beta:+beta.toFixed(3), alpha:+alpha.toFixed(4),
    smb:+smb.toFixed(3), hml:+hml.toFixed(3), rmw:+rmw.toFixed(3), cma:+cma.toFixed(3),
    score:+score.toFixed(4),
    signal: score>0.1?'BUY':score<-0.1?'SELL':'HOLD',
    interpretation: {
      beta: beta>1.2?'High risk/reward':beta<0.7?'Defensive':'Market-neutral',
      value: hml<0.1?'Growth stock':hml>0.4?'Deep value':'Moderate value',
      quality: rmw>0.3?'Profitable trend':rmw<-0.3?'Deteriorating':'Mixed',
    }
  }
}
