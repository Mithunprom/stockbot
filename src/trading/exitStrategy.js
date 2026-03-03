/**
 * exitStrategy.js — Professional exit strategies used by top traders
 *
 * Strategies:
 *   1. Fixed Stop Loss / Take Profit
 *   2. Trailing Stop — locks in gains as price rises
 *   3. ATR-based — volatility-adjusted (Renaissance, Citadel style)
 *   4. Signal Reversal — exit when model flips
 *   5. Time Stop — max hold period
 *   6. Portfolio Heat — reduce when portfolio down X%
 */

export const EXIT_PRESETS = {
  conservative: { label:'Conservative', icon:'🛡️', desc:'Tight stops. Capital preservation first.', stopLoss:0.05, takeProfit:0.10, trailingStop:0.04, maxDays:10, signalReversal:true },
  moderate:     { label:'Moderate',     icon:'⚖️', desc:'Balanced risk/reward. Standard pro setup.', stopLoss:0.08, takeProfit:0.20, trailingStop:0.06, maxDays:20, signalReversal:true },
  aggressive:   { label:'Aggressive',   icon:'🚀', desc:'Wider stops for bigger moves.', stopLoss:0.15, takeProfit:0.40, trailingStop:0.10, maxDays:45, signalReversal:false },
  atr:          { label:'ATR-Based',    icon:'📐', desc:'Volatility-adjusted. Used by quant funds.', stopLoss:null, takeProfit:null, trailingStop:null, atrMultiplier:2.0, atrPeriod:14, maxDays:30, signalReversal:true },
}

export function calcATR(bars, period = 14) {
  if (!bars || bars.length < period + 1) return null
  const trs = bars.slice(-(period+1)).map((b,i,a) => {
    if(i===0) return b.h-b.l
    return Math.max(b.h-b.l, Math.abs(b.h-a[i-1].c), Math.abs(b.l-a[i-1].c))
  })
  return trs.reduce((a,b)=>a+b,0)/trs.length
}

export function checkExitSignal(position, currentPrice, currentSignal, bars, preset = EXIT_PRESETS.moderate) {
  if(!position || !currentPrice) return null
  const { avgPrice, side, entryTime, highWater } = position
  const isLong = side === 'LONG'
  const pnlPct = isLong ? (currentPrice-avgPrice)/avgPrice : (avgPrice-currentPrice)/avgPrice
  const daysHeld = entryTime ? Math.floor((Date.now()-entryTime)/86400000) : 0
  const hwm = isLong ? Math.max(highWater||avgPrice, currentPrice) : Math.min(highWater||avgPrice, currentPrice)

  // ATR stops
  if (preset.atrMultiplier && bars) {
    const atr = calcATR(bars, preset.atrPeriod||14)
    if (atr) {
      const stop = atr * preset.atrMultiplier
      const target = stop * 3
      if (isLong && currentPrice <= avgPrice - stop) return { shouldExit:true, reason:`ATR Stop (${(stop/avgPrice*100).toFixed(1)}% ATR)`, urgency:'HIGH', pnlPct }
      if (isLong && currentPrice >= avgPrice + target) return { shouldExit:true, reason:`ATR Target Hit (3:1 R/R)`, urgency:'TAKE_PROFIT', pnlPct }
    }
  }

  // Fixed stop loss
  if (preset.stopLoss && pnlPct <= -preset.stopLoss)
    return { shouldExit:true, reason:`Stop Loss ${(preset.stopLoss*100).toFixed(0)}%`, urgency:'HIGH', pnlPct }

  // Take profit
  if (preset.takeProfit && pnlPct >= preset.takeProfit)
    return { shouldExit:true, reason:`Take Profit ${(preset.takeProfit*100).toFixed(0)}%`, urgency:'TAKE_PROFIT', pnlPct }

  // Trailing stop
  if (preset.trailingStop && hwm) {
    const ddFromHigh = isLong ? (hwm-currentPrice)/hwm : (currentPrice-hwm)/hwm
    if (ddFromHigh >= preset.trailingStop)
      return { shouldExit:true, reason:`Trailing Stop ${(preset.trailingStop*100).toFixed(0)}% from peak`, urgency:'MEDIUM', pnlPct }
  }

  // Signal reversal
  if (preset.signalReversal) {
    if (isLong && currentSignal?.signal==='SELL' && currentSignal?.confidence>0.6)
      return { shouldExit:true, reason:`Signal Flip → SELL (${(currentSignal.confidence*100).toFixed(0)}% conf)`, urgency:'MEDIUM', pnlPct }
    if (!isLong && currentSignal?.signal==='BUY' && currentSignal?.confidence>0.6)
      return { shouldExit:true, reason:`Signal Flip → BUY (cover short)`, urgency:'MEDIUM', pnlPct }
  }

  // Time stop
  if (preset.maxDays && daysHeld >= preset.maxDays)
    return { shouldExit:true, reason:`Time Stop — ${daysHeld} days`, urgency:'LOW', pnlPct }

  // Warning
  if (preset.stopLoss && pnlPct <= -preset.stopLoss*0.7)
    return { shouldExit:false, reason:`⚠️ Approaching stop (${(pnlPct*100).toFixed(1)}%)`, urgency:'WARNING', pnlPct }

  return null
}

export function checkPortfolioHeat(portfolio, prices, threshold=0.05) {
  let val = portfolio.cash
  for (const [t,p] of Object.entries(portfolio.positions)) {
    val += p.shares * (prices[t]?.price || p.avgPrice)
  }
  const dd = (100000 - val) / 100000
  if (dd >= threshold) return { shouldReduce:true, drawdown:dd, reason:`Portfolio down ${(dd*100).toFixed(1)}%` }
  return null
}
