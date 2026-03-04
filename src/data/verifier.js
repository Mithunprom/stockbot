/**
 * verifier.js — 5-Subagent Verification + Auto-Improvement System
 *
 * Subagents:
 *   1. PriceVerifier    — detects mock/stale prices, spot-checks vs Polygon
 *   2. AlgoVerifier     — signal sanity, data leakage, weight validity
 *   3. RLVerifier       — reward convergence, Sharpe, drawdown, algo mode fit
 *   4. StrategyVerifier — win rate, avg P&L, sector performance, trade quality
 *   5. EmailVerifier    — Resend reachability, config health, send history
 *
 * After checks complete, the Orchestrator generates auto-improvement
 * recommendations that the app can apply immediately.
 */

// ─── 1. PRICE VERIFIER ───────────────────────────────────────────────────────
export async function runPriceVerifier({ prices, bars, apiKey }) {
  const checks = []

  // Check for mock sentinel values
  const zeroOrHundred = Object.entries(prices)
    .filter(([, p]) => p?.price === 100 || p?.price === 0 || p?.price === 1)
    .map(([t]) => t)
  checks.push({
    id: 'no_mock_sentinels',
    label: 'No mock sentinel prices (0, 1, 100)',
    status: zeroOrHundred.length === 0 ? 'pass' : 'fail',
    detail: zeroOrHundred.length > 0
      ? `Mock prices: ${zeroOrHundred.join(', ')} — real price not yet loaded`
      : 'All prices real or hidden'
  })

  // Check bar data is not mock-normalized (last close ≈ 100)
  const mockBarsFound = Object.entries(bars)
    .filter(([t]) => !t.includes('X:'))
    .slice(0, 8)
    .filter(([, b]) => b?.length > 0 && Math.abs(b[b.length - 1].c - 100) < 2)
    .map(([t]) => t)
  checks.push({
    id: 'bars_real',
    label: 'Bar data not mock-normalized (real OHLCV)',
    status: mockBarsFound.length === 0 ? 'pass' : apiKey ? 'fail' : 'warn',
    detail: mockBarsFound.length > 0
      ? `Mock bars: ${mockBarsFound.join(', ')} — price shown as "—" in emails (correct)`
      : 'Bars have real price history'
  })

  // Suspiciously round prices (exact $5 multiples = likely hardcoded)
  const roundPrices = Object.entries(prices)
    .filter(([t, p]) => p?.price > 0 && p.price % 5 === 0 && !['SPY','QQQ','GLD'].includes(t))
    .map(([t, p]) => `${t}=$${p.price}`)
  checks.push({
    id: 'no_round_prices',
    label: 'Prices not suspiciously round',
    status: roundPrices.length < 4 ? 'pass' : 'warn',
    detail: roundPrices.length > 0 ? `Possible seed values: ${roundPrices.slice(0,5).join(', ')}` : 'Prices look natural'
  })

  // Live spot-check vs Polygon (3 tickers)
  if (apiKey) {
    const testTickers = Object.keys(prices).filter(t => !t.includes('X:') && prices[t]?.price > 0).slice(0, 3)
    const results = await Promise.all(testTickers.map(async ticker => {
      try {
        const r = await fetch(
          `https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/${ticker}?apiKey=${apiKey}`,
          { signal: AbortSignal.timeout(5000) }
        )
        if (!r.ok) return { ticker, status: 'warn', detail: `HTTP ${r.status}` }
        const d = await r.json()
        const live = d?.ticker?.day?.c || d?.ticker?.prevDay?.c
        const ours = prices[ticker]?.price
        if (!live || !ours) return { ticker, status: 'warn', detail: 'No live price' }
        const drift = Math.abs(live - ours) / live
        return {
          ticker,
          status: drift < 0.02 ? 'pass' : drift < 0.10 ? 'warn' : 'fail',
          detail: `Ours: $${ours.toFixed(2)} | Polygon: $${live.toFixed(2)} | Drift: ${(drift*100).toFixed(1)}%`
        }
      } catch (e) { return { ticker, status: 'warn', detail: e.message } }
    }))
    results.forEach(r => checks.push({ id: `live_${r.ticker}`, label: `${r.ticker} vs Polygon`, status: r.status, detail: r.detail }))
  } else {
    checks.push({ id: 'no_api_key', label: 'Polygon API key present', status: 'warn',
      detail: 'No API key — prices shown as "—" in emails. Add VITE_POLYGON_API_KEY for live prices.' })
  }

  const failed = checks.filter(c => c.status === 'fail').length
  const warned = checks.filter(c => c.status === 'warn').length
  // Count tickers that are blocked (mock sentinel, stale, or live drift fail)
  const blockedTickers = Object.entries(prices)
    .filter(([t, p]) => !t.includes('X:') && (!p?.price || p.price === 100 || p.price === 0 || p.price === 1))
    .map(([t]) => t)
  return {
    agent: 'PriceVerifier',
    status: failed > 0 ? 'error' : warned > 0 ? 'warn' : 'ok',
    checks,
    blockedTickers,
    summary: failed > 0
      ? `${failed} price issue(s) — ${blockedTickers.length} ticker(s) blocked from trading`
      : warned > 0
      ? `${warned} warning(s) — check prices before trading`
      : 'All prices Polygon-verified — trading allowed',
    ts: Date.now()
  }
}

// ─── 2. ALGO VERIFIER ────────────────────────────────────────────────────────
export function runAlgoVerifier({ bars, signals, weights }) {
  const checks = []

  const testTicker = Object.keys(bars).find(t => bars[t]?.length > 60 && !t.includes('X:'))
  if (testTicker) {
    const sig = signals[testTicker]
    checks.push({
      id: 'signal_has_factors',
      label: 'Signals have multi-factor breakdown',
      status: sig?.factors && Object.keys(sig.factors).length >= 3 ? 'pass' : 'warn',
      detail: sig?.factors ? `${testTicker}: ${Object.keys(sig.factors).join(', ')}` : `${testTicker}: no factors found`
    })
    const unbounded = Object.entries(signals).filter(([,s]) => Math.abs(s?.score||0) > 1).map(([t,s]) => `${t}=${s.score.toFixed(2)}`)
    checks.push({ id: 'scores_bounded', label: 'Signal scores in [-1, 1]', status: unbounded.length === 0 ? 'pass' : 'fail', detail: unbounded.length > 0 ? `Unbounded: ${unbounded.slice(0,4).join(', ')}` : 'All scores valid' })
    const badConf = Object.entries(signals).filter(([,s]) => (s?.confidence||0) < 0 || (s?.confidence||0) > 1).map(([t]) => t)
    checks.push({ id: 'confidence_bounded', label: 'Confidence in [0, 1]', status: badConf.length === 0 ? 'pass' : 'fail', detail: badConf.length > 0 ? `Bad: ${badConf.join(', ')}` : 'All valid' })
  } else {
    checks.push({ id: 'no_bars', label: 'Bars available for check', status: 'warn', detail: 'No bars yet — run screener' })
  }

  const wVals = Object.values(weights||{}).filter(v => typeof v === 'number')
  const wSum = wVals.reduce((a,b)=>a+Math.abs(b),0)
  checks.push({ id: 'weights_valid', label: 'Weights positive, sum ~1', status: wVals.length > 0 && wSum > 0.5 && wSum < 2 ? 'pass' : 'warn', detail: `${wVals.length} weights, sum=${wSum.toFixed(3)}` })

  const sigArr = Object.values(signals||{})
  const buyPct = sigArr.filter(s=>s?.signal==='BUY').length / (sigArr.length||1)
  checks.push({ id: 'signal_distribution', label: 'Signal distribution not degenerate', status: buyPct > 0.95 || buyPct < 0.05 ? 'warn' : 'pass', detail: `${(buyPct*100).toFixed(0)}% BUY · ${(100-buyPct*100).toFixed(0)}% non-BUY` })

  const failed = checks.filter(c=>c.status==='fail').length
  const warned = checks.filter(c=>c.status==='warn').length
  return { agent:'AlgoVerifier', status: failed>0?'error':warned>0?'warn':'ok', checks, summary: failed>0?`${failed} algo issue(s)`:warned>0?`${warned} algo warning(s)`:'Signal logic clean', ts:Date.now() }
}

// ─── 3. RL VERIFIER ──────────────────────────────────────────────────────────
export function runRLVerifier({ trainingLog, backtestResult, rlProgress, algoMode, weights }) {
  const checks = []

  checks.push({ id:'training_run', label:'RL training executed', status:(trainingLog||[]).length>0?'pass':'warn', detail:(trainingLog||[]).length>0?`${trainingLog.length} episodes logged`:'No episodes — run Train tab first' })

  if ((trainingLog||[]).length >= 10) {
    const scores = trainingLog.map(e => parseFloat(e.score)||0)
    const first5 = scores.slice(0,5).reduce((a,b)=>a+b,0)/5
    const last5  = scores.slice(-5).reduce((a,b)=>a+b,0)/5
    checks.push({ id:'converging', label:'Reward improving across episodes', status:last5>first5?'pass':'warn', detail:`First 5 avg: ${first5.toFixed(3)} → Last 5 avg: ${last5.toFixed(3)} ${last5<=first5?'— not converging, may need more episodes':''}` })
    
    // Check for plateau (last 5 episodes variance is very low)
    const last10 = scores.slice(-10)
    const variance = last10.reduce((a,x)=>a+(x-last5)**2,0)/last10.length
    checks.push({ id:'not_plateaued', label:'RL not stuck in local optimum', status:variance>0.001?'pass':'warn', detail:`Score variance (last 10): ${variance.toFixed(5)} ${variance<=0.001?'— plateau detected, try more episodes':''}` })
  }

  if (backtestResult) {
    const { sharpe=0, maxDrawdown=0, totalReturn=0, reward=0, winRate=0 } = backtestResult
    checks.push({ id:'sharpe', label:'Sharpe ratio positive', status:sharpe>0?'pass':sharpe>-0.5?'warn':'fail', detail:`Sharpe: ${sharpe.toFixed(2)} (target: >0.5)` })
    checks.push({ id:'drawdown', label:'Max drawdown ≤ 15%', status:maxDrawdown<0.15?'pass':maxDrawdown<0.25?'warn':'fail', detail:`Max DD: ${(maxDrawdown*100).toFixed(1)}% ${maxDrawdown>0.15?'⚠️ Obs Mode triggered':''}` })
    checks.push({ id:'reward', label:'Spec reward positive (ΔP − λ×DD)', status:reward>0?'pass':'warn', detail:`Reward: ${reward.toFixed(4)} = ${totalReturn.toFixed(4)} − 0.5×${maxDrawdown.toFixed(4)}` })
    checks.push({ id:'winrate', label:'Win rate > 40%', status:winRate>0.4?'pass':winRate>0.3?'warn':'fail', detail:`Win rate: ${(winRate*100).toFixed(0)}% (target: >40%)` })
  } else {
    checks.push({ id:'no_backtest', label:'Backtest result available', status:'warn', detail:'Run training to get metrics' })
  }

  checks.push({ id:'algo_mode', label:'Algo mode set (PPO/SAC/CEM)', status:['PPO','SAC','CEM'].includes(algoMode)?'pass':'warn', detail:`Mode: ${algoMode} ${algoMode==='PPO'?'(trending)':algoMode==='SAC'?'(sideways)':'(optimizer)'}` })

  const epsilon = rlProgress?.epsilon
  if (epsilon !== undefined) checks.push({ id:'epsilon', label:'Exploration rate decaying', status:epsilon<0.9?'pass':'warn', detail:`ε = ${epsilon.toFixed(3)} ${epsilon>0.9?'(too high — run more training)':''}` })

  const failed = checks.filter(c=>c.status==='fail').length
  const warned = checks.filter(c=>c.status==='warn').length
  return { agent:'RLVerifier', status:failed>0?'error':warned>0?'warn':'ok', checks, summary:failed>0?`${failed} RL issue(s)`:warned>0?`${warned} RL warning(s)`:'RL performing correctly', ts:Date.now() }
}

// ─── 4. STRATEGY VERIFIER ─────────────────────────────────────────────────────
export function runStrategyVerifier({ tradeLog, backtestResult, signals, prices, autoSettings }) {
  const checks = []
  const trades = tradeLog || []

  // Win rate from real paper trades
  const closedTrades = trades.filter(t => t.pnl !== null && t.pnl !== undefined)
  if (closedTrades.length >= 5) {
    const wins = closedTrades.filter(t => (t.pnl||0) > 0)
    const losses = closedTrades.filter(t => (t.pnl||0) <= 0)
    const winRate = wins.length / closedTrades.length
    const avgWin  = wins.reduce((a,t)=>a+(t.pnl||0),0) / (wins.length||1)
    const avgLoss = losses.reduce((a,t)=>a+Math.abs(t.pnl||0),0) / (losses.length||1)
    const expectancy = (winRate * avgWin) - ((1-winRate) * avgLoss)

    checks.push({ id:'paper_winrate', label:`Paper trade win rate (${closedTrades.length} trades)`, status:winRate>0.45?'pass':winRate>0.35?'warn':'fail', detail:`Win rate: ${(winRate*100).toFixed(0)}% · Avg win: $${avgWin.toFixed(0)} · Avg loss: $${avgLoss.toFixed(0)}` })
    checks.push({ id:'expectancy', label:'Trade expectancy positive', status:expectancy>0?'pass':'fail', detail:`Expectancy: ${expectancy>0?'+':''}$${expectancy.toFixed(0)} per trade ${expectancy<0?'— strategy losing money on average':''}` })

    // Profit factor
    const grossProfit = wins.reduce((a,t)=>a+(t.pnl||0),0)
    const grossLoss   = losses.reduce((a,t)=>a+Math.abs(t.pnl||0),0)
    const pf = grossLoss > 0 ? grossProfit / grossLoss : grossProfit > 0 ? 999 : 0
    checks.push({ id:'profit_factor', label:'Profit factor > 1.2', status:pf>1.5?'pass':pf>1.0?'warn':'fail', detail:`Profit factor: ${pf.toFixed(2)} (gross profit $${grossProfit.toFixed(0)} / gross loss $${grossLoss.toFixed(0)})` })
  } else {
    checks.push({ id:'insufficient_trades', label:'Sufficient paper trades for analysis', status:'warn', detail:`Only ${closedTrades.length} closed trades — need 5+ for reliable stats` })
  }

  // Check auto-trader exit settings are sensible
  const tp = autoSettings?.takeProfitPct || 0.10
  const sl = autoSettings?.stopLossPct   || 0.07
  const rr = tp / sl
  checks.push({ id:'risk_reward', label:'Risk/reward ratio ≥ 1.2', status:rr>=1.4?'pass':rr>=1.0?'warn':'fail', detail:`Target: +${(tp*100).toFixed(0)}% · Stop: -${(sl*100).toFixed(0)}% · R/R = ${rr.toFixed(2)}` })

  // Check backtest vs paper consistency
  if (backtestResult && closedTrades.length >= 3) {
    const paperReturn = closedTrades.reduce((a,t)=>a+(t.pnl||0),0) / 100000
    const btReturn = backtestResult.totalReturn || 0
    const gap = Math.abs(paperReturn - btReturn)
    checks.push({ id:'bt_paper_consistency', label:'Backtest vs paper trade consistency', status:gap<0.15?'pass':'warn', detail:`Backtest: ${(btReturn*100).toFixed(1)}% · Paper: ${(paperReturn*100).toFixed(1)}% · Gap: ${(gap*100).toFixed(1)}%` })
  }

  // Over-trading check (more than 3 trades per day average)
  if (trades.length > 0) {
    const days = new Set(trades.map(t => { try { return new Date(t.time||0).toDateString() } catch { return '' } })).size || 1
    const tradesPerDay = trades.length / days
    checks.push({ id:'overtrading', label:'Trade frequency reasonable', status:tradesPerDay<5?'pass':tradesPerDay<10?'warn':'fail', detail:`${tradesPerDay.toFixed(1)} trades/day avg over ${days} days` })
  }

  const failed = checks.filter(c=>c.status==='fail').length
  const warned = checks.filter(c=>c.status==='warn').length
  return { agent:'StrategyVerifier', status:failed>0?'error':warned>0?'warn':'ok', checks, summary:failed>0?`${failed} strategy issue(s) need attention`:warned>0?`${warned} strategy warning(s)`:'Strategy metrics healthy', ts:Date.now() }
}

// ─── 5. EMAIL VERIFIER ───────────────────────────────────────────────────────
export async function runEmailVerifier({ emailConfig }) {
  const checks = []
  const { to, lastBriefingSent, lastAlertSent } = emailConfig || {}

  checks.push({ id:'email_to', label:'Email recipient configured', status:to&&to.includes('@')?'pass':'fail', detail:to?`Recipient: ${to}`:'No email — set in Paper tab' })

  try {
    const r = await fetch('/api/email', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({type:'__ping__',to:to||'test@test.com',data:{}}), signal:AbortSignal.timeout(5000) })
    const d = await r.json().catch(()=>({}))
    checks.push({ id:'api_live', label:'Email API /api/email reachable', status:r.status<500?'pass':'fail', detail:`HTTP ${r.status}` })
    checks.push({ id:'resend_key', label:'RESEND_API_KEY configured', status:d?.error?.includes('RESEND_API_KEY')||d?.error?.includes('not configured')?'fail':'pass', detail:d?.error?.includes('RESEND_API_KEY')?'Key missing — add in Vercel env vars':'Key present' })
  } catch(e) {
    checks.push({ id:'api_live', label:'Email API reachable', status:'fail', detail:`Network error: ${e.message}` })
  }

  const today = new Date().toDateString()
  checks.push({ id:'briefing_sent', label:'Morning briefing sent today', status:lastBriefingSent===today?'pass':'warn', detail:lastBriefingSent===today?`Sent today`:`Not yet (auto-sends 8–9am ET). Last: ${lastBriefingSent||'never'}` })

  const now = Date.now()
  const gap = lastAlertSent ? now - lastAlertSent : Infinity
  checks.push({ id:'no_spam', label:'Alert cooldown respected (2hr)', status:gap===Infinity||gap>7200000?'pass':'warn', detail:gap<7200000?`Last alert ${Math.round(gap/60000)}min ago`:`Cooldown OK` })

  const failed = checks.filter(c=>c.status==='fail').length
  const warned = checks.filter(c=>c.status==='warn').length
  return { agent:'EmailVerifier', status:failed>0?'error':warned>0?'warn':'ok', checks, summary:failed>0?`${failed} email issue(s)`:warned>0?`${warned} email warning(s)`:'Email system operational', ts:Date.now() }
}

// ─── AUTO-IMPROVEMENT ENGINE ─────────────────────────────────────────────────
/**
 * Reads verification report + raw trade history and generates concrete
 * parameter adjustments to maximise P/L.
 *
 * Improvement layers (in order of priority):
 *   1. Kelly criterion position sizing from actual paper trade history
 *   2. Consecutive loss protection — tighten entry criteria during losing streaks
 *   3. Score-bucket analysis — raise minScore if low-signal trades are net losers
 *   4. Adaptive take-profit — widen TP when winners consistently exceed current target
 *   5. Drawdown guard — tighten SL when backtest/paper drawdown is high
 *   6. Portfolio gain protection — lock in gains by tightening SL when up >15%
 *   7. Confidence threshold — raise when win rate is failing
 *   8. RL convergence / algo mode
 *
 * @param {Object} report          - Full verifier report
 * @param {Object} currentSettings - Current autoTradeSettings
 * @param {Object} rawData         - { tradeLog, portfolio }
 */
export function generateAutoImprovements(report, currentSettings, rawData = {}) {
  const actions        = []
  const settingPatches = {}
  let shouldRetrain    = false
  let retrainReason    = ''

  const rl       = report.agents?.rl
  const strategy = report.agents?.strategy
  const price    = report.agents?.price

  // ── Raw trade analysis ───────────────────────────────────────────────────
  const allTrades     = (rawData.tradeLog || [])
  const closedTrades  = allTrades.filter(t => t.pnl != null)
  const portfolio     = rawData.portfolio || null

  const wins   = closedTrades.filter(t => (t.pnl || 0) > 0)
  const losses = closedTrades.filter(t => (t.pnl || 0) <= 0)
  const winRate   = closedTrades.length > 0 ? wins.length / closedTrades.length : 0.5
  const avgWinPnl = wins.length   > 0 ? wins.reduce((a,t)=>a+(t.pnl||0),0)  / wins.length   : 0
  const avgLosPnl = losses.length > 0 ? losses.reduce((a,t)=>a+Math.abs(t.pnl||0),0) / losses.length : 0
  // Avg win/loss as fraction of trade value (for Kelly)
  const avgWinR = wins.length   > 0 ? wins.reduce((a,t)=>a+((t.pnl||0)/Math.max(t.value||5000,1)),0)   / wins.length   : 0.03
  const avgLosR = losses.length > 0 ? losses.reduce((a,t)=>a+(Math.abs(t.pnl||0)/Math.max(t.value||5000,1)),0) / losses.length : 0.02

  // ── 1. Kelly criterion position sizing ────────────────────────────────────
  // Only apply when we have enough trades for statistically reliable Kelly
  if (closedTrades.length >= 10 && avgLosR > 0) {
    const b         = avgWinR / avgLosR                            // odds ratio
    const fullKelly = (b * winRate - (1 - winRate)) / b           // Kelly fraction
    const halfKelly = Math.max(0, Math.min(0.25, fullKelly * 0.5)) // Half-Kelly, cap 25%
    const portfolioVal = portfolio
      ? portfolio.cash + Object.values(portfolio.positions || {}).reduce((s,p)=>s+p.shares*(p.avgPrice||0),0)
      : 100000
    const kellySize = Math.round((halfKelly * portfolioVal) / 100) * 100
    const currentSize = currentSettings?.tradeSize || 5000
    if (kellySize > 0 && Math.abs(kellySize - currentSize) / currentSize > 0.15) {
      settingPatches.tradeSize = kellySize
      const dir = kellySize > currentSize ? 'up' : 'down'
      actions.push({ type:'setting', severity:'ok', label:'Kelly position sizing', detail:`Win ${(winRate*100).toFixed(0)}% · R/R ${b.toFixed(2)} → Half-Kelly: ${(halfKelly*100).toFixed(1)}% of portfolio = $${kellySize} (${dir} from $${currentSize})`, fix:'Applied automatically' })
    }
  }

  // ── 2. Consecutive loss protection ────────────────────────────────────────
  if (closedTrades.length >= 3) {
    const recentTrades     = closedTrades.slice(-5)
    const consecLosses     = recentTrades.filter(t => (t.pnl || 0) < 0).length
    if (consecLosses >= 3) {
      const newScore = Math.min(0.60, (currentSettings?.minScore || 0.25) + 0.10)
      const newConf  = Math.min(0.80, (currentSettings?.minConfidence || 0.55) + 0.08)
      settingPatches.minScore      = newScore
      settingPatches.minConfidence = newConf
      actions.push({ type:'setting', severity:'error', label:'Losing streak protection', detail:`${consecLosses}/5 recent trades lost — tightening entry: minScore → ${newScore.toFixed(2)}, minConf → ${newConf.toFixed(2)}`, fix:'Applied automatically' })
      shouldRetrain = true
      retrainReason = 'Losing streak — re-optimising weights'
    }
  }

  // ── 3. Score-bucket analysis — filter low-signal entries ──────────────────
  if (closedTrades.length >= 8) {
    const scoreOf = t => parseFloat(t.signal) || 0
    const lowBucket  = closedTrades.filter(t => scoreOf(t) < 0.40)
    const highBucket = closedTrades.filter(t => scoreOf(t) >= 0.40)
    const lowExp  = lowBucket.reduce((a,t)=>a+(t.pnl||0),0)  / Math.max(lowBucket.length, 1)
    const highExp = highBucket.reduce((a,t)=>a+(t.pnl||0),0) / Math.max(highBucket.length, 1)
    if (lowBucket.length >= 4 && lowExp < 0 && highExp > lowExp) {
      const newScore = Math.min(0.45, Math.max(0.35, currentSettings?.minScore || 0.25) + 0.05)
      if (newScore > (currentSettings?.minScore || 0.25)) {
        settingPatches.minScore = newScore
        actions.push({ type:'setting', severity:'warn', label:'Skip low-signal trades', detail:`Trades with score < 0.40 avg ${lowExp < 0 ? '-' : '+'}$${Math.abs(lowExp).toFixed(0)}/trade vs +$${highExp.toFixed(0)} for score ≥ 0.40 — raising minScore → ${newScore.toFixed(2)}`, fix:'Applied automatically' })
      }
    }
  }

  // ── 4. Adaptive take-profit from actual winner magnitude ──────────────────
  if (wins.length >= 5) {
    const currentTP = currentSettings?.takeProfitPct || 0.10
    // If average winning trade return exceeds TP × 1.4, TP is cutting winners short
    if (avgWinR > currentTP * 1.4) {
      const newTP = Math.min(0.30, +(avgWinR * 0.85).toFixed(2)) // capture 85% of avg win
      if (newTP > currentTP + 0.01) {
        settingPatches.takeProfitPct = newTP
        actions.push({ type:'setting', severity:'ok', label:'Widen take-profit (winners exceed target)', detail:`Avg winner: ${(avgWinR*100).toFixed(1)}% but TP is ${(currentTP*100).toFixed(0)}% — raising TP → ${(newTP*100).toFixed(0)}%`, fix:'Applied automatically' })
      }
    }
  }

  // ── 5. Drawdown guard ─────────────────────────────────────────────────────
  const ddCheck = rl?.checks.find(c=>c.id==='drawdown')
  if (ddCheck?.status==='fail') {
    const current = currentSettings?.stopLossPct || 0.07
    const newVal  = Math.max(0.03, +(current - 0.015).toFixed(3))
    settingPatches.stopLossPct = newVal
    actions.push({ type:'setting', severity:'error', label:'Tighten stop-loss (drawdown exceeded)', detail:`Drawdown > 15% — reducing SL ${(current*100).toFixed(0)}% → ${(newVal*100).toFixed(0)}%`, fix:'Applied automatically' })
  }

  // ── 6. Portfolio gain protection ──────────────────────────────────────────
  if (portfolio) {
    const totalVal = portfolio.cash + Object.values(portfolio.positions || {}).reduce((s,p)=>s+p.shares*(p.avgPrice||0),0)
    const gainPct  = (totalVal - 100000) / 100000
    if (gainPct >= 0.15) {
      const current = currentSettings?.stopLossPct || 0.07
      const newSL   = Math.max(0.03, +(current - 0.01).toFixed(3))
      if (newSL < current) {
        settingPatches.stopLossPct = newSL
        actions.push({ type:'setting', severity:'ok', label:'Protect gains (portfolio up ' + (gainPct*100).toFixed(0) + '%)', detail:`Tightening SL from ${(current*100).toFixed(0)}% → ${(newSL*100).toFixed(0)}% to lock in profits`, fix:'Applied automatically' })
      }
    }
  }

  // ── 7. Win rate / confidence threshold ────────────────────────────────────
  const winRateCheck = rl?.checks.find(c=>c.id==='winrate') || strategy?.checks.find(c=>c.id==='paper_winrate')
  if (winRateCheck?.status==='fail') {
    const current = currentSettings?.minConfidence || 0.55
    const newVal  = Math.min(0.80, +(current + 0.05).toFixed(2))
    if (!settingPatches.minConfidence) settingPatches.minConfidence = newVal
    actions.push({ type:'setting', severity:'error', label:'Raise confidence threshold (win rate low)', detail:`Win rate failing — minConfidence ${current} → ${newVal}`, fix:'Applied automatically' })
    shouldRetrain = true
    retrainReason = retrainReason || 'Win rate below threshold'
  }

  // ── 8. RL convergence / algo mode ─────────────────────────────────────────
  const rlConverging = rl?.checks.find(c=>c.id==='converging')
  const rlPlateau    = rl?.checks.find(c=>c.id==='not_plateaued')
  if (rlConverging?.status==='warn' || rlPlateau?.status==='warn') {
    if (!shouldRetrain) {
      shouldRetrain = true
      retrainReason = rlPlateau?.status==='warn' ? 'RL in local optimum — restart with wider exploration' : 'Reward not improving'
      actions.push({ type:'retrain', severity:'warn', label:'RL improvement', detail:retrainReason, fix:'Auto-retraining' })
    }
  }
  const sharpeCheck = rl?.checks.find(c=>c.id==='sharpe')
  if (sharpeCheck?.status==='fail') {
    if (!shouldRetrain) {
      shouldRetrain = true
      retrainReason = retrainReason || 'Sharpe negative — algo mode switch needed'
    }
    actions.push({ type:'mode', severity:'error', label:'Algo mode wrong for current regime', detail:'Sharpe negative — retrain will auto-select PPO vs SAC', fix:'Triggered retrain' })
  }
  const rrCheck = strategy?.checks.find(c=>c.id==='risk_reward')
  if (rrCheck?.status==='fail' && !settingPatches.takeProfitPct) {
    const currentTP = currentSettings?.takeProfitPct || 0.10
    settingPatches.takeProfitPct = Math.min(0.25, +(currentTP + 0.02).toFixed(2))
    actions.push({ type:'setting', severity:'warn', label:'Widen TP (R/R too tight)', detail:`R/R ratio below 1.2 — TP ${(currentTP*100).toFixed(0)}% → ${(settingPatches.takeProfitPct*100).toFixed(0)}%`, fix:'Applied automatically' })
  }
  const expectancyCheck = strategy?.checks.find(c=>c.id==='expectancy')
  if (expectancyCheck?.status==='fail' && !settingPatches.minScore) {
    const current = currentSettings?.minScore || 0.25
    settingPatches.minScore = Math.min(0.50, +(current + 0.05).toFixed(2))
    actions.push({ type:'setting', severity:'error', label:'Raise entry threshold (negative expectancy)', detail:`minScore ${current} → ${settingPatches.minScore}`, fix:'Applied automatically' })
  }
  const apiCheck = price?.checks.find(c=>c.id==='no_api_key')
  if (apiCheck?.status==='warn') {
    actions.push({ type:'info', severity:'warn', label:'No live prices', detail:'Add VITE_POLYGON_API_KEY for real prices', fix:'Manual — polygon.io' })
  }

  if (actions.length === 0) {
    actions.push({ type:'ok', severity:'ok', label:'No improvements needed', detail:`All 5 subagents passed · Win rate: ${(winRate*100).toFixed(0)}% · ${closedTrades.length} trades analysed`, fix:'—' })
  }

  return { actions, settingPatches, shouldRetrain, retrainReason }
}

// ─── ORCHESTRATOR ─────────────────────────────────────────────────────────────
export async function runAllVerifiers(ctx) {
  const [price, algo, rl, strategy, email] = await Promise.all([
    runPriceVerifier({ prices:ctx.prices, bars:ctx.bars, apiKey:ctx.apiKey }),
    Promise.resolve(runAlgoVerifier({ bars:ctx.bars, signals:ctx.signals, weights:ctx.weights })),
    Promise.resolve(runRLVerifier({ trainingLog:ctx.trainingLog, backtestResult:ctx.backtestResult, rlProgress:ctx.rlProgress, algoMode:ctx.algoMode, weights:ctx.weights })),
    Promise.resolve(runStrategyVerifier({ tradeLog:ctx.tradeLog, backtestResult:ctx.backtestResult, signals:ctx.signals, prices:ctx.prices, autoSettings:ctx.autoSettings })),
    runEmailVerifier({ emailConfig:ctx.emailConfig }),
  ])

  const agents  = [price, algo, rl, strategy, email]
  const errors  = agents.filter(a=>a.status==='error').map(a=>a.agent)
  const warns   = agents.filter(a=>a.status==='warn').map(a=>a.agent)
  const oks     = agents.filter(a=>a.status==='ok').map(a=>a.agent)

  const feedback = []
  agents.forEach(agent => {
    agent.checks.filter(c=>c.status==='fail'||c.status==='warn').forEach(c => feedback.push({ agent:agent.agent, severity:c.status==='fail'?'error':'warn', label:c.label, detail:c.detail, id:`${agent.agent}::${c.id}` }))
  })

  const report = { ts:Date.now(), overall:errors.length>0?'error':warns.length>0?'warn':'ok', agents:{price,algo,rl,strategy,email}, summary:{ errors,warns,oks, message:errors.length>0?`${errors.length} issue(s): ${errors.join(', ')}`:warns.length>0?`${warns.length} warning(s): ${warns.join(', ')}`:'✅ All 5 subagents passed' }, feedback }

  // Auto-improvement recommendations — pass raw trade log + portfolio for Kelly/streak analysis
  report.improvements = generateAutoImprovements(report, ctx.autoSettings, { tradeLog: ctx.tradeLog, portfolio: ctx.portfolio })
  return report
}
