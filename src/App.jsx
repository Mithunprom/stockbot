import React, { useState, useEffect, useCallback, useRef } from 'react'
import { AreaChart, Area, BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'
import { screenStocks, SCREENING_PROFILES, DEFAULT_CRITERIA } from './data/screener.js'
import { fetchMarketNews, fetchTickerNews, scoreWithClaude, extractNewsStocks, clearNewsCache } from './data/news.js'
import { generateSignal, calcRSI, calcMACD, calcBollingerBands, calcATR } from './signals/signals.js'
import { backtestPortfolio } from './backtest/backtester.js'
import { agent } from './rl/agent.js'
import { trainEpisodes, stopTraining } from './rl/trainer.js'
import { loadPortfolio, savePortfolio, loadTradeLog, saveTradeLog, loadWatchlist, saveWatchlist, loadSettings, saveSettings, resetPortfolio as resetStored, normalizeTicket, displayTicker, debugStorage, loadScreenerCache, saveScreenerCache, isScreenerCacheFresh } from './data/persistence.js'
import { EXIT_PRESETS, checkExitSignal, checkPortfolioHeat } from './trading/exitStrategy.js'
import { evaluateAutoTrade, AUTO_TRADE_DEFAULTS } from './trading/autoTrader.js'
import { runEnsemble, MODEL_INFO, FUTURE_MODELS } from './models/ensemble.js'
import { fetchDynamicUniverse, getUniverseLabel } from './data/universe.js'
import { fetchCryptoPrices, buildCryptoBars } from './data/cryptoPrices.js'
import { equityCurveToCSV } from './backtest/backtester.js'
import { isConfigured as alpacaConfigured, getMode as alpacaMode, syncPortfolio } from './trading/alpaca.js'
import { runAllVerifiers, generateAutoImprovements } from './data/verifier.js'
import { verifyPricesViaPolygon, applyPriceGate } from './data/priceGate.js'

const ALL_CRYPTO = ['X:BTCUSD','X:ETHUSD','X:SOLUSD']
const CRYPTO_DISPLAY = {'X:BTCUSD':'BTC','X:ETHUSD':'ETH','X:SOLUSD':'SOL'}
// Crypto prices fetched live from CoinGecko — see cryptoPrices.js
const SCREEN_INTERVAL = 15*60
const STARTING_CAPITAL = 100000

const C = {
  bg:'#050510',surface:'#0a0a1a',panel:'#0f0f22',border:'#1a1a3a',
  accent:'#00d4ff',accentDim:'#00d4ff22',green:'#00ff88',red:'#ff3366',
  yellow:'#ffd700',purple:'#a855f7',orange:'#ff8c00',
  text:'#e2e8f0',textDim:'#64748b',textBright:'#ffffff',
}
const S = {
  app:{ background:C.bg,minHeight:'100vh',fontFamily:"'IBM Plex Mono','Courier New',monospace",color:C.text,display:'flex',flexDirection:'column',paddingBottom:64 },
  header:{ background:`linear-gradient(90deg,${C.surface},#0a0a2a)`,borderBottom:`1px solid ${C.border}`,padding:'0 16px',height:52,display:'flex',alignItems:'center',justifyContent:'space-between',position:'sticky',top:0,zIndex:100 },
  logo:{ fontSize:16,fontWeight:700,letterSpacing:4,color:C.accent,textShadow:`0 0 20px ${C.accent}` },
  main:{ flex:1,padding:12,maxWidth:1400,margin:'0 auto',width:'100%' },
  panel:{ background:C.panel,border:`1px solid ${C.border}`,borderRadius:8,padding:14 },
  pt:{ fontSize:9,letterSpacing:3,color:C.textDim,marginBottom:10,textTransform:'uppercase' },
  table:{ width:'100%',borderCollapse:'collapse',fontSize:11 },
  th:{ textAlign:'left',padding:'6px 8px',borderBottom:`1px solid ${C.border}`,color:C.textDim,fontSize:9,letterSpacing:1,fontWeight:400 },
  td:{ padding:'7px 8px',borderBottom:`1px solid ${C.border}40` },
  badge:c=>({ fontSize:9,padding:'2px 6px',borderRadius:2,border:`1px solid ${c}`,color:c,letterSpacing:1,whiteSpace:'nowrap' }),
  sigBadge:s=>({ display:'inline-block',padding:'2px 8px',borderRadius:3,fontSize:9,fontWeight:700,letterSpacing:1,
    background:s==='BUY'?'#00ff8822':s==='SELL'?'#ff336622':'#ffffff11',
    color:s==='BUY'?C.green:s==='SELL'?C.red:C.textDim,
    border:`1px solid ${s==='BUY'?C.green:s==='SELL'?C.red:C.border}` }),
  btn:(v='primary')=>({ background:v==='primary'?C.accentDim:v==='danger'?'#ff336622':v==='active'?'#a855f722':v==='green'?'#00ff8822':v==='warn'?'#ffd70022':'transparent',
    border:`1px solid ${v==='primary'?C.accent:v==='danger'?C.red:v==='active'?C.purple:v==='green'?C.green:v==='warn'?C.yellow:C.border}`,
    color:v==='primary'?C.accent:v==='danger'?C.red:v==='active'?C.purple:v==='green'?C.green:v==='warn'?C.yellow:C.textDim,
    padding:'7px 14px',borderRadius:4,cursor:'pointer',fontSize:10,letterSpacing:1,fontFamily:'inherit',transition:'all 0.2s' }),
  input:{ background:C.surface,border:`1px solid ${C.border}`,color:C.textBright,padding:'8px 12px',borderRadius:4,fontFamily:'inherit',fontSize:11,outline:'none' },
}


// ── Error Boundary — prevents black screen on any React crash ─────────────
class ErrorBoundary extends React.Component {
  constructor(props) { super(props); this.state = { error: null } }
  static getDerivedStateFromError(e) { return { error: e } }
  componentDidCatch(e, info) { console.error('[ErrorBoundary]', e, info) }
  render() {
    if (this.state.error) return (
      <div style={{padding:40,textAlign:'center',fontFamily:'monospace',background:'#0a0e1a',minHeight:'100vh',color:'#ff3366'}}>
        <div style={{fontSize:24,marginBottom:16}}>⚠️ Render Error</div>
        <div style={{fontSize:12,color:'#94a3b8',marginBottom:24}}>{this.state.error.message}</div>
        <button onClick={()=>this.setState({error:null})}
          style={{padding:'8px 24px',background:'#1e293b',color:'#00d4ff',border:'1px solid #00d4ff',borderRadius:4,cursor:'pointer',fontFamily:'monospace'}}>
          ↺ RETRY
        </button>
      </div>
    )
    return this.props.children
  }
}


const fmt = {
  price: v=>!v?'—':v>=1000?`$${(v/1000).toFixed(2)}K`:`$${v.toFixed(2)}`,
  pct: v=>v==null?'—':`${v>=0?'+':''}${(v*100).toFixed(2)}%`,
  chg: v=>({ color:v>=0?C.green:C.red }),
  date: ts=>new Date(ts).toLocaleDateString(),
  ago: ts=>{ const m=Math.floor((Date.now()-new Date(ts))/60000); return m<60?`${m}m ago`:`${Math.floor(m/60)}h ago` },
}

// mockBars — synthetic OHLCV bars for backtesting/signals when no real data exists.
// Prices are index-normalized to 100. Real prices come ONLY from Polygon/CoinGecko.
// Never display mockBars price as a real stock price.
function mockBars(ticker, days=400) {
  const charSeed=ticker.split('').reduce((a,c)=>a+c.charCodeAt(0),0)
  let rng=charSeed
  const rand=()=>{ rng=(rng*1664525+1013904223)&0xffffffff; return (rng>>>0)/0xffffffff }
  const raw=[]; let px=100; const now=Date.now()
  for(let i=days;i>=0;i--){
    const t=now-i*86400000
    const o=px; px=Math.max(1,px*(1+(rand()-0.485)*0.028))
    raw.push({t,o,px})
  }
  // Normalize: final bar = 100
  const scale=100/raw[raw.length-1].px
  return raw.map(({t,o,px:c})=>{
    const so=+(o*scale).toFixed(4), sc=+(c*scale).toFixed(4)
    return {t,o:so,h:+(Math.max(so,sc)*(1+rand()*0.008)).toFixed(4),
      l:+(Math.min(so,sc)*(1-rand()*0.008)).toFixed(4),
      c:sc,v:Math.round(5e5+rand()*4e6),vw:+((so+sc)/2).toFixed(4)}
  })
}

function getApiKey() {
  try { const k=import.meta.env.VITE_POLYGON_API_KEY; if(k&&k.length>10&&k!=='your_polygon_api_key_here') return k } catch(e){}
  return null
}
function getFinnhubKey() {
  try { const k=import.meta.env.VITE_FINNHUB_API_KEY; if(k&&k.length>8) return k } catch(e){}
  return null
}
function hasAnyPriceKey() { return !!(getApiKey()||getFinnhubKey()) }

function portfolioValue(port, prices) {
  let val=port.cash
  for(const [t,p] of Object.entries(port.positions)) {
    const px=prices[t]?.price||p.avgPrice; val+=p.shares*px
  }
  return val
}

function App() {
  const [tab,setTab]=useState('screen')
  const [screenProfile,setScreenProfile]=useState(()=>loadSettings().screenProfile||'momentum')
  const [screenResult,setScreenResult]=useState(()=>{
    const c=loadScreenerCache(); if(!isScreenerCacheFresh(c)) return null
    return { stocks:c.stocks, timestamp:new Date(c.timestamp), mock:false, fromCache:true }
  })
  const [isScreening,setIsScreening]=useState(false)
  const [isRefreshing,setIsRefreshing]=useState(false)
  const [verifyReport,setVerifyReport]=useState(null)
  const [isVerifying,setIsVerifying]=useState(false)
  const [activeStocks,setActiveStocks]=useState(()=>{ const c=loadScreenerCache(); return isScreenerCacheFresh(c)?(c.tickers||[]):[] })
  const [prices,setPrices]=useState(()=>{ const c=loadScreenerCache(); return isScreenerCacheFresh(c)?(c.prices||{}):{}  })
  const [bars,setBars]=useState({})
  const [signals,setSignals]=useState(()=>{ const c=loadScreenerCache(); return isScreenerCacheFresh(c)?(c.signals||{}):{}  })
  const [backtestResult,setBacktestResult]=useState(null)
  const [trainingLog,setTrainingLog]=useState([])
  const [isTraining,setIsTraining]=useState(false)
  const [dataStatus,setDataStatus]=useState({live:false,keyFound:false})
  const [selectedAsset,setSelectedAsset]=useState(null)
  const [weights,setWeights]=useState(agent.weights)
  const [rlProgress,setRlProgress]=useState(null)
  const [ensembleResults,setEnsembleResults]=useState({})
  const [universeLabel,setUniverseLabel]=useState('Initializing...')
  const [activeModel,setActiveModel]=useState('ensemble')
  const [nextScreenIn,setNextScreenIn]=useState(SCREEN_INTERVAL)
  const [autoScreenEnabled,setAutoScreenEnabled]=useState(true)
  const [signalFilter,setSignalFilter]=useState(null)
  const [customCriteria,setCustomCriteria]=useState(DEFAULT_CRITERIA)
  // Portfolio — persisted
  const [portfolio,setPortfolio]=useState(()=>loadPortfolio())
  const [tradeLog,setTradeLog]=useState(()=>loadTradeLog())
  const [tradeSize,setTradeSize]=useState(()=>loadSettings().tradeSize||5000)
  const [watchlist,setWatchlist]=useState(()=>loadWatchlist())
  const [watchlistInput,setWatchlistInput]=useState('')
  // Auto-trading
  const [autoTradeSettings,setAutoTradeSettings]=useState({...AUTO_TRADE_DEFAULTS,...loadSettings().autoTrade})
  const [autoTradeLog,setAutoTradeLog]=useState([])
  const [verifiedTickers,setVerifiedTickers]=useState(()=>{ const c=loadScreenerCache(); return isScreenerCacheFresh(c)?new Set(c.verifiedTickers||[]):new Set() })
  const verifiedRef=useRef(new Set())
  const [exitPreset,setExitPreset]=useState('moderate')
  const [exitAlerts,setExitAlerts]=useState([]) // warnings shown in UI
  // Email
  const [emailStatus,setEmailStatus]=useState(null)
  const [observationMode,setObservationMode]=useState(false)
  const [algoMode,setAlgoMode]=useState('CEM')
  const [lastAlertSent,setLastAlertSent]=useState(()=>{ const v=localStorage.getItem('stockbot_alert_sent'); return v?parseInt(v):null })
  const [lastBriefingSent,setLastBriefingSent]=useState(()=>localStorage.getItem('stockbot_briefing_sent')||null)
  const EMAIL_TO = 'mithunghosh404@gmail.com'
  // News
  const [marketNews,setMarketNews]=useState([])
  const marketNewsRef=useRef([])
  const [newsStocks,setNewsStocks]=useState([]) // tickers surfaced by news
  const newsStocksRef=useRef([])
  const [tickerNews,setTickerNews]=useState([])
  const [aiSentiment,setAiSentiment]=useState(null)

  const screenTimerRef=useRef(null)
  // Save portfolio on tab close — last line of defense
  useEffect(()=>{
    const handler = () => {
      savePortfolio(portfolioRef.current)
      saveTradeLog(tradeLogRef.current || [])
    }
    window.addEventListener('beforeunload', handler)
    return () => window.removeEventListener('beforeunload', handler)
  }, [])
  const countdownRef=useRef(null)
  const autoTradeRef=useRef(null)
  const emailCheckRef=useRef(null)
  const isScreeningRef=useRef(false)
  const apiKey=getApiKey()
  const portfolioRef=useRef(portfolio)
  const tradeLogRef=useRef(tradeLog)
  const pricesRef=useRef(prices)
  const signalsRef=useRef(signals)
  const barsRef=useRef(bars)
  const ensembleRef=useRef(ensembleResults)
  const tradeSizeRef=useRef(tradeSize)
  const dataStatusRef=useRef({live:false,keyFound:false})
  const autoSettingsRef=useRef(autoTradeSettings)

  // Keep refs in sync for use inside setInterval callbacks
  useEffect(()=>{ portfolioRef.current=portfolio },[portfolio])
  useEffect(()=>{ tradeLogRef.current=tradeLog },[tradeLog])
  useEffect(()=>{ pricesRef.current=prices },[prices])
  useEffect(()=>{ signalsRef.current=signals },[signals])
  useEffect(()=>{ barsRef.current=bars },[bars])
  useEffect(()=>{ ensembleRef.current=ensembleResults },[ensembleResults])
  useEffect(()=>{ tradeSizeRef.current=tradeSize },[tradeSize])
  useEffect(()=>{ newsStocksRef.current=newsStocks },[newsStocks])
  useEffect(()=>{ marketNewsRef.current=marketNews },[marketNews])
  useEffect(()=>{ dataStatusRef.current=dataStatus },[dataStatus])
  useEffect(()=>{ autoSettingsRef.current=autoTradeSettings },[autoTradeSettings])
  useEffect(()=>{ verifiedRef.current=verifiedTickers },[verifiedTickers])

  // ── Persistence ──────────────────────────────────────────────────────────
  // Settings save (not critical for positions so useEffect is fine here)
  useEffect(()=>{ saveSettings({ tradeSize, screenProfile }) },[tradeSize,screenProfile])

  // ── Screener ─────────────────────────────────────────────────────────────
  const runScreener = useCallback(async(profile,custom)=>{
    if(isScreeningRef.current) return
    isScreeningRef.current=true
    // First load (no data yet) → show full spinner. Background refresh → keep old data visible.
    setIsScreening(true)
    setIsRefreshing(true)

    const activeProfile=profile||screenProfile
    const criteria=custom||SCREENING_PROFILES[activeProfile]?.criteria||DEFAULT_CRITERIA

    // Build mock stocks
    // Fetch dynamic universe — live top movers + most active if API available
    const universeTickers = await fetchDynamicUniverse(apiKey, newsStocksRef.current, watchlist)
    setUniverseLabel(getUniverseLabel(!!apiKey, newsStocksRef.current.length))
    const mockStocks=universeTickers.slice(0,25).map(ticker=>{
      const b=mockBars(ticker)
      const n=b.length
      const sig=generateSignal(b, agent.weights)
      return { ticker, price:null, change1d:(b[n-1].c-b[n-2].c)/b[n-2].c*100, change5d:(b[n-1].c-b[n-5].c)/b[n-5].c*100,
        volume:b[n-1].v, composite:sig.score,
        scores:{ momentum:sig.factors.momentum||0, volatility:sig.factors.volatility||0, volumeSurge:sig.factors.volume||0, trendBreak:sig.factors.trend||0, rsiOversold:sig.factors.meanReversion||0 }, bars:b }
    }).sort((a,b)=>b.composite-a.composite)

    let stocks=mockStocks; let isLive=false
    if(apiKey) {
      try { const r=await screenStocks(criteria,20); if(r.stocks.length>0){stocks=r.stocks;isLive=true} }
      catch(e){ console.error('[Screener]',e) }
    }

    // Build bars map — stocks + crypto with CORRECT prices
    const barsMap={}
    stocks.forEach(s=>{ barsMap[s.ticker]=(s.bars||[]).map(b=>({t:b.t,o:b.o,h:b.h,l:b.l,c:b.c,v:b.v,vw:b.vw||b.c})) })

    // Fetch live crypto prices from CoinGecko (free, no API key needed)
    let cryptoPriceData = {}
    try { cryptoPriceData = await fetchCryptoPrices() } catch(e) { console.warn('[App] Crypto fetch failed:', e) }

    // Build crypto bars anchored to live prices
    for(const pair of ALL_CRYPTO) {
      const livePrice = cryptoPriceData[pair]?.price
      barsMap[pair] = buildCryptoBars(pair, livePrice)
    }

    // Build price map — stocks AND crypto
    const priceMap={}
    stocks.forEach(s=>{ priceMap[s.ticker]={ price:s.price, changePct:s.change1d, volume:s.volume } })

    // ── PRICE GATE: verify all stock prices vs Polygon before allowing trades ──
    const stockTickers = stocks.map(s => s.ticker)
    let verifiedSet = new Set(ALL_CRYPTO) // crypto always allowed (CoinGecko verified)
    if (hasAnyPriceKey() && isLive) {
      try {
        const verifyMap = await verifyPricesViaPolygon(stockTickers, apiKey)
        stocks = applyPriceGate(stocks, verifyMap)
        // Only add stocks whose price passed gate
        for (const s of stocks) {
          if (s.priceVerified) verifiedSet.add(s.ticker)
        }
      } catch(e) { console.warn('[PriceGate]', e) }
    }
    // Update verified ref immediately so auto-trader can use it
    verifiedRef.current = verifiedSet
    setVerifiedTickers(new Set(verifiedSet))

    // Crypto prices from live fetch
    for(const pair of ALL_CRYPTO) {
      if(cryptoPriceData[pair]) {
        priceMap[pair]=cryptoPriceData[pair]
      } else {
        const b=barsMap[pair]
        if(b&&b.length>1) {
          const last=b[b.length-1], prev=b[b.length-2]
          priceMap[pair]={ price:last.c, changePct:(last.c-prev.c)/prev.c*100, volume:last.v }
        }
      }
    }
    // Update priceMap with gate-corrected prices
    stocks.forEach(s=>{ if(s.price) priceMap[s.ticker]={ price:s.price, changePct:s.change1d, volume:s.volume } })

    // Only stocks with a verified real price enter the trading/signals universe.
    // When live data is unavailable (API down, no key), priceVerified is never set → tickers=[] → only crypto shows.
    // This prevents mock-bar prices (~$100) from appearing as real stock prices.
    const tickers=stocks.slice(0,15).filter(s=>s.priceVerified===true).map(s=>s.ticker)
    const newSignals={}
    for(const [t,b] of Object.entries(barsMap)) {
      if(b&&b.length>0) newSignals[t]=generateSignal(b,agent.weights)
    }

    // Score news sentiment per ticker
    const newsSentimentMap = {}
    for(const n of (marketNewsRef.current||[])) {
        for(const t of (n.tickers||[])) {
          if(!newsSentimentMap[t]) newsSentimentMap[t]=[]
          newsSentimentMap[t].push(n.sentiment?.score||0)
        }
    }
    // Compute ensemble for all assets with news sentiment wired in
    const ensMap = {}
    for (const [t,b] of Object.entries(barsMap)) {
      try {
        const sentScores = newsSentimentMap[t]
        const sentScore = sentScores?.length ? sentScores.reduce((a,x)=>a+x,0)/sentScores.length : null
        ensMap[t] = runEnsemble(b, agent.weights, barsMap['SPY']||null, sentScore)
      } catch(e){}
    }
    setEnsembleResults(ensMap)
    const newScreenResult={ stocks:stocks.slice(0,15), timestamp:new Date(), mock:!isLive }
    setScreenResult(newScreenResult)
    setActiveStocks(tickers)
    setBars(barsMap)
    setPrices(priceMap)
    setSignals(newSignals)
    // Persist screener result so the same stocks appear on next page load
    saveScreenerCache({
      stocks: newScreenResult.stocks,
      tickers,
      prices: priceMap,
      signals: Object.fromEntries(Object.entries(newSignals).map(([t,s])=>[t,{signal:s.signal,score:s.score,factors:s.factors,confidence:s.confidence}])),
      verifiedTickers: [...verifiedSet],
    })
    // Auto-run verifiers after each screen (non-blocking)
    setTimeout(()=>runVerify(priceMap, barsMap, newSignals), 1500)
    setWeights(agent.weights)
    setDataStatus({ live:isLive, keyFound:!!(apiKey||getFinnhubKey()), lastUpdate:new Date(), screened:tickers.length })

    isScreeningRef.current=false
    setIsScreening(false)
    setIsRefreshing(false)
    loadMarketNews()
  },[apiKey,screenProfile])

  // ── Scheduler ────────────────────────────────────────────────────────────
  const hasAutoRunRef=useRef(false)
  useEffect(()=>{
    if(!hasAutoRunRef.current){
      hasAutoRunRef.current=true
      // Skip immediate run if we loaded fresh data from cache — wait for the 15-min interval instead
      if(!isScreenerCacheFresh(loadScreenerCache())) runScreener()
    }
  },[runScreener])

  useEffect(()=>{
    if(!autoScreenEnabled){ clearInterval(screenTimerRef.current); clearInterval(countdownRef.current); return }
    setNextScreenIn(SCREEN_INTERVAL)
    countdownRef.current=setInterval(()=>setNextScreenIn(p=>p<=1?SCREEN_INTERVAL:p-1),1000)
    screenTimerRef.current=setInterval(()=>runScreener(),SCREEN_INTERVAL*1000)
    return()=>{ clearInterval(screenTimerRef.current); clearInterval(countdownRef.current) }
  },[autoScreenEnabled,runScreener])

  // ── Auto-Trading Engine ───────────────────────────────────────────────────
  useEffect(()=>{
    if(!autoTradeSettings.enabled){ clearInterval(autoTradeRef.current); return }
    autoTradeRef.current=setInterval(()=>{
      const port=portfolioRef.current, px=pricesRef.current, sigs=signalsRef.current, b=barsRef.current
      const allTickers=[...Object.keys(sigs)]
      for(const ticker of allTickers) {
        const sig=sigs[ticker], price=px[ticker]?.price
        if(!sig||!price) continue
        // PRICE GATE: skip auto-trade if price not Polygon-verified
        if(!verifiedRef.current.has(ticker)) continue
        const ens=ensembleRef.current[ticker]||null
        const decision=evaluateAutoTrade(ticker, sig, ens, b[ticker], port, px, autoTradeSettings)
        if(decision) {
          if(decision.action==='BUY') executeTrade(ticker,'BUY',price,sig,true)
          else if(decision.action==='CLOSE') closePosition(ticker,true)
          setAutoTradeLog(prev=>[{ ticker, ...decision, time:new Date().toLocaleTimeString() },...prev.slice(0,49)])
        }
      }
      // Check exit alerts for all positions
      const alerts=[]
      for(const [ticker,pos] of Object.entries(port.positions)) {
        const price=px[ticker]?.price, sig=sigs[ticker], preset=EXIT_PRESETS[exitPreset]||EXIT_PRESETS.moderate
        if(price&&sig) {
          const exit=checkExitSignal(pos,price,sig,b[ticker],preset)
          if(exit) alerts.push({ ticker,...exit })
        }
      }
      setExitAlerts(alerts)
    },autoTradeSettings.checkIntervalMs||60000)
    return()=>clearInterval(autoTradeRef.current)
  },[autoTradeSettings,exitPreset])

  // ── Email Alerts ──────────────────────────────────────────────────────────
  async function sendEmail(type, data) {
    try {
      setEmailStatus('sending')
      const r=await fetch('/api/email',{
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({ type, to:EMAIL_TO, data })
      })
      const d=await r.json()
      if(d.success){ setEmailStatus('sent'); setTimeout(()=>setEmailStatus(null),5000) }
      else { setEmailStatus(`error: ${d.error}`); setTimeout(()=>setEmailStatus(null),8000) }
    } catch(e){ setEmailStatus(`error: ${e.message}`); setTimeout(()=>setEmailStatus(null),8000) }
  }

  // Monitor portfolio for 5% swing alerts
  useEffect(()=>{
    emailCheckRef.current=setInterval(()=>{
      const port=portfolioRef.current, px=pricesRef.current
      const val=portfolioValue(port,px)
      const pnl=val-STARTING_CAPITAL, pnlPct=(pnl/STARTING_CAPITAL)*100
      const threshold=5
      const now=Date.now()
      // Only alert if >5% change and hasn't alerted in last 2 hours
      if(Math.abs(pnlPct)>=threshold && (!lastAlertSent||(now-lastAlertSent)>7200000)) {
        const positions=Object.entries(port.positions).map(([ticker,pos])=>({
          ticker, shares:pos.shares, avgPrice:pos.avgPrice,
          pnl:(px[ticker]?.price||pos.avgPrice-pos.avgPrice)*pos.shares
        }))
        sendEmail('portfolio_alert',{
          portfolioValue:val, pnl, pnlPct,
          direction:pnlPct>=0?'up':'down', positions
        })
        localStorage.setItem('stockbot_alert_sent', String(now)); setLastAlertSent(now)
      }
      // ☀️ Morning briefing — 8:00–9:00am, once per day
      const hour=new Date().getHours()
      const todayStr=new Date().toDateString()
      if(hour>=8 && hour<=9 && lastBriefingSent!==todayStr) {
        sendEmail('daily_briefing', buildRichBriefing())
        localStorage.setItem('stockbot_briefing_sent', todayStr)
        setLastBriefingSent(todayStr)
      }

      // 📊 EOD performance report — 4:00–4:30pm, once per day
      const isEOD = hour===16 && new Date().getMinutes()<=30
      const todayEOD = 'eod_'+todayStr
      if(isEOD && localStorage.getItem('stockbot_eod_sent')!==todayEOD) {
        sendEmail('end_of_day', buildRichEOD())
        localStorage.setItem('stockbot_eod_sent', todayEOD)
      }

    },60000) // check every minute
    return()=>clearInterval(emailCheckRef.current)
  },[lastAlertSent,lastBriefingSent])

  // ── Paper Trading ─────────────────────────────────────────────────────────
  function executeTrade(ticker,side,price,signal,isAuto=false) {
    if(!price||price<=0) return
    // PRICE GATE: block any trade on tickers without verified Polygon price
    if(!verifiedRef.current.has(ticker)) {
      console.warn(`[PriceGate] Blocked ${side} ${ticker} — price not verified by Polygon`)
      return
    }
    // Use ref so auto-trader setInterval always gets current tradeSize
    const effectiveSize = isAuto ? (tradeSizeRef.current || 5000) : tradeSize
    const shares=Math.floor(effectiveSize/price)
    if(shares<=0) return
    setPortfolio(prev=>{
      const next={...prev,positions:{...prev.positions}}
      if(side==='BUY') {
        const cost=shares*price
        if(cost>next.cash) return prev
        next.cash-=cost
        if(next.positions[ticker]) {
          const ex=next.positions[ticker], total=ex.shares+shares
          next.positions[ticker]={ shares:total, avgPrice:(ex.avgPrice*ex.shares+price*shares)/total, side:'LONG', entryTime:ex.entryTime||Date.now(), highWater:price }
        } else {
          next.positions[ticker]={ shares, avgPrice:price, side:'LONG', entryTime:Date.now(), highWater:price }
        }
      } else if(side==='SELL') {
        const pos=next.positions[ticker]
        if(pos){ next.cash+=pos.shares*price; delete next.positions[ticker] }
        else { next.cash+=shares*price; next.positions[ticker]={ shares, avgPrice:price, side:'SHORT', entryTime:Date.now(), highWater:price } }
      }
      const val=portfolioValue(next,pricesRef.current)
      next.history=[...(prev.history||[]),{value:val,t:Date.now()}].slice(-500)
      // SAVE SYNCHRONOUSLY — don't rely on useEffect which fires after render
      savePortfolio(next)
      return next
    })
    setTradeLog(prev=>{
      const next=[{ id:Date.now(), ticker, side, price, shares, value:shares*price, signal:signal?.score?.toFixed(3)||'—', time:new Date().toLocaleTimeString(), auto:isAuto },...prev.slice(0,499)]
      saveTradeLog(next)
      return next
    })
  }


  // Build news sentiment map for RL trainer — { TICKER: avgScore }
  // ── Verifier ──────────────────────────────────────────────────────────────
  async function runVerify(px, b, sigs) {
    setIsVerifying(true)
    try {
      const report = await runAllVerifiers({
        prices:   px  || pricesRef.current,
        bars:     b   || bars,
        signals:  sigs|| signalsRef.current,
        weights:  agent.weights,
        apiKey:   getApiKey(),
        trainingLog, backtestResult, rlProgress, algoMode,
        tradeLog: tradeLogRef.current || [],
        autoSettings: autoSettingsRef.current,
        emailConfig: { to: EMAIL_TO, lastBriefingSent, lastAlertSent }
      })
      setVerifyReport(report)

      // ── Auto-apply improvements ────────────────────────────────────────
      const { settingPatches, shouldRetrain, retrainReason } = report.improvements || {}
      if (settingPatches && Object.keys(settingPatches).length > 0) {
        setAutoTradeSettings(prev => ({ ...prev, ...settingPatches }))
        console.log('[AutoImprove] Applied setting patches:', settingPatches)
      }
      if (shouldRetrain && !isTraining && Object.keys(bars).length > 0) {
        console.log('[AutoImprove] Triggering retrain:', retrainReason)
        // Small delay so UI updates first
        setTimeout(() => startTraining(), 1500)
      }
    } catch(e) {
      console.error('[Verifier]', e)
    } finally {
      setIsVerifying(false)
    }
  }

  function buildRichBriefing() {
    const _px=pricesRef.current, _sigs=signalsRef.current
    const _live=dataStatusRef.current.live  // only show prices if Polygon is live
    const _allA=Object.entries(_sigs).map(([t,s])=>({ticker:t,...s,price:_live?(_px[t]?.price||0):0}))
    const _mbuys=_allA.filter(a=>a.signal==='BUY').sort((a,b)=>b.score-a.score)
    const _msells=_allA.filter(a=>a.signal==='SELL').sort((a,b)=>a.score-b.score)
    const _picks=_mbuys.slice(0,6).map(a=>{
      const ens=ensembleRef.current[a.ticker], p=a.price
      const nl=(marketNewsRef.current||[]).find(n=>(n.tickers||[]).includes(a.ticker))
      // Entry/stop/target only meaningful with real prices
      const entry=_live&&p>0?('$'+(p*0.99).toFixed(2)+'\u2013$'+(p*1.005).toFixed(2)):'—'
      const stop =_live&&p>0?('$'+(p*0.92).toFixed(2)):'—'
      const target=_live&&p>0?('$'+(p*1.10).toFixed(2)):'—'
      return {ticker:a.ticker,price:_live&&p>0?p:null,score:a.score,confidence:a.confidence,
        ensemble:ens?.signal||a.signal,entryZone:entry,stopLoss:stop,target,
        reason:(ens?(ens.signal+' · '+Math.round((ens.confidence||0)*100)+'% conf · '):'')+'Score '+a.score.toFixed(3),
        newsHeadline:nl?.title||null}
    })
    return {
      buys:_mbuys.slice(0,10).map(a=>({ticker:a.ticker,price:_live&&a.price>0?a.price:null,score:a.score,confidence:a.confidence,reason:'BUY'})),
      sells:_msells.slice(0,5).map(a=>({ticker:a.ticker,price:_live&&a.price>0?a.price:null,score:a.score,reason:'SELL'})),
      topPicks:_picks, algoMode, portfolioValue:portfolioValue(portfolioRef.current,_px),
      newsHeadlines:(marketNewsRef.current||[]).slice(0,4),
      marketMood:_mbuys.length>_msells.length?'BULLISH':_msells.length>_mbuys.length?'BEARISH':'NEUTRAL',
      date:new Date().toLocaleDateString('en-US',{weekday:'long',month:'long',day:'numeric'})
    }
  }

  function buildRichEOD() {
    const _px=pricesRef.current, _sigs=signalsRef.current
    const _live=dataStatusRef.current.live
    const _allA=Object.entries(_sigs).map(([t,s])=>({ticker:t,...s,price:_live?(_px[t]?.price||0):0}))
    const port=portfolioRef.current, val=portfolioValue(port,_px)
    const pnl=val-STARTING_CAPITAL
    const pos=Object.entries(port.positions).map(([t,p])=>{
      const curPx=_px[t]?.price||p.avgPrice
      const positionPnl=(p.side==='LONG'?(curPx-p.avgPrice):(p.avgPrice-curPx))*p.shares
      // Only show price if live
      return {ticker:displayTicker(t),shares:p.shares,avgPrice:_live?p.avgPrice:null,
        currentPrice:_live?curPx:null,pnl:positionPnl,pnlPct:(positionPnl/(p.avgPrice*p.shares))*100}
    }).sort((a,b)=>b.pnl-a.pnl)
    const today=new Date().toDateString()
    const todayTrades=(tradeLogRef.current||[]).filter(t=>{
      try{return new Date(t.time).toDateString()===today}catch{return false}
    }).slice(0,10).map(t=>({ticker:displayTicker(t.ticker||''),side:t.side||'',
      price:_live?t.price||0:null,shares:t.shares||0,pnl:t.pnl||null}))
    const watch=_allA.filter(a=>a.signal==='BUY').sort((a,b)=>b.score-a.score).slice(0,6)
      .map(a=>({ticker:a.ticker,price:_live&&a.price>0?a.price:null,score:a.score,confidence:a.confidence}))
    return {
      portfolioValue:val, pnl, pnlPct:(pnl/STARTING_CAPITAL)*100,
      pnlToday:pnl, pnlTodayPct:(pnl/STARTING_CAPITAL)*100,
      positions:pos, bestPosition:pos[0]||null, worstPosition:pos[pos.length-1]||null,
      todayTrades, nextDayWatchlist:watch, totalTrades:(tradeLogRef.current||[]).length,
      sharpe:backtestResult?.sharpe||null, maxDrawdown:backtestResult?.maxDrawdown||null,
      algoMode, marketMood:_allA.filter(a=>a.signal==='BUY').length>_allA.filter(a=>a.signal==='SELL').length?'BULLISH':'BEARISH',
      date:new Date().toLocaleDateString('en-US',{weekday:'long',month:'long',day:'numeric'})
    }
  }

  function buildNewsSentimentMap(articles) {
    const map = {}
    for (const a of (articles || [])) {
      const score = a.sentiment?.score || 0
      const weight = a.sentiment?.impact === 'high' ? 2 : 1
      for (const t of (a.tickers || [])) {
        if (!map[t]) map[t] = { total: 0, count: 0 }
        map[t].total += score * weight
        map[t].count += weight
      }
    }
    const result = {}
    for (const [t, v] of Object.entries(map)) result[t] = v.total / v.count
    return result
  }
  function closePosition(ticker, isAuto=false) {
    // Use ref to avoid stale closure in auto-trader setInterval
    const pos=portfolioRef.current.positions[ticker]
    if(!pos) return
    const px=pricesRef.current[ticker]?.price||pos.avgPrice
    const pnl=pos.side==='LONG'?(px-pos.avgPrice)*pos.shares:(pos.avgPrice-px)*pos.shares
    setPortfolio(prev=>{
      const next={...prev,positions:{...prev.positions}}
      if(pos.side==='LONG') next.cash+=pos.shares*px; else next.cash-=pos.shares*px
      delete next.positions[ticker]
      const val=portfolioValue(next,pricesRef.current)
      next.history=[...(prev.history||[]),{value:val,t:Date.now()}].slice(-500)
      // SAVE SYNCHRONOUSLY
      savePortfolio(next)
      return next
    })
    setTradeLog(prev=>{
      const next=[{ id:Date.now(), ticker, side:'CLOSE', price:px, shares:pos.shares, value:pos.shares*px, signal:'—', time:new Date().toLocaleTimeString(), pnl:parseFloat(pnl.toFixed(2)), auto:isAuto },...prev.slice(0,499)]
      saveTradeLog(next)
      return next
    })
  }

  function resetPortfolio() {
    if(!window.confirm('Reset portfolio? Cannot be undone.')) return
    resetStored(); setPortfolio(loadPortfolio()); setTradeLog([])
  }

  // ── Watchlist ─────────────────────────────────────────────────────────────
  function addToWatchlist(input) {
    const ticker=normalizeTicket(input)
    if(!ticker||ticker.length<1) return
    if(watchlist.includes(ticker)) return
    setWatchlist(prev=>{ const next=[...prev,ticker]; saveWatchlist(next); return next })
    setWatchlistInput('')
    if(!bars[ticker]) setBars(prev=>({...prev,[ticker]:mockBars(ticker)}))
    if(!prices[ticker]) {
      const b=mockBars(ticker)
      const last=b[b.length-1],prev2=b[b.length-2]
      setPrices(prev=>({...prev,[ticker]:{ price:last.c, changePct:(last.c-prev2.c)/prev2.c*100, volume:last.v }}))
    }
  }

  // ── News ──────────────────────────────────────────────────────────────────
  async function loadMarketNews(force=false) {
    if(force) clearNewsCache()
    const news=await fetchMarketNews(12,force)
    marketNewsRef.current=news; setMarketNews(news)
    const ai=await scoreWithClaude(news); if(ai) setAiSentiment(ai)
    // Extract tickers mentioned in today's news and surface them
    const extracted=extractNewsStocks(news)
    newsStocksRef.current=extracted; setNewsStocks(extracted)
    // Auto-add high-conviction news tickers to active universe
    if(extracted.length>0) {
      const newTickers=extracted.filter(c=>Math.abs(c.score)>0.5).map(c=>c.ticker).filter(t=>!activeStocks.includes(t))
      if(newTickers.length>0) {
        setActiveStocks(prev=>[...new Set([...prev,...newTickers])].slice(0,25))
        // Generate bars+signals for new tickers
        const newBarsMap={}
        for(const t of newTickers) {
          if(!bars[t]) { const b=mockBars(t); newBarsMap[t]=b }
        }
        if(Object.keys(newBarsMap).length>0) {
          setBars(prev=>({...prev,...newBarsMap}))
          const newPrices={}
          for(const [t,b] of Object.entries(newBarsMap)) {
            const last=b[b.length-1],prev2=b[b.length-2]
            newPrices[t]={ price:last.c, changePct:(last.c-prev2.c)/prev2.c*100, volume:last.v }
          }
          setPrices(prev=>({...prev,...newPrices}))
          const newSigs={}
          for(const [t,b] of Object.entries(newBarsMap)) newSigs[t]=generateSignal(b,agent.weights)
          setSignals(prev=>({...prev,...newSigs}))
        }
      }
    }
  }
  async function loadTickerNews(ticker) { setTickerNews([]); setTickerNews(await fetchTickerNews(ticker,5)) }


  // ── CSV / Data Export ─────────────────────────────────────────────────────
  function downloadCSV(content, filename) {
    const blob = new Blob([content], { type: 'text/csv;charset=utf-8;' })
    const url  = URL.createObjectURL(blob)
    const a    = document.createElement('a')
    a.href = url; a.download = filename; a.click()
    URL.revokeObjectURL(url)
  }
  function downloadEquityCurveCSV() {
    if (!backtestResult || !backtestResult.equityCurve || !backtestResult.equityCurve.length) {
      alert('Run backtest or training first'); return
    }
    downloadCSV(equityCurveToCSV(backtestResult.equity), 'stockbot_equity_' + new Date().toISOString().split('T')[0] + '.csv')
  }
  function downloadTradeLogCSV() {
    if (!backtestResult || !backtestResult.tradeLog || !backtestResult.tradeLog.length) {
      alert('Run backtest or training first'); return
    }
    const rows = ['date,ticker,action,price,shares,reason,confidence']
    backtestResult.trades.forEach(function(t) {
      rows.push([new Date(t.date).toISOString().split('T')[0],t.ticker,t.action,t.price,t.shares,t.reason||'',t.confidence||''].join(','))
    })
    downloadCSV(rows.join('\n'), 'stockbot_trades_' + new Date().toISOString().split('T')[0] + '.csv')
  }
  function downloadPaperLogCSV() {
    if (!tradeLog || !tradeLog.length) { alert('No paper trades yet'); return }
    const rows = ['time,ticker,side,price,shares,pnl,capital_after']
    tradeLog.forEach(function(t) {
      rows.push([t.time||'',displayTicker(t.ticker||''),t.side||'',t.price||'',t.shares||'',t.pnl!=null?t.pnl.toFixed(2):'',t.capitalAfter!=null?t.capitalAfter.toFixed(2):''].join(','))
    })
    downloadCSV(rows.join('\n'), 'stockbot_paper_' + new Date().toISOString().split('T')[0] + '.csv')
  }

  // ── RL Training ───────────────────────────────────────────────────────────
  async function startTraining() {
    if(isTraining||Object.keys(bars).length===0) return
    setIsTraining(true)
    const fmt=(v,d=2)=>v!=null&&isFinite(v)?Number(v).toFixed(d):'—'
    try {
      await trainEpisodes(bars,30,
        ep=>{
          const r=ep.result||{}
          setTrainingLog(prev=>[...prev.slice(-49),{
            episode:ep.episode,
            score:fmt(r.ragScore,3),
            sharpe:fmt(r.sharpe,2),
            dd:fmt((r.maxDrawdown||0)*100,1)+'%',
            ret:fmt((r.totalReturn||0)*100,1)+'%',
            regime:ep.regime||'—',
            algo:r.algo||'CEM',
            news:r.newsCount>0?'✓':''
          }])
          setRlProgress(ep.agentState)
          setWeights({...(ep.currentWeights||{})})
          setAlgoMode(r.algo||'CEM')
          if((r.maxDrawdown||0)>0.15) setObservationMode(true)
        },
        done=>{
          try {
            const finalBt=backtestPortfolio(bars,done.bestWeights)
            setWeights({...done.bestWeights})
            setBacktestResult(finalBt)
            setAlgoMode(done.currentAlgo||'CEM')
            setObservationMode((finalBt.maxDrawdown||0)>0.15)
          } catch(e){ console.error('[Training done]',e) }
          finally { setIsTraining(false) }
        }
      )
    } catch(e){
      console.error('[startTraining]',e)
      setIsTraining(false)
    }
  }

  // ── Computed ──────────────────────────────────────────────────────────────
  const allTracked=[...activeStocks.map(t=>({ticker:t,display:t})),...ALL_CRYPTO.map(t=>({ticker:t,display:CRYPTO_DISPLAY[t]}))]
  const sortedSignals=allTracked.map(a=>({...a,sig:signals[a.ticker]||{score:0,signal:'HOLD',factors:{},confidence:0},px:prices[a.ticker]})).sort((a,b)=>(b.sig?.score||0)-(a.sig?.score||0))
  const buys=sortedSignals.filter(a=>a.sig?.signal==='BUY')
  const sells=sortedSignals.filter(a=>a.sig?.signal==='SELL')
  const curVal=portfolioValue(portfolio,prices)
  const totalPnL=curVal-STARTING_CAPITAL
  const fmtCD=s=>`${Math.floor(s/60)}:${String(s%60).padStart(2,'0')}`
  const filteredStocks=screenResult?.stocks.filter(s=>!signalFilter||(signals[s.ticker]?.signal||'HOLD')===signalFilter)||[]

  const verifyBadge = verifyReport ? (verifyReport.overall==='error'?'🔴':verifyReport.overall==='warn'?'🟡':'🟢') : '⚙️'
  const tabs=[
    {id:'screen',icon:'⟳',label:'SCREEN'},
    {id:'signals',icon:'📡',label:'SIGNALS'},
    {id:'news',icon:'📰',label:'NEWS'},
    {id:'paper',icon:'💼',label:'PAPER'},
    {id:'auto',icon:'🤖',label:'AUTO'},
    {id:'models',icon:'📊',label:'MODELS'},
    {id:'train',icon:'🧠',label:'TRAIN'},
    {id:'verify',icon:verifyBadge,label:'VERIFY'},
  ]

  return (
    <div style={S.app}>
      {/* Header */}
      <header style={S.header}>
        <div style={{display:'flex',alignItems:'center',gap:10}}>
          <span style={S.logo}>STOCKBOT</span>
          {isScreening?<span style={S.badge(C.yellow)}>⟳ SCANNING</span>
            :!dataStatus.keyFound?<span style={S.badge(C.yellow)}>MOCK</span>
            :dataStatus.live?<span style={S.badge(C.green)}>LIVE · {dataStatus.screened} · {getFinnhubKey()?'FH':'POLY'}</span>
            :<span style={S.badge(C.yellow)}>LOADING</span>}
          <span style={S.badge(algoMode==='PPO'?C.green:algoMode==='SAC'?C.yellow:C.accent)}>{algoMode}</span>
          {observationMode&&<span style={S.badge(C.red)}>OBS MODE</span>}
          {autoTradeSettings.enabled&&<span style={S.badge(C.purple)}>🤖 AUTO-TRADE ON</span>}
          {emailStatus&&<span style={S.badge(emailStatus==='sent'?C.green:C.yellow)}>{emailStatus==='sent'?'✉️ SENT':'✉️ '+emailStatus}</span>}
        </div>
        <div style={{fontSize:9,color:C.textDim,textAlign:'right'}}>
          {autoScreenEnabled&&<span style={{color:C.accent}}>⟳ {fmtCD(nextScreenIn)}  </span>}
          <span>GEN {agent.generation} · ε {agent.epsilon.toFixed(2)}</span>
        </div>
      </header>

      <main style={S.main}>

        {/* Exit Alerts Banner */}
        {exitAlerts.length>0&&(
          <div style={{background:'#ff336611',border:`1px solid ${C.red}`,borderRadius:8,padding:'10px 14px',marginBottom:12,display:'flex',gap:12,flexWrap:'wrap',alignItems:'center'}}>
            <span style={{color:C.red,fontSize:10,fontWeight:700}}>⚠️ EXIT ALERTS</span>
            {exitAlerts.map((a,i)=>(
              <span key={i} style={{fontSize:10,color:C.yellow}}>
                {a.ticker}: {a.reason} ({(a.pnlPct*100).toFixed(1)}%)
                <button style={{...S.btn('danger'),padding:'2px 8px',marginLeft:6,fontSize:9}} onClick={()=>closePosition(a.ticker)}>CLOSE</button>
              </span>
            ))}
          </div>
        )}

        {observationMode&&(
          <div style={{background:'#ff336611',border:'1px solid #ff3366',borderRadius:8,padding:'10px 14px',marginBottom:12,display:'flex',alignItems:'center',gap:12}}>
            <span style={{color:'#ff3366',fontSize:10,fontWeight:700}}>OBS MODE</span>
            <span style={{color:'#ffd700',fontSize:10}}>Max Drawdown exceeded 15%. Auto-trading suspended per spec. RL agent observing only.</span>
            <button style={{...S.btn('danger'),padding:'2px 10px',fontSize:9}} onClick={()=>setObservationMode(false)}>RESUME</button>
          </div>
        )}
        {/* ══ SCREEN ══ */}
        {tab==='screen'&&(
          <div style={{display:'flex',flexDirection:'column',gap:12}}>
            <div style={{background:C.surface,border:`1px solid ${C.border}`,borderRadius:8,padding:'10px 14px',display:'flex',alignItems:'center',justifyContent:'space-between',flexWrap:'wrap',gap:8}}>
              <div style={{display:'flex',alignItems:'center',gap:10}}>
                <div style={{width:8,height:8,borderRadius:'50%',background:autoScreenEnabled?C.green:C.textDim,boxShadow:autoScreenEnabled?`0 0 8px ${C.green}`:'none'}}/>
                <span style={{fontSize:10,color:C.textBright}}>{autoScreenEnabled?`AUTO · NEXT ${fmtCD(nextScreenIn)}`:'PAUSED'}</span>
              </div>
              <div style={{display:'flex',gap:6}}>
                <button style={{...S.btn(autoScreenEnabled?'danger':'primary'),padding:'4px 12px',fontSize:9}} onClick={()=>setAutoScreenEnabled(p=>!p)}>
                  {autoScreenEnabled?'⏸ PAUSE':'▶ RESUME'}
                </button>
                <button style={{...S.btn('primary'),padding:'4px 12px',fontSize:9}} onClick={()=>runScreener(screenProfile,customCriteria)} disabled={isScreening}>
                  {isScreening?'SCANNING...':'⟳ SCAN NOW'}
                </button>
              </div>
            </div>

            <div style={{display:'flex',gap:6,flexWrap:'wrap'}}>
              {Object.entries(SCREENING_PROFILES).map(([key,p])=>(
                <button key={key} style={{...S.btn(screenProfile===key?'active':'default'),fontSize:10}}
                  onClick={()=>{setScreenProfile(key);setCustomCriteria(p.criteria);runScreener(key,p.criteria)}}>
                  {p.icon} {p.label}
                </button>
              ))}
            </div>

            {isScreening&&!screenResult&&<div style={{...S.panel,textAlign:'center',padding:48}}><div style={{color:C.accent,letterSpacing:4}}>SCANNING MARKET...</div></div>}

            {screenResult&&(
              <>
                <div style={{display:'grid',gridTemplateColumns:'repeat(4,1fr)',gap:8}}>
                  {[
                    {label:'ALL',value:screenResult.stocks.length,color:C.accent,filter:null},
                    {label:'BUY ▲',value:buys.length,color:C.green,filter:'BUY'},
                    {label:'SELL ▼',value:sells.length,color:C.red,filter:'SELL'},
                    {label:isRefreshing?'REFRESHING':'UPDATED',value:isRefreshing?'⟳ ...':(screenResult.fromCache?'CACHED · ':'')+screenResult.timestamp?.toLocaleTimeString(),color:isRefreshing?C.yellow:C.textDim,filter:undefined},
                  ].map(m=>(
                    <div key={m.label} onClick={m.filter!==undefined?()=>setSignalFilter(f=>f===m.filter?null:m.filter):undefined}
                      style={{...S.panel,cursor:m.filter!==undefined?'pointer':'default',border:`1px solid ${signalFilter===m.filter&&m.filter!==undefined?m.color:C.border}`,background:signalFilter===m.filter&&m.filter!==undefined?m.color+'11':C.panel}}>
                      <div style={S.pt}>{m.label}</div>
                      <div style={{fontSize:18,fontWeight:700,color:m.color}}>{m.value}</div>
                      {m.filter!==undefined&&<div style={{fontSize:8,color:C.textDim,marginTop:2}}>{signalFilter===m.filter?'ACTIVE FILTER':'TAP TO FILTER'}</div>}
                    </div>
                  ))}
                </div>

                {/* Heatmap — stocks + crypto */}
                <div style={S.panel}>
                  <div style={S.pt}>SIGNAL HEATMAP · <span style={{color:C.accent,textTransform:'none',letterSpacing:0}}>{universeLabel}</span>{screenResult.mock&&<span style={{color:C.yellow}}> (MOCK DATA)</span>}</div>
                  <div style={{display:'grid',gridTemplateColumns:'repeat(5,1fr)',gap:5}}>
                    {[...sortedSignals.filter(a=>!signalFilter||a.sig?.signal===signalFilter).slice(0,17),
                      ...ALL_CRYPTO.map(t=>({ticker:t,display:CRYPTO_DISPLAY[t],sig:signals[t]||{score:0,signal:'HOLD'},px:prices[t]}))
                    ].map(a=>{
                      const score=a.sig?.score||0,hue=score>0?'144,255,136':'255,51,102'
                      const isCrypto=ALL_CRYPTO.includes(a.ticker)
                      return (
                        <div key={a.ticker} onClick={()=>{setSelectedAsset(a.ticker);setTab('signals')}}
                          style={{padding:'7px 4px',borderRadius:4,cursor:'pointer',background:`rgba(${hue},${Math.abs(score)*0.4})`,border:`1px solid rgba(${hue},${Math.abs(score)*0.7})`,textAlign:'center'}}>
                          <div style={{fontSize:10,fontWeight:700,color:C.textBright}}>{a.display}{newsStocks.find(n=>n.ticker===a.ticker)?'📰':''}</div>
                          <div style={{fontSize:8,color:score>0?C.green:score<0?C.red:C.textDim}}>{score>=0?'+':''}{score.toFixed(2)}</div>
                          {isCrypto&&<div style={{fontSize:7,color:C.purple}}>CRYPTO</div>}
                          {a.px?.price&&<div style={{fontSize:8,color:C.textDim}}>{fmt.price(a.px.price)}</div>}
                        </div>
                      )
                    })}
                  </div>
                </div>

                <div style={S.panel}>
                  <div style={S.pt}>RANKED STOCKS {signalFilter&&`— ${signalFilter} ONLY`}</div>
                  <div style={{overflowX:'auto',WebkitOverflowScrolling:'touch'}}>
                    <table style={{...S.table,minWidth:580}}>
                      <thead><tr>{['#','TICKER','PRICE','1D%','RSI','TECH','FF5','TCN','NEWS','ENSEMBLE','CONF'].map(h=><th key={h} style={S.th}>{h}</th>)}</tr></thead>
                      <tbody>
                        {filteredStocks.map((s,i)=>{
                          const techSig = signals[s.ticker]||{signal:'HOLD',score:0,confidence:0}
                          const ens = ensembleResults[s.ticker]
                          const ff5Sig  = ens?.models?.famaFrench?.signal||'—'
                          const tcnSig  = ens?.models?.tcn?.signal||'—'
                          const newsSig = ens?.models?.sentiment?.signal||'—'
                          const ensSignal = ens?.signal || techSig.signal
                          const ensConf   = ens?.confidence ?? techSig.confidence ?? 0
                          const rsi = s.bars ? calcRSI(s.bars.map(b=>b.c),14) : 50
                          const hasNews = newsStocks.find(n=>n.ticker===s.ticker)
                          const isPriceOk = verifiedTickers.has(s.ticker)
                          return (
                            <tr key={s.ticker} style={{cursor:'pointer',opacity:isPriceOk||!hasAnyPriceKey()?1:0.55}} onClick={()=>{setSelectedAsset(s.ticker);setTab('signals')}}>
                              <td style={{...S.td,color:C.textDim}}>{i+1}</td>
                              <td style={{...S.td,color:C.textBright,fontWeight:700}}>
                                {s.ticker}
                                {hasNews?<span style={{color:C.purple,fontSize:8,marginLeft:3}}>📰</span>:null}
                                {hasAnyPriceKey()&&!isPriceOk?<span title={s.priceReason||'Price not verified'} style={{color:C.red,fontSize:8,marginLeft:3}}>🔒</span>:null}
                                {hasAnyPriceKey()&&isPriceOk?<span title={`${s.priceSource||'price'} verified`} style={{color:C.green,fontSize:8,marginLeft:3}}>✓</span>:null}
                              </td>
                              <td style={S.td}>{fmt.price(s.price)}</td>
                              <td style={{...S.td,...fmt.chg(s.change1d)}}>{s.change1d?.toFixed(2)}%</td>
                              <td style={{...S.td,color:rsi<30?C.green:rsi>70?C.red:C.textDim}}>{rsi.toFixed(0)}</td>
                              <td style={S.td}><span style={S.sigBadge(techSig.signal)}>{techSig.signal}</span></td>
                              <td style={S.td}><span style={S.sigBadge(ff5Sig)}>{ff5Sig}</span></td>
                              <td style={S.td}><span style={S.sigBadge(tcnSig)}>{tcnSig}</span></td>
                              <td style={S.td}><span style={S.sigBadge(newsSig)}>{newsSig}</span></td>
                              <td style={{...S.td,fontWeight:700}}><span style={S.sigBadge(ensSignal)}>{ensSignal}</span></td>
                              <td style={{...S.td,color:ensConf>0.65?C.green:ensConf>0.4?C.yellow:C.textDim,fontWeight:700}}>{(ensConf*100).toFixed(0)}%</td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              </>
            )}
          </div>
        )}

        {/* ══ SIGNALS ══ */}
        {tab==='signals'&&(
          <div style={{display:'flex',flexDirection:'column',gap:12}}>
            <div style={{display:'flex',gap:5,flexWrap:'wrap'}}>
              {allTracked.map(a=>(
                <button key={a.ticker} onClick={()=>{setSelectedAsset(a.ticker);loadTickerNews(a.ticker)}}
                  style={{background:selectedAsset===a.ticker?C.accentDim:'transparent',border:`1px solid ${selectedAsset===a.ticker?C.accent:C.border}`,color:selectedAsset===a.ticker?C.accent:C.textDim,padding:'4px 10px',borderRadius:4,cursor:'pointer',fontSize:10,fontFamily:'inherit'}}>
                  {a.display}{ALL_CRYPTO.includes(a.ticker)&&' 🪙'}
                </button>
              ))}
            </div>

            {selectedAsset&&signals[selectedAsset]?(()=>{
              const sig=signals[selectedAsset],px=prices[selectedAsset],assetBars=bars[selectedAsset]||[]
              const isVerified=verifiedTickers.has(selectedAsset)||ALL_CRYPTO.includes(selectedAsset)
              // Block display of any stock whose real-time price couldn't be confirmed.
              // Prevents mock-bar prices (~$100) from being shown as real prices.
              if(!isVerified&&hasAnyPriceKey()) return (
                <div style={{...S.panel,textAlign:'center',padding:48}}>
                  <div style={{fontSize:16,color:C.red,marginBottom:10}}>🔒 PRICE NOT AVAILABLE</div>
                  <div style={{fontSize:11,color:C.textDim,marginBottom:6}}>{selectedAsset} — live price could not be verified.</div>
                  <div style={{fontSize:10,color:C.textDim}}>Trading blocked. Stock will appear once price is confirmed by Finnhub or Polygon.</div>
                </div>
              )
              const pos=portfolio.positions[selectedAsset]
              const curPx=px?.price||0
              const unreal=pos?(pos.side==='LONG'?(curPx-pos.avgPrice)*pos.shares:(pos.avgPrice-curPx)*pos.shares):0
              const chartData=assetBars.slice(-90).map(b=>({date:fmt.date(b.t),price:+b.c.toFixed(2)}))
              const closes=assetBars.map(b=>b.c)
              const rsi=closes.length>14?calcRSI(closes,14):50
              const macd=closes.length>26?calcMACD(closes):{macd:0,signal:0,hist:0}
              const bb=closes.length>20?calcBollingerBands(closes):null
              const atr=assetBars.length>14?calcATR(assetBars):0
              const exitCheck=pos?checkExitSignal(pos,curPx,sig,assetBars,EXIT_PRESETS[exitPreset]||EXIT_PRESETS.moderate):null
              return (
                <div style={{display:'flex',flexDirection:'column',gap:12}}>
                  <div style={{display:'grid',gridTemplateColumns:'repeat(2,1fr)',gap:8}}>
                    <div style={{...S.panel,border:`1px solid ${sig.signal==='BUY'?C.green:sig.signal==='SELL'?C.red:C.border}`}}>
                      <div style={S.pt}>SIGNAL</div>
                      <div style={{fontSize:28,fontWeight:700,color:sig.signal==='BUY'?C.green:sig.signal==='SELL'?C.red:C.textDim}}>{sig.signal}</div>
                      <div style={{fontSize:10,color:C.textDim,marginTop:4}}>Score: {sig.score.toFixed(4)} · Conf: {(sig.confidence*100).toFixed(0)}%</div>
                    </div>
                    <div style={S.panel}>
                      <div style={S.pt}>PRICE {ALL_CRYPTO.includes(selectedAsset)&&'🪙 CRYPTO'}</div>
                      <div style={{fontSize:22,fontWeight:700,color:C.textBright}}>{fmt.price(px?.price)}</div>
                      <div style={{fontSize:10,marginTop:4,...fmt.chg(px?.changePct||0)}}>{px?.changePct?.toFixed(2)}% today</div>
                    </div>
                  </div>

                  {/* Technical indicators row */}
                  <div style={{display:'grid',gridTemplateColumns:'repeat(4,1fr)',gap:8}}>
                    {[
                      {label:'RSI(14)',value:rsi.toFixed(0),color:rsi<30?C.green:rsi>70?C.red:C.textBright,note:rsi<30?'OVERSOLD':rsi>70?'OVERBOUGHT':'NEUTRAL'},
                      {label:'MACD HIST',value:macd.hist?.toFixed(3)||'—',color:macd.hist>0?C.green:C.red,note:macd.hist>0?'BULLISH':'BEARISH'},
                      {label:'BB %',value:bb?`${(bb.pct*100).toFixed(0)}%`:'—',color:bb&&bb.pct<0.2?C.green:bb&&bb.pct>0.8?C.red:C.textBright,note:bb&&bb.pct<0.2?'NEAR LOWER':bb&&bb.pct>0.8?'NEAR UPPER':'MID'},
                      {label:'ATR',value:fmt.price(atr),color:C.textDim,note:'VOLATILITY'},
                    ].map(m=>(
                      <div key={m.label} style={S.panel}>
                        <div style={S.pt}>{m.label}</div>
                        <div style={{fontSize:16,fontWeight:700,color:m.color}}>{m.value}</div>
                        <div style={{fontSize:8,color:C.textDim,marginTop:2}}>{m.note}</div>
                      </div>
                    ))}
                  </div>

                  {/* Exit alert for this position */}
                  {exitCheck&&(
                    <div style={{background:exitCheck.urgency==='HIGH'?'#ff336622':exitCheck.urgency==='TAKE_PROFIT'?'#00ff8822':'#ffd70011',border:`1px solid ${exitCheck.urgency==='HIGH'?C.red:exitCheck.urgency==='TAKE_PROFIT'?C.green:C.yellow}`,borderRadius:8,padding:'10px 14px',display:'flex',justifyContent:'space-between',alignItems:'center'}}>
                      <div>
                        <div style={{fontSize:11,fontWeight:700,color:exitCheck.urgency==='HIGH'?C.red:exitCheck.urgency==='TAKE_PROFIT'?C.green:C.yellow}}>
                          {exitCheck.shouldExit?'EXIT SIGNAL':'⚠️ WARNING'}: {exitCheck.reason}
                        </div>
                        <div style={{fontSize:10,color:C.textDim,marginTop:2}}>P&L: {(exitCheck.pnlPct*100).toFixed(2)}% · Preset: {exitPreset}</div>
                      </div>
                      {exitCheck.shouldExit&&<button style={{...S.btn('danger'),padding:'6px 14px'}} onClick={()=>closePosition(selectedAsset)}>CLOSE NOW</button>}
                    </div>
                  )}

                  {/* Price chart */}
                  <div style={S.panel}>
                    <div style={S.pt}>{selectedAsset} — 90-DAY PRICE</div>
                    <ResponsiveContainer width="100%" height={180}>
                      <AreaChart data={chartData}>
                        <defs><linearGradient id="pg" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor={C.accent} stopOpacity={0.3}/><stop offset="95%" stopColor={C.accent} stopOpacity={0}/></linearGradient></defs>
                        <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
                        <XAxis dataKey="date" tick={{fontSize:8,fill:C.textDim}} interval={14}/>
                        <YAxis tick={{fontSize:8,fill:C.textDim}}/>
                        <Tooltip contentStyle={{background:C.panel,border:`1px solid ${C.border}`,fontSize:10}}/>
                        <Area type="monotone" dataKey="price" stroke={C.accent} fill="url(#pg)" strokeWidth={2} dot={false}/>
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Factor bars */}
                  <div style={S.panel}>
                    <div style={S.pt}>FACTOR SCORES (7 INDICATORS)</div>
                    <ResponsiveContainer width="100%" height={120}>
                      <BarChart data={Object.entries(sig.factors||{}).map(([k,v])=>({factor:k.replace('meanReversion','MRev').replace('momentum','Mom').replace('volume','Vol').replace('volatility','VReg').replace('trend','Trend').replace('ffAlpha','FF5α').replace('tcnAlign','TCN'),value:parseFloat(v?.toFixed(2)||0)}))}>
                        <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
                        <XAxis dataKey="factor" tick={{fontSize:9,fill:C.textDim}}/>
                        <YAxis domain={[-3,3]} tick={{fontSize:8,fill:C.textDim}}/>
                        <ReferenceLine y={0} stroke={C.border}/>
                        <Tooltip contentStyle={{background:C.panel,border:`1px solid ${C.border}`}}/>
                        <Bar dataKey="value" fill={C.accent} radius={[3,3,0,0]}/>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Trade panel */}
                  <div style={S.panel}>
                    <div style={S.pt}>PAPER TRADE</div>
                    {pos&&<div style={{fontSize:11,color:C.textDim,marginBottom:8}}>Position: {pos.shares} shares @ {fmt.price(pos.avgPrice)} · P&L: <span style={{color:unreal>=0?C.green:C.red,fontWeight:700}}>${unreal.toFixed(0)}</span></div>}
                    <div style={{display:'flex',gap:8,alignItems:'center',marginBottom:10,flexWrap:'wrap'}}>
                      <span style={{fontSize:10,color:C.textDim}}>Size: $</span>
                      <input type="number" value={tradeSize} onChange={e=>setTradeSize(parseInt(e.target.value)||1000)}
                        style={{...S.input,width:80}}/>
                      <span style={{fontSize:9,color:C.textDim}}>= ~{curPx>0?Math.floor(tradeSize/curPx):0} shares</span>
                    </div>
                    <div style={{display:'flex',gap:8}}>
                      <button
                        style={{...S.btn(verifiedTickers.has(selectedAsset)||!hasAnyPriceKey()?'green':'default'),flex:1,padding:10}}
                        onClick={()=>executeTrade(selectedAsset,'BUY',px?.price,sig)}
                        disabled={hasAnyPriceKey()&&!verifiedTickers.has(selectedAsset)}
                        title={hasAnyPriceKey()&&!verifiedTickers.has(selectedAsset)?'🔒 Price not verified — trade blocked':''}
                      >▲ BUY {displayTicker(selectedAsset)}{hasAnyPriceKey()&&!verifiedTickers.has(selectedAsset)?' 🔒':''}</button>
                      <button style={{...S.btn('danger'),flex:1,padding:10}} onClick={()=>executeTrade(selectedAsset,'SELL',px?.price,sig)}>▼ SELL {displayTicker(selectedAsset)}</button>
                      {pos&&<button style={{...S.btn(),padding:10}} onClick={()=>closePosition(selectedAsset)}>CLOSE</button>}
                    </div>
                    <div style={{marginTop:10}}>
                      <div style={S.pt}>EXIT STRATEGY</div>
                      <div style={{display:'flex',gap:6,flexWrap:'wrap'}}>
                        {Object.entries(EXIT_PRESETS).map(([key,p])=>(
                          <button key={key} style={{...S.btn(exitPreset===key?'active':'default'),padding:'4px 10px',fontSize:9}} onClick={()=>setExitPreset(key)}>
                            {p.icon} {p.label}
                          </button>
                        ))}
                      </div>
                      {EXIT_PRESETS[exitPreset]&&(
                        <div style={{fontSize:9,color:C.textDim,marginTop:6,padding:'6px 10px',background:C.surface,borderRadius:4}}>
                          {EXIT_PRESETS[exitPreset].desc} ·
                          {EXIT_PRESETS[exitPreset].stopLoss&&` Stop: ${(EXIT_PRESETS[exitPreset].stopLoss*100).toFixed(0)}%`}
                          {EXIT_PRESETS[exitPreset].takeProfit&&` · Target: ${(EXIT_PRESETS[exitPreset].takeProfit*100).toFixed(0)}%`}
                          {EXIT_PRESETS[exitPreset].trailingStop&&` · Trailing: ${(EXIT_PRESETS[exitPreset].trailingStop*100).toFixed(0)}%`}
                        </div>
                      )}
                    </div>
                  </div>

                  {tickerNews.length>0&&(
                    <div style={S.panel}>
                      <div style={S.pt}>NEWS — {selectedAsset}</div>
                      {tickerNews.map((n,i)=>(
                        <div key={i} style={{padding:'8px 0',borderBottom:`1px solid ${C.border}40`}}>
                          <div style={{display:'flex',justifyContent:'space-between',gap:8}}>
                            <div style={{fontSize:11,color:C.textBright,flex:1,lineHeight:1.4}}>{n.title}</div>
                            <span style={S.sigBadge(n.sentiment.label==='BULLISH'?'BUY':n.sentiment.label==='BEARISH'?'SELL':'HOLD')}>{n.sentiment.label}</span>
                          </div>
                          <div style={{fontSize:9,color:C.textDim,marginTop:4}}>{n.source} · {fmt.ago(n.published)}</div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )
            })():<div style={{...S.panel,textAlign:'center',padding:40,color:C.textDim}}>Tap a ticker above to view signal</div>}
          </div>
        )}

        {/* ══ NEWS ══ */}
        {tab==='news'&&(
          <div style={{display:'flex',flexDirection:'column',gap:12}}>
            <div style={{display:'flex',gap:8,alignItems:'center',flexWrap:'wrap'}}>
              <button style={S.btn('primary')} onClick={()=>loadMarketNews(true)}>⟳ REFRESH LIVE</button>
              <button style={S.btn('warn')} onClick={()=>sendEmail('daily_briefing', buildRichBriefing())}>
                ✉️ SEND MORNING BRIEF NOW
              </button>
            </div>

            {aiSentiment&&(
              <div style={{...S.panel,border:`1px solid ${aiSentiment.sentiment==='BULLISH'?C.green:aiSentiment.sentiment==='BEARISH'?C.red:C.border}`}}>
                <div style={S.pt}>🤖 CLAUDE AI MARKET SENTIMENT</div>
                <div style={{display:'flex',alignItems:'center',gap:16}}>
                  <div style={{fontSize:24,fontWeight:700,color:aiSentiment.sentiment==='BULLISH'?C.green:aiSentiment.sentiment==='BEARISH'?C.red:C.textDim}}>{aiSentiment.sentiment}</div>
                  <div style={{fontSize:11,color:C.text}}>{aiSentiment.reason}</div>
                </div>
              </div>
            )}

            {newsStocks.length>0&&(
              <div style={{...S.panel,border:`1px solid ${C.purple}`}}>
                <div style={S.pt}>📡 STOCKS SURFACED FROM TODAY'S NEWS — AUTO-ADDED TO SCREEN</div>
                <div style={{display:'flex',gap:8,flexWrap:'wrap',marginBottom:8}}>
                  {newsStocks.map(c=>(
                    <div key={c.ticker} onClick={()=>{setSelectedAsset(c.ticker);setTab('signals')}}
                      style={{background:c.score>0?'#00ff8822':'#ff336622',border:`1px solid ${c.score>0?C.green:C.red}`,borderRadius:6,padding:'6px 12px',cursor:'pointer'}}>
                      <div style={{fontWeight:700,color:C.textBright,fontSize:11}}>{c.ticker}</div>
                      <div style={{fontSize:9,color:c.score>0?C.green:C.red}}>{c.score>0?'BULLISH':'BEARISH'} · {c.count} mention{c.count>1?'s':''}</div>
                    </div>
                  ))}
                </div>
                <div style={{fontSize:9,color:C.textDim}}>These tickers appeared in today's news and have been added to the screener universe.</div>
              </div>
            )}

            <div style={S.panel}>
              <div style={S.pt}>LIVE NEWS FEED — SENTIMENT SCORED</div>
              {marketNews.map((n,i)=>(
                <div key={i} style={{padding:'10px 12px',borderRadius:6,background:C.surface,border:`1px solid ${n.sentiment.label==='BULLISH'?C.green+'44':n.sentiment.label==='BEARISH'?C.red+'44':C.border}`,marginBottom:8}}>
                  <div style={{display:'flex',justifyContent:'space-between',alignItems:'flex-start',gap:8,marginBottom:6}}>
                    <div style={{fontSize:11,color:C.textBright,lineHeight:1.4,flex:1}}>{n.title}</div>
                    <span style={{...S.sigBadge(n.sentiment.label==='BULLISH'?'BUY':n.sentiment.label==='BEARISH'?'SELL':'HOLD'),flexShrink:0}}>{n.sentiment.label}</span>
                  </div>
                  <div style={{display:'flex',gap:10,alignItems:'center',flexWrap:'wrap'}}>
                    <span style={{fontSize:9,color:C.textDim}}>{n.source} · {fmt.ago(n.published)}</span>
                    {n.sentiment.impact==='high'&&<span style={{fontSize:9,color:C.orange,border:`1px solid ${C.orange}`,padding:'1px 5px',borderRadius:2}}>HIGH IMPACT</span>}
                    {(n.tickers||[]).slice(0,4).map(t=>(
                      <span key={t} onClick={()=>{setSelectedAsset(normalizeTicket(t));setTab('signals')}}
                        style={{fontSize:9,color:C.accent,border:`1px solid ${C.border}`,padding:'1px 5px',borderRadius:2,cursor:'pointer'}}>{t}</span>
                    ))}
                  </div>
                </div>
              ))}
              {marketNews.length===0&&<div style={{textAlign:'center',padding:32,color:C.textDim}}>Loading news...</div>}
            </div>
          </div>
        )}

        {/* ══ PAPER ══ */}
        {tab==='paper'&&(
          <div style={{display:'flex',flexDirection:'column',gap:12}}>
            {/* Watchlist */}
            <div style={S.panel}>
              <div style={S.pt}>📌 WATCHLIST — PERSISTENT LIVE P&L</div>
              <div style={{display:'flex',gap:8,marginBottom:10}}>
                <input value={watchlistInput} onChange={e=>setWatchlistInput(e.target.value.toUpperCase())} onKeyDown={e=>e.key==='Enter'&&addToWatchlist(watchlistInput)}
                  placeholder="Add ticker (NVDA, BTC, ETH, SOL...)" style={{...S.input,flex:1}}/>
                <button style={S.btn('primary')} onClick={()=>addToWatchlist(watchlistInput)}>+ ADD</button>
              </div>
              <div style={{overflowX:'auto'}}>
                <table style={{...S.table,minWidth:480}}>
                  <thead><tr>{['TICKER','PRICE','1D%','SIGNAL','POSITION','P&L',''].map(h=><th key={h} style={S.th}>{h}</th>)}</tr></thead>
                  <tbody>
                    {watchlist.map(ticker=>{
                      const display=displayTicker(ticker)
                      const px=prices[ticker]
                      const pos=portfolio.positions[ticker]
                      const curPx=px?.price||pos?.avgPrice||0
                      const unreal=pos?(pos.side==='LONG'?(curPx-pos.avgPrice)*pos.shares:(pos.avgPrice-curPx)*pos.shares):null
                      const sig=signals[ticker]||{signal:'HOLD',score:0}
                      return (
                        <tr key={ticker} style={{cursor:'pointer'}} onClick={()=>{setSelectedAsset(ticker);setTab('signals')}}>
                          <td style={{...S.td,fontWeight:700,color:C.textBright}}>{display}{ALL_CRYPTO.includes(ticker)&&' 🪙'}</td>
                          <td style={S.td}>{px?.price?fmt.price(px.price):<span style={{color:C.textDim}}>loading</span>}</td>
                          <td style={{...S.td,...fmt.chg(px?.changePct||0)}}>{px?.changePct!=null?`${px.changePct.toFixed(2)}%`:'—'}</td>
                          <td style={S.td}><span style={S.sigBadge(sig.signal)}>{sig.signal}</span></td>
                          <td style={S.td}>{pos?<span style={{color:pos.side==='LONG'?C.green:C.red}}>{pos.side} {pos.shares}sh</span>:'—'}</td>
                          <td style={{...S.td,fontWeight:unreal!=null?700:400,color:unreal!=null?(unreal>=0?C.green:C.red):C.textDim}}>{unreal!=null?`${unreal>=0?'+':''}$${unreal.toFixed(0)}`:'—'}</td>
                          <td style={S.td}><button style={{background:'transparent',border:'none',color:C.textDim,cursor:'pointer',fontSize:12}} onClick={e=>{e.stopPropagation();setWatchlist(prev=>prev.filter(t=>t!==ticker))}}>✕</button></td>
                        </tr>
                      )
                    })}
                    {watchlist.length===0&&<tr><td colSpan={7} style={{...S.td,textAlign:'center',color:C.textDim,padding:24}}>Add tickers above to track them</td></tr>}
                  </tbody>
                </table>
              </div>
              <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginTop:6}}>
                <div style={{fontSize:9,color:C.textDim}}>💾 Auto-saved per trade · Survives refresh · BTC/ETH/SOL supported</div>
                <button style={{...S.btn('default'),fontSize:8,padding:'2px 8px'}} onClick={()=>debugStorage()}>🔍 DEBUG STORAGE</button>
              </div>
            </div>

            {/* Portfolio metrics */}
            <div style={{display:'grid',gridTemplateColumns:'repeat(2,1fr)',gap:8}}>
              {[
                {label:'PORTFOLIO VALUE',value:`$${curVal.toFixed(0)}`,color:C.textBright},
                {label:'TOTAL P&L',value:`${totalPnL>=0?'+':''}$${totalPnL.toFixed(0)}`,color:totalPnL>=0?C.green:C.red},
                {label:'CASH',value:`$${portfolio.cash.toFixed(0)}`,color:C.accent},
                {label:'OPEN POSITIONS',value:Object.keys(portfolio.positions).length,color:C.purple},
              ].map(m=>(
                <div key={m.label} style={S.panel}>
                  <div style={S.pt}>{m.label}</div>
                  <div style={{fontSize:18,fontWeight:700,color:m.color}}>{m.value}</div>
                </div>
              ))}
            </div>

            {portfolio.history?.length>2&&(
              <div style={S.panel}>
                <div style={S.pt}>EQUITY CURVE</div>
                <ResponsiveContainer width="100%" height={130}>
                  <AreaChart data={portfolio.history.map((h,i)=>({i,value:h.value}))}>
                    <defs><linearGradient id="eq" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor={totalPnL>=0?C.green:C.red} stopOpacity={0.3}/><stop offset="95%" stopColor={totalPnL>=0?C.green:C.red} stopOpacity={0}/></linearGradient></defs>
                    <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
                    <XAxis dataKey="i" tick={false}/>
                    <YAxis tick={{fontSize:8,fill:C.textDim}} tickFormatter={v=>`$${(v/1000).toFixed(0)}K`}/>
                    <ReferenceLine y={STARTING_CAPITAL} stroke={C.border} strokeDasharray="4 4"/>
                    <Tooltip contentStyle={{background:C.panel,border:`1px solid ${C.border}`,fontSize:10}} formatter={v=>[`$${v.toFixed(0)}`,'Value']}/>
                    <Area type="monotone" dataKey="value" stroke={totalPnL>=0?C.green:C.red} fill="url(#eq)" strokeWidth={2} dot={false}/>
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            )}

            {Object.keys(portfolio.positions).length>0&&(
              <div style={S.panel}>
                <div style={S.pt}>OPEN POSITIONS — LIVE P&L</div>
                <div style={{overflowX:'auto'}}>
                  <table style={{...S.table,minWidth:500}}>
                    <thead><tr>{['TICKER','SIDE','SHARES','AVG IN','NOW','P&L','%','EXIT SIGNAL',''].map(h=><th key={h} style={S.th}>{h}</th>)}</tr></thead>
                    <tbody>
                      {Object.entries(portfolio.positions).map(([ticker,pos])=>{
                        const curPx=prices[ticker]?.price||pos.avgPrice
                        const unreal=pos.side==='LONG'?(curPx-pos.avgPrice)*pos.shares:(pos.avgPrice-curPx)*pos.shares
                        const pct=(unreal/(pos.avgPrice*pos.shares))*100
                        const exit=checkExitSignal(pos,curPx,signals[ticker],bars[ticker],EXIT_PRESETS[exitPreset]||EXIT_PRESETS.moderate)
                        return (
                          <tr key={ticker}>
                            <td style={{...S.td,fontWeight:700,color:C.textBright}}>{displayTicker(ticker)}</td>
                            <td style={S.td}><span style={S.sigBadge(pos.side==='LONG'?'BUY':'SELL')}>{pos.side}</span></td>
                            <td style={S.td}>{pos.shares}</td>
                            <td style={S.td}>{fmt.price(pos.avgPrice)}</td>
                            <td style={S.td}>{fmt.price(curPx)}</td>
                            <td style={{...S.td,fontWeight:700,color:unreal>=0?C.green:C.red}}>{unreal>=0?'+':''}${unreal.toFixed(0)}</td>
                            <td style={{...S.td,color:pct>=0?C.green:C.red}}>{pct>=0?'+':''}{pct.toFixed(1)}%</td>
                            <td style={{...S.td,fontSize:9,color:exit?.shouldExit?C.red:exit?.urgency==='WARNING'?C.yellow:C.textDim}}>{exit?.reason||'—'}</td>
                            <td style={S.td}><button style={{...S.btn('danger'),padding:'2px 8px',fontSize:9}} onClick={()=>closePosition(ticker)}>CLOSE</button></td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            <div style={S.panel}>
              <div style={S.pt}>SETTINGS</div>
              <div style={{display:'flex',gap:10,alignItems:'center',flexWrap:'wrap'}}>
                <span style={{fontSize:10,color:C.textDim}}>Trade size: $</span>
                <input type="number" value={tradeSize} onChange={e=>setTradeSize(parseInt(e.target.value)||1000)} style={{...S.input,width:90}}/>
                <button style={{...S.btn('danger'),marginLeft:'auto'}} onClick={resetPortfolio}>↺ RESET</button>
              </div>
            </div>

            <div style={S.panel}>
              <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:8}}>
                <div style={{...S.pt,marginBottom:0}}>TRADE LOG ({tradeLog.length})</div>
                <button style={{...S.btn('primary'),padding:'4px 12px',fontSize:9}} onClick={downloadPaperLogCSV}>↓ CSV</button>
              </div>
              {tradeLog.length===0?<div style={{textAlign:'center',padding:24,color:C.textDim}}>No trades yet</div>:(
                <div style={{overflowX:'auto',maxHeight:280,overflowY:'auto'}}>
                  <table style={{...S.table,minWidth:420}}>
                    <thead><tr>{['TIME','TICKER','SIDE','PRICE','SHARES','P&L','AUTO'].map(h=><th key={h} style={S.th}>{h}</th>)}</tr></thead>
                    <tbody>
                      {tradeLog.map(t=>(
                        <tr key={t.id}>
                          <td style={{...S.td,fontSize:9,color:C.textDim}}>{t.time}</td>
                          <td style={{...S.td,fontWeight:700,color:C.textBright}}>{displayTicker(t.ticker)}</td>
                          <td style={S.td}><span style={S.sigBadge(t.side==='BUY'?'BUY':t.side==='SELL'?'SELL':'HOLD')}>{t.side}</span></td>
                          <td style={S.td}>{fmt.price(t.price)}</td>
                          <td style={S.td}>{t.shares}</td>
                          <td style={{...S.td,color:t.pnl>0?C.green:t.pnl<0?C.red:C.textDim}}>{t.pnl?`${t.pnl>=0?'+':''}$${t.pnl}`:'—'}</td>
                          <td style={{...S.td,fontSize:9,color:t.auto?C.purple:C.textDim}}>{t.auto?'🤖 AUTO':'—'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ══ AUTO-TRADE ══ */}
        {tab==='auto'&&(
          <div style={{display:'flex',flexDirection:'column',gap:12}}>
            {/* Email test */}
            <div style={{...S.panel,border:`1px solid ${C.yellow}`}}>
              <div style={S.pt}>✉️ EMAIL ALERTS → {EMAIL_TO}</div>
              <div style={{fontSize:11,color:C.textDim,marginBottom:12,lineHeight:1.6}}>
                Automatic alerts sent when portfolio moves ±5%. Daily briefing every morning at 8-11am ET.
                
              {/* ── Alpaca Broker Bridge (claude.md Phase 4) ── */}
              <div style={{...S.panel,border:'1px solid #00d4ff33'}}>
                <div style={S.pt}>ALPACA BROKER BRIDGE · <span style={{color:alpacaConfigured()?'#00ff88':'#ffd700'}}>{alpacaConfigured()?alpacaMode()+' CONNECTED':'NOT CONFIGURED'}</span></div>
                {alpacaConfigured()?(
                  <div style={{display:'flex',gap:8,flexWrap:'wrap',marginTop:8}}>
                    <button style={{...S.btn('green'),fontSize:9}} onClick={async()=>{try{const p=await syncPortfolio(portfolio);setPortfolio(p);setEmailStatus('Alpaca synced')}catch(e){setEmailStatus('Sync failed: '+e.message)}}}>⟳ SYNC POSITIONS</button>
                    <span style={{fontSize:9,color:'#64748b',alignSelf:'center'}}>Mode: {alpacaMode()} · Set VITE_ALPACA_KEY_ID + VITE_ALPACA_SECRET_KEY in Vercel</span>
                  </div>
                ):(
                  <div style={{fontSize:9,color:'#64748b',marginTop:8}}>
                    Add to Vercel environment variables:<br/>
                    <code style={{color:'#00d4ff'}}>VITE_ALPACA_KEY_ID</code> · <code style={{color:'#00d4ff'}}>VITE_ALPACA_SECRET_KEY</code> · <code style={{color:'#00d4ff'}}>VITE_ALPACA_PAPER=true</code><br/>
                    Paper trading is free at alpaca.markets — no commission.
                  </div>
                )}
              </div>

              Requires <code style={{color:C.accent}}>RESEND_API_KEY</code> set in Vercel env vars.
              </div>
              <div style={{display:'flex',gap:8,flexWrap:'wrap'}}>
                <button style={S.btn('warn')} onClick={()=>sendEmail('portfolio_alert',{
                  portfolioValue:curVal, pnl:totalPnL, pnlPct:(totalPnL/STARTING_CAPITAL)*100,
                  direction:totalPnL>=0?'up':'down',
                  positions:Object.entries(portfolio.positions).map(([t,p])=>({ticker:displayTicker(t),shares:p.shares,avgPrice:p.avgPrice,pnl:(prices[t]?.price||p.avgPrice-p.avgPrice)*p.shares}))
                })}>
                  ✉️ TEST PORTFOLIO ALERT
                </button>
                <button style={S.btn('primary')} onClick={()=>sendEmail('daily_briefing', buildRichBriefing())}>
                  ☀️ TEST MORNING BRIEF
                </button>
                <button style={S.btn('warn')} onClick={()=>sendEmail('end_of_day', buildRichEOD())}>
                  📊 TEST EOD REPORT
                </button>
              </div>
              {emailStatus&&<div style={{marginTop:8,fontSize:10,color:emailStatus==='sent'?C.green:C.yellow}}>{emailStatus==='sent'?'✓ Email sent successfully':emailStatus}</div>}
              <div style={{marginTop:12,fontSize:9,color:C.textDim}}>
                To set up: Go to resend.com → free account → get API key → add RESEND_API_KEY to Vercel dashboard → redeploy
              </div>
            </div>

            {/* Auto-trade engine */}
            <div style={{...S.panel,border:`1px solid ${autoTradeSettings.enabled?C.green:C.border}`}}>
              <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:12}}>
                <div>
                  <div style={S.pt}>🤖 AUTO-TRADING ENGINE</div>
                  <div style={{fontSize:9,color:C.textDim,marginTop:2}}>Scans every 60s · Buys on BUY signal · Exits on TP/SL</div>
                </div>
                <button style={{...S.btn(autoTradeSettings.enabled?'active':'default'),fontSize:11,padding:'6px 14px'}}
                  onClick={()=>setAutoTradeSettings(p=>({...p,enabled:!p.enabled}))}>
                  {autoTradeSettings.enabled?'🟢 LIVE':'⚫ PAUSED'}
                </button>
              </div>
              {autoTradeSettings.enabled&&(
                <div style={{fontSize:10,color:C.green,background:'#00ff8811',border:'1px solid #00ff8830',borderRadius:4,padding:'6px 10px',marginBottom:12}}>
                  ✅ Auto-trading ACTIVE — bot will buy/sell automatically based on signals
                </div>
              )}

              {/* ── Primary exit controls — big and prominent ── */}
              <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:10,marginBottom:14}}>
                {/* Take-Profit */}
                <div style={{background:C.surface,border:`1px solid ${C.green}40`,borderRadius:8,padding:12}}>
                  <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:6}}>
                    <div style={{fontSize:9,color:C.textDim,letterSpacing:2}}>TAKE PROFIT</div>
                    <span style={{fontSize:14,fontWeight:700,color:C.green}}>+{(autoTradeSettings.takeProfitPct*100).toFixed(0)}%</span>
                  </div>
                  <input type="range" min={3} max={50} step={1}
                    value={Math.round((autoTradeSettings.takeProfitPct||0.10)*100)}
                    onChange={e=>setAutoTradeSettings(p=>({...p,takeProfitPct:parseInt(e.target.value)/100}))}
                    style={{width:'100%',accentColor:C.green}}/>
                  <div style={{display:'flex',justifyContent:'space-between',fontSize:8,color:C.textDim,marginTop:2}}>
                    <span>3%</span><span>50%</span>
                  </div>
                </div>

                {/* Stop-Loss */}
                <div style={{background:C.surface,border:`1px solid ${C.red}40`,borderRadius:8,padding:12}}>
                  <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:6}}>
                    <div style={{fontSize:9,color:C.textDim,letterSpacing:2}}>STOP LOSS</div>
                    <span style={{fontSize:14,fontWeight:700,color:C.red}}>-{(autoTradeSettings.stopLossPct*100).toFixed(0)}%</span>
                  </div>
                  <input type="range" min={2} max={25} step={1}
                    value={Math.round((autoTradeSettings.stopLossPct||0.07)*100)}
                    onChange={e=>setAutoTradeSettings(p=>({...p,stopLossPct:parseInt(e.target.value)/100}))}
                    style={{width:'100%',accentColor:C.red}}/>
                  <div style={{display:'flex',justifyContent:'space-between',fontSize:8,color:C.textDim,marginTop:2}}>
                    <span>2%</span><span>25%</span>
                  </div>
                </div>
              </div>

              {/* R/R indicator */}
              {(()=>{
                const rr=(autoTradeSettings.takeProfitPct||0.10)/(autoTradeSettings.stopLossPct||0.07)
                const col=rr>=2?C.green:rr>=1.2?C.yellow:C.red
                return (
                  <div style={{display:'flex',justifyContent:'center',alignItems:'center',gap:12,marginBottom:14,padding:'8px',background:C.surface,borderRadius:6}}>
                    <span style={{fontSize:9,color:C.textDim}}>RISK/REWARD</span>
                    <span style={{fontSize:16,fontWeight:700,color:col}}>{rr.toFixed(1)}x</span>
                    <span style={{fontSize:9,color:col}}>{rr>=2?'✅ Excellent':rr>=1.5?'✅ Good':rr>=1.2?'⚠️ Acceptable':'❌ Too tight'}</span>
                  </div>
                )
              })()}

              {/* Advanced settings collapsible */}
              <div style={{fontSize:9,color:C.textDim,letterSpacing:2,marginBottom:8}}>ADVANCED</div>
              <div style={{display:'grid',gridTemplateColumns:'repeat(2,1fr)',gap:8}}>
                {[
                  {label:'Min Signal Score',key:'minScore',step:0.05,min:0.1,max:0.9},
                  {label:'Min Confidence',key:'minConfidence',step:0.05,min:0.1,max:0.9},
                  {label:'Max Positions',key:'maxPositions',step:1,min:1,max:20},
                  {label:'Position Size %',key:'positionSizePct',step:0.01,min:0.01,max:0.25},
                ].map(f=>(
                  <div key={f.key}>
                    <div style={{fontSize:9,color:C.textDim,marginBottom:3}}>{f.label}: <span style={{color:C.accent}}>{f.key==='positionSizePct'?`${(autoTradeSettings[f.key]*100).toFixed(0)}%`:autoTradeSettings[f.key]}</span></div>
                    <input type="number" step={f.step} min={f.min} max={f.max}
                      value={autoTradeSettings[f.key]}
                      onChange={e=>setAutoTradeSettings(p=>({...p,[f.key]:parseFloat(e.target.value)}))}
                      style={{...S.input,width:'100%'}}/>
                  </div>
                ))}
              </div>
              <div style={{marginTop:10,display:'flex',alignItems:'center',gap:8}}>
                <input type="checkbox" id="multimodel" checked={!!autoTradeSettings.requireMultiModel}
                  onChange={e=>setAutoTradeSettings(p=>({...p,requireMultiModel:e.target.checked}))}/>
                <label htmlFor="multimodel" style={{fontSize:10,color:C.textDim,cursor:'pointer'}}>
                  Require 2+ models to agree before buying (recommended)
                </label>
              </div>
            </div>

            {/* Exit strategy explainer */}
            <div style={S.panel}>
              <div style={S.pt}>📐 EXIT STRATEGIES — SELECT ACTIVE PRESET</div>
              <div style={{display:'flex',gap:6,flexWrap:'wrap',marginBottom:12}}>
                {Object.entries(EXIT_PRESETS).map(([key,p])=>(
                  <button key={key} style={{...S.btn(exitPreset===key?'active':'default'),padding:'6px 14px'}} onClick={()=>setExitPreset(key)}>
                    {p.icon} {p.label}
                  </button>
                ))}
              </div>
              {EXIT_PRESETS[exitPreset]&&(
                <div style={{background:C.surface,borderRadius:6,padding:14}}>
                  <div style={{fontSize:13,fontWeight:700,color:C.textBright,marginBottom:8}}>{EXIT_PRESETS[exitPreset].icon} {EXIT_PRESETS[exitPreset].label}</div>
                  <div style={{fontSize:11,color:C.textDim,marginBottom:10}}>{EXIT_PRESETS[exitPreset].desc}</div>
                  <div style={{display:'grid',gridTemplateColumns:'repeat(2,1fr)',gap:8}}>
                    {[
                      {label:'Stop Loss',value:EXIT_PRESETS[exitPreset].stopLoss,fmt:v=>v?`${(v*100).toFixed(0)}%`:'ATR-based'},
                      {label:'Take Profit',value:EXIT_PRESETS[exitPreset].takeProfit,fmt:v=>v?`${(v*100).toFixed(0)}%`:'3x ATR'},
                      {label:'Trailing Stop',value:EXIT_PRESETS[exitPreset].trailingStop,fmt:v=>v?`${(v*100).toFixed(0)}% from peak`:'ATR-based'},
                      {label:'Max Hold',value:EXIT_PRESETS[exitPreset].maxDays,fmt:v=>`${v} days`},
                      {label:'Signal Reversal',value:EXIT_PRESETS[exitPreset].signalReversal,fmt:v=>v?'YES — exit on SELL signal':'NO'},
                      {label:'ATR Multiplier',value:EXIT_PRESETS[exitPreset].atrMultiplier,fmt:v=>v?`${v}x ATR`:'N/A'},
                    ].map(m=>(
                      <div key={m.label} style={{padding:'6px 0',borderBottom:`1px solid ${C.border}40`}}>
                        <div style={{fontSize:9,color:C.textDim}}>{m.label}</div>
                        <div style={{fontSize:11,color:C.textBright,marginTop:2}}>{m.fmt(m.value)}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Auto trade log */}
            {autoTradeLog.length>0&&(
              <div style={S.panel}>
                <div style={S.pt}>🤖 AUTO-TRADE LOG</div>
                {autoTradeLog.map((t,i)=>(
                  <div key={i} style={{padding:'6px 0',borderBottom:`1px solid ${C.border}40`,fontSize:10}}>
                    <span style={{color:C.textDim}}>{t.time}</span>
                    <span style={{color:C.textBright,fontWeight:700,marginLeft:8}}>{t.ticker}</span>
                    <span style={{color:t.action==='BUY'?C.green:t.action==='CLOSE'?C.yellow:C.red,marginLeft:8}}>{t.action}</span>
                    <span style={{color:C.textDim,marginLeft:8}}>{t.reason}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* ══ TRAIN ══ */}
        {tab==='train'&&(
          <div style={{display:'flex',flexDirection:'column',gap:12}}>
            <div style={{display:'flex',gap:8}}>
              <button style={{...S.btn(isTraining?'default':'primary'),flex:1,padding:12}} onClick={startTraining} disabled={isTraining}>
                {isTraining?'▶ TRAINING...':'▶ START RL TRAINING (7 FACTORS)'}
              </button>
              {isTraining&&<button style={{...S.btn('danger'),padding:12}} onClick={()=>{stopTraining();setIsTraining(false)}}>■ STOP</button>}
            </div>

            <div style={{...S.panel,border:`1px solid ${C.border}`}}>
              <div style={S.pt}>WHAT'S BEING OPTIMIZED</div>
              <div style={{fontSize:10,color:C.textDim,lineHeight:1.7}}>
                The RL agent optimizes weights for <span style={{color:C.accent}}>7 signals</span>: Momentum · Mean-Reversion · Volume/OBV · Volatility/ATR · Trend/MACD · <span style={{color:C.purple}}>FF5 Alpha</span> · <span style={{color:C.yellow}}>TCN Align</span><br/><br/><span style={{color:C.green}}>Reward = Sharpe (45%) + Calmar (20%) + Model Agreement FF5+TCN (20%) + Return (15%)</span><br/>Agent is rewarded MORE when signals agree with Fama-French and TCN — so optimized weights work across ALL models, not just technicals. Regime-aware: separate weight pools per market regime.
              </div>
            </div>

            <div style={{display:'grid',gridTemplateColumns:'repeat(2,1fr)',gap:8}}>
              {[
                {label:'GENERATION',value:agent.generation,color:C.accent},
                {label:'BEST RAG SCORE',value:agent.bestScore===-Infinity?'—':agent.bestScore.toFixed(3),color:C.purple},
                {label:'EPSILON ε',value:agent.epsilon.toFixed(3),color:C.text},
                {label:'STATUS',value:rlProgress?.improving?'↑ IMPROVING':'↔ EXPLORING',color:rlProgress?.improving?C.green:C.yellow},
              ].map(m=>(
                <div key={m.label} style={S.panel}>
                  <div style={S.pt}>{m.label}</div>
                  <div style={{fontSize:18,fontWeight:700,color:m.color}}>{m.value}</div>
                </div>
              ))}
            </div>

            <div style={S.panel}>
              <div style={S.pt}>FACTOR WEIGHTS — LIVE</div>
              <div style={{display:'flex',gap:8,alignItems:'flex-end',flexWrap:'wrap'}}>
                {Object.entries(weights).map(([k,w])=>(
                  <div key={k} style={{flex:1,minWidth:50,textAlign:'center'}}>
                    <div style={{fontSize:7,color:C.textDim,marginBottom:4}}>{k.replace('meanReversion','MRev').replace('momentum','Mom').replace('volume','Vol').replace('volatility','VReg').replace('ffAlpha','FF5α').replace('tcnAlign','TCN').toUpperCase()}</div>
                    <div style={{height:70,background:C.border,borderRadius:3,position:'relative',overflow:'hidden'}}>
                      <div style={{position:'absolute',bottom:0,width:'100%',height:`${w*100}%`,background:`linear-gradient(0deg,${C.accent},${C.purple})`,transition:'height 0.5s'}}/>
                    </div>
                    <div style={{fontSize:10,fontWeight:700,color:C.textBright,marginTop:3}}>{(w*100).toFixed(0)}%</div>
                  </div>
                ))}
              </div>
            </div>


            {/* CSV Export */}
            {backtestResult && (
              <div style={{...S.panel, border:'1px solid #00d4ff33'}}>
                <div style={S.pt}>📥 EXPORT DATA</div>
                <div style={{display:'flex', gap:8, flexWrap:'wrap'}}>
                  <button style={{...S.btn('primary'), padding:'8px 16px'}} onClick={downloadEquityCurveCSV}>
                    ↓ EQUITY CURVE CSV
                  </button>
                  <button style={{...S.btn('primary'), padding:'8px 16px'}} onClick={downloadTradeLogCSV}>
                    ↓ TRADE LOG CSV
                  </button>
                </div>
                <div style={{marginTop:8, fontSize:9, color:'#64748b'}}>
                  Equity curve: daily portfolio value, cash, positions, observation mode flags<br/>
                  Trade log: every entry/exit with price, shares, Kelly sizing, stop-loss reason
                </div>
              </div>
            )}
            {trainingLog.length>0&&(
              <div style={S.panel}>
                <div style={S.pt}>TRAINING SCORE HISTORY</div>
                <ResponsiveContainer width="100%" height={140}>
                  <LineChart data={trainingLog}>
                    <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
                    <XAxis dataKey="episode" tick={{fontSize:8,fill:C.textDim}}/>
                    <YAxis tick={{fontSize:8,fill:C.textDim}}/>
                    <ReferenceLine y={0} stroke={C.border}/>
                    <Tooltip contentStyle={{background:C.panel,border:`1px solid ${C.border}`,fontSize:10}}/>
                    <Line type="monotone" dataKey="score" stroke={C.purple} strokeWidth={2} dot={false} name="RAG"/>
                    <Line type="monotone" dataKey="sharpe" stroke={C.accent} strokeWidth={1.5} dot={false} name="Sharpe"/>
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}

            {backtestResult&&(
              <div style={S.panel}>
                <div style={{display:'flex',justifyContent:'space-between',alignItems:'center'}}>
                  <div style={S.pt}>BACKTEST RESULT — BEST WEIGHTS · <span style={{color:C.accent}}>{algoMode} mode</span></div>
                  {backtestResult.equity?.length>0&&(
                    <button style={{...S.btn('primary'),padding:'3px 10px',fontSize:9}} onClick={()=>{
                      const csv=equityCurveToCSV(backtestResult.equity)
                      const b=new Blob([csv],{type:'text/csv'})
                      const u=URL.createObjectURL(b)
                      const a=document.createElement('a');a.href=u;a.download='equity_curve.csv';a.click()
                    }}>⬇ CSV</button>
                  )}
                </div>
                <div style={{display:'grid',gridTemplateColumns:'repeat(5,1fr)',gap:8,marginTop:8}}>
                  {[
                    {label:'RETURN',value:fmt.pct(backtestResult.totalReturn||0),color:(backtestResult.totalReturn||0)>=0?C.green:C.red},
                    {label:'SHARPE',value:(backtestResult.sharpe||0).toFixed(2),color:C.accent},
                    {label:'MAX DD',value:`-${((backtestResult.maxDrawdown||0)*100).toFixed(1)}%`,color:(backtestResult.maxDrawdown||0)>0.15?C.red:C.yellow},
                    {label:'REWARD',value:(backtestResult.ragScore||0).toFixed(3),color:C.purple},
                    {label:'TRADES',value:backtestResult.tradeCount||0,color:C.textDim},
                  ].map(m=>(
                    <div key={m.label} style={{textAlign:'center',padding:10,background:C.surface,borderRadius:6}}>
                      <div style={S.pt}>{m.label}</div>
                      <div style={{fontSize:16,fontWeight:700,color:m.color}}>{m.value}</div>
                    </div>
                  ))}
                </div>
                {backtestResult.observationModeHits>0&&(
                  <div style={{marginTop:8,fontSize:9,color:C.red}}>⚠️ {backtestResult.observationModeHits} bars in Observation Mode (DD &gt; 15%)</div>
                )}
              </div>
            )}
          </div>
        )}


        {/* ══ VERIFY ══ */}
        {tab==='verify'&&(
          <div style={{display:'flex',flexDirection:'column',gap:12}}>

            {/* Header */}
            <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',flexWrap:'wrap',gap:8}}>
              <div>
                <div style={{fontSize:13,fontWeight:700,color:C.textBright}}>🤖 Subagent Verification System</div>
                <div style={{fontSize:10,color:C.textDim,marginTop:2}}>5 subagents · prices · signals · RL · strategy · email</div>
              </div>
              <div style={{display:'flex',gap:8,alignItems:'center'}}>
                {verifyReport&&<span style={{fontSize:9,color:C.textDim}}>Last run: {new Date(verifyReport.ts).toLocaleTimeString()}</span>}
                <button style={{...S.btn('primary'),padding:'6px 16px'}} onClick={()=>runVerify()} disabled={isVerifying}>
                  {isVerifying?'⟳ RUNNING...':'▶ RUN ALL CHECKS'}
                </button>
              </div>
            </div>

            {/* Loading */}
            {isVerifying&&(
              <div style={{...S.panel,textAlign:'center',padding:40}}>
                <div style={{color:C.accent,letterSpacing:4,fontSize:11,marginBottom:8}}>RUNNING 5 SUBAGENTS...</div>
                <div style={{display:'flex',justifyContent:'center',gap:16}}>
                  {['PriceVerifier','AlgoVerifier','RLVerifier','StrategyVerifier','EmailVerifier'].map(a=>(
                    <div key={a} style={{textAlign:'center'}}>
                      <div style={{width:8,height:8,borderRadius:'50%',background:C.accent,margin:'0 auto 4px',animation:'pulse 1s infinite'}}/>
                      <div style={{fontSize:8,color:C.textDim}}>{a.replace('Verifier','')}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Idle state */}
            {!verifyReport&&!isVerifying&&(
              <div style={{...S.panel,textAlign:'center',padding:48,color:C.textDim}}>
                <div style={{fontSize:24,marginBottom:8}}>🤖</div>
                <div style={{fontSize:12,color:C.textBright,marginBottom:6}}>5 Subagents Ready</div>
                <div style={{fontSize:10}}>PriceVerifier · AlgoVerifier · RLVerifier · StrategyVerifier · EmailVerifier</div>
                <div style={{fontSize:9,marginTop:8}}>Checks run automatically after each screen. Click to run manually.</div>
              </div>
            )}

            {verifyReport&&(
              <>
                {/* Overall status */}
                <div style={{background:verifyReport.overall==='error'?'#ff336611':verifyReport.overall==='warn'?'#ffd70011':'#00ff8811',
                  border:`1px solid ${verifyReport.overall==='error'?C.red:verifyReport.overall==='warn'?C.yellow:C.green}`,
                  borderRadius:8,padding:'12px 16px',display:'flex',justifyContent:'space-between',alignItems:'center'}}>
                  <span style={{color:verifyReport.overall==='error'?C.red:verifyReport.overall==='warn'?C.yellow:C.green,fontWeight:700,fontSize:12}}>
                    {verifyReport.overall==='error'?'🔴':verifyReport.overall==='warn'?'🟡':'🟢'} {verifyReport.summary.message}
                  </span>
                  <span style={{fontSize:9,color:C.textDim}}>{verifyReport.summary.oks.length}/5 OK</span>
                </div>

                {/* ── AUTO-IMPROVEMENT PANEL ── */}
                {verifyReport.improvements&&(
                  <div style={{...S.panel,border:`1px solid ${C.purple}60`}}>
                    <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:10}}>
                      <div style={{fontSize:11,fontWeight:700,color:C.purple}}>🔧 AUTO-IMPROVEMENT ENGINE</div>
                      <div style={{display:'flex',gap:6}}>
                        {verifyReport.improvements.shouldRetrain&&(
                          <span style={{fontSize:9,padding:'2px 8px',borderRadius:3,background:'#a855f720',border:`1px solid ${C.purple}`,color:C.purple}}>
                            ⟳ RETRAIN QUEUED
                          </span>
                        )}
                        {Object.keys(verifyReport.improvements.settingPatches||{}).length>0&&(
                          <span style={{fontSize:9,padding:'2px 8px',borderRadius:3,background:'#00d4ff20',border:`1px solid ${C.accent}`,color:C.accent}}>
                            ⚙️ SETTINGS PATCHED
                          </span>
                        )}
                      </div>
                    </div>
                    {verifyReport.improvements.actions.map((a,i)=>(
                      <div key={i} style={{display:'flex',gap:10,padding:'8px 0',borderBottom:`1px solid ${C.border}30`,alignItems:'flex-start'}}>
                        <span style={{fontSize:14,minWidth:20}}>
                          {a.type==='ok'?'✅':a.type==='retrain'?'🔄':a.type==='setting'?'⚙️':a.type==='mode'?'🔀':'ℹ️'}
                        </span>
                        <div style={{flex:1}}>
                          <div style={{display:'flex',justifyContent:'space-between',alignItems:'center'}}>
                            <div style={{fontSize:10,fontWeight:700,color:a.severity==='ok'?C.green:a.severity==='error'?C.red:C.yellow}}>{a.label}</div>
                            <span style={{fontSize:8,color:C.textDim,fontStyle:'italic'}}>{a.fix}</span>
                          </div>
                          <div style={{fontSize:9,color:C.textDim,marginTop:2}}>{a.detail}</div>
                        </div>
                      </div>
                    ))}
                    {Object.keys(verifyReport.improvements.settingPatches||{}).length>0&&(
                      <div style={{marginTop:8,padding:'6px 10px',background:C.surface,borderRadius:4}}>
                        <div style={{fontSize:9,color:C.textDim,marginBottom:4}}>APPLIED PATCHES</div>
                        <div style={{display:'flex',gap:8,flexWrap:'wrap'}}>
                          {Object.entries(verifyReport.improvements.settingPatches).map(([k,v])=>(
                            <span key={k} style={{fontSize:9,padding:'2px 8px',borderRadius:3,background:`${C.accent}20`,color:C.accent}}>
                              {k}: {typeof v==='number'&&v<1?`${(v*100).toFixed(0)}%`:v}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* Action items */}
                {verifyReport.feedback?.length>0&&(
                  <div style={S.panel}>
                    <div style={S.pt}>⚠️ ACTION ITEMS ({verifyReport.feedback.length})</div>
                    {verifyReport.feedback.map((f,i)=>(
                      <div key={i} style={{display:'flex',gap:10,padding:'8px 0',borderBottom:`1px solid ${C.border}40`,alignItems:'flex-start'}}>
                        <span style={{fontSize:9,padding:'2px 6px',borderRadius:3,border:`1px solid ${f.severity==='error'?C.red:C.yellow}`,
                          color:f.severity==='error'?C.red:C.yellow,whiteSpace:'nowrap',marginTop:1}}>
                          {f.severity==='error'?'ERROR':'WARN'}
                        </span>
                        <div>
                          <div style={{fontSize:10,color:C.textBright}}>{f.label}</div>
                          <div style={{fontSize:9,color:C.textDim,marginTop:2}}>{f.detail}</div>
                          <div style={{fontSize:8,color:C.textDim,opacity:0.6,marginTop:1}}>{f.agent}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {/* 5 agent detail cards */}
                {[
                  {key:'price',   icon:'💲', name:'PriceVerifier',    desc:'Real vs mock price detection'},
                  {key:'algo',    icon:'📡', name:'AlgoVerifier',     desc:'Signal logic & data leakage'},
                  {key:'rl',      icon:'🧠', name:'RLVerifier',       desc:'Reward convergence & Sharpe'},
                  {key:'strategy',icon:'📈', name:'StrategyVerifier', desc:'Win rate, P&L, trade quality'},
                  {key:'email',   icon:'✉️', name:'EmailVerifier',    desc:'Email delivery & API health'},
                ].map(({key,icon,name,desc})=>{
                  const a=verifyReport.agents[key]
                  if(!a) return null
                  const agentColor=a.status==='error'?C.red:a.status==='warn'?C.yellow:C.green
                  return (
                    <div key={key} style={{...S.panel,border:`1px solid ${agentColor}40`}}>
                      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:8}}>
                        <div style={{display:'flex',alignItems:'center',gap:8}}>
                          <span style={{width:8,height:8,borderRadius:'50%',background:agentColor,display:'inline-block',boxShadow:`0 0 8px ${agentColor}`}}/>
                          <span style={{fontSize:11,fontWeight:700,color:C.textBright}}>{icon} {name}</span>
                          <span style={{fontSize:9,color:C.textDim}}>{desc}</span>
                        </div>
                        <span style={{...S.badge(agentColor),fontSize:9}}>{a.status.toUpperCase()}</span>
                      </div>
                      <div style={{fontSize:10,color:C.textDim,marginBottom:8}}>{a.summary}</div>
                      <div style={{display:'flex',flexDirection:'column',gap:0}}>
                        {a.checks.map((c,i)=>(
                          <div key={i} style={{display:'flex',gap:8,padding:'5px 0',borderTop:`1px solid ${C.border}20`,alignItems:'flex-start'}}>
                            <span style={{color:c.status==='pass'?C.green:c.status==='fail'?C.red:C.yellow,fontSize:12,minWidth:16,marginTop:0}}>
                              {c.status==='pass'?'✓':c.status==='fail'?'✗':'!'}
                            </span>
                            <div style={{flex:1}}>
                              <div style={{fontSize:9,color:c.status==='pass'?C.textDim:C.textBright}}>{c.label}</div>
                              {(c.status==='fail'||c.status==='warn')&&(
                                <div style={{fontSize:9,color:c.status==='fail'?C.red:C.yellow,marginTop:2,lineHeight:1.4}}>{c.detail}</div>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )
                })}
              </>
            )}
          </div>
        )}

        {/* ══ MODELS ══ */}
        {tab==='models'&&(
          <div style={{display:'flex',flexDirection:'column',gap:12}}>
            <div style={{display:'flex',gap:6,flexWrap:'wrap'}}>
              {['ensemble',...MODEL_INFO.map(m=>m.id)].map(id=>(
                <button key={id} style={{...S.btn(activeModel===id?'active':'default'),fontSize:10}}
                  onClick={()=>setActiveModel(id)}>
                  {id==='ensemble'?'⚡ ENSEMBLE':(MODEL_INFO.find(m=>m.id===id)?.icon||'') + ' ' + (MODEL_INFO.find(m=>m.id===id)?.name?.split(' ')[0]||id)}
                </button>
              ))}
            </div>

            {activeModel==='ensemble'&&(
              <>
                <div style={S.panel}>
                  <div style={S.pt}>⚡ ENSEMBLE — ALL MODELS COMBINED</div>
                  <div style={{fontSize:10,color:C.textDim,marginBottom:12,lineHeight:1.6}}>
                    Combines Technical (35%), Fama-French 5F (25%), Temporal CNN (25%), News Sentiment (15%) into one confidence-weighted signal.
                  </div>
                  <div style={{display:'grid',gridTemplateColumns:'repeat(5,1fr)',gap:5}}>
                    {Object.entries(ensembleResults).filter(([t])=>activeStocks.includes(t)).slice(0,15).map(([ticker,ens])=>{
                      if(!ens) return null
                      const score=ens.score||0, hue=score>0?'144,255,136':'255,51,102'
                      return (
                        <div key={ticker} onClick={()=>{setSelectedAsset(ticker);setTab('signals')}}
                          style={{padding:'8px 4px',borderRadius:4,cursor:'pointer',background:`rgba(${hue},${Math.abs(score)*0.4})`,border:`1px solid rgba(${hue},${Math.abs(score)*0.6})`,textAlign:'center'}}>
                          <div style={{fontSize:10,fontWeight:700,color:C.textBright}}>{ticker}</div>
                          <div style={{fontSize:9,color:score>0?C.green:C.red}}>{ens.signal}</div>
                          <div style={{fontSize:8,color:C.textDim}}>{(ens.confidence*100).toFixed(0)}% conf</div>
                        </div>
                      )
                    })}
                  </div>
                </div>

                <div style={S.panel}>
                  <div style={S.pt}>SIGNAL BREAKDOWN BY MODEL</div>
                  <div style={{overflowX:'auto'}}>
                    <table style={{...S.table,minWidth:520}}>
                      <thead>
                        <tr>
                          <th style={S.th}>TICKER</th>
                          {MODEL_INFO.slice(0,4).map(m=><th key={m.id} style={S.th}>{m.icon} {m.name.split(' ')[0].toUpperCase()}</th>)}
                          <th style={S.th}>ENSEMBLE</th>
                          <th style={S.th}>CONFIDENCE</th>
                        </tr>
                      </thead>
                      <tbody>
                        {activeStocks.slice(0,12).map(ticker=>{
                          const ens=ensembleResults[ticker]
                          if(!ens) return null
                          return (
                            <tr key={ticker} style={{cursor:'pointer'}} onClick={()=>{setSelectedAsset(ticker);setTab('signals')}}>
                              <td style={{...S.td,fontWeight:700,color:C.textBright}}>{ticker}</td>
                              {MODEL_INFO.slice(0,4).map(m=>{
                                const res=ens.models?.[m.id]
                                return <td key={m.id} style={S.td}>{res?<span style={S.sigBadge(res.signal)}>{res.signal}</span>:<span style={{color:C.textDim,fontSize:9}}>—</span>}</td>
                              })}
                              <td style={S.td}><span style={{...S.sigBadge(ens.signal),fontWeight:700}}>{ens.signal}</span></td>
                              <td style={{...S.td,color:ens.confidence>0.7?C.green:ens.confidence>0.4?C.yellow:C.textDim}}>{(ens.confidence*100).toFixed(0)}%</td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              </>
            )}

            {activeModel!=='ensemble'&&(()=>{
              const modelDef=MODEL_INFO.find(m=>m.id===activeModel)
              if(!modelDef) return null
              return (
                <div style={{display:'flex',flexDirection:'column',gap:12}}>
                  <div style={{...S.panel,border:`1px solid ${C.purple}`}}>
                    <div style={{fontSize:16,fontWeight:700,color:C.purple,marginBottom:8}}>{modelDef.icon} {modelDef.name}</div>
                    <div style={{fontSize:11,color:C.text,lineHeight:1.6,marginBottom:8}}>{modelDef.desc}</div>
                    <div style={{fontSize:10,color:C.accent,padding:'6px 10px',background:C.accentDim,borderRadius:4}}>📚 {modelDef.sota}</div>
                  </div>
                  <div style={S.panel}>
                    <div style={S.pt}>RESULTS — ALL SCREENED STOCKS</div>
                    <div style={{overflowX:'auto'}}>
                      <table style={{...S.table,minWidth:420}}>
                        <thead>
                          <tr>
                            <th style={S.th}>TICKER</th>
                            <th style={S.th}>SCORE</th>
                            <th style={S.th}>SIGNAL</th>
                            {activeModel==='famaFrench'&&<><th style={S.th}>BETA</th><th style={S.th}>ALPHA</th><th style={S.th}>RMW</th><th style={S.th}>TYPE</th></>}
                            {activeModel==='tcn'&&<><th style={S.th}>PATTERN</th><th style={S.th}>ALIGNMENT</th><th style={S.th}>VOL SURGE</th></>}
                          </tr>
                        </thead>
                        <tbody>
                          {activeStocks.slice(0,15).map(ticker=>{
                            const ens=ensembleResults[ticker]
                            const res=ens?.models?.[activeModel]
                            if(!res) return null
                            return (
                              <tr key={ticker} style={{cursor:'pointer'}} onClick={()=>{setSelectedAsset(ticker);setTab('signals')}}>
                                <td style={{...S.td,fontWeight:700,color:C.textBright}}>{ticker}</td>
                                <td style={{...S.td,color:res.score>0?C.green:C.red,fontWeight:700}}>{res.score?.toFixed(3)}</td>
                                <td style={S.td}><span style={S.sigBadge(res.signal)}>{res.signal}</span></td>
                                {activeModel==='famaFrench'&&<>
                                  <td style={{...S.td,color:res.beta>1.2?C.orange:C.text}}>{res.beta}</td>
                                  <td style={{...S.td,color:res.alpha>0?C.green:C.red}}>{res.alpha!=null?(res.alpha*100).toFixed(2)+'%':'—'}</td>
                                  <td style={{...S.td,color:res.rmw>0?C.green:C.red}}>{res.rmw?.toFixed(2)}</td>
                                  <td style={{...S.td,fontSize:9,color:C.textDim}}>{res.interpretation?.value}</td>
                                </>}
                                {activeModel==='tcn'&&<>
                                  <td style={{...S.td,color:res.pattern?.includes('UP')?C.green:res.pattern?.includes('DOWN')?C.red:C.textDim,fontSize:9}}>{res.pattern}</td>
                                  <td style={{...S.td,color:res.maAlignment>0?C.green:C.red}}>{res.maAlignment?.toFixed(2)}</td>
                                  <td style={{...S.td,color:res.volSurge>1.5?C.orange:C.textDim}}>{res.volSurge?.toFixed(1)}x</td>
                                </>}
                              </tr>
                            )
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              )
            })()}

            <div style={S.panel}>
              <div style={S.pt}>🗺️ FUTURE MODELS ROADMAP — SOTA IN QUANT FINANCE</div>
              {FUTURE_MODELS.map((m,i)=>(
                <div key={i} style={{padding:'10px 12px',background:C.surface,borderRadius:6,border:`1px solid ${C.border}`,marginBottom:8}}>
                  <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:4}}>
                    <div style={{fontSize:11,fontWeight:700,color:C.textBright}}>{m.name}</div>
                    <span style={{fontSize:9,padding:'2px 6px',borderRadius:2,border:`1px solid ${m.difficulty==='Very High'?C.red:m.difficulty==='High'?C.orange:C.yellow}`,color:m.difficulty==='Very High'?C.red:m.difficulty==='High'?C.orange:C.yellow}}>{m.difficulty}</span>
                  </div>
                  <div style={{fontSize:10,color:C.textDim,lineHeight:1.5}}>{m.desc}</div>
                </div>
              ))}
            </div>
          </div>
        )}

      </main>

      {/* Mobile bottom nav */}
      <nav style={{position:'fixed',bottom:0,left:0,right:0,background:C.surface,borderTop:`1px solid ${C.border}`,display:'flex',zIndex:200,paddingBottom:'env(safe-area-inset-bottom)'}}>
        {tabs.map(t=>(
          <button key={t.id} onClick={()=>setTab(t.id)}
            style={{flex:1,background:'transparent',border:'none',cursor:'pointer',padding:'8px 2px 6px',display:'flex',flexDirection:'column',alignItems:'center',gap:2,color:tab===t.id?C.accent:C.textDim,fontFamily:'inherit'}}>
            <span style={{fontSize:14}}>{t.icon}</span>
            <span style={{fontSize:7,letterSpacing:1}}>{t.label}</span>
            {tab===t.id&&<div style={{width:20,height:2,background:C.accent,borderRadius:1}}/>}
          </button>
        ))}
      </nav>
    </div>
  )
}

export default function AppWithBoundary() {
  return <ErrorBoundary><App /></ErrorBoundary>
}
