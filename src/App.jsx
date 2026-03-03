import { useState, useEffect, useCallback, useRef } from 'react'
import { AreaChart, Area, BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'
import { screenStocks, SCREENING_PROFILES, DEFAULT_CRITERIA } from './data/screener.js'
import { fetchMarketNews, fetchTickerNews, scoreWithClaude } from './data/news.js'
import { generateSignal, calcRSI, calcMACD, calcBollingerBands, calcATR } from './signals/signals.js'
import { backtestPortfolio } from './backtest/backtester.js'
import { agent } from './rl/agent.js'
import { trainEpisodes, stopTraining } from './rl/trainer.js'
import { loadPortfolio, savePortfolio, loadTradeLog, saveTradeLog, loadWatchlist, saveWatchlist, loadSettings, saveSettings, resetPortfolio as resetStored, normalizeTicket, displayTicker, debugStorage } from './data/persistence.js'
import { EXIT_PRESETS, checkExitSignal, checkPortfolioHeat } from './trading/exitStrategy.js'
import { evaluateAutoTrade, AUTO_TRADE_DEFAULTS } from './trading/autoTrader.js'
import { runEnsemble, MODEL_INFO, FUTURE_MODELS } from './models/ensemble.js'
import { fetchDynamicUniverse, getUniverseLabel, ALL_SEED } from './data/universe.js'
import { fetchCryptoPrices, buildCryptoBars } from './data/cryptoPrices.js'

const ALL_CRYPTO = ['X:BTCUSD','X:ETHUSD','X:SOLUSD']
const CRYPTO_DISPLAY = {'X:BTCUSD':'BTC','X:ETHUSD':'ETH','X:SOLUSD':'SOL'}
// Crypto prices fetched live from CoinGecko — see cryptoPrices.js
const SCREEN_INTERVAL = 30*60
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

const fmt = {
  price: v=>!v?'—':v>=1000?`$${(v/1000).toFixed(2)}K`:`$${v.toFixed(2)}`,
  pct: v=>`${v>=0?'+':''}${(v*100).toFixed(2)}%`,
  chg: v=>({ color:v>=0?C.green:C.red }),
  date: ts=>new Date(ts).toLocaleDateString(),
  ago: ts=>{ const m=Math.floor((Date.now()-new Date(ts))/60000); return m<60?`${m}m ago`:`${Math.floor(m/60)}h ago` },
}

function mockBars(ticker, days=400) {
  const seed=ticker.split('').reduce((a,c)=>a+c.charCodeAt(0),0)
  const basePx = 50+(seed%200)  // crypto gets real prices via buildCryptoBars
  const bars=[]; let price=basePx; const now=Date.now()
  for(let i=days;i>=0;i--) {
    const t=now-i*86400000,change=(Math.random()-0.48)*price*0.03,o=price
    price=Math.max(1,price+change)
    bars.push({t,o,h:Math.max(o,price)*(1+Math.random()*0.01),l:Math.min(o,price)*(1-Math.random()*0.01),c:price,v:1e6+Math.random()*5e6,vw:(o+price)/2})
  }
  return bars
}

function getApiKey() {
  try { const k=import.meta.env.VITE_POLYGON_API_KEY; if(k&&k.length>10&&k!=='your_polygon_api_key_here') return k } catch(e){}
  return null
}

function portfolioValue(port, prices) {
  let val=port.cash
  for(const [t,p] of Object.entries(port.positions)) {
    const px=prices[t]?.price||p.avgPrice; val+=p.shares*px
  }
  return val
}

export default function App() {
  const [tab,setTab]=useState('screen')
  const [screenProfile,setScreenProfile]=useState(()=>loadSettings().screenProfile||'momentum')
  const [screenResult,setScreenResult]=useState(null)
  const [isScreening,setIsScreening]=useState(false)
  const [activeStocks,setActiveStocks]=useState([])
  const [prices,setPrices]=useState({})
  const [bars,setBars]=useState({})
  const [signals,setSignals]=useState({})
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
  const [exitPreset,setExitPreset]=useState('moderate')
  const [exitAlerts,setExitAlerts]=useState([]) // warnings shown in UI
  // Email
  const [emailStatus,setEmailStatus]=useState(null)
  const [lastAlertSent,setLastAlertSent]=useState(null)
  const [lastBriefingSent,setLastBriefingSent]=useState(null)
  const EMAIL_TO = 'mithunghosh404@gmail.com'
  // News
  const [marketNews,setMarketNews]=useState([])
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

  // Keep refs in sync for use inside setInterval callbacks
  useEffect(()=>{ portfolioRef.current=portfolio },[portfolio])
  useEffect(()=>{ tradeLogRef.current=tradeLog },[tradeLog])
  useEffect(()=>{ pricesRef.current=prices },[prices])
  useEffect(()=>{ signalsRef.current=signals },[signals])
  useEffect(()=>{ barsRef.current=bars },[bars])
  useEffect(()=>{ ensembleRef.current=ensembleResults },[ensembleResults])
  useEffect(()=>{ tradeSizeRef.current=tradeSize },[tradeSize])

  // ── Persistence ──────────────────────────────────────────────────────────
  // Settings save (not critical for positions so useEffect is fine here)
  useEffect(()=>{ saveSettings({ tradeSize, screenProfile }) },[tradeSize,screenProfile])

  // ── Screener ─────────────────────────────────────────────────────────────
  const runScreener = useCallback(async(profile,custom)=>{
    if(isScreeningRef.current) return
    isScreeningRef.current=true
    setIsScreening(true); setScreenResult(null)

    const activeProfile=profile||screenProfile
    const criteria=custom||SCREENING_PROFILES[activeProfile]?.criteria||DEFAULT_CRITERIA

    // Build mock stocks
    // Fetch dynamic universe — live top movers + most active if API available
    const universeTickers = await fetchDynamicUniverse(apiKey)
    setUniverseLabel(getUniverseLabel(!!apiKey))
    const mockStocks=universeTickers.slice(0,25).map(ticker=>{
      const b=mockBars(ticker)
      const n=b.length
      const sig=generateSignal(b, agent.weights)
      return { ticker, price:b[n-1].c, change1d:(b[n-1].c-b[n-2].c)/b[n-2].c*100, change5d:(b[n-1].c-b[n-5].c)/b[n-5].c*100,
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

    const tickers=stocks.slice(0,15).map(s=>s.ticker)
    const newSignals={}
    for(const [t,b] of Object.entries(barsMap)) {
      if(b&&b.length>0) newSignals[t]=generateSignal(b,agent.weights)
    }

    // Compute ensemble for all assets
    const ensMap = {}
    for (const [t,b] of Object.entries(barsMap)) {
      try { ensMap[t] = runEnsemble(b, agent.weights, barsMap['SPY']||null, null) } catch(e){}
    }
    setEnsembleResults(ensMap)
    setScreenResult({ stocks:stocks.slice(0,15), timestamp:new Date(), mock:!isLive })
    setActiveStocks(tickers)
    setBars(barsMap)
    setPrices(priceMap)
    setSignals(newSignals)
    setWeights(agent.weights)
    setDataStatus({ live:isLive, keyFound:!!apiKey, lastUpdate:new Date(), screened:tickers.length })

    isScreeningRef.current=false
    setIsScreening(false)
    loadMarketNews()
  },[apiKey,screenProfile])

  // ── Scheduler ────────────────────────────────────────────────────────────
  const hasAutoRunRef=useRef(false)
  useEffect(()=>{
    if(!hasAutoRunRef.current){ hasAutoRunRef.current=true; runScreener() }
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
        setLastAlertSent(now)
      }
      // Daily briefing — send at 7-8am if not sent today
      const hour=new Date().getHours()
      const todayStr=new Date().toDateString()
      if(hour===7 && lastBriefingSent!==todayStr) {
        const sigs=signalsRef.current
        const allAssets=Object.entries(sigs).map(([ticker,s])=>({ticker,...s}))
        const buys=allAssets.filter(a=>a.signal==='BUY').sort((a,b)=>b.score-a.score)
        const sells=allAssets.filter(a=>a.signal==='SELL').sort((a,b)=>a.score-b.score)
        const topPicks=buys.slice(0,5).map(a=>({
          ticker:a.ticker, score:a.score, confidence:a.confidence,
          reason:`${a.signal} · conf ${Math.round(a.confidence*100)}%`
        }))
        sendEmail('daily_briefing',{
          buys, sells, topPicks,
          marketMood: buys.length>sells.length?'BULLISH':sells.length>buys.length?'BEARISH':'NEUTRAL',
          date: new Date().toLocaleDateString('en-US',{weekday:'long',month:'long',day:'numeric'})
        })
        setLastBriefingSent(todayStr)
      }
    },60000) // check every minute
    return()=>clearInterval(emailCheckRef.current)
  },[lastAlertSent,lastBriefingSent])

  // ── Paper Trading ─────────────────────────────────────────────────────────
  function executeTrade(ticker,side,price,signal,isAuto=false) {
    if(!price||price<=0) return
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
  async function loadMarketNews() {
    const news=await fetchMarketNews(10); setMarketNews(news)
    const ai=await scoreWithClaude(news); if(ai) setAiSentiment(ai)
  }
  async function loadTickerNews(ticker) { setTickerNews([]); setTickerNews(await fetchTickerNews(ticker,5)) }

  // ── RL Training ───────────────────────────────────────────────────────────
  async function startTraining() {
    if(isTraining||Object.keys(bars).length===0) return
    setIsTraining(true)
    await trainEpisodes(bars,30,
      ep=>{ setTrainingLog(prev=>[...prev.slice(-50),{episode:ep.episode,score:ep.result.ragScore.toFixed(3),sharpe:ep.result.sharpe.toFixed(2),ret:(ep.result.totalReturn*100).toFixed(1)+'%',regime:ep.regime}]); setRlProgress(ep.agentState); setWeights({...ep.currentWeights}) },
      done=>{ setWeights({...done.bestWeights}); setBacktestResult(backtestPortfolio(bars,done.bestWeights)); setIsTraining(false) }
    )
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

  const tabs=[
    {id:'screen',icon:'⟳',label:'SCREEN'},
    {id:'signals',icon:'📡',label:'SIGNALS'},
    {id:'news',icon:'📰',label:'NEWS'},
    {id:'paper',icon:'💼',label:'PAPER'},
    {id:'auto',icon:'🤖',label:'AUTO'},
    {id:'models',icon:'📊',label:'MODELS'},
    {id:'train',icon:'🧠',label:'TRAIN'},
  ]

  return (
    <div style={S.app}>
      {/* Header */}
      <header style={S.header}>
        <div style={{display:'flex',alignItems:'center',gap:10}}>
          <span style={S.logo}>STOCKBOT</span>
          {isScreening?<span style={S.badge(C.yellow)}>⟳ SCANNING</span>
            :!dataStatus.keyFound?<span style={S.badge(C.yellow)}>MOCK</span>
            :dataStatus.live?<span style={S.badge(C.green)}>LIVE · {dataStatus.screened}</span>
            :<span style={S.badge(C.yellow)}>LOADING</span>}
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

            {isScreening&&<div style={{...S.panel,textAlign:'center',padding:48}}><div style={{color:C.accent,letterSpacing:4}}>SCANNING MARKET...</div></div>}

            {!isScreening&&screenResult&&(
              <>
                <div style={{display:'grid',gridTemplateColumns:'repeat(4,1fr)',gap:8}}>
                  {[
                    {label:'ALL',value:screenResult.stocks.length,color:C.accent,filter:null},
                    {label:'BUY ▲',value:buys.length,color:C.green,filter:'BUY'},
                    {label:'SELL ▼',value:sells.length,color:C.red,filter:'SELL'},
                    {label:'UPDATED',value:screenResult.timestamp?.toLocaleTimeString(),color:C.textDim,filter:undefined},
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
                          <div style={{fontSize:10,fontWeight:700,color:C.textBright}}>{a.display}</div>
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
                      <thead><tr>{['#','TICKER','PRICE','1D%','5D%','RSI','MOMENTUM','VOL SURGE','TREND','SCORE','SIGNAL'].map(h=><th key={h} style={S.th}>{h}</th>)}</tr></thead>
                      <tbody>
                        {filteredStocks.map((s,i)=>{
                          const sig=signals[s.ticker]||{signal:'HOLD',score:0}
                          const rsi=s.bars?calcRSI(s.bars.map(b=>b.c),14):50
                          return (
                            <tr key={s.ticker} style={{cursor:'pointer'}} onClick={()=>{setSelectedAsset(s.ticker);setTab('signals')}}>
                              <td style={{...S.td,color:C.textDim}}>{i+1}</td>
                              <td style={{...S.td,color:C.textBright,fontWeight:700}}>{s.ticker}</td>
                              <td style={S.td}>{fmt.price(s.price)}</td>
                              <td style={{...S.td,...fmt.chg(s.change1d)}}>{s.change1d?.toFixed(2)}%</td>
                              <td style={{...S.td,...fmt.chg(s.change5d)}}>{s.change5d?.toFixed(2)}%</td>
                              <td style={{...S.td,color:rsi<30?C.green:rsi>70?C.red:C.textDim}}>{rsi.toFixed(0)}</td>
                              <td style={{...S.td,color:s.scores.momentum>0?C.green:C.red}}>{(s.scores.momentum*100).toFixed(0)}%</td>
                              <td style={{...S.td,color:s.scores.volumeSurge>0.5?C.orange:C.textDim}}>{s.scores.volumeSurge>0?'+':''}{(s.scores.volumeSurge*100).toFixed(0)}%</td>
                              <td style={{...S.td,color:s.scores.trendBreak>0?C.green:C.red}}>{s.scores.trendBreak>0?'↑':'↓'}</td>
                              <td style={{...S.td,color:s.composite>0?C.green:C.red,fontWeight:700}}>{s.composite.toFixed(3)}</td>
                              <td style={S.td}><span style={S.sigBadge(sig.signal)}>{sig.signal}</span></td>
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
                      <button style={{...S.btn('green'),flex:1,padding:10}} onClick={()=>executeTrade(selectedAsset,'BUY',px?.price,sig)}>▲ BUY {displayTicker(selectedAsset)}</button>
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
              <button style={S.btn('primary')} onClick={loadMarketNews}>⟳ REFRESH</button>
              <button style={S.btn('warn')} onClick={()=>sendEmail('daily_briefing',{
                buys:sortedSignals.filter(a=>a.sig?.signal==='BUY').slice(0,8).map(a=>({ticker:a.display,score:a.sig.score,confidence:a.sig.confidence,reason:`${a.sig.signal}`})),
                sells:sortedSignals.filter(a=>a.sig?.signal==='SELL').slice(0,5).map(a=>({ticker:a.display,score:a.sig.score,reason:'SELL signal'})),
                topPicks:sortedSignals.filter(a=>a.sig?.signal==='BUY').slice(0,5).map(a=>({ticker:a.display,score:a.sig.score,confidence:a.sig.confidence,reason:`Score ${a.sig.score.toFixed(3)}`})),
                marketMood:buys.length>sells.length?'BULLISH':sells.length>buys.length?'BEARISH':'NEUTRAL',
                date:new Date().toLocaleDateString()
              })}>
                ✉️ SEND BRIEFING NOW
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
              <div style={S.pt}>TRADE LOG ({tradeLog.length})</div>
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
                Automatic alerts sent when portfolio moves ±5%. Daily briefing every morning at 7am.
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
                <button style={S.btn('primary')} onClick={()=>sendEmail('daily_briefing',{
                  buys:sortedSignals.filter(a=>a.sig?.signal==='BUY').slice(0,8).map(a=>({ticker:a.display,score:a.sig.score,confidence:a.sig.confidence,reason:`Score ${a.sig.score.toFixed(3)}`})),
                  sells:sortedSignals.filter(a=>a.sig?.signal==='SELL').slice(0,5).map(a=>({ticker:a.display,score:a.sig.score,reason:'SELL signal'})),
                  topPicks:sortedSignals.filter(a=>a.sig?.signal==='BUY').slice(0,5).map(a=>({ticker:a.display,score:a.sig.score,confidence:a.sig.confidence,reason:`Ensemble BUY · conf ${Math.round(a.sig.confidence*100)}%`})),
                  marketMood:buys.length>sells.length?'BULLISH':sells.length>buys.length?'BEARISH':'NEUTRAL',
                  date:new Date().toLocaleDateString()
                })}>
                  ✉️ TEST DAILY BRIEFING
                </button>
              </div>
              {emailStatus&&<div style={{marginTop:8,fontSize:10,color:emailStatus==='sent'?C.green:C.yellow}}>{emailStatus==='sent'?'✓ Email sent successfully':emailStatus}</div>}
              <div style={{marginTop:12,fontSize:9,color:C.textDim}}>
                To set up: Go to resend.com → free account → get API key → add RESEND_API_KEY to Vercel dashboard → redeploy
              </div>
            </div>

            {/* Auto-trade toggle */}
            <div style={{...S.panel,border:`1px solid ${autoTradeSettings.enabled?C.purple:C.border}`}}>
              <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:12}}>
                <div style={S.pt}>🤖 AUTO-TRADING ENGINE</div>
                <button style={S.btn(autoTradeSettings.enabled?'active':'default')} onClick={()=>setAutoTradeSettings(p=>({...p,enabled:!p.enabled}))}>
                  {autoTradeSettings.enabled?'🟢 ON — CLICK TO DISABLE':'⚫ OFF — CLICK TO ENABLE'}
                </button>
              </div>
              {autoTradeSettings.enabled&&<div style={{fontSize:10,color:C.purple,marginBottom:12}}>⚠️ Auto-trading is ACTIVE. Bot will buy/sell automatically based on signals.</div>}
              <div style={{display:'grid',gridTemplateColumns:'repeat(2,1fr)',gap:10}}>
                {[
                  {label:'Min Signal Score',key:'minScore',type:'number',step:0.05},
                  {label:'Min Confidence %',key:'minConfidence',type:'number',step:0.05},
                  {label:'Max Positions',key:'maxPositions',type:'number',step:1},
                  {label:'Position Size %',key:'positionSizePct',type:'number',step:0.01},
                ].map(f=>(
                  <div key={f.key}>
                    <div style={{fontSize:9,color:C.textDim,marginBottom:4}}>{f.label}</div>
                    <input type="number" step={f.step} value={autoTradeSettings[f.key]}
                      onChange={e=>setAutoTradeSettings(p=>({...p,[f.key]:parseFloat(e.target.value)}))}
                      style={{...S.input,width:'100%'}}/>
                  </div>
                ))}
              </div>
              <div style={{marginTop:12}}>
                <div style={{fontSize:9,color:C.textDim,marginBottom:6}}>EXIT PRESET FOR AUTO-TRADES</div>
                <div style={{display:'flex',gap:6,flexWrap:'wrap'}}>
                  {Object.entries(EXIT_PRESETS).map(([key,p])=>(
                    <button key={key} style={{...S.btn(autoTradeSettings.exitPreset===key?'active':'default'),padding:'4px 10px',fontSize:9}} onClick={()=>setAutoTradeSettings(pr=>({...pr,exitPreset:key}))}>
                      {p.icon} {p.label}
                    </button>
                  ))}
                </div>
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
                <div style={S.pt}>BACKTEST RESULT — BEST WEIGHTS</div>
                <div style={{display:'grid',gridTemplateColumns:'repeat(3,1fr)',gap:8}}>
                  {[
                    {label:'RETURN',value:fmt.pct(backtestResult.totalReturn),color:backtestResult.totalReturn>=0?C.green:C.red},
                    {label:'SHARPE',value:backtestResult.sharpe.toFixed(2),color:C.accent},
                    {label:'MAX DD',value:`-${(backtestResult.maxDrawdown*100).toFixed(1)}%`,color:C.red},
                  ].map(m=>(
                    <div key={m.label} style={{textAlign:'center',padding:12,background:C.surface,borderRadius:6}}>
                      <div style={S.pt}>{m.label}</div>
                      <div style={{fontSize:18,fontWeight:700,color:m.color}}>{m.value}</div>
                    </div>
                  ))}
                </div>
              </div>
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