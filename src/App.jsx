import { useState, useEffect, useCallback, useRef } from 'react'
import { AreaChart, Area, BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'
import { screenStocks, SCREENING_PROFILES, DEFAULT_CRITERIA } from './data/screener.js'
import { fetchMarketNews, fetchTickerNews, scoreWithClaude } from './data/news.js'
import { generateSignal } from './signals/signals.js'
import { backtestPortfolio } from './backtest/backtester.js'
import { agent } from './rl/agent.js'
import { trainEpisodes, stopTraining } from './rl/trainer.js'
import { runEnsemble, MODEL_INFO, FUTURE_MODELS } from './models/ensemble.js'
import { loadPortfolio, savePortfolio, loadTradeLog, saveTradeLog, loadWatchlist, saveWatchlist, loadSettings, saveSettings, resetPortfolio as resetStoredPortfolio, normalizeTicket, displayTicker } from './data/persistence.js'
import { getCryptoBars } from './data/polygonClient.js'

const ALL_CRYPTO = ['X:BTCUSD','X:ETHUSD','X:SOLUSD']
const CRYPTO_DISPLAY = {'X:BTCUSD':'BTC','X:ETHUSD':'ETH','X:SOLUSD':'SOL'}
const SCREEN_INTERVAL_SECS = 30 * 60
const STARTING_CAPITAL = 100000

const C = {
  bg:'#050510', surface:'#0a0a1a', panel:'#0f0f22', border:'#1a1a3a',
  accent:'#00d4ff', accentDim:'#00d4ff22', green:'#00ff88', red:'#ff3366',
  yellow:'#ffd700', purple:'#a855f7', orange:'#ff8c00',
  text:'#e2e8f0', textDim:'#64748b', textBright:'#ffffff',
}

const S = {
  app:{ background:C.bg, minHeight:'100vh', fontFamily:"'IBM Plex Mono','Courier New',monospace", color:C.text, display:'flex', flexDirection:'column', paddingBottom:64 },
  header:{ background:`linear-gradient(90deg,${C.surface},#0a0a2a)`, borderBottom:`1px solid ${C.border}`, padding:'0 16px', height:52, display:'flex', alignItems:'center', justifyContent:'space-between', position:'sticky', top:0, zIndex:100 },
  logo:{ fontSize:16, fontWeight:700, letterSpacing:4, color:C.accent, textShadow:`0 0 20px ${C.accent}` },
  badge:c=>({ fontSize:9, padding:'2px 6px', borderRadius:2, border:`1px solid ${c}`, color:c, letterSpacing:1, whiteSpace:'nowrap' }),
  main:{ flex:1, padding:12, maxWidth:1400, margin:'0 auto', width:'100%' },
  panel:{ background:C.panel, border:`1px solid ${C.border}`, borderRadius:8, padding:14 },
  panelTitle:{ fontSize:9, letterSpacing:3, color:C.textDim, marginBottom:10, textTransform:'uppercase' },
  table:{ width:'100%', borderCollapse:'collapse', fontSize:11 },
  th:{ textAlign:'left', padding:'6px 8px', borderBottom:`1px solid ${C.border}`, color:C.textDim, fontSize:9, letterSpacing:1, fontWeight:400 },
  td:{ padding:'7px 8px', borderBottom:`1px solid ${C.border}40` },
  sigBadge:s=>({ display:'inline-block', padding:'2px 8px', borderRadius:3, fontSize:9, fontWeight:700, letterSpacing:1, background:s==='BUY'?'#00ff8822':s==='SELL'?'#ff336622':'#ffffff11', color:s==='BUY'?C.green:s==='SELL'?C.red:C.textDim, border:`1px solid ${s==='BUY'?C.green:s==='SELL'?C.red:C.border}` }),
  btn:(v='primary')=>({ background:v==='primary'?C.accentDim:v==='danger'?'#ff336622':v==='active'?'#a855f722':v==='green'?'#00ff8822':'transparent', border:`1px solid ${v==='primary'?C.accent:v==='danger'?C.red:v==='active'?C.purple:v==='green'?C.green:C.border}`, color:v==='primary'?C.accent:v==='danger'?C.red:v==='active'?C.purple:v==='green'?C.green:C.textDim, padding:'7px 14px', borderRadius:4, cursor:'pointer', fontSize:10, letterSpacing:1, fontFamily:'inherit', transition:'all 0.2s' }),
}

const fmt = {
  price:v=>!v?'—':v>=1000?`$${(v/1000).toFixed(2)}K`:`$${v.toFixed(2)}`,
  pct:v=>`${v>=0?'+':''}${(v*100).toFixed(2)}%`,
  chgPct:v=>({ color:v>=0?C.green:C.red }),
  date:ts=>new Date(ts).toLocaleDateString(),
  time:ts=>new Date(ts).toLocaleTimeString(),
  ago:ts=>{ const m=Math.floor((Date.now()-new Date(ts))/60000); return m<60?`${m}m ago`:`${Math.floor(m/60)}h ago` },
}

function mockBars(ticker, days=400) {
  const seed=ticker.split('').reduce((a,c)=>a+c.charCodeAt(0),0)
  const bars=[]; let price=50+(seed%200); const now=Date.now()
  for(let i=days;i>=0;i--){
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

// ── Education content ──────────────────────────────────────────────────────
const EDUCATION = {
  momentum: {
    icon:'🚀', title:'Momentum', color:C.accent,
    what:'Momentum measures how strongly a stock has been trending. Stocks that have been going up tend to keep going up — this is one of the most well-documented effects in finance.',
    how:'We look at 1-week and 1-month price returns. A stock up 20% this month scores higher than one up 2%.',
    when:'Works best in trending bull markets. Less reliable during sideways or volatile markets.',
    example:'NVDA rallied from $400 → $900 in 6 months. Momentum signal would have flagged it early based on consistent weekly gains.',
    risk:'Can reverse sharply when trend breaks. Always use with a stop loss.',
  },
  breakout: {
    icon:'⚡', title:'Breakout', color:C.yellow,
    what:'Breakout signals occur when a stock crosses above key price levels (moving averages) on unusually high volume — suggesting institutional buying.',
    how:'We compare today\'s volume to the 20-day average. A 3x volume spike with price above both 5-day and 20-day MA = strong breakout.',
    when:'Best at the start of new trends. Often follows earnings beats, news catalysts, or sector rotation.',
    example:'PLTR broke above its 200-day MA on 5x normal volume after a Pentagon contract announcement — signaling a new uptrend.',
    risk:'False breakouts are common. Wait for confirmation (2nd day above level) to reduce fakeouts.',
  },
  meanReversion: {
    icon:'🔄', title:'Mean Reversion', color:C.purple,
    what:'Mean reversion assumes that stocks which have fallen sharply will bounce back toward their average. When RSI drops below 30, the stock is "oversold" and likely to recover.',
    how:'RSI (Relative Strength Index) measures recent gains vs losses over 14 days. Below 30 = oversold = potential buy. Above 70 = overbought = potential sell.',
    when:'Works best in stable, range-bound markets. Less effective in strong downtrends.',
    example:'A stock drops 15% in a week on no fundamental news. RSI hits 25. Mean reversion signal fires — stock recovers 8% the next week.',
    risk:'A falling stock can keep falling. Never catch a falling knife without checking why it\'s dropping.',
  },
  volatility: {
    icon:'⚡', title:'Volatility', color:C.orange,
    what:'Volatility measures how wildly a stock\'s price swings. High volatility = big moves both up and down. Low volatility = stable, quiet stock.',
    how:'We calculate the standard deviation of daily returns over 5 and 20 days. A spike in short-term volatility vs historical = opportunity or warning.',
    when:'High vol = good for options traders and momentum players. Low vol = better for position traders holding weeks/months.',
    example:'Iran conflict news spikes oil prices. OXY and CVX show 3x their normal daily volatility — signaling a major move is underway.',
    risk:'Volatility cuts both ways. High vol stocks can lose 10% as fast as they gain it.',
  },
  trend: {
    icon:'📈', title:'Trend', color:C.green,
    what:'Trend factor measures whether a stock is trading above or below its key price levels (VWAP and 20-day moving average). Price above both = bullish trend.',
    how:'VWAP (Volume Weighted Average Price) is the average price weighted by volume — used by institutions as a fair value reference. Price > VWAP = buyers in control.',
    when:'Strong trends persist. A stock above its 20-day MA for 3+ weeks is in a confirmed uptrend.',
    example:'MSFT trades above both VWAP and 20-day MA every day for a month. Trend factor stays positive, suggesting institutional accumulation.',
    risk:'Trend can reverse quickly on macro news. Check volume — a trend with declining volume is weakening.',
  },
  rl: {
    icon:'🤖', title:'RL Agent', color:C.purple,
    what:'The Reinforcement Learning agent learns which combination of the 5 factors works best in the current market regime, optimizing for risk-adjusted returns.',
    how:'The agent tries 30 different weight combinations per training batch, backtests each one, and keeps the weights that produced the best Sharpe + Calmar ratio.',
    when:'Run training when market conditions change significantly — after a crash, a sector rotation, or a major macro shift.',
    example:'In a bull market, the agent learns to weight momentum 50% and volume 30%. In a bear market, it shifts to mean reversion 60% and volatility 20%.',
    risk:'Past performance doesn\'t guarantee future results. The agent optimizes for historical data — real markets may behave differently.',
  },
}

// ── Paper Trading Engine ───────────────────────────────────────────────────
function createPortfolio() {
  return { cash: STARTING_CAPITAL, positions: {}, history: [{ value: STARTING_CAPITAL, t: Date.now() }] }
}

function portfolioValue(portfolio, prices) {
  let val = portfolio.cash
  for (const [ticker, pos] of Object.entries(portfolio.positions)) {
    const px = prices[ticker]?.price || pos.avgPrice
    val += pos.shares * px
  }
  return val
}

export default function App() {
  const [tab, setTab] = useState('screen')
  const [screenProfile, setScreenProfile] = useState('momentum')
  const [screenResult, setScreenResult] = useState(null)
  const [isScreening, setIsScreening] = useState(false)
  const [activeStocks, setActiveStocks] = useState([])
  const [prices, setPrices] = useState({})
  const [bars, setBars] = useState({})
  const [signals, setSignals] = useState({})
  const [backtestResult, setBacktestResult] = useState(null)
  const [trainingLog, setTrainingLog] = useState([])
  const [isTraining, setIsTraining] = useState(false)
  const [dataStatus, setDataStatus] = useState({ live:false, usingMock:true, keyFound:false, lastUpdate:null })
  const [selectedAsset, setSelectedAsset] = useState(null)
  const [rlProgress, setRlProgress] = useState(null)
  const [weights, setWeights] = useState(agent.weights)
  const [nextScreenIn, setNextScreenIn] = useState(SCREEN_INTERVAL_SECS)
  const [autoScreenEnabled, setAutoScreenEnabled] = useState(true)
  const [hasAutoRun, setHasAutoRun] = useState(false)
  const [signalFilter, setSignalFilter] = useState(null) // null=all, 'BUY', 'SELL'
  const [customCriteria, setCustomCriteria] = useState(DEFAULT_CRITERIA)
  const [ensembleResults, setEnsembleResults] = useState({})
  const [activeModel, setActiveModel] = useState('ensemble')
  // Paper trading
  const [portfolio, setPortfolio] = useState(() => loadPortfolio())
  const [tradeLog, setTradeLog] = useState(() => loadTradeLog())
  const [tradeSize, setTradeSize] = useState(() => loadSettings().tradeSize || 2000)
  const [watchlist, setWatchlist] = useState(() => loadWatchlist())
  const [watchlistInput, setWatchlistInput] = useState('')
  const [addTickerError, setAddTickerError] = useState('')
  // News
  const [marketNews, setMarketNews] = useState([])
  const [tickerNews, setTickerNews] = useState([])
  const [newsLoading, setNewsLoading] = useState(false)
  const [aiSentiment, setAiSentiment] = useState(null)
  // Education
  const [eduModal, setEduModal] = useState(null)

  const screenTimerRef = useRef(null)
  const countdownRef = useRef(null)
  const isScreeningRef = useRef(false)
  const apiKey = getApiKey()

  // ── Persistence — auto-save on changes ──────────────────────────────────
  useEffect(() => { savePortfolio(portfolio) }, [portfolio])
  useEffect(() => { saveTradeLog(tradeLog) }, [tradeLog])
  useEffect(() => { saveWatchlist(watchlist) }, [watchlist])
  useEffect(() => { saveSettings({ tradeSize, screenProfile, autoScreenEnabled: true }) }, [tradeSize, screenProfile])

  // ── Screener ──────────────────────────────────────────────────────────────
  const runScreener = useCallback(async (profile, custom) => {
    if (isScreeningRef.current) return
    isScreeningRef.current = true
    setIsScreening(true)
    setScreenResult(null)

    const activeProfile = profile || screenProfile
    const criteria = custom || SCREENING_PROFILES[activeProfile]?.criteria || DEFAULT_CRITERIA

    const buildMock = () => {
      const tickers = ['NVDA','TSLA','AMD','MSTR','PLTR','COIN','META','AAPL','AMZN','GOOGL','MSFT','ARM','SMCI','AVGO','MARA','IONQ','RKLB','HOOD','ACHR','RIOT']
      const stocks = tickers.map(ticker => {
        const b = mockBars(ticker)
        const last=b[b.length-1], prev2=b[b.length-2], prev5=b[b.length-5]
        return { ticker, price:last.c, change1d:(last.c-prev2.c)/prev2.c*100, change5d:(last.c-prev5.c)/prev5.c*100, volume:last.v,
          composite:Math.random()*2-0.3, scores:{ momentum:Math.random()-0.2, volatility:Math.random()*0.05, volumeSurge:Math.random()*2-0.5, trendBreak:Math.random()-0.2, rsiOversold:Math.random()-0.3 }, bars:b }
      }).sort((a,b)=>b.composite-a.composite)
      return stocks
    }

    let stocks = buildMock()
    let isLive = false

    if (apiKey) {
      try {
        const result = await screenStocks(criteria, 20)
        if (result.stocks.length > 0) { stocks = result.stocks; isLive = true }
      } catch(e) { console.error('[Screener]', e) }
    }

    const barsMap = {}
    stocks.forEach(s => { barsMap[s.ticker] = (s.bars||[]).map(b=>({t:b.t,o:b.o,h:b.h,l:b.l,c:b.c,v:b.v,vw:b.vw||b.c})) })
    ALL_CRYPTO.forEach(t => { if(!barsMap[t]) barsMap[t] = mockBars(CRYPTO_DISPLAY[t]) })

    // Compute ensemble for all assets
    const ensembleMap = {}
    for (const [t,b] of Object.entries(barsMap)) {
      try { ensembleMap[t] = runEnsemble(b, agent.weights, barsMap['SPY']||null, null) } catch(e){}
    }
    setEnsembleResults(ensembleMap)

    const priceMap = {}
    stocks.forEach(s => { priceMap[s.ticker] = { price:s.price, changePct:s.change1d, change:s.price*s.change1d/100, volume:s.volume } })

    const tickers = stocks.slice(0,15).map(s=>s.ticker)
    setScreenResult({ stocks:stocks.slice(0,15), errors:[], criteria, timestamp:new Date(), mock:!isLive })
    setActiveStocks(tickers)
    setBars(barsMap)
    setPrices(priceMap)
    recomputeSignals(barsMap)
    setDataStatus({ live:isLive, usingMock:!isLive, keyFound:!!apiKey, lastUpdate:new Date(), screened:tickers.length })

    isScreeningRef.current = false
    setIsScreening(false)

    // Auto-load market news
    loadMarketNews()
  }, [apiKey, screenProfile])

  function recomputeSignals(barsMap) {
    const w = agent.weights
    const sigs = {}
    for (const [t,b] of Object.entries(barsMap)) { if(b&&b.length>0) sigs[t] = generateSignal(b,w) }
    setSignals(sigs)
    setWeights(w)
  }

  // ── Scheduler ─────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!hasAutoRun) { setHasAutoRun(true); runScreener() }
  }, [runScreener, hasAutoRun])

  useEffect(() => {
    if (!autoScreenEnabled) { clearInterval(screenTimerRef.current); clearInterval(countdownRef.current); return }
    setNextScreenIn(SCREEN_INTERVAL_SECS)
    countdownRef.current = setInterval(() => setNextScreenIn(p => p<=1?SCREEN_INTERVAL_SECS:p-1), 1000)
    screenTimerRef.current = setInterval(() => runScreener(), SCREEN_INTERVAL_SECS*1000)
    return () => { clearInterval(screenTimerRef.current); clearInterval(countdownRef.current) }
  }, [autoScreenEnabled, runScreener])

  // ── News ──────────────────────────────────────────────────────────────────
  async function loadMarketNews() {
    setNewsLoading(true)
    const news = await fetchMarketNews(10)
    setMarketNews(news)
    setNewsLoading(false)
    // Try Claude AI sentiment on top headlines
    const ai = await scoreWithClaude(news)
    if (ai) setAiSentiment(ai)
  }

  async function loadTickerNews(ticker) {
    setTickerNews([])
    const news = await fetchTickerNews(ticker, 5)
    setTickerNews(news)
  }

  // ── Paper Trading ─────────────────────────────────────────────────────────
  function executeTrade(ticker, side, price, signal) {
    if (!price || price <= 0) return
    const shares = Math.floor(tradeSize / price)
    if (shares <= 0) return

    setPortfolio(prev => {
      const next = { ...prev, positions: { ...prev.positions } }

      if (side === 'BUY') {
        const cost = shares * price
        if (cost > next.cash) return prev // not enough cash
        next.cash -= cost
        if (next.positions[ticker]) {
          const existing = next.positions[ticker]
          const totalShares = existing.shares + shares
          next.positions[ticker] = { shares:totalShares, avgPrice:(existing.avgPrice*existing.shares + price*shares)/totalShares, side:'LONG' }
        } else {
          next.positions[ticker] = { shares, avgPrice:price, side:'LONG' }
        }
      } else if (side === 'SELL') {
        const pos = next.positions[ticker]
        if (pos && pos.shares > 0) {
          // Close existing long
          const proceeds = pos.shares * price
          next.cash += proceeds
          delete next.positions[ticker]
        } else {
          // Short sell
          const proceeds = shares * price
          next.cash += proceeds
          next.positions[ticker] = { shares, avgPrice:price, side:'SHORT' }
        }
      }

      const val = portfolioValue(next, prices)
      next.history = [...(prev.history||[]), { value:val, t:Date.now() }].slice(-200)
      return next
    })

    setTradeLog(prev => [{
      id: Date.now(), ticker, side, price, shares,
      value: shares * price,
      signal: signal?.score?.toFixed(3) || '—',
      time: new Date().toLocaleTimeString(),
      pnl: null,
    }, ...prev.slice(0,99)])
  }

  function closePosition(ticker) {
    const pos = portfolio.positions[ticker]
    const px = prices[ticker]?.price || pos.avgPrice
    if (!pos) return

    const pnl = pos.side === 'LONG'
      ? (px - pos.avgPrice) * pos.shares
      : (pos.avgPrice - px) * pos.shares

    setPortfolio(prev => {
      const next = { ...prev, positions: { ...prev.positions } }
      if (pos.side === 'LONG') next.cash += pos.shares * px
      else next.cash -= pos.shares * px
      delete next.positions[ticker]
      const val = portfolioValue(next, prices)
      next.history = [...(prev.history||[]), { value:val, t:Date.now() }].slice(-200)
      return next
    })

    setTradeLog(prev => [{
      id: Date.now(), ticker, side:'CLOSE', price:px, shares:pos.shares,
      value: pos.shares * px, signal:'—', time:new Date().toLocaleTimeString(),
      pnl: pnl.toFixed(2),
    }, ...prev.slice(0,99)])
  }

  function resetPortfolio() {
    resetStoredPortfolio()
    setPortfolio(loadPortfolio())
    setTradeLog([])
  }

  // ── Watchlist management ─────────────────────────────────────────────────
  function addToWatchlist(input) {
    const ticker = normalizeTicket(input)
    if (!ticker || ticker.length < 1) { setAddTickerError('Enter a ticker symbol'); return }
    if (watchlist.includes(ticker)) { setAddTickerError(`${displayTicker(ticker)} already in watchlist`); return }
    setAddTickerError('')
    setWatchlist(prev => [...prev, ticker])
    setWatchlistInput('')
    // Add mock bars if not already tracked
    if (!bars[ticker]) {
      setBars(prev => ({ ...prev, [ticker]: mockBars(displayTicker(ticker)) }))
    }
  }

  function removeFromWatchlist(ticker) {
    setWatchlist(prev => prev.filter(t => t !== ticker))
  }

  // ── RL Training ───────────────────────────────────────────────────────────
  async function startTraining() {
    if (isTraining || Object.keys(bars).length===0) return
    setIsTraining(true)
    await trainEpisodes(bars, 30,
      ep => {
        setTrainingLog(prev=>[...prev.slice(-50),{ episode:ep.episode, score:ep.result.ragScore.toFixed(3), sharpe:ep.result.sharpe.toFixed(2), ret:(ep.result.totalReturn*100).toFixed(1)+'%', regime:ep.regime }])
        setRlProgress(ep.agentState)
        setWeights({...ep.currentWeights})
      },
      done => { setWeights({...done.bestWeights}); recomputeSignals(bars); setBacktestResult(backtestPortfolio(bars,done.bestWeights)); setIsTraining(false) }
    )
  }

  // ── Computed ──────────────────────────────────────────────────────────────
  const allTracked = [...activeStocks.map(t=>({ticker:t,display:t})), ...ALL_CRYPTO.map(t=>({ticker:t,display:CRYPTO_DISPLAY[t]}))]
  const sortedSignals = allTracked.map(a=>({...a,sig:signals[a.ticker]||{score:0,signal:'HOLD',factors:{},confidence:0},px:prices[a.ticker]})).sort((a,b)=>(b.sig?.score||0)-(a.sig?.score||0))
  const buys = sortedSignals.filter(a=>a.sig?.signal==='BUY')
  const sells = sortedSignals.filter(a=>a.sig?.signal==='SELL')
  const curPortfolioValue = portfolioValue(portfolio, prices)
  const totalPnL = curPortfolioValue - STARTING_CAPITAL
  const fmtCD = s=>`${Math.floor(s/60)}:${String(s%60).padStart(2,'0')}`

  const statusBadge = () => {
    if (isScreening) return <span style={S.badge(C.yellow)}>⟳ SCANNING...</span>
    if (!dataStatus.keyFound) return <span style={S.badge(C.yellow)}>MOCK — NO KEY</span>
    if (dataStatus.live) return <span style={S.badge(C.green)}>LIVE · {dataStatus.screened} STOCKS</span>
    return <span style={S.badge(C.yellow)}>LOADING...</span>
  }

  // ── Education Modal ───────────────────────────────────────────────────────
  const EduModal = ({ item, onClose }) => (
    <div style={{ position:'fixed', inset:0, background:'#000000cc', zIndex:1000, display:'flex', alignItems:'center', justifyContent:'center', padding:16 }} onClick={onClose}>
      <div style={{ background:C.panel, border:`1px solid ${item.color}`, borderRadius:12, padding:24, maxWidth:480, width:'100%', maxHeight:'80vh', overflowY:'auto' }} onClick={e=>e.stopPropagation()}>
        <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:16 }}>
          <div style={{ fontSize:18, fontWeight:700, color:item.color }}>{item.icon} {item.title}</div>
          <button onClick={onClose} style={{ background:'transparent', border:'none', color:C.textDim, fontSize:18, cursor:'pointer' }}>✕</button>
        </div>
        {[
          { label:'WHAT IT IS', text:item.what },
          { label:'HOW WE CALCULATE IT', text:item.how },
          { label:'WHEN IT WORKS BEST', text:item.when },
          { label:'REAL EXAMPLE', text:item.example },
          { label:'⚠️ RISK', text:item.risk },
        ].map(s => (
          <div key={s.label} style={{ marginBottom:14 }}>
            <div style={{ fontSize:9, letterSpacing:3, color:C.textDim, marginBottom:6 }}>{s.label}</div>
            <div style={{ fontSize:12, color:C.text, lineHeight:1.6 }}>{s.text}</div>
          </div>
        ))}
        <button style={{ ...S.btn('primary'), width:'100%', marginTop:8 }} onClick={onClose}>GOT IT</button>
      </div>
    </div>
  )

  const tabs = [
    { id:'screen', icon:'⟳', label:'SCREEN' },
    { id:'news', icon:'📰', label:'NEWS' },
    { id:'signals', icon:'📡', label:'SIGNALS' },
    { id:'paper', icon:'💼', label:'PAPER' },
    { id:'models', icon:'🧠', label:'MODELS' },
    { id:'train', icon:'🤖', label:'TRAIN' },
  ]

  return (
    <div style={S.app}>
      {eduModal && <EduModal item={EDUCATION[eduModal]} onClose={()=>setEduModal(null)}/>}

      {/* Header */}
      <header style={S.header}>
        <div style={{ display:'flex', alignItems:'center', gap:10 }}>
          <span style={S.logo}>STOCKBOT</span>
          {statusBadge()}
        </div>
        <div style={{ fontSize:9, color:C.textDim, textAlign:'right' }}>
          {autoScreenEnabled && <span style={{ color:C.accent }}>⟳ {fmtCD(nextScreenIn)}  </span>}
          <span>GEN {agent.generation}</span>
        </div>
      </header>

      <main style={S.main}>

        {/* ══ SCREEN ══ */}
        {tab==='screen' && (
          <div style={{ display:'flex', flexDirection:'column', gap:12 }}>

            {/* Scheduler bar */}
            <div style={{ background:C.surface, border:`1px solid ${C.border}`, borderRadius:8, padding:'10px 14px', display:'flex', alignItems:'center', justifyContent:'space-between', flexWrap:'wrap', gap:8 }}>
              <div style={{ display:'flex', alignItems:'center', gap:10 }}>
                <div style={{ width:8, height:8, borderRadius:'50%', background:autoScreenEnabled?C.green:C.textDim, boxShadow:autoScreenEnabled?`0 0 8px ${C.green}`:'none' }}/>
                <span style={{ fontSize:10, color:C.textBright }}>{autoScreenEnabled?`AUTO-SCREEN · NEXT IN ${fmtCD(nextScreenIn)}`:'PAUSED'}</span>
              </div>
              <div style={{ display:'flex', gap:6 }}>
                <button style={{ ...S.btn(autoScreenEnabled?'danger':'primary'), padding:'4px 12px', fontSize:9 }} onClick={()=>setAutoScreenEnabled(p=>!p)}>
                  {autoScreenEnabled?'⏸ PAUSE':'▶ RESUME'}
                </button>
                <button style={{ ...S.btn('primary'), padding:'4px 12px', fontSize:9 }} onClick={()=>runScreener(screenProfile,customCriteria)} disabled={isScreening}>
                  {isScreening?'⟳ SCANNING...':'⟳ SCAN NOW'}
                </button>
              </div>
            </div>

            {/* Profile buttons + learn */}
            <div style={{ display:'flex', gap:6, flexWrap:'wrap', alignItems:'center' }}>
              {Object.entries(SCREENING_PROFILES).map(([key,p])=>(
                <button key={key} style={{ ...S.btn(screenProfile===key?'active':'default'), fontSize:10 }}
                  onClick={()=>{ setScreenProfile(key); setCustomCriteria(p.criteria); runScreener(key,p.criteria) }}>
                  {p.icon} {p.label}
                </button>
              ))}
              <button style={{ ...S.btn(), padding:'6px 10px', fontSize:9, marginLeft:'auto' }} onClick={()=>setEduModal('momentum')}>
                📖 LEARN INDICATORS
              </button>
            </div>

            {/* Learn buttons per factor */}
            <div style={{ display:'flex', gap:6, flexWrap:'wrap' }}>
              {['momentum','breakout','meanReversion','volatility','trend','rl'].map(k=>(
                <button key={k} style={{ background:'transparent', border:`1px solid ${C.border}`, color:C.textDim, padding:'3px 10px', borderRadius:12, cursor:'pointer', fontSize:9, fontFamily:'inherit' }}
                  onClick={()=>setEduModal(k)}>
                  {EDUCATION[k]?.icon} {EDUCATION[k]?.title}
                </button>
              ))}
            </div>

            {isScreening && (
              <div style={{ ...S.panel, textAlign:'center', padding:48 }}>
                <div style={{ fontSize:28, marginBottom:8 }}>⟳</div>
                <div style={{ color:C.accent, letterSpacing:4, fontSize:11 }}>SCANNING MARKET...</div>
              </div>
            )}

            {!isScreening && screenResult && (
              <>
                <div style={{ display:'grid', gridTemplateColumns:'repeat(4,1fr)', gap:8 }}>
                  {[
                    { label:'ALL', value:screenResult.stocks.length, color:C.accent, filter:null },
                    { label:'BUY ▲', value:buys.length, color:C.green, filter:'BUY' },
                    { label:'SELL ▼', value:sells.length, color:C.red, filter:'SELL' },
                    { label:'UPDATED', value:screenResult.timestamp?.toLocaleTimeString(), color:C.textDim, filter:undefined },
                  ].map(m=>(
                    <div key={m.label}
                      onClick={m.filter!==undefined ? ()=>setSignalFilter(f=>f===m.filter?null:m.filter) : undefined}
                      style={{ ...S.panel, cursor:m.filter!==undefined?'pointer':'default',
                        border:`1px solid ${signalFilter===m.filter&&m.filter!==undefined?m.color:C.border}`,
                        background:signalFilter===m.filter&&m.filter!==undefined?m.color+'11':C.panel }}>
                      <div style={S.panelTitle}>{m.label}</div>
                      <div style={{ fontSize:18, fontWeight:700, color:m.color }}>{m.value}</div>
                      {m.filter!==undefined && <div style={{ fontSize:8, color:C.textDim, marginTop:2 }}>{signalFilter===m.filter?'CLICK TO CLEAR':'CLICK TO FILTER'}</div>}
                    </div>
                  ))}
                </div>

                {/* Heatmap */}
                <div style={S.panel}>
                  <div style={S.panelTitle}>SIGNAL HEATMAP {screenResult.mock&&<span style={{ color:C.yellow }}>(MOCK)</span>}</div>
                  <div style={{ display:'grid', gridTemplateColumns:'repeat(5,1fr)', gap:5 }}>
                    {sortedSignals.filter(a=>!signalFilter||a.sig?.signal===signalFilter).slice(0,20).map(a=>{
                      const score=a.sig?.score||0, hue=score>0?'144,255,136':'255,51,102'
                      return (
                        <div key={a.ticker} onClick={()=>{ setSelectedAsset(a.ticker); setTab('signals') }}
                          style={{ padding:'7px 4px', borderRadius:4, cursor:'pointer', background:`rgba(${hue},${Math.abs(score)*0.4})`, border:`1px solid rgba(${hue},${Math.abs(score)*0.7})`, textAlign:'center' }}>
                          <div style={{ fontSize:10, fontWeight:700, color:C.textBright }}>{a.display}</div>
                          <div style={{ fontSize:9, color:score>0?C.green:score<0?C.red:C.textDim }}>{score>=0?'+':''}{score.toFixed(2)}</div>
                        </div>
                      )
                    })}
                  </div>
                </div>

                {/* Ranked table */}
                <div style={S.panel}>
                  <div style={S.panelTitle}>RANKED STOCKS</div>
                  <div style={{ overflowX:'auto', WebkitOverflowScrolling:'touch' }}>
                    <table style={{ ...S.table, minWidth:580 }}>
                      <thead>
                        <tr>{['#','TICKER','PRICE','1D%','5D%','MOMENTUM','VOL SURGE','TREND','SCORE','SIGNAL'].map(h=><th key={h} style={S.th}>{h}</th>)}</tr>
                      </thead>
                      <tbody>
                        {screenResult.stocks.filter(s=>!signalFilter||(signals[s.ticker]||{signal:'HOLD'}).signal===signalFilter).map((s,i)=>{
                          const sig=signals[s.ticker]||{signal:'HOLD',score:0}
                          return (
                            <tr key={s.ticker} style={{ cursor:'pointer' }} onClick={()=>{ setSelectedAsset(s.ticker); setTab('signals') }}>
                              <td style={{ ...S.td, color:C.textDim }}>{i+1}</td>
                              <td style={{ ...S.td, color:C.textBright, fontWeight:700 }}>{s.ticker}</td>
                              <td style={S.td}>{fmt.price(s.price)}</td>
                              <td style={{ ...S.td, ...fmt.chgPct(s.change1d) }}>{s.change1d?.toFixed(2)}%</td>
                              <td style={{ ...S.td, ...fmt.chgPct(s.change5d) }}>{s.change5d?.toFixed(2)}%</td>
                              <td style={{ ...S.td, color:s.scores.momentum>0?C.green:C.red }}>{(s.scores.momentum*100).toFixed(0)}%</td>
                              <td style={{ ...S.td, color:s.scores.volumeSurge>0.5?C.orange:C.textDim }}>{s.scores.volumeSurge>0?'+':''}{(s.scores.volumeSurge*100).toFixed(0)}%</td>
                              <td style={{ ...S.td, color:s.scores.trendBreak>0?C.green:C.red }}>{s.scores.trendBreak>0?'↑':'↓'}{Math.abs(s.scores.trendBreak*100).toFixed(0)}%</td>
                              <td style={{ ...S.td, color:s.composite>0?C.green:C.red, fontWeight:700 }}>{s.composite.toFixed(3)}</td>
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

        {/* ══ NEWS ══ */}
        {tab==='news' && (
          <div style={{ display:'flex', flexDirection:'column', gap:12 }}>
            <div style={{ display:'flex', gap:8, alignItems:'center' }}>
              <button style={S.btn('primary')} onClick={loadMarketNews} disabled={newsLoading}>
                {newsLoading?'⟳ LOADING...':'⟳ REFRESH NEWS'}
              </button>
              <span style={{ fontSize:10, color:C.textDim }}>Auto-loads on each screen scan · powers signal strength</span>
            </div>

            {/* AI Sentiment summary */}
            {aiSentiment && (
              <div style={{ ...S.panel, border:`1px solid ${aiSentiment.sentiment==='BULLISH'?C.green:aiSentiment.sentiment==='BEARISH'?C.red:C.border}` }}>
                <div style={S.panelTitle}>🤖 CLAUDE AI MARKET SENTIMENT</div>
                <div style={{ display:'flex', alignItems:'center', gap:16 }}>
                  <div style={{ fontSize:24, fontWeight:700, color:aiSentiment.sentiment==='BULLISH'?C.green:aiSentiment.sentiment==='BEARISH'?C.red:C.textDim }}>
                    {aiSentiment.sentiment}
                  </div>
                  <div>
                    <div style={{ fontSize:11, color:C.text }}>{aiSentiment.reason}</div>
                    <div style={{ fontSize:10, color:C.textDim, marginTop:4 }}>
                      Score: {aiSentiment.score?.toFixed(2)} · Tradeable: {aiSentiment.tradeable?'YES':'NO'}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Market news feed */}
            <div style={S.panel}>
              <div style={S.panelTitle}>MARKET NEWS FEED — SENTIMENT SCORED</div>
              {marketNews.length === 0 && !newsLoading && (
                <div style={{ textAlign:'center', padding:32, color:C.textDim }}>Loading news...</div>
              )}
              <div style={{ display:'flex', flexDirection:'column', gap:8 }}>
                {marketNews.map((n,i)=>(
                  <div key={i} style={{ padding:'10px 12px', borderRadius:6, background:C.surface, border:`1px solid ${n.sentiment.label==='BULLISH'?C.green+'44':n.sentiment.label==='BEARISH'?C.red+'44':C.border}` }}>
                    <div style={{ display:'flex', justifyContent:'space-between', alignItems:'flex-start', gap:8, marginBottom:6 }}>
                      <div style={{ fontSize:11, color:C.textBright, lineHeight:1.4, flex:1 }}>{n.title}</div>
                      <span style={{ ...S.sigBadge(n.sentiment.label==='BULLISH'?'BUY':n.sentiment.label==='BEARISH'?'SELL':'HOLD'), flexShrink:0 }}>
                        {n.sentiment.label}
                      </span>
                    </div>
                    <div style={{ display:'flex', gap:12, alignItems:'center', flexWrap:'wrap' }}>
                      <span style={{ fontSize:9, color:C.textDim }}>{n.source}</span>
                      <span style={{ fontSize:9, color:C.textDim }}>{fmt.ago(n.published)}</span>
                      {n.sentiment.impact==='high' && <span style={{ fontSize:9, color:C.orange, border:`1px solid ${C.orange}`, padding:'1px 6px', borderRadius:2 }}>HIGH IMPACT</span>}
                      <div style={{ display:'flex', gap:4, flexWrap:'wrap' }}>
                        {(n.tickers||[]).slice(0,4).map(t=>(
                          <span key={t} onClick={()=>{ setSelectedAsset(t); setTab('signals') }}
                            style={{ fontSize:9, color:C.accent, border:`1px solid ${C.border}`, padding:'1px 5px', borderRadius:2, cursor:'pointer' }}>
                            {t}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* ══ SIGNALS ══ */}
        {tab==='signals' && (
          <div style={{ display:'flex', flexDirection:'column', gap:12 }}>
            <div style={{ display:'flex', gap:5, flexWrap:'wrap' }}>
              {allTracked.map(a=>(
                <button key={a.ticker} onClick={()=>{ setSelectedAsset(a.ticker); loadTickerNews(a.ticker) }}
                  style={{ background:selectedAsset===a.ticker?C.accentDim:'transparent', border:`1px solid ${selectedAsset===a.ticker?C.accent:C.border}`, color:selectedAsset===a.ticker?C.accent:C.textDim, padding:'4px 10px', borderRadius:4, cursor:'pointer', fontSize:10, fontFamily:'inherit' }}>
                  {a.display}
                </button>
              ))}
            </div>

            {selectedAsset && signals[selectedAsset] ? (() => {
              const sig=signals[selectedAsset], px=prices[selectedAsset], assetBars=bars[selectedAsset]||[]
              const chartData=assetBars.slice(-60).map(b=>({date:fmt.date(b.t),price:b.c}))
              const pos=portfolio.positions[selectedAsset]
              const curPx=px?.price||0
              const unrealized=pos?(pos.side==='LONG'?(curPx-pos.avgPrice)*pos.shares:(pos.avgPrice-curPx)*pos.shares):0
              return (
                <div style={{ display:'flex', flexDirection:'column', gap:12 }}>
                  <div style={{ display:'grid', gridTemplateColumns:'repeat(2,1fr)', gap:8 }}>
                    <div style={{ ...S.panel, border:`1px solid ${sig.signal==='BUY'?C.green:sig.signal==='SELL'?C.red:C.border}` }}>
                      <div style={S.panelTitle}>SIGNAL</div>
                      <div style={{ fontSize:28, fontWeight:700, color:sig.signal==='BUY'?C.green:sig.signal==='SELL'?C.red:C.textDim }}>{sig.signal}</div>
                      <div style={{ fontSize:10, color:C.textDim, marginTop:4 }}>Score: {sig.score.toFixed(4)} · Confidence: {(sig.confidence*100).toFixed(0)}%</div>
                    </div>
                    <div style={S.panel}>
                      <div style={S.panelTitle}>PRICE</div>
                      <div style={{ fontSize:22, fontWeight:700, color:C.textBright }}>{fmt.price(px?.price)}</div>
                      <div style={{ fontSize:10, marginTop:4, ...fmt.chgPct(px?.changePct||0) }}>{px?.changePct?.toFixed(2)}% today</div>
                    </div>
                  </div>

                  {/* Price chart */}
                  <div style={S.panel}>
                    <div style={S.panelTitle}>{selectedAsset} — 60-DAY PRICE</div>
                    <ResponsiveContainer width="100%" height={160}>
                      <AreaChart data={chartData}>
                        <defs><linearGradient id="pg" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor={C.accent} stopOpacity={0.3}/><stop offset="95%" stopColor={C.accent} stopOpacity={0}/></linearGradient></defs>
                        <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
                        <XAxis dataKey="date" tick={{ fontSize:8, fill:C.textDim }} interval={9}/>
                        <YAxis tick={{ fontSize:8, fill:C.textDim }}/>
                        <Tooltip contentStyle={{ background:C.panel, border:`1px solid ${C.border}`, fontSize:10 }}/>
                        <Area type="monotone" dataKey="price" stroke={C.accent} fill="url(#pg)" strokeWidth={2} dot={false}/>
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Factor bars */}
                  <div style={S.panel}>
                    <div style={{ display:'flex', justifyContent:'space-between', marginBottom:8 }}>
                      <div style={S.panelTitle}>FACTOR SCORES</div>
                      <button style={{ background:'transparent', border:'none', color:C.accent, fontSize:9, cursor:'pointer', fontFamily:'inherit' }} onClick={()=>setEduModal('momentum')}>📖 LEARN</button>
                    </div>
                    <ResponsiveContainer width="100%" height={120}>
                      <BarChart data={Object.entries(sig.factors).map(([k,v])=>({factor:k.slice(0,4).toUpperCase(),value:parseFloat(v.toFixed(2))}))}>
                        <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
                        <XAxis dataKey="factor" tick={{ fontSize:9, fill:C.textDim }}/>
                        <YAxis domain={[-3,3]} tick={{ fontSize:8, fill:C.textDim }}/>
                        <ReferenceLine y={0} stroke={C.border}/>
                        <Tooltip contentStyle={{ background:C.panel, border:`1px solid ${C.border}` }}/>
                        <Bar dataKey="value" fill={C.accent} radius={[3,3,0,0]}/>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Quick trade */}
                  <div style={S.panel}>
                    <div style={S.panelTitle}>QUICK PAPER TRADE</div>
                    <div style={{ display:'flex', gap:8, alignItems:'center', flexWrap:'wrap' }}>
                      <div style={{ fontSize:10, color:C.textDim }}>Size: $</div>
                      <input type="number" value={tradeSize} onChange={e=>setTradeSize(parseInt(e.target.value)||1000)}
                        style={{ background:C.surface, border:`1px solid ${C.border}`, color:C.textBright, padding:'6px 10px', borderRadius:4, width:80, fontFamily:'inherit', fontSize:11 }}/>
                      {pos && <div style={{ fontSize:10, color:C.textDim }}>Position: {pos.shares} shares @ {fmt.price(pos.avgPrice)} · P&L: <span style={{ color:unrealized>=0?C.green:C.red }}>${unrealized.toFixed(0)}</span></div>}
                    </div>
                    <div style={{ display:'flex', gap:8, marginTop:10 }}>
                      <button style={{ ...S.btn('green'), flex:1, padding:10 }} onClick={()=>executeTrade(selectedAsset,'BUY',px?.price,sig)}>
                        ▲ BUY {selectedAsset}
                      </button>
                      <button style={{ ...S.btn('danger'), flex:1, padding:10 }} onClick={()=>executeTrade(selectedAsset,'SELL',px?.price,sig)}>
                        ▼ SELL {selectedAsset}
                      </button>
                      {pos && <button style={{ ...S.btn(), padding:10 }} onClick={()=>closePosition(selectedAsset)}>CLOSE</button>}
                    </div>
                  </div>

                  {/* Ticker news */}
                  {tickerNews.length > 0 && (
                    <div style={S.panel}>
                      <div style={S.panelTitle}>RECENT NEWS — {selectedAsset}</div>
                      {tickerNews.map((n,i)=>(
                        <div key={i} style={{ padding:'8px 0', borderBottom:`1px solid ${C.border}40` }}>
                          <div style={{ display:'flex', justifyContent:'space-between', gap:8 }}>
                            <div style={{ fontSize:11, color:C.textBright, flex:1, lineHeight:1.4 }}>{n.title}</div>
                            <span style={S.sigBadge(n.sentiment.label==='BULLISH'?'BUY':n.sentiment.label==='BEARISH'?'SELL':'HOLD')}>{n.sentiment.label}</span>
                          </div>
                          <div style={{ fontSize:9, color:C.textDim, marginTop:4 }}>{n.source} · {fmt.ago(n.published)}</div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )
            })() : (
              <div style={{ ...S.panel, textAlign:'center', padding:40, color:C.textDim }}>Tap a ticker above to view its signal</div>
            )}
          </div>
        )}

        {/* ══ PAPER TRADING ══ */}
        {tab==='paper' && (
          <div style={{ display:'flex', flexDirection:'column', gap:12 }}>

            {/* ── Watchlist & Add Ticker ── */}
            <div style={S.panel}>
              <div style={S.panelTitle}>📌 WATCHLIST — PERSISTENT LIVE P&L</div>
              <div style={{ display:'flex', gap:8, marginBottom:12, flexWrap:'wrap' }}>
                <input
                  value={watchlistInput}
                  onChange={e=>{ setWatchlistInput(e.target.value.toUpperCase()); setAddTickerError('') }}
                  onKeyDown={e=>{ if(e.key==='Enter') addToWatchlist(watchlistInput) }}
                  placeholder="Add ticker (NVDA, BTC, ETH...)"
                  style={{ background:C.surface, border:`1px solid ${addTickerError?C.red:C.border}`, color:C.textBright, padding:'8px 12px', borderRadius:4, fontFamily:'inherit', fontSize:11, flex:1, minWidth:160, outline:'none' }}/>
                <button style={{ ...S.btn('primary'), padding:'8px 16px' }} onClick={()=>addToWatchlist(watchlistInput)}>
                  + ADD
                </button>
              </div>
              {addTickerError && <div style={{ fontSize:10, color:C.red, marginBottom:8 }}>{addTickerError}</div>}

              {/* Watchlist table with live P&L */}
              <div style={{ overflowX:'auto' }}>
                <table style={{ ...S.table, minWidth:480 }}>
                  <thead>
                    <tr>{['TICKER','PRICE','1D%','SIGNAL','POSITION','AVG PRICE','UNREAL P&L',''].map(h=><th key={h} style={S.th}>{h}</th>)}</tr>
                  </thead>
                  <tbody>
                    {watchlist.map(ticker=>{
                      const display = displayTicker(ticker)
                      const px = prices[ticker] || prices[display]
                      const pos = portfolio.positions[ticker]
                      const curPx = px?.price || pos?.avgPrice || 0
                      const unreal = pos ? (pos.side==='LONG' ? (curPx-pos.avgPrice)*pos.shares : (pos.avgPrice-curPx)*pos.shares) : null
                      const sig = signals[ticker] || { signal:'HOLD', score:0 }
                      return (
                        <tr key={ticker} style={{ cursor:'pointer' }} onClick={()=>{ setSelectedAsset(ticker); setTab('signals') }}>
                          <td style={{ ...S.td, fontWeight:700, color:C.textBright }}>{display}</td>
                          <td style={S.td}>{px?.price ? `$${px.price.toFixed(2)}` : <span style={{ color:C.textDim }}>loading...</span>}</td>
                          <td style={{ ...S.td, color:px?.changePct>=0?C.green:C.red }}>{px?.changePct!=null?`${px.changePct.toFixed(2)}%`:'—'}</td>
                          <td style={S.td}><span style={S.sigBadge(sig.signal)}>{sig.signal}</span></td>
                          <td style={S.td}>{pos ? <span style={{ color:pos.side==='LONG'?C.green:C.red }}>{pos.side} {pos.shares}sh</span> : <span style={{ color:C.textDim }}>—</span>}</td>
                          <td style={S.td}>{pos ? `$${pos.avgPrice.toFixed(2)}` : '—'}</td>
                          <td style={{ ...S.td, fontWeight:pos?700:400, color:unreal!=null?(unreal>=0?C.green:C.red):C.textDim }}>
                            {unreal!=null ? `${unreal>=0?'+':''}$${unreal.toFixed(0)}` : '—'}
                          </td>
                          <td style={S.td}>
                            <button style={{ background:'transparent', border:'none', color:C.textDim, cursor:'pointer', fontSize:12, padding:'0 4px' }}
                              onClick={e=>{ e.stopPropagation(); removeFromWatchlist(ticker) }}>✕</button>
                          </td>
                        </tr>
                      )
                    })}
                    {watchlist.length===0&&(
                      <tr><td colSpan={8} style={{ ...S.td, textAlign:'center', color:C.textDim, padding:24 }}>Add tickers above to track them</td></tr>
                    )}
                  </tbody>
                </table>
              </div>
              <div style={{ fontSize:9, color:C.textDim, marginTop:8 }}>
                💾 Watchlist saved to your browser — persists across sessions. Tap any row to trade.
              </div>
            </div>

            {/* Portfolio summary */}
            <div style={{ display:'grid', gridTemplateColumns:'repeat(2,1fr)', gap:8 }}>
              {[
                { label:'PORTFOLIO VALUE', value:`$${curPortfolioValue.toFixed(0)}`, color:C.textBright },
                { label:'TOTAL P&L', value:`${totalPnL>=0?'+':''}$${totalPnL.toFixed(0)}`, color:totalPnL>=0?C.green:C.red },
                { label:'CASH REMAINING', value:`$${portfolio.cash.toFixed(0)}`, color:C.accent },
                { label:'OPEN POSITIONS', value:Object.keys(portfolio.positions).length, color:C.purple },
              ].map(m=>(
                <div key={m.label} style={S.panel}>
                  <div style={S.panelTitle}>{m.label}</div>
                  <div style={{ fontSize:18, fontWeight:700, color:m.color }}>{m.value}</div>
                  {m.label==='PORTFOLIO VALUE'&&<div style={{ fontSize:9, color:C.textDim, marginTop:4 }}>💾 Auto-saved · survives refresh</div>}
                </div>
              ))}
            </div>

            {/* Equity curve */}
            {portfolio.history && portfolio.history.length > 2 && (
              <div style={S.panel}>
                <div style={S.panelTitle}>EQUITY CURVE — ALL TIME</div>
                <ResponsiveContainer width="100%" height={140}>
                  <AreaChart data={portfolio.history.map((h,i)=>({i,value:h.value}))}>
                    <defs><linearGradient id="pg2" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor={totalPnL>=0?C.green:C.red} stopOpacity={0.3}/><stop offset="95%" stopColor={totalPnL>=0?C.green:C.red} stopOpacity={0}/></linearGradient></defs>
                    <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
                    <XAxis dataKey="i" tick={false}/>
                    <YAxis tick={{ fontSize:8, fill:C.textDim }} tickFormatter={v=>`$${(v/1000).toFixed(0)}K`}/>
                    <ReferenceLine y={100000} stroke={C.border} strokeDasharray="4 4"/>
                    <Tooltip contentStyle={{ background:C.panel, border:`1px solid ${C.border}`, fontSize:10 }} formatter={v=>[`$${v.toFixed(0)}`,'Value']}/>
                    <Area type="monotone" dataKey="value" stroke={totalPnL>=0?C.green:C.red} fill="url(#pg2)" strokeWidth={2} dot={false}/>
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Open positions with close button */}
            {Object.keys(portfolio.positions).length > 0 && (
              <div style={S.panel}>
                <div style={S.panelTitle}>OPEN POSITIONS — LIVE P&L</div>
                <div style={{ overflowX:'auto' }}>
                  <table style={{ ...S.table, minWidth:480 }}>
                    <thead><tr>{['TICKER','SIDE','SHARES','AVG IN','CURRENT','UNREAL P&L','%',''].map(h=><th key={h} style={S.th}>{h}</th>)}</tr></thead>
                    <tbody>
                      {Object.entries(portfolio.positions).map(([ticker,pos])=>{
                        const curPx = prices[ticker]?.price || prices[displayTicker(ticker)]?.price || pos.avgPrice
                        const unreal = pos.side==='LONG' ? (curPx-pos.avgPrice)*pos.shares : (pos.avgPrice-curPx)*pos.shares
                        const unrealPct = (unreal / (pos.avgPrice*pos.shares)) * 100
                        return (
                          <tr key={ticker}>
                            <td style={{ ...S.td, fontWeight:700, color:C.textBright }}>{displayTicker(ticker)}</td>
                            <td style={S.td}><span style={S.sigBadge(pos.side==='LONG'?'BUY':'SELL')}>{pos.side}</span></td>
                            <td style={S.td}>{pos.shares}</td>
                            <td style={S.td}>${pos.avgPrice.toFixed(2)}</td>
                            <td style={S.td}>${curPx.toFixed(2)}</td>
                            <td style={{ ...S.td, fontWeight:700, color:unreal>=0?C.green:C.red }}>{unreal>=0?'+':''}${unreal.toFixed(0)}</td>
                            <td style={{ ...S.td, color:unrealPct>=0?C.green:C.red }}>{unrealPct>=0?'+':''}{unrealPct.toFixed(1)}%</td>
                            <td style={S.td}><button style={{ ...S.btn('danger'), padding:'2px 10px', fontSize:9 }} onClick={()=>closePosition(ticker)}>CLOSE</button></td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Trade size + reset */}
            <div style={S.panel}>
              <div style={S.panelTitle}>SETTINGS</div>
              <div style={{ display:'flex', gap:10, alignItems:'center', flexWrap:'wrap' }}>
                <span style={{ fontSize:10, color:C.textDim }}>Trade size: $</span>
                <input type="number" value={tradeSize} onChange={e=>setTradeSize(parseInt(e.target.value)||1000)}
                  style={{ background:C.surface, border:`1px solid ${C.border}`, color:C.textBright, padding:'6px 10px', borderRadius:4, width:90, fontFamily:'inherit', fontSize:11 }}/>
                <span style={{ fontSize:9, color:C.textDim }}>Starting capital: $100,000</span>
                <button style={{ ...S.btn('danger'), marginLeft:'auto' }} onClick={()=>{ if(window.confirm('Reset portfolio? This cannot be undone.')) resetPortfolio() }}>↺ RESET</button>
              </div>
            </div>

            {/* Trade log */}
            <div style={S.panel}>
              <div style={S.panelTitle}>TRADE LOG — {tradeLog.length} TRADES (PERSISTED)</div>
              {tradeLog.length===0 ? (
                <div style={{ textAlign:'center', padding:24, color:C.textDim }}>No trades yet — go to SIGNALS and click BUY or SELL</div>
              ) : (
                <div style={{ overflowX:'auto', maxHeight:300, overflowY:'auto' }}>
                  <table style={{ ...S.table, minWidth:400 }}>
                    <thead><tr>{['TIME','TICKER','SIDE','PRICE','SHARES','VALUE','P&L'].map(h=><th key={h} style={S.th}>{h}</th>)}</tr></thead>
                    <tbody>
                      {tradeLog.map(t=>(
                        <tr key={t.id}>
                          <td style={{ ...S.td, fontSize:9, color:C.textDim }}>{t.time}</td>
                          <td style={{ ...S.td, fontWeight:700, color:C.textBright }}>{displayTicker(t.ticker)}</td>
                          <td style={S.td}><span style={S.sigBadge(t.side==='BUY'?'BUY':t.side==='SELL'?'SELL':'HOLD')}>{t.side}</span></td>
                          <td style={S.td}>${t.price?.toFixed(2)||'—'}</td>
                          <td style={S.td}>{t.shares}</td>
                          <td style={S.td}>${t.value?.toFixed(0)}</td>
                          <td style={{ ...S.td, color:t.pnl>0?C.green:t.pnl<0?C.red:C.textDim }}>{t.pnl?`${t.pnl>=0?'+':''}$${t.pnl}`:'—'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ══ TRAIN ══ */}
        {tab==='train' && (
          <div style={{ display:'flex', flexDirection:'column', gap:12 }}>
            <div style={{ display:'flex', gap:8 }}>
              <button style={{ ...S.btn(isTraining?'default':'primary'), flex:1, padding:12 }} onClick={startTraining} disabled={isTraining}>
                {isTraining?'▶ TRAINING...':'▶ START RL TRAINING'}
              </button>
              {isTraining&&<button style={{ ...S.btn('danger'), padding:12 }} onClick={()=>{ stopTraining(); setIsTraining(false) }}>■ STOP</button>}
              <button style={{ ...S.btn(), padding:12 }} onClick={()=>setEduModal('rl')}>📖 HOW IT WORKS</button>
            </div>

            <div style={{ display:'grid', gridTemplateColumns:'repeat(2,1fr)', gap:8 }}>
              {[
                { label:'GENERATION', value:agent.generation, color:C.accent },
                { label:'BEST RAG', value:agent.bestScore===-Infinity?'—':agent.bestScore.toFixed(3), color:C.purple },
                { label:'EPSILON ε', value:agent.epsilon.toFixed(3), color:C.text },
                { label:'STATUS', value:rlProgress?.improving?'↑ IMPROVING':'↔ EXPLORING', color:rlProgress?.improving?C.green:C.yellow },
              ].map(m=>(
                <div key={m.label} style={S.panel}>
                  <div style={S.panelTitle}>{m.label}</div>
                  <div style={{ fontSize:18, fontWeight:700, color:m.color }}>{m.value}</div>
                </div>
              ))}
            </div>

            <div style={S.panel}>
              <div style={S.panelTitle}>FACTOR WEIGHTS — LIVE OPTIMIZATION</div>
              <div style={{ display:'flex', gap:10, alignItems:'flex-end' }}>
                {Object.entries(weights).map(([k,w])=>(
                  <div key={k} style={{ flex:1, textAlign:'center' }}>
                    <div style={{ fontSize:8, color:C.textDim, marginBottom:6 }}>{k.slice(0,4).toUpperCase()}</div>
                    <div style={{ height:80, background:C.border, borderRadius:3, position:'relative', overflow:'hidden' }}>
                      <div style={{ position:'absolute', bottom:0, width:'100%', height:`${w*100}%`, background:`linear-gradient(0deg,${C.accent},${C.purple})`, transition:'height 0.5s' }}/>
                    </div>
                    <div style={{ fontSize:11, fontWeight:700, color:C.textBright, marginTop:4 }}>{(w*100).toFixed(0)}%</div>
                  </div>
                ))}
              </div>
            </div>

            {trainingLog.length>0&&(
              <div style={S.panel}>
                <div style={S.panelTitle}>SCORE HISTORY</div>
                <ResponsiveContainer width="100%" height={140}>
                  <LineChart data={trainingLog}>
                    <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
                    <XAxis dataKey="episode" tick={{ fontSize:8, fill:C.textDim }}/>
                    <YAxis tick={{ fontSize:8, fill:C.textDim }}/>
                    <ReferenceLine y={0} stroke={C.border}/>
                    <Tooltip contentStyle={{ background:C.panel, border:`1px solid ${C.border}`, fontSize:10 }}/>
                    <Line type="monotone" dataKey="score" stroke={C.purple} strokeWidth={2} dot={false} name="RAG"/>
                    <Line type="monotone" dataKey="sharpe" stroke={C.accent} strokeWidth={1.5} dot={false} name="Sharpe"/>
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}

            {backtestResult&&(
              <div style={S.panel}>
                <div style={S.panelTitle}>LATEST BACKTEST RESULT</div>
                <div style={{ display:'grid', gridTemplateColumns:'repeat(3,1fr)', gap:8 }}>
                  {[
                    { label:'RETURN', value:fmt.pct(backtestResult.totalReturn), color:backtestResult.totalReturn>=0?C.green:C.red },
                    { label:'SHARPE', value:backtestResult.sharpe.toFixed(2), color:C.accent },
                    { label:'MAX DD', value:`-${(backtestResult.maxDrawdown*100).toFixed(1)}%`, color:C.red },
                  ].map(m=>(
                    <div key={m.label} style={{ textAlign:'center' }}>
                      <div style={S.panelTitle}>{m.label}</div>
                      <div style={{ fontSize:16, fontWeight:700, color:m.color }}>{m.value}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}


        {/* ══ MODELS ══ */}
        {tab==='models' && (
          <div style={{ display:'flex', flexDirection:'column', gap:12 }}>

            {/* Model selector */}
            <div style={{ display:'flex', gap:6, flexWrap:'wrap' }}>
              {['ensemble',...MODEL_INFO.map(m=>m.id)].map(id=>(
                <button key={id}
                  style={{ ...S.btn(activeModel===id?'active':'default'), fontSize:10 }}
                  onClick={()=>setActiveModel(id)}>
                  {id==='ensemble'?'⚡ ENSEMBLE':MODEL_INFO.find(m=>m.id===id)?.icon+' '+MODEL_INFO.find(m=>m.id===id)?.name||id}
                </button>
              ))}
            </div>

            {/* Ensemble overview */}
            {activeModel==='ensemble' && (
              <>
                <div style={S.panel}>
                  <div style={S.panelTitle}>⚡ ENSEMBLE MODEL — ALL SIGNALS COMBINED</div>
                  <div style={{ fontSize:11, color:C.textDim, lineHeight:1.6, marginBottom:12 }}>
                    Combines Technical, Fama-French 5F, Temporal CNN, and News Sentiment into one weighted signal.
                    Models with higher recent accuracy get more weight.
                  </div>
                  <div style={{ display:'grid', gridTemplateColumns:'repeat(5,1fr)', gap:5 }}>
                    {Object.entries(ensembleResults).filter(([t])=>activeStocks.includes(t)).slice(0,15).map(([ticker,ens])=>{
                      if(!ens) return null
                      const score=ens.score||0, hue=score>0?'144,255,136':'255,51,102'
                      return (
                        <div key={ticker} onClick={()=>{ setSelectedAsset(ticker); setTab('signals') }}
                          style={{ padding:'8px 6px', borderRadius:4, cursor:'pointer', background:`rgba(${hue},${Math.abs(score)*0.4})`, border:`1px solid rgba(${hue},${Math.abs(score)*0.6})`, textAlign:'center' }}>
                          <div style={{ fontSize:10, fontWeight:700, color:C.textBright }}>{ticker}</div>
                          <div style={{ fontSize:9, color:score>0?C.green:C.red }}>{ens.signal}</div>
                          <div style={{ fontSize:8, color:C.textDim }}>{(ens.confidence*100).toFixed(0)}% conf</div>
                        </div>
                      )
                    })}
                  </div>
                </div>

                {/* Model breakdown table */}
                <div style={S.panel}>
                  <div style={S.panelTitle}>SIGNAL BREAKDOWN BY MODEL</div>
                  <div style={{ overflowX:'auto' }}>
                    <table style={{ ...S.table, minWidth:500 }}>
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
                            <tr key={ticker} style={{ cursor:'pointer' }} onClick={()=>{ setSelectedAsset(ticker); setTab('signals') }}>
                              <td style={{ ...S.td, fontWeight:700, color:C.textBright }}>{ticker}</td>
                              {MODEL_INFO.slice(0,4).map(m=>{
                                const res=ens.models?.[m.id]
                                return <td key={m.id} style={S.td}>{res?<span style={S.sigBadge(res.signal)}>{res.signal}</span>:<span style={{ color:C.textDim, fontSize:9 }}>—</span>}</td>
                              })}
                              <td style={S.td}><span style={{ ...S.sigBadge(ens.signal), fontWeight:700 }}>{ens.signal}</span></td>
                              <td style={{ ...S.td, color:ens.confidence>0.7?C.green:ens.confidence>0.4?C.yellow:C.textDim }}>{(ens.confidence*100).toFixed(0)}%</td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              </>
            )}

            {/* Individual model detail */}
            {activeModel!=='ensemble' && (() => {
              const modelDef = MODEL_INFO.find(m=>m.id===activeModel)
              if(!modelDef) return null
              return (
                <div style={{ display:'flex', flexDirection:'column', gap:12 }}>
                  <div style={{ ...S.panel, border:`1px solid ${C.purple}` }}>
                    <div style={{ fontSize:16, fontWeight:700, color:C.purple, marginBottom:8 }}>{modelDef.icon} {modelDef.name}</div>
                    <div style={{ fontSize:11, color:C.text, lineHeight:1.6, marginBottom:8 }}>{modelDef.desc}</div>
                    <div style={{ fontSize:10, color:C.accent, padding:'6px 10px', background:C.accentDim, borderRadius:4 }}>
                      📚 USED BY: {modelDef.sota}
                    </div>
                  </div>

                  <div style={S.panel}>
                    <div style={S.panelTitle}>RESULTS — ALL SCREENED STOCKS</div>
                    <div style={{ overflowX:'auto' }}>
                      <table style={{ ...S.table, minWidth:400 }}>
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
                              <tr key={ticker} style={{ cursor:'pointer' }} onClick={()=>{ setSelectedAsset(ticker); setTab('signals') }}>
                                <td style={{ ...S.td, fontWeight:700, color:C.textBright }}>{ticker}</td>
                                <td style={{ ...S.td, color:res.score>0?C.green:C.red, fontWeight:700 }}>{res.score?.toFixed(3)}</td>
                                <td style={S.td}><span style={S.sigBadge(res.signal)}>{res.signal}</span></td>
                                {activeModel==='famaFrench'&&<>
                                  <td style={{ ...S.td, color:res.beta>1.2?C.orange:C.text }}>{res.beta}</td>
                                  <td style={{ ...S.td, color:res.alpha>0?C.green:C.red }}>{(res.alpha*100).toFixed(2)}%</td>
                                  <td style={{ ...S.td, color:res.rmw>0?C.green:C.red }}>{res.rmw?.toFixed(2)}</td>
                                  <td style={{ ...S.td, fontSize:9, color:C.textDim }}>{res.interpretation?.value}</td>
                                </>}
                                {activeModel==='tcn'&&<>
                                  <td style={{ ...S.td, color:res.pattern?.includes('UP')?C.green:res.pattern?.includes('DOWN')?C.red:C.textDim, fontSize:9 }}>{res.pattern}</td>
                                  <td style={{ ...S.td, color:res.maAlignment>0?C.green:C.red }}>{res.maAlignment?.toFixed(2)}</td>
                                  <td style={{ ...S.td, color:res.volSurge>1.5?C.orange:C.textDim }}>{res.volSurge?.toFixed(1)}x</td>
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

            {/* SOTA roadmap */}
            <div style={S.panel}>
              <div style={S.panelTitle}>🗺️ FUTURE MODELS ROADMAP — SOTA IN QUANT FINANCE</div>
              <div style={{ display:'flex', flexDirection:'column', gap:8 }}>
                {FUTURE_MODELS.map((m,i)=>(
                  <div key={i} style={{ padding:'10px 12px', background:C.surface, borderRadius:6, border:`1px solid ${C.border}` }}>
                    <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:4 }}>
                      <div style={{ fontSize:11, fontWeight:700, color:C.textBright }}>{m.name}</div>
                      <span style={{ fontSize:9, padding:'2px 6px', borderRadius:2, border:`1px solid ${m.difficulty==='Very High'?C.red:m.difficulty==='High'?C.orange:C.yellow}`, color:m.difficulty==='Very High'?C.red:m.difficulty==='High'?C.orange:C.yellow }}>
                        {m.difficulty}
                      </span>
                    </div>
                    <div style={{ fontSize:10, color:C.textDim, lineHeight:1.5 }}>{m.desc}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

      </main>

      {/* Mobile bottom nav */}
      <nav style={{ position:'fixed', bottom:0, left:0, right:0, background:C.surface, borderTop:`1px solid ${C.border}`, display:'flex', zIndex:200, paddingBottom:'env(safe-area-inset-bottom)' }}>
        {tabs.map(t=>(
          <button key={t.id} onClick={()=>setTab(t.id)}
            style={{ flex:1, background:'transparent', border:'none', cursor:'pointer', padding:'8px 4px 6px', display:'flex', flexDirection:'column', alignItems:'center', gap:2, color:tab===t.id?C.accent:C.textDim, fontFamily:'inherit' }}>
            <span style={{ fontSize:16 }}>{t.icon}</span>
            <span style={{ fontSize:8, letterSpacing:1 }}>{t.label}</span>
            {tab===t.id&&<div style={{ width:20, height:2, background:C.accent, borderRadius:1 }}/>}
          </button>
        ))}
      </nav>
    </div>
  )
}
