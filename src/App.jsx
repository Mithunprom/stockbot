import { useState, useEffect, useCallback, useRef } from 'react'
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine
} from 'recharts'
import { getHistoricalBars, REFRESH_INTERVAL } from './data/polygonClient.js'
import { screenStocks, SCREENING_PROFILES, DEFAULT_CRITERIA } from './data/screener.js'
import { generateSignal } from './signals/signals.js'
import { backtestPortfolio } from './backtest/backtester.js'
import { agent } from './rl/agent.js'
import { trainEpisodes, stopTraining } from './rl/trainer.js'

const ALL_CRYPTO = ['X:BTCUSD', 'X:ETHUSD', 'X:SOLUSD']
const CRYPTO_DISPLAY = { 'X:BTCUSD': 'BTC', 'X:ETHUSD': 'ETH', 'X:SOLUSD': 'SOL' }
const BACKTEST_DAYS = 365
const SCREEN_INTERVAL_SECS = 30 * 60

const C = {
  bg: '#050510', surface: '#0a0a1a', panel: '#0f0f22', border: '#1a1a3a',
  accent: '#00d4ff', accentDim: '#00d4ff33', green: '#00ff88', red: '#ff3366',
  yellow: '#ffd700', purple: '#a855f7', orange: '#ff8c00',
  text: '#e2e8f0', textDim: '#64748b', textBright: '#ffffff',
}

const S = {
  app: { background: C.bg, minHeight: '100vh', fontFamily: "'IBM Plex Mono','Courier New',monospace", color: C.text, display: 'flex', flexDirection: 'column' },
  header: { background: `linear-gradient(90deg,${C.surface},#0a0a2a)`, borderBottom: `1px solid ${C.border}`, padding: '0 16px', height: 52, display: 'flex', alignItems: 'center', justifyContent: 'space-between', position: 'sticky', top: 0, zIndex: 100 },
  logo: { fontSize: 16, fontWeight: 700, letterSpacing: 4, color: C.accent, textShadow: `0 0 20px ${C.accent}` },
  badge: c => ({ fontSize: 9, padding: '2px 6px', borderRadius: 2, border: `1px solid ${c}`, color: c, letterSpacing: 1, whiteSpace: 'nowrap' }),
  main: { flex: 1, padding: '16px', maxWidth: 1600, margin: '0 auto', width: '100%' },
  panel: { background: C.panel, border: `1px solid ${C.border}`, borderRadius: 8, padding: 16 },
  panelTitle: { fontSize: 9, letterSpacing: 3, color: C.textDim, marginBottom: 12, textTransform: 'uppercase' },
  metric: { fontSize: 24, fontWeight: 700, color: C.textBright, letterSpacing: -1 },
  table: { width: '100%', borderCollapse: 'collapse', fontSize: 11 },
  th: { textAlign: 'left', padding: '6px 8px', borderBottom: `1px solid ${C.border}`, color: C.textDim, fontSize: 9, letterSpacing: 1, fontWeight: 400 },
  td: { padding: '8px 8px', borderBottom: `1px solid ${C.border}50` },
  sigBadge: s => ({ display: 'inline-block', padding: '2px 8px', borderRadius: 3, fontSize: 9, fontWeight: 700, letterSpacing: 1, background: s==='BUY'?'#00ff8822':s==='SELL'?'#ff336622':'#ffffff11', color: s==='BUY'?C.green:s==='SELL'?C.red:C.textDim, border: `1px solid ${s==='BUY'?C.green:s==='SELL'?C.red:C.border}` }),
  btn: (v='primary') => ({ background: v==='primary'?C.accentDim:v==='danger'?'#ff336622':v==='active'?'#a855f722':'transparent', border: `1px solid ${v==='primary'?C.accent:v==='danger'?C.red:v==='active'?C.purple:C.border}`, color: v==='primary'?C.accent:v==='danger'?C.red:v==='active'?C.purple:C.textDim, padding: '7px 14px', borderRadius: 4, cursor: 'pointer', fontSize: 10, letterSpacing: 1, fontFamily: 'inherit', transition: 'all 0.2s' }),
}

const fmt = {
  pct: v => `${v>=0?'+':''}${(v*100).toFixed(2)}%`,
  price: v => !v?'—':v>=1000?`$${(v/1000).toFixed(2)}K`:`$${v.toFixed(2)}`,
  chg: v => ({ color: v>=0?C.green:C.red }),
  date: ts => new Date(ts).toLocaleDateString(),
}

function generateMockBars(ticker, days=400) {
  const seed = ticker.split('').reduce((a,c)=>a+c.charCodeAt(0),0)
  const bars=[]; let price=50+(seed%200); const now=Date.now()
  for(let i=days;i>=0;i--){
    const t=now-i*86400000, change=(Math.random()-0.48)*price*0.03, o=price
    price=Math.max(1,price+change)
    bars.push({t,o,h:Math.max(o,price)*(1+Math.random()*0.01),l:Math.min(o,price)*(1-Math.random()*0.01),c:price,v:1e6+Math.random()*5e6,vw:(o+price)/2})
  }
  return bars
}

function getApiKey() {
  try {
    const k = import.meta.env.VITE_POLYGON_API_KEY
    if (k && k.length > 10 && k !== 'your_polygon_api_key_here') return k
  } catch(e) {}
  return null
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
  const [dataStatus, setDataStatus] = useState({ live: false, usingMock: true, keyFound: false, lastUpdate: null })
  const [selectedAsset, setSelectedAsset] = useState(null)
  const [rlProgress, setRlProgress] = useState(null)
  const [weights, setWeights] = useState(agent.weights)
  const [paperTrades, setPaperTrades] = useState([])
  const [customCriteria, setCustomCriteria] = useState(DEFAULT_CRITERIA)
  const [nextScreenIn, setNextScreenIn] = useState(SCREEN_INTERVAL_SECS)
  const [autoScreenEnabled, setAutoScreenEnabled] = useState(true)
  const [hasAutoRun, setHasAutoRun] = useState(false)

  const screenTimerRef = useRef(null)
  const countdownRef = useRef(null)
  const isScreeningRef = useRef(false)
  const apiKey = getApiKey()

  // ── Screener ──────────────────────────────────────────────────────────────
  const runScreener = useCallback(async (profile, custom) => {
    if (isScreeningRef.current) return
    isScreeningRef.current = true
    setIsScreening(true)
    setScreenResult(null)

    const activeProfile = profile || screenProfile
    const criteria = custom || SCREENING_PROFILES[activeProfile]?.criteria || DEFAULT_CRITERIA

    if (!apiKey) {
      // Mock mode — generate ranked stocks immediately
      const mockTickers = ['NVDA','TSLA','AMD','MSTR','PLTR','COIN','META','AAPL','AMZN','GOOGL','MSFT','ARM','SMCI','AVGO','MARA','IONQ','RKLB','HOOD','ACHR','RIOT']
      const mockStocks = mockTickers.map(ticker => {
        const b = generateMockBars(ticker)
        const last = b[b.length-1], prev2 = b[b.length-2], prev5 = b[b.length-5]
        return {
          ticker, price: last.c,
          change1d: (last.c-prev2.c)/prev2.c*100,
          change5d: (last.c-prev5.c)/prev5.c*100,
          volume: last.v,
          composite: (Math.random()*2-0.5),
          scores: { momentum: Math.random()-0.3, volatility: Math.random()*0.05, volumeSurge: Math.random()*2-0.5, trendBreak: Math.random()-0.3, rsiOversold: Math.random()-0.3 },
          bars: b,
        }
      }).sort((a,b)=>b.composite-a.composite)

      const tickers = mockStocks.slice(0,15).map(s=>s.ticker)
      const barsMap = {}
      mockStocks.slice(0,15).forEach(s => { barsMap[s.ticker] = s.bars })
      ALL_CRYPTO.forEach(t => { barsMap[t] = generateMockBars(CRYPTO_DISPLAY[t]) })

      const priceMap = {}
      mockStocks.slice(0,15).forEach(s => {
        priceMap[s.ticker] = { price: s.price, changePct: s.change1d, change: s.price*s.change1d/100, volume: s.volume }
      })

      setScreenResult({ stocks: mockStocks.slice(0,15), errors:[], criteria, timestamp: new Date(), mock: true })
      setActiveStocks(tickers)
      setBars(barsMap)
      setPrices(priceMap)
      recomputeSignals(barsMap)
      setDataStatus({ live: false, usingMock: true, keyFound: false, lastUpdate: new Date(), screened: tickers.length })
    } else {
      try {
        const result = await screenStocks(criteria, 20)
        const tickers = result.stocks.map(s=>s.ticker)
        const barsMap = {}
        result.stocks.forEach(s => {
          barsMap[s.ticker] = (s.bars||[]).map(b=>({t:b.t,o:b.o,h:b.h,l:b.l,c:b.c,v:b.v,vw:b.vw||b.c}))
        })
        ALL_CRYPTO.forEach(t => { barsMap[t] = generateMockBars(CRYPTO_DISPLAY[t]) })

        const priceMap = {}
        result.stocks.forEach(s => {
          priceMap[s.ticker] = { price: s.price, changePct: s.change1d, change: s.price*s.change1d/100, volume: s.volume }
        })

        setScreenResult({ ...result, mock: false })
        setActiveStocks(tickers)
        setBars(barsMap)
        setPrices(priceMap)
        recomputeSignals(barsMap)
        setDataStatus({ live: true, usingMock: false, keyFound: true, lastUpdate: new Date(), screened: tickers.length })
      } catch(err) {
        console.error('[Screener]', err)
        setDataStatus(prev => ({ ...prev, error: err.message, keyFound: true }))
      }
    }

    isScreeningRef.current = false
    setIsScreening(false)
  }, [apiKey, screenProfile])

  function recomputeSignals(barsMap) {
    const w = agent.weights
    const sigs = {}
    for (const [t,b] of Object.entries(barsMap)) {
      if (b && b.length > 0) sigs[t] = generateSignal(b, w)
    }
    setSignals(sigs)
    setWeights(w)
  }

  // ── Scheduler ─────────────────────────────────────────────────────────────
  useEffect(() => {
    // Run screener immediately on first load
    if (!hasAutoRun) {
      setHasAutoRun(true)
      runScreener()
    }
  }, [runScreener, hasAutoRun])

  useEffect(() => {
    if (!autoScreenEnabled) {
      clearInterval(screenTimerRef.current)
      clearInterval(countdownRef.current)
      return
    }

    setNextScreenIn(SCREEN_INTERVAL_SECS)

    // Countdown tick
    countdownRef.current = setInterval(() => {
      setNextScreenIn(prev => (prev <= 1 ? SCREEN_INTERVAL_SECS : prev - 1))
    }, 1000)

    // Auto-screen every 30 min
    screenTimerRef.current = setInterval(() => {
      runScreener()
    }, SCREEN_INTERVAL_SECS * 1000)

    return () => {
      clearInterval(screenTimerRef.current)
      clearInterval(countdownRef.current)
    }
  }, [autoScreenEnabled, runScreener])

  // ── RL Training ───────────────────────────────────────────────────────────
  async function startTraining() {
    if (isTraining || Object.keys(bars).length === 0) return
    setIsTraining(true)
    await trainEpisodes(bars, 30,
      ep => {
        setTrainingLog(prev => [...prev.slice(-50), { episode: ep.episode, score: ep.result.ragScore.toFixed(3), sharpe: ep.result.sharpe.toFixed(2), ret: (ep.result.totalReturn*100).toFixed(1)+'%', regime: ep.regime }])
        setRlProgress(ep.agentState)
        setWeights({...ep.currentWeights})
      },
      done => { setWeights({...done.bestWeights}); recomputeSignals(bars); runBacktest(done.bestWeights); setIsTraining(false) }
    )
  }

  function runBacktest(w=weights) { setBacktestResult(backtestPortfolio(bars, w)) }

  // ── Computed ──────────────────────────────────────────────────────────────
  const allTracked = [
    ...activeStocks.map(t=>({ticker:t,display:t,type:'stock'})),
    ...ALL_CRYPTO.map(t=>({ticker:t,display:CRYPTO_DISPLAY[t],type:'crypto'})),
  ]
  const sortedBySignal = allTracked.map(a=>({...a,sig:signals[a.ticker]||{score:0,signal:'HOLD',factors:{},confidence:0},px:prices[a.ticker]})).sort((a,b)=>(b.sig?.score||0)-(a.sig?.score||0))
  const buys = sortedBySignal.filter(a=>a.sig?.signal==='BUY')
  const sells = sortedBySignal.filter(a=>a.sig?.signal==='SELL')

  const fmtCountdown = s => `${Math.floor(s/60)}:${String(s%60).padStart(2,'0')}`

  const statusBadge = () => {
    if (isScreening) return <span style={S.badge(C.yellow)}>⟳ SCANNING...</span>
    if (!dataStatus.keyFound) return <span style={S.badge(C.yellow)}>MOCK — NO KEY</span>
    if (dataStatus.error) return <span style={S.badge(C.red)}>API ERROR</span>
    if (dataStatus.live) return <span style={S.badge(C.green)}>LIVE · {dataStatus.screened} STOCKS</span>
    return <span style={S.badge(C.yellow)}>LOADING...</span>
  }

  // ── Mobile bottom nav ─────────────────────────────────────────────────────
  const tabs = ['screen','signals','backtest','train','paper']

  return (
    <div style={S.app}>
      {/* Header */}
      <header style={S.header}>
        <div style={{ display:'flex', alignItems:'center', gap:10 }}>
          <span style={S.logo}>STOCKBOT</span>
          {statusBadge()}
        </div>
        {/* Desktop nav */}
        <nav style={{ display:'flex', gap:2, '@media(maxWidth:640px)':{display:'none'} }}>
          {tabs.map(t => (
            <button key={t} style={{ background:tab===t?C.accentDim:'transparent', border:`1px solid ${tab===t?C.accent:'transparent'}`, color:tab===t?C.accent:C.textDim, padding:'5px 12px', borderRadius:4, cursor:'pointer', fontSize:10, letterSpacing:2, fontFamily:'inherit' }}
              onClick={()=>setTab(t)}>{t.toUpperCase()}</button>
          ))}
        </nav>
        <div style={{ fontSize:9, color:C.textDim, textAlign:'right' }}>
          {autoScreenEnabled && <div style={{ color:C.accent }}>⟳ {fmtCountdown(nextScreenIn)}</div>}
          <div>GEN {agent.generation}</div>
        </div>
      </header>

      <main style={S.main}>

        {/* ══ SCREEN ══ */}
        {tab === 'screen' && (
          <div style={{ display:'flex', flexDirection:'column', gap:12 }}>

            {/* Scheduler bar */}
            <div style={{ background:C.surface, border:`1px solid ${C.border}`, borderRadius:8, padding:'10px 14px', display:'flex', alignItems:'center', justifyContent:'space-between', flexWrap:'wrap', gap:8 }}>
              <div style={{ display:'flex', alignItems:'center', gap:10 }}>
                <div style={{ width:8, height:8, borderRadius:'50%', background:autoScreenEnabled?C.green:C.textDim, boxShadow:autoScreenEnabled?`0 0 8px ${C.green}`:'none' }}/>
                <span style={{ fontSize:10, color:C.textBright, letterSpacing:2 }}>{autoScreenEnabled?'AUTO-SCREEN ON':'PAUSED'}</span>
                {autoScreenEnabled && <span style={{ fontSize:10, color:C.textDim }}>next in <span style={{ color:C.accent, fontWeight:700 }}>{fmtCountdown(nextScreenIn)}</span></span>}
              </div>
              <div style={{ display:'flex', gap:6 }}>
                <button style={{ ...S.btn(autoScreenEnabled?'danger':'primary'), padding:'4px 12px' }} onClick={()=>setAutoScreenEnabled(p=>!p)}>
                  {autoScreenEnabled?'⏸ PAUSE':'▶ RESUME'}
                </button>
                <button style={{ ...S.btn('primary'), padding:'4px 12px' }} onClick={()=>runScreener(screenProfile, customCriteria)} disabled={isScreening}>
                  {isScreening ? '⟳ SCANNING...' : '⟳ SCAN NOW'}
                </button>
              </div>
            </div>

            {/* Profile buttons */}
            <div style={{ display:'flex', gap:6, flexWrap:'wrap' }}>
              {Object.entries(SCREENING_PROFILES).map(([key,p])=>(
                <button key={key} style={{ ...S.btn(screenProfile===key?'active':'default'), fontSize:10 }}
                  onClick={()=>{ setScreenProfile(key); setCustomCriteria(p.criteria); runScreener(key, p.criteria) }}>
                  {p.icon} {p.label}
                </button>
              ))}
            </div>

            {/* Loading state */}
            {isScreening && (
              <div style={{ ...S.panel, textAlign:'center', padding:48 }}>
                <div style={{ fontSize:28, marginBottom:8 }}>⟳</div>
                <div style={{ color:C.accent, letterSpacing:4, fontSize:11 }}>SCANNING MARKET...</div>
                <div style={{ color:C.textDim, fontSize:10, marginTop:8 }}>Fetching bars · scoring momentum, volume, trend, RSI</div>
              </div>
            )}

            {/* Results */}
            {!isScreening && screenResult && (
              <>
                {/* Stats row */}
                <div style={{ display:'grid', gridTemplateColumns:'repeat(4,1fr)', gap:10 }}>
                  {[
                    { label:'FOUND', value:screenResult.stocks.length, color:C.accent },
                    { label:'BUY', value:buys.length, color:C.green },
                    { label:'SELL', value:sells.length, color:C.red },
                    { label:'UPDATED', value:screenResult.timestamp?.toLocaleTimeString(), color:C.textDim },
                  ].map(m=>(
                    <div key={m.label} style={S.panel}>
                      <div style={S.panelTitle}>{m.label}</div>
                      <div style={{ fontSize:20, fontWeight:700, color:m.color }}>{m.value}</div>
                    </div>
                  ))}
                </div>

                {/* Heatmap */}
                <div style={S.panel}>
                  <div style={S.panelTitle}>SIGNAL HEATMAP {screenResult.mock && <span style={{ color:C.yellow }}>(MOCK)</span>}</div>
                  <div style={{ display:'grid', gridTemplateColumns:'repeat(5,1fr)', gap:5 }}>
                    {sortedBySignal.slice(0, 20).map(a=>{
                      const score = a.sig?.score||0
                      const hue = score>0?'144,255,136':'255,51,102'
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

                {/* Table — scrollable on mobile */}
                <div style={S.panel}>
                  <div style={S.panelTitle}>RANKED STOCKS</div>
                  <div style={{ overflowX:'auto', WebkitOverflowScrolling:'touch' }}>
                    <table style={{ ...S.table, minWidth:600 }}>
                      <thead>
                        <tr>{['#','TICKER','PRICE','1D%','5D%','VOL','MOMENTUM','VOL SURGE','TREND','SCORE','SIGNAL'].map(h=><th key={h} style={S.th}>{h}</th>)}</tr>
                      </thead>
                      <tbody>
                        {screenResult.stocks.map((s,i)=>{
                          const sig = signals[s.ticker]||{signal:'HOLD',score:0}
                          return (
                            <tr key={s.ticker} style={{ cursor:'pointer' }} onClick={()=>{ setSelectedAsset(s.ticker); setTab('signals') }}>
                              <td style={{ ...S.td, color:C.textDim }}>{i+1}</td>
                              <td style={{ ...S.td, color:C.textBright, fontWeight:700 }}>{s.ticker}</td>
                              <td style={S.td}>{fmt.price(s.price)}</td>
                              <td style={{ ...S.td, ...fmt.chg(s.change1d) }}>{s.change1d?.toFixed(2)}%</td>
                              <td style={{ ...S.td, ...fmt.chg(s.change5d) }}>{s.change5d?.toFixed(2)}%</td>
                              <td style={{ ...S.td, color:C.textDim }}>{(s.volume/1e6).toFixed(1)}M</td>
                              <td style={{ ...S.td, color:s.scores.momentum>0?C.green:C.red }}>{(s.scores.momentum*100).toFixed(1)}%</td>
                              <td style={{ ...S.td, color:s.scores.volumeSurge>0.5?C.orange:C.textDim }}>{s.scores.volumeSurge>0?'+':''}{(s.scores.volumeSurge*100).toFixed(0)}%</td>
                              <td style={{ ...S.td, color:s.scores.trendBreak>0?C.green:C.red }}>{s.scores.trendBreak>0?'↑':'↓'}{Math.abs(s.scores.trendBreak*100).toFixed(1)}%</td>
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

            {/* Empty state */}
            {!isScreening && !screenResult && (
              <div style={{ ...S.panel, textAlign:'center', padding:60, color:C.textDim }}>
                <div style={{ fontSize:32, marginBottom:12 }}>⟳</div>
                <div>Loading market data...</div>
              </div>
            )}
          </div>
        )}

        {/* ══ SIGNALS ══ */}
        {tab === 'signals' && (
          <div style={{ display:'flex', flexDirection:'column', gap:12 }}>
            <div style={{ display:'flex', gap:5, flexWrap:'wrap' }}>
              {allTracked.map(a=>(
                <button key={a.ticker} onClick={()=>setSelectedAsset(a.ticker)}
                  style={{ background:selectedAsset===a.ticker?C.accentDim:'transparent', border:`1px solid ${selectedAsset===a.ticker?C.accent:C.border}`, color:selectedAsset===a.ticker?C.accent:C.textDim, padding:'4px 10px', borderRadius:4, cursor:'pointer', fontSize:10, fontFamily:'inherit' }}>
                  {a.display}
                </button>
              ))}
            </div>

            {selectedAsset && signals[selectedAsset] && (() => {
              const sig = signals[selectedAsset]
              const assetBars = bars[selectedAsset]||[]
              const px = prices[selectedAsset]
              const chartData = assetBars.slice(-60).map(b=>({date:fmt.date(b.t),price:b.c}))
              return (
                <div style={{ display:'flex', flexDirection:'column', gap:12 }}>
                  <div style={{ display:'grid', gridTemplateColumns:'repeat(2,1fr)', gap:10 }}>
                    <div style={S.panel}>
                      <div style={S.panelTitle}>SIGNAL</div>
                      <div style={{ fontSize:28, fontWeight:700, color:sig.signal==='BUY'?C.green:sig.signal==='SELL'?C.red:C.textDim }}>{sig.signal}</div>
                      <div style={{ fontSize:10, color:C.textDim, marginTop:4 }}>Score: {sig.score.toFixed(4)}</div>
                    </div>
                    <div style={S.panel}>
                      <div style={S.panelTitle}>PRICE</div>
                      <div style={{ fontSize:22, fontWeight:700, color:C.textBright }}>{fmt.price(px?.price)}</div>
                      <div style={{ fontSize:10, marginTop:4, ...fmt.chg(px?.changePct||0) }}>{px?.changePct?.toFixed(2)}% today</div>
                    </div>
                  </div>
                  <div style={S.panel}>
                    <div style={S.panelTitle}>{selectedAsset} — 60-DAY PRICE</div>
                    <ResponsiveContainer width="100%" height={180}>
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
                  <div style={S.panel}>
                    <div style={S.panelTitle}>FACTOR SCORES</div>
                    <ResponsiveContainer width="100%" height={140}>
                      <BarChart data={Object.entries(sig.factors).map(([k,v])=>({factor:k.slice(0,4),value:v}))}>
                        <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
                        <XAxis dataKey="factor" tick={{ fontSize:9, fill:C.textDim }}/>
                        <YAxis domain={[-3,3]} tick={{ fontSize:8, fill:C.textDim }}/>
                        <ReferenceLine y={0} stroke={C.border}/>
                        <Tooltip contentStyle={{ background:C.panel, border:`1px solid ${C.border}` }}/>
                        <Bar dataKey="value" fill={C.accent} radius={[3,3,0,0]}/>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <div style={{ padding:'0 4px' }}>
                    <button style={{ ...S.btn('primary'), width:'100%', padding:12, fontSize:12 }}
                      onClick={()=>setPaperTrades(p=>[{ticker:selectedAsset,side:sig.signal,price:px?.price,size:1000,score:sig.score.toFixed(3),time:new Date().toLocaleTimeString()},...p.slice(0,49)])}>
                      PAPER TRADE {selectedAsset}
                    </button>
                  </div>
                </div>
              )
            })()}

            {!selectedAsset && (
              <div style={{ ...S.panel, textAlign:'center', padding:40, color:C.textDim }}>
                Tap a ticker above to see its signal
              </div>
            )}
          </div>
        )}

        {/* ══ BACKTEST ══ */}
        {tab === 'backtest' && (
          <div style={{ display:'flex', flexDirection:'column', gap:12 }}>
            <button style={{ ...S.btn('primary'), width:'100%', padding:12 }} onClick={()=>runBacktest()}>
              ▶ RUN BACKTEST ON SCREENED STOCKS
            </button>
            {backtestResult && (() => {
              const r = backtestResult
              const equityData = r.equity.map((v,i)=>({i,value:v}))
              const ddData = []; let peak=-Infinity
              for (const {i,value} of equityData) { if(value>peak) peak=value; ddData.push({i,dd:((peak-value)/peak)*-100}) }
              return (
                <div style={{ display:'flex', flexDirection:'column', gap:12 }}>
                  <div style={{ display:'grid', gridTemplateColumns:'repeat(2,1fr)', gap:10 }}>
                    {[
                      { label:'TOTAL RETURN', value:fmt.pct(r.totalReturn), color:r.totalReturn>=0?C.green:C.red },
                      { label:'SHARPE', value:r.sharpe.toFixed(2), color:r.sharpe>=1?C.green:C.yellow },
                      { label:'MAX DRAWDOWN', value:`-${(r.maxDrawdown*100).toFixed(1)}%`, color:r.maxDrawdown<0.2?C.yellow:C.red },
                      { label:'RAG SCORE', value:r.ragScore.toFixed(3), color:C.purple },
                      { label:'WIN RATE', value:`${(r.winRate*100).toFixed(0)}%`, color:r.winRate>=0.5?C.green:C.yellow },
                      { label:'FINAL VALUE', value:`$${r.finalValue.toFixed(0)}`, color:C.textBright },
                    ].map(m=>(
                      <div key={m.label} style={S.panel}>
                        <div style={S.panelTitle}>{m.label}</div>
                        <div style={{ fontSize:18, fontWeight:700, color:m.color }}>{m.value}</div>
                      </div>
                    ))}
                  </div>
                  <div style={S.panel}>
                    <div style={S.panelTitle}>EQUITY CURVE</div>
                    <ResponsiveContainer width="100%" height={200}>
                      <AreaChart data={equityData}>
                        <defs><linearGradient id="eg" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor={C.green} stopOpacity={0.3}/><stop offset="95%" stopColor={C.green} stopOpacity={0}/></linearGradient></defs>
                        <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
                        <XAxis dataKey="i" tick={false}/>
                        <YAxis tick={{ fontSize:8, fill:C.textDim }} tickFormatter={v=>`$${(v/1000).toFixed(0)}K`}/>
                        <Tooltip contentStyle={{ background:C.panel, border:`1px solid ${C.border}`, fontSize:10 }} formatter={v=>[`$${v.toFixed(0)}`,'Value']}/>
                        <ReferenceLine y={100000} stroke={C.border} strokeDasharray="4 4"/>
                        <Area type="monotone" dataKey="value" stroke={C.green} fill="url(#eg)" strokeWidth={2} dot={false}/>
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )
            })()}
            {!backtestResult && <div style={{ ...S.panel, textAlign:'center', padding:48, color:C.textDim }}>Run screener first, then backtest</div>}
          </div>
        )}

        {/* ══ TRAIN ══ */}
        {tab === 'train' && (
          <div style={{ display:'flex', flexDirection:'column', gap:12 }}>
            <div style={{ display:'flex', gap:8 }}>
              <button style={{ ...S.btn(isTraining?'default':'primary'), flex:1, padding:12 }} onClick={startTraining} disabled={isTraining}>
                {isTraining?'▶ TRAINING...':'▶ START RL TRAINING'}
              </button>
              {isTraining && <button style={{ ...S.btn('danger'), padding:12 }} onClick={()=>{ stopTraining(); setIsTraining(false) }}>■ STOP</button>}
            </div>
            <div style={{ display:'grid', gridTemplateColumns:'repeat(2,1fr)', gap:10 }}>
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
              <div style={S.panelTitle}>FACTOR WEIGHTS</div>
              <div style={{ display:'flex', gap:12, alignItems:'flex-end' }}>
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
            {trainingLog.length > 0 && (
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
          </div>
        )}

        {/* ══ PAPER ══ */}
        {tab === 'paper' && (
          <div style={{ display:'flex', flexDirection:'column', gap:12 }}>
            <div style={{ ...S.panel, textAlign:'center', padding: paperTrades.length?16:48, color: paperTrades.length?C.text:C.textDim }}>
              {paperTrades.length === 0 ? (
                <>
                  <div style={{ fontSize:28, marginBottom:8 }}>📋</div>
                  <div>Go to SIGNALS → tap a ticker → PAPER TRADE</div>
                </>
              ) : (
                <>
                  <div style={S.panelTitle}>PAPER TRADE LOG</div>
                  <div style={{ overflowX:'auto' }}>
                    <table style={{ ...S.table, minWidth:400 }}>
                      <thead><tr>{['TIME','TICKER','SIDE','PRICE','SCORE'].map(h=><th key={h} style={S.th}>{h}</th>)}</tr></thead>
                      <tbody>
                        {paperTrades.map((t,i)=>(
                          <tr key={i}>
                            <td style={{ ...S.td, fontSize:9, color:C.textDim }}>{t.time}</td>
                            <td style={{ ...S.td, fontWeight:700, color:C.textBright }}>{t.ticker}</td>
                            <td style={S.td}><span style={S.sigBadge(t.side)}>{t.side}</span></td>
                            <td style={S.td}>{fmt.price(t.price)}</td>
                            <td style={{ ...S.td, color:parseFloat(t.score)>0?C.green:C.red }}>{t.score}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              )}
            </div>
          </div>
        )}

      </main>

      {/* ── Mobile bottom navigation ── */}
      <nav style={{ position:'fixed', bottom:0, left:0, right:0, background:C.surface, borderTop:`1px solid ${C.border}`, display:'flex', zIndex:200, paddingBottom:'env(safe-area-inset-bottom)' }}>
        {[
          { id:'screen', icon:'⟳', label:'SCREEN' },
          { id:'signals', icon:'📡', label:'SIGNALS' },
          { id:'backtest', icon:'📈', label:'BACKTEST' },
          { id:'train', icon:'🤖', label:'TRAIN' },
          { id:'paper', icon:'📋', label:'PAPER' },
        ].map(t=>(
          <button key={t.id} onClick={()=>setTab(t.id)}
            style={{ flex:1, background:'transparent', border:'none', cursor:'pointer', padding:'8px 4px', display:'flex', flexDirection:'column', alignItems:'center', gap:2, color:tab===t.id?C.accent:C.textDim, fontFamily:'inherit' }}>
            <span style={{ fontSize:16 }}>{t.icon}</span>
            <span style={{ fontSize:8, letterSpacing:1 }}>{t.label}</span>
            {tab===t.id && <div style={{ width:20, height:2, background:C.accent, borderRadius:1 }}/>}
          </button>
        ))}
      </nav>

      {/* Bottom padding for mobile nav */}
      <div style={{ height:60 }}/>
    </div>
  )
}
