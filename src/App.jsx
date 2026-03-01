import { useState, useEffect, useCallback, useRef } from 'react'
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine
} from 'recharts'
import { getSnapshots, getCryptoSnapshots, getHistoricalBars, REFRESH_INTERVAL } from './data/polygonClient.js'
import { screenStocks, SCREENING_PROFILES, DEFAULT_CRITERIA } from './data/screener.js'
import { generateSignal } from './signals/signals.js'
import { backtestPortfolio } from './backtest/backtester.js'
import { agent } from './rl/agent.js'
import { trainEpisodes, stopTraining } from './rl/trainer.js'

const ALL_CRYPTO = ['X:BTCUSD', 'X:ETHUSD', 'X:SOLUSD']
const CRYPTO_DISPLAY = { 'X:BTCUSD': 'BTC', 'X:ETHUSD': 'ETH', 'X:SOLUSD': 'SOL' }
const BACKTEST_DAYS = 365

// ── Styles ─────────────────────────────────────────────────────────────────
const C = {
  bg: '#050510', surface: '#0a0a1a', panel: '#0f0f22', border: '#1a1a3a',
  accent: '#00d4ff', accentDim: '#00d4ff33', green: '#00ff88', red: '#ff3366',
  yellow: '#ffd700', purple: '#a855f7', orange: '#ff8c00',
  text: '#e2e8f0', textDim: '#64748b', textBright: '#ffffff',
}

const S = {
  app: { background: C.bg, minHeight: '100vh', fontFamily: "'IBM Plex Mono','Courier New',monospace", color: C.text, display: 'flex', flexDirection: 'column' },
  header: { background: `linear-gradient(90deg,${C.surface},#0a0a2a)`, borderBottom: `1px solid ${C.border}`, padding: '0 24px', height: 56, display: 'flex', alignItems: 'center', justifyContent: 'space-between', position: 'sticky', top: 0, zIndex: 100 },
  logo: { fontSize: 18, fontWeight: 700, letterSpacing: 4, color: C.accent, textShadow: `0 0 20px ${C.accent}` },
  badge: c => ({ fontSize: 10, padding: '2px 8px', borderRadius: 2, border: `1px solid ${c}`, color: c, letterSpacing: 2 }),
  nav: { display: 'flex', gap: 4 },
  navBtn: a => ({ background: a ? C.accentDim : 'transparent', border: `1px solid ${a ? C.accent : 'transparent'}`, color: a ? C.accent : C.textDim, padding: '6px 16px', borderRadius: 4, cursor: 'pointer', fontSize: 11, letterSpacing: 2, fontFamily: 'inherit', transition: 'all 0.2s' }),
  main: { flex: 1, padding: 24, maxWidth: 1600, margin: '0 auto', width: '100%' },
  panel: { background: C.panel, border: `1px solid ${C.border}`, borderRadius: 8, padding: 20 },
  panelTitle: { fontSize: 10, letterSpacing: 4, color: C.textDim, marginBottom: 16, textTransform: 'uppercase' },
  metric: { fontSize: 28, fontWeight: 700, color: C.textBright, letterSpacing: -1 },
  metricSub: { fontSize: 11, color: C.textDim, marginTop: 4, letterSpacing: 1 },
  table: { width: '100%', borderCollapse: 'collapse', fontSize: 12 },
  th: { textAlign: 'left', padding: '8px 12px', borderBottom: `1px solid ${C.border}`, color: C.textDim, fontSize: 10, letterSpacing: 2, fontWeight: 400 },
  td: { padding: '10px 12px', borderBottom: `1px solid ${C.border}80` },
  sigBadge: s => ({ display: 'inline-block', padding: '2px 10px', borderRadius: 3, fontSize: 10, fontWeight: 700, letterSpacing: 2, background: s==='BUY'?'#00ff8822':s==='SELL'?'#ff336622':'#ffffff11', color: s==='BUY'?C.green:s==='SELL'?C.red:C.textDim, border: `1px solid ${s==='BUY'?C.green:s==='SELL'?C.red:C.border}` }),
  btn: (v='primary') => ({ background: v==='primary'?C.accentDim:v==='danger'?'#ff336622':v==='active'?'#a855f722':'transparent', border: `1px solid ${v==='primary'?C.accent:v==='danger'?C.red:v==='active'?C.purple:C.border}`, color: v==='primary'?C.accent:v==='danger'?C.red:v==='active'?C.purple:C.textDim, padding: '8px 20px', borderRadius: 4, cursor: 'pointer', fontSize: 11, letterSpacing: 2, fontFamily: 'inherit', transition: 'all 0.2s' }),
  grid2: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 },
  grid4: { display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 16 },
}

const fmt = {
  pct: v => `${v>=0?'+':''}${(v*100).toFixed(2)}%`,
  price: v => !v ? '—' : v>=1000?`$${(v/1000).toFixed(2)}K`:`$${v.toFixed(2)}`,
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

// ── App ────────────────────────────────────────────────────────────────────
export default function App() {
  const [tab, setTab] = useState('screen')
  const [screenProfile, setScreenProfile] = useState('momentum')
  const [screenResult, setScreenResult] = useState(null)
  const [isScreening, setIsScreening] = useState(false)
  const [activeStocks, setActiveStocks] = useState([]) // currently tracked tickers from screener
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
  const [nextScreenIn, setNextScreenIn] = useState(null)
  const [autoScreenEnabled, setAutoScreenEnabled] = useState(true)
  const screenTimerRef = useRef(null)
  const countdownRef = useRef(null)

  const apiKey = getApiKey()

  // ── Screen stocks ────────────────────────────────────────────────────────
  const runScreener = useCallback(async (profile = screenProfile, custom = null) => {
    setIsScreening(true)
    setScreenResult(null)

    const criteria = custom || SCREENING_PROFILES[profile]?.criteria || DEFAULT_CRITERIA

    if (!apiKey) {
      // Mock screening — generate fake ranked stocks
      const mockStocks = ['NVDA','TSLA','AMD','MSTR','PLTR','COIN','META','AAPL','AMZN','GOOGL',
        'MSFT','ARM','SMCI','AVGO','MARA','IONQ','RKLB','HOOD','ACHR','RIOT'].map((ticker, i) => {
        const mockBars = generateMockBars(ticker)
        const last = mockBars[mockBars.length-1]
        const prev5 = mockBars[mockBars.length-5]
        const prev20 = mockBars[mockBars.length-20]
        return {
          ticker, price: last.c,
          change1d: (last.c - mockBars[mockBars.length-2].c) / mockBars[mockBars.length-2].c * 100,
          change5d: (last.c - prev5.c) / prev5.c * 100,
          volume: last.v,
          composite: Math.random() * 2 - 0.5,
          scores: { momentum: Math.random()-0.3, volatility: Math.random()*0.05, volumeSurge: Math.random()*2-0.5, trendBreak: Math.random()-0.3, rsiOversold: Math.random()-0.3 },
          bars: mockBars,
        }
      }).sort((a,b) => b.composite - a.composite)

      const tickers = mockStocks.slice(0,15).map(s=>s.ticker)
      setScreenResult({ stocks: mockStocks.slice(0,15), errors: [], criteria, timestamp: new Date(), mock: true })
      setActiveStocks(tickers)
      loadBarsForTickers(tickers, mockStocks.slice(0,15).reduce((acc,s)=>({...acc,[s.ticker]:s.bars}),{}))
      setIsScreening(false)
      return
    }

    try {
      const result = await screenStocks(criteria, 20)
      const tickers = result.stocks.map(s => s.ticker)
      setScreenResult(result)
      setActiveStocks(tickers)

      // Use bars from screener result (already fetched)
      const barsMap = result.stocks.reduce((acc, s) => ({ ...acc, [s.ticker]: s.bars.map(b=>({t:b.t||b.timestamp,o:b.o,h:b.h,l:b.l,c:b.c,v:b.v,vw:b.vw||b.c})) }), {})

      // Add crypto mock bars
      for (const t of ALL_CRYPTO) barsMap[t] = generateMockBars(CRYPTO_DISPLAY[t])

      const priceMap = result.stocks.reduce((acc, s) => ({
        ...acc,
        [s.ticker]: { price: s.price, changePct: s.change1d, change: s.price * s.change1d / 100, volume: s.volume }
      }), {})

      setBars(barsMap)
      setPrices(priceMap)
      recomputeSignals(barsMap)
      setDataStatus({ live: true, usingMock: false, keyFound: true, lastUpdate: new Date(), screened: tickers.length })
    } catch (err) {
      console.error('[Screener] Error:', err)
      setDataStatus(prev => ({ ...prev, error: err.message }))
    }
    setIsScreening(false)
  }, [apiKey, screenProfile])

  function loadBarsForTickers(tickers, existingBars = {}) {
    const barsMap = { ...existingBars }
    for (const t of tickers) if (!barsMap[t]) barsMap[t] = generateMockBars(t)
    for (const t of ALL_CRYPTO) barsMap[t] = generateMockBars(CRYPTO_DISPLAY[t])
    setBars(barsMap)
    const priceMap = {}
    for (const t of tickers) {
      const b = barsMap[t]
      if (b && b.length > 1) {
        const last = b[b.length-1], prev = b[b.length-2]
        priceMap[t] = { price: last.c, change: last.c-prev.c, changePct: (last.c-prev.c)/prev.c*100, volume: last.v }
      }
    }
    setPrices(priceMap)
    recomputeSignals(barsMap)
  }

  function recomputeSignals(barsMap) {
    const w = agent.weights
    const sigs = {}
    for (const [t, b] of Object.entries(barsMap)) {
      if (b && b.length > 0) sigs[t] = generateSignal(b, w)
    }
    setSignals(sigs)
    setWeights(w)
  }

  // ── Auto-screening scheduler ─────────────────────────────────────────────
  const SCREEN_INTERVAL = 30 * 60 // 30 minutes in seconds

  function startScheduler() {
    // Clear any existing timers
    if (screenTimerRef.current) clearInterval(screenTimerRef.current)
    if (countdownRef.current) clearInterval(countdownRef.current)

    setNextScreenIn(SCREEN_INTERVAL)

    // Countdown every second
    countdownRef.current = setInterval(() => {
      setNextScreenIn(prev => {
        if (prev <= 1) return SCREEN_INTERVAL // reset after trigger
        return prev - 1
      })
    }, 1000)

    // Run screener every 30 minutes
    screenTimerRef.current = setInterval(() => {
      console.log('[Scheduler] Auto-screening triggered')
      runScreener()
    }, SCREEN_INTERVAL * 1000)
  }

  function stopScheduler() {
    if (screenTimerRef.current) clearInterval(screenTimerRef.current)
    if (countdownRef.current) clearInterval(countdownRef.current)
    setNextScreenIn(null)
  }

  // Auto-run screener on mount + start scheduler
  useEffect(() => {
    runScreener()
    startScheduler()
    return () => stopScheduler() // cleanup on unmount
  }, [])

  // Toggle auto-screen
  useEffect(() => {
    if (autoScreenEnabled) startScheduler()
    else stopScheduler()
  }, [autoScreenEnabled])

  // ── RL Training ──────────────────────────────────────────────────────────
  async function startTraining() {
    if (isTraining || Object.keys(bars).length === 0) return
    setIsTraining(true)
    await trainEpisodes(bars, 30,
      ep => {
        setTrainingLog(prev => [...prev.slice(-50), {
          episode: ep.episode, score: ep.result.ragScore.toFixed(3),
          sharpe: ep.result.sharpe.toFixed(2), ret: (ep.result.totalReturn*100).toFixed(1)+'%',
          regime: ep.regime, improving: ep.agentState.improving,
        }])
        setRlProgress(ep.agentState)
        setWeights({...ep.currentWeights})
      },
      done => {
        setWeights({...done.bestWeights})
        recomputeSignals(bars)
        runBacktest(done.bestWeights)
        setIsTraining(false)
      }
    )
  }

  function runBacktest(w = weights) {
    setBacktestResult(backtestPortfolio(bars, w))
  }

  // ── Computed ──────────────────────────────────────────────────────────────
  const allTracked = [
    ...activeStocks.map(t => ({ ticker: t, display: t, type: 'stock' })),
    ...ALL_CRYPTO.map(t => ({ ticker: t, display: CRYPTO_DISPLAY[t], type: 'crypto' })),
  ]

  const sortedBySignal = allTracked
    .map(a => ({ ...a, sig: signals[a.ticker] || { score:0, signal:'HOLD', factors:{}, confidence:0 }, px: prices[a.ticker] }))
    .sort((a,b) => (b.sig?.score||0) - (a.sig?.score||0))

  const buys = sortedBySignal.filter(a => a.sig?.signal === 'BUY')
  const sells = sortedBySignal.filter(a => a.sig?.signal === 'SELL')

  const statusBadge = () => {
    if (!dataStatus.keyFound) return <span style={S.badge(C.yellow)}>MOCK — NO API KEY</span>
    if (dataStatus.error) return <span style={S.badge(C.red)}>API ERROR</span>
    if (dataStatus.live) return <span style={S.badge(C.green)}>LIVE · {dataStatus.screened} SCREENED</span>
    return <span style={S.badge(C.yellow)}>LOADING...</span>
  }

  return (
    <div style={S.app}>
      {/* Header */}
      <header style={S.header}>
        <div style={{ display:'flex', alignItems:'center', gap:16 }}>
          <span style={S.logo}>STOCKBOT</span>
          <span style={S.badge(C.accent)}>RL·ENABLED</span>
          {statusBadge()}
        </div>
        <nav style={S.nav}>
          {['screen','signals','backtest','train','paper'].map(t => (
            <button key={t} style={S.navBtn(tab===t)} onClick={()=>setTab(t)}>{t.toUpperCase()}</button>
          ))}
        </nav>
        <div style={{ fontSize:10, color:C.textDim, textAlign:'right' }}>
          {dataStatus.lastUpdate ? <div>UPDATED {dataStatus.lastUpdate.toLocaleTimeString()}</div> : 'INITIALIZING...'}
          <div style={{ display:'flex', gap:8, justifyContent:'flex-end', alignItems:'center' }}>
            {autoScreenEnabled && nextScreenIn !== null && (
              <span style={{ color:C.accent }}>
                ⟳ {Math.floor(nextScreenIn/60)}:{String(nextScreenIn%60).padStart(2,'0')}
              </span>
            )}
            <span>GEN {agent.generation} · ε={agent.epsilon.toFixed(2)}</span>
          </div>
        </div>
      </header>

      <main style={S.main}>

        {/* ══ SCREENER ══ */}
        {tab === 'screen' && (
          <div style={{ display:'flex', flexDirection:'column', gap:16 }}>

            {/* Scheduler status bar */}
            <div style={{ background:C.surface, border:`1px solid ${C.border}`, borderRadius:8, padding:'12px 20px', display:'flex', alignItems:'center', justifyContent:'space-between' }}>
              <div style={{ display:'flex', alignItems:'center', gap:16 }}>
                <div style={{ width:8, height:8, borderRadius:'50%', background:autoScreenEnabled?C.green:C.textDim, boxShadow:autoScreenEnabled?`0 0 8px ${C.green}`:'none' }}/>
                <span style={{ fontSize:11, color:C.textBright, letterSpacing:2 }}>
                  {autoScreenEnabled ? 'AUTO-SCREEN ON' : 'AUTO-SCREEN OFF'}
                </span>
                {autoScreenEnabled && nextScreenIn !== null && (
                  <span style={{ fontSize:11, color:C.textDim }}>
                    next scan in{' '}
                    <span style={{ color:C.accent, fontWeight:700 }}>
                      {Math.floor(nextScreenIn/60)}:{String(nextScreenIn%60).padStart(2,'0')}
                    </span>
                  </span>
                )}
              </div>
              <div style={{ display:'flex', gap:8, alignItems:'center' }}>
                <span style={{ fontSize:10, color:C.textDim }}>INTERVAL: 30 MIN</span>
                <button
                  style={{ ...S.btn(autoScreenEnabled?'danger':'primary'), padding:'4px 14px', fontSize:10 }}
                  onClick={() => setAutoScreenEnabled(p => !p)}>
                  {autoScreenEnabled ? '⏸ PAUSE' : '▶ RESUME'}
                </button>
                <button
                  style={{ ...S.btn('primary'), padding:'4px 14px', fontSize:10 }}
                  onClick={() => { startScheduler(); runScreener(screenProfile, customCriteria) }}
                  disabled={isScreening}>
                  ⟳ SCAN NOW
                </button>
              </div>
            </div>

            {/* Profile selector */}
            <div style={S.panel}>
              <div style={S.panelTitle}>SCREENING PROFILE — SELECT STRATEGY</div>
              <div style={{ display:'flex', gap:10, flexWrap:'wrap', marginBottom:16 }}>
                {Object.entries(SCREENING_PROFILES).map(([key, p]) => (
                  <button key={key}
                    style={{ ...S.btn(screenProfile===key?'active':'default'), fontSize:11 }}
                    onClick={() => setScreenProfile(key)}>
                    {p.icon} {p.label}
                  </button>
                ))}
              </div>

              {/* Show active profile description */}
              <div style={{ fontSize:11, color:C.textDim, marginBottom:16 }}>
                {SCREENING_PROFILES[screenProfile]?.description} — criteria weights:
                <span style={{ color:C.accent, marginLeft:8 }}>
                  {Object.entries(SCREENING_PROFILES[screenProfile]?.criteria || {})
                    .filter(([,v]) => v > 0)
                    .map(([k,v]) => `${k} ${(v*100).toFixed(0)}%`)
                    .join('  ·  ')}
                </span>
              </div>

              {/* Criteria sliders */}
              <div style={{ display:'grid', gridTemplateColumns:'repeat(5,1fr)', gap:12, marginBottom:16 }}>
                {Object.entries(customCriteria).filter(([k]) => !['minPrice','maxPrice','minVolume'].includes(k)).map(([k, v]) => (
                  <div key={k}>
                    <div style={{ fontSize:9, color:C.textDim, marginBottom:4, letterSpacing:1 }}>{k.toUpperCase()}</div>
                    <input type="range" min="0" max="1" step="0.05" value={v}
                      onChange={e => setCustomCriteria(prev => ({ ...prev, [k]: parseFloat(e.target.value) }))}
                      style={{ width:'100%', accentColor:C.accent }} />
                    <div style={{ fontSize:10, color:C.textBright }}>{(v*100).toFixed(0)}%</div>
                  </div>
                ))}
              </div>

              <div style={{ display:'flex', gap:10 }}>
                <button style={S.btn('primary')} onClick={() => runScreener(screenProfile, customCriteria)} disabled={isScreening}>
                  {isScreening ? '⟳ SCREENING MARKET...' : '⟳ RUN SCREENER'}
                </button>
                <button style={S.btn()} onClick={() => {
                  const c = SCREENING_PROFILES[screenProfile].criteria
                  setCustomCriteria(c)
                  runScreener(screenProfile, c)
                }}>
                  RESET TO PROFILE
                </button>
                <span style={{ fontSize:11, color:C.textDim, alignSelf:'center' }}>
                  Scans {apiKey ? '25 stocks' : '20 stocks (mock)'} · ranks by composite score
                </span>
              </div>
            </div>

            {/* Results */}
            {screenResult && (
              <>
                <div style={S.grid4}>
                  {[
                    { label:'STOCKS FOUND', value:screenResult.stocks.length, color:C.accent },
                    { label:'BUY SIGNALS', value:buys.length, color:C.green },
                    { label:'SELL SIGNALS', value:sells.length, color:C.red },
                    { label:'SCREENED AT', value:screenResult.timestamp?.toLocaleTimeString(), color:C.textDim },
                  ].map(m => (
                    <div key={m.label} style={S.panel}>
                      <div style={S.panelTitle}>{m.label}</div>
                      <div style={{ ...S.metric, fontSize:22, color:m.color }}>{m.value}</div>
                    </div>
                  ))}
                </div>

                <div style={S.panel}>
                  <div style={S.panelTitle}>
                    SCREENER RESULTS — RANKED BY {SCREENING_PROFILES[screenProfile]?.label}
                    {screenResult.mock && <span style={{ color:C.yellow, marginLeft:8 }}>(MOCK DATA)</span>}
                  </div>
                  <div style={{ overflowX:'auto' }}>
                    <table style={S.table}>
                      <thead>
                        <tr>
                          {['#','TICKER','PRICE','1D %','5D %','VOLUME','MOMENTUM','VOL SURGE','TREND','COMPOSITE','SIGNAL'].map(h =>
                            <th key={h} style={S.th}>{h}</th>
                          )}
                        </tr>
                      </thead>
                      <tbody>
                        {screenResult.stocks.map((s, i) => {
                          const sig = signals[s.ticker] || { signal:'HOLD', score:0 }
                          return (
                            <tr key={s.ticker} style={{ cursor:'pointer' }}
                              onClick={() => { setSelectedAsset(s.ticker); setTab('signals') }}>
                              <td style={{ ...S.td, color:C.textDim }}>{i+1}</td>
                              <td style={{ ...S.td, color:C.textBright, fontWeight:700 }}>{s.ticker}</td>
                              <td style={S.td}>{fmt.price(s.price)}</td>
                              <td style={{ ...S.td, ...fmt.chg(s.change1d) }}>{s.change1d?.toFixed(2)}%</td>
                              <td style={{ ...S.td, ...fmt.chg(s.change5d) }}>{s.change5d?.toFixed(2)}%</td>
                              <td style={{ ...S.td, color:C.textDim }}>{(s.volume/1e6).toFixed(1)}M</td>
                              <td style={{ ...S.td, color:s.scores.momentum>0?C.green:C.red }}>{(s.scores.momentum*100).toFixed(1)}%</td>
                              <td style={{ ...S.td, color:s.scores.volumeSurge>0.5?C.orange:C.textDim }}>{s.scores.volumeSurge>0?'+':''}{(s.scores.volumeSurge*100).toFixed(0)}%</td>
                              <td style={{ ...S.td, color:s.scores.trendBreak>0?C.green:C.red }}>{s.scores.trendBreak>0?'↑':'↓'} {Math.abs(s.scores.trendBreak*100).toFixed(1)}%</td>
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

            {isScreening && (
              <div style={{ ...S.panel, textAlign:'center', padding:60 }}>
                <div style={{ fontSize:24, marginBottom:12 }}>⟳</div>
                <div style={{ color:C.accent, letterSpacing:4, fontSize:12 }}>SCANNING MARKET...</div>
                <div style={{ color:C.textDim, fontSize:11, marginTop:8 }}>
                  Fetching bars for 25 stocks · scoring momentum, volatility, volume, trend, RSI
                </div>
              </div>
            )}
          </div>
        )}

        {/* ══ SIGNALS ══ */}
        {tab === 'signals' && (
          <div style={{ display:'flex', flexDirection:'column', gap:16 }}>
            <div style={{ display:'flex', gap:6, flexWrap:'wrap' }}>
              {allTracked.map(a => (
                <button key={a.ticker} onClick={() => setSelectedAsset(a.ticker)} style={{ ...S.navBtn(selectedAsset===a.ticker), fontSize:11 }}>
                  {a.display}
                </button>
              ))}
            </div>

            {/* Signal heatmap */}
            <div style={S.panel}>
              <div style={S.panelTitle}>SIGNAL HEATMAP — SCREENED UNIVERSE</div>
              <div style={{ display:'grid', gridTemplateColumns:'repeat(5,1fr)', gap:6 }}>
                {sortedBySignal.map(a => {
                  const score = a.sig?.score || 0
                  const hue = score > 0 ? '144,255,136' : '255,51,102'
                  return (
                    <div key={a.ticker} onClick={() => setSelectedAsset(a.ticker)}
                      style={{ padding:'8px 6px', borderRadius:4, cursor:'pointer', background:`rgba(${hue},${Math.abs(score)*0.4})`, border:`1px solid rgba(${hue},${Math.abs(score)*0.8})`, textAlign:'center' }}>
                      <div style={{ fontSize:11, fontWeight:700, color:C.textBright }}>{a.display}</div>
                      <div style={{ fontSize:10, color:score>0?C.green:score<0?C.red:C.textDim }}>{score>=0?'+':''}{score.toFixed(2)}</div>
                    </div>
                  )
                })}
              </div>
            </div>

            {/* Selected asset detail */}
            {selectedAsset && signals[selectedAsset] && (() => {
              const sig = signals[selectedAsset]
              const assetBars = bars[selectedAsset] || []
              const px = prices[selectedAsset]
              const chartData = assetBars.slice(-60).map(b => ({ date:fmt.date(b.t), price:b.c }))

              return (
                <div style={{ display:'flex', flexDirection:'column', gap:16 }}>
                  <div style={S.grid4}>
                    <div style={S.panel}>
                      <div style={S.panelTitle}>SIGNAL</div>
                      <div style={{ ...S.metric, color:sig.signal==='BUY'?C.green:sig.signal==='SELL'?C.red:C.textDim }}>{sig.signal}</div>
                      <div style={S.metricSub}>Score: {sig.score.toFixed(4)}</div>
                    </div>
                    <div style={S.panel}>
                      <div style={S.panelTitle}>PRICE</div>
                      <div style={S.metric}>{fmt.price(px?.price)}</div>
                      <div style={{ ...S.metricSub, ...fmt.chg(px?.changePct||0) }}>{px?.changePct?.toFixed(2)}% today</div>
                    </div>
                    <div style={S.panel}>
                      <div style={S.panelTitle}>CONFIDENCE</div>
                      <div style={S.metric}>{(sig.confidence*100).toFixed(0)}%</div>
                    </div>
                    <div style={S.panel}>
                      <div style={S.panelTitle}>ACTION</div>
                      <button style={S.btn('primary')} onClick={() => setPaperTrades(p => [{ ticker:selectedAsset, side:sig.signal, price:px?.price, size:1000, score:sig.score.toFixed(3), time:new Date().toLocaleTimeString() }, ...p.slice(0,49)])}>
                        PAPER TRADE
                      </button>
                    </div>
                  </div>

                  <div style={S.panel}>
                    <div style={S.panelTitle}>{selectedAsset} — 60-DAY PRICE</div>
                    <ResponsiveContainer width="100%" height={200}>
                      <AreaChart data={chartData}>
                        <defs>
                          <linearGradient id="pg" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor={C.accent} stopOpacity={0.3}/>
                            <stop offset="95%" stopColor={C.accent} stopOpacity={0}/>
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
                        <XAxis dataKey="date" tick={{ fontSize:9, fill:C.textDim }} interval={9}/>
                        <YAxis tick={{ fontSize:9, fill:C.textDim }}/>
                        <Tooltip contentStyle={{ background:C.panel, border:`1px solid ${C.border}` }}/>
                        <Area type="monotone" dataKey="price" stroke={C.accent} fill="url(#pg)" strokeWidth={2} dot={false}/>
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>

                  <div style={S.panel}>
                    <div style={S.panelTitle}>FACTOR DECOMPOSITION</div>
                    <ResponsiveContainer width="100%" height={160}>
                      <BarChart data={Object.entries(sig.factors).map(([k,v])=>({factor:k,value:v}))}>
                        <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
                        <XAxis dataKey="factor" tick={{ fontSize:10, fill:C.textDim }}/>
                        <YAxis domain={[-3,3]} tick={{ fontSize:9, fill:C.textDim }}/>
                        <ReferenceLine y={0} stroke={C.border}/>
                        <Tooltip contentStyle={{ background:C.panel, border:`1px solid ${C.border}` }}/>
                        <Bar dataKey="value" fill={C.accent} radius={[4,4,0,0]}/>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )
            })()}
          </div>
        )}

        {/* ══ BACKTEST ══ */}
        {tab === 'backtest' && (
          <div style={{ display:'flex', flexDirection:'column', gap:16 }}>
            <div style={{ display:'flex', gap:12, alignItems:'center' }}>
              <button style={S.btn('primary')} onClick={() => runBacktest()}>RUN BACKTEST</button>
              <span style={{ fontSize:11, color:C.textDim }}>
                {dataStatus.usingMock?'MOCK':'REAL'} data · {activeStocks.length} screened stocks · {BACKTEST_DAYS} days · current RL weights
              </span>
            </div>

            {backtestResult && (() => {
              const r = backtestResult
              const equityData = r.equity.map((v,i)=>({i,value:v}))
              const ddData = []; let peak=-Infinity
              for (const {i,value} of equityData) {
                if(value>peak) peak=value
                ddData.push({i, dd:((peak-value)/peak)*-100})
              }
              return (
                <div style={{ display:'flex', flexDirection:'column', gap:16 }}>
                  <div style={{ display:'grid', gridTemplateColumns:'repeat(4,1fr)', gap:16 }}>
                    {[
                      { label:'TOTAL RETURN', value:fmt.pct(r.totalReturn), color:r.totalReturn>=0?C.green:C.red },
                      { label:'SHARPE RATIO', value:r.sharpe.toFixed(2), color:r.sharpe>=1?C.green:r.sharpe>=0?C.yellow:C.red },
                      { label:'MAX DRAWDOWN', value:`-${(r.maxDrawdown*100).toFixed(1)}%`, color:r.maxDrawdown<0.1?C.green:r.maxDrawdown<0.2?C.yellow:C.red },
                      { label:'RAG SCORE', value:r.ragScore.toFixed(3), color:C.purple },
                      { label:'CALMAR', value:r.calmar.toFixed(2), color:C.accent },
                      { label:'WIN RATE', value:`${(r.winRate*100).toFixed(0)}%`, color:r.winRate>=0.5?C.green:C.yellow },
                      { label:'TRADES', value:r.tradeCount, color:C.text },
                      { label:'FINAL VALUE', value:`$${r.finalValue.toFixed(0)}`, color:C.textBright },
                    ].map(m => (
                      <div key={m.label} style={S.panel}>
                        <div style={S.panelTitle}>{m.label}</div>
                        <div style={{ ...S.metric, fontSize:22, color:m.color }}>{m.value}</div>
                      </div>
                    ))}
                  </div>

                  <div style={S.panel}>
                    <div style={S.panelTitle}>EQUITY CURVE</div>
                    <ResponsiveContainer width="100%" height={240}>
                      <AreaChart data={equityData}>
                        <defs>
                          <linearGradient id="eg" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor={C.green} stopOpacity={0.3}/>
                            <stop offset="95%" stopColor={C.green} stopOpacity={0}/>
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
                        <XAxis dataKey="i" tick={false}/>
                        <YAxis tick={{ fontSize:9, fill:C.textDim }} tickFormatter={v=>`$${(v/1000).toFixed(0)}K`}/>
                        <Tooltip contentStyle={{ background:C.panel, border:`1px solid ${C.border}` }} formatter={v=>[`$${v.toFixed(0)}`,'Value']}/>
                        <ReferenceLine y={100000} stroke={C.border} strokeDasharray="4 4"/>
                        <Area type="monotone" dataKey="value" stroke={C.green} fill="url(#eg)" strokeWidth={2} dot={false}/>
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>

                  <div style={S.panel}>
                    <div style={S.panelTitle}>DRAWDOWN</div>
                    <ResponsiveContainer width="100%" height={120}>
                      <AreaChart data={ddData}>
                        <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
                        <XAxis dataKey="i" tick={false}/>
                        <YAxis tick={{ fontSize:9, fill:C.textDim }}/>
                        <Tooltip contentStyle={{ background:C.panel, border:`1px solid ${C.border}` }} formatter={v=>[`${v.toFixed(2)}%`,'DD']}/>
                        <Area type="monotone" dataKey="dd" stroke={C.red} fill={`${C.red}33`} strokeWidth={1.5} dot={false}/>
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )
            })()}
            {!backtestResult && <div style={{ ...S.panel, textAlign:'center', padding:60, color:C.textDim }}>Click RUN BACKTEST to evaluate screened stocks</div>}
          </div>
        )}

        {/* ══ TRAIN ══ */}
        {tab === 'train' && (
          <div style={{ display:'flex', flexDirection:'column', gap:16 }}>
            <div style={{ display:'flex', gap:12, alignItems:'center' }}>
              <button style={S.btn(isTraining?'default':'primary')} onClick={startTraining} disabled={isTraining}>
                {isTraining?'▶ TRAINING...':'▶ START TRAINING'}
              </button>
              {isTraining && <button style={S.btn('danger')} onClick={() => { stopTraining(); setIsTraining(false) }}>■ STOP</button>}
              <button style={S.btn()} onClick={() => { agent.reset(); setTrainingLog([]); setWeights(agent.weights); setRlProgress(null) }}>↺ RESET</button>
              <span style={{ fontSize:11, color:C.textDim }}>Trains on {activeStocks.length} screened stocks · optimizes factor weights for max Sharpe+Calmar</span>
            </div>

            <div style={S.grid4}>
              {[
                { label:'GENERATION', value:agent.generation },
                { label:'BEST RAG SCORE', value:agent.bestScore===-Infinity?'—':agent.bestScore.toFixed(3) },
                { label:'EXPLORATION ε', value:agent.epsilon.toFixed(3) },
                { label:'TREND', value:rlProgress?.improving?'↑ IMPROVING':rlProgress?'↔ EXPLORING':'—', color:rlProgress?.improving?C.green:C.yellow },
              ].map(m=>(
                <div key={m.label} style={S.panel}>
                  <div style={S.panelTitle}>{m.label}</div>
                  <div style={{ fontSize:22, fontWeight:700, color:m.color||C.textBright }}>{m.value}</div>
                </div>
              ))}
            </div>

            <div style={S.panel}>
              <div style={S.panelTitle}>FACTOR WEIGHTS — LIVE OPTIMIZATION</div>
              <div style={{ display:'flex', gap:24, alignItems:'flex-end' }}>
                {Object.entries(weights).map(([k,w])=>(
                  <div key={k} style={{ flex:1, textAlign:'center' }}>
                    <div style={{ fontSize:10, color:C.textDim, marginBottom:8 }}>{k.toUpperCase()}</div>
                    <div style={{ height:100, background:C.border, borderRadius:4, position:'relative', overflow:'hidden' }}>
                      <div style={{ position:'absolute', bottom:0, width:'100%', height:`${w*100}%`, background:`linear-gradient(0deg,${C.accent},${C.purple})`, transition:'height 0.5s' }}/>
                    </div>
                    <div style={{ fontSize:14, fontWeight:700, color:C.textBright, marginTop:6 }}>{(w*100).toFixed(1)}%</div>
                  </div>
                ))}
              </div>
            </div>

            {trainingLog.length > 0 && (
              <div style={S.panel}>
                <div style={S.panelTitle}>TRAINING LOG</div>
                <ResponsiveContainer width="100%" height={160}>
                  <LineChart data={trainingLog}>
                    <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
                    <XAxis dataKey="episode" tick={{ fontSize:9, fill:C.textDim }}/>
                    <YAxis tick={{ fontSize:9, fill:C.textDim }}/>
                    <ReferenceLine y={0} stroke={C.border}/>
                    <Tooltip contentStyle={{ background:C.panel, border:`1px solid ${C.border}` }}/>
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
          <div style={{ display:'flex', flexDirection:'column', gap:16 }}>
            <div style={S.panel}>
              <div style={S.panelTitle}>PAPER TRADE LOG</div>
              {paperTrades.length > 0 ? (
                <table style={S.table}>
                  <thead><tr>{['TIME','TICKER','SIDE','PRICE','SIZE','SCORE'].map(h=><th key={h} style={S.th}>{h}</th>)}</tr></thead>
                  <tbody>
                    {paperTrades.map((t,i)=>(
                      <tr key={i}>
                        <td style={{ ...S.td, fontSize:10, color:C.textDim }}>{t.time}</td>
                        <td style={{ ...S.td, fontWeight:700, color:C.textBright }}>{t.ticker}</td>
                        <td style={S.td}><span style={S.sigBadge(t.side)}>{t.side}</span></td>
                        <td style={S.td}>{fmt.price(t.price)}</td>
                        <td style={{ ...S.td, color:C.green }}>${t.size}</td>
                        <td style={{ ...S.td, color:parseFloat(t.score)>0?C.green:C.red }}>{t.score}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <div style={{ textAlign:'center', padding:60, color:C.textDim }}>
                  Go to SIGNALS tab → select asset → click PAPER TRADE
                </div>
              )}
            </div>
          </div>
        )}

      </main>
    </div>
  )
}
