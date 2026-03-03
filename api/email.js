// Vercel serverless — /api/email
// Uses Resend (free 3k/mo). Set RESEND_API_KEY in Vercel env vars.
// Types: portfolio_alert | daily_briefing | end_of_day

export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin','*')
  res.setHeader('Access-Control-Allow-Methods','POST,OPTIONS')
  res.setHeader('Access-Control-Allow-Headers','Content-Type')
  if(req.method==='OPTIONS') return res.status(200).end()
  if(req.method!=='POST') return res.status(405).json({error:'Method not allowed'})

  const { type, to, data } = req.body
  if(!to||!type) return res.status(400).json({error:'Missing to or type'})
  const apiKey = process.env.RESEND_API_KEY
  if(!apiKey) return res.status(500).json({error:'RESEND_API_KEY not configured'})

  const C = {
    bg:'#050510', panel:'#0f0f22', border:'#1a1a3a', surface:'#0a0a2a',
    green:'#00ff88', red:'#ff3366', accent:'#00d4ff', yellow:'#ffd700',
    purple:'#a855f7', text:'#e2e8f0', dim:'#64748b'
  }
  const base = c =>
    `<div style="background:${C.bg};color:${C.text};font-family:'Courier New',monospace;padding:24px;max-width:640px;margin:0 auto">${c}` +
    `<p style="color:${C.dim};font-size:10px;text-align:center;margin-top:24px;border-top:1px solid ${C.border};padding-top:12px">` +
    `⚠️ Not financial advice · STOCKBOT · <a href="https://stockbot-six-woad.vercel.app" style="color:${C.accent}">Open Dashboard</a></p></div>`
  const badge = (t,c) => `<span style="display:inline-block;padding:2px 8px;border:1px solid ${c};color:${c};font-size:9px;border-radius:3px;letter-spacing:1px;margin-right:4px">${t}</span>`
  const section = (title, color, content) =>
    `<div style="background:${C.panel};border:1px solid ${color||C.border};border-radius:8px;padding:16px;margin-bottom:12px">` +
    `<div style="font-size:9px;letter-spacing:3px;color:${C.dim};margin-bottom:10px;text-transform:uppercase">${title}</div>${content}</div>`
  const divider = `<div style="border-bottom:1px solid ${C.border}40;margin:6px 0"></div>`
  const row = (cols) => `<div style="display:flex;justify-content:space-between;align-items:center;padding:8px 0">${cols.map(c=>`<span>${c}</span>`).join('')}</div>`

  let subject = '', html = ''

  // ── PORTFOLIO ALERT ───────────────────────────────────────────────────────
  if (type === 'portfolio_alert') {
    const { portfolioValue, pnl, pnlPct, positions, direction } = data
    const isUp = direction === 'up', color = isUp ? C.green : C.red
    subject = `${isUp?'📈':'📉'} Stockbot: Portfolio ${isUp?'UP':'DOWN'} ${Math.abs(pnlPct).toFixed(1)}% — Action may be needed`
    html = base(
      `<div style="border:2px solid ${color};border-radius:8px;padding:16px;margin-bottom:12px">
        <h2 style="color:${color};margin:0;font-size:15px">${isUp?'📈':'📉'} Portfolio ${isUp?'Surge':'Drop'} Alert</h2>
        <div style="font-size:28px;font-weight:700;color:#fff;margin:10px 0">$${portfolioValue.toLocaleString()}</div>
        <div style="font-size:18px;color:${color}">${isUp?'+':''}$${pnl.toFixed(0)} &nbsp;(${isUp?'+':''}${pnlPct.toFixed(2)}%)</div>
      </div>` +
      section('Open Positions', null,
        (positions||[]).map(p =>
          `<div style="display:grid;grid-template-columns:2fr 1fr 1fr;gap:4px;padding:8px 0;border-bottom:1px solid ${C.border}40">
            <span style="color:#fff;font-weight:700">${p.ticker}</span>
            <span style="color:${C.dim}">${p.shares} sh @ $${(p.avgPrice||0).toFixed(2)}</span>
            <span style="color:${p.pnl>=0?C.green:C.red};font-weight:700;text-align:right">${p.pnl>=0?'+':''}$${(p.pnl||0).toFixed(0)}</span>
          </div>`
        ).join('') || `<div style="color:${C.dim}">No open positions</div>`
      )
    )
  }

  // ── MORNING BRIEFING ─────────────────────────────────────────────────────
  if (type === 'daily_briefing') {
    const { topPicks, buys, sells, marketMood, date, algoMode, portfolioValue, newsHeadlines } = data
    const moodColor = marketMood==='BULLISH'?C.green:marketMood==='BEARISH'?C.red:C.yellow
    const picks = topPicks || buys || []
    subject = `☀️ Morning Brief ${date}: ${picks.slice(0,3).map(s=>s.ticker).join(' · ')} — ${marketMood}`
    html = base(
      // Header
      `<div style="border:2px solid ${C.accent};border-radius:8px;padding:16px;margin-bottom:12px">
        <div style="display:flex;justify-content:space-between;align-items:flex-start">
          <div>
            <h2 style="color:${C.accent};margin:0;font-size:15px">☀️ Morning Market Briefing</h2>
            <div style="color:${C.dim};font-size:10px;margin-top:4px">${date} · Open in 30 min</div>
          </div>
          <div style="text-align:right">
            ${badge(marketMood, moodColor)}
            ${algoMode ? badge(algoMode+' MODE', C.accent) : ''}
          </div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-top:14px">
          <div style="background:${C.surface};padding:10px;border-radius:6px;text-align:center">
            <div style="font-size:8px;color:${C.dim};letter-spacing:2px">PORTFOLIO</div>
            <div style="font-size:16px;font-weight:700;color:#fff">$${portfolioValue ? portfolioValue.toLocaleString() : '—'}</div>
          </div>
          <div style="background:${C.surface};padding:10px;border-radius:6px;text-align:center">
            <div style="font-size:8px;color:${C.dim};letter-spacing:2px">BUY SIGNALS</div>
            <div style="font-size:16px;font-weight:700;color:${C.green}">${buys.length}</div>
          </div>
          <div style="background:${C.surface};padding:10px;border-radius:6px;text-align:center">
            <div style="font-size:8px;color:${C.dim};letter-spacing:2px">SELL SIGNALS</div>
            <div style="font-size:16px;font-weight:700;color:${C.red}">${sells.length}</div>
          </div>
        </div>
      </div>` +

      // Top picks with prices and entry zones
      (picks.length > 0 ? section('⭐ High Conviction — Watch at Open', C.purple,
        picks.slice(0,6).map((s,i) => {
          const px = s.price != null && s.price > 0 ? s.price : null  // null = no live price
          const entry = px ? (s.entryZone || `$${(px*0.99).toFixed(2)}–$${(px*1.005).toFixed(2)}`) : '—'
          const stop  = px ? (s.stopLoss  || `$${(px*0.92).toFixed(2)}`) : '—'
          const target= px ? (s.target    || `$${(px*1.10).toFixed(2)}`) : '—'
          const ensStr = s.ensemble ? badge(s.ensemble, s.ensemble==='BUY'?C.green:C.red) : ''
          return `<div style="padding:12px 0;border-bottom:1px solid ${C.border}40">
            <div style="display:flex;justify-content:space-between;align-items:center">
              <span style="color:#fff;font-weight:700;font-size:14px">#${i+1} ${s.ticker}</span>
              <span>${ensStr}${badge('Conf '+Math.round((s.confidence||0)*100)+'%', C.accent)}</span>
            </div>
            ${px ? `<div style="color:#fff;font-size:16px;font-weight:700;margin:4px 0">$${px.toFixed(2)}</div>` : `<div style="font-size:10px;color:${C.dim};margin:4px 0">Price: add Polygon API key for live prices</div>`}
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-top:8px">
              <div style="background:${C.surface};padding:6px 8px;border-radius:4px">
                <div style="font-size:8px;color:${C.dim}">ENTRY ZONE</div>
                <div style="font-size:10px;color:${C.green}">${entry}</div>
              </div>
              <div style="background:${C.surface};padding:6px 8px;border-radius:4px">
                <div style="font-size:8px;color:${C.dim}">STOP LOSS</div>
                <div style="font-size:10px;color:${C.red}">${stop}</div>
              </div>
              <div style="background:${C.surface};padding:6px 8px;border-radius:4px">
                <div style="font-size:8px;color:${C.dim}">TARGET</div>
                <div style="font-size:10px;color:${C.yellow}">${target}</div>
              </div>
            </div>
            <div style="font-size:10px;color:${C.dim};margin-top:6px">${s.reason || ''}</div>
            ${s.newsHeadline ? `<div style="font-size:9px;color:${C.purple};margin-top:4px">📰 ${s.newsHeadline}</div>` : ''}
          </div>`
        }).join('')
      ) : '') +

      // All buy signals compact
      (buys.length > 0 ? section(`🚀 All Buy Signals (${buys.length})`, C.green,
        buys.slice(0,10).map(s =>
          `<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid ${C.border}30">
            <span style="color:#fff;font-weight:700">${s.ticker}</span>
            ${s.price ? `<span style="color:${C.dim}">$${s.price.toFixed(2)}</span>` : ''}
            <span style="color:${C.green}">${(s.score||0).toFixed(3)}</span>
            <span style="color:${C.dim};font-size:9px">Conf ${Math.round((s.confidence||0)*100)}%</span>
          </div>`
        ).join('')
      ) : '') +

      // News catalysts
      (newsHeadlines && newsHeadlines.length > 0 ? section('📰 Market Moving News', C.yellow,
        newsHeadlines.slice(0,4).map(n =>
          `<div style="padding:8px 0;border-bottom:1px solid ${C.border}30">
            <div style="font-size:11px;color:#fff">${n.title}</div>
            <div style="display:flex;gap:8px;margin-top:4px">
              ${n.tickers && n.tickers.length ? n.tickers.map(t=>`<span style="color:${C.accent};font-size:9px">${t}</span>`).join('') : ''}
              ${badge(n.sentiment?.label||'NEUTRAL', n.sentiment?.label==='BULLISH'?C.green:n.sentiment?.label==='BEARISH'?C.red:C.dim)}
              <span style="color:${C.dim};font-size:9px">${n.source||''}</span>
            </div>
          </div>`
        ).join('')
      ) : '') +

      // Sell/avoid signals
      (sells.length > 0 ? section(`⚠️ Avoid / Watch for Shorts (${sells.length})`, C.red,
        sells.slice(0,5).map(s =>
          `<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid ${C.border}30">
            <span style="color:${C.red};font-weight:700">${s.ticker}</span>
            ${s.price ? `<span style="color:${C.dim}">$${s.price.toFixed(2)}</span>` : ''}
            <span style="color:${C.dim};font-size:9px">${s.reason||'SELL signal'}</span>
          </div>`
        ).join('')
      ) : '')
    )
  }

  // ── END-OF-DAY PERFORMANCE REPORT ────────────────────────────────────────
  if (type === 'end_of_day') {
    const { portfolioValue, pnl, pnlPct, pnlToday, pnlTodayPct, positions,
            bestPosition, worstPosition, nextDayWatchlist, todayTrades,
            totalTrades, marketMood, date, sharpe, maxDrawdown, algoMode } = data
    const isUp = pnl >= 0, color = isUp ? C.green : C.red
    const isTodayUp = (pnlToday||0) >= 0
    const moodColor = marketMood==='BULLISH'?C.green:marketMood==='BEARISH'?C.red:C.yellow
    subject = `${isUp?'📈':'📉'} EOD Report ${date}: ${isTodayUp?'+':''}${(pnlTodayPct||pnlPct||0).toFixed(2)}% today · ${(nextDayWatchlist||[]).slice(0,2).map(s=>s.ticker).join('/')}`
    html = base(
      // Header
      `<div style="border:2px solid ${color};border-radius:8px;padding:16px;margin-bottom:12px">
        <div style="display:flex;justify-content:space-between;align-items:flex-start">
          <div>
            <h2 style="color:${color};margin:0;font-size:15px">📊 End-of-Day Performance</h2>
            <div style="color:${C.dim};font-size:10px;margin-top:4px">${date} · Market Closed</div>
          </div>
          <div>${badge(marketMood, moodColor)}${algoMode ? ' '+badge(algoMode, C.accent) : ''}</div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:10px;margin-top:14px">
          <div style="background:${C.surface};padding:10px;border-radius:6px;text-align:center">
            <div style="font-size:8px;color:${C.dim};letter-spacing:2px">PORTFOLIO</div>
            <div style="font-size:15px;font-weight:700;color:#fff">$${portfolioValue.toLocaleString()}</div>
          </div>
          <div style="background:${C.surface};padding:10px;border-radius:6px;text-align:center">
            <div style="font-size:8px;color:${C.dim};letter-spacing:2px">TODAY P&L</div>
            <div style="font-size:15px;font-weight:700;color:${isTodayUp?C.green:C.red}">${isTodayUp?'+':''}${(pnlTodayPct||pnlPct||0).toFixed(2)}%</div>
          </div>
          <div style="background:${C.surface};padding:10px;border-radius:6px;text-align:center">
            <div style="font-size:8px;color:${C.dim};letter-spacing:2px">SHARPE</div>
            <div style="font-size:15px;font-weight:700;color:${C.accent}">${sharpe ? sharpe.toFixed(2) : '—'}</div>
          </div>
          <div style="background:${C.surface};padding:10px;border-radius:6px;text-align:center">
            <div style="font-size:8px;color:${C.dim};letter-spacing:2px">MAX DD</div>
            <div style="font-size:15px;font-weight:700;color:${(maxDrawdown||0)>0.15?C.red:C.yellow}">${maxDrawdown ? (maxDrawdown*100).toFixed(1)+'%' : '—'}</div>
          </div>
        </div>
      </div>` +

      // Today's trades
      (todayTrades && todayTrades.length > 0 ? section(`Today's Trades (${todayTrades.length})`, C.accent,
        todayTrades.map(t =>
          `<div style="display:grid;grid-template-columns:2fr 1fr 1fr 1fr;gap:4px;padding:7px 0;border-bottom:1px solid ${C.border}30">
            <span style="color:#fff;font-weight:700">${t.ticker}</span>
            <span style="${t.side==='BUY'?`color:${C.green}`:`color:${C.red}`}">${t.side}</span>
            <span style="color:${C.dim}">${t.shares}sh @ $${(t.price||0).toFixed(2)}</span>
            <span style="color:${t.pnl!=null?(t.pnl>=0?C.green:C.red):C.dim};text-align:right">${t.pnl!=null?(t.pnl>=0?'+':'')+'$'+t.pnl.toFixed(0) : ''}</span>
          </div>`
        ).join('')
      ) : '') +

      // Open positions
      section('Open Positions at Close', null,
        (positions||[]).length > 0
          ? (positions||[]).map(p => {
              const showPx = p.currentPrice != null && p.currentPrice > 0
              return `<div style="display:grid;grid-template-columns:2fr 1fr 1fr 1fr;gap:4px;padding:8px 0;border-bottom:1px solid ${C.border}30">
                <span style="color:#fff;font-weight:700">${p.ticker}</span>
                <span style="color:${C.dim}">${p.shares}sh</span>
                <span style="color:${C.dim}">${showPx?'$'+(p.currentPrice).toFixed(2):'—'}</span>
                <span style="color:${(p.pnl||0)>=0?C.green:C.red};font-weight:700;text-align:right">${(p.pnl||0)>=0?'+':''}$${(p.pnl||0).toFixed(0)}</span>
              </div>`
            }).join('') +
            `<div style="display:flex;gap:16px;margin-top:10px;font-size:10px">
              ${bestPosition ? `<span style="color:${C.green}">★ Best: ${bestPosition.ticker} +${(bestPosition.pnlPct||0).toFixed(1)}%</span>` : ''}
              ${worstPosition ? `<span style="color:${C.red}">▼ Worst: ${worstPosition.ticker} ${(worstPosition.pnlPct||0).toFixed(1)}%</span>` : ''}
            </div>`
          : `<div style="color:${C.dim}">No open positions</div>`
      ) +

      // Watchlist for tomorrow
      (nextDayWatchlist && nextDayWatchlist.length > 0 ? section('🔭 Watch Tomorrow at Open', C.purple,
        nextDayWatchlist.slice(0,6).map((s,i) => {
          const px = s.price != null && s.price > 0 ? s.price : null
          return `<div style="display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid ${C.border}30">
            <span style="color:#fff;font-weight:700">#${i+1} ${s.ticker}</span>
            ${px ? `<span style="color:${C.dim}">$${px.toFixed(2)}</span>` : ''}
            ${badge('BUY', C.green)}
            <span style="color:${C.accent}">Score ${(s.score||0).toFixed(3)}</span>
            <span style="color:${C.dim};font-size:9px">Conf ${Math.round((s.confidence||0)*100)}%</span>
          </div>`
        }).join('')
      ) : '')
    )
  }

  if (!subject) return res.status(400).json({ error: 'Unknown email type: ' + type })

  try {
    const r = await fetch('https://api.resend.com/emails', {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ from: 'STOCKBOT <alerts@resend.dev>', to, subject, html })
    })
    const d = await r.json()
    if (!r.ok) return res.status(r.status).json({ error: d.message||'Send failed' })
    return res.status(200).json({ success: true, id: d.id })
  } catch(e) {
    return res.status(500).json({ error: e.message })
  }
}
