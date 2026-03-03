// Vercel serverless — /api/email
// Uses Resend (free 3k/mo). Set RESEND_API_KEY in Vercel env vars.
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

  let subject='', html=''

  if(type==='portfolio_alert') {
    const { portfolioValue, pnl, pnlPct, positions, direction } = data
    const isUp = direction==='up'
    const color = isUp?'#00ff88':'#ff3366'
    subject = `${isUp?'📈':'📉'} Stockbot: Portfolio ${isUp?'UP':'DOWN'} ${Math.abs(pnlPct).toFixed(1)}%`
    html = `<div style="background:#050510;color:#e2e8f0;font-family:monospace;padding:24px;max-width:600px;margin:0 auto">
<div style="border:2px solid ${color};border-radius:8px;padding:20px;margin-bottom:16px">
<h2 style="color:${color};margin:0">${isUp?'📈':'📉'} Portfolio Alert</h2>
<div style="font-size:32px;font-weight:700;color:#fff;margin:12px 0">$${portfolioValue.toLocaleString()}</div>
<div style="color:${color};font-size:18px">${isUp?'+':''}$${pnl.toFixed(0)} (${isUp?'+':''}${pnlPct.toFixed(2)}%)</div>
</div>
<div style="background:#0f0f22;border:1px solid #1a1a3a;border-radius:6px;padding:16px;margin-bottom:16px">
<div style="color:#64748b;font-size:11px;letter-spacing:3px;margin-bottom:12px">OPEN POSITIONS</div>
${(positions||[]).map(p=>`<div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #1a1a3a">
  <span style="color:#fff;font-weight:700">${p.ticker}</span>
  <span style="color:#64748b">${p.shares} @ $${p.avgPrice?.toFixed(2)}</span>
  <span style="color:${p.pnl>=0?'#00ff88':'#ff3366'};font-weight:700">${p.pnl>=0?'+':''}$${p.pnl?.toFixed(0)}</span>
</div>`).join('')}
</div>
<p style="color:#64748b;font-size:10px;text-align:center">STOCKBOT · ${new Date().toLocaleString()}</p>
</div>`
  }

  if(type==='daily_briefing') {
    const { buys, sells, topPicks, marketMood, date } = data
    const moodColor = marketMood==='BULLISH'?'#00ff88':marketMood==='BEARISH'?'#ff3366':'#ffd700'
    subject = `☀️ Stockbot Daily: ${(topPicks||buys||[]).slice(0,3).map(s=>s.ticker).join(', ')} look strong today`
    html = `<div style="background:#050510;color:#e2e8f0;font-family:monospace;padding:24px;max-width:600px;margin:0 auto">
<div style="border:2px solid #00d4ff;border-radius:8px;padding:20px;margin-bottom:16px">
<h2 style="color:#00d4ff;margin:0">☀️ Daily Market Briefing</h2>
<div style="color:#64748b;font-size:11px;margin-top:4px">${date}</div>
</div>
<div style="background:#0f0f22;border:1px solid ${moodColor};border-radius:6px;padding:16px;margin-bottom:16px">
<div style="color:#64748b;font-size:10px;letter-spacing:3px;margin-bottom:8px">MARKET MOOD</div>
<div style="font-size:22px;font-weight:700;color:${moodColor}">${marketMood}</div>
</div>
${(topPicks||[]).length>0?`<div style="background:#0a0a2a;border:1px solid #a855f7;border-radius:6px;padding:16px;margin-bottom:16px">
<div style="color:#a855f7;font-size:10px;letter-spacing:3px;margin-bottom:12px">⭐ HIGH CONVICTION PICKS</div>
${topPicks.slice(0,6).map((s,i)=>`<div style="padding:10px 0;border-bottom:1px solid #1a1a3a40">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <span style="color:#fff;font-weight:700;font-size:14px">#${i+1} ${s.ticker}</span>
    <span style="color:#00ff88;font-weight:700">Score: ${s.score?.toFixed(3)||'—'}</span>
  </div>
  <div style="color:#64748b;font-size:10px;margin-top:4px">${s.reason||''} · Confidence: ${s.confidence?Math.round(s.confidence*100)+'%':'—'}</div>
</div>`).join('')}
</div>`:''}
${(buys||[]).length>0?`<div style="background:#0f0f22;border:1px solid #1a1a3a;border-radius:6px;padding:16px;margin-bottom:16px">
<div style="color:#00ff88;font-size:10px;letter-spacing:3px;margin-bottom:12px">🚀 BUY SIGNALS (${buys.length})</div>
${buys.slice(0,8).map(s=>`<div style="padding:8px 0;border-bottom:1px solid #1a1a3a40">
  <span style="color:#fff;font-weight:700">${s.ticker}</span>
  <span style="color:#64748b;font-size:10px;margin-left:10px">Score: ${s.score?.toFixed(3)||'—'} · ${s.reason||''}</span>
</div>`).join('')}
</div>`:''}
${(sells||[]).length>0?`<div style="background:#0f0f22;border:1px solid #ff3366;border-radius:6px;padding:16px;margin-bottom:16px">
<div style="color:#ff3366;font-size:10px;letter-spacing:3px;margin-bottom:12px">⚠️ AVOID / SELL (${sells.length})</div>
${sells.slice(0,5).map(s=>`<div style="padding:6px 0;border-bottom:1px solid #1a1a3a40">
  <span style="color:#ff3366;font-weight:700">${s.ticker}</span>
  <span style="color:#64748b;font-size:10px;margin-left:10px">${s.reason||''}</span>
</div>`).join('')}
</div>`:''}
<p style="color:#64748b;font-size:10px;text-align:center;margin-top:16px">⚠️ Not financial advice · STOCKBOT · <a href="https://stockbot-six-woad.vercel.app" style="color:#00d4ff">Open Dashboard</a></p>
</div>`
  }

  try {
    const r = await fetch('https://api.resend.com/emails', {
      method:'POST',
      headers:{ 'Authorization':`Bearer ${apiKey}`, 'Content-Type':'application/json' },
      body: JSON.stringify({ from:'STOCKBOT <alerts@resend.dev>', to, subject, html })
    })
    const d = await r.json()
    if(!r.ok) return res.status(r.status).json({error:d.message||'Send failed'})
    return res.status(200).json({success:true, id:d.id})
  } catch(e) {
    return res.status(500).json({error:e.message})
  }
}
