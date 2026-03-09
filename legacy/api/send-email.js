/**
 * Vercel Serverless Function — /api/send-email
 * Sends portfolio alerts and daily briefings via email.
 * Uses nodemailer with Gmail SMTP (free).
 */

const nodemailer = require('nodemailer')

module.exports = async (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*')
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS')
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type')
  if (req.method === 'OPTIONS') return res.status(200).end()
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' })

  const { type, to, subject, data } = req.body

  if (!to || !subject) return res.status(400).json({ error: 'Missing required fields' })

  const transporter = nodemailer.createTransport({
    service: 'gmail',
    auth: {
      user: process.env.EMAIL_USER,
      pass: process.env.EMAIL_PASS, // Gmail App Password
    },
  })

  let html = ''

  if (type === 'portfolio_alert') {
    const { portfolioValue, pnl, pnlPct, positions, topMovers } = data
    const isUp = pnlPct >= 0
    html = `
<!DOCTYPE html>
<html>
<body style="background:#050510;color:#e2e8f0;font-family:monospace;padding:24px;max-width:600px;margin:0 auto">
  <div style="border:1px solid ${isUp?'#00ff88':'#ff3366'};border-radius:8px;padding:24px;margin-bottom:16px">
    <h1 style="color:${isUp?'#00ff88':'#ff3366'};font-size:24px;margin:0 0 8px">
      ${isUp?'📈':'📉'} Portfolio ${isUp?'UP':'DOWN'} ${Math.abs(pnlPct).toFixed(1)}%
    </h1>
    <p style="color:#64748b;font-size:12px;margin:0">STOCKBOT ALERT · ${new Date().toLocaleString()}</p>
  </div>

  <div style="display:grid;gap:12px;margin-bottom:16px">
    <div style="background:#0f0f22;border:1px solid #1a1a3a;border-radius:6px;padding:16px">
      <div style="font-size:10px;color:#64748b;letter-spacing:3px;margin-bottom:8px">PORTFOLIO VALUE</div>
      <div style="font-size:28px;font-weight:700;color:#ffffff">$${portfolioValue.toLocaleString()}</div>
      <div style="font-size:14px;color:${isUp?'#00ff88':'#ff3366'};margin-top:4px">
        ${isUp?'+':''}$${pnl.toFixed(0)} (${isUp?'+':''}${pnlPct.toFixed(2)}%)
      </div>
    </div>
  </div>

  <div style="background:#0f0f22;border:1px solid #1a1a3a;border-radius:6px;padding:16px;margin-bottom:16px">
    <div style="font-size:10px;color:#64748b;letter-spacing:3px;margin-bottom:12px">OPEN POSITIONS</div>
    ${positions.map(p => `
      <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #1a1a3a20">
        <span style="font-weight:700;color:#ffffff">${p.ticker}</span>
        <span style="color:#64748b">${p.shares} shares @ $${p.avgPrice.toFixed(2)}</span>
        <span style="color:${p.unrealPnl>=0?'#00ff88':'#ff3366'};font-weight:700">${p.unrealPnl>=0?'+':''}$${p.unrealPnl.toFixed(0)}</span>
      </div>
    `).join('')}
  </div>

  ${topMovers?.length ? `
  <div style="background:#0f0f22;border:1px solid #1a1a3a;border-radius:6px;padding:16px">
    <div style="font-size:10px;color:#64748b;letter-spacing:3px;margin-bottom:12px">TOP MOVERS TODAY</div>
    ${topMovers.map(m => `
      <div style="padding:6px 0;border-bottom:1px solid #1a1a3a20">
        <span style="color:#00d4ff;font-weight:700">${m.ticker}</span>
        <span style="color:${m.change>=0?'#00ff88':'#ff3366'};margin-left:12px">${m.change>=0?'+':''}${m.change.toFixed(2)}%</span>
        <span style="color:#64748b;margin-left:8px;font-size:10px">${m.signal}</span>
      </div>
    `).join('')}
  </div>` : ''}

  <p style="color:#64748b;font-size:10px;margin-top:24px;text-align:center">
    STOCKBOT · Automated Alert · <a href="${process.env.VERCEL_URL||'https://stockbot.vercel.app'}" style="color:#00d4ff">Open Dashboard</a>
  </p>
</body>
</html>`
  }

  if (type === 'daily_briefing') {
    const { buys, sells, marketSentiment, topPicks, date } = data
    html = `
<!DOCTYPE html>
<html>
<body style="background:#050510;color:#e2e8f0;font-family:monospace;padding:24px;max-width:600px;margin:0 auto">
  <div style="border:1px solid #00d4ff;border-radius:8px;padding:24px;margin-bottom:16px">
    <h1 style="color:#00d4ff;font-size:20px;margin:0 0 4px">☀️ STOCKBOT DAILY BRIEFING</h1>
    <p style="color:#64748b;font-size:11px;margin:0">${date} · Pre-market analysis</p>
  </div>

  <div style="background:#0f0f22;border:1px solid ${marketSentiment==='BULLISH'?'#00ff88':marketSentiment==='BEARISH'?'#ff3366':'#1a1a3a'};border-radius:6px;padding:16px;margin-bottom:16px">
    <div style="font-size:10px;color:#64748b;letter-spacing:3px;margin-bottom:8px">MARKET SENTIMENT</div>
    <div style="font-size:22px;font-weight:700;color:${marketSentiment==='BULLISH'?'#00ff88':marketSentiment==='BEARISH'?'#ff3366':'#e2e8f0'}">${marketSentiment}</div>
  </div>

  <div style="background:#0f0f22;border:1px solid #1a1a3a;border-radius:6px;padding:16px;margin-bottom:16px">
    <div style="font-size:10px;color:#00ff88;letter-spacing:3px;margin-bottom:12px">🚀 TOP BUY SIGNALS TODAY (${buys.length})</div>
    ${buys.slice(0,8).map((s,i) => `
      <div style="display:flex;justify-content:space-between;align-items:center;padding:10px 0;border-bottom:1px solid #1a1a3a40">
        <div>
          <span style="font-weight:700;color:#ffffff;font-size:13px">#${i+1} ${s.ticker}</span>
          <span style="color:#64748b;font-size:10px;margin-left:8px">${s.reason||''}</span>
        </div>
        <div style="text-align:right">
          <div style="color:#00ff88;font-weight:700">Score: ${s.score.toFixed(3)}</div>
          <div style="color:#64748b;font-size:10px">Conf: ${(s.confidence*100).toFixed(0)}%</div>
        </div>
      </div>
    `).join('')}
  </div>

  ${sells.length>0?`
  <div style="background:#0f0f22;border:1px solid #1a1a3a;border-radius:6px;padding:16px;margin-bottom:16px">
    <div style="font-size:10px;color:#ff3366;letter-spacing:3px;margin-bottom:12px">⚠️ SELL / AVOID (${sells.length})</div>
    ${sells.slice(0,5).map(s => `
      <div style="padding:8px 0;border-bottom:1px solid #1a1a3a40">
        <span style="font-weight:700;color:#ff3366">${s.ticker}</span>
        <span style="color:#64748b;font-size:10px;margin-left:8px">Score: ${s.score.toFixed(3)}</span>
      </div>
    `).join('')}
  </div>`:''}

  ${topPicks?.length?`
  <div style="background:#0a0a2a;border:1px solid #a855f7;border-radius:6px;padding:16px;margin-bottom:16px">
    <div style="font-size:10px;color:#a855f7;letter-spacing:3px;margin-bottom:12px">⭐ HIGH CONVICTION PICKS (Ensemble Confirmed)</div>
    ${topPicks.map(p => `
      <div style="padding:10px 0;border-bottom:1px solid #1a1a3a40">
        <div style="display:flex;justify-content:space-between">
          <span style="font-weight:700;color:#ffffff;font-size:13px">${p.ticker}</span>
          <span style="color:#a855f7">Models agree: ${p.modelAgreement}/4</span>
        </div>
        <div style="font-size:10px;color:#64748b;margin-top:4px">${p.reason}</div>
      </div>
    `).join('')}
  </div>`:''}

  <p style="color:#64748b;font-size:10px;margin-top:24px;text-align:center">
    ⚠️ Not financial advice. Always do your own research.<br>
    STOCKBOT · <a href="${process.env.VERCEL_URL||'https://stockbot.vercel.app'}" style="color:#00d4ff">Open Dashboard</a>
  </p>
</body>
</html>`
  }

  try {
    await transporter.sendMail({
      from: `STOCKBOT <${process.env.EMAIL_USER}>`,
      to,
      subject,
      html,
    })
    res.status(200).json({ success: true })
  } catch (err) {
    console.error('Email error:', err)
    res.status(500).json({ error: err.message })
  }
}
