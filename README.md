# STOCKBOT 🤖📈

Institutional-grade algorithmic trading system with reinforcement learning strategy optimization.

## Features

- **Real Market Data** — Polygon.io integration (15-min delayed on free tier)
- **Multi-Factor Signals** — Momentum, Mean-Reversion, Volume, Volatility, Trend
- **Vectorized Backtester** — Zero lookahead bias, realistic transaction costs, Kelly sizing
- **RL Agent** — Cross-Entropy Method learns optimal factor weights via risk-adjusted growth reward
- **Paper Trading** — Test signals without real money
- **One-Click Deploy** — Vercel ready

## Architecture

```
stockbot/
├── src/
│   ├── data/
│   │   └── polygonClient.js     # Polygon.io API client (rate-limited, cached)
│   ├── signals/
│   │   └── signals.js           # 5-factor signal generation engine
│   ├── backtest/
│   │   └── backtester.js        # Vectorized backtester (single + portfolio)
│   ├── rl/
│   │   ├── agent.js             # CEM + Q-learning RL agent (persistent)
│   │   └── trainer.js           # Training loop orchestrator
│   ├── risk/
│   │   └── kelly.js             # Kelly criterion position sizing
│   ├── App.jsx                  # Main React dashboard (5 screens)
│   └── main.jsx                 # Entry point
├── index.html
├── vite.config.js
├── vercel.json
└── .env.example
```

## Quickstart

### 1. Create GitHub repo

```bash
git clone https://github.com/YOUR_USERNAME/stockbot
cd stockbot
# Copy all these files in
```

Or create from scratch:
```bash
mkdir stockbot && cd stockbot
git init
git remote add origin https://github.com/YOUR_USERNAME/stockbot.git
```

### 2. Install dependencies

```bash
npm install
```

### 3. Set up environment variables

```bash
cp .env.example .env.local
```

Edit `.env.local`:
```
VITE_POLYGON_API_KEY=your_key_from_polygon.io
VITE_ANTHROPIC_API_KEY=your_anthropic_key  # optional, for sentiment
```

> **Free Polygon.io key:** Sign up at https://polygon.io — free tier gives 15-min delayed data, 5 req/min.

### 4. Run locally

```bash
npm run dev
# Open http://localhost:3000
```

### 5. Deploy to Vercel

```bash
npm install -g vercel
vercel
```

Or connect your GitHub repo on vercel.com → New Project → Import → set env vars.

**Vercel env vars to set:**
- `VITE_POLYGON_API_KEY` → your Polygon.io key
- `VITE_ANTHROPIC_API_KEY` → your Anthropic key (optional)

## How the RL Loop Works

1. **Agent samples** candidate factor weights (explore or exploit elite population)
2. **Backtester evaluates** weights on historical OHLCV data
3. **Reward** = Risk-Adjusted Growth = `0.6 × Sharpe + 0.4 × Calmar`
4. **Agent records** episode, updates elite population (CEM)
5. **Weights converge** toward what actually worked in real market data
6. **Regime detection** lets agent specialize weights per market regime (bull/bear × vol)

The agent persists across page refreshes via `localStorage`.

## Signal Factors

| Factor | Description | Source |
|--------|-------------|--------|
| Momentum | 12-1 month price return | Jegadeesh & Titman (1993) |
| Mean Reversion | RSI-based contrarian | Lehmann (1990) |
| Volume | Unusual volume × price direction | Blume, Easley & O'Hara (1994) |
| Volatility | Low-vol regime = risk-on | Ang et al. (2006) |
| Trend | Price vs VWAP + 20-day MA | Classic technical |

## Risk Management

- **Kelly Criterion**: Half-Kelly position sizing based on signal strength
- **Max Position**: 25% cap per single asset
- **Dynamic Stop**: 7% trailing stop loss
- **Transaction Cost**: 0.05% per side (institutional estimate)

## Dashboard Screens

- **Dashboard** — Signal heatmap, top buys, live RL weights
- **Signals** — Per-asset deep-dive, factor decomposition, price chart
- **Backtest** — Equity curve, drawdown, per-asset performance table
- **Train** — RL agent control, live weight visualization, episode log
- **Paper** — Simulated trade execution log

## Mock Mode

If no `VITE_POLYGON_API_KEY` is set, the app runs in mock mode with synthetic price data. All features work — signals, backtesting, RL training — just on simulated data.

## Roadmap

- [ ] WebSocket real-time prices (Polygon.io Advanced tier)
- [ ] News sentiment via Anthropic API
- [ ] Portfolio optimization (Black-Litterman)
- [ ] Live broker integration (Alpaca)
- [ ] Multi-timeframe signals (1h + 1d)
- [ ] Walk-forward optimization
