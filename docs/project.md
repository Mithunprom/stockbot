# Stockbot — Project Documentation

## Objective
An institutional-grade algorithmic trading dashboard that:
1. Screens the S&P 500 universe for the top 20 stocks using multi-factor alpha scoring
2. Generates buy/sell/hold signals via a 7-factor signal engine
3. Backtests strategies with zero lookahead bias and realistic transaction costs
4. Trains a Reinforcement Learning agent to optimize factor weights per market regime
5. Supports paper trading and (roadmap) live execution via Alpaca

---

## Architecture

```
stockbot/
├── src/
│   ├── data/
│   │   ├── screener.js        # S&P 500 screener — grouped daily bars
│   │   ├── polygonClient.js   # Polygon.io API client (rate-limited, cached)
│   │   ├── universe.js        # Trading universe management
│   │   ├── verifier.js        # Stock verification before universe entry
│   │   ├── persistence.js     # localStorage helpers
│   │   ├── news.js            # News sentiment engine
│   │   ├── priceGate.js       # Rate-limiting gate for price fetches
│   │   ├── stockPriceSeeds.js # Synthetic data for mock mode
│   │   └── cryptoPrices.js    # Crypto price feed
│   ├── signals/
│   │   └── signals.js         # 7-factor signal engine
│   ├── backtest/
│   │   └── backtester.js      # Vectorized backtester
│   ├── rl/
│   │   ├── agent.js           # CEM + Q-learning RL agent
│   │   └── trainer.js         # Training loop orchestrator
│   ├── risk/
│   │   └── kelly.js           # Kelly criterion position sizing
│   ├── models/
│   │   ├── ensemble.js        # Ensemble model
│   │   ├── famaFrench.js      # Fama-French factor proxy
│   │   └── temporalCNN.js     # Temporal CNN multi-scale alignment
│   ├── trading/
│   │   ├── alpaca.js          # Alpaca broker integration
│   │   ├── autoTrader.js      # Automated execution
│   │   └── exitStrategy.js    # Exit strategy (trailing stop)
│   ├── ui/components/         # React UI components
│   ├── App.jsx                # Main dashboard (5 screens)
│   └── main.jsx               # Vite entry point
├── api/
│   ├── email.js               # Email notification handler
│   └── send-email.js          # Vercel serverless email endpoint
├── CLAUDE.md                  # Claude Code instructions
├── docs/
│   ├── project.md             # This file
│   └── skills.md              # Role & expertise profile
├── .env.example               # Environment variable template
├── vercel.json                # Vercel deployment config
└── vite.config.js             # Vite build config
```

---

## Modules

### Screener (`src/data/screener.js`)
Screens ~250 S&P 500 components using Polygon.io grouped daily bars.

**API Strategy:**
- `GET /v2/aggs/grouped/locale/us/market/stocks/{date}` returns ALL US stocks in one call
- Fetches last 5 trading days on first run (5 API calls, spaced 13s for rate limiting)
- `groupedDayCache` is in-memory with no TTL — daily bars are immutable, so re-screens cost 0 API calls

**Scoring:**
```
composite = dayReturn × 0.6 × adjMomentum
          + weekReturn × 0.4 × adjMomentum
          + volSurge × volumeSurge_weight
          + trendScore × trendBreak_weight
          + newsScore × 0.20
```

**Screening Profiles:**

| Profile | Momentum | Volume Surge | Trend Break | RSI Oversold |
|---------|----------|--------------|-------------|--------------|
| MOMENTUM | 0.50 | 0.20 | 0.20 | 0.00 |
| BREAKOUT | 0.20 | 0.40 | 0.30 | 0.00 |
| MEAN REVERSION | 0.00 | 0.10 | -0.30 | 0.60 |
| HIGH VOLATILITY | 0.20 | 0.30 | 0.10 | 0.00 |
| BALANCED | 0.25 | 0.25 | 0.25 | 0.00 |

---

### Signal Engine (`src/signals/signals.js`)
Generates composite BUY/SELL/HOLD signals from 7 factors.

**Factors:**

| Factor | Description | Default Weight |
|--------|-------------|----------------|
| Momentum | 1-month + 3-month return, MACD confirmation | 0.28 |
| Mean Reversion | RSI + Bollinger Bands + Stochastic | 0.12 |
| Volume | Volume ratio × price direction + OBV trend | 0.18 |
| Volatility | ATR + vol ratio (low vol = bullish for momentum) | 0.12 |
| Trend | Price vs VWAP/MA20/MA50 + EMA golden cross | 0.20 |
| FF Alpha | Fama-French alpha proxy (residual return + RMW) | 0.08 |
| TCN Align | Multi-scale MA alignment (5/10/20/50-day) | 0.08 |

**Technical Indicators computed:**
- RSI (14-period)
- MACD (12/26/9)
- Bollinger Bands (20-period, 2σ)
- Stochastic Oscillator (14/3)
- ATR (14-period)
- OBV (On-Balance Volume)

**Signal thresholds:**
- `score > 0.15` → BUY
- `score < -0.15` → SELL
- otherwise → HOLD

---

### Backtester (`src/backtest/backtester.js`)
Vectorized backtester with institutional-grade accuracy.

**Key properties:**
- Zero lookahead bias — `vectorizeSignals()` uses fixed lookback window sliced at each bar
- Transaction costs: 0.05% per side
- Kelly criterion position sizing (half-Kelly, 25% cap per asset)
- 7% dynamic trailing stop loss

**Reward function:**
```
Reward = 0.6 × Sharpe Ratio + 0.4 × Calmar Ratio
```

---

### RL Agent (`src/rl/agent.js`)
Cross-Entropy Method + Q-learning hybrid, regime-aware.

**Algorithm:**
1. Sample candidate factor weights (exploit elite population or explore randomly)
2. Backtest the weights on historical OHLCV data
3. Record episode score (Sharpe + Calmar reward)
4. Update elite population (top 25% of episodes)
5. Derive new weights as weighted mean of elites
6. Adapt epsilon (exploration rate) based on improvement trend

**Regime detection:**
```
regime = bull/bear × lowvol/medvol/highvol × trending/ranging/declining
```
The agent maintains per-regime populations and best weights, enabling specialization.

**Persistence:** `localStorage` key `stockbot_rl_v3` — survives page refreshes.

---

### Risk Management (`src/risk/kelly.js`)
- Half-Kelly position sizing based on signal strength and confidence
- Maximum 25% of portfolio per single asset
- 7% trailing stop loss (dynamic)

---

## Dashboard Screens

| Screen | Description |
|--------|-------------|
| **Dashboard** | Signal heatmap, top 20 screened stocks, live RL factor weights |
| **Signals** | Per-asset deep-dive: factor decomposition, price chart, indicators |
| **Backtest** | Equity curve, max drawdown, per-asset performance table |
| **Train** | RL agent controls, live weight visualization, episode log, regime stats |
| **Paper** | Simulated trade execution log with P&L tracking |

---

## Data Sources

| Source | Usage | Notes |
|--------|-------|-------|
| Polygon.io (free) | Grouped daily bars, per-ticker OHLCV | 15-min delayed, 5 req/min |
| Polygon.io (paid) | WebSocket real-time prices | Roadmap |
| Anthropic API | News sentiment | Optional / Roadmap |
| Alpaca Markets | Paper + live order execution | Roadmap |

---

## Environment Setup

```bash
cp .env.example .env.local
# Edit .env.local:
# VITE_POLYGON_API_KEY=your_key
# VITE_ANTHROPIC_API_KEY=your_key  (optional)

npm install
npm run dev       # http://localhost:3000
```

### Mock Mode
If `VITE_POLYGON_API_KEY` is not set, the app runs entirely on synthetic price data. All features (signals, backtest, RL training) work in mock mode.

---

## Deployment

```bash
# One-time setup
npm install -g vercel
vercel link

# Deploy
vercel --prod

# Set env vars in Vercel dashboard:
# VITE_POLYGON_API_KEY
# VITE_ANTHROPIC_API_KEY (optional)
```

---

## Roadmap

| Phase | Feature | Status |
|-------|---------|--------|
| 1 | Historical data pipeline + technical indicators | Done |
| 2 | Multi-factor signal engine | Done |
| 3 | Vectorized backtester | Done |
| 4 | CEM + Q-learning RL agent | Done |
| 5 | S&P 500 screener (grouped bars) | Done |
| 6 | Paper trading dashboard | Done |
| 7 | WebSocket real-time prices | Roadmap |
| 8 | News sentiment (Anthropic API) | Roadmap |
| 9 | Black-Litterman portfolio optimization | Roadmap |
| 10 | Live broker integration (Alpaca) | Roadmap |
| 11 | Multi-timeframe signals (1h + 1d) | Roadmap |
| 12 | Walk-forward optimization | Roadmap |
