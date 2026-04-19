# StockBot — Trading Strategy & System Design

**System Design Diagram (FigJam):**
https://www.figma.com/online-whiteboard/create-diagram/b2b80e6d-9cb3-4d58-a262-fc133dd6035b

---

## Overview

StockBot is an autonomous paper-to-live trading system. It combines three ML signal
models with a Reinforcement Learning execution agent, hard risk controls, and a fleet
of sub-agents that continuously improve the strategy. All changes go through paper
trading before touching live capital.

---

## 1. Universe Selection

**How stocks are chosen:**

| Layer | Description |
|---|---|
| Nightly screener | Scans all S&P 500 stocks after market close (18:00 ET) |
| Momentum score | `0.35 × 5d return + 0.35 × 20d return + 0.20 × volume surge + 0.10 × RS vs SPY` |
| Liquidity filter | Avg daily volume > 500k, price $5–$2000 |
| Anchor tickers | Always included: AAPL, MSFT, NVDA, AMZN, GOOGL, TSLA, AVGO, AMD, SNDK, JPM, V, MA, PLTR, ARM, MSTR, XOM, CVX |
| Excluded tickers | META (permanent exclusion) |
| Max universe | 40 stocks |
| Hot-swap | Universe updates nightly without server restart |

Defense stocks (LMT, RTX, NOC, GD, BA) rotate in naturally when their momentum
qualifies — they are not hardcoded.

---

## 2. Data Pipeline

### 2a. Price Data
- **Source:** Alpaca REST bar poller (free tier, polls every 60s)
  - WebSocket disabled on Railway (1-connection-per-account limit breaks rolling deploys)
- **Granularity:** 1-minute OHLCV bars
- **Storage:** TimescaleDB `ohlcv_1m` table
- **Backfill:** Last 5 minutes fetched per poll cycle; 300 bars on startup

### 2b. News & Sentiment
- **Source:** NewsAPI + Benzinga (polled every 5 minutes)
- **Scoring:** FinBERT via Hugging Face Inference API (`ProsusAI/finbert`)
- **Output:** `sentiment_score ∈ [-1, +1]` per article, stored in `news_raw` table
- **Relevance:** Ticker mention frequency in headline + body

### 2c. Options Flow
- **Source:** yfinance options chain (free, polled every 5 minutes)
- **Metrics computed:**
  - Put/Call ratio
  - Volume/OI ratio (unusual flow flag if > 5×)
  - Net GEX (gamma exposure approximation)
  - Smart money score (ITM volume directional bias)
  - IV rank (put IV vs call IV skew)
- **Storage:** `options_flow` table

---

## 3. Feature Engineering

**FFSA (Financial Feature Significance Analysis)**
- LightGBM + SHAP selects the 30 most predictive features
- Current IC (Information Coefficient): 0.0265
- Features recalculated nightly; top-30 list stored in `config/ffsa_features.json`

**Computed indicators (pure pandas/numpy, no external TA library):**

| Category | Indicators |
|---|---|
| Trend | EMA(9/21/50), MACD(12/26/9), ADX(14) |
| Momentum | RSI(14), Stochastic(14), Williams %R |
| Volatility | ATR(14), Bollinger Bands(20), Keltner Channel |
| Volume | OBV, MFI(14), Volume ratio |
| Price | VWAP, CCI(14) |

All indicators shifted by 1 bar (no lookahead bias).

---

## 4. Signal Generation

### Two Pipelines

StockBot supports two signal pipelines. **Pipeline B is currently active
(standalone mode since 2026-04-19).** Pipeline A is disabled pending live IC
validation.

| Aspect | Pipeline A (ML) | Pipeline B (Rules) — ACTIVE |
|--------|----------------|----------------------------|
| Signal source | LightGBM + Transformer + TCN + FinBERT | Technical rules + fundamentals + regime + sentiment + social |
| Training needed | Yes | No |
| Live performance | IC~0, Sharpe -3.5 (failed) | TBD |
| Config | `ACTIVE_PIPELINE=a` | `ACTIVE_PIPELINE=b` |

> Full Pipeline B documentation: **[docs/pipeline_b.md](pipeline_b.md)**

### 4a. Pipeline A — ML Ensemble (INACTIVE)

```
ensemble_signal = 0.60 × LightGBM + 0.10 × Transformer + 0.10 × TCN + 0.20 × Sentiment
```

- LightGBM: val IC=0.11, dir_acc=52% (primary signal)
- Transformer: val IC~0 (effectively random — disabled)
- TCN: val IC~0 (effectively random — disabled)
- FinBERT sentiment: 20% weight

**Post-mortem (2026-03-27):** 155 trades, win rate 32%, profit factor 0.348,
Sharpe -3.5. Signal strength showed zero conditional predictive power. See
`src/agents/quant_research_agent.md` for full analysis.

### 4b. Pipeline B — Rules-Based Ensemble (ACTIVE)

```
ensemble = 0.30 × technicals + 0.25 × fundamentals + 0.20 × regime
         + 0.20 × sentiment + 0.05 × social
```

**Technical (30%):** RSI, MACD, Bollinger %B, VWAP deviation, ADX, MFI,
Stochastic, OBV, multi-timeframe confluence, divergence, candlestick patterns,
supply/demand zones, Parabolic SAR, Donchian breakout.

**Fundamental (25%):** P/E ratio, forward P/E compression, earnings surprise,
YoY revenue growth (via yfinance, daily cache).

**Market Regime (20%):** VIX level + percentile, SPY/QQQ 5-bar and 15-bar
momentum, QQQ-SPY risk-on spread.

**Sentiment (20%):** FinBERT (ProsusAI/finbert) via HuggingFace Inference API.
Rolling sentiment index, recency-weighted (12h half-life).

**Social (5%):** Reddit finance subreddits (r/wallstreetbets, r/stocks,
r/investing, r/options). Keyword heuristic + upvote weighting.

### Signal Strength Classification
| Range | Strength |
|---|---|
| \|signal\| >= 0.60 | Strong |
| \|signal\| >= 0.40 | Moderate |
| \|signal\| >= 0.20 | Weak |
| \|signal\| < 0.20 | Flat (no trade) |

**Entry threshold:** \|ensemble_signal\| >= 0.40 (moderate or stronger)

---

## 5. Trade Execution

### 5a. Position Sizing (SmartPositionSizer)

Kelly-inspired conviction buckets based on `dir_prob`:

| dir_prob | Max cap | Description |
|----------|---------|-------------|
| [0.55, 0.65) | 2% | Low conviction |
| [0.65, 0.80) | 4% | Medium conviction |
| [0.80, 1.00] | 6% | High conviction |

Hard caps: max 25% single position, max 80% portfolio heat, min 10% cash buffer.

Entry gate requires `|pred_return| > 0.0015` (cost threshold) AND `dir_prob`
outside [0.45, 0.55] dead zone.

### 5b. ATR-Adaptive Exits (since 2026-04-16)

Stop-loss, trailing stop, and take-profit scale per-ticker by 1-minute ATR:

```
stop_loss     = max(ATR% × 15, 0.5%)
trailing_stop = max(ATR% × 20, 0.8%)
take_profit   = max(ATR% × 35, 1.5%)
max_hold      = 45 bars (45 minutes)
```

Volatile stocks (NVDA: 1.35% stop) get wider stops than low-vol stocks
(AAPL: 0.6% stop), preventing noise-triggered exits.

### 5c. Order Type
- Market orders for paper trading (immediate fill)
- Fill polling: checks every second for up to 30 seconds
- Shorts: disabled (paper account needs margin approval)

### 5d. RL Agent (INACTIVE)
- PPO position-sizing agent trained (Sharpe 18.4 in simulation)
- Replaced by SmartPositionSizer for now (simpler, more transparent)
- Re-enable when live IC > 0.05 validated

---

## 6. Risk Management

Seven hard circuit breakers (never disabled):

| Breaker | Threshold | Action |
|---|---|---|
| Daily loss limit | -3% portfolio | Halt trading for the day |
| Max drawdown | -8% from peak | Pause, alert human |
| Position size | >25% single position | Reject order |
| Portfolio heat | >80% deployed | No new entries |
| VIX spike | VIX > 35 | Go to cash |
| Consecutive losses | 5 in a row | Reduce position size 50% |
| Flash crash | Price drop >20% in 1 bar | Halt, alert human |

**Resume after halt:** requires explicit human confirmation — never automatic.

---

## 7. Sub-Agents

| Agent | Frequency | Scope |
|---|---|---|
| **Risk Agent** | Every 15 min (market hours) | Monitors all 7 circuit breakers, writes `reports/risk/live.json` |
| **Latency Agent** | Hourly (market hours) | Measures pipeline latency per stage, escalates if p95 > 200ms |
| **Profit Agent** | Daily 16:30 ET | PnL attribution, proposes new ensemble weights to `config/staging/` |
| **Screener Agent** | Nightly 18:00 ET | Rotates universe based on S&P 500 momentum |
| **Model Drift Agent** | Weekly (planned) | Monitors PSI and accuracy, queues retraining |

All sub-agent changes go to `config/staging/` — never directly to live config.

---

## 8. Infrastructure

| Component | Technology |
|---|---|
| Server | Railway Cloud (FastAPI + uvicorn) |
| Database | TimescaleDB (PostgreSQL) on Railway |
| Model storage | AWS S3 (`stockbot-models` bucket) |
| FinBERT inference | Hugging Face Inference API |
| Real-time data | Alpaca IEX WebSocket (free tier) |
| Scheduling | APScheduler (AsyncIO) |
| Frontend | React dashboard → WebSocket `/ws/dashboard` |

---

## 9. API Endpoints

| Endpoint | Description |
|---|---|
| `GET /health` | Liveness check (shows active_pipeline: a/b/ab) |
| `GET /signals` | Latest ensemble signals for all tickers |
| `GET /signals/actionable` | Signals that pass entry gate with trade/no-trade flags |
| `GET /trades` | Trade history with full attribution |
| `GET /positions/detail` | Open positions with ATR stops, unrealized P&L |
| `GET /portfolio/summary` | Portfolio diagnostics (ATR multipliers, thresholds, positions) |
| `GET /diagnostics` | Full pipeline diagnostics (A and/or B) |
| `GET /status` | System status (models, weights, agents, FFSA, positions) |
| `GET /ab/status` | A/B test comparison (when both pipelines active) |
| `GET /reports/{name}` | Latest sub-agent report (risk, latency, drift) |
| `POST /admin/resume-trading` | Resume after circuit breaker halt |
| `WS /ws/dashboard` | Real-time push: signals, positions, PnL, risk |

**Live URL:** `https://stockbot-production-cbde.up.railway.app`

---

## 10. Paper-to-Live Transition Criteria

The bot moves from paper to live only when ALL of these are met:

- [ ] 3 months of paper trading
- [ ] Rolling Sharpe ≥ 1.5 (2-week window)
- [ ] Max drawdown ≤ 8% in paper
- [ ] Win rate ≥ 45%
- [ ] Profit factor ≥ 1.3
- [ ] RL agent Sharpe ≥ 1.0 (retrained on real paper data)
- [ ] Model drift PSI < 0.25 on all features
- [ ] Explicit human approval ("go live" command)

---

## 11. Training Schedule

| Model | When | Trigger |
|---|---|---|
| Transformer / TCN | Overnight (pre-market) | Weekly or PSI > 0.25 |
| RL Agent | Overnight (pre-market) | ≥500 paper trades available |
| FFSA re-ranking | Monthly | Feature distribution shift |
| Ensemble weights | Daily (Profit Agent) | Proposes, human approves |

---

## Glossary

| Term | Definition |
|---|---|
| FFSA | Financial Feature Significance Analysis — SHAP-based feature selection |
| TCN | Temporal Convolutional Network — efficient time-series model |
| PPO | Proximal Policy Optimization — RL algorithm |
| IC | Information Coefficient — signal predictive accuracy (-1 to +1) |
| PSI | Population Stability Index — feature distribution drift measure |
| GEX | Gamma Exposure — options dealer hedging pressure |
| SI | Sentiment Index — FinBERT rolling weighted score |
| RS | Relative Strength — stock return vs SPY benchmark |
| Sharpe | Return / volatility (annualized). Target ≥ 1.5 for live |
| Portfolio Heat | % of capital currently deployed in positions |
