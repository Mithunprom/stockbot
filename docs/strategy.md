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
- **Source:** Alpaca IEX WebSocket (free, real-time)
- **Granularity:** 1-minute OHLCV bars
- **Storage:** TimescaleDB `ohlcv_1m` table
- **Backfill:** Last 300 bars fetched on startup

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

## 4. Signal Generation — Ensemble Model

```
ensemble_signal = 0.45 × (transformer_conf × transformer_dir)
               + 0.35 × (tcn_conf × tcn_dir)
               + 0.20 × sentiment_index

ensemble_signal ∈ [-1, +1]
```

### 4a. Transformer Model
- Architecture: 4-head attention, 3 layers, cross-attention on options flow
- Input: 60-bar sequence × 30 features (1m resolution)
- Output: direction ∈ {-1, 0, +1}, confidence ∈ [0, 1]
- Best checkpoint: Sharpe 0.896, val_accuracy 37.5%
- Stored on: AWS S3 → downloaded to Railway at startup

### 4b. TCN (Temporal Convolutional Network)
- Architecture: Dual-stream (1m + 5m bars), dilated causal convolutions
- Input: 1m and 5m feature sequences
- Output: direction + confidence
- Best checkpoint: Sharpe 0.776, val_accuracy 36.9%
- Stored on: AWS S3 → downloaded to Railway at startup

### 4c. FinBERT Sentiment
- Model: `ProsusAI/finbert` (financial domain FinBERT)
- Inference: Hugging Face Inference API (no local GPU needed)
- Rolling Sentiment Index: recency-decayed (half-life 12h) × relevance-weighted
- Contributes 20% to ensemble signal

### 4d. Rule-Based Fallback
Used only when ML models unavailable:
- RSI(14) < 35 AND MACD bullish crossover → +0.50
- RSI(14) > 65 AND MACD bearish crossover → -0.50

### Signal Strength Classification
| Range | Strength |
|---|---|
| \|signal\| ≥ 0.60 | Strong |
| \|signal\| ≥ 0.40 | Moderate |
| \|signal\| ≥ 0.20 | Weak |
| \|signal\| < 0.20 | Flat (no trade) |

**Entry threshold:** \|ensemble_signal\| ≥ 0.40 (moderate or stronger)

---

## 5. Trade Execution

### 5a. RL Agent (PPO)
- Algorithm: Proximal Policy Optimization (Stable-Baselines3)
- State space: 27-dimensional observation (ensemble signal, position, FFSA features, options flow)
- Action space: 9 discrete actions (hold, buy small/medium/large, sell 25/50/100%, short small/large)
- Current status: 500k steps trained, Sharpe -9.7 (needs 2M+ steps to converge)
- Retraining trigger: ≥500 paper trades logged + rolling Sharpe ≥ 0.5

### 5b. Threshold Fallback (active while RL matures)
- Buy: `ensemble_signal ≥ 0.40` and no open position
- Sell: position open and `ensemble_signal < -0.20`

### 5c. Position Sizing
- Base size: 5% of portfolio per position
- Scaling: volatility-scaled (ATR-based, targets 1% daily vol per position)
- Max single position: 25% of portfolio
- Max portfolio heat: 80% deployed

### 5d. Order Type
- Limit orders only, 0.1% above/below mid-price
- Fill polling: checks every second for up to 60 seconds
- Shorts: disabled (paper account needs margin approval)

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
| `GET /health` | Liveness check |
| `GET /signals` | Latest ensemble signals for all tickers |
| `GET /trades` | Trade history with full attribution |
| `GET /status` | Full system status (models, weights, agents, FFSA, positions) |
| `GET /reports/{name}` | Latest sub-agent report (risk, latency, drift) |
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
