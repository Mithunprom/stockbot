# Stockbot — Project Documentation

## Objective
An autonomous intraday paper trading system that:
1. Screens S&P 500 for top 20-40 stocks via multi-factor momentum scoring
2. Generates buy/sell signals via a 5-dimension rules-based engine (Pipeline B)
3. Executes paper trades on Alpaca with ATR-adaptive exits
4. Monitors performance via sub-agents (risk, latency, profit, drift)
5. Targets transition to live trading after meeting Sharpe/drawdown criteria

**Current mode:** Paper trading, Pipeline B standalone (since 2026-04-19)

---

## Architecture

```
stockbot/
├── main.py                    # FastAPI entry point, lifespan, API routes
├── src/
│   ├── config.py              # Settings (pydantic-settings, .env)
│   ├── data/
│   │   ├── db.py              # TimescaleDB schema (ohlcv_1m, feature_matrix, trades)
│   │   ├── alpaca_ws.py       # AlpacaDataStreamClient + RestBarPoller
│   │   ├── news.py            # NewsPoller (Polygon.io + Benzinga)
│   │   ├── options_flow.py    # Options flow poller (yfinance chains)
│   │   ├── fundamentals.py    # FundamentalsCache (yfinance P/E, earnings)
│   │   ├── market_regime.py   # MarketRegimeMonitor (VIX + SPY/QQQ)
│   │   └── social_stocktwits.py # Reddit social sentiment feed
│   ├── features/
│   │   ├── indicators.py      # Full pandas-ta indicator pipeline
│   │   ├── live.py            # LiveFeatureComputer (incremental on each bar)
│   │   ├── ffsa.py            # LightGBM + SHAP feature selection
│   │   ├── regime.py          # Regime classification + gate thresholds
│   │   └── psi.py             # Population Stability Index (drift)
│   ├── models/
│   │   ├── pipeline_b.py      # PipelineBEngine (rules-based signal — ACTIVE)
│   │   ├── lgbm.py            # LGBMSignalModel (inactive)
│   │   ├── transformer.py     # TransformerSignalModel (inactive)
│   │   ├── tcn.py             # TCNSignalModel (inactive)
│   │   ├── sentiment.py       # FinBERT via HuggingFace API
│   │   └── ensemble.py        # EnsembleEngine + EnsembleSignal dataclass
│   ├── execution/
│   │   ├── alpaca.py          # AlpacaOrderRouter (paper/live)
│   │   ├── position_manager.py # PositionManager (tracking, heat, sync)
│   │   └── position_sizer.py  # SmartPositionSizer (Kelly-inspired)
│   ├── risk/
│   │   └── circuit_breakers.py # 7 hard circuit breakers
│   └── agents/
│       ├── signal_loop.py     # SignalLoop (base class — execution, exits, ATR)
│       ├── signal_loop_b.py   # SignalLoopB (Pipeline B — overrides _tick())
│       ├── scheduler.py       # APScheduler for sub-agents
│       ├── risk_agent.py      # Every 15 min
│       ├── latency_agent.py   # Hourly
│       ├── profit_agent.py    # Daily 16:30 ET
│       ├── screener_agent.py  # Nightly 18:00 ET
│       ├── drift_agent.py     # Weekly
│       ├── live_ic_tracker.py # Prediction vs actual tracking
│       ├── critique_agent.py  # Trade critique
│       ├── retrain_agent.py   # Retrain scheduler
│       └── quant_research_agent.md # Research spec
├── scripts/
│   ├── test_pipeline_b_steps.py  # Step-by-step Pipeline B test
│   ├── test_signal_scan.py       # Full universe signal scanner
│   ├── test_trade_roundtrip.py   # Buy/sell execution test
│   ├── test_pipeline_health.py   # End-to-end health check
│   ├── daily_performance.py      # Performance plots (A vs B)
│   ├── ab_report.py              # A/B comparison report
│   ├── train_lgbm.py             # Train LightGBM
│   ├── train_models.py           # Train Transformer/TCN
│   ├── train_rl.py               # Train RL agent
│   ├── run_ffsa.py               # FFSA feature selection
│   ├── build_features.py         # Rebuild feature matrix
│   └── backfill.py               # Backfill OHLCV bars
├── config/
│   ├── universe.json          # Active trading universe (screener-managed)
│   ├── paper.yaml             # Paper trading config
│   ├── live.yaml              # Live config (human-managed, never auto-modified)
│   └── staging/               # Sub-agent proposed changes
├── models/                    # Saved model checkpoints (S3 synced)
├── reports/                   # Sub-agent output (risk, latency, drift, performance)
├── docs/
│   ├── project.md             # This file
│   ├── strategy.md            # Trading strategy & system design
│   ├── pipeline_b.md          # Pipeline B detailed documentation
│   ├── model_improvement_brief.md # ML model analysis
│   └── skills.md              # Role & expertise profile
├── frontend/                  # React dashboard (npm run dev → :5173)
├── CLAUDE.md                  # Claude Code instructions
└── SKILL.md                   # Technical skill guide
```

---

## Modules

### Signal Engine — Pipeline B (`src/models/pipeline_b.py`)

The active signal engine. Scores each ticker across 5 dimensions:

```
ensemble = 0.30 × technicals + 0.25 × fundamentals + 0.20 × regime
         + 0.20 × sentiment + 0.05 × social
```

See [docs/pipeline_b.md](pipeline_b.md) for full scoring rules and thresholds.

---

### Screener Agent (`src/agents/screener_agent.py`)

Runs nightly at 18:00 ET to rotate the trading universe.

**Scoring:**
```
composite = 0.35 × 5d_return + 0.35 × 20d_return
          + 0.20 × volume_surge + 0.10 × RS_vs_SPY
```

**Filters:** Avg daily volume > 500k, price $5-$2000, max 40 stocks.
**Anchor tickers:** Always included (AAPL, MSFT, NVDA, AMZN, GOOGL, TSLA, etc.)
**Output:** `config/universe.json` (hot-swapped without restart)

---

### Feature Engineering (`src/features/indicators.py`)

All indicators computed from raw OHLCV via pure pandas/numpy:

| Category | Indicators |
|---|---|
| Trend | EMA(9/21/50), MACD(12/26/9), ADX(14), Parabolic SAR |
| Momentum | RSI(14), Stochastic(14), Williams %R |
| Volatility | ATR(14), Bollinger Bands(20), Keltner Channel, Donchian |
| Volume | OBV, MFI(14), Volume ratio, VPIN |
| Price | VWAP, CCI(14), supply/demand zones |
| Multi-TF | Multi-timeframe confluence, divergence detection |
| Composite | orb_vpin_interact, vwap_time_interact, atr_vol_interact |

All indicators shifted by 1 bar (no lookahead bias).

**FFSA:** LightGBM + SHAP selects top 30 features. Current IC = 0.1385.

---

### Risk Management (`src/risk/circuit_breakers.py`)

Seven hard circuit breakers (never disabled):

| Breaker | Threshold | Action |
|---|---|---|
| Daily loss limit | -3% portfolio | Halt trading for the day |
| Max drawdown | -8% from peak | Pause, alert human |
| Position size | >25% single position | Reject order |
| Portfolio heat | >80% deployed | No new entries |
| VIX spike | VIX > 35 | Go to cash |
| Consecutive losses | 5 in a row | Reduce position size 50% |
| Flash crash | >20% drop in 1 bar | Halt, alert human |

**Resume after halt:** requires explicit human confirmation — never automatic.

---

## Data Sources (All Free Tier)

| Source | Data | Frequency |
|--------|------|-----------|
| Alpaca (free) | 1-min OHLCV bars (IEX), paper execution | REST poll every 60s |
| Polygon.io (free) | News articles, daily screener bars | Poll every 5 min |
| yfinance (free) | VIX, P/E, earnings, revenue, options chains | Daily cache + 1-min regime |
| HuggingFace API (free) | FinBERT sentiment inference | Per article |
| Reddit public JSON | Social sentiment (WSB, stocks, investing) | Poll every 5 min |

---

## Environment Setup

```bash
# Backend (from stockbot/)
cp .env.example .env
# Required: ALPACA_API_KEY, ALPACA_SECRET_KEY, DATABASE_URL, POLYGON_API_KEY
uvicorn main:app --port 8000

# Frontend (from stockbot/frontend/)
npm install
npm run dev    # → http://localhost:5173 (proxied to :8000)

# PostgreSQL must be running:
brew services start postgresql@14
```

---

## Deployment (Railway)

```bash
# Link to Railway project
railway link

# Set env vars
railway variables set ACTIVE_PIPELINE=b
railway variables set ENABLE_ALPACA_WS=false

# Deploy (auto-deploys on git push to main)
git push origin main

# Check status
railway logs
```

**Live URL:** `https://stockbot-production-cbde.up.railway.app`

---

## Roadmap

| Phase | Feature | Status |
|-------|---------|--------|
| 1 | Data pipeline (Alpaca bars, TimescaleDB) | Done |
| 2 | Feature engineering (30 FFSA features) | Done |
| 3 | ML signal models (LightGBM, Transformer, TCN) | Done (inactive — IC~0 live) |
| 4 | Paper trading (Pipeline A) | Done (underperformed) |
| 5 | Pipeline B (rules-based signal engine) | Done — ACTIVE |
| 6 | A/B testing framework (Pipeline A vs B) | Done |
| 7 | ATR-adaptive exits (per-ticker volatility) | Done |
| 8 | Sub-agents (risk, latency, profit, drift, screener) | Done |
| 9 | Daily performance plots | Done |
| - | Validate live IC > 0.05 (Pipeline B) | In progress |
| - | 3 months paper trading meeting Sharpe >= 1.5 | Pending |
| - | Live broker integration | Pending (requires human approval) |
