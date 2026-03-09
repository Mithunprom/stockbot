---
name: autonomous-stockbot
description: >
  Use this skill whenever the user wants to build, run, improve, or reason about
  an autonomous trading bot. Triggers include: stock trading automation, algorithmic
  trading, paper trading, live trading, RL trading agent, market data ingestion,
  options flow analysis, TCN/Transformer models for finance, news sentiment trading,
  reinforcement learning for profit maximization, trading sub-agents, Sharpe ratio
  optimization, drawdown control, trade execution, or any request involving buy/sell
  automation. Also trigger when the user asks about FFSA, TCN, Transformer+Options Flow,
  or any ML-based market prediction pipeline. Use this skill even if the user only
  mentions one component (e.g. "help me set up news sentiment for my bot").
---

# Autonomous StockBot — Skill Guide

## Overview

This skill guides Claude in building and operating a fully autonomous trading bot.
The system ingests real-time market data, runs a multi-model signal ensemble
(FFSA, Transformer + Options Flow, TCN, News Sentiment), then feeds all signals
into a Reinforcement Learning (RL) agent that decides when to buy, sell, or hold
to maximize risk-adjusted profit.

The lifecycle is:

1. **Backtest** — validate strategy on historical data
2. **Paper Trade** — simulate in real time, no real money
3. **Live Trade** — deploy only after consistent paper-trading performance

A fleet of sub-agents runs alongside the main bot, continuously tuning latency,
model performance, risk parameters, and strategy drift.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DATA INGESTION LAYER                  │
│  Real-Time Prices │ Options Flow │ Order Book │ News API │
└────────────────────────────┬────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────┐
│                  FEATURE ENGINEERING                     │
│  Technical Indicators │ FFSA │ Sentiment Scores │ Vega   │
└────────────────────────────┬────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────┐
│               MULTI-MODEL SIGNAL ENSEMBLE                │
│   Transformer + Options Flow │ TCN │ News Sentiment NLP  │
└────────────────────────────┬────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────┐
│              RL TRADING AGENT (PPO / SAC)                │
│     State → Policy → Action (Buy / Sell / Hold / Size)  │
└────────────────────────────┬────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────┐
│              EXECUTION & RISK MANAGEMENT                 │
│   Order Router │ Position Sizing │ Stop-Loss │ Drawdown  │
└────────────────────────────┬────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────┐
│                SUB-AGENT IMPROVEMENT LOOP                │
│  Latency Agent │ Profit Agent │ Risk Agent │ Drift Agent │
└─────────────────────────────────────────────────────────┘
```

---

## 1. Data Ingestion Layer

### Real-Time Market Data

- **Price & Volume**: Connect to a broker WebSocket (Alpaca, Interactive Brokers,
  Polygon.io) for tick-level OHLCV data.
- **Order Book (L2/L3)**: Capture bid/ask depth to detect large order imbalances.
- **Options Flow**: Subscribe to unusual options activity feeds (e.g., Unusual Whales,
  Market Chameleon API, or Tradier options chain). Key fields: strike, expiry, premium,
  volume vs. open interest, put/call ratio, implied volatility (IV), delta, gamma, vega.
- **News & Filings**: Pull from NewsAPI, Benzinga, SEC EDGAR (for earnings/8-K filings),
  and Twitter/Reddit sentiment streams.

### Data Normalization

- Align all feeds to a unified timestamp (UTC).
- Resample tick data to multiple timeframes: 1m, 5m, 15m, 1h, 1d.
- Store raw data in a time-series database (InfluxDB or TimescaleDB).
- Maintain a rolling window in memory for live inference (default: 200 bars).

---

## 2. Feature Engineering

### Technical Indicators (baseline signals)

Compute the following for each timeframe:

- **Trend**: EMA(9), EMA(21), EMA(50), MACD, ADX
- **Momentum**: RSI(14), Stochastic(14,3), Williams %R, CCI
- **Volatility**: Bollinger Bands(20,2), ATR(14), Keltner Channels
- **Volume**: VWAP, OBV, Volume Profile, Money Flow Index (MFI)
- **Market Microstructure**: Bid-ask spread, order book imbalance ratio, trade
  flow toxicity (VPIN)

### FFSA — Financial Feature Significance Analysis

FFSA is a signal selection layer that scores each feature's predictive power before
feeding the model ensemble. This prevents noisy or redundant indicators from diluting
signal quality.

**How to implement FFSA:**

1. Compute a large feature matrix (50–200 features per bar).
2. Run a rolling importance score using one or more of:
   - **Mutual Information** between each feature and forward returns
   - **SHAP values** from a lightweight gradient-boosted tree (XGBoost/LightGBM)
   - **Variance Inflation Factor (VIF)** to drop highly collinear features
3. Rank features by importance score. Keep the top-K (e.g., top 30).
4. Re-run FFSA on a weekly schedule to adapt to regime changes.
5. Pass the selected feature subset to the downstream models.

This keeps the model ensemble lean and prevents overfitting — especially important
when market regimes shift.

---

## 3. Multi-Model Signal Ensemble

Each model outputs a **directional signal** (long, short, neutral) and a
**confidence score** (0–1). These are combined into a weighted ensemble signal
that feeds the RL agent's state vector.

### Model A — Transformer + Options Flow

The Transformer captures long-range temporal dependencies in price data while
incorporating options flow as an additional attention context.

**Architecture:**

- Input: FFSA-selected price/volume features + options flow features
  (IV rank, put/call ratio, unusual flow flag, gamma exposure / GEX)
- Positional encoding over the time dimension (sequence length: 60–120 bars)
- Multi-head self-attention (4–8 heads)
- Options flow is injected as a cross-attention key/value stream (separate encoder)
- Output head: 3-class softmax (up / down / flat) + confidence logit

**Why options flow matters:** Large options positioning often precedes big moves
("smart money" hedging or speculating). The model learns to weight directional
options bets alongside price action.

**Training tips:**

- Use focal loss to handle class imbalance (most bars are "flat").
- Augment with synthetic regime data (trending, mean-reverting, high-vol).
- Retrain weekly or when model drift is detected.

### Model B — TCN (Temporal Convolutional Network)

TCNs excel at capturing multi-scale temporal patterns with lower latency than
Transformers, making them ideal for shorter-term signals.

**Architecture:**

- Dilated causal convolutions (dilation factors: 1, 2, 4, 8, 16)
- Residual connections between blocks
- Input: FFSA-selected features at 1m and 5m resolution
- Output: regression of next-bar return + direction probability

**Why TCN alongside Transformer:**
The Transformer handles longer context (hours/days), the TCN focuses on
short-term momentum bursts. Together they cover different time horizons, making
the ensemble more robust across market conditions.

### Model C — News Sentiment NLP

Converts raw news text into a real-time sentiment signal using a fine-tuned
financial language model.

**Implementation:**

- Use `FinBERT` (pre-trained on financial text) or fine-tune `DistilBERT` on
  earnings call transcripts + financial news.
- For each article/headline: predict sentiment (positive/negative/neutral) +
  relevance score (is this about the ticker we're trading?).
- Aggregate into a rolling **Sentiment Index**:
  `SI(t) = weighted_avg(sentiment_score * relevance * recency_decay)`
- Spike detection: flag sudden sentiment reversals as high-conviction signals.
- Include social sentiment (StockTwits, Reddit r/wallstreetbets) with lower weight.

### Ensemble Combination

```
ensemble_signal = (
    w_transformer * transformer_confidence * transformer_direction +
    w_tcn         * tcn_confidence         * tcn_direction +
    w_sentiment   * sentiment_index
)
```

- Initial weights: `w_transformer=0.45, w_tcn=0.35, w_sentiment=0.20`
- Weights are re-optimized quarterly by the Profit Sub-Agent using Bayesian
  optimization over paper-trading performance.
- Direction encoding: long=+1, short=-1, neutral=0

---

## 4. Reinforcement Learning Trading Agent

The RL agent is the decision-maker. It observes the current market state and
outputs a trading action, learning over time which actions maximize cumulative
risk-adjusted profit.

### State Space

At each timestep, the agent observes:

```
state = [
  ensemble_signal,          # weighted model output
  transformer_confidence,   # per-model confidence
  tcn_confidence,
  sentiment_index,
  current_position,         # -1 (short), 0 (flat), +1 (long)
  unrealized_pnl,           # normalized
  time_in_trade,            # bars since entry
  portfolio_heat,           # % capital at risk
  vix_level,                # market-wide fear gauge
  regime_label,             # trending / mean-reverting / choppy (from HMM)
  recent_drawdown,          # max drawdown in last N bars
  # + top FFSA features (10–20 features)
]
```

### Action Space

```
actions = {
  0: hold,
  1: buy_small   (5% of portfolio),
  2: buy_medium  (10% of portfolio),
  3: buy_large   (20% of portfolio),
  4: sell_small  (close 25% of position),
  5: sell_medium (close 50% of position),
  6: sell_all    (close 100% of position),
  7: short_small (5% short),
  8: short_large (20% short),
}
```

Position sizing is baked into the action space to let the agent learn
risk management through experience.

### Reward Function

The reward function must balance profit-seeking with risk control:

```python
def reward(pnl, drawdown, trade_cost, holding_time):
    sharpe_component  = pnl / (rolling_volatility + 1e-8)
    drawdown_penalty  = -2.0 * max(drawdown - max_drawdown_threshold, 0)
    cost_penalty      = -trade_cost * 0.1
    time_decay        = -0.001 * holding_time  # discourage overholding
    return sharpe_component + drawdown_penalty + cost_penalty + time_decay
```

- Primary objective: maximize **Sharpe ratio**, not raw PnL (prevents reckless risk-taking)
- Drawdown threshold: default 5% — penalty ramps steeply beyond this
- Include transaction costs and slippage in every reward calculation

### RL Algorithm

Use **PPO (Proximal Policy Optimization)** as the default — it's stable,
well-tested, and handles continuous/mixed action spaces well.

For higher-frequency strategies, consider **SAC (Soft Actor-Critic)** which
maximizes entropy alongside reward (encourages exploration and robustness).

**Training setup:**

- Framework: `Stable-Baselines3` or `RLlib`
- Environment: custom `gym.Env` wrapping the backtester
- Training data: minimum 2 years of historical data
- Evaluation episodes: walk-forward validation (no lookahead)
- Checkpoint every 10,000 steps; keep the best checkpoint by Sharpe ratio

**Warm start:** Pre-train the policy using Behavioral Cloning on a rules-based
strategy (e.g., MACD crossover) before switching to full RL. This accelerates
convergence significantly.

---

## 5. Execution & Risk Management

### Order Execution

- Connect to broker via REST/WebSocket (Alpaca recommended for algo trading,
  Interactive Brokers for institutional features).
- Route market orders for entries/exits when signal confidence > threshold.
- Use **limit orders** within 0.1% of mid-price to reduce slippage.
- Enforce a **minimum hold time** (default: 3 bars) to prevent overtrading.

### Position Sizing

- Base sizing from RL action (5/10/20% of portfolio)
- Adjust for volatility: `size *= (target_vol / realized_vol)`
- Hard cap: never exceed 25% portfolio in any single trade
- Maintain a cash buffer: always keep ≥ 10% uninvested

### Risk Controls (hard limits, enforced outside the RL agent)

| Control            | Default Threshold            | Action                   |
| ------------------ | ---------------------------- | ------------------------ |
| Daily loss limit   | -3% portfolio                | Halt trading for the day |
| Max drawdown       | -8% portfolio                | Pause and alert          |
| Max position size  | 25% portfolio                | Reject oversized orders  |
| Consecutive losses | 5 in a row                   | Reduce size by 50%       |
| Volatility spike   | VIX > 35                     | Switch to cash-only mode |
| Earnings blackout  | 2 days before/after earnings | No new positions         |

---

## 6. Paper Trading → Live Trading Transition

### Paper Trading Phase

Run the full bot in simulation mode before committing real capital. Paper trading
should mirror live conditions exactly: use real-time data, simulate realistic
slippage (0.05–0.10%), and charge actual brokerage fees.

**Minimum paper trading period:** 3 months (or 500+ trades)

### Promotion Criteria (all must be met)

Before going live, the following benchmarks must be satisfied over the paper
trading period:

- **Sharpe Ratio** ≥ 1.5 (annualized)
- **Win Rate** ≥ 52%
- **Profit Factor** ≥ 1.4 (gross profit / gross loss)
- **Max Drawdown** ≤ 8%
- **Consistency**: Profitable in ≥ 75% of rolling 2-week windows
- **Sortino Ratio** ≥ 2.0
- **No single trade** accounts for > 30% of total profit (concentration risk)

### Gradual Live Rollout

1. **Week 1–2**: 10% of intended capital, monitor execution quality vs. paper
2. **Week 3–4**: 25% of capital if slippage/latency within 10% of paper trading
3. **Month 2**: 50% of capital if drawdown and Sharpe remain on target
4. **Month 3+**: Full capital deployment

---

## 7. Sub-Agent Improvement Loop

A set of specialized sub-agents runs continuously alongside the main bot,
each focused on a specific dimension of improvement. They operate on a schedule
and push updates to the main system.

### Sub-Agent 1 — Latency Agent

**Goal**: Minimize end-to-end signal-to-order latency.
**Runs**: Every hour
**Tasks**:

- Profile each pipeline stage (data fetch → feature compute → model inference → order submit)
- Flag any stage exceeding its SLA (target: < 100ms total)
- Recommend optimizations: vectorize features, cache model weights, batch API calls
- Report: latency histogram, p50/p95/p99 breakdown

### Sub-Agent 2 — Profit Agent

**Goal**: Continuously improve strategy profitability.
**Runs**: Daily
**Tasks**:

- Analyze last 24h of trades: winners vs. losers, by signal source
- Re-weight ensemble model contributions using Bayesian optimization
- Suggest RL reward function tweaks if Sharpe is declining
- Backtest proposed changes on last 30 days before applying
- Report: PnL attribution by model, trade duration analysis, best/worst setups

### Sub-Agent 3 — Risk Agent

**Goal**: Keep portfolio risk within defined limits.
**Runs**: Every 15 minutes during market hours
**Tasks**:

- Monitor live drawdown, VaR (Value at Risk), correlation between open positions
- Detect regime shifts (HMM-based market regime classifier)
- Auto-reduce position sizes if volatility spikes
- Trigger emergency halt if daily loss limit is breached
- Report: risk dashboard, drawdown waterfall, position heat map

### Sub-Agent 4 — Model Drift Agent

**Goal**: Detect when models are degrading and trigger retraining.
**Runs**: Weekly
**Tasks**:

- Compare last week's model accuracy vs. training-time accuracy
- Run PSI (Population Stability Index) on feature distributions
- Flag models with accuracy drop > 5% or PSI > 0.25 for retraining
- Queue and execute automated retraining pipeline
- Report: model performance scorecard, drift metrics per feature

### Sub-Agent 5 — Opportunity Agent (optional, advanced)

**Goal**: Discover new alpha sources and instruments.
**Runs**: Weekly
**Tasks**:

- Scan new tickers for signal quality using FFSA
- Test new features (e.g., new sentiment sources, alternative data)
- A/B test new model architectures in paper trading
- Report: ranked opportunity list with expected Sharpe improvement

---

## 8. Implementation Checklist

When helping the user build this system, work through these stages in order:

- [ ] **Stage 1 — Data Pipeline**: Set up broker connection, data normalization,
      time-series DB. Verify data quality (gaps, outliers, timestamp alignment).
- [ ] **Stage 2 — Feature Engineering**: Implement technical indicators + FFSA
      pipeline. Validate feature distributions on historical data.
- [ ] **Stage 3 — Model Training**: Train Transformer, TCN, and Sentiment models
      on historical data. Evaluate on held-out test set (no lookahead).
- [ ] **Stage 4 — Backtester**: Build the `gym.Env` wrapper. Run RL training with
      walk-forward validation. Record Sharpe, drawdown, trade stats.
- [ ] **Stage 5 — Risk Controls**: Implement all hard limits. Test circuit breakers
      with synthetic stress scenarios (flash crash, earnings surprise).
- [ ] **Stage 6 — Paper Trading**: Deploy in live simulation. Run for minimum
      3 months. Track all promotion criteria.
- [ ] **Stage 7 — Sub-Agents**: Activate sub-agents one by one. Start with the
      Risk Agent (critical) before the Profit and Drift agents.
- [ ] **Stage 8 — Live Rollout**: Follow the gradual capital deployment schedule.
      Never skip the partial-capital phase.

---

## 9. Key Libraries & Tools

| Purpose                  | Recommended Library                         |
| ------------------------ | ------------------------------------------- |
| Broker API               | `alpaca-trade-api`, `ib_insync`             |
| Real-time data           | `polygon-api-client`, `websockets`          |
| Time-series DB           | `influxdb-client`, `timescaledb`            |
| Technical indicators     | `ta-lib`, `pandas-ta`                       |
| Feature selection (FFSA) | `shap`, `sklearn`, `lightgbm`               |
| Transformer model        | `torch`, `transformers` (HuggingFace)       |
| TCN                      | `torch` (custom) or `keras-tcn`             |
| NLP Sentiment            | `FinBERT` via HuggingFace `pipeline`        |
| RL Training              | `stable-baselines3`, `gymnasium`            |
| Backtesting              | `backtrader`, `vectorbt`, or custom gym env |
| Monitoring               | `prometheus` + `grafana`, or `wandb`        |
| Scheduling (sub-agents)  | `APScheduler`, `celery`                     |

---

## 10. Important Warnings

- **Never skip paper trading.** Consistent paper trading profit is not a guarantee
  of live profit, but inconsistent paper trading is a guarantee of live losses.
- **RL agents can overfit to historical data.** Always use walk-forward validation.
  If the agent looks too good in backtest (Sharpe > 3), it is almost certainly
  overfitting.
- **Options flow signals are powerful but noisy.** Require at least 2 confirming
  signals before entering a trade based on options flow alone.
- **News sentiment can spike on false or irrelevant news.** Always filter by
  relevance score before acting on sentiment signals.
- **Sub-agents must not push changes to the live bot without human review.**
  All sub-agent recommendations should be staged in a config file for human
  approval before deployment.
- **Slippage and fees compound.** A strategy with Sharpe 2.0 before costs can
  easily become unprofitable after costs at high trade frequency.
