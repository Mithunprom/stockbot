# StockBot — Project Overview

## Vision

Build a fully autonomous, self-improving trading bot that ingests real-time
market data, applies a multi-model signal ensemble, and uses Reinforcement
Learning to execute trades with the goal of maximizing risk-adjusted profit.
The system graduates from backtesting → paper trading → live trading only
when consistent, measurable performance benchmarks are met. A dedicated fleet
of sub-agents continuously improves the system's performance, risk controls,
and model quality without human intervention.

---

## Goals

### Primary Goal

Deploy a live autonomous trading bot capable of generating consistent
risk-adjusted returns (target: annualized Sharpe ≥ 1.5, max drawdown ≤ 8%).

### Secondary Goals

- Build a modular, extensible system where new signal sources and models can
  be added without re-architecting the core.
- Create a self-improving loop: the bot and its sub-agents should get
  measurably better every week without manual tuning.
- Maintain full observability: every trade decision must be explainable and
  logged with contributing signal sources.

---

## Project Phases

### Phase 1 — Foundation

**Goal**: Get data flowing and features computing reliably.

| Task                                            | Owner | Status  |
| ----------------------------------------------- | ----- | ------- |
| Set up broker API connection (Alpaca/IBKR)      | Dev   | ⬜ Todo |
| Build real-time OHLCV + options flow pipeline   | Dev   | ⬜ Todo |
| Connect news/sentiment API (NewsAPI, Benzinga)  | Dev   | ⬜ Todo |
| Implement technical indicators (50+ features)   | Dev   | ⬜ Todo |
| Build FFSA feature selection pipeline           | Dev   | ⬜ Todo |
| Set up InfluxDB / TimescaleDB for storage       | Dev   | ⬜ Todo |
| Data quality checks (gaps, outliers, alignment) | Dev   | ⬜ Todo |

**Exit Criteria**: Clean, real-time feature matrix updating every bar with
< 50ms latency. All data sources validated on 1 year of history.

---

### Phase 2 — Model Development

**Goal**: Train all three signal models and validate signal quality.

| Task                                   | Owner | Status  |
| -------------------------------------- | ----- | ------- |
| Train Transformer + Options Flow model | ML    | ⬜ Todo |
| Train TCN model (1m + 5m resolution)   | ML    | ⬜ Todo |
| Fine-tune FinBERT for news sentiment   | ML    | ⬜ Todo |
| Build ensemble combination layer       | ML    | ⬜ Todo |
| Validate signals on held-out test set  | ML    | ⬜ Todo |
| Build signal quality dashboard         | ML    | ⬜ Todo |

**Exit Criteria**: Ensemble signal has positive IC (Information Coefficient > 0.05)
on held-out data. No lookahead bias confirmed.

---

### Phase 3 — RL Agent & Backtesting

**Goal**: Train the RL agent and validate strategy via rigorous backtesting.

| Task                                              | Owner | Status  |
| ------------------------------------------------- | ----- | ------- |
| Build custom `gym.Env` backtesting environment    | Dev   | ⬜ Todo |
| Pre-train policy via Behavioral Cloning           | ML    | ⬜ Todo |
| Train PPO agent (2+ years historical data)        | ML    | ⬜ Todo |
| Walk-forward validation (no lookahead)            | ML    | ⬜ Todo |
| Stress test: flash crash, high-vol regimes        | QA    | ⬜ Todo |
| Implement all risk hard limits / circuit breakers | Dev   | ⬜ Todo |
| Achieve backtest Sharpe ≥ 1.5, drawdown ≤ 8%      | ML    | ⬜ Todo |

**Exit Criteria**: Walk-forward backtest meets promotion criteria. All circuit
breakers tested and verified working.

---

### Phase 4 — Paper Trading

**Goal**: Run in live simulation for minimum 3 months. Prove consistency.

| Task                                             | Owner | Status  |
| ------------------------------------------------ | ----- | ------- |
| Deploy bot in paper trading mode                 | Dev   | ⬜ Todo |
| Activate Risk Agent sub-agent                    | Dev   | ⬜ Todo |
| Activate Latency Agent sub-agent                 | Dev   | ⬜ Todo |
| Set up live monitoring dashboard (Grafana/W&B)   | Dev   | ⬜ Todo |
| Weekly performance review vs. promotion criteria | Team  | ⬜ Todo |
| Activate Profit Agent after 4 weeks              | Dev   | ⬜ Todo |
| Activate Model Drift Agent after 4 weeks         | Dev   | ⬜ Todo |
| Document slippage vs. paper trading estimates    | QA    | ⬜ Todo |

**Promotion Criteria (all required before Phase 5):**

- [ ] Sharpe Ratio ≥ 1.5 (annualized, over 3 months)
- [ ] Win Rate ≥ 52%
- [ ] Profit Factor ≥ 1.4
- [ ] Max Drawdown ≤ 8%
- [ ] Profitable in ≥ 75% of rolling 2-week windows
- [ ] Sortino Ratio ≥ 2.0
- [ ] No single trade > 30% of total profit

---

### Phase 5 — Live Trading Rollout (Month 7+)

**Goal**: Graduated capital deployment with continuous sub-agent monitoring.

| Task                               | Capital   | Duration      | Status  |
| ---------------------------------- | --------- | ------------- | ------- |
| Deploy at 10% capital              | $X \* 10% | 2 weeks       | ⬜ Todo |
| Review execution quality vs. paper | —         | End of week 2 | ⬜ Todo |
| Scale to 25% capital               | $X \* 25% | 2 weeks       | ⬜ Todo |
| Scale to 50% capital               | $X \* 50% | Month 2       | ⬜ Todo |
| Full capital deployment            | $X        | Month 3+      | ⬜ Todo |
| Activate Opportunity Agent         | —         | Month 3       | ⬜ Todo |

**Go / No-Go at each scale-up**: Sharpe and drawdown must remain within
10% of paper trading benchmarks. Any breach triggers a hold at current level.

---

## Key Performance Metrics

Track these metrics weekly across all phases:

| Metric                      | Target  | Alert Threshold |
| --------------------------- | ------- | --------------- |
| Sharpe Ratio (annualized)   | ≥ 1.5   | < 1.0           |
| Sortino Ratio (annualized)  | ≥ 2.0   | < 1.2           |
| Max Drawdown                | ≤ 8%    | > 10%           |
| Win Rate                    | ≥ 52%   | < 48%           |
| Profit Factor               | ≥ 1.4   | < 1.1           |
| Daily Loss Limit            | ≤ 3%    | > 2.5%          |
| Signal-to-Order Latency     | < 100ms | > 200ms         |
| Model Drift (PSI)           | < 0.1   | > 0.25          |
| % Profitable 2-week windows | ≥ 75%   | < 60%           |

---

## Risk Management Policy

- **Hard stop — daily loss limit**: If portfolio drops 3% in a single day, halt
  trading for the rest of the day. Requires manual review before resuming.
- **Hard stop — drawdown**: If portfolio drops 8% from peak, pause all trading.
  Do not resume until root cause is identified.
- **Emergency halt**: Any sub-agent can trigger a full halt. Resuming requires
  human confirmation.
- **Leverage**: No leverage in Phase 4 (paper) or Phase 5 initial rollout.
  Re-evaluate after 6 months of live performance.
- **Concentration**: No single position > 25% of portfolio at any time.
- **Sub-agent changes**: All sub-agent recommendations are staged for human
  approval. No automated pushes to the live bot.

---

## Tech Stack Summary

| Layer               | Technology                                        |
| ------------------- | ------------------------------------------------- |
| Broker / Execution  | Alpaca (paper + live), IBKR (optional)            |
| Real-time data      | Polygon.io WebSocket, Unusual Whales              |
| Storage             | TimescaleDB (time-series), PostgreSQL (trades)    |
| Feature engineering | Python, pandas-ta, TA-Lib, SHAP                   |
| ML Models           | PyTorch (Transformer, TCN), HuggingFace (FinBERT) |
| RL Training         | Stable-Baselines3, Gymnasium                      |
| Backtesting         | vectorbt / custom gym.Env                         |
| Scheduling          | APScheduler (sub-agents)                          |
| Monitoring          | Grafana + Prometheus, W&B (model tracking)        |
| Deployment          | Docker, Linux server or cloud VM                  |

---

## Team & Responsibilities

| Role           | Responsibilities                                         |
| -------------- | -------------------------------------------------------- |
| Lead Developer | Architecture, data pipeline, broker integration          |
| ML Engineer    | Model training, FFSA, RL agent, retraining pipeline      |
| QA / Risk      | Backtesting validation, stress testing, circuit breakers |
| Operations     | Sub-agent scheduling, monitoring, alerting               |

_(Assign roles as the team grows. Initially, one person may cover multiple roles.)_

---

## Open Questions

- Which broker for live trading? (Alpaca has simple API; IBKR has more instruments)
- What instruments? (US equities only, or include ETFs, crypto, futures?)
- What is the starting capital for live trading?
- Is short selling allowed, or long-only?
- What is the target trade frequency? (intraday vs. swing trading)
- Are there any regulatory constraints (PDT rule for accounts < $25k)?

---

## Changelog

| Date | Version | Notes                |
| ---- | ------- | -------------------- |
| —    | v0.1    | Initial project plan |
