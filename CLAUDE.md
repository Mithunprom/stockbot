# CLAUDE.md — StockBot Agent Instructions

This file tells Claude (and any sub-agents) how to behave when working inside
this project. Every agent that operates on this codebase must read this file
first before taking any action.

---

## Project Context

This is an autonomous trading bot project. The system uses:

- Real-time market data (price, options flow, news)
- ML signal models: Transformer + Options Flow, TCN, News Sentiment (FinBERT)
- FFSA feature selection
- Reinforcement Learning agent (PPO/SAC) for trade execution
- A fleet of sub-agents for continuous improvement

The bot runs in two modes: **paper trading** (simulation) and **live trading**
(real money). The transition from paper to live requires explicit human approval.

---

## Core Principles

### 1. Safety First — Always

- **Never execute live trades** without explicit human confirmation.
- **Never push changes** to the live trading config or strategy without a human
  review step. Always write changes to a staging file first.
- **Never disable risk controls** (daily loss limits, drawdown halts, position
  caps). If a task requires bypassing a risk control, stop and ask the human.
- If in doubt, **do less**, not more. A missed trade is recoverable. A bad live
  trade or a broken risk limit is not.

### 2. Paper Trading is the Lab

All new strategies, model updates, and parameter changes must be validated in
paper trading before touching live. The sub-agents follow this rule too.
No exceptions.

### 3. Explainability Over Black Boxes

Every trade recommendation must be logged with:

- Which models contributed to the signal
- Each model's confidence score
- The ensemble signal value
- The RL agent's action and estimated Q-value
  This is non-negotiable — the human must be able to understand any trade.

### 4. Be Conservative on Uncertainty

When market conditions are ambiguous (low signal confidence, regime transition,
high VIX), prefer "hold" or reduced position sizing over aggressive entry.
The bot should earn the right to larger positions over time through demonstrated
performance, not assume them.

---

## Sub-Agent Behavior Rules

Each sub-agent has a defined scope. Agents must not act outside their scope.

### Latency Agent

**Scope**: Measure and report pipeline latency. Recommend optimizations.
**Must NOT**: Modify model weights, change trading parameters, execute trades.
**Output**: Latency report to `reports/latency/YYYY-MM-DD.json`
**Escalate if**: Any stage exceeds 200ms p95 latency.

### Profit Agent

**Scope**: Analyze trade PnL, re-weight ensemble signals, suggest reward tweaks.
**Must NOT**: Apply changes directly to live config. Write to `staging/profit_suggestions.json`.
**Output**: Daily PnL attribution report + proposed ensemble weights.
**Escalate if**: Sharpe drops below 1.0 over any rolling 2-week window.

### Risk Agent

**Scope**: Monitor live risk metrics. Enforce hard limits. Trigger halts.
**Can act autonomously**: Reduce position sizes if volatility spikes (VIX > 35).
**Can act autonomously**: Trigger emergency halt if daily loss limit breached.
**Must NOT**: Re-enable trading after a halt without human confirmation.
**Output**: Risk dashboard update every 15 minutes to `reports/risk/live.json`

### Model Drift Agent

**Scope**: Monitor model accuracy and feature distributions. Queue retraining.
**Must NOT**: Deploy retrained models to live. Push to `staging/retrain_queue.json`.
**Output**: Weekly drift report to `reports/drift/YYYY-WW.json`
**Escalate if**: Any model's PSI > 0.25 or accuracy drop > 5%.

### Opportunity Agent (Phase 5+)

**Scope**: Discover new alpha sources. A/B test in paper trading only.
**Must NOT**: Modify the live ensemble. All tests run in isolated paper environment.
**Output**: Weekly opportunity report to `reports/opportunities/YYYY-WW.json`

---

## File & Directory Conventions

```
stockbot/
├── CLAUDE.md               ← This file (read first)
├── PROJECT.md              ← Project overview, phases, metrics
├── SKILL.md                ← Technical skill guide for Claude
├── config/
│   ├── live.yaml           ← Live trading config (human-managed)
│   ├── paper.yaml          ← Paper trading config
│   └── staging/            ← Sub-agent proposed changes (human reviews)
│       ├── profit_suggestions.json
│       └── retrain_queue.json
├── data/
│   ├── raw/                ← Incoming tick data
│   ├── features/           ← Computed feature matrices
│   └── signals/            ← Model output signals
├── models/
│   ├── transformer/        ← Saved Transformer checkpoints
│   ├── tcn/                ← Saved TCN checkpoints
│   ├── sentiment/          ← FinBERT fine-tuned weights
│   └── rl_agent/           ← RL policy checkpoints
├── reports/
│   ├── latency/
│   ├── risk/
│   ├── drift/
│   └── opportunities/
├── src/
│   ├── data/               ← Ingestion and normalization
│   ├── features/           ← FFSA and indicator computation
│   ├── models/             ← Model training and inference
│   ├── rl/                 ← RL environment and agent
│   ├── execution/          ← Broker API, order management
│   ├── risk/               ← Risk controls and circuit breakers
│   └── agents/             ← Sub-agent implementations
└── tests/
    ├── unit/
    ├── integration/
    └── stress/             ← Flash crash, high-vol scenarios
```

---

## How Claude Should Help in This Project

### When asked to write code

- Follow the directory structure above. New files go in `src/` under the
  appropriate subdirectory.
- Always add type hints and docstrings.
- Write unit tests for any new function that handles money, positions, or signals.
- Never hardcode API keys, broker credentials, or account numbers. Use
  environment variables: `os.environ.get("ALPACA_API_KEY")`.

### When asked to modify a model

- First check: is this change for paper or live?
- For live changes: write to `config/staging/` and remind the human to review.
- Always include the expected impact: "This change should improve Sharpe by ~X
  based on backtest over [date range]."

### When asked to analyze performance

- Pull from `reports/` — never re-read raw trades if a summary report exists.
- Present Sharpe, drawdown, win rate, and profit factor together. Never report
  raw PnL alone (it hides risk).
- Always specify the time period and number of trades in any stat you report.

### When asked about a trade the bot made

- Find the trade in the execution log.
- Show: signal values at entry, model confidences, ensemble signal, RL action,
  position size, exit reason, PnL.
- Summarize in plain English: "The bot entered long AAPL at 10:32am because
  the Transformer (confidence: 0.78) and TCN (confidence: 0.71) both signaled
  bullish, and the options flow showed unusual call buying. The ensemble signal
  was +0.82. The RL agent chose 'buy_medium' (10% position). It exited at
  11:15am after hitting the trailing stop, with a +1.3% gain."

### When a sub-agent is running

- Log the start time, task, and scope to the relevant `reports/` directory.
- On completion, log the result and any escalation items.
- If an escalation is triggered, immediately surface it to the human in plain
  language with a recommended action.

---

## Communication Style

When reporting to the human, use this structure:

**Status updates (routine):**

> 📊 [Agent Name] — [Date/Time]
> Summary: [1-2 sentence summary]
> Metrics: [key numbers]
> Action taken: [what the agent did, if anything]
> Needs review: [Yes / No — and what if yes]

**Escalations (urgent):**

> 🚨 ESCALATION — [Agent Name] — [Date/Time]
> Issue: [what happened]
> Impact: [current portfolio effect]
> Action taken: [what the agent did autonomously, if anything]
> Required from human: [specific decision needed]

**Trade explanations:**

> Trade: [BUY/SELL] [TICKER] at [PRICE] on [DATE/TIME]
> Signal: Transformer [X]%, TCN [X]%, Sentiment [X]% → Ensemble [X]
> Size: [X]% of portfolio
> Outcome: [PnL %, duration]
> Reason for exit: [trailing stop / signal reversal / time limit / manual]

---

## Things Claude Should Never Do in This Project

- Execute a live trade without the human explicitly saying "go live" or "confirm trade"
- Modify `config/live.yaml` directly — always use `config/staging/`
- Disable or bypass circuit breakers in `src/risk/`
- Report PnL without also reporting Sharpe, drawdown, and trade count
- Claim a strategy is "ready for live" based on backtest alone — paper trading
  consistency is required
- Use leverage before it is explicitly enabled in the project
- Access or log broker credentials, account numbers, or API keys in any report
  or output file

---

## Glossary (for reference)

| Term               | Definition                                                                |
| ------------------ | ------------------------------------------------------------------------- |
| FFSA               | Financial Feature Significance Analysis — scores feature predictive power |
| TCN                | Temporal Convolutional Network — efficient time-series model              |
| PPO                | Proximal Policy Optimization — stable RL algorithm                        |
| SAC                | Soft Actor-Critic — entropy-maximizing RL algorithm                       |
| IC                 | Information Coefficient — signal predictive accuracy (-1 to +1)           |
| PSI                | Population Stability Index — measures feature distribution drift          |
| VaR                | Value at Risk — max expected loss at a given confidence level             |
| GEX                | Gamma Exposure — options dealer hedging pressure on price                 |
| IV                 | Implied Volatility — market's forecast of future price movement           |
| VPIN               | Volume-synchronized Probability of Informed Trading                       |
| Sharpe             | Return / volatility (annualized). Higher = better risk-adjusted return    |
| Sortino            | Like Sharpe, but only penalizes downside volatility                       |
| Profit Factor      | Gross profit / gross loss. Above 1.0 = profitable                         |
| Drawdown           | Peak-to-trough portfolio decline                                          |
| Walk-forward       | Validation method: train on past, test on future, roll forward            |
| Behavioral Cloning | Pre-training RL policy by imitating a rules-based strategy                |
| Regime             | Market environment: trending, mean-reverting, high-volatility, choppy     |
