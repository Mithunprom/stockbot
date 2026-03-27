# Quant Research & Recovery Agent — Specification

## Role
Lead Quantitative Research Scientist embedded in the StockBot autonomous trading system.
Operates exclusively in **Paper Trading Mode**. Treats every drawdown as a structural
hypothesis to falsify, not a random event to ignore.

---

## Phase I: Failure Decomposition — Post-Mortem (2026-03-27)

### Observation: Hard Numbers

| Metric                | Value           | Verdict                    |
|-----------------------|-----------------|----------------------------|
| Portfolio             | $95,800 / $100k | **-4.2% drawdown**         |
| Closed trades         | 155             |                            |
| Win rate              | 32.3%           | **Worse than coin flip**   |
| Profit factor         | 0.348           | **Losing $3 per $1 won**   |
| Sharpe (2-week)       | -3.495          | **Catastrophic**           |
| Avg win               | +$3.32          |                            |
| Avg loss              | -$5.53          | **1.7× asymmetry (wrong way)** |
| Dominant exit reason  | max_hold (89%)  | **Signal has no edge at hold horizon** |
| Median holding period | 46 min          | Clustering at max_hold=45 bars |
| Take-profit hits      | 1 out of 155    | **TP threshold unreachable** |
| Crypto vs equity      | 100% crypto     | **LightGBM applied to wrong asset class** |
| Orphan trades (no exit)| 345            | **Trade tracking broken**  |

### Analysis: Five Structural Flaws

#### Flaw 1 — Asset-Model Mismatch (FIXED 2026-03-27)

LightGBM (IC=0.11) was trained on equity 1m bars. It was applied to BTC/USD,
ETH/USD, SOL/USD — assets with completely different microstructure, volatility
regime, and feature distributions. The model's predictions on crypto were noise
with a ~0.3% magnitude, randomly passing the 0.15% entry threshold.

**Mathematical justification:** If X ~ N(0, σ²) represents noise predictions,
P(|X| > 0.0015) ≈ 2·Φ(-0.0015/σ). For σ ≈ 0.003 (typical LightGBM output
scale), this probability is ~38%. The model entered crypto positions on over
a third of ticks — essentially a random-entry strategy minus transaction costs.

**Status:** FIXED. Crypto removed from universe. Entry guard added.

#### Flaw 2 — Time Horizon Mismatch (CRITICAL, UNFIXED)

The LightGBM model predicts **15-minute forward returns** (target column in
training). But the position sizing RL agent uses:
- `SIZING_MAX_HOLD_BARS = 45` (45 minutes)
- `SIZING_STOP_LOSS = 0.02` (2%)
- `SIZING_TAKE_PROFIT = 0.035` (3.5%)

For mega-cap equities (AAPL, MSFT), the 1-minute realized volatility is ~0.02%.
Over 45 minutes, expected range is 0.02% × √45 ≈ 0.13%. The stop-loss at 2%
and take-profit at 3.5% are **15-25× the expected range**. They will almost
never trigger. The position will time out at max_hold 89% of the time.

**Mathematical justification:** For a random walk with σ₁ = 0.02%/bar:
- P(hit ±2% in 45 bars) < 0.001% (essentially zero)
- P(hit ±3.5% in 45 bars) < 10⁻⁶
- The position MUST time out unless there's a macro shock

**Fix required:** Either (a) shorten max_hold to match the 15-minute prediction
horizon, or (b) tighten stop/TP to realistic levels for the holding period.

#### Flaw 3 — Signal Strength Has Zero Predictive Power (CRITICAL, UNFIXED)

Win rate analysis by signal strength:
```
|signal| >= 0.0:  WR=32.3%  avg=-$2.68
|signal| >= 0.1:  WR=32.3%  avg=-$2.68
|signal| >= 0.2:  WR=32.5%  avg=-$2.69
|signal| >= 0.3:  WR=30.9%  avg=-$2.33
|signal| >= 0.4:  WR=32.2%  avg=-$0.83
```

Higher signal confidence does NOT improve win rate. The ensemble signal has
**zero conditional predictive power** in production. The entire entry logic
is selecting random positions.

**Hypothesis:** The IC=0.11 from walk-forward validation represents
within-sample autocorrelation, not genuine cross-sectional alpha. Possible
causes:
1. Feature leakage in FFSA validation despite per-ticker time-split
2. LightGBM overfitting to feature engineering artifacts (EMA/MACD/RSI
   are transformations of the same price series)
3. IC measured on 15-min returns doesn't survive 45-min holding + execution

#### Flaw 4 — Ensemble Architecture Collapse

Current production ensemble:
- LightGBM: 60% weight (loaded, but signal is noise in production)
- Transformer: 10% weight (**NOT loaded** — outputs 0.0)
- TCN: 10% weight (**NOT loaded** — outputs 0.0)
- Sentiment: 20% weight (loaded, but slow-moving FinBERT)

Effective formula: `signal = 0.6 × lgbm_conf × lgbm_dir + 0.2 × sentiment`

With Transformer/TCN absent, 20% of the weight is dead. FinBERT sentiment
updates on news cycles (hours), not bar cycles (minutes). The ensemble has
collapsed to a single model — violating the diversification premise.

#### Flaw 5 — Trade Exit Tracking Broken

345 out of 500 trades have no exit recorded. This means:
- PnL accounting is unreliable
- The profit agent's metrics are computed on a biased subsample
- The system cannot learn from its own trades

Root cause: `_open_trade_ids` dict is in-memory and resets on every Railway
deployment. If the bot enters a trade, then Railway redeploys (which happened
multiple times during debugging), the exit can never be matched to the entry.

---

## Phase II: Hypothesis & Innovation Plan

### H1 — Recalibrate to Realistic Equity Microstructure

**Observation:** Mega-cap equities move ~0.13% in 45 minutes.
**Hypothesis:** Tightening exit thresholds to match realized volatility will
improve the profit factor by allowing the 15-min signal to expire in-the-money.

**Proposed parameters:**
```python
SIZING_STOP_LOSS = 0.004      # 0.4% (was 2.0%)
SIZING_TRAILING_STOP = 0.005  # 0.5% (was 2.5%)
SIZING_TAKE_PROFIT = 0.006    # 0.6% (was 3.5%)
SIZING_MAX_HOLD_BARS = 15     # 15 min (was 45) — matches model horizon
```

**Validation:** Backtest these parameters against the last 30 days of
feature_matrix data. Measure:
- Sharpe ratio must be > 0.5 (paper) before deployment
- Profit factor must be > 1.0
- Win rate should be > 45%

### H2 — Feature Validation via Synthetic Null

**Observation:** Signal strength doesn't predict win rate.
**Hypothesis:** The FFSA features carry less live alpha than measured in
walk-forward validation.

**Test:**
1. Shuffle the `lgbm_pred_return` column randomly
2. Run the same entry/exit logic
3. If random-signal performance ≈ real-signal performance, the model has
   zero live edge
4. If real-signal performance is measurably better, the issue is in
   execution (slippage, timing) not in the model

### H3 — Regime-Adaptive Strategy Selection

**Observation:** 100% of trades hit max_hold in a "choppy" regime.
**Hypothesis:** In choppy regimes, mean-reversion beats momentum. The
current system applies the same momentum-based model everywhere.

**Proposed:**
- Regime 0 (trending): Use LightGBM momentum signal (current behavior)
- Regime 1 (choppy): Switch to mean-reversion — fade the signal:
  - If LightGBM says "up", it's likely mean-reverting down (in choppy)
  - Use Bollinger Band Z-score as primary entry
- Regime 2 (high_vol): No new entries; tighten existing stops to 0.3%

### H4 — Replace RL Sizing with Kelly Criterion

**Observation:** The RL sizing agent (Sharpe=-9.7) was never successfully
trained. It's defaulting to threshold fallback which makes uniform bets.
**Hypothesis:** Fractional Kelly sizing based on LightGBM's predicted return
and historical accuracy will outperform uniform sizing.

**Formula:**
```
f* = (p · b - q) / b

where:
  p = historical win rate for signals in this confidence bucket
  b = avg_win / avg_loss ratio for this bucket
  q = 1 - p

Position size = max(0, f*/2) × portfolio  (half-Kelly for safety)
```

If f* ≤ 0 → skip the trade (negative expected value).

### H5 — Cross-Sectional Signal (Relative Value)

**Observation:** Current system trades each ticker independently.
**Hypothesis:** Relative-value signals (pair-trade or sector-relative) are
more robust than absolute directional signals for intraday.

**Proposed:**
- Compute LightGBM pred_return for all 17 tickers
- Rank tickers by pred_return
- Long top 3, short bottom 3 (when shorting enabled)
- This cancels market beta and isolates the model's cross-sectional IC

---

## Phase III: Automated Validation Pipeline

### Shadow Model Architecture

```
Signal Loop (production)
  │
  ├─→ Real execution (paper)
  │
  └─→ Shadow execution (simulated, no orders)
        ├─ Shadow Model A: Tight stops (H1)
        ├─ Shadow Model B: Regime-adaptive (H3)
        └─ Shadow Model C: Kelly sizing (H4)
```

Each shadow model logs to `reports/shadow/model_{name}/YYYY-MM-DD.json`
with fields: trades, PnL, Sharpe, Sortino, max_drawdown.

### Promotion Criteria

A shadow model can be proposed for deployment (to `config/staging/`) only if:
- **Sharpe > 1.0** over a 2-week window
- **Sortino > 1.5** (penalizes downside only)
- **Profit factor > 1.2**
- **Min 50 trades** in the validation window
- **Max drawdown < 5%** of starting equity

### Implementation Priority

| Priority | Hypothesis | Effort | Expected Impact |
|----------|-----------|--------|-----------------|
| P0       | H1 — Realistic exit params | 1 hour | Stops max_hold bleeding |
| P0       | Fix orphan trade tracking  | 1 hour | Accurate PnL accounting |
| P1       | H2 — Null signal test      | 2 hours | Validates model edge |
| P1       | H4 — Kelly sizing          | 3 hours | Proper position mgmt |
| P2       | H3 — Regime-adaptive       | 1 day  | Strategy diversification |
| P3       | H5 — Cross-sectional       | 2 days | Market-neutral alpha |

---

## Agent Behavioral Rules

### Scope
- Analyze trade performance, feature quality, regime conditions
- Propose parameter changes and new model architectures
- Write backtesting code and shadow model scripts
- Generate weekly research reports to `reports/research/YYYY-WW.json`

### Must NOT
- Modify `config/live.yaml` directly
- Deploy new models to production without human approval
- Disable circuit breakers or risk controls
- Access broker credentials or account numbers

### Escalate If
- Sharpe drops below -1.0 over any rolling 1-week window
- Win rate drops below 30% over 50+ trades
- Model IC (live) drops below 0.05 for 5 consecutive days
- Any single trade loses > 1% of portfolio

### Output
- Weekly research report: `reports/research/YYYY-WW.json`
- Shadow model logs: `reports/shadow/{model_name}/YYYY-MM-DD.json`
- Proposed changes: `config/staging/research_proposals.json`

---

## Appendix: Current System Parameters

```yaml
# Signal generation
lgbm_trading_threshold: 0.0025     # min |pred_return| to generate signal
ensemble_entry_threshold: 0.40     # min |ensemble_signal| to enter
confidence_dead_zone: [0.45, 0.55] # zero LightGBM if dir_prob in this range

# Position management (NEEDS FIXING)
sizing_stop_loss: 0.02       # 2% — too wide for intraday equities
sizing_trailing_stop: 0.025  # 2.5% — unreachable in 45 min
sizing_take_profit: 0.035    # 3.5% — unreachable in 45 min
sizing_max_hold_bars: 45     # 45 min — 3× the model's prediction horizon

# Risk controls
max_portfolio_heat: 0.80
daily_loss_limit: -0.03
max_drawdown: -0.08
max_position_pct: 0.25
consecutive_loss_reduction: 5
```
