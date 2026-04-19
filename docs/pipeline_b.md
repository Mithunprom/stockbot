# Pipeline B — Rules-Based Trading System

**Status:** Active (standalone mode since 2026-04-19)
**Config:** `ACTIVE_PIPELINE=b`, `AB_TEST_ENABLED=false`

---

## Overview

Pipeline B is a rules-based signal engine that replaces Pipeline A's ML models
(LightGBM, Transformer, TCN) with transparent, interpretable scoring across
five dimensions. Every signal component can be traced to specific indicator
values — no black-box inference.

Pipeline A's ML models achieved IC=0.11 in walk-forward validation but showed
zero predictive power in live trading (win rate 32%, Sharpe -3.5). Pipeline B
was built as an alternative hypothesis: hand-tuned rules that can be debugged
and improved without retraining.

---

## Architecture

```
Market Data (Alpaca REST poller, 1-min bars)
  → OHLCV_1m (TimescaleDB)
  → LiveFeatureComputer (incremental indicators)
  → feature_matrix (30 FFSA features per ticker per bar)

Pipeline B Signal Loop (every 60s during market hours):
  → Fetch latest feature row per ticker
  → Score 5 dimensions:
      1. Technical score (30%)  — from feature_matrix indicators
      2. Fundamental score (25%) — from yfinance (daily cache)
      3. Regime score (20%)     — VIX + SPY/QQQ momentum
      4. Sentiment score (20%)  — FinBERT via HuggingFace API
      5. Social score (5%)      — Reddit r/wallstreetbets + finance subs
  → Weighted ensemble → EnsembleSignal
  → Entry gate (cost threshold + conviction)
  → Position sizer (Kelly-inspired buckets)
  → ATR-adaptive exits (per-ticker volatility scaling)
  → Alpaca paper execution
```

---

## Signal Dimensions

### 1. Technical Score (30%) — `_score_technicals()`

Scores each ticker from its latest indicator values. Returns [-1, +1].

| Indicator | Bullish Signal | Score | Bearish Signal | Score |
|-----------|---------------|-------|----------------|-------|
| RSI(14) | < 30 (oversold) | +0.40 | > 70 (overbought) | -0.40 |
| RSI(14) | < 40 | +0.15 | > 60 | -0.15 |
| MACD histogram | > 0 + above signal | +0.25 | < 0 + below signal | -0.25 |
| Bollinger %B | < 0.10 (lower band) | +0.20 | > 0.90 (upper band) | -0.20 |
| VWAP deviation | < -1.5% | +0.20 | > +1.5% | -0.20 |
| MFI(14) | < 20 | +0.15 | > 80 | -0.15 |
| Stochastic K | < 20 | +0.15 | > 80 | -0.15 |
| OBV % change | > 2% | +0.10 | < -2% | -0.10 |
| Multi-TF confluence | All aligned bullish | +0.20 | All aligned bearish | -0.20 |
| Divergence | Bullish divergence | +0.075 | Bearish divergence | -0.075 |
| Candlestick patterns | Engulfing/hammer/morning star | +0.10 | Evening star | -0.10 |
| Supply/demand zones | At demand zone | +0.15 | At supply zone | -0.15 |
| Parabolic SAR | Bullish | +0.05 | Bearish | -0.05 |
| Donchian breakout | Upside breakout | +0.10 | Downside breakout | -0.10 |

**ADX trend filter** dampens signal in weak trends:
- ADX < 15: score × 0.60
- ADX < 20: score × 0.85

**Source:** `src/models/pipeline_b.py:_score_technicals()`

---

### 2. Fundamental Score (25%) — `_score_fundamentals()`

Scores from daily-cached yfinance data. Returns [-1, +1].

| Factor | Bullish | Score | Bearish | Score |
|--------|---------|-------|---------|-------|
| Trailing P/E | < 12 (deep value) | +0.30 | > 40 (very expensive) | -0.20 |
| Trailing P/E | < 18 (fair value) | +0.15 | 25–40 (somewhat expensive) | -0.10 |
| P/E compression | Forward P/E 20%+ below trailing | +0.15 | Forward > trailing by 10%+ | -0.10 |
| Earnings surprise | > 10% beat | +0.30 | > 10% miss | -0.30 |
| Earnings surprise | > 5% beat | +0.20 | > 5% miss | -0.20 |
| Revenue growth YoY | > 25% (hypergrowth) | +0.20 | < -10% (shrinking fast) | -0.20 |
| Revenue growth YoY | > 10% | +0.10 | < 0% | -0.10 |

**Data source:** yfinance (free, no API key). Cached 24 hours.
**Source:** `src/data/fundamentals.py`, `src/models/pipeline_b.py:_score_fundamentals()`

---

### 3. Market Regime Score (20%)

Classifies macro environment as risk_on / neutral / risk_off.
Returns regime_score [-1, +1].

| Component | Bullish | Bearish |
|-----------|---------|---------|
| VIX level | < 15: +0.30 | > 30: -0.40 |
| VIX 30d percentile | < 30th: +0.15 | > 70th: -0.15 |
| SPY 5-bar return | > 0.2%: +0.20 | < -0.2%: -0.20 |
| SPY 15-bar return | > 0.5%: +0.15 | < -0.5%: -0.15 |
| QQQ-SPY spread | QQQ outperforming: +0.10 | QQQ underperforming: -0.10 |

**Regime classification:**
- score >= 0.20 → `risk_on`
- score <= -0.20 → `risk_off`
- otherwise → `neutral`

**Data source:** VIX via yfinance (sync, in thread executor); SPY/QQQ from ohlcv_1m table.
**Poll interval:** Every 60 seconds.
**Source:** `src/data/market_regime.py`

---

### 4. Sentiment Score (20%)

FinBERT (ProsusAI/finbert) scores news articles from Polygon.io and
Benzinga feeds. Returns rolling sentiment index [-1, +1] per ticker.

- Articles scored via HuggingFace Inference API
- Recency-weighted (half-life 12 hours)
- Relevance-weighted (ticker mention frequency)
- Stored in `news_raw` table

**Known issue:** HuggingFace API token expired (401 errors since ~2026-04-16).
Sentiment returns 0.0 until token is refreshed. This means 20% of the signal
weight is effectively dead.

**Source:** `src/models/sentiment.py`, `src/data/news.py`

---

### 5. Social Score (5%)

Reddit-based social sentiment from finance subreddits.

**Subreddits polled:** r/wallstreetbets, r/stocks, r/investing, r/options
**Poll interval:** Every 5 minutes
**Scoring method:**
1. Fetch latest 50 posts per subreddit
2. Filter posts mentioning the ticker ($AAPL or word boundary match)
3. Score each post:
   - FinBERT if available, else keyword heuristic (bullish/bearish word matching)
   - Weight by log2(upvotes) × recency (1.0 for last hour, decays to 0.3 at 24h)
4. Weighted average → score [-1, +1]

**Keyword examples:** bullish words (buy, calls, moon, squeeze, tendies);
bearish words (sell, puts, crash, overvalued, rug pull)

**Source:** `src/data/social_stocktwits.py`

---

## Composite Signal

```python
ensemble = (0.30 × tech_score
          + 0.25 × fund_score
          + 0.20 × regime_score
          + 0.20 × sentiment_score
          + 0.05 × social_score)

# Clipped to [-1, +1]
```

### Mapping to Execution Layer

Pipeline B produces `EnsembleSignal` objects compatible with Pipeline A's
execution stack. The mapping:

| EnsembleSignal field | Pipeline B source |
|---------------------|-------------------|
| `lgbm_dir_prob` | 0.5 + ensemble × 0.30 (maps [-1,+1] → [0.2, 0.8]) |
| `lgbm_pred_return` | ensemble × 0.024 (moderate signal ±0.125 → ±0.003) |
| `transformer_direction` | Fundamental direction (+1/-1/0) |
| `transformer_confidence` | |fund_score| |
| `tcn_direction` | Regime direction (+1/-1/0) |
| `tcn_confidence` | |regime_score| |
| `sentiment_index` | Sentiment score directly |
| `ensemble_signal` | Composite ensemble value |

---

## Entry Gate

Trades are only entered when both conditions are met:

1. **Cost gate:** `|pred_return| > 0.0015` (SIZING_COST_THRESHOLD)
   - A moderate signal of ±0.125 maps to ±0.003 pred_return, which passes
   - Weak signals (|ensemble| < 0.0625) are filtered out

2. **Conviction gate:** `dir_prob` NOT in [0.45, 0.55] dead zone
   - Neutral signals are zeroed

3. **Regime gate** (inherited from Pipeline A):
   - Trending (regime=0): threshold 0.40, full size
   - Choppy (regime=1): threshold 0.55, 70% size
   - High vol (regime=2): threshold 0.55, 50% size

---

## Position Sizing

Kelly-inspired conviction buckets:

| dir_prob range | Max position % | Description |
|---------------|---------------|-------------|
| [0.55, 0.65) | 2% of portfolio | Low conviction |
| [0.65, 0.80) | 4% of portfolio | Medium conviction |
| [0.80, 1.00] | 6% of portfolio | High conviction |

Hard caps:
- Max single position: 25% of portfolio
- Max portfolio heat: 80% deployed
- Min cash buffer: 10%

---

## Exit Management — ATR-Adaptive

Stops and targets scale per-ticker by 1-minute ATR percentage:

```python
stop_loss    = max(ATR% × 15, 0.5%)    # floor prevents micro-stops
trailing_stop = max(ATR% × 20, 0.8%)
take_profit  = max(ATR% × 35, 1.5%)
max_hold     = 45 bars (45 minutes)
```

**Example:**
| Ticker | 1m ATR% | Stop | Trail | Target |
|--------|---------|------|-------|--------|
| AAPL | 0.04% | 0.6% | 0.8% | 1.5% (floor) |
| NVDA | 0.09% | 1.35% | 1.8% | 3.15% |
| MSTR | 0.19% | 2.85% | 3.8% | 6.65% |

**Source:** `src/agents/signal_loop.py:_atr_exits()`

---

## Data Sources (All Free)

| Source | Data | Frequency | API Key? |
|--------|------|-----------|----------|
| Alpaca IEX | 1-min OHLCV bars | REST poll every 60s | Yes (free account) |
| yfinance | VIX, P/E, earnings, revenue | Daily cache (24h TTL) | No |
| Polygon.io | News articles | Poll every 5 min | Yes (free tier) |
| HuggingFace API | FinBERT sentiment inference | Per article | Yes (free tier) |
| Reddit public JSON | Social sentiment (WSB etc.) | Poll every 5 min | No |

---

## Key Files

| File | Purpose |
|------|---------|
| `src/models/pipeline_b.py` | PipelineBEngine, _score_technicals, _score_fundamentals |
| `src/agents/signal_loop_b.py` | SignalLoopB — execution loop (inherits from SignalLoop) |
| `src/data/fundamentals.py` | FundamentalsCache — yfinance P/E, earnings, revenue |
| `src/data/market_regime.py` | MarketRegimeMonitor — VIX + SPY/QQQ regime classification |
| `src/data/social_stocktwits.py` | StockTwitsFeed — Reddit social sentiment |
| `src/models/sentiment.py` | SentimentScorer — FinBERT via HuggingFace |
| `src/agents/signal_loop.py` | Parent class — execution, exits, ATR logic, circuit breakers |
| `src/features/regime.py` | REGIME_GATE thresholds per market regime |

---

## Test Scripts

| Script | Purpose |
|--------|---------|
| `scripts/test_pipeline_b_steps.py` | Tests each scoring dimension individually |
| `scripts/test_signal_scan.py` | Scans full universe, shows ranked signals |
| `scripts/test_trade_roundtrip.py` | Buys/sells 1 share to verify execution |
| `scripts/daily_performance.py` | Daily P&L plots — cumulative, drawdown, per-ticker, etc. |
| `scripts/test_pipeline_health.py` | End-to-end health check (DB, API, signals, ATR) |

---

## Comparison: Pipeline A vs Pipeline B

| Aspect | Pipeline A (ML) | Pipeline B (Rules) |
|--------|----------------|-------------------|
| Signal source | LightGBM (IC=0.11), Transformer, TCN, FinBERT | Technical rules, fundamentals, regime, sentiment, social |
| Interpretability | Black box — SHAP needed to explain | Every score traceable to indicator values |
| Training required | Yes — FFSA, LightGBM, Transformer, TCN, RL | No training — hand-tuned rules |
| Adaptability | Retrain on new data | Adjust weights and thresholds manually |
| Live IC (2026-03-27) | ~0 (zero conditional predictive power) | TBD — insufficient trades yet |
| Execution stack | Shared (ATR exits, position sizer, circuit breakers) | Same — inherits from SignalLoop |
| Data dependencies | feature_matrix, ML checkpoints (S3) | feature_matrix, yfinance, Reddit |

---

## Configuration

### Environment Variables

| Variable | Value | Effect |
|----------|-------|--------|
| `ACTIVE_PIPELINE` | `b` | Run Pipeline B standalone (full capital) |
| `ACTIVE_PIPELINE` | `a` | Run Pipeline A standalone (default) |
| `AB_TEST_ENABLED` | `true` | Run both pipelines, 50/50 capital split |
| `AB_CAPITAL_SPLIT` | `0.5` | Fraction of capital for Pipeline A (when A/B) |

### Pipeline B Weights (in `src/models/pipeline_b.py`)

```python
@dataclass
class PipelineBWeights:
    technicals:   0.30
    fundamentals: 0.25
    regime:       0.20
    sentiment:    0.20
    social:       0.05
```

To adjust, modify `PipelineBWeights` defaults or pass custom weights to
`PipelineBEngine(weights=PipelineBWeights(technicals=0.35, ...))`.

---

## Known Issues

1. **HuggingFace 401** — Sentiment scoring returns 0.0. 20% of signal weight is dead.
   Fix: refresh the HF API token in Railway env vars.

2. **Social score is thin** — Reddit mentions of individual tickers are sparse
   outside mega-caps. Most tickers get 0.0 social score.

3. **Fundamental cache cold start** — yfinance fetches are slow (2-3s per ticker).
   Universe of 20+ tickers takes ~1 minute on first boot. Caches are pre-populated
   at startup to avoid missing the first tick.

4. **No short selling** — Paper account lacks margin approval. All positions are
   long-only, so bearish signals are used to avoid entry (not to short).
