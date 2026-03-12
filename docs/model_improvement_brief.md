# StockBot ML Model Improvement Brief
*Prepared for NotebookLM — request: analyze and return a model improvement strategy*

---

## 1. Project Goal

An autonomous intraday paper-trading bot that predicts 5-bar (5-minute) forward price direction on 1-minute bars for US equities and crypto. The model outputs a 3-class direction signal (up / flat / down) with a confidence score. These signals feed an ensemble that drives a live RL-based execution agent.

**The core problem we need help with**: The Transformer model consistently achieves val_acc ≈ 0.33–0.37, which is barely above random chance for a balanced 3-class problem. Training loss converges but no generalization occurs.

---

## 2. Training Data

### Source
- TimescaleDB (PostgreSQL) table `feature_matrix`
- 1-minute OHLCV bars from Alpaca IEX (free tier), ~2 years history
- Current universe: 17 tickers (AAPL, MSFT, NVDA, AMZN, GOOGL, TSLA, AVGO, AMD, JPM, V, MA, LLY, UNH, COST, NFLX, XOM, CVX)
- SP500_TOP50 backfill in progress (will add 33 more tickers, ~5M total rows)

### Feature Engineering
All features computed from raw OHLCV via pure pandas/numpy (no TA libraries):

**Top 30 features selected by FFSA (LightGBM + SHAP, IC = 0.0265):**
```
atr_pct, adx, atr_14, vol_ratio, obv, bb_width, macd, day_of_week,
vol_seasonal_ratio, vpin_50, vpin_zscore, time_to_close, dmp, dmn,
macd_signal, obv_pct, cci_20, returns_15b, macd_hist, bb_upper,
mfi_14, high_low_range, returns_5b, ema_50, returns_1b, bb_lower,
stoch_d, ema_9, rsi_14, stoch_k
```

**Critical observation: FFSA validation IC = 0.0265** — this is extremely low. The SHAP importances are uniformly tiny (max = 6.5e-5), suggesting no single feature has meaningful predictive power for the 5-bar forward return label.

### Label Construction
```python
FORWARD_N = 5        # predict 5-bar (5-minute) forward return
UP_DOWN_PCT = 0.33   # percentile split: bottom 33% = down, top 33% = up, middle = flat
SEQ_LEN = 60         # 60 bars (1 hour) of history per sample
STRIDE = 3           # 3-bar stride between overlapping samples
```

Labels are computed per-ticker using percentile thresholds on `forward_return = close.pct_change(5).shift(-5)`. This gives balanced 33/33/33 class distribution but the actual return magnitude at the 33rd/67th percentiles is typically ±0.05–0.15% for liquid large-caps, which is within microstructure noise.

---

## 3. Model Architectures

### 3A. TransformerSignalModel
```
Input: (batch, seq_len=60, n_features=30)
Architecture:
  - LayerNorm(30) → Linear(30→128) → SinusoidalPositionalEncoding
  - TransformerEncoder: 5 layers, 8 heads, d_ff=512, dropout=0.1
  - Cross-attention: options flow features as key/value (currently zeros — no historical options data)
  - AdaptiveAvgPool1d(1) → Linear(128→64) → GELU → Linear(64→3)  [classifier]
  - Separate: Linear(128→32) → GELU → Linear(32→1) → Sigmoid  [confidence head]
Output: (logits: B×3, confidence: B×1)
Total params: 1,075,200
```

### 3B. TCNSignalModel
```
Input: x_1m (batch, n_features=30, seq_len=60), x_5m (batch, n_features=30, seq_5m=12)
Architecture:
  - Dual stream: each stream has 6 TemporalBlocks with dilations [1,2,4,8,16,32]
  - Channels: 128, kernel_size=3
  - Receptive field: ~126 bars
  - LayerNorm before first TemporalBlock (critical: prevents NaN from OBV scale ~1e7)
  - Last time-step pooling → cat([h_1m, h_5m]) → Linear(256→128) → GELU
  - return_head: Linear(128→1)     [next-bar return regression]
  - direction_head: Linear(128→3)  [direction logits]
  - confidence_head: Linear(128→1) → Sigmoid
Total params: ~1,140,000
```

---

## 4. Training Setup

```python
# Optimizer
AdamW(lr=5e-5, weight_decay=1e-4)
CosineAnnealingLR(T_max=epochs)
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Loss
FocalLoss(gamma=0.5, weight=None)   # gamma=0 → plain CE; 0.5 = light focal

# Data split
Walk-forward: per-ticker, train on oldest 80%, val on newest 20%
Batch size: 512
Epochs: 15
```

### Training Results History

**Run 0 (old architecture: d_model=64, 3 layers, 4 heads — ~200k params):**
- Transformer: val_acc=0.375, best sharpe=0.896
- TCN: val_acc=0.369, best sharpe=0.776
- Problem: val_loss diverged after epoch 1 on this architecture too

**Run 1 (new architecture: d_model=128, 5 layers, 8 heads — 1.07M params, 17 tickers):**
- Epoch 1: train_loss=0.8943, val_loss=0.8907, val_acc=0.366
- Epoch 2: val_loss=0.8912, val_acc=0.366 (flat)
- Epoch 3: val_loss=0.8954, val_acc=0.347 (diverging)
- Epoch 4: val_loss=0.8976, val_acc=0.348 (diverging)
- Diagnosis: overfitting on 17 tickers with 1M+ params

**Run 2 (SP500_TOP50, 50 tickers, lr=5e-5):**
- Epoch 1: train_loss=0.8976, val_loss=0.8972, val_acc=0.337
- Epoch 2: val_loss=0.8971, val_acc=0.333
- Epoch 3: val_loss=0.8970, val_acc=0.337
- **Stuck at random-chance accuracy despite 50 tickers and 12× more data**

---

## 5. Bug Fixes Already Applied

The following training bugs were found and fixed (do NOT suggest re-fixing these):

1. ✅ **Input normalization**: Added `LayerNorm` before first projection in both Transformer and TCN. Without it, OBV (~1e7) vs returns (~0.001) caused gradient collapse.
2. ✅ **Label balance**: Changed `UP_DOWN_PCT` from 0.15 → 0.33 to force 33/33/33 class balance. Previous 0.15 gave heavily imbalanced classes → model collapsed to majority class.
3. ✅ **Focal loss**: Changed from `FocalLoss(gamma=2.0, weight=class_weights)` → `FocalLoss(gamma=0.5, weight=None)`. Gamma=2.0 over-penalized hard examples on noisy 1m data.
4. ✅ **RL Sharpe bug**: Fixed episode-level std=0 bug in `_evaluate_sharpe()` that produced Sharpe=-5.5T.
5. ✅ **Learning rate**: Reduced from 1e-4 → 5e-5 for larger model.
6. ✅ **`d_ff` in checkpoint config**: Added `d_ff` to saved config dict to prevent state dict mismatch on reload.
7. ✅ **Gradient clipping**: `clip_grad_norm_(max_norm=1.0)` already in place.

---

## 6. Root Cause Hypothesis

We believe the core issue is **label quality and signal-to-noise**, not architecture:

1. **FFSA IC = 0.0265 is near zero**: A LightGBM model with SHAP can barely predict 5-bar returns better than random. If the best tree-based method achieves IC=0.0265, a neural network on the same features likely cannot do better.

2. **5-minute forward returns are microstructure noise at 1m resolution**: At 1m bar granularity, a 5-bar forward return of ±0.05–0.15% is within the bid-ask spread and order flow noise. There may not be sufficient signal.

3. **No microstructure features**: The features are all technical indicators (trend, momentum, volatility, volume). There are no order flow, level-2, or tick data features that would capture real short-term predictability.

4. **Overlapping sequences with stride=3**: Consecutive samples share 57/60 bars — high autocorrelation in training data may inflate apparent train performance while val performance suffers.

5. **Cross-attention options features are all zeros**: The Transformer's options cross-attention path receives zeros (no historical options data yet), contributing no signal. But this shouldn't cause random-chance accuracy — it should just be ignored.

6. **Percentile labels are noisy by design**: By construction, the "up" class is simply the top 33% of 5-min returns on a given ticker, day, and market condition. In choppy markets, the top 33% return might be +0.02%. Predicting this from 1 hour of indicators is extremely hard.

---

## 7. What We've NOT Tried

These are areas where we want improvement suggestions:

### Label construction alternatives
- Longer forward horizon (e.g. FORWARD_N = 15, 30, or 60 bars)
- Binary classification (up vs not-up) instead of 3-class
- Threshold-based labels (fixed ±0.3% instead of percentile) with class reweighting
- Trend-following labels (label based on directional consistency over N bars, not just N-bar return)
- Multi-horizon labels (predict 5m, 15m, 30m simultaneously — multi-task learning)

### Feature improvements
- Per-ticker feature normalization (z-score per ticker over rolling window)
- Return features normalized by ATR (regime-adaptive)
- Lag features: rolling return autocorrelation, mean-reversion signals
- Cross-sectional features: relative strength vs sector ETF
- Calendar/event features: earnings dates, FOMC days, options expiry

### Architecture alternatives
- **Smaller model**: The IC=0.0265 suggests the signal is weak. A smaller model (d_model=64, 2 layers) with strong regularization might generalize better.
- **Multi-task learning**: Predict 5m AND 15m AND 30m returns simultaneously — the shared representations may learn more robust features.
- **Binary classification head**: Replace 3-class with binary (up/not-up) + separate (down/not-down), reducing label noise.
- **Temporal self-supervised pre-training**: Pre-train on masked bar prediction before fine-tuning on direction.
- **Sequence length**: Shorter (30 bars) or longer (120 bars) sequences.
- **Per-ticker embeddings**: Add a learned ticker embedding to capture ticker-specific behavior.

### Training procedure
- **Larger stride**: Reduce sequence overlap from stride=3 to stride=15 or stride=30 to reduce training autocorrelation.
- **Data augmentation**: Add Gaussian noise to features, random scaling of returns.
- **Curriculum learning**: Start with easier samples (higher |forward_return|) and gradually include harder ones.
- **Walk-forward cross-validation**: Multiple rolling train/val splits instead of a single 80/20.
- **Feature standardization**: Currently features are NOT normalized before model input (LayerNorm is applied per-sequence, but features like OBV have absolute scale differences across tickers).

---

## 8. Constraints

- **No paid data**: Using Alpaca IEX free tier (no Level-2, no tick data, no options chain history)
- **CPU/MPS inference** on Railway Cloud (no CUDA in production)
- **Training locally** on Apple Silicon (MPS)
- **Real-time requirement**: Inference must complete in <200ms per tick for all 24 tickers
- **Free options data**: yfinance options chains — 5-minute snapshots, not historical
- **No lookahead**: All indicators are shifted by 1 bar before training and inference

---

## 9. Current System Architecture (context)

```
Live data (Alpaca IEX WebSocket, 1m bars)
  → LiveFeatureComputer (incremental indicators on each bar)
  → feature_matrix DB (TimescaleDB)

Signal loop (every 1m):
  → fetch 60 bars from feature_matrix
  → TransformerSignalModel.predict() → (direction, confidence)
  → TCNSignalModel.predict() → (direction, confidence)
  → FinBERT sentiment (disabled — missing transformers package)
  → Ensemble: 0.45*Transformer + 0.35*TCN + 0.20*Sentiment
  → Regime gate: choppy→threshold=0.55, high_vol→threshold=0.55+50%size
  → RL agent (PPO, 500k steps, Sharpe=-9.7, not ready)
  → AlpacaOrderRouter → paper account
```

---

## 10. Key Questions for NotebookLM

1. **Is 5-minute forward return predictable from 1-hour of technical indicators?** Given IC=0.0265 with a LightGBM model, what is the theoretical ceiling for a deep learning model on this problem?

2. **What label construction would give a learnable problem?** Should we change `FORWARD_N`, the percentile split, or the labeling strategy entirely?

3. **Is the architecture too large for the signal strength?** With IC~0.03, would a simpler model (logistic regression, shallow MLP, or small Transformer) outperform the 1M-param model?

4. **What features would most increase IC?** Given the current feature set (all lagged technical indicators), what categories of features are we missing?

5. **What training procedure changes would help?** Specifically addressing the stride/autocorrelation issue and the cross-ticker generalization issue.

6. **Should we abandon 3-class classification?** Would binary (directional) classification or regression (predict continuous return) be a better proxy task?

7. **Is the SP500_TOP50 multi-ticker approach correct?** The model has no ticker embedding — it must learn universal indicator patterns. Is this a reasonable assumption for intraday signals?

---

## 11. Success Metrics

We consider the model "ready for paper trading evaluation" when:
- val_acc > 0.40 (7 percentage points above random for 3-class)
- OR val_IC (on held-out data) > 0.05
- OR backtest Sharpe on validation period > 1.0

Current best: val_acc = 0.375 (old small architecture), val_IC ≈ 0.03
