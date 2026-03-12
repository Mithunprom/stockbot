"""Market regime detection.

3-class regime label computed from price/volatility indicators:
  0 = trending      (ADX > 25, normal volatility)
  1 = choppy        (ADX < 20, directionless price action)
  2 = high_vol      (ATR or Bollinger width > 2× rolling median)

Used in:
  - build_features.py: adds 'regime' column to feature_matrix via indicators.py
  - signal_loop.py: gates entry threshold and position size per regime
  - trading_env.py: RL observation slot (was hardcoded 0.0)

Signal loop gating:
  regime=0 (trending):  normal threshold (0.40), full position size
  regime=1 (choppy):    raised threshold (0.55), 70% size — avoid whipsaws
  regime=2 (high_vol):  raised threshold (0.55), 50% size — reduce risk
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ─── Thresholds ───────────────────────────────────────────────────────────────

_ADX_TREND_THRESHOLD = 25.0      # ADX > 25 → trending
_ADX_CHOPPY_THRESHOLD = 20.0     # ADX < 20 → choppy
_VOL_SPIKE_MULTIPLIER = 2.0      # ATR_pct > 2× rolling median → high-vol
_VOL_LOOKBACK = 200              # bars for median ATR baseline (~3h of 1m bars)
_ADX_SMOOTH = 60                 # bars for rolling ADX smoothing (reduces 1m noise)

# Regime int labels — match trading_env.py TradingEnv comment
REGIME_TRENDING = 0
REGIME_CHOPPY = 1
REGIME_HIGH_VOL = 2


def compute_regime(feat_df: pd.DataFrame) -> pd.Series:
    """Classify each bar into a market regime.

    Args:
        feat_df: Feature DataFrame with DatetimeIndex.
                 Expected columns: adx, atr_pct, bb_width (pre-computed).

    Returns:
        Integer Series (same index as feat_df):
          0 = trending, 1 = choppy, 2 = high_vol
    """
    regime = pd.Series(REGIME_CHOPPY, index=feat_df.index, dtype=np.int8)

    # ── Step 1: High-vol detection (highest priority) ─────────────────────────
    # ATR_pct spike: current > 2× rolling median of past 200 bars
    if "atr_pct" in feat_df.columns:
        atr_pct = feat_df["atr_pct"].ffill().fillna(0.0)
        atr_median = atr_pct.rolling(_VOL_LOOKBACK, min_periods=20).median().clip(lower=1e-9)
        regime[atr_pct > _VOL_SPIKE_MULTIPLIER * atr_median] = REGIME_HIGH_VOL

    # BB width expansion: second confirming signal
    if "bb_width" in feat_df.columns:
        bb_w = feat_df["bb_width"].ffill().fillna(0.0)
        bb_median = bb_w.rolling(_VOL_LOOKBACK, min_periods=20).median().clip(lower=1e-9)
        bb_spike = bb_w > _VOL_SPIKE_MULTIPLIER * bb_median
        # Only escalate if not already high_vol — avoid double-counting
        regime[(bb_spike) & (regime != REGIME_HIGH_VOL)] = REGIME_HIGH_VOL

    # ── Step 2: Trend / choppy (only on non-high-vol bars) ────────────────────
    non_highvol = regime != REGIME_HIGH_VOL

    if "adx" in feat_df.columns:
        adx = feat_df["adx"].ffill().fillna(0.0)
        # Smooth over 60 bars to suppress 1m-bar noise
        adx_smooth = adx.rolling(_ADX_SMOOTH, min_periods=10).mean()
        regime[(adx_smooth > _ADX_TREND_THRESHOLD) & non_highvol] = REGIME_TRENDING
        # choppy (regime=1) is the default; ADX < 20 stays as-is

    return regime


def regime_label(regime_val: int) -> str:
    """Human-readable regime name for logging and reports."""
    return {
        REGIME_TRENDING: "trending",
        REGIME_CHOPPY: "choppy",
        REGIME_HIGH_VOL: "high_vol",
    }.get(int(regime_val), "unknown")


# ─── Signal loop gate parameters ─────────────────────────────────────────────

# (threshold, size_scale) per regime — used in signal_loop._act_on_signal()
REGIME_GATE: dict[int, tuple[float, float]] = {
    REGIME_TRENDING:  (0.40, 1.00),   # normal operation
    REGIME_CHOPPY:    (0.55, 0.70),   # tighter threshold, smaller size
    REGIME_HIGH_VOL:  (0.55, 0.50),   # same threshold, half size
}
