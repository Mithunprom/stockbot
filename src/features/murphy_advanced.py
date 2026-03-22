"""Advanced Murphy indicators — divergence, Fibonacci, volume climax, intermarket.

From John J. Murphy, "Technical Analysis of the Financial Markets" (1999):

1. RSI / MACD Divergence (Chapters 10, 13):
   Price makes new high but oscillator doesn't → bearish divergence (reversal)
   Price makes new low but oscillator doesn't → bullish divergence (reversal)
   One of the most reliable reversal signals in technical analysis.

2. Fibonacci Retracement (Chapter 13):
   After a swing move, price tends to retrace to 38.2%, 50%, or 61.8%
   before resuming. Proximity to these levels provides S/R signals.

3. Volume Climax (Chapter 7):
   Extreme volume at price extremes confirms reversals. A spike in volume
   (>2σ) at a swing high/low suggests exhaustion and potential reversal.

4. Intermarket Proxies (Chapter 17):
   Cross-asset signals without external data — derived from the existing
   universe: sector-relative momentum, volatility regime vs returns.

All features use only past data. No lookahead.
Shift-by-1 is handled at the pipeline level (indicators.py), NOT here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ─── 1. Divergence Detection ────────────────────────────────────────────────

def _find_swing_highs(series: pd.Series, lookback: int = 10) -> pd.Series:
    """1.0 where series is a local maximum over ±lookback bars."""
    rolling_max = series.rolling(2 * lookback + 1, center=True, min_periods=lookback).max()
    return (series == rolling_max).astype(float)


def _find_swing_lows(series: pd.Series, lookback: int = 10) -> pd.Series:
    """1.0 where series is a local minimum over ±lookback bars."""
    rolling_min = series.rolling(2 * lookback + 1, center=True, min_periods=lookback).min()
    return (series == rolling_min).astype(float)


def rsi_divergence(
    close: pd.Series,
    rsi: pd.Series,
    lookback: int = 10,
    window: int = 50,
) -> pd.Series:
    """Detect RSI divergence: +1 bullish, −1 bearish, 0 none.

    Bearish divergence: price makes higher high but RSI makes lower high.
    Bullish divergence: price makes lower low but RSI makes higher low.

    Uses rolling window to compare current swing vs previous swing.
    """
    result = pd.Series(0.0, index=close.index)

    # Rolling highs/lows over the window
    price_hi = close.rolling(window, min_periods=lookback).max()
    price_lo = close.rolling(window, min_periods=lookback).min()
    rsi_at_price_hi = rsi.rolling(window, min_periods=lookback).apply(
        lambda x: x.iloc[-1] if len(x) > 0 else np.nan, raw=False
    )

    # Previous window highs/lows for comparison
    prev_price_hi = price_hi.shift(lookback)
    prev_price_lo = price_lo.shift(lookback)
    prev_rsi_hi = rsi.rolling(window, min_periods=lookback).max().shift(lookback)
    prev_rsi_lo = rsi.rolling(window, min_periods=lookback).min().shift(lookback)

    current_rsi_hi = rsi.rolling(lookback, min_periods=5).max()
    current_rsi_lo = rsi.rolling(lookback, min_periods=5).min()

    # Bearish divergence: higher price high + lower RSI high
    bearish = (
        (close.rolling(lookback, min_periods=5).max() > prev_price_hi) &
        (current_rsi_hi < prev_rsi_hi) &
        prev_price_hi.notna() & prev_rsi_hi.notna()
    )
    result[bearish] = -1.0

    # Bullish divergence: lower price low + higher RSI low
    bullish = (
        (close.rolling(lookback, min_periods=5).min() < prev_price_lo) &
        (current_rsi_lo > prev_rsi_lo) &
        prev_price_lo.notna() & prev_rsi_lo.notna()
    )
    result[bullish] = 1.0

    return result


def macd_divergence(
    close: pd.Series,
    macd_hist: pd.Series,
    lookback: int = 10,
    window: int = 50,
) -> pd.Series:
    """Detect MACD histogram divergence: +1 bullish, −1 bearish, 0 none.

    Bearish: price higher high, MACD hist lower high.
    Bullish: price lower low, MACD hist higher low.
    """
    result = pd.Series(0.0, index=close.index)

    prev_price_hi = close.rolling(window, min_periods=lookback).max().shift(lookback)
    prev_price_lo = close.rolling(window, min_periods=lookback).min().shift(lookback)
    prev_macd_hi = macd_hist.rolling(window, min_periods=lookback).max().shift(lookback)
    prev_macd_lo = macd_hist.rolling(window, min_periods=lookback).min().shift(lookback)

    current_macd_hi = macd_hist.rolling(lookback, min_periods=5).max()
    current_macd_lo = macd_hist.rolling(lookback, min_periods=5).min()

    bearish = (
        (close.rolling(lookback, min_periods=5).max() > prev_price_hi) &
        (current_macd_hi < prev_macd_hi) &
        prev_price_hi.notna() & prev_macd_hi.notna()
    )
    result[bearish] = -1.0

    bullish = (
        (close.rolling(lookback, min_periods=5).min() < prev_price_lo) &
        (current_macd_lo > prev_macd_lo) &
        prev_price_lo.notna() & prev_macd_lo.notna()
    )
    result[bullish] = 1.0

    return result


def divergence_strength(
    close: pd.Series,
    rsi: pd.Series,
    macd_hist: pd.Series,
    lookback: int = 10,
    window: int = 50,
) -> pd.Series:
    """Combined divergence score: sum of RSI + MACD divergence ∈ [-2, +2].

    ±2 = both RSI and MACD diverging (strongest signal).
    ±1 = one diverging.
     0 = no divergence.
    """
    rsi_div = rsi_divergence(close, rsi, lookback, window)
    macd_div = macd_divergence(close, macd_hist, lookback, window)
    return rsi_div + macd_div


# ─── 2. Fibonacci Retracement ───────────────────────────────────────────────

def fibonacci_features(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    swing_lookback: int = 50,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Fibonacci retracement level proximity features.

    Finds the recent swing high/low over `swing_lookback` bars, computes
    Fibonacci levels (38.2%, 50%, 61.8%), and measures how close the
    current price is to each level.

    Returns:
        fib_retrace_pos   : position in swing range [0=swing_low, 1=swing_high]
        fib_nearest_level : nearest Fibonacci level (0.382, 0.500, or 0.618)
        fib_dist          : absolute distance to nearest Fib level (in range units)
    """
    swing_hi = high.rolling(swing_lookback, min_periods=10).max()
    swing_lo = low.rolling(swing_lookback, min_periods=10).min()
    swing_range = (swing_hi - swing_lo).clip(lower=1e-9)

    # Position in range [0, 1]: 0 = at swing low, 1 = at swing high
    retrace_pos = (close - swing_lo) / swing_range

    # Fibonacci levels as retracement from high
    fib_levels = np.array([0.382, 0.500, 0.618])

    # Distance to nearest Fib level
    fib_nearest = pd.Series(np.nan, index=close.index)
    fib_dist = pd.Series(np.nan, index=close.index)

    for i in range(len(close)):
        pos = retrace_pos.iloc[i]
        if np.isnan(pos):
            continue
        distances = np.abs(pos - fib_levels)
        nearest_idx = distances.argmin()
        fib_nearest.iloc[i] = fib_levels[nearest_idx]
        fib_dist.iloc[i] = distances[nearest_idx]

    return retrace_pos, fib_nearest, fib_dist


def fibonacci_features_vectorized(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    swing_lookback: int = 50,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Vectorized Fibonacci features (fast, no loop).

    Returns:
        fib_retrace_pos   : position in swing range [0, 1]
        fib_nearest_level : nearest Fib level (0.382, 0.500, 0.618)
        fib_dist          : distance to nearest Fib level (in range units)
    """
    swing_hi = high.rolling(swing_lookback, min_periods=10).max()
    swing_lo = low.rolling(swing_lookback, min_periods=10).min()
    swing_range = (swing_hi - swing_lo).clip(lower=1e-9)

    retrace_pos = (close - swing_lo) / swing_range

    # Distance to each Fib level
    fib_levels = np.array([0.382, 0.500, 0.618])
    pos_vals = retrace_pos.values.astype(float)
    nan_mask = np.isnan(pos_vals)

    # Fill NaN positions with 0.5 for computation, then mask back
    safe_pos = np.where(nan_mask, 0.5, pos_vals)
    dists = np.abs(safe_pos[:, None] - fib_levels[None, :])  # (N, 3)

    fib_dist_vals = dists.min(axis=1)
    nearest_idx = dists.argmin(axis=1)
    fib_nearest_vals = fib_levels[nearest_idx]

    # Restore NaN where input was NaN
    fib_dist_vals[nan_mask] = np.nan
    fib_nearest_vals[nan_mask] = np.nan

    fib_dist = pd.Series(fib_dist_vals, index=close.index)
    fib_nearest = pd.Series(fib_nearest_vals, index=close.index)

    return retrace_pos, fib_nearest, fib_dist


# ─── 3. Volume Climax Detection ─────────────────────────────────────────────

def volume_climax(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    vol_lookback: int = 50,
    zscore_thresh: float = 2.0,
    price_lookback: int = 20,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Detect volume climax events at price extremes.

    A volume climax occurs when volume spikes (>2σ above mean) at a
    swing high or low — indicating exhaustion and potential reversal.

    Returns:
        vol_climax_signal : +1 climax at low (bullish), −1 climax at high (bearish)
        vol_zscore        : volume z-score (how extreme is current volume)
        vol_climax_intensity : vol_zscore × price_extreme_score [0, ∞)
    """
    vol_mean = volume.rolling(vol_lookback, min_periods=10).mean()
    vol_std = volume.rolling(vol_lookback, min_periods=10).std().clip(lower=1e-9)
    vol_z = (volume - vol_mean) / vol_std

    # Is price at a local extreme?
    price_hi = high.rolling(price_lookback, min_periods=5).max()
    price_lo = low.rolling(price_lookback, min_periods=5).min()
    price_range = (price_hi - price_lo).clip(lower=1e-9)

    # How close to the high or low? [0, 1]
    near_high = (close - price_lo) / price_range  # 1.0 = at high
    near_low = 1 - near_high                      # 1.0 = at low

    is_vol_spike = vol_z > zscore_thresh

    # Climax at high (bearish) = volume spike + price near high
    climax_at_high = is_vol_spike & (near_high > 0.8)
    # Climax at low (bullish) = volume spike + price near low
    climax_at_low = is_vol_spike & (near_low > 0.8)

    signal = pd.Series(0.0, index=close.index)
    signal[climax_at_low] = 1.0
    signal[climax_at_high] = -1.0

    # Intensity: how extreme the climax is
    intensity = vol_z.clip(lower=0) * pd.concat([near_high, near_low], axis=1).max(axis=1)
    intensity[~is_vol_spike] = 0.0

    return signal, vol_z, intensity


# ─── 4. Intermarket Proxies ─────────────────────────────────────────────────

def intermarket_proxies(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Intermarket-style indicators derived from single-asset data.

    Without external data (bonds, VIX, commodities), we approximate:

    1. vol_regime_return: Realized vol regime × recent return direction.
       High-vol bearish moves are more likely to continue (fear cascades).
       High-vol bullish moves more likely to mean-revert (short squeezes).

    2. momentum_quality: Trend strength confirmed by declining volatility.
       A move on falling vol = healthy trend. Rising vol = unstable.

    3. exhaustion_index: Combination of extreme readings suggesting
       a move is running out of steam (overbought + declining volume + narrow range).

    Returns:
        vol_regime_return   : vol_rank × signed_return
        momentum_quality    : momentum × inverse_vol_change
        exhaustion_index    : composite exhaustion score [0, 1]
    """
    # Realized vol rank (percentile over 200 bars)
    returns = close.pct_change(1)
    realized_vol = returns.rolling(20, min_periods=5).std()
    vol_rank = realized_vol.rolling(200, min_periods=20).rank(pct=True)

    # Recent return direction (5-bar momentum)
    momentum_5 = close.pct_change(5)

    # 1. Vol-regime × return: captures fear cascades vs short squeezes
    vol_regime_return = vol_rank * np.sign(momentum_5)

    # 2. Momentum quality: strong momentum on calm vol = sustainable
    vol_change = realized_vol.pct_change(10)
    # Invert: falling vol → positive quality, rising vol → negative
    momentum_quality = momentum_5.abs() * (-vol_change).clip(-1, 1)

    # 3. Exhaustion index: composite [0, 1]
    # Components: extreme return rank + declining volume + narrowing range
    ret_rank = momentum_5.rolling(100, min_periods=20).rank(pct=True)
    ret_extreme = ((ret_rank > 0.9) | (ret_rank < 0.1)).astype(float)

    vol_declining = (volume < volume.rolling(20, min_periods=5).mean()).astype(float)

    hl_range = (high - low) / close.clip(lower=1e-9)
    range_narrowing = (
        hl_range < hl_range.rolling(20, min_periods=5).median()
    ).astype(float)

    exhaustion_index = (ret_extreme + vol_declining + range_narrowing) / 3.0

    return vol_regime_return, momentum_quality, exhaustion_index


# ─── Public API ──────────────────────────────────────────────────────────────

def compute_murphy_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all advanced Murphy features to a DataFrame.

    Requires: open, high, low, close, volume, rsi_14, macd_hist columns.
    """
    h, l, c, v = df["high"], df["low"], df["close"], df["volume"]

    # ── 1. Divergence ────────────────────────────────────────────────────────
    rsi = df.get("rsi_14")
    macd_h = df.get("macd_hist")

    if rsi is not None and macd_h is not None:
        df["div_rsi"] = rsi_divergence(c, rsi)
        df["div_macd"] = macd_divergence(c, macd_h)
        df["div_strength"] = df["div_rsi"] + df["div_macd"]
    else:
        df["div_rsi"] = 0.0
        df["div_macd"] = 0.0
        df["div_strength"] = 0.0

    # ── 2. Fibonacci ─────────────────────────────────────────────────────────
    df["fib_retrace_pos"], df["fib_nearest_level"], df["fib_dist"] = (
        fibonacci_features_vectorized(h, l, c)
    )

    # ── 3. Volume climax ─────────────────────────────────────────────────────
    df["vol_climax_signal"], df["vol_climax_zscore"], df["vol_climax_intensity"] = (
        volume_climax(h, l, c, v)
    )

    # ── 4. Intermarket proxies ───────────────────────────────────────────────
    df["im_vol_regime_return"], df["im_momentum_quality"], df["im_exhaustion_index"] = (
        intermarket_proxies(c, h, l, v)
    )

    return df
