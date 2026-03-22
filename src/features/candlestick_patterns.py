"""Candlestick pattern detection — numeric features for ML models.

Based on Course 1 §5.3 (Candlestick Patterns Every Trader Needs to Know)
and §6.2 (The Secret of Candlestick Patterns):

Single-candle features:
  - body_ratio      : |close−open| / (high−low) ∈ [0,1] — Marubozu=1, Doji=0
  - upper_shadow     : upper wick / range ∈ [0,1]
  - lower_shadow     : lower wick / range ∈ [0,1]
  - is_doji          : body < 10% of range (indecision candle)
  - is_hammer        : +1 hammer (bullish), −1 shooting star (bearish)

Multi-candle patterns:
  - engulfing        : +1 bullish engulfing, −1 bearish engulfing
  - morning_eve_star : +1 morning star (bullish), −1 evening star (bearish)
  - three_soldiers   : +1 three white soldiers, −1 three black crows
  - inside_bar       : 1.0 if current bar is inside previous bar's range
  - tweezer          : +1 tweezer bottom, −1 tweezer top

All features are computed per-bar without future data.
Shift-by-1 is handled at the pipeline level (indicators.py), NOT here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_range(high: pd.Series, low: pd.Series) -> pd.Series:
    """High − Low, floored to avoid division by zero."""
    return (high - low).clip(lower=1e-9)


# ─── Single-candle shape features ────────────────────────────────────────────

def candle_body_ratio(
    o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series,
) -> pd.Series:
    """Body size relative to full range. Marubozu ≈ 1.0, Doji ≈ 0.0."""
    return (c - o).abs() / _safe_range(h, l)


def candle_upper_shadow(
    o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series,
) -> pd.Series:
    """Upper wick as fraction of range. Long upper shadow = selling pressure."""
    body_top = pd.concat([o, c], axis=1).max(axis=1)
    return (h - body_top) / _safe_range(h, l)


def candle_lower_shadow(
    o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series,
) -> pd.Series:
    """Lower wick as fraction of range. Long lower shadow = buying pressure."""
    body_bottom = pd.concat([o, c], axis=1).min(axis=1)
    return (body_bottom - l) / _safe_range(h, l)


def candle_direction(o: pd.Series, c: pd.Series) -> pd.Series:
    """Candle direction: +1 bull, −1 bear, 0 flat."""
    return np.sign(c - o)


def is_doji(
    o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series,
    thresh: float = 0.10,
) -> pd.Series:
    """1.0 if body < thresh × range (indecision candle)."""
    return (candle_body_ratio(o, h, l, c) < thresh).astype(float)


def is_hammer(
    o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series,
) -> pd.Series:
    """Detect hammer (+1) and shooting star (−1).

    Hammer: lower shadow > 2× body, upper shadow < 0.3× range, bullish.
    Shooting star: upper shadow > 2× body, lower shadow < 0.3× range, bearish.
    """
    rng = _safe_range(h, l)
    body = (c - o).abs()
    body_top = pd.concat([o, c], axis=1).max(axis=1)
    body_bot = pd.concat([o, c], axis=1).min(axis=1)
    upper = (h - body_top) / rng
    lower = (body_bot - l) / rng

    hammer = ((lower > 0.6) & (upper < 0.3) & (body / rng < 0.35)).astype(float)
    shooting = ((upper > 0.6) & (lower < 0.3) & (body / rng < 0.35)).astype(float)
    return hammer - shooting


# ─── Multi-candle pattern features ───────────────────────────────────────────

def engulfing(
    o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series,
) -> pd.Series:
    """Detect engulfing patterns: +1 bullish, −1 bearish.

    Bullish engulfing: prev candle is bearish, current candle's body fully
    engulfs the previous candle's body, current is bullish.
    """
    prev_o, prev_c = o.shift(1), c.shift(1)
    prev_bear = prev_c < prev_o
    prev_bull = prev_c > prev_o
    curr_bull = c > o
    curr_bear = c < o

    body_bot = pd.concat([o, c], axis=1).min(axis=1)
    body_top = pd.concat([o, c], axis=1).max(axis=1)
    prev_body_bot = pd.concat([prev_o, prev_c], axis=1).min(axis=1)
    prev_body_top = pd.concat([prev_o, prev_c], axis=1).max(axis=1)

    bull_engulf = (
        prev_bear & curr_bull &
        (body_bot <= prev_body_bot) & (body_top >= prev_body_top)
    ).astype(float)

    bear_engulf = (
        prev_bull & curr_bear &
        (body_bot <= prev_body_bot) & (body_top >= prev_body_top)
    ).astype(float)

    return bull_engulf - bear_engulf


def morning_evening_star(
    o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series,
) -> pd.Series:
    """Detect morning star (+1) and evening star (−1).

    Morning star: big bear candle → small body (doji/spinning) → big bull candle.
    Evening star: big bull candle → small body → big bear candle.
    """
    rng = _safe_range(h, l)
    body = (c - o).abs()
    body_ratio = body / rng

    # Big body = ratio > 0.6, small body = ratio < 0.3
    big_body = body_ratio > 0.6
    small_body = body_ratio < 0.3
    bull = c > o
    bear = c < o

    big_bear_2ago = bear.shift(2) & big_body.shift(2)
    small_1ago = small_body.shift(1)
    big_bull_now = bull & big_body

    morning = (big_bear_2ago & small_1ago & big_bull_now).astype(float)

    big_bull_2ago = bull.shift(2) & big_body.shift(2)
    big_bear_now = bear & big_body

    evening = (big_bull_2ago & small_1ago & big_bear_now).astype(float)

    return morning - evening


def three_soldiers_crows(
    o: pd.Series, c: pd.Series,
) -> pd.Series:
    """Detect three white soldiers (+1) and three black crows (−1).

    Three white soldiers: 3 consecutive bullish candles, each closing higher.
    Three black crows: 3 consecutive bearish candles, each closing lower.
    """
    bull = (c > o).astype(float)
    bear = (c < o).astype(float)
    higher_close = (c > c.shift(1)).astype(float)
    lower_close = (c < c.shift(1)).astype(float)

    soldiers = (
        (bull == 1) & (bull.shift(1) == 1) & (bull.shift(2) == 1) &
        (higher_close == 1) & (higher_close.shift(1) == 1)
    ).astype(float)

    crows = (
        (bear == 1) & (bear.shift(1) == 1) & (bear.shift(2) == 1) &
        (lower_close == 1) & (lower_close.shift(1) == 1)
    ).astype(float)

    return soldiers - crows


def inside_bar(h: pd.Series, l: pd.Series) -> pd.Series:
    """1.0 if current bar's range is entirely within previous bar's range.

    Inside bars indicate consolidation / indecision before a breakout.
    """
    return ((h <= h.shift(1)) & (l >= l.shift(1))).astype(float)


def tweezer(
    o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series,
    tol_pct: float = 0.001,
) -> pd.Series:
    """Detect tweezer bottom (+1) and tweezer top (−1).

    Tweezer bottom: two candles with nearly equal lows, first bearish, second bullish.
    Tweezer top: two candles with nearly equal highs, first bullish, second bearish.
    """
    price_tol = c * tol_pct  # tolerance scales with price level

    equal_lows = (l - l.shift(1)).abs() < price_tol
    equal_highs = (h - h.shift(1)).abs() < price_tol

    prev_bear = c.shift(1) < o.shift(1)
    prev_bull = c.shift(1) > o.shift(1)
    curr_bull = c > o
    curr_bear = c < o

    bottom = (equal_lows & prev_bear & curr_bull).astype(float)
    top = (equal_highs & prev_bull & curr_bear).astype(float)

    return bottom - top


# ─── Aggregate function ─────────────────────────────────────────────────────

def compute_candlestick_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all candlestick pattern features to a DataFrame with OHLCV columns.

    Returns the input DataFrame with new columns appended (no copy).
    """
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]

    # Single-candle shape
    df["candle_body_ratio"] = candle_body_ratio(o, h, l, c)
    df["candle_upper_shadow"] = candle_upper_shadow(o, h, l, c)
    df["candle_lower_shadow"] = candle_lower_shadow(o, h, l, c)
    df["candle_direction"] = candle_direction(o, c)
    df["candle_doji"] = is_doji(o, h, l, c)
    df["candle_hammer"] = is_hammer(o, h, l, c)

    # Multi-candle patterns
    df["candle_engulfing"] = engulfing(o, h, l, c)
    df["candle_morning_eve_star"] = morning_evening_star(o, h, l, c)
    df["candle_three_soldiers"] = three_soldiers_crows(o, c)
    df["candle_inside_bar"] = inside_bar(h, l)
    df["candle_tweezer"] = tweezer(o, h, l, c)

    return df
