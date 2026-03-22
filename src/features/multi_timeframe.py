"""Multi-timeframe trend alignment — numeric features for ML models.

Based on Course 1 §9.1–§9.4:
  - Multiple timeframes maximize reward/risk by aligning entry with trend
  - 3 timeframe layers: Entry TF → Trading TF → Trending TF (factor 3–6×)
  - Confluence: all timeframes agree → strongest signal
  - Non-confluence: counter-trend on entry TF → use strongest demand/supply zone

For 1-minute bars, we simulate higher timeframes by resampling:
  - 5m  (entry TF for scalping)
  - 15m (trading TF for scalping)
  - 60m (trending TF / swing)

Features produced:
  mtf_trend_5m      : EMA(9) > EMA(21) on 5m bars → +1 bull, −1 bear
  mtf_trend_15m     : EMA(9) > EMA(21) on 15m bars
  mtf_trend_60m     : EMA(9) > EMA(21) on 60m bars
  mtf_rsi_15m       : RSI(14) on 15m bars (rescaled to [-1,1])
  mtf_confluence     : sum of 3 trend signals ∈ [-3, +3] — alignment score
  mtf_aligned        : 1.0 if all 3 timeframes agree, else 0.0

Also from Murphy (Technical Analysis of the Financial Markets):
  donchian_pos       : position within Donchian channel [0, 1]
  donchian_breakout  : +1 new high, −1 new low, 0 otherwise
  ad_line_slope      : 10-bar slope of Accumulation/Distribution line
  psar_signal        : +1 price above PSAR, −1 below

All features use only past data. No lookahead.
Shift-by-1 is handled at the pipeline level (indicators.py), NOT here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ─── EMA helper ──────────────────────────────────────────────────────────────

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ─── Multi-timeframe resampling ──────────────────────────────────────────────

def _resample_trend(df: pd.DataFrame, period: str) -> pd.Series:
    """Resample OHLCV to a higher timeframe and compute EMA trend direction.

    Returns +1 (bullish), −1 (bearish) mapped back to original 1m index.
    """
    resampled = df[["open", "high", "low", "close", "volume"]].resample(period).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["close"])

    if len(resampled) < 21:
        return pd.Series(0.0, index=df.index)

    ema_fast = _ema(resampled["close"], 9)
    ema_slow = _ema(resampled["close"], 21)
    trend = np.sign(ema_fast - ema_slow)

    # Forward-fill back to 1m index (each higher TF bar covers N 1m bars)
    return trend.reindex(df.index, method="ffill").fillna(0.0)


def _resample_rsi(df: pd.DataFrame, period: str, n: int = 14) -> pd.Series:
    """Compute RSI on resampled timeframe, mapped back to 1m index.

    Returns RSI rescaled to [-1, 1] (centered at 50).
    """
    resampled = df["close"].resample(period).last().dropna()

    if len(resampled) < n + 5:
        return pd.Series(0.0, index=df.index)

    rsi_vals = _rsi(resampled, n)
    # Rescale: 0→-1, 50→0, 100→+1
    rsi_centered = (rsi_vals - 50) / 50

    return rsi_centered.reindex(df.index, method="ffill").fillna(0.0)


# ─── Murphy indicators ──────────────────────────────────────────────────────

def _donchian(
    high: pd.Series, low: pd.Series, close: pd.Series, n: int = 20,
) -> tuple[pd.Series, pd.Series]:
    """Donchian channel position and breakout signal.

    donchian_pos: (close − channel_low) / (channel_high − channel_low) ∈ [0, 1]
    donchian_breakout: +1 if close = channel_high, −1 if close = channel_low
    """
    ch_high = high.rolling(n, min_periods=5).max()
    ch_low = low.rolling(n, min_periods=5).min()
    ch_range = (ch_high - ch_low).clip(lower=1e-9)

    pos = (close - ch_low) / ch_range

    breakout = pd.Series(0.0, index=close.index)
    breakout[close >= ch_high] = 1.0
    breakout[close <= ch_low] = -1.0

    return pos, breakout


def _ad_line_slope(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    slope_bars: int = 10,
) -> pd.Series:
    """Accumulation/Distribution line normalized slope.

    AD = cumsum(CLV × volume), where CLV = ((close−low) − (high−close)) / (high−low).
    Returns the 10-bar rate-of-change of the AD line, normalized by volume.
    """
    clv = ((close - low) - (high - close)) / (high - low).clip(lower=1e-9)
    ad = (clv * volume).cumsum()
    vol_ma = volume.rolling(slope_bars, min_periods=3).mean().clip(lower=1e-9)
    return ad.diff(slope_bars) / (vol_ma * slope_bars)


def _parabolic_sar(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    af_start: float = 0.02,
    af_step: float = 0.02,
    af_max: float = 0.20,
) -> pd.Series:
    """Parabolic SAR direction signal: +1 if price above SAR, −1 if below.

    Simplified implementation suitable for feature generation.
    """
    n = len(close)
    if n < 5:
        return pd.Series(0.0, index=close.index)

    sar = np.zeros(n)
    direction = np.ones(n)  # 1 = long, -1 = short
    af = af_start
    ep = high.iloc[0]  # extreme point
    sar[0] = low.iloc[0]

    for i in range(1, n):
        prev_sar = sar[i - 1]

        if direction[i - 1] == 1:  # long position
            sar[i] = prev_sar + af * (ep - prev_sar)
            sar[i] = min(sar[i], low.iloc[i - 1])
            if i >= 2:
                sar[i] = min(sar[i], low.iloc[i - 2])

            if low.iloc[i] < sar[i]:  # reversal to short
                direction[i] = -1
                sar[i] = ep
                ep = low.iloc[i]
                af = af_start
            else:
                direction[i] = 1
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + af_step, af_max)
        else:  # short position
            sar[i] = prev_sar + af * (ep - prev_sar)
            sar[i] = max(sar[i], high.iloc[i - 1])
            if i >= 2:
                sar[i] = max(sar[i], high.iloc[i - 2])

            if high.iloc[i] > sar[i]:  # reversal to long
                direction[i] = 1
                sar[i] = ep
                ep = high.iloc[i]
                af = af_start
            else:
                direction[i] = -1
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + af_step, af_max)

    return pd.Series(direction, index=close.index)


# ─── Public API ──────────────────────────────────────────────────────────────

def compute_multi_timeframe_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add multi-timeframe and Murphy indicator features.

    Requires: DatetimeIndex, open/high/low/close/volume columns.
    """
    h, l, c, v = df["high"], df["low"], df["close"], df["volume"]

    # ── Multi-timeframe trend alignment ──────────────────────────────────────
    df["mtf_trend_5m"] = _resample_trend(df, "5min")
    df["mtf_trend_15m"] = _resample_trend(df, "15min")
    df["mtf_trend_60m"] = _resample_trend(df, "60min")
    df["mtf_rsi_15m"] = _resample_rsi(df, "15min")

    # Confluence: sum of trend signals (−3 to +3)
    df["mtf_confluence"] = (
        df["mtf_trend_5m"] + df["mtf_trend_15m"] + df["mtf_trend_60m"]
    )
    # All aligned = 1.0
    df["mtf_aligned"] = (df["mtf_confluence"].abs() == 3).astype(float)

    # ── Murphy indicators ────────────────────────────────────────────────────
    df["donchian_pos"], df["donchian_breakout"] = _donchian(h, l, c)
    df["ad_line_slope"] = _ad_line_slope(h, l, c, v)
    df["psar_signal"] = _parabolic_sar(h, l, c)

    return df
