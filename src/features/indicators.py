"""Technical indicator pipeline — pure pandas/numpy, no external TA library.

Computes the full indicator set per ticker from OHLCV bars.
All indicators are shifted by 1 bar to prevent lookahead bias.

Usage:
    df = compute_indicators(ohlcv_df)
    # Returns DataFrame with all indicator columns appended.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

_REQUIRED_COLS = {"open", "high", "low", "close", "volume"}


def _validate(df: pd.DataFrame) -> None:
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"OHLCV DataFrame missing columns: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")


# ─── Pure-pandas indicator helpers ───────────────────────────────────────────

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = _ema(close, fast) - _ema(close, slow)
    signal_line = _ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def _bbands(close: pd.Series, n: int = 20, std: float = 2.0):
    mid = close.rolling(n).mean()
    sigma = close.rolling(n).std()
    upper = mid + std * sigma
    lower = mid - std * sigma
    return upper, mid, lower


def _stoch(high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3):
    lowest_low = low.rolling(k).min()
    highest_high = high.rolling(k).max()
    k_pct = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-9)
    d_pct = k_pct.rolling(d).mean()
    return k_pct, d_pct


def _willr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    highest_high = high.rolling(n).max()
    lowest_low = low.rolling(n).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low + 1e-9)


def _cci(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 20) -> pd.Series:
    tp = (high + low + close) / 3
    ma = tp.rolling(n).mean()
    mad = tp.rolling(n).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - ma) / (0.015 * mad + 1e-9)


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14):
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    dm_plus = np.where((high - prev_high) > (prev_low - low), (high - prev_high).clip(lower=0), 0.0)
    dm_minus = np.where((prev_low - low) > (high - prev_high), (prev_low - low).clip(lower=0), 0.0)
    dm_plus = pd.Series(dm_plus, index=close.index)
    dm_minus = pd.Series(dm_minus, index=close.index)
    atr_n = tr.ewm(alpha=1 / n, adjust=False).mean()
    di_plus = 100 * dm_plus.ewm(alpha=1 / n, adjust=False).mean() / atr_n.replace(0, np.nan)
    di_minus = 100 * dm_minus.ewm(alpha=1 / n, adjust=False).mean() / atr_n.replace(0, np.nan)
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-9)
    adx = dx.ewm(alpha=1 / n, adjust=False).mean()
    return adx, di_plus, di_minus


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def _mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, n: int = 14) -> pd.Series:
    tp = (high + low + close) / 3
    rmf = tp * volume
    pos_flow = rmf.where(tp > tp.shift(1), 0.0).rolling(n).sum()
    neg_flow = rmf.where(tp < tp.shift(1), 0.0).rolling(n).sum()
    mfr = pos_flow / neg_flow.replace(0, np.nan)
    return 100 - (100 / (1 + mfr))


def _keltner(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 20, mult: float = 2.0):
    mid = _ema(close, n)
    atr_n = _atr(high, low, close, n)
    return mid + mult * atr_n, mid, mid - mult * atr_n


def _vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    tp = (high + low + close) / 3
    return (tp * volume).cumsum() / volume.cumsum().replace(0, np.nan)


# ─── Main indicator pipeline ──────────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame, shift: bool = True) -> pd.DataFrame:
    """Compute full indicator set on an OHLCV DataFrame.

    Args:
        df: DataFrame with DatetimeIndex and open/high/low/close/volume columns.
        shift: Shift all indicator columns by 1 bar to prevent lookahead (default True).

    Returns:
        DataFrame with all indicator columns appended.
    """
    _validate(df)
    out = df.copy()
    o, h, l, c, v = out["open"], out["high"], out["low"], out["close"], out["volume"]

    # ── Trend ──────────────────────────────────────────────────────────────────
    out["ema_9"] = _ema(c, 9)
    out["ema_21"] = _ema(c, 21)
    out["ema_50"] = _ema(c, 50)
    out["ema_cross_9_21"] = (out["ema_9"] > out["ema_21"]).astype(float)
    out["ema_cross_21_50"] = (out["ema_21"] > out["ema_50"]).astype(float)

    macd_line, macd_sig, macd_hist = _macd(c)
    out["macd"] = macd_line
    out["macd_signal"] = macd_sig
    out["macd_hist"] = macd_hist

    adx_val, dmp, dmn = _adx(h, l, c)
    out["adx"] = adx_val
    out["dmp"] = dmp
    out["dmn"] = dmn

    # ── Momentum ───────────────────────────────────────────────────────────────
    out["rsi_14"] = _rsi(c, 14)
    out["stoch_k"], out["stoch_d"] = _stoch(h, l, c)
    out["willr_14"] = _willr(h, l, c)
    out["cci_20"] = _cci(h, l, c)
    out["mom_10"] = c.diff(10)
    out["roc_10"] = c.pct_change(10) * 100

    # ── Volatility ─────────────────────────────────────────────────────────────
    out["bb_upper"], out["bb_mid"], out["bb_lower"] = _bbands(c)
    out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / out["bb_mid"].replace(0, np.nan)
    out["bb_pct"] = (c - out["bb_lower"]) / (out["bb_upper"] - out["bb_lower"] + 1e-9)

    out["atr_14"] = _atr(h, l, c, 14)
    out["atr_pct"] = out["atr_14"] / c.replace(0, np.nan)

    out["kc_upper"], out["kc_mid"], out["kc_lower"] = _keltner(h, l, c)

    # ── Volume ─────────────────────────────────────────────────────────────────
    out["vwap"] = _vwap(h, l, c, v)
    out["obv"] = _obv(c, v)
    out["obv_pct"] = out["obv"].pct_change(5)
    out["mfi_14"] = _mfi(h, l, c, v)
    out["vol_ratio"] = v / v.rolling(20).mean().replace(0, np.nan)

    # ── Price action ───────────────────────────────────────────────────────────
    out["returns_1b"] = c.pct_change(1)
    out["returns_5b"] = c.pct_change(5)
    out["returns_15b"] = c.pct_change(15)
    out["high_low_range"] = (h - l) / c.replace(0, np.nan)
    out["close_vs_high"] = (c - l) / (h - l + 1e-9)
    out["gap_pct"] = (o - c.shift(1)) / c.shift(1).replace(0, np.nan)

    # ── VPIN ───────────────────────────────────────────────────────────────────
    _hl = (h - l).clip(lower=1e-9)
    _buy_frac = 0.5 * (1 + (c - o) / _hl)
    _buy_vol = v * _buy_frac
    _sell_vol = v * (1 - _buy_frac)
    _imbalance = (_buy_vol - _sell_vol).abs() / v.clip(lower=1e-9)
    out["vpin_50"] = _imbalance.rolling(50).mean()
    _vpin_mean = out["vpin_50"].rolling(200).mean()
    _vpin_std = out["vpin_50"].rolling(200).std().clip(lower=1e-9)
    out["vpin_zscore"] = (out["vpin_50"] - _vpin_mean) / _vpin_std

    # ── Intraday seasonality ───────────────────────────────────────────────────
    _idx = out.index
    _min_since_open = ((_idx.hour * 60 + _idx.minute) - (9 * 60 + 30)) % (24 * 60)
    _min_since_open = pd.Series(_min_since_open, index=_idx).clip(0, 389)
    out["min_since_open"] = _min_since_open / 389.0
    out["day_of_week"] = _idx.dayofweek / 4.0
    out["is_open_window"] = (_min_since_open <= 30).astype(float)
    out["is_close_window"] = (_min_since_open >= 360).astype(float)
    out["is_lunch"] = ((_min_since_open >= 150) & (_min_since_open <= 210)).astype(float)
    out["time_to_close"] = (389 - _min_since_open) / 389.0

    if len(out) >= 390 * 5:
        _min_key = (_idx.hour * 60 + _idx.minute).astype(str)
        _vol_series = v.copy()
        _vol_series.index = pd.MultiIndex.from_arrays([_min_key, _idx], names=["min_key", "time"])
        _seasonal_vol = (
            _vol_series.groupby(level="min_key")
            .transform(lambda x: x.rolling(20, min_periods=5).mean())
        )
        _seasonal_vol.index = _idx
        out["vol_seasonal_ratio"] = (v / _seasonal_vol.clip(lower=1)).clip(upper=10)
    else:
        out["vol_seasonal_ratio"] = np.nan

    # ── Regime ─────────────────────────────────────────────────────────────────
    from src.features.regime import compute_regime
    out["regime"] = compute_regime(out).astype(float)

    # ── Prevent lookahead ──────────────────────────────────────────────────────
    if shift:
        indicator_cols = [c for c in out.columns if c not in df.columns]
        out[indicator_cols] = out[indicator_cols].shift(1)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def compute_indicators_for_universe(
    bars: dict[str, pd.DataFrame], shift: bool = True
) -> dict[str, pd.DataFrame]:
    """Compute indicators for a dict of {ticker: ohlcv_df}."""
    results: dict[str, pd.DataFrame] = {}
    for ticker, df in bars.items():
        if len(df) < 60:
            logger.debug("skip_short_series: %s bars=%d", ticker, len(df))
            continue
        try:
            results[ticker] = compute_indicators(df, shift=shift)
        except Exception as exc:
            logger.error("indicator_error: %s: %s", ticker, exc)
    return results


FEATURE_COLUMNS: list[str] = [
    "ema_9", "ema_21", "ema_50", "ema_cross_9_21", "ema_cross_21_50",
    "macd", "macd_signal", "macd_hist",
    "adx", "dmp", "dmn",
    "rsi_14", "stoch_k", "stoch_d", "willr_14", "cci_20", "mom_10", "roc_10",
    "bb_upper", "bb_mid", "bb_lower", "bb_width", "bb_pct",
    "atr_14", "atr_pct",
    "kc_upper", "kc_mid", "kc_lower",
    "vwap", "obv", "obv_pct", "mfi_14", "vol_ratio",
    "returns_1b", "returns_5b", "returns_15b",
    "high_low_range", "close_vs_high", "gap_pct",
    "vpin_50", "vpin_zscore",
    "min_since_open", "day_of_week",
    "is_open_window", "is_close_window", "is_lunch", "time_to_close",
    "vol_seasonal_ratio",
    "gex_net", "gex_zscore", "gex_call_pct",
    "regime",
]
