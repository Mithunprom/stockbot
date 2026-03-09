"""Technical indicator pipeline using pandas-ta.

Computes the full indicator set per ticker from OHLCV bars.
All indicators are shifted by 1 bar to prevent lookahead bias.

Usage:
    df = compute_indicators(ohlcv_df)
    # Returns DataFrame with all indicator columns appended.
"""

from __future__ import annotations

import logging
import structlog
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pandas_ta as ta

logger = structlog.get_logger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Column expectations ──────────────────────────────────────────────────────

_REQUIRED_COLS = {"open", "high", "low", "close", "volume"}


def _validate(df: pd.DataFrame) -> None:
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"OHLCV DataFrame missing columns: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")


# ─── Indicator computation ────────────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame, shift: bool = True) -> pd.DataFrame:
    """Compute the full indicator set on an OHLCV DataFrame.

    Args:
        df: DataFrame with DatetimeIndex and open/high/low/close/volume columns.
        shift: If True (default), shift all indicator columns by 1 bar to prevent
               lookahead. Must be True for live training/inference.

    Returns:
        A new DataFrame with all indicator columns appended. The original OHLCV
        columns are preserved.
    """
    _validate(df)
    out = df.copy()

    # ── Trend ──────────────────────────────────────────────────────────────────
    out["ema_9"] = ta.ema(out["close"], length=9)
    out["ema_21"] = ta.ema(out["close"], length=21)
    out["ema_50"] = ta.ema(out["close"], length=50)
    out["ema_cross_9_21"] = (out["ema_9"] > out["ema_21"]).astype(float)
    out["ema_cross_21_50"] = (out["ema_21"] > out["ema_50"]).astype(float)

    macd = ta.macd(out["close"], fast=12, slow=26, signal=9)
    if macd is not None:
        out["macd"] = macd["MACD_12_26_9"]
        out["macd_signal"] = macd["MACDs_12_26_9"]
        out["macd_hist"] = macd["MACDh_12_26_9"]

    adx = ta.adx(out["high"], out["low"], out["close"], length=14)
    if adx is not None:
        out["adx"] = adx["ADX_14"]
        out["dmp"] = adx["DMP_14"]   # +DI
        out["dmn"] = adx["DMN_14"]   # -DI

    # ── Momentum ───────────────────────────────────────────────────────────────
    out["rsi_14"] = ta.rsi(out["close"], length=14)

    stoch = ta.stoch(out["high"], out["low"], out["close"], k=14, d=3)
    if stoch is not None:
        out["stoch_k"] = stoch["STOCHk_14_3_3"]
        out["stoch_d"] = stoch["STOCHd_14_3_3"]

    out["willr_14"] = ta.willr(out["high"], out["low"], out["close"], length=14)
    out["cci_20"] = ta.cci(out["high"], out["low"], out["close"], length=20)
    out["mom_10"] = ta.mom(out["close"], length=10)
    out["roc_10"] = ta.roc(out["close"], length=10)

    # ── Volatility ─────────────────────────────────────────────────────────────
    bb = ta.bbands(out["close"], length=20, std=2)
    if bb is not None:
        # pandas-ta ≥0.4: column names gained an extra _2.0 suffix
        bb_upper_col = next((c for c in bb.columns if c.startswith("BBU")), None)
        bb_mid_col   = next((c for c in bb.columns if c.startswith("BBM")), None)
        bb_lower_col = next((c for c in bb.columns if c.startswith("BBL")), None)
        if bb_upper_col:
            out["bb_upper"] = bb[bb_upper_col]
            out["bb_mid"]   = bb[bb_mid_col]
            out["bb_lower"] = bb[bb_lower_col]
        out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / out["bb_mid"]
        out["bb_pct"] = (out["close"] - out["bb_lower"]) / (
            out["bb_upper"] - out["bb_lower"]
        )

    out["atr_14"] = ta.atr(out["high"], out["low"], out["close"], length=14)
    out["atr_pct"] = out["atr_14"] / out["close"]

    kc = ta.kc(out["high"], out["low"], out["close"], length=20)
    if kc is not None:
        # Column order: Lower, Basis, Upper (KCLe, KCBe, KCUe)
        kc_lower_col = next((c for c in kc.columns if "Le" in c), None)
        kc_mid_col   = next((c for c in kc.columns if "Be" in c), None)
        kc_upper_col = next((c for c in kc.columns if "Ue" in c), None)
        if kc_upper_col:
            out["kc_upper"] = kc[kc_upper_col]
            out["kc_mid"]   = kc[kc_mid_col]
            out["kc_lower"] = kc[kc_lower_col]

    # ── Volume ─────────────────────────────────────────────────────────────────
    out["vwap"] = ta.vwap(out["high"], out["low"], out["close"], out["volume"])
    out["obv"] = ta.obv(out["close"], out["volume"])
    out["obv_pct"] = out["obv"].pct_change(periods=5)
    out["mfi_14"] = ta.mfi(out["high"], out["low"], out["close"], out["volume"], length=14)

    # Volume relative to 20-bar average
    out["vol_ratio"] = out["volume"] / out["volume"].rolling(20).mean()

    # ── Price action features ──────────────────────────────────────────────────
    out["returns_1b"] = out["close"].pct_change(1)
    out["returns_5b"] = out["close"].pct_change(5)
    out["returns_15b"] = out["close"].pct_change(15)
    out["high_low_range"] = (out["high"] - out["low"]) / out["close"]
    out["close_vs_high"] = (out["close"] - out["low"]) / (
        out["high"] - out["low"] + 1e-9
    )   # 0=low, 1=high (intrabar strength)

    # Gap (vs. prior close)
    out["gap_pct"] = (out["open"] - out["close"].shift(1)) / out["close"].shift(1)

    # ── VPIN (Volume-Synchronized Probability of Informed Trading) ─────────────
    # Bulk Volume Classification (BVC) approximation of Easley et al.:
    # Each bar's volume is split into buy/sell using price direction within the bar.
    # buy_vol  = V × 0.5 × (1 + (close - open) / (high - low + ε))
    # sell_vol = V - buy_vol
    # VPIN = rolling_mean(|buy_vol - sell_vol| / V, window=50)
    # Range: [0, 1]. High VPIN → informed trading likely → larger upcoming move.
    _hl = (out["high"] - out["low"]).clip(lower=1e-9)
    _buy_frac = 0.5 * (1 + (out["close"] - out["open"]) / _hl)
    _buy_vol  = out["volume"] * _buy_frac
    _sell_vol = out["volume"] * (1 - _buy_frac)
    _imbalance = (_buy_vol - _sell_vol).abs() / (out["volume"].clip(lower=1e-9))
    out["vpin_50"] = _imbalance.rolling(50).mean()
    # Normalized VPIN z-score (deviation from its own 200-bar mean)
    _vpin_mean = out["vpin_50"].rolling(200).mean()
    _vpin_std  = out["vpin_50"].rolling(200).std().clip(lower=1e-9)
    out["vpin_zscore"] = (out["vpin_50"] - _vpin_mean) / _vpin_std

    # ── Intraday Seasonality ───────────────────────────────────────────────────
    # Markets have strong time-of-day patterns: open surge, lunch lull, close rush.
    # All features are derived purely from the bar's timestamp — zero lookahead.
    _idx = out.index
    # Minutes since market open (9:30 ET). Works correctly for UTC-stored bars.
    # Alpaca IEX bars: 14:30–21:00 UTC = 9:30–16:00 ET
    _min_since_open = ((_idx.hour * 60 + _idx.minute) - (9 * 60 + 30)) % (24 * 60)
    _min_since_open = pd.Series(_min_since_open, index=_idx).clip(0, 389)
    out["min_since_open"] = _min_since_open / 389.0  # normalized 0→1

    # Day-of-week (0=Mon … 4=Fri) as fraction — Monday=0.0, Friday=1.0
    out["day_of_week"] = _idx.dayofweek / 4.0

    # Binary session windows
    out["is_open_window"]  = (_min_since_open <= 30).astype(float)   # first 30 min
    out["is_close_window"] = (_min_since_open >= 360).astype(float)  # last 30 min
    out["is_lunch"]        = ((_min_since_open >= 150) & (_min_since_open <= 210)).astype(float)

    # Time-to-close fraction (1.0 = just opened, 0.0 = about to close)
    out["time_to_close"] = (389 - _min_since_open) / 389.0

    # Historical volume seasonality: vol relative to the same minute's 20-day avg
    # Uses a pivot on minute-of-day to build the seasonal baseline.
    if len(out) >= 390 * 5:   # need at least 5 days of data
        _min_key = (_idx.hour * 60 + _idx.minute).astype(str)
        _vol_series = out["volume"].copy()
        _vol_series.index = pd.MultiIndex.from_arrays([_min_key, _idx], names=["min_key", "time"])
        _seasonal_vol = (
            _vol_series.groupby(level="min_key")
            .transform(lambda x: x.rolling(20, min_periods=5).mean())
        )
        _seasonal_vol.index = _idx
        out["vol_seasonal_ratio"] = (out["volume"] / _seasonal_vol.clip(lower=1)).clip(upper=10)
    else:
        out["vol_seasonal_ratio"] = np.nan

    # ── Prevent lookahead ─────────────────────────────────────────────────────
    if shift:
        indicator_cols = [c for c in out.columns if c not in df.columns]
        out[indicator_cols] = out[indicator_cols].shift(1)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


# ─── Per-ticker batch computation ─────────────────────────────────────────────

def compute_indicators_for_universe(
    bars: dict[str, pd.DataFrame], shift: bool = True
) -> dict[str, pd.DataFrame]:
    """Compute indicators for a dict of {ticker: ohlcv_df}.

    Silently skips tickers with insufficient data (< 60 bars).
    """
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


# ─── Feature column registry ──────────────────────────────────────────────────

FEATURE_COLUMNS: list[str] = [
    # Trend
    "ema_9", "ema_21", "ema_50", "ema_cross_9_21", "ema_cross_21_50",
    "macd", "macd_signal", "macd_hist",
    "adx", "dmp", "dmn",
    # Momentum
    "rsi_14", "stoch_k", "stoch_d", "willr_14", "cci_20", "mom_10", "roc_10",
    # Volatility
    "bb_upper", "bb_mid", "bb_lower", "bb_width", "bb_pct",
    "atr_14", "atr_pct",
    "kc_upper", "kc_mid", "kc_lower",
    # Volume
    "vwap", "obv", "obv_pct", "mfi_14", "vol_ratio",
    # Price action
    "returns_1b", "returns_5b", "returns_15b",
    "high_low_range", "close_vs_high", "gap_pct",
    # VPIN — informed trading probability
    "vpin_50", "vpin_zscore",
    # Intraday seasonality
    "min_since_open", "day_of_week",
    "is_open_window", "is_close_window", "is_lunch", "time_to_close",
    "vol_seasonal_ratio",
    # GEX injected externally (see features/gex.py)
    "gex_net", "gex_zscore", "gex_call_pct",
]
