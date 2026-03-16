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
    """Legacy cumulative VWAP (no daily reset) — kept for backward compat."""
    tp = (high + low + close) / 3
    return (tp * volume).cumsum() / volume.cumsum().replace(0, np.nan)


def _to_et_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Convert a DatetimeIndex to US/Eastern, handling both aware and naive."""
    if idx.tz is None:
        return idx.tz_localize("UTC").tz_convert("America/New_York")
    return idx.tz_convert("America/New_York")


def _vwap_daily(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """VWAP with proper daily reset using ET calendar dates.

    Groups by ET date so the daily accumulation resets at 9:30 AM ET, not
    at midnight UTC. The legacy _vwap() above accumulates from the start of
    the entire DataFrame which contaminates cross-session signal.

    Lookahead: none (cumulative within day, shifted at pipeline level).
    """
    tp = (high + low + close) / 3
    tpv = tp * volume
    # ET date guarantees the reset aligns with the trading session, not midnight UTC
    idx_et = _to_et_index(close.index)
    date_key = pd.Series(idx_et.date, index=close.index)
    cum_tpv = tpv.groupby(date_key).cumsum()
    cum_vol = volume.groupby(date_key).cumsum().replace(0, np.nan)
    return cum_tpv / cum_vol


def _orb_features(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr_14: pd.Series,
    orb_bars: int = 30,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Opening Range Breakout (ORB) features — first 30 bars of each session.

    Bug fix: must convert to ET timezone before computing minutes-since-open.
    DB timestamps are UTC (9:30 AM ET = 13:30 UTC). Using raw UTC hours
    would look for bars at 5:30 AM ET which don't exist → all NaN.

    Returns:
        orb_range_atr : (ORB_high − ORB_low) / ATR14  — daily volatility regime [0,∞)
        orb_dev       : (close − ORB_low) / (ORB_high − ORB_low)  — position in range [0,1]
        orb_break_up  : 1.0 if close > ORB_high else 0.0
        orb_break_dn  : 1.0 if close < ORB_low  else 0.0
    """
    idx_et = _to_et_index(close.index)
    min_since_open = (idx_et.hour * 60 + idx_et.minute) - (9 * 60 + 30)
    is_orb = pd.Series((min_since_open >= 0) & (min_since_open < orb_bars), index=close.index)

    # Group by ET date so ORB resets correctly each session
    date_key = pd.Series(idx_et.date, index=close.index)

    # transform('max'/'min') on masked series: NaN (outside ORB) excluded,
    # so each bar of the day gets the ORB high/low as a day-constant value.
    orb_high = high.where(is_orb).groupby(date_key).transform("max")
    orb_low  = low.where(is_orb).groupby(date_key).transform("min")

    orb_range     = (orb_high - orb_low).clip(lower=0)
    orb_range_atr = orb_range / atr_14.replace(0, np.nan)
    orb_dev       = (close - orb_low) / (orb_range + 1e-9)
    orb_break_up  = (close > orb_high).astype(float)
    orb_break_dn  = (close < orb_low).astype(float)

    return orb_range_atr, orb_dev, orb_break_up, orb_break_dn


def _vwap_slope(vwap: pd.Series) -> tuple[pd.Series, pd.Series]:
    """VWAP rate-of-change over 5 and 15 bars, normalized by VWAP level.

    Dividing by the lagged VWAP converts the slope to fractional units (≈ %/bar)
    so that a +0.001 reading means the same intraday trend strength regardless
    of whether the stock trades at $10 or $500.

    Returns (slope_5, slope_15).
    Lookahead: none (uses only past VWAP values).
    """
    slope_5  = vwap.diff(5)  / vwap.shift(5).replace(0, np.nan)
    slope_15 = vwap.diff(15) / vwap.shift(15).replace(0, np.nan)
    return slope_5, slope_15


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

    # ── Volume & VWAP ──────────────────────────────────────────────────────────
    # Daily-reset VWAP: resets each session, no cross-day contamination
    out["vwap"] = _vwap(h, l, c, v)            # legacy (kept for compat)
    vwap_d = _vwap_daily(h, l, c, v)
    out["vwap_daily"] = vwap_d

    # VWAP deviation: (close − VWAP) / VWAP → %-units, comparable across tickers
    out["vwap_dev"] = (c - vwap_d) / vwap_d.replace(0, np.nan)
    out["vwap_above"] = (c > vwap_d).astype(float)

    # VWAP slope: fractional change over 5 and 15 bars (intraday trend strength)
    out["vwap_slope_5"], out["vwap_slope_15"] = _vwap_slope(vwap_d)

    # ORB: opening-range features — normalized by ATR so signal is cross-ticker
    out["orb_range_atr"], out["orb_dev"], out["orb_break_up"], out["orb_break_dn"] = (
        _orb_features(h, l, c, out["atr_14"])
    )

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

    # ── Regime-Time interaction ───────────────────────────────────────────────
    # ATR percentile × time_to_close: a volatility spike at 9:30 AM has very
    # different predictive value than the same spike at 3:55 PM.
    # Rolling ATR percentile over 200 bars gives [0, 1] regime strength.
    _atr_pctile = out["atr_pct"].rolling(200, min_periods=20).rank(pct=True)
    out["regime_time"] = _atr_pctile * out["time_to_close"]

    # ── Composite interaction features ────────────────────────────────────────
    # Informed breakout intensity: ORB breakout conviction × informed flow
    out["orb_vpin_interact"] = out["orb_dev"] * out["vpin_50"]
    # Mean-reversion urgency: VWAP deviation that persists late in day
    out["vwap_time_interact"] = out["vwap_dev"] * out["time_to_close"]
    # Volatility confirmation: high ATR with high relative volume = real move
    out["atr_vol_interact"] = out["atr_pct"] * out["vol_ratio"]

    # ── Prevent lookahead ──────────────────────────────────────────────────────
    if shift:
        indicator_cols = [c for c in out.columns if c not in df.columns]
        out[indicator_cols] = out[indicator_cols].shift(1)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def compute_indicators_for_universe(
    bars: dict[str, pd.DataFrame], shift: bool = True
) -> dict[str, pd.DataFrame]:
    """Compute indicators for a dict of {ticker: ohlcv_df}.

    After per-ticker indicators are computed, adds cross-sectional relative
    strength features that require the full universe:
        rs_15m : ticker 15m return − universe-mean 15m return
        rs_1m  : ticker 1m  return − universe-mean 1m  return

    These capture idiosyncratic alpha (e.g. NVDA outrunning the group on
    earnings day) and are among the strongest short-term IC features in
    the literature.  The universe-mean is computed from the same already-
    shifted returns_15b column, so no lookahead is introduced.
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

    if len(results) < 2:
        return results

    # ── Cross-sectional relative strength ─────────────────────────────────────
    # Align all tickers on a common timestamp index, take universe average,
    # then subtract from each ticker's own shifted returns.
    # Weight of self in universe mean: 1/N — negligible for N≥10.
    try:
        ret15 = {t: df["returns_15b"] for t, df in results.items() if "returns_15b" in df.columns}
        ret1  = {t: df["returns_1b"]  for t, df in results.items() if "returns_1b"  in df.columns}
        vdev  = {t: df["vwap_dev"]    for t, df in results.items() if "vwap_dev"    in df.columns}

        if ret15:
            univ15 = pd.concat(ret15, axis=1).mean(axis=1)   # NaN-safe mean per timestamp
            for ticker, df in results.items():
                if "returns_15b" in df.columns:
                    results[ticker]["rs_15m"] = df["returns_15b"] - univ15.reindex(df.index)

        if ret1:
            univ1  = pd.concat(ret1,  axis=1).mean(axis=1)
            for ticker, df in results.items():
                if "returns_1b" in df.columns:
                    results[ticker]["rs_1m"]  = df["returns_1b"]  - univ1.reindex(df.index)

        # Cross-sectional VWAP deviation: isolates idiosyncratic mean-reversion
        # from market-wide VWAP drift (e.g. NVDA pulling away while SPY flat)
        if vdev:
            univ_vwap = pd.concat(vdev, axis=1).mean(axis=1)
            for ticker, df in results.items():
                if "vwap_dev" in df.columns:
                    results[ticker]["rs_vwap_dev"] = df["vwap_dev"] - univ_vwap.reindex(df.index)
    except Exception as exc:
        logger.warning("relative_strength_failed: %s", exc)

    return results


FEATURE_COLUMNS: list[str] = [
    # ── Trend ──────────────────────────────────────────────────────────────────
    # EMAs excluded: raw price level, not cross-ticker comparable.
    # Binary ema_cross signals PRUNED (SHAP < 1e-6 in FFSA W10 report).
    "ema_cross_21_50",                     # kept: SHAP 6.5e-6, barely above threshold
    "macd", "macd_signal", "macd_hist",
    "adx", "dmp", "dmn",

    # ── Momentum ───────────────────────────────────────────────────────────────
    "rsi_14", "stoch_k", "stoch_d", "willr_14", "cci_20", "roc_10",

    # ── Volatility ─────────────────────────────────────────────────────────────
    "bb_width", "bb_pct",
    "atr_pct",

    # ── VWAP features (intraday mean-reversion signal) ─────────────────────────
    "vwap_dev",                            # (close − daily VWAP) / VWAP → %-units
    "vwap_slope_5",                        # 5-bar VWAP rate-of-change (fractional)
    "vwap_slope_15",                       # 15-bar VWAP rate-of-change (fractional)
    # PRUNED: vwap_above (SHAP 1.6e-6, binary — near zero signal)

    # ── Opening Range Breakout (daily volatility regime) ───────────────────────
    "orb_range_atr",                       # (ORB high − low) / ATR14 — regime width
    "orb_dev",                             # position within ORB range [0, 1]
    # PRUNED: orb_break_up (1.4e-6), orb_break_dn (1.3e-6) — binary, near-zero SHAP

    # ── Volume ─────────────────────────────────────────────────────────────────
    "obv_pct",
    "mfi_14", "vol_ratio",

    # ── Price action ───────────────────────────────────────────────────────────
    "returns_1b", "returns_5b", "returns_15b",
    "high_low_range", "gap_pct",
    # PRUNED: close_vs_high (SHAP 4.8e-6)

    # ── Relative strength (cross-sectional alpha isolation) ────────────────────
    "rs_15m",                              # ticker 15m ret − universe mean 15m ret
    "rs_1m",                               # ticker 1m  ret − universe mean 1m  ret
    "rs_vwap_dev",                         # ticker VWAP dev − universe mean VWAP dev

    # ── Microstructure ─────────────────────────────────────────────────────────
    "vpin_50", "vpin_zscore",

    # ── Intraday seasonality ───────────────────────────────────────────────────
    "min_since_open", "day_of_week", "time_to_close",
    "vol_seasonal_ratio",
    # PRUNED: is_open_window (0.0), is_close_window (1.7e-6), is_lunch (0.0)

    # ── Options flow (non-zero after ~90 days of collection) ───────────────────
    "gex_net", "gex_zscore", "gex_call_pct",

    # ── Market regime ──────────────────────────────────────────────────────────
    "regime",
    "regime_time",                         # ATR_percentile × time_to_close interaction

    # ── Composite interaction features ────────────────────────────────────────
    "orb_vpin_interact",                   # orb_dev × vpin_50 — informed breakout intensity
    "vwap_time_interact",                  # vwap_dev × time_to_close — mean-reversion urgency
    "atr_vol_interact",                    # atr_pct × vol_ratio — volatility confirmation
]
