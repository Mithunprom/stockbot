"""Daily technical, HMM-style regime, and causal sequence predictor.

Inspired by kasyap1234/stock_bot_langgraph (40+ indicators, HMM, LSTM) and
the existing StockBot pandas indicator pipeline, but computed on *daily*
bars so the research harness does not need 1-minute history.

Every derived column that can leak the future is shifted by 1 bar when
``shift=True`` (the default).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Integer HMM-style regimes — match src.features.regime labels.
REGIME_TRENDING = 0
REGIME_CHOPPY = 1
REGIME_HIGH_VOL = 2


def regime_int(value: object, default: int = REGIME_CHOPPY) -> int:
    """Parse an HMM regime label.

    Trending is ``0``, so callers must never write ``int(x or default)`` —
    that silently maps a valid trending bar onto choppy.
    """
    if value is None:
        return default
    try:
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev).abs(), (low - prev).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def _macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    line = _ema(close, 12) - _ema(close, 26)
    signal = _ema(line, 9)
    return line, signal, line - signal


def _stoch(high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3):
    ll = low.rolling(k).min()
    hh = high.rolling(k).max()
    k_pct = 100 * (close - ll) / (hh - ll + 1e-9)
    return k_pct, k_pct.rolling(d).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14):
    prev_high, prev_low, prev_close = high.shift(1), low.shift(1), close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    dm_plus = np.where(
        (high - prev_high) > (prev_low - low),
        (high - prev_high).clip(lower=0),
        0.0,
    )
    dm_minus = np.where(
        (prev_low - low) > (high - prev_high),
        (prev_low - low).clip(lower=0),
        0.0,
    )
    atr_n = tr.ewm(alpha=1 / n, adjust=False).mean()
    di_p = 100 * pd.Series(dm_plus, index=close.index).ewm(alpha=1 / n, adjust=False).mean() / atr_n.replace(0, np.nan)
    di_m = 100 * pd.Series(dm_minus, index=close.index).ewm(alpha=1 / n, adjust=False).mean() / atr_n.replace(0, np.nan)
    dx = 100 * (di_p - di_m).abs() / (di_p + di_m + 1e-9)
    return dx.ewm(alpha=1 / n, adjust=False).mean(), di_p, di_m


def _wilder_di(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14):
    adx, di_p, di_m = _adx(high, low, close, n)
    return adx, di_p, di_m


def compute_daily_indicators(df: pd.DataFrame, shift: bool = True) -> pd.DataFrame:
    """Append 40+ daily indicators. Index must be DatetimeIndex."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")
    out = df.copy()
    o, h, l, c, v = out["open"], out["high"], out["low"], out["close"], out["volume"]

    # Trend / MAs
    for n in (5, 10, 20, 50, 100, 200):
        out[f"sma_{n}"] = c.rolling(n).mean()
        out[f"ema_{n}"] = _ema(c, n)
        out[f"close_vs_sma_{n}"] = c / out[f"sma_{n}"].replace(0, np.nan) - 1.0
    out["ema_stack"] = (
        (out["ema_10"] > out["ema_20"]).astype(float)
        + (out["ema_20"] > out["ema_50"]).astype(float)
        + (out["ema_50"] > out["ema_200"]).astype(float)
        - 1.5
    ) / 1.5

    macd, macd_sig, macd_hist = _macd(c)
    out["macd"] = macd
    out["macd_signal"] = macd_sig
    out["macd_hist"] = macd_hist
    out["macd_hist_z"] = macd_hist / (macd_hist.rolling(50).std().replace(0, np.nan) + 1e-9)

    adx, di_p, di_m = _wilder_di(h, l, c)
    out["adx"] = adx
    out["di_plus"] = di_p
    out["di_minus"] = di_m

    # Momentum
    out["rsi_14"] = _rsi(c, 14)
    out["rsi_7"] = _rsi(c, 7)
    out["stoch_k"], out["stoch_d"] = _stoch(h, l, c)
    out["willr"] = -100 * (h.rolling(14).max() - c) / (h.rolling(14).max() - l.rolling(14).min() + 1e-9)
    tp = (h + l + c) / 3
    tp_ma = tp.rolling(20).mean()
    tp_mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    out["cci"] = (tp - tp_ma) / (0.015 * tp_mad + 1e-9)
    out["roc_10"] = c.pct_change(10)
    out["roc_21"] = c.pct_change(21)
    out["mom_10"] = c.diff(10)

    # Volatility
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    out["bb_upper"] = bb_mid + 2 * bb_std
    out["bb_lower"] = bb_mid - 2 * bb_std
    out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / bb_mid.replace(0, np.nan)
    out["bb_pct"] = (c - out["bb_lower"]) / (out["bb_upper"] - out["bb_lower"] + 1e-9)
    out["atr_14"] = _atr(h, l, c, 14)
    out["atr_pct"] = out["atr_14"] / c.replace(0, np.nan)
    kc_mid = _ema(c, 20)
    out["kc_upper"] = kc_mid + 2 * out["atr_14"]
    out["kc_lower"] = kc_mid - 2 * out["atr_14"]

    # Donchian / breakout / Ichimoku-ish
    out["donchian_high"] = h.rolling(20).max()
    out["donchian_low"] = l.rolling(20).min()
    out["donchian_pos"] = (c - out["donchian_low"]) / (
        out["donchian_high"] - out["donchian_low"] + 1e-9
    )
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
    out["ichimoku_tenkan"] = tenkan
    out["ichimoku_kijun"] = kijun
    out["ichimoku_span_a"] = (tenkan + kijun) / 2
    out["ichimoku_span_b"] = (h.rolling(52).max() + l.rolling(52).min()) / 2
    out["ichimoku_cloud_dir"] = np.sign(out["ichimoku_span_a"] - out["ichimoku_span_b"])
    out["ichimoku_price_vs_kijun"] = c / kijun.replace(0, np.nan) - 1.0

    # Volume
    direction = np.sign(c.diff()).fillna(0)
    out["obv"] = (direction * v).cumsum()
    out["obv_slope"] = out["obv"].pct_change(10).replace([np.inf, -np.inf], np.nan)
    rmf = tp * v
    pos_flow = rmf.where(tp > tp.shift(1), 0.0).rolling(14).sum()
    neg_flow = rmf.where(tp < tp.shift(1), 0.0).rolling(14).sum()
    out["mfi"] = 100 - (100 / (1 + pos_flow / neg_flow.replace(0, np.nan)))
    out["vol_ratio"] = v / v.rolling(20).mean().replace(0, np.nan)
    out["vol_z"] = (v - v.rolling(60).mean()) / v.rolling(60).std().replace(0, np.nan)
    ad = ((c - l) - (h - c)) / (h - l + 1e-9) * v
    out["cmf"] = ad.rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
    out["force_index"] = c.diff() * v
    out["force_index_z"] = out["force_index"] / (
        out["force_index"].rolling(20).std().replace(0, np.nan) + 1e-9
    )

    # Range / geometry
    out["ret_1"] = c.pct_change(1)
    out["ret_5"] = c.pct_change(5)
    out["ret_21"] = c.pct_change(21)
    out["gap"] = o / c.shift(1).replace(0, np.nan) - 1.0
    out["high_low_range"] = (h - l) / c.replace(0, np.nan)
    roll_hi = h.rolling(60).max()
    roll_lo = l.rolling(60).min()
    out["fib_pos"] = (c - roll_lo) / (roll_hi - roll_lo + 1e-9)
    out["dist_52w_high"] = c / h.rolling(252, min_periods=60).max() - 1.0
    out["drawdown_from_peak"] = c / c.cummax() - 1.0

    # Aroon / choppiness / PPO / TRIX-lite
    n_aroon = 25
    out["aroon_up"] = 100 * h.rolling(n_aroon + 1).apply(
        lambda x: float(n_aroon - np.argmax(x[::-1])), raw=True
    ) / n_aroon
    out["aroon_down"] = 100 * l.rolling(n_aroon + 1).apply(
        lambda x: float(n_aroon - np.argmin(x[::-1])), raw=True
    ) / n_aroon
    out["aroon_osc"] = out["aroon_up"] - out["aroon_down"]
    atr_sum = (h - l).rolling(14).sum()
    out["chop"] = 100 * np.log10(atr_sum / (h.rolling(14).max() - l.rolling(14).min() + 1e-9)) / np.log10(14)
    out["ppo"] = 100 * (_ema(c, 12) - _ema(c, 26)) / _ema(c, 26).replace(0, np.nan)
    logc = np.log(c.replace(0, np.nan))
    out["trix"] = _ema(_ema(_ema(logc, 15), 15), 15).diff() * 100

    # Elder ray / Hull-ish
    out["bull_power"] = h - _ema(c, 13)
    out["bear_power"] = l - _ema(c, 13)
    wma9 = c.rolling(9).apply(lambda x: np.dot(x, np.arange(1, 10)) / 45.0, raw=True)
    wma5 = c.rolling(5).apply(lambda x: np.dot(x, np.arange(1, 6)) / 15.0, raw=True)
    out["hull_approx"] = _ema(2 * wma5 - wma9, 3)
    out["price_vs_hull"] = c / out["hull_approx"].replace(0, np.nan) - 1.0

    # Realized vol
    out["rvol_20"] = c.pct_change().rolling(20).std() * np.sqrt(252)
    out["rvol_60"] = c.pct_change().rolling(60).std() * np.sqrt(252)

    derived = [col for col in out.columns if col not in df.columns]
    if shift:
        out[derived] = out[derived].shift(1)
    return out


def causal_hmm_regime(close: pd.Series, spy_close: pd.Series | None = None) -> pd.Series:
    """Causal 3-state regime (trending / choppy / high-vol).

    Uses expanding-window percentiles of 20d realized vol and 50d return so a
    bar never sees its own future. This is the HMM stand-in used in search:
    a full Baum-Welch fit on the whole sample would leak.
    """
    ret = close.pct_change()
    vol20 = ret.rolling(20, min_periods=10).std()
    ret50 = close.pct_change(50)
    # Expanding z-score of vol — causal, O(n). Equivalent in spirit to an
    # 80th-percentile HMM high-vol state without the O(n²) expanding rank.
    vol_mu = vol20.expanding(min_periods=40).mean()
    vol_sd = vol20.expanding(min_periods=40).std().replace(0, np.nan)
    vol_z = (vol20 - vol_mu) / vol_sd
    adx_proxy = (close / close.rolling(20).mean() - 1.0).abs()
    regime = pd.Series(REGIME_CHOPPY, index=close.index, dtype=np.int8)
    high_vol = vol_z > 0.85
    trending = (~high_vol) & (adx_proxy > 0.04) & (ret50.abs() > 0.04)
    regime[high_vol] = REGIME_HIGH_VOL
    regime[trending] = REGIME_TRENDING
    if spy_close is not None:
        spy_aligned = spy_close.reindex(close.index).ffill()
        spy_sma = spy_aligned.rolling(200, min_periods=50).mean()
        # If SPY is in a strong trend, upgrade choppy → trending.
        spy_trend = (spy_aligned - spy_sma).abs() / spy_sma.replace(0, np.nan) > 0.04
        regime[(regime == REGIME_CHOPPY) & spy_trend.fillna(False).astype(bool)] = REGIME_TRENDING
    return regime.shift(1).fillna(REGIME_CHOPPY).astype(np.int8)


def causal_sequence_predictor(feat: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """Walk-forward ridge on lagged features — LSTM stand-in.

    Retrains every 21 bars on *past* rows only. Predicts ``horizon``-day
    forward return. Output is shifted so it is known at close of day t
    (uses features already shifted by ``compute_daily_indicators``).
    """
    cols = [
        c
        for c in (
            "ret_1",
            "ret_5",
            "ret_21",
            "rsi_14",
            "macd_hist",
            "bb_pct",
            "vol_z",
            "adx",
            "atr_pct",
            "donchian_pos",
        )
        if c in feat.columns
    ]
    if not cols or "close" not in feat.columns:
        return pd.Series(0.0, index=feat.index)

    y = feat["close"].pct_change(horizon).shift(-horizon)
    X = feat[cols].replace([np.inf, -np.inf], np.nan)
    pred = pd.Series(0.0, index=feat.index)
    values = X.fillna(0.0).to_numpy(dtype=np.float64)
    target = y.fillna(0.0).to_numpy(dtype=np.float64)
    n = len(feat)
    min_train = 80
    step = 21
    last_coef = None
    last_mu = None
    last_sig = None
    last_intercept = 0.0
    ridge = 1.0
    for start in range(min_train, n, step):
        train = slice(0, start)
        mu = values[train].mean(axis=0)
        sig = values[train].std(axis=0)
        sig = np.where(sig < 1e-8, 1.0, sig)
        Z = (values[train] - mu) / sig
        yt = target[train]
        # Drop the last ``horizon`` rows of train — their y is not yet known.
        usable = start - horizon
        if usable < min_train:
            continue
        Zt = Z[:usable]
        ytt = yt[:usable]
        xtx = Zt.T @ Zt + ridge * np.eye(Zt.shape[1])
        xty = Zt.T @ ytt
        try:
            coef = np.linalg.solve(xtx, xty)
        except np.linalg.LinAlgError:
            continue
        intercept = float(ytt.mean() - (Zt.mean(axis=0) @ coef))
        last_coef, last_mu, last_sig, last_intercept = coef, mu, sig, intercept
        stop = min(start + step, n)
        Zte = (values[start:stop] - mu) / sig
        pred.iloc[start:stop] = Zte @ coef + intercept
    if last_coef is not None:
        # Fill the warm-up with zeros; remaining tail already written.
        pass
    return pred.clip(-0.15, 0.15)


def attach_quant_overlay(
    feat: pd.DataFrame,
    spy_close: pd.Series | None = None,
) -> pd.DataFrame:
    """Add HMM regime, sequence prediction, and relative-strength vs SPY.

    Overlay columns that are computed from unshifted closes are lagged one
    bar so a signal at timestamp t never sees t's close (fill is t's open).
    ``hmm_regime`` already lags inside ``causal_hmm_regime``; ``seq_pred``
    reads indicator columns that ``compute_daily_indicators`` already shifted.
    """
    out = feat.copy()
    out["hmm_regime"] = causal_hmm_regime(feat["close"], spy_close)
    out["seq_pred"] = causal_sequence_predictor(feat)
    if spy_close is not None:
        spy = spy_close.reindex(feat.index).ffill()
        spy_ret_21 = spy.pct_change(21)
        out["rs_spy_21"] = (feat["close"].pct_change(21) - spy_ret_21).shift(1)
        out["spy_vs_sma200"] = (spy / spy.rolling(200, min_periods=50).mean() - 1.0).shift(1)
        out["spy_rvol_20"] = (spy.pct_change().rolling(20).std() * np.sqrt(252)).shift(1)
    else:
        rvol = out["rvol_20"] if "rvol_20" in out.columns else 0.15
        out["rs_spy_21"] = 0.0
        out["spy_vs_sma200"] = 0.0
        out["spy_rvol_20"] = rvol
    return out
