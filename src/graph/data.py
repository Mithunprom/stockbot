"""Daily OHLCV loader for hybrid-graph research.

Tries (in order): on-disk cache, Yahoo chart API, yfinance, Stooq.
Never looks into the future — callers pass an explicit end date.

Look-ahead: none. Bars are daily. Yahoo adjclose is applied as a ratio to
open/high/low/close so OHLC stays on one scale after splits and dividends.
"""

from __future__ import annotations

import json
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

CACHE_DIR = Path("data/research/hybrid_adj")
_USER_AGENT = "StockBotHybrid/1.0 (research backtest)"


def _as_float_array(values: object) -> np.ndarray:
    if values is None:
        return np.array([], dtype=float)
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def apply_split_adjustment(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    adjclose: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scale raw OHLC by adjclose/close so every column is split-adjusted.

    Using adjclose as ``close`` while leaving open/high/low on the quote
    scale makes stops and ATR percentages jump on ex-div / split days.
    """
    if adjclose is None or adjclose.size == 0:
        return open_, high, low, close
    ratio = np.ones_like(close, dtype=float)
    ok = np.isfinite(close) & (close != 0.0) & np.isfinite(adjclose)
    ratio[ok] = adjclose[ok] / close[ok]
    adj_close = np.where(np.isfinite(adjclose), adjclose, close)
    return open_ * ratio, high * ratio, low * ratio, adj_close


def cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker.upper()}.csv.gz"


def synthetic_ohlcv(
    n: int = 750,
    seed: int = 0,
    start: str = "2023-01-03",
    drift: float = 0.00035,
    vol: float = 0.014,
    price0: float = 100.0,
) -> pd.DataFrame:
    """Deterministic geometric Brownian OHLCV for unit tests (business days)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n, tz="UTC")
    shocks = rng.standard_normal(n)
    rets = drift + vol * shocks
    close = price0 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[price0], close[:-1]]) * (1 + 0.15 * vol * rng.standard_normal(n))
    high = np.maximum(open_, close) * (1 + 0.4 * vol * rng.random(n))
    low = np.minimum(open_, close) * (1 - 0.4 * vol * rng.random(n))
    volume = rng.integers(1_000_000, 8_000_000, size=n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def _to_naive_utc_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower().strip(): c for c in df.columns}
    rename = {}
    for want in ("open", "high", "low", "close", "volume"):
        if want in cols:
            rename[cols[want]] = want
        elif want.title() in df.columns:
            rename[want.title()] = want
        elif want.capitalize() in df.columns:
            rename[want.capitalize()] = want
    out = df.rename(columns=rename)
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in out.columns]
    out = out[keep].copy()
    out.index = pd.to_datetime(out.index, utc=True)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out = out.dropna(subset=["close"])
    if "volume" not in out.columns:
        out["volume"] = 1.0
    return out


def _fetch_yahoo_chart(ticker: str, start: date, end: date) -> pd.DataFrame:
    p1 = int(datetime(start.year, start.month, start.day, tzinfo=timezone.utc).timestamp())
    p2 = int(datetime(end.year, end.month, end.day, 23, 59, tzinfo=timezone.utc).timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={p1}&period2={p2}&interval=1d&events=div%7Csplit"
    )
    req = Request(url, headers={"User-Agent": _USER_AGENT})
    with urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    result = payload["chart"]["result"][0]
    ts = result["timestamp"]
    q = result["indicators"]["quote"][0]
    adj_raw = result["indicators"].get("adjclose", [{}])[0].get("adjclose")
    open_, high, low, close = apply_split_adjustment(
        _as_float_array(q["open"]),
        _as_float_array(q["high"]),
        _as_float_array(q["low"]),
        _as_float_array(q["close"]),
        _as_float_array(adj_raw) if adj_raw else None,
    )
    idx = pd.to_datetime(ts, unit="s", utc=True)
    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": q["volume"],
        },
        index=idx,
    )
    return _normalize_ohlcv(df)


def _fetch_stooq(ticker: str) -> pd.DataFrame:
    symbol = ticker.lower()
    if not symbol.endswith(".us"):
        symbol = f"{symbol}.us"
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    req = Request(url, headers={"User-Agent": _USER_AGENT})
    with urlopen(req, timeout=30) as resp:
        raw = resp.read()
    from io import BytesIO

    df = pd.read_csv(BytesIO(raw))
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.set_index("date")
    return _normalize_ohlcv(df)


def _fetch_yfinance(ticker: str, start: date, end: date) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(
        ticker,
        start=start.isoformat(),
        end=end.isoformat(),
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"yfinance empty for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return _normalize_ohlcv(df)


def fetch_daily(
    ticker: str,
    start: date,
    end: date,
    *,
    use_cache: bool = True,
    sleep_s: float = 0.25,
) -> pd.DataFrame:
    """Fetch daily OHLCV for ``ticker`` in ``[start, end]`` (inclusive)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = cache_path(ticker)
    if use_cache and path.exists():
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df = _normalize_ohlcv(df)
    else:
        errors: list[str] = []
        df = None
        for fn in (
            lambda: _fetch_yahoo_chart(ticker, start, end),
            lambda: _fetch_yfinance(ticker, start, end),
            lambda: _fetch_stooq(ticker),
        ):
            try:
                df = fn()
                break
            except (URLError, TimeoutError, KeyError, ValueError, RuntimeError, OSError) as exc:
                errors.append(f"{fn.__name__ if hasattr(fn, '__name__') else 'fetch'}: {exc}")
                time.sleep(sleep_s)
        if df is None or df.empty:
            raise RuntimeError(f"all fetchers failed for {ticker}: {errors}")
        df.to_csv(path, compression="gzip")
        time.sleep(sleep_s)

    lo = pd.Timestamp(start, tz="UTC")
    hi = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    return df.loc[(df.index >= lo) & (df.index < hi)]


def load_universe_bars(
    tickers: Iterable[str],
    start: date,
    end: date,
    *,
    use_cache: bool = True,
    min_bars: int = 200,
) -> dict[str, pd.DataFrame]:
    """Load as many tickers as possible; skip failures instead of aborting."""
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            df = fetch_daily(t, start, end, use_cache=use_cache)
            if len(df) >= min_bars:
                out[t.upper()] = df
        except Exception:
            continue
    return out
