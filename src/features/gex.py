"""Gamma Exposure (GEX) feature computation.

GEX measures dealer (market-maker) hedging pressure at the current price.

Formula per contract:
    GEX = Gamma × Open_Interest × 100 × Spot² / 1e6
    (×100 because each contract covers 100 shares; /1e6 to express in $M)

Net GEX = Σ(Call GEX) − Σ(Put GEX)
  - Positive → dealers are long gamma → they buy dips, sell rallies → mean-reversion
  - Negative → dealers are short gamma → they amplify price moves → momentum

Key derived features:
  gex_net       : net dealer gamma exposure ($M)
  gex_zscore    : gex_net normalized by its rolling 20-day std
  gex_call_pct  : call GEX / (|call GEX| + |put GEX|), range [0,1]

Data source: options_flow table (populated by Unusual Whales poller).
If options_flow is empty, GEX features are returned as NaN (non-blocking).

Usage:
    from src.features.gex import compute_gex_features, attach_gex_to_features
    # Standalone:
    gex_df = await compute_gex_features("AAPL", spot_series, session)
    # As part of build_features pipeline:
    feat_df = await attach_gex_to_features(ticker, feat_df, session)
"""

from __future__ import annotations

import logging
import structlog
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.data.db import OptionsFlow

logger = structlog.get_logger(__name__)


# ─── Core GEX computation ─────────────────────────────────────────────────────


def _gex_from_df(flow_df: pd.DataFrame, spot: float) -> dict[str, float]:
    """Compute net GEX from a snapshot of open options contracts.

    Args:
        flow_df: DataFrame with columns [option_type, gamma, open_interest, strike]
        spot: Current spot price of the underlying.

    Returns:
        dict with gex_net, gex_call, gex_put, gex_call_pct (all in $M).
    """
    if flow_df.empty or spot <= 0:
        return {"gex_net": 0.0, "gex_call": 0.0, "gex_put": 0.0, "gex_call_pct": 0.5}

    flow_df = flow_df.dropna(subset=["gamma", "open_interest"])

    # GEX per contract = Gamma × OI × 100 × Spot² / 1e6
    flow_df = flow_df.copy()
    flow_df["contract_gex"] = (
        flow_df["gamma"].abs()
        * flow_df["open_interest"]
        * 100
        * spot ** 2
        / 1e6
    )

    calls = flow_df[flow_df["option_type"].str.lower() == "call"]["contract_gex"].sum()
    puts  = flow_df[flow_df["option_type"].str.lower() == "put"]["contract_gex"].sum()

    gex_net = calls - puts
    total   = abs(calls) + abs(puts)
    gex_call_pct = calls / total if total > 0 else 0.5

    return {
        "gex_net":      round(gex_net, 4),
        "gex_call":     round(calls,   4),
        "gex_put":      round(puts,    4),
        "gex_call_pct": round(gex_call_pct, 4),
    }


# ─── DB-backed GEX time series ────────────────────────────────────────────────


async def compute_gex_features(
    ticker: str,
    spot_series: pd.Series,           # DatetimeIndex → close price
    session: AsyncSession,
    lookback_days: int = 5,           # how far back to load options data
) -> pd.DataFrame:
    """Compute per-bar GEX features aligned to a spot price series.

    For each bar in spot_series, we use the most recent options snapshot
    (within the last lookback_days) to compute net GEX. The GEX is then
    z-scored over a 20-day rolling window to normalize across tickers.

    Returns:
        DataFrame indexed like spot_series with columns:
          gex_net, gex_zscore, gex_call_pct
    """
    if spot_series.empty:
        return pd.DataFrame(index=spot_series.index, columns=["gex_net", "gex_zscore", "gex_call_pct"])

    start = spot_series.index[0] - timedelta(days=lookback_days)
    end   = spot_series.index[-1]

    # Load all options flow for this ticker in the window
    result = await session.execute(
        select(OptionsFlow)
        .where(OptionsFlow.ticker == ticker)
        .where(OptionsFlow.time >= start)
        .where(OptionsFlow.time <= end)
        .order_by(OptionsFlow.time)
    )
    flow_rows = result.scalars().all()

    if not flow_rows:
        logger.warning("gex_no_data: %s — returning NaN GEX features", ticker)
        empty = pd.DataFrame(
            {"gex_net": np.nan, "gex_zscore": np.nan, "gex_call_pct": np.nan},
            index=spot_series.index,
        )
        return empty

    # Build a timed DataFrame of flow events
    flow_df = pd.DataFrame(
        [
            {
                "time":        r.time,
                "option_type": r.option_type,
                "gamma":       r.gamma,
                "open_interest": r.open_interest,
                "strike":      r.strike,
                "expiry":      r.expiry,
            }
            for r in flow_rows
        ]
    )
    flow_df["time"] = pd.to_datetime(flow_df["time"], utc=True)
    flow_df = flow_df.set_index("time").sort_index()

    # For each bar: use flow snapshot from the past 24h (options data is sparse)
    gex_records = []
    for ts, spot in spot_series.items():
        window_start = ts - timedelta(hours=24)
        snap = flow_df.loc[window_start:ts]
        g = _gex_from_df(snap.reset_index(drop=True), float(spot))
        g["time"] = ts
        gex_records.append(g)

    gex = pd.DataFrame(gex_records).set_index("time")

    # Z-score gex_net over a 20-day rolling window (20 × 390 bars ≈ 7800 bars)
    _window = min(7800, len(gex))
    _mean = gex["gex_net"].rolling(_window, min_periods=10).mean()
    _std  = gex["gex_net"].rolling(_window, min_periods=10).std().clip(lower=1e-9)
    gex["gex_zscore"] = (gex["gex_net"] - _mean) / _std

    return gex[["gex_net", "gex_zscore", "gex_call_pct"]]


# ─── Attach GEX to existing feature DataFrame ─────────────────────────────────


async def attach_gex_to_features(
    ticker: str,
    feat_df: pd.DataFrame,
    session: AsyncSession,
) -> pd.DataFrame:
    """Merge GEX columns into an existing feature DataFrame.

    If options data is unavailable, GEX columns are filled with NaN
    (the FFSA will naturally down-weight features with too many NaN values).

    Args:
        ticker:   Ticker symbol.
        feat_df:  Feature DataFrame with DatetimeIndex (output of compute_indicators).
        session:  Active AsyncSession.

    Returns:
        feat_df with gex_net, gex_zscore, gex_call_pct columns added.
    """
    spot_series = feat_df.get("close") if "close" in feat_df.columns else None

    # Fall back to using index timestamps with NaN spot if close not present
    if spot_series is None:
        feat_df["gex_net"]      = np.nan
        feat_df["gex_zscore"]   = np.nan
        feat_df["gex_call_pct"] = np.nan
        return feat_df

    try:
        gex_df = await compute_gex_features(ticker, spot_series, session)
        feat_df = feat_df.join(gex_df[["gex_net", "gex_zscore", "gex_call_pct"]], how="left")
    except Exception as exc:
        logger.warning("gex_attach_failed %s: %s — using NaN", ticker, exc)
        feat_df["gex_net"]      = np.nan
        feat_df["gex_zscore"]   = np.nan
        feat_df["gex_call_pct"] = np.nan

    return feat_df


# ─── Real-time GEX snapshot (used by live signal engine) ──────────────────────


async def get_live_gex(
    ticker: str,
    spot: float,
    session: AsyncSession,
    lookback_hours: int = 24,
) -> dict[str, float]:
    """Get the current GEX snapshot for a ticker (used in live inference).

    Returns a dict with gex_net, gex_call_pct ready to be added to the
    live feature vector alongside the indicator features.

    Runs in <1ms if options data is already loaded. Non-blocking on failure.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    try:
        result = await session.execute(
            select(OptionsFlow)
            .where(OptionsFlow.ticker == ticker)
            .where(OptionsFlow.time >= cutoff)
        )
        flow_rows = result.scalars().all()
        if not flow_rows:
            return {"gex_net": 0.0, "gex_zscore": 0.0, "gex_call_pct": 0.5}

        flow_df = pd.DataFrame(
            [{"option_type": r.option_type, "gamma": r.gamma, "open_interest": r.open_interest}
             for r in flow_rows]
        )
        return _gex_from_df(flow_df, spot)
    except Exception as exc:
        logger.debug("get_live_gex failed for %s: %s", ticker, exc)
        return {"gex_net": 0.0, "gex_zscore": 0.0, "gex_call_pct": 0.5}
