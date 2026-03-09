"""Incremental live feature computation.

Triggered after every completed 1m bar is written to ohlcv_1m.
Loads the last WARMUP_BARS of history for that ticker, runs
compute_indicators (shift=True to match training convention), and
upserts the latest feature row into feature_matrix.

This keeps feature_matrix fresh during live/paper trading so the
signal loop always has up-to-date features without needing a manual
build_features.py run.
"""

from __future__ import annotations

import asyncio
import numpy as np
import structlog
from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from src.data.db import FeatureMatrix, OHLCV1m, get_session_factory
from src.features.indicators import compute_indicators

logger = structlog.get_logger(__name__)

# Enough bars for longest indicator (EMA-50, VPIN-50) to fully warm up.
WARMUP_BARS = 300
FFSA_VERSION = "v1"


class LiveFeatureComputer:
    """Computes and upserts features for one ticker on each completed 1m bar.

    Args:
        feature_cols: Ordered list of FFSA-selected feature column names.
    """

    def __init__(self, feature_cols: list[str]) -> None:
        self._feature_cols = feature_cols
        self._sf = get_session_factory()
        # Simple per-ticker throttle: skip if a compute is already in flight
        self._in_flight: set[str] = set()

    async def on_bar(self, ticker: str, bar_time: datetime) -> None:
        """Called after a completed 1m bar for `ticker` has been written to DB."""
        if ticker in self._in_flight:
            return  # previous compute still running — skip this bar
        self._in_flight.add(ticker)
        try:
            await self._compute_and_write(ticker, bar_time)
        except Exception as exc:
            logger.warning(
                "live_feature_error",
                ticker=ticker,
                bar_time=str(bar_time),
                error=str(exc),
            )
        finally:
            self._in_flight.discard(ticker)

    # ── Internal ─────────────────────────────────────────────────────────────

    async def _compute_and_write(self, ticker: str, bar_time: datetime) -> None:
        # 1. Load the last WARMUP_BARS from ohlcv_1m
        df = await self._load_ohlcv(ticker)
        if df.empty or len(df) < 60:
            logger.debug("live_feature_skip_insufficient_bars", ticker=ticker, n=len(df))
            return

        # 2. Compute indicators — shift=True matches training convention
        feat_df = compute_indicators(df, shift=True)

        # 3. Take only the last row (the just-completed bar)
        last_row = feat_df.iloc[-1]
        feat_dict: dict[str, float | None] = {}
        for col in feat_df.columns:
            if col in {"open", "high", "low", "close", "volume", "vwap"}:
                continue
            val = last_row[col]
            feat_dict[col] = (
                None if not isinstance(val, (int, float)) or not np.isfinite(float(val))
                else round(float(val), 8)
            )

        # 4. Upsert into feature_matrix
        row = {
            "time": last_row.name,  # DatetimeIndex
            "ticker": ticker,
            "features": feat_dict,
            "ffsa_version": FFSA_VERSION,
        }
        async with self._sf() as session:
            stmt = insert(FeatureMatrix).values([row])
            stmt = stmt.on_conflict_do_update(
                index_elements=["time", "ticker"],
                set_={
                    "features": stmt.excluded.features,
                    "ffsa_version": stmt.excluded.ffsa_version,
                },
            )
            await session.execute(stmt)
            await session.commit()

        logger.debug(
            "live_feature_written",
            ticker=ticker,
            bar_time=str(last_row.name),
            n_features=len(feat_dict),
        )

    async def _load_ohlcv(self, ticker: str) -> pd.DataFrame:
        """Load the last WARMUP_BARS 1m bars for a ticker from DB."""
        async with self._sf() as session:
            result = await session.execute(
                select(OHLCV1m)
                .where(OHLCV1m.ticker == ticker)
                .order_by(OHLCV1m.time.desc())
                .limit(WARMUP_BARS)
            )
            rows = list(reversed(result.scalars().all()))

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(
            [
                {
                    "time": r.time,
                    "open":   float(r.open),
                    "high":   float(r.high),
                    "low":    float(r.low),
                    "close":  float(r.close),
                    "volume": float(r.volume),
                    "vwap":   float(r.vwap or r.close),
                }
                for r in rows
            ]
        )
        df["time"] = pd.to_datetime(df["time"], utc=True)
        return df.set_index("time").sort_index()
