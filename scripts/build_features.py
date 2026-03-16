"""Feature engineering pipeline.

Reads OHLCV bars from the DB, computes technical indicators via indicators.py,
and writes results to the feature_matrix table.

Usage:
    python scripts/build_features.py                  # all tickers, all data
    python scripts/build_features.py --tickers AAPL NVDA
    python scripts/build_features.py --since 2026-02-01
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.db import FeatureMatrix, OHLCV1m, get_session_factory, init_db
from src.features.gex import attach_gex_to_features
from src.features.indicators import compute_indicators, compute_indicators_for_universe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_features")

FFSA_VERSION = "v1"
WRITE_CHUNK = 500  # rows per DB insert


async def load_ohlcv(ticker: str, since: datetime | None) -> pd.DataFrame:
    """Load 1m bars for a ticker from the DB into a pandas DataFrame."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        q = select(OHLCV1m).where(OHLCV1m.ticker == ticker)
        if since:
            q = q.where(OHLCV1m.time >= since)
        q = q.order_by(OHLCV1m.time)
        result = await session.execute(q)
        rows = result.scalars().all()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        [
            {
                "time": r.time,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
                "vwap": r.vwap or r.close,
            }
            for r in rows
        ]
    )
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()
    return df


_REQUIRED_FEATURES = ["rsi_14", "macd", "ema_9", "atr_14", "vpin_50"]


async def write_features(ticker: str, feat_df: pd.DataFrame) -> int:
    """Write feature rows to feature_matrix table."""
    # Only require core indicators to be non-null.
    # GEX and other optional features (e.g. vol_seasonal_ratio) may be NaN
    # when data sources are unavailable — the FFSA will down-weight them.
    required = [c for c in _REQUIRED_FEATURES if c in feat_df.columns]
    feat_df = feat_df.dropna(subset=required)
    if feat_df.empty:
        return 0

    rows = []
    for ts, row in feat_df.iterrows():
        # Convert to plain dict, replace inf/-inf with None
        feat_dict = row.to_dict()
        feat_dict = {
            k: (None if not np.isfinite(v) else round(float(v), 8))
            for k, v in feat_dict.items()
        }
        rows.append(
            {
                "time": ts,
                "ticker": ticker,
                "features": feat_dict,   # pass dict directly — SQLAlchemy handles JSONB
                "ffsa_version": FFSA_VERSION,
            }
        )

    session_factory = get_session_factory()
    written = 0
    for i in range(0, len(rows), WRITE_CHUNK):
        chunk = rows[i : i + WRITE_CHUNK]
        async with session_factory() as session:
            stmt = insert(FeatureMatrix).values(chunk)
            stmt = stmt.on_conflict_do_update(
                index_elements=["time", "ticker"],
                set_={
                    "features": stmt.excluded.features,
                    "ffsa_version": stmt.excluded.ffsa_version,
                },
            )
            await session.execute(stmt)
            await session.commit()
        written += len(chunk)
    return written


async def process_ticker(ticker: str, since: datetime | None) -> None:
    logger.info("Processing %s ...", ticker)

    df = await load_ohlcv(ticker, since)
    if df.empty:
        logger.warning("%s: no OHLCV data found — skipping", ticker)
        return

    logger.info("  %s: %d bars loaded (%s → %s)", ticker, len(df), df.index[0].date(), df.index[-1].date())

    try:
        feat_df = compute_indicators(df, shift=True)
    except Exception as exc:
        logger.error("  %s: indicator computation failed: %s", ticker, exc)
        return

    # Attach GEX features (NaN if options_flow table is empty — non-blocking)
    session_factory = get_session_factory()
    async with session_factory() as session:
        feat_df = await attach_gex_to_features(ticker, feat_df, session)

    # Keep only indicator columns (drop raw OHLCV to avoid duplication)
    indicator_cols = [c for c in feat_df.columns if c not in {"open", "high", "low", "close", "volume", "vwap"}]
    feat_df = feat_df[indicator_cols]

    written = await write_features(ticker, feat_df)
    logger.info("  %s: %d feature rows written", ticker, written)


async def run(tickers: list[str], since: datetime | None) -> None:
    await init_db()

    # ── Pass 1: Load OHLCV + compute per-ticker indicators ────────────────────
    logger.info("Pass 1: Computing per-ticker indicators ...")
    bars: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = await load_ohlcv(ticker, since)
        if df.empty:
            logger.warning("%s: no OHLCV data — skipping", ticker)
            continue
        logger.info("  %s: %d bars (%s → %s)", ticker, len(df), df.index[0].date(), df.index[-1].date())
        bars[ticker] = df

    if not bars:
        logger.error("No data loaded for any ticker.")
        return

    # ── Pass 2: Compute indicators + cross-sectional features (rs_15m, rs_1m, rs_vwap_dev) ─
    logger.info("Pass 2: Computing cross-sectional relative strength features ...")
    results = compute_indicators_for_universe(bars, shift=True)

    # ── Pass 3: Attach GEX + write to DB ──────────────────────────────────────
    logger.info("Pass 3: Attaching GEX + writing to DB ...")
    session_factory = get_session_factory()
    for ticker, feat_df in results.items():
        async with session_factory() as session:
            feat_df = await attach_gex_to_features(ticker, feat_df, session)

        # Keep only indicator columns (drop raw OHLCV to avoid duplication)
        indicator_cols = [c for c in feat_df.columns if c not in {"open", "high", "low", "close", "volume", "vwap"}]
        feat_df = feat_df[indicator_cols]

        written = await write_features(ticker, feat_df)
        logger.info("  %s: %d feature rows written", ticker, written)

    # Summary
    async with session_factory() as session:
        result = await session.execute(
            text("SELECT COUNT(*) FROM feature_matrix")
        )
        total = result.scalar()
    logger.info("Done. Total feature rows in DB: %d", total)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build feature matrix from OHLCV bars")
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--since", default=None, help="Only process bars after this date (YYYY-MM-DD)")
    args = parser.parse_args()

    if args.tickers:
        tickers = args.tickers
    else:
        from main import _DEFAULT_UNIVERSE
        tickers = _DEFAULT_UNIVERSE

    since = None
    if args.since:
        since = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    asyncio.run(run(tickers=tickers, since=since))


if __name__ == "__main__":
    main()
