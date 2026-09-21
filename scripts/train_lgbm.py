"""Train LightGBM signal model.

Uses the same walk-forward split as neural net training but trains
in seconds instead of hours. LightGBM achieves IC=0.21 on proper
per-ticker temporal validation — far exceeding Transformer/TCN (IC≈0).

Usage:
    python scripts/train_lgbm.py
    python scripts/train_lgbm.py --top-n 30 --forward-n 15
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.lgbm import LGBMSignalModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_lgbm")

FORWARD_N = 15

# Below this, the archive is not yet a better training set than the live tables.
# 20 sessions ~= one trading month; the models that actually worked were fit on
# months of bars, not the 3 days DB retention leaves behind.
_MIN_ARCHIVE_DAYS = 20
DIRECTION_EPSILON = 0.0001


def _load_from_archive(
    feature_cols: list[str], tickers: list[str], max_rows: int,
) -> pd.DataFrame | None:
    """Build the training frame from the durable Parquet archive.

    Returns None when the archive holds too little to be worth using, so the
    caller can fall back to the (short-retention) live tables.
    """
    from datetime import date as _date, timedelta as _td

    from src.data.feature_archive import available_days, load_range

    days = available_days()
    if len(days) < _MIN_ARCHIVE_DAYS:
        logger.info("Archive has %d day(s) — below the %d-day minimum",
                    len(days), _MIN_ARCHIVE_DAYS)
        return None

    start = _date.fromisoformat(days[0])
    end = _date.fromisoformat(days[-1])
    logger.info("Loading archive %s → %s (%d sessions)", start, end, len(days))

    raw = load_range(start, end, tickers=tickers)
    if raw.empty or "close" not in raw.columns:
        return None

    frames: list[pd.DataFrame] = []
    for ticker, grp in raw.groupby("ticker"):
        grp = grp.sort_values("time")
        close_s = pd.to_numeric(grp["close"], errors="coerce")
        fwd = close_s.pct_change(FORWARD_N).shift(-FORWARD_N)

        out = pd.DataFrame({"time": grp["time"].values, "ticker": ticker})
        for f in feature_cols:
            out[f] = (pd.to_numeric(grp[f], errors="coerce").fillna(0.0).values
                      if f in grp.columns else 0.0)
        out["forward_return"] = fwd.values
        out = out.dropna(subset=["forward_return"])
        if max_rows:
            out = out.tail(max_rows)
        if not out.empty:
            frames.append(out)
            logger.info("  %s: %d rows (archive)", ticker, len(out))

    if not frames:
        return None
    merged = pd.concat(frames, ignore_index=True)
    logger.info("Total from archive: %d rows, %d features",
                len(merged), len(feature_cols))
    return merged


async def load_data(
    top_n: int, max_rows: int, tickers: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Load features + forward returns, preferring the durable archive."""
    from sqlalchemy import select

    from src.data.db import FeatureMatrix, OHLCV1m, get_session_factory, init_db

    await init_db()
    session_factory = get_session_factory()

    ffsa_files = sorted(Path("reports/drift").glob("ffsa_*.json"), reverse=True)
    if not ffsa_files:
        raise FileNotFoundError("No FFSA report — run scripts/run_ffsa.py first")
    with open(ffsa_files[0]) as f:
        ffsa = json.load(f)
    feature_cols: list[str] = ffsa["selected_features"][:top_n]
    logger.info("Using %d FFSA features from %s", len(feature_cols), ffsa_files[0].name)

    from main import _DEFAULT_UNIVERSE

    if tickers is None:
        tickers = _DEFAULT_UNIVERSE
    tickers = [t for t in tickers if "/" not in t]
    logger.info("Universe: %d tickers, max %d rows each", len(tickers), max_rows)

    # Prefer the durable archive. `feature_matrix` is pruned at 3 days and
    # `ohlcv_1m` at 7, so training only off the live tables fits an intraday
    # model to ~3 days of bars — which is how checkpoints with val_ic of -0.06
    # came to exist. The archive holds months.
    archived = _load_from_archive(feature_cols, tickers, max_rows)
    if archived is not None:
        return archived, [c for c in feature_cols if c in archived.columns]

    logger.warning(
        "Feature archive empty — falling back to the live DB. This trains on at "
        "most the retention window (3 days) and will produce an unstable model."
    )

    all_frames: list[pd.DataFrame] = []

    for ticker in tickers:
        async with session_factory() as session:
            result = await session.execute(
                select(FeatureMatrix.time, FeatureMatrix.features)
                .where(FeatureMatrix.ticker == ticker)
                .order_by(FeatureMatrix.time.desc())
                .limit(max_rows)
            )
            feat_rows = result.all()

        if not feat_rows:
            logger.warning("  %s: no rows — skipping", ticker)
            continue

        feat_rows = list(reversed(feat_rows))
        records = []
        for ts, feat_dict in feat_rows:
            if not isinstance(feat_dict, dict):
                continue
            rec: dict = {"time": ts}
            for f in feature_cols:
                rec[f] = float(feat_dict.get(f) or 0.0)
            records.append(rec)

        ticker_df = pd.DataFrame(records)
        ticker_df["time"] = pd.to_datetime(ticker_df["time"], utc=True)

        async with session_factory() as session:
            result = await session.execute(
                select(OHLCV1m.time, OHLCV1m.close)
                .where(OHLCV1m.ticker == ticker)
                .order_by(OHLCV1m.time)
            )
            price_rows = result.all()

        close_s = pd.Series(
            {r.time: float(r.close) for r in price_rows}, name="close"
        )
        close_s.index = pd.to_datetime(close_s.index, utc=True)

        fwd = close_s.pct_change(FORWARD_N).shift(-FORWARD_N).rename("forward_return")

        ticker_df = ticker_df.set_index("time").sort_index()
        ticker_df = ticker_df.join(fwd, how="inner")
        ticker_df = ticker_df.dropna(subset=["forward_return"])
        ticker_df = ticker_df.reset_index()
        ticker_df["ticker"] = ticker

        all_frames.append(ticker_df)
        logger.info("  %s: %d rows", ticker, len(ticker_df))

    merged = pd.concat(all_frames, ignore_index=True)
    feature_cols = [c for c in feature_cols if c in merged.columns]
    logger.info("Total: %d rows, %d features", len(merged), len(feature_cols))
    return merged, feature_cols


def walk_forward_split(
    df: pd.DataFrame, val_frac: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-ticker walk-forward split (last val_frac% of each ticker's bars)."""
    train_frames, val_frames = [], []
    time_col = "time" if "time" in df.columns else "index"
    for ticker, grp in df.groupby("ticker"):
        grp = grp.sort_values(time_col)
        cutoff = int(len(grp) * (1 - val_frac))
        train_frames.append(grp.iloc[:cutoff])
        val_frames.append(grp.iloc[cutoff:])
    return (
        pd.concat(train_frames, ignore_index=True),
        pd.concat(val_frames, ignore_index=True),
    )


async def main(
    top_n: int, max_rows: int, forward_n: int, tickers: list[str] | None,
) -> None:
    global FORWARD_N
    FORWARD_N = forward_n

    merged, feature_cols = await load_data(top_n, max_rows, tickers)
    train_df, val_df = walk_forward_split(merged)
    logger.info("Split: train=%d, val=%d", len(train_df), len(val_df))

    X_train = train_df[feature_cols].fillna(0).astype(np.float32)
    y_train = train_df["forward_return"].values.astype(np.float32)
    X_val = val_df[feature_cols].fillna(0).astype(np.float32)
    y_val = val_df["forward_return"].values.astype(np.float32)

    model = LGBMSignalModel(feature_cols=feature_cols)
    logger.info("Training LightGBM (regressor + classifier) ...")
    metrics = model.train(X_train, y_train, X_val, y_val, DIRECTION_EPSILON)

    print("\n" + "═" * 50)
    print("  LightGBM Signal Model — Results")
    print("═" * 50)
    print(f"  Train IC:     {metrics['train_ic']:.4f}")
    print(f"  Val IC:       {metrics['val_ic']:.4f}")
    print(f"  Val Dir Acc:  {metrics['val_dir_acc']:.4f}")
    print(f"  Features:     {len(feature_cols)}")
    print(f"  Train rows:   {len(X_train):,}")
    print(f"  Val rows:     {len(X_val):,}")
    print("═" * 50)

    path = model.save()
    logger.info("Model saved → %s", path)
    print(f"\n  Saved → {path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LightGBM signal model")
    parser.add_argument("--top-n", type=int, default=30,
                        help="Number of FFSA features to use")
    parser.add_argument("--max-rows", type=int, default=100_000,
                        help="Max rows per ticker from DB")
    parser.add_argument("--forward-n", type=int, default=15,
                        help="Forward return horizon in bars")
    parser.add_argument("--tickers", nargs="*", default=None)
    args = parser.parse_args()

    asyncio.run(main(args.top_n, args.max_rows, args.forward_n, args.tickers))
