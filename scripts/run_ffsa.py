"""Run FFSA — LightGBM + SHAP feature selection.

Reads pre-computed features from the feature_matrix table, trains a LightGBM
model to predict N-bar forward returns, and uses SHAP to rank all features
by predictive power. Saves top-N to reports/drift/.

Usage:
    python scripts/run_ffsa.py
    python scripts/run_ffsa.py --top-n 30 --forward-n 15
    python scripts/run_ffsa.py --forward-n 15 --sample-pct 15
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_ffsa")

FORWARD_BARS = 15       # default: 15-bar (~15 min) forward return — matches training horizon
MIN_SAMPLES  = 5_000    # minimum rows to proceed


# ─── Load from DB ─────────────────────────────────────────────────────────────

async def load_feature_matrix(sample_pct: float = 8.0) -> pd.DataFrame:
    """Load a sampled slice of feature_matrix rows into a single DataFrame.

    Uses TABLESAMPLE SYSTEM for fast block-level sampling — avoids full sequential
    scan of 3.7M JSONB rows which would OOM on a typical dev machine.

    Args:
        sample_pct: Percentage of rows to sample (default 8% ≈ 300k rows).
    """
    from src.data.db import get_session_factory, init_db
    from sqlalchemy import text

    await init_db()
    session_factory = get_session_factory()

    # Block-level sampling — fast, no full table scan
    logger.info("Loading feature_matrix from DB (TABLESAMPLE %.0f%%) ...", sample_pct)
    async with session_factory() as session:
        result = await session.execute(
            text("""
                SELECT time, ticker, features
                FROM feature_matrix TABLESAMPLE SYSTEM(:pct)
                ORDER BY ticker, time
            """),
            {"pct": sample_pct},
        )
        rows = result.all()

    logger.info("  %d feature rows loaded", len(rows))

    # Expand JSONB → flat columns
    records = []
    for time, ticker, feat_dict in rows:
        if not isinstance(feat_dict, dict):
            continue
        rec = {"time": time, "ticker": ticker}
        rec.update(feat_dict)
        records.append(rec)

    df = pd.DataFrame(records)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


async def load_close_prices() -> pd.DataFrame:
    """Load close prices to compute forward returns (target variable).

    Loads only the last 12 months of closes — enough for forward return labeling
    and avoids pulling 3.8M rows into memory.
    """
    from src.data.db import OHLCV1m, get_session_factory
    from sqlalchemy import select, text

    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            text("""
                SELECT time, ticker, close
                FROM ohlcv_1m
                WHERE time >= NOW() - INTERVAL '12 months'
                ORDER BY ticker, time
            """)
        )
        rows = result.all()

    df = pd.DataFrame(rows, columns=["time", "ticker", "close"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    logger.info("  %d close price rows loaded (last 12 months)", len(df))
    return df


# ─── Build X, y ───────────────────────────────────────────────────────────────

def build_Xy(feat_df: pd.DataFrame, close_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Join features with forward returns to produce (X, y)."""
    # Compute per-ticker forward returns from close prices
    target_frames = []
    for ticker, grp in close_df.groupby("ticker"):
        grp = grp.set_index("time").sort_index()
        fwd = grp["close"].pct_change(FORWARD_BARS).shift(-FORWARD_BARS)
        fwd = fwd.rename("target").reset_index()
        fwd["ticker"] = ticker
        target_frames.append(fwd)

    target_df = pd.concat(target_frames, ignore_index=True)
    target_df["time"] = pd.to_datetime(target_df["time"], utc=True)

    # Merge features + target
    merged = feat_df.merge(target_df, on=["time", "ticker"], how="inner")
    merged = merged.dropna(subset=["target"])

    # Drop time/ticker, keep only numeric feature columns
    drop_cols = {"time", "ticker", "target"}
    feat_cols = [c for c in merged.columns if c not in drop_cols]

    # Drop columns that are entirely NaN (e.g. GEX when no options data)
    X = merged[feat_cols].copy()
    null_pct = X.isnull().mean()
    too_sparse = null_pct[null_pct > 0.5].index.tolist()
    if too_sparse:
        logger.info("  Dropping %d sparse features (>50%% null): %s", len(too_sparse), too_sparse)
        X = X.drop(columns=too_sparse)

    # Fill remaining NaN with column median (warmup NaN at series start)
    X = X.fillna(X.median(numeric_only=True))
    y = merged["target"]

    logger.info("  X shape: %s, y shape: %s", X.shape, y.shape)
    return X, y


# ─── FFSA ─────────────────────────────────────────────────────────────────────

def run_ffsa(X: pd.DataFrame, y: pd.Series, top_n: int, feat_df: pd.DataFrame | None = None) -> dict:
    """Train LightGBM + compute SHAP importances. Returns ranked feature dict.

    Uses per-ticker walk-forward validation (last 20% of each ticker's bars)
    to match the training pipeline and avoid ticker-level leakage.
    """
    import shap
    from lightgbm import LGBMRegressor

    logger.info("Training LightGBM on %d samples, %d features ...", len(X), X.shape[1])

    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    # Per-ticker walk-forward split (matches training pipeline)
    # Old approach used TimeSeriesSplit on ticker-sorted rows — leaked info
    if feat_df is not None and "ticker" in feat_df.columns and "time" in feat_df.columns:
        train_mask = pd.Series(False, index=X.index)
        val_mask = pd.Series(False, index=X.index)
        for ticker, grp in feat_df.groupby("ticker"):
            _time_col = "time" if "time" in grp.columns else "index"
            grp_sorted = grp.sort_values(_time_col)
            grp_idx = grp_sorted.index.intersection(X.index)
            cutoff = int(len(grp_idx) * 0.80)
            train_mask.loc[grp_idx[:cutoff]] = True
            val_mask.loc[grp_idx[cutoff:]] = True
        train_idx = X.index[train_mask]
        val_idx = X.index[val_mask]
        logger.info("  Walk-forward split: train=%d, val=%d (per-ticker 80/20)", len(train_idx), len(val_idx))
    else:
        # Fallback: simple 80/20 split by row order
        cutoff = int(len(X) * 0.80)
        train_idx = X.index[:cutoff]
        val_idx = X.index[cutoff:]
        logger.info("  Fallback split: train=%d, val=%d (no ticker info)", len(train_idx), len(val_idx))

    model.fit(
        X.loc[train_idx], y.loc[train_idx],
        eval_set=[(X.loc[val_idx], y.loc[val_idx])],
        callbacks=[],
    )

    # Validation IC (Information Coefficient) — per-ticker temporal validation
    val_y    = y.loc[val_idx].reset_index(drop=True)
    val_pred = pd.Series(model.predict(X.loc[val_idx]))
    mask     = val_y.notna() & val_pred.notna()
    ic       = val_pred[mask].corr(val_y[mask]) if mask.sum() > 10 else float("nan")
    logger.info("  Validation IC: %.4f (target > 0.05)", ic)

    # SHAP feature importance
    logger.info("Computing SHAP values on %d sample rows ...", min(5000, len(X)))
    sample = X.sample(n=min(5000, len(X)), random_state=42)
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importances = dict(zip(X.columns, mean_abs_shap.tolist()))

    ranked = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    selected = [feat for feat, _ in ranked[:top_n]]

    return {
        "ic": round(ic, 4),
        "ranked": ranked,
        "selected": selected,
        "importances": dict(ranked),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main(top_n: int, sample_pct: float = 8.0, forward_n: int = FORWARD_BARS) -> None:
    global FORWARD_BARS
    FORWARD_BARS = forward_n
    logger.info("FFSA target: %d-bar forward return (~%d min)", forward_n, forward_n)

    feat_df  = await load_feature_matrix(sample_pct=sample_pct)
    close_df = await load_close_prices()

    X, y = build_Xy(feat_df, close_df)

    if len(X) < MIN_SAMPLES:
        logger.error("Only %d samples — need %d. Run backfill first.", len(X), MIN_SAMPLES)
        return

    result = run_ffsa(X, y, top_n, feat_df=feat_df)

    # ── Print ranked features ─────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print(f"  FFSA RESULTS — Top {top_n} Features by SHAP Importance")
    print(f"  Validation IC: {result['ic']:.4f}  (good if > 0.05)")
    print("═" * 60)
    for rank, (feat, importance) in enumerate(result["ranked"][:top_n], 1):
        bar = "█" * int(importance / result["ranked"][0][1] * 30)
        print(f"  {rank:2d}. {feat:<28s}  {importance:.5f}  {bar}")
    print("═" * 60)

    dropped = [feat for feat, _ in result["ranked"][top_n:]]
    print(f"\n  Dropped ({len(dropped)} features): {', '.join(dropped[:10])}" +
          (f" ... +{len(dropped)-10} more" if len(dropped) > 10 else ""))

    # ── Save report ───────────────────────────────────────────────────────────
    report_dir = Path("reports/drift")
    report_dir.mkdir(parents=True, exist_ok=True)
    date_str  = datetime.now(timezone.utc).strftime("%Y-W%W")
    out_path  = report_dir / f"ffsa_{date_str}.json"

    payload = {
        "version":           datetime.now(timezone.utc).isoformat(),
        "forward_bars":      forward_n,
        "validation_ic":     result["ic"],
        "top_n":             top_n,
        "selected_features": result["selected"],
        "shap_importances":  result["importances"],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    logger.info("FFSA report saved → %s", out_path)
    print(f"\n  Report saved → {out_path}")
    print(f"  These {top_n} features will be used for Transformer + TCN training.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n",      type=int,   default=30,
                        help="Number of features to select (default 30)")
    parser.add_argument("--sample-pct", type=float, default=8.0,
                        help="Percentage of feature_matrix rows to sample (default 8%%)")
    parser.add_argument("--forward-n",  type=int,   default=FORWARD_BARS,
                        help=f"Forward return horizon in bars (default {FORWARD_BARS} = ~{FORWARD_BARS}min)")
    args = parser.parse_args()
    asyncio.run(main(top_n=args.top_n, sample_pct=args.sample_pct, forward_n=args.forward_n))
