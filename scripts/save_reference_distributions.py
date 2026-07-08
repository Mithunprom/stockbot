"""Save reference distributions for PSI drift detection.

Loads the current LightGBM model's feature columns from its JSON sidecar,
pulls the feature matrix from the DB (same data the model was trained on),
and computes + saves the reference bin-edge distributions to JSON.

Run this once after each model retrain:
    python scripts/save_reference_distributions.py
    python scripts/save_reference_distributions.py --model-path models/lgbm/lgbm_ic_0.1775
    python scripts/save_reference_distributions.py --bins 15 --max-rows 200000
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

from src.features.psi import save_reference_distribution, DEFAULT_REFERENCE_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("save_reference_distributions")


def find_best_model_sidecar() -> Path:
    """Find the highest-IC LightGBM model JSON sidecar."""
    model_dir = Path("models/lgbm")
    json_files = sorted(model_dir.glob("lgbm_ic_*.json"), reverse=True)
    if not json_files:
        raise FileNotFoundError(
            "No LightGBM model sidecar found in models/lgbm/. "
            "Run scripts/train_lgbm.py first."
        )
    # Files are named lgbm_ic_0.XXXX.json — sort descending picks highest IC
    return json_files[0]


def load_feature_cols(model_path: str | None) -> list[str]:
    """Load feature column names from the model's JSON sidecar.

    Args:
        model_path: Base path (without extension) to model files,
                    e.g. "models/lgbm/lgbm_ic_0.1775". If None, auto-selects best.

    Returns:
        List of feature column names.
    """
    if model_path:
        json_path = Path(model_path).with_suffix(".json")
    else:
        json_path = find_best_model_sidecar()

    if not json_path.exists():
        raise FileNotFoundError(f"Model sidecar not found: {json_path}")

    with open(json_path) as f:
        metadata = json.load(f)

    feature_cols = metadata["feature_cols"]
    logger.info(
        "Loaded %d feature columns from %s (val_ic=%.4f)",
        len(feature_cols),
        json_path.name,
        metadata.get("val_ic", 0.0),
    )
    return feature_cols


async def load_feature_matrix(
    feature_cols: list[str],
    max_rows: int,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Load feature values from the DB into a DataFrame.

    Follows the same loading pattern as scripts/train_lgbm.py.

    Args:
        feature_cols: Feature column names to extract.
        max_rows: Maximum rows per ticker.
        tickers: Optional ticker filter. Defaults to _DEFAULT_UNIVERSE.

    Returns:
        DataFrame with one column per feature.
    """
    from sqlalchemy import select

    from src.data.db import FeatureMatrix, get_session_factory, init_db

    await init_db()
    session_factory = get_session_factory()

    if tickers is None:
        from main import _DEFAULT_UNIVERSE
        tickers = [t for t in _DEFAULT_UNIVERSE if "/" not in t]

    logger.info("Loading features for %d tickers (max %d rows each)", len(tickers), max_rows)

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
            logger.debug("  %s: no rows — skipping", ticker)
            continue

        records = []
        for ts, feat_dict in feat_rows:
            if not isinstance(feat_dict, dict):
                continue
            rec: dict = {}
            for col in feature_cols:
                val = feat_dict.get(col)
                rec[col] = float(val) if val is not None else np.nan
            records.append(rec)

        if records:
            ticker_df = pd.DataFrame(records)
            all_frames.append(ticker_df)
            logger.info("  %s: %d rows loaded", ticker, len(ticker_df))

    if not all_frames:
        raise RuntimeError("No feature data loaded from DB. Run build_features.py first.")

    combined = pd.concat(all_frames, ignore_index=True)
    logger.info("Total: %d rows, %d features", len(combined), len(feature_cols))
    return combined


async def main(
    model_path: str | None,
    output_path: str | None,
    bins: int,
    max_rows: int,
    tickers: list[str] | None,
) -> None:
    # 1. Load feature columns from model sidecar
    feature_cols = load_feature_cols(model_path)

    # 2. Load feature matrix from DB
    df = await load_feature_matrix(feature_cols, max_rows, tickers)

    # 3. Compute and save reference distributions
    out_path = Path(output_path) if output_path else DEFAULT_REFERENCE_PATH

    save_reference_distribution(
        df=df,
        feature_cols=feature_cols,
        path=out_path,
        bins=bins,
    )

    print()
    print("=" * 60)
    print("  Reference Distributions Saved")
    print("=" * 60)
    print(f"  Features:    {len(feature_cols)}")
    print(f"  Rows:        {len(df):,}")
    print(f"  Bins:        {bins}")
    print(f"  Output:      {out_path}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Save reference distributions for PSI drift detection"
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Base path to model files (e.g. models/lgbm/lgbm_ic_0.1775). "
             "Auto-selects best if not provided.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=f"Output JSON path (default: {DEFAULT_REFERENCE_PATH})",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=10,
        help="Number of quantile bins (default: 10)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=100_000,
        help="Max rows per ticker from DB (default: 100000)",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="Optional ticker filter (default: full universe)",
    )
    args = parser.parse_args()

    asyncio.run(main(
        model_path=args.model_path,
        output_path=args.output,
        bins=args.bins,
        max_rows=args.max_rows,
        tickers=args.tickers,
    ))
