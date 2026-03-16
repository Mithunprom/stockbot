"""Train the RL agent (PPO) on historical bar data.

Loads feature_matrix + OHLCV from DB, formats bar dicts for TradingEnv,
runs behavioral cloning pre-training, then full PPO training.

Usage:
    python scripts/train_rl.py
    python scripts/train_rl.py --timesteps 100000 --ticker AAPL
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_rl")


# ─── Data loading ─────────────────────────────────────────────────────────────

async def load_bars(
    ticker: str,
    feature_cols: list[str],
    lgbm_model: Any = None,
) -> list[dict[str, Any]]:
    """Load feature matrix + close prices and format into bar dicts for TradingEnv.

    Args:
        ticker: Stock ticker symbol.
        feature_cols: FFSA-selected feature columns.
        lgbm_model: Optional LGBMSignalModel — if provided, populates
                     lgbm_pred_return and lgbm_dir_prob per bar.
    """
    from sqlalchemy import select

    from src.data.db import FeatureMatrix, OHLCV1m, get_session_factory, init_db

    await init_db()
    sf = get_session_factory()

    # Feature matrix for this ticker
    async with sf() as session:
        result = await session.execute(
            select(FeatureMatrix.time, FeatureMatrix.features)
            .where(FeatureMatrix.ticker == ticker)
            .order_by(FeatureMatrix.time)
        )
        feat_rows = result.all()

    # Close prices
    async with sf() as session:
        result = await session.execute(
            select(OHLCV1m.time, OHLCV1m.close)
            .where(OHLCV1m.ticker == ticker)
            .order_by(OHLCV1m.time)
        )
        price_rows = result.all()

    if not feat_rows or not price_rows:
        return []

    # Build price lookup
    price_map = {r.time: float(r.close) for r in price_rows}

    # Top-10 FFSA feature names (for the TradingEnv state)
    top10 = feature_cols[:10]

    # LightGBM feature columns (may differ from FFSA top-10)
    lgbm_feat_cols = lgbm_model.feature_cols if lgbm_model else []

    bars: list[dict[str, Any]] = []
    for ts, feat_dict in feat_rows:
        if not isinstance(feat_dict, dict):
            continue
        close = price_map.get(ts)
        if close is None or close <= 0:
            continue

        # Extract top-10 FFSA feature values as an ordered list
        feat_values = [float(feat_dict.get(f, 0.0) or 0.0) for f in top10]

        # LightGBM predictions for this bar
        lgbm_pred_return = 0.0
        lgbm_dir_prob = 0.5
        if lgbm_model and lgbm_feat_cols:
            full_feat = np.array(
                [float(feat_dict.get(f, 0.0) or 0.0) for f in lgbm_feat_cols]
            )
            _, _, lgbm_pred_return, lgbm_dir_prob = lgbm_model.predict(full_feat)

        bars.append({
            "time":             ts,
            "close":            close,
            "features":         feat_values,
            # ML model signals — set to 0 for initial training
            # (RL agent learns from raw FFSA features in state obs)
            "ensemble_signal":  0.0,
            "transformer_conf": 0.0,
            "tcn_conf":         0.0,
            "sentiment_index":  0.0,
            # LightGBM predictions (fed into RL obs)
            "lgbm_pred_return": lgbm_pred_return,
            "lgbm_dir_prob":    lgbm_dir_prob,
            # Market context
            "vix":              20.0,   # placeholder until VIX feed is wired
            "regime":           0.0,    # 0=neutral
            # Pass through feature dict for MACD expert policy (BC)
            **{k: float(v or 0.0) for k, v in feat_dict.items()
               if k in ("macd", "macd_signal", "rsi_14")},
        })

    if lgbm_model:
        logger.info("  LightGBM predictions populated for %d bars", len(bars))

    return bars


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main(ticker: str, timesteps: int, bc_epochs: int, env_mode: str = "sizing") -> None:
    # Load FFSA feature list
    ffsa_files = sorted(Path("reports/drift").glob("ffsa_*.json"), reverse=True)
    if not ffsa_files:
        logger.error("No FFSA report found — run scripts/run_ffsa.py first")
        return
    with open(ffsa_files[0]) as f:
        ffsa = json.load(f)
    feature_cols: list[str] = ffsa["selected_features"]
    logger.info("FFSA features: %d selected from %s", len(feature_cols), ffsa_files[0].name)

    # Load LightGBM model for RL observations
    from src.models.lgbm import load_best_checkpoint
    lgbm_model = load_best_checkpoint()
    if lgbm_model:
        logger.info("LightGBM loaded: val_ic=%.4f, %d features", lgbm_model.val_ic, len(lgbm_model.feature_cols))
    else:
        logger.warning("No LightGBM checkpoint — RL obs will have lgbm_pred_return=0, lgbm_dir_prob=0.5")

    # Load bars
    logger.info("Loading bars for %s ...", ticker)
    bars = await load_bars(ticker, feature_cols, lgbm_model=lgbm_model)
    if len(bars) < 1000:
        logger.error("Too few bars for %s (%d). Need at least 1000.", ticker, len(bars))
        return
    logger.info("  %d bars loaded (%s → %s)", len(bars), bars[0]["time"], bars[-1]["time"])

    # Walk-forward split (80% train, 20% val)
    split = int(len(bars) * 0.80)
    bars_train, bars_val = bars[:split], bars[split:]
    logger.info("  Train: %d bars | Val: %d bars", len(bars_train), len(bars_val))

    # Train
    from src.rl.trainer import RLTrainer
    if env_mode == "sizing":
        from src.rl.position_sizing_env import PositionSizingEnv, SizingEnvConfig
        env_class = PositionSizingEnv
        cfg = SizingEnvConfig()
        bc_epochs = 0  # no BC for sizing env
        logger.info("Using PositionSizingEnv (5 sizing actions, LightGBM gates entry/exit)")
    else:
        from src.rl.trading_env import TradingEnv, EnvConfig
        env_class = TradingEnv
        cfg = EnvConfig()
        logger.info("Using TradingEnv (9 full actions)")
    trainer = RLTrainer(bars_train, bars_val, cfg=cfg, env_class=env_class)
    model = trainer.run(total_timesteps=timesteps, bc_epochs=bc_epochs)

    # Backtest on validation set
    print("\n" + "═" * 60)
    print("  BACKTEST — Validation Set")
    print("═" * 60)
    metrics = trainer.backtest(model)
    for k, v in metrics.items():
        print(f"  {k:<22s}: {v}")
    print("═" * 60)

    # Save backtest report
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / f"rl_backtest_{ticker}.json"
    with open(report_path, "w") as f:
        json.dump({"ticker": ticker, "timesteps": timesteps, **metrics}, f, indent=2)
    logger.info("Backtest report saved → %s", report_path)

    # Promotion gate check
    print()
    sharpe_ok = metrics.get("sharpe", 0) >= 1.5
    dd_ok     = metrics.get("max_drawdown", 1.0) <= 0.08
    wr_ok     = metrics.get("win_rate", 0) >= 0.52
    pf_ok     = metrics.get("profit_factor", 0) >= 1.4

    print(f"  Sharpe ≥ 1.5:      {'✓' if sharpe_ok else '✗'}  ({metrics.get('sharpe', 0):.3f})")
    print(f"  Max DD ≤ 8%:       {'✓' if dd_ok     else '✗'}  ({metrics.get('max_drawdown', 0)*100:.1f}%)")
    print(f"  Win rate ≥ 52%:    {'✓' if wr_ok     else '✗'}  ({metrics.get('win_rate', 0)*100:.1f}%)")
    print(f"  Profit factor ≥ 1.4: {'✓' if pf_ok   else '✗'}  ({metrics.get('profit_factor', 0):.2f})")

    if all([sharpe_ok, dd_ok, wr_ok, pf_ok]):
        print("\n  ✓ ALL GATES PASSED — ready for paper trading")
    else:
        print("\n  ✗ Not yet ready — needs more training data and model improvement")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker",     default="AAPL")
    parser.add_argument("--timesteps",  type=int, default=50_000)
    parser.add_argument("--bc-epochs",  type=int, default=10)
    parser.add_argument("--env",        choices=["full", "sizing"], default="sizing")
    args = parser.parse_args()
    asyncio.run(main(ticker=args.ticker, timesteps=args.timesteps, bc_epochs=args.bc_epochs, env_mode=args.env))
