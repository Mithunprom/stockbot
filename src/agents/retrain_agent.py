"""Pre-market retraining agent.

Runs daily at 08:00 ET (before 09:30 open):
  1. Build features — incremental (yesterday's new bars only)
  2. Retrain LightGBM on full dataset with latest features
  3. Upload new model checkpoint to S3 backup
  4. Hot-reload model in EnsembleEngine for today's trading

Total runtime: ~30-60 seconds (feature build) + ~22 seconds (LightGBM train).
"""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class RetrainAgent:
    """Orchestrates daily pre-market model retraining.

    Args:
        ensemble: EnsembleEngine instance (for hot-reload after retrain).
        session_factory: SQLAlchemy async_sessionmaker.
        universe: List of ticker symbols.
    """

    def __init__(
        self,
        ensemble: Any,
        session_factory: Any,
        universe: list[str],
    ) -> None:
        self._ensemble = ensemble
        self._sf = session_factory
        self._universe = universe

    async def run(self) -> dict[str, Any]:
        """Run the full retrain pipeline. Returns a summary dict."""
        start = datetime.now(timezone.utc)
        logger.info("retrain_agent_started")
        report: dict[str, Any] = {"started_at": start.isoformat()}

        try:
            # Step 1: Build features (incremental — last 2 days to catch gaps)
            since = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%d")
            feat_count = await self._build_features(since)
            report["features_updated"] = feat_count
            logger.info("retrain_features_done", rows=feat_count)

            # Step 2: Retrain LightGBM
            metrics = await self._train_lgbm()
            report["lgbm_metrics"] = metrics
            logger.info("retrain_lgbm_done", val_ic=metrics.get("val_ic"), dir_acc=metrics.get("val_dir_acc"))

            # Step 3: Upload to S3
            s3_result = await self._upload_to_s3()
            report["s3_upload"] = s3_result
            logger.info("retrain_s3_upload_done", result=s3_result)

            # Step 4: Hot-reload LightGBM in ensemble
            self._reload_model()
            report["model_reloaded"] = True
            logger.info("retrain_model_reloaded")

            report["status"] = "success"
        except Exception as exc:
            logger.error("retrain_agent_failed", error=str(exc), exc_info=True)
            report["status"] = "failed"
            report["error"] = str(exc)

        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        report["elapsed_seconds"] = round(elapsed, 1)
        logger.info("retrain_agent_complete", status=report["status"], elapsed=elapsed)

        # Write report
        self._write_report(report)
        return report

    async def _build_features(self, since: str) -> int:
        """Build features incrementally for recent bars."""
        from datetime import datetime as dt

        from scripts.build_features import run

        since_dt = dt.strptime(since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        # Filter to equity tickers only (no crypto — they don't have OHLCV in DB)
        equity_tickers = [t for t in self._universe if "/" not in t]

        await run(tickers=equity_tickers, since=since_dt)

        # Count rows written (approximate from DB)
        from sqlalchemy import text
        async with self._sf() as session:
            result = await session.execute(text("SELECT COUNT(*) FROM feature_matrix"))
            total = result.scalar()
        return total or 0

    async def _train_lgbm(self) -> dict[str, float]:
        """Retrain LightGBM on full dataset."""
        import json

        import numpy as np
        import pandas as pd

        from scripts.train_lgbm import load_data, walk_forward_split
        from src.models.lgbm import LGBMSignalModel

        merged, feature_cols = await load_data(top_n=30, max_rows=100_000)
        train_df, val_df = walk_forward_split(merged)

        X_train = train_df[feature_cols].fillna(0).astype(np.float32)
        y_train = train_df["forward_return"].values.astype(np.float32)
        X_val = val_df[feature_cols].fillna(0).astype(np.float32)
        y_val = val_df["forward_return"].values.astype(np.float32)

        model = LGBMSignalModel(feature_cols=feature_cols)
        metrics = model.train(X_train, y_train, X_val, y_val, direction_epsilon=0.0001)
        path = model.save()
        logger.info("lgbm_saved", path=str(path), val_ic=metrics["val_ic"])
        return metrics

    async def _upload_to_s3(self) -> dict[str, str]:
        """Upload latest model checkpoints to S3."""
        from main import upload_models_to_s3
        return await upload_models_to_s3()

    def _reload_model(self) -> None:
        """Hot-reload LightGBM in the ensemble engine."""
        self._ensemble._load_lgbm()
        logger.info("lgbm_hot_reloaded", ic=getattr(self._ensemble._lgbm, "val_ic", None))

    def _write_report(self, report: dict[str, Any]) -> None:
        """Write retrain report to reports/retrain/."""
        import json

        report_dir = Path("reports/retrain")
        report_dir.mkdir(parents=True, exist_ok=True)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = report_dir / f"retrain_{today}.json"

        # Make report JSON-serializable
        serializable = {}
        for k, v in report.items():
            if isinstance(v, dict):
                serializable[k] = {str(kk): str(vv) for kk, vv in v.items()}
            else:
                serializable[k] = v

        with open(path, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        logger.info("retrain_report_saved", path=str(path))
