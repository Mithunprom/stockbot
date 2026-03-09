"""FFSA — Financial Feature Significance Analysis.

Uses LightGBM + SHAP to rank features by predictive power for forward returns.
Selected top-N features are stored and used by all downstream models.

Weekly APScheduler job re-runs FFSA to adapt to regime changes.
Results are logged to reports/drift/ for the Model Drift Agent.

Usage:
    ffsa = FFSAPipeline()
    selected = await ffsa.run(ticker_dfs)     # returns list of feature names
    ffsa.save("reports/drift/")
"""

from __future__ import annotations

import json
import logging
import structlog
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit

from src.features.indicators import FEATURE_COLUMNS, compute_indicators

logger = structlog.get_logger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

TOP_N_FEATURES = 30
FORWARD_RETURN_BARS = 5         # predict 5-bar forward return
MIN_SAMPLES = 2000              # minimum rows required to run FFSA
FFSA_REPORT_DIR = Path("reports/drift")


# ─── FFSA Pipeline ────────────────────────────────────────────────────────────

class FFSAPipeline:
    """Run LightGBM + SHAP feature selection on historical bar data.

    Thread-safe: each run creates a fresh model instance.
    """

    def __init__(self, top_n: int = TOP_N_FEATURES) -> None:
        self.top_n = top_n
        self.selected_features: list[str] = []
        self.shap_importances: dict[str, float] = {}
        self.version: str = ""

    # ── Data prep ─────────────────────────────────────────────────────────────

    def _build_feature_matrix(
        self, ticker_dfs: dict[str, pd.DataFrame]
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Stack all tickers into a single (X, y) matrix."""
        frames: list[pd.DataFrame] = []
        for ticker, raw_df in ticker_dfs.items():
            if len(raw_df) < 60:
                continue
            try:
                df = compute_indicators(raw_df, shift=True)
            except Exception as exc:
                logger.warning("ffsa_indicator_error: %s: %s", ticker, exc)
                continue

            available_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
            df = df[["close"] + available_cols].copy()

            # Forward return target (shifted back so it's the future)
            df["target"] = df["close"].pct_change(FORWARD_RETURN_BARS).shift(-FORWARD_RETURN_BARS)
            df = df.dropna()

            df = df[available_cols + ["target"]]
            df["ticker"] = ticker
            frames.append(df)

        if not frames:
            raise ValueError("No valid ticker data for FFSA")

        combined = pd.concat(frames, ignore_index=True)
        logger.info("ffsa_matrix_built: %d rows, %d tickers", len(combined), len(frames))

        y = combined["target"]
        X = combined.drop(columns=["target", "ticker"])
        return X, y

    # ── Model training ────────────────────────────────────────────────────────

    def _train_lgbm(self, X: pd.DataFrame, y: pd.Series) -> LGBMRegressor:
        """Train LightGBM regressor with time-series cross-validation."""
        model = LGBMRegressor(
            n_estimators=300,
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

        # Use last fold for training (most recent data)
        tscv = TimeSeriesSplit(n_splits=3)
        splits = list(tscv.split(X))
        train_idx, val_idx = splits[-1]

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[],  # silence verbose output
        )

        logger.info("lgbm_trained: train=%d val=%d", len(X_train), len(X_val))
        return model

    # ── SHAP importance ───────────────────────────────────────────────────────

    def _compute_shap_importance(
        self, model: LGBMRegressor, X: pd.DataFrame
    ) -> dict[str, float]:
        """Compute mean |SHAP| for each feature on a sample of X."""
        sample_size = min(5000, len(X))
        X_sample = X.sample(n=sample_size, random_state=42)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        return dict(zip(X.columns, mean_abs_shap.tolist()))

    # ── Main run ──────────────────────────────────────────────────────────────

    async def run(self, ticker_dfs: dict[str, pd.DataFrame]) -> list[str]:
        """Run the full FFSA pipeline. Returns list of selected feature names.

        This is CPU-intensive — consider running in a thread pool for production.
        """
        import asyncio

        loop = asyncio.get_event_loop()
        selected = await loop.run_in_executor(None, self._run_sync, ticker_dfs)
        return selected

    def _run_sync(self, ticker_dfs: dict[str, pd.DataFrame]) -> list[str]:
        """Synchronous FFSA execution (runs in thread pool)."""
        X, y = self._build_feature_matrix(ticker_dfs)

        if len(X) < MIN_SAMPLES:
            logger.warning("ffsa_insufficient_data: %d rows, need %d", len(X), MIN_SAMPLES)
            # Fall back to full feature set
            self.selected_features = [c for c in FEATURE_COLUMNS if c in X.columns]
            return self.selected_features

        model = self._train_lgbm(X, y)
        importances = self._compute_shap_importance(model, X)

        # Rank and select top N
        ranked = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        self.shap_importances = dict(ranked)
        self.selected_features = [feat for feat, _ in ranked[: self.top_n]]

        self.version = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M")
        logger.info(
            "ffsa_complete: top5=%s total=%d version=%s",
            self.selected_features[:5], len(self.selected_features), self.version,
        )
        return self.selected_features

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, report_dir: Path | str | None = None) -> Path:
        """Serialize FFSA results to a JSON report file."""
        out_dir = Path(report_dir or FFSA_REPORT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)

        date_str = datetime.now(timezone.utc).strftime("%Y-W%W")
        out_path = out_dir / f"ffsa_{date_str}.json"

        payload = {
            "version": self.version,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "top_n": self.top_n,
            "selected_features": self.selected_features,
            "shap_importances": self.shap_importances,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)

        logger.info("ffsa_saved: %s", out_path)
        return out_path

    @classmethod
    def load(cls, report_path: Path | str) -> FFSAPipeline:
        """Load a previously saved FFSA result."""
        with open(report_path) as f:
            data = json.load(f)

        pipeline = cls(top_n=data["top_n"])
        pipeline.version = data["version"]
        pipeline.selected_features = data["selected_features"]
        pipeline.shap_importances = data["shap_importances"]
        return pipeline

    @classmethod
    def load_latest(cls, report_dir: Path | str | None = None) -> FFSAPipeline | None:
        """Load the most recently saved FFSA result, or None if none exists."""
        out_dir = Path(report_dir or FFSA_REPORT_DIR)
        files = sorted(out_dir.glob("ffsa_*.json"), reverse=True)
        if not files:
            return None
        return cls.load(files[0])


# ─── APScheduler job ──────────────────────────────────────────────────────────

async def weekly_ffsa_job(ticker_dfs: dict[str, pd.DataFrame]) -> None:
    """Scheduled weekly FFSA re-run. Called by APScheduler in agents/drift.py."""
    pipeline = FFSAPipeline()
    await pipeline.run(ticker_dfs)
    pipeline.save()
    logger.info("weekly_ffsa_complete", features=pipeline.selected_features[:5])
