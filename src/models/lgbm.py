"""LightGBM signal model — primary signal generator (replaces Transformer/TCN).

Uses point-in-time features (single bar) rather than sequences.
Proven IC=0.21 on walk-forward validation vs IC≈0 for neural nets.
The signal is non-linear — tree models capture it, neural nets can't.

Two internal models:
  1. LGBMRegressor  — predicts 15m forward return (continuous)
  2. LGBMClassifier — predicts direction probability (binary up/down)

Inference:
  direction  = +1 if pred_return > threshold, -1 if < -threshold, else 0
  confidence = calibrated probability scaled to [0, 1]
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

TRADING_THRESHOLD = 0.0005  # 0.05% — direction filter; entry gate (SIZING_COST_THRESHOLD) handles actual trade gating
MODEL_DIR = Path("models/lgbm")


class LGBMSignalModel:
    """LightGBM-based signal model with regressor + classifier heads."""

    def __init__(self, feature_cols: list[str] | None = None) -> None:
        self.regressor: Any | None = None
        self.classifier: Any | None = None
        self.feature_cols: list[str] = feature_cols or []
        self.train_ic: float = 0.0
        self.val_ic: float = 0.0
        self.val_dir_acc: float = 0.0

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        direction_epsilon: float = 0.0001,
    ) -> dict[str, float]:
        """Train regressor + classifier. Returns metrics dict."""
        from lightgbm import LGBMClassifier, LGBMRegressor

        lgbm_params = dict(
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

        # Head A: Regressor — predict raw 15m forward return
        self.regressor = LGBMRegressor(**lgbm_params)
        self.regressor.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[],
        )

        # Head B: Classifier — predict direction (up/down)
        dir_train = (y_train > direction_epsilon).astype(int)
        dir_val = (y_val > direction_epsilon).astype(int)

        self.classifier = LGBMClassifier(**lgbm_params)
        self.classifier.fit(
            X_train, dir_train,
            eval_set=[(X_val, dir_val)],
            callbacks=[],
        )

        # Compute metrics
        val_pred = self.regressor.predict(X_val)
        ic = float(np.corrcoef(val_pred, y_val)[0, 1])
        if np.isnan(ic):
            ic = 0.0
        self.val_ic = ic

        dir_pred = self.classifier.predict(X_val)
        self.val_dir_acc = float((dir_pred == dir_val).mean())

        train_pred = self.regressor.predict(X_train)
        train_ic = float(np.corrcoef(train_pred, y_train)[0, 1])
        if np.isnan(train_ic):
            train_ic = 0.0
        self.train_ic = train_ic

        return {
            "train_ic": self.train_ic,
            "val_ic": self.val_ic,
            "val_dir_acc": self.val_dir_acc,
        }

    def predict(self, features: np.ndarray) -> tuple[float, float, float, float]:
        """Predict direction, confidence, raw return, and dir probability.

        Args:
            features: (n_features,) array of raw feature values.

        Returns:
            (direction, confidence, pred_return, dir_prob) where:
                direction: +1.0 (long), -1.0 (short), or 0.0 (flat)
                confidence: [0, 1] — scaled distance of dir prob from 0.5
                pred_return: raw predicted 15m forward return
                dir_prob: raw classifier probability of up direction [0, 1]
        """
        if self.regressor is None or self.classifier is None:
            return 0.0, 0.0, 0.0, 0.5

        n_expected = self.regressor.n_features_
        if features.shape[0] < n_expected:
            # Pad missing features with 0 (handles old DB rows with fewer features)
            padded = np.zeros(n_expected, dtype=np.float32)
            padded[:features.shape[0]] = features
            features = padded
        elif features.shape[0] > n_expected:
            features = features[:n_expected]

        x = features.reshape(1, -1)
        pred_return = float(self.regressor.predict(x)[0])
        dir_prob = float(self.classifier.predict_proba(x)[0, 1])

        if pred_return > TRADING_THRESHOLD:
            direction = 1.0
        elif pred_return < -TRADING_THRESHOLD:
            direction = -1.0
        else:
            direction = 0.0

        confidence = min(abs(dir_prob - 0.5) * 2, 1.0)
        return direction, confidence, pred_return, dir_prob

    def predict_return(self, features: np.ndarray) -> float:
        """Predict raw 15m forward return."""
        if self.regressor is None:
            return 0.0
        return float(self.regressor.predict(features.reshape(1, -1))[0])

    def save(self, path: Path | None = None) -> Path:
        """Save model to disk (pickle + JSON metadata)."""
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path = path or MODEL_DIR / f"lgbm_ic_{self.val_ic:.4f}.pkl"

        data = {
            "regressor": self.regressor,
            "classifier": self.classifier,
            "feature_cols": self.feature_cols,
            "train_ic": self.train_ic,
            "val_ic": self.val_ic,
            "val_dir_acc": self.val_dir_acc,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

        meta_path = path.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump({
                "feature_cols": self.feature_cols,
                "train_ic": self.train_ic,
                "val_ic": self.val_ic,
                "val_dir_acc": self.val_dir_acc,
                "n_features": len(self.feature_cols),
            }, f, indent=2)

        return path

    @classmethod
    def load(cls, path: Path | None = None) -> LGBMSignalModel:
        """Load model from disk. Auto-selects best checkpoint by IC."""
        if path is None:
            candidates = sorted(MODEL_DIR.glob("lgbm_ic_*.pkl"), reverse=True)
            if not candidates:
                raise FileNotFoundError(f"No LightGBM checkpoints in {MODEL_DIR}")
            path = candidates[0]

        with open(path, "rb") as f:
            data = pickle.load(f)

        model = cls(feature_cols=data["feature_cols"])
        model.regressor = data["regressor"]
        model.classifier = data["classifier"]
        model.train_ic = data.get("train_ic", 0.0)
        model.val_ic = data.get("val_ic", 0.0)
        model.val_dir_acc = data.get("val_dir_acc", 0.0)
        return model


def load_best_checkpoint() -> LGBMSignalModel | None:
    """Load the best LightGBM checkpoint, or None if unavailable."""
    try:
        return LGBMSignalModel.load()
    except (FileNotFoundError, Exception) as e:
        logger.warning("lgbm_checkpoint_not_found: %s", e)
        return None
