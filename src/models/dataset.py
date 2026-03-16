"""StockSequenceDataset — PyTorch Dataset for Transformer + TCN training.

HYBRID mode (v3 — Regression + Binary Direction):
  - Head A target: raw 15m forward return (regression via Huber loss)
  - Head B target: binary direction label (1=Up if return > epsilon, 0=Down/Flat)
  - Combined loss: HuberLoss(Head A) + λ * BCEWithLogitsLoss(Head B)
  - The binary head acts as a "forcing function" preventing prediction collapse
    to zero — the model MUST commit to a direction.
  - Trading thresholds applied at INFERENCE time: |predicted_return| > 0.25%.
  - Stride = 15 — reduces autocorrelation in training data.
  - Data Augmentation: Gaussian noise injection + target scaling during training.

Each sample returns:
    x_1m:        (seq_len, n_features)   — 1m feature sequence
    x_5m:        (seq_5m,  n_features)   — 5m-resampled (every 5th bar)
    target_15m:  float  — regression target: 15m forward log-return
    target_5m:   float  — auxiliary regression: 5m forward log-return
    target_30m:  float  — auxiliary regression: 30m forward log-return
    dir_label:   float  — binary direction: 1.0 if return > epsilon, else 0.0
    ticker_id:   int    — ticker index for learned ticker embeddings

Walk-forward split is done externally by time cutoff before passing to Dataset.
"""

from __future__ import annotations

import logging
import structlog

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = structlog.get_logger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

SEQ_LEN    = 60    # 1m bars per sample (1 hour of context)
SEQ_5M     = 12    # 5m bars per sample (SEQ_LEN // 5)
STRIDE     = 15    # larger stride reduces training autocorrelation

# Multi-horizon forward bars
FORWARD_N  = 15    # PRIMARY target: 15-minute return
FORWARD_5  = 5     # auxiliary: 5-minute return
FORWARD_30 = 30    # auxiliary: 30-minute return

# Trading threshold applied at INFERENCE time (not during training)
TRADING_THRESHOLD = 0.0025   # 0.25% — only generate signal if |pred| > this

# Binary direction label epsilon: returns within ±epsilon are treated as "flat/down" (label=0)
# This reduces noise from micro-moves that have no directional signal.
DIRECTION_EPSILON = 0.0001   # 0.01% — below this is "no direction"

# Data augmentation constants
NOISE_STD = 0.01            # Gaussian noise std for input features
TARGET_SCALE_RANGE = 0.05   # ±5% random scaling of regression targets

# Legacy alias
UP_DOWN_PCT = 0.33


# ─── Normalization modes ─────────────────────────────────────────────────────

ATR_NORMALIZED_FEATURES = {
    "macd", "macd_signal", "macd_hist", "mom_10",
    "bb_upper", "bb_lower", "bb_mid",
    "kc_upper", "kc_mid", "kc_lower",
    "ema_9", "ema_21", "ema_50",
    "atr_14", "vwap", "vwap_daily",
}

VOLUME_NORMALIZED_FEATURES = {
    "obv",
}

PASSTHROUGH_FEATURES = {
    "rsi_14", "stoch_k", "stoch_d", "willr_14", "cci_20", "mfi_14",
    "bb_width", "bb_pct", "atr_pct", "vwap_dev", "vwap_slope_5", "vwap_slope_15",
    "orb_range_atr", "orb_dev", "returns_1b", "returns_5b", "returns_15b",
    "roc_10", "high_low_range", "close_vs_high", "gap_pct",
    "obv_pct", "vol_ratio", "vpin_50", "vpin_zscore",
    "min_since_open", "day_of_week", "time_to_close",
    "vol_seasonal_ratio", "adx", "dmp", "dmn", "regime",
    "rs_15m", "rs_1m", "rs_vwap_dev", "regime_time",
    "ema_cross_9_21", "ema_cross_21_50", "vwap_above",
    "orb_break_up", "orb_break_dn",
    "is_open_window", "is_close_window", "is_lunch",
    "gex_net", "gex_zscore", "gex_call_pct",
}


# ─── Dataset ──────────────────────────────────────────────────────────────────

class StockSequenceDataset(Dataset):
    """Overlapping sequence dataset with regression + binary direction targets.

    _index_map entries: (ticker_idx, end_row, target_15m, target_5m, target_30m, dir_label)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        seq_len: int = SEQ_LEN,
        stride: int = STRIDE,
        ticker_to_id: dict[str, int] | None = None,
    ) -> None:
        self.feature_cols = feature_cols
        self.seq_len      = seq_len
        self.seq_5m       = max(1, seq_len // 5)
        self.stride       = stride
        self._augment     = False

        self._feats:      list[np.ndarray] = []
        self._ticker_ids: list[int] = []

        # (ticker_idx, end_row, target_15m, target_5m, target_30m, dir_label)
        self._index_map: list[tuple[int, int, float, float, float, float]] = []

        if ticker_to_id is None:
            unique_tickers = sorted(df["ticker"].unique())
            self.ticker_to_id = {t: i for i, t in enumerate(unique_tickers)}
        else:
            self.ticker_to_id = ticker_to_id
        self.n_tickers = max(self.ticker_to_id.values()) + 1 if self.ticker_to_id else 1

        self._atr_mask = np.array([c in ATR_NORMALIZED_FEATURES for c in feature_cols])
        self._vol_mask = np.array([c in VOLUME_NORMALIZED_FEATURES for c in feature_cols])
        self._pass_mask = ~(self._atr_mask | self._vol_mask)

        max_forward = max(FORWARD_N, FORWARD_5, FORWARD_30)

        for t_idx, (ticker, grp) in enumerate(df.groupby("ticker")):
            grp = grp.sort_values("time").reset_index(drop=True)
            feats = grp[feature_cols].values.astype(np.float32)

            fwd_15 = grp["forward_return"].values.astype(np.float32)
            fwd_5  = grp.get("forward_return_5m",  pd.Series(np.nan, index=grp.index)).values.astype(np.float32)
            fwd_30 = grp.get("forward_return_30m", pd.Series(np.nan, index=grp.index)).values.astype(np.float32)

            self._feats.append(feats)
            self._ticker_ids.append(self.ticker_to_id.get(ticker, 0))

            n = len(grp)
            for end in range(seq_len, n - max_forward, stride):
                r15 = fwd_15[end]
                r5  = fwd_5[end]
                r30 = fwd_30[end]

                if np.isnan(r15):
                    continue
                window = feats[end - seq_len : end]
                if np.isnan(window).mean() > 0.20:
                    continue

                r5  = r5 if not np.isnan(r5) else 0.0
                r30 = r30 if not np.isnan(r30) else 0.0

                # Binary direction label: 1.0 if return > epsilon, 0.0 otherwise
                dir_label = 1.0 if r15 > DIRECTION_EPSILON else 0.0

                self._index_map.append((t_idx, end, float(r15), float(r5), float(r30), dir_label))

        # Log distribution statistics
        targets_15 = np.array([m[2] for m in self._index_map])
        dir_labels = np.array([m[5] for m in self._index_map])
        up_pct = dir_labels.mean() * 100
        logger.info(
            "dataset_built: %d seqs | tickers=%d | target_15m: mean=%.5f std=%.5f | "
            "dir_up=%.1f%% dir_down=%.1f%% | stride=%d",
            len(self._index_map), len(self._feats),
            targets_15.mean(), targets_15.std(),
            up_pct, 100 - up_pct, stride,
        )

    # ── Per-ticker sample weights (for WeightedRandomSampler) ────────────────

    def get_ticker_sample_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for t_idx, _, _, _, _, _ in self._index_map:
            counts[t_idx] = counts.get(t_idx, 0) + 1
        return counts

    def get_sample_weights(self) -> np.ndarray:
        """Per-sample weight inversely proportional to ticker frequency."""
        counts = self.get_ticker_sample_counts()
        total = sum(counts.values())
        n_tickers_actual = len(counts)
        weights = np.zeros(len(self._index_map), dtype=np.float64)
        for i, (t_idx, _, _, _, _, _) in enumerate(self._index_map):
            weights[i] = total / (n_tickers_actual * counts[t_idx])
        return weights

    # ── Augmentation control ─────────────────────────────────────────────────

    def set_augmentation(self, enabled: bool) -> None:
        self._augment = enabled

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, float, float, float, float, int]:
        t_idx, end, tgt_15, tgt_5, tgt_30, dir_label = self._index_map[idx]
        seq = self._feats[t_idx][end - self.seq_len : end].copy()
        np.nan_to_num(seq, copy=False, nan=0.0)

        # ── Regime-aware normalization ─────────────────────────────────────────
        if self._atr_mask.any():
            atr_col_idx = np.where(self._atr_mask)[0]
            for ci in atr_col_idx:
                col_data = seq[:, ci]
                scale = np.abs(col_data).mean()
                if scale > 1e-8:
                    seq[:, ci] = col_data / scale

        if self._vol_mask.any():
            vol_col_idx = np.where(self._vol_mask)[0]
            for ci in vol_col_idx:
                col_data = seq[:, ci]
                scale = np.abs(col_data).mean()
                if scale > 1e-8:
                    seq[:, ci] = col_data / scale

        if self._pass_mask.any():
            pass_col_idx = np.where(self._pass_mask)[0]
            seq[:, pass_col_idx] = np.clip(seq[:, pass_col_idx], -10.0, 10.0)

        # ── Data augmentation (training only) ──────────────────────────────────
        if self._augment:
            noise = np.random.normal(0.0, NOISE_STD, size=seq.shape).astype(np.float32)
            seq = seq + noise
            scale_factor = 1.0 + np.random.uniform(-TARGET_SCALE_RANGE, TARGET_SCALE_RANGE)
            tgt_15 = tgt_15 * scale_factor
            tgt_5  = tgt_5  * scale_factor
            tgt_30 = tgt_30 * scale_factor
            # Note: dir_label is NOT scaled — it's derived from the original return

        seq_5m = seq[4::5][-self.seq_5m :]
        if len(seq_5m) < self.seq_5m:
            pad = np.zeros((self.seq_5m - len(seq_5m), seq.shape[1]), dtype=np.float32)
            seq_5m = np.vstack([pad, seq_5m])

        x_1m = torch.from_numpy(seq)
        x_5m = torch.from_numpy(seq_5m)
        ticker_id = self._ticker_ids[t_idx]
        return x_1m, x_5m, tgt_15, tgt_5, tgt_30, dir_label, ticker_id


# ─── Legacy helpers ──────────────────────────────────────────────────────────

def compute_class_weights(dataset: StockSequenceDataset) -> torch.Tensor:
    """Legacy helper — returns [1, 1, 1]."""
    return torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
