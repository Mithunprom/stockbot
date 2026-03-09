"""StockSequenceDataset — PyTorch Dataset for Transformer + TCN training.

Loads FFSA features from a merged DataFrame and builds overlapping sequences
with 3-class direction labels computed from 5-bar forward returns.

Each sample returns:
    x_1m:  (seq_len, n_features)  — 1m feature sequence
    x_5m:  (seq_5m, n_features)   — 5m-resampled features (every 5th bar)
    label: int                     — 0=down, 1=flat, 2=up

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

SEQ_LEN   = 60     # 1m bars per sample (1 hour)
SEQ_5M    = 12     # 5m bars per sample (SEQ_LEN // 5)
STRIDE    = 3      # step between samples (avoids redundant overlap)
THRESHOLD = 0.003  # ±0.3% fallback threshold (overridden by percentile labels)
FORWARD_N = 5      # forward bars used for the return label

# Percentile-based labelling: bottom UP_DOWN_PCT → down, top → up, middle → flat
# 0.33 gives ~33/33/33 balanced classes, preventing model collapse under weighted loss.
UP_DOWN_PCT = 0.33  # 33% each side → balanced 3-class split


# ─── Class-weight helper ──────────────────────────────────────────────────────

def compute_class_weights(dataset: "StockSequenceDataset") -> torch.Tensor:
    """Compute inverse-frequency class weights for focal/CE loss."""
    labels = np.array([dataset._index_map[i][2] for i in range(len(dataset))])
    counts = np.bincount(labels, minlength=3).astype(float)
    counts = np.maximum(counts, 1.0)  # avoid div-by-zero
    weights = counts.sum() / (3 * counts)
    return torch.tensor(weights, dtype=torch.float32)


# ─── Dataset ──────────────────────────────────────────────────────────────────

class StockSequenceDataset(Dataset):
    """Memory-efficient overlapping sequence dataset.

    Stores one feature array per ticker; samples are resolved via an index map
    that records (ticker_idx, end_row, label) — no per-sample tensor copies.
    """

    def __init__(
        self,
        df: pd.DataFrame,           # must contain feature_cols + 'forward_return' + 'ticker'
        feature_cols: list[str],
        seq_len: int = SEQ_LEN,
        stride: int = STRIDE,
        up_down_pct: float = UP_DOWN_PCT,
    ) -> None:
        self.feature_cols = feature_cols
        self.seq_len      = seq_len
        self.seq_5m       = max(1, seq_len // 5)
        self.stride       = stride

        # Raw arrays keyed by ticker index
        self._feats:   list[np.ndarray] = []   # (n_rows, n_features) float32
        self._targets: list[np.ndarray] = []   # (n_rows,)            float32

        # (ticker_idx, end_row, label)
        self._index_map: list[tuple[int, int, int]] = []

        for t_idx, (ticker, grp) in enumerate(df.groupby("ticker")):
            grp = grp.sort_values("time").reset_index(drop=True)
            feats   = grp[feature_cols].values.astype(np.float32)
            targets = grp["forward_return"].values.astype(np.float32)

            self._feats.append(feats)
            self._targets.append(targets)

            # Per-ticker percentile thresholds — adapts to each stock's volatility
            valid_targets = targets[~np.isnan(targets)]
            lo = float(np.percentile(valid_targets, up_down_pct * 100))
            hi = float(np.percentile(valid_targets, (1 - up_down_pct) * 100))

            n = len(grp)
            for end in range(seq_len, n, stride):
                fwd = targets[end]
                if np.isnan(fwd):
                    continue
                window = feats[end - seq_len : end]
                if np.isnan(window).mean() > 0.20:   # >20% NaN → skip (warmup)
                    continue

                label = 2 if fwd >= hi else (0 if fwd <= lo else 1)
                self._index_map.append((t_idx, end, label))

        labels_arr = np.array([m[2] for m in self._index_map])
        counts = np.bincount(labels_arr, minlength=3)
        logger.info(
            "dataset_built: %d seqs | tickers=%d | down=%d flat=%d up=%d",
            len(self._index_map),
            len(self._feats),
            counts[0], counts[1], counts[2],
        )

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        t_idx, end, label = self._index_map[idx]
        seq = self._feats[t_idx][end - self.seq_len : end].copy()
        np.nan_to_num(seq, copy=False, nan=0.0)

        # 5m resample: every 5th bar, take the most-recent seq_5m bars
        seq_5m = seq[4::5][-self.seq_5m :]
        if len(seq_5m) < self.seq_5m:
            pad = np.zeros((self.seq_5m - len(seq_5m), seq.shape[1]), dtype=np.float32)
            seq_5m = np.vstack([pad, seq_5m])

        x_1m = torch.from_numpy(seq)       # (seq_len, n_features)
        x_5m = torch.from_numpy(seq_5m)    # (seq_5m,  n_features)
        return x_1m, x_5m, label
