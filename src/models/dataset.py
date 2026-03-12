"""StockSequenceDataset — PyTorch Dataset for Transformer + TCN training.

Multi-horizon labels (NotebookLM recommendation 2026-03-12):
  - PRIMARY label: 15m forward return (FORWARD_N=15) — more predictable than 5m
  - AUXILIARY labels: 5m and 30m — multi-task regularization for shared repr.
  - Fixed ±% thresholds instead of percentile splits — only label economically
    significant moves (above bid-ask noise); flat/ambiguous moves stay class=1.
  - Stride increased from 3 → 15 — reduces autocorrelation in training data.

Each sample returns:
    x_1m:        (seq_len, n_features)   — 1m feature sequence
    x_5m:        (seq_5m,  n_features)   — 5m-resampled (every 5th bar)
    label_15m:   int  — PRIMARY: 0=down, 1=flat, 2=up at +15 bars
    label_5m:    int  — auxiliary: direction at +5 bars
    label_30m:   int  — auxiliary: direction at +30 bars

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
STRIDE     = 15    # was 3 — larger stride reduces training autocorrelation

# Multi-horizon forward bars
FORWARD_N  = 15    # PRIMARY target: 15-minute direction (was 5)
FORWARD_5  = 5     # auxiliary: 5-minute direction
FORWARD_30 = 30    # auxiliary: 30-minute direction

# Fixed threshold labels — only predict moves above bid-ask noise.
# Anything inside ±THRESH is labeled "flat" (class=1).
# These are calibrated for liquid large-caps on Alpaca IEX:
#   5m  ±0.10%: above typical spread, detects meaningful 5m momentum
#   15m ±0.20%: captures genuine intraday moves
#   30m ±0.30%: half-hour trend confirmation
THRESH_5M  = 0.0010   # ±0.10%
THRESH_15M = 0.0020   # ±0.20%
THRESH_30M = 0.0030   # ±0.30%

# Legacy alias used by train_models.py for DB query
UP_DOWN_PCT = 0.33    # kept for backward compat — not used for labeling anymore


# ─── Class-weight helper ──────────────────────────────────────────────────────

def compute_class_weights(dataset: "StockSequenceDataset") -> torch.Tensor:
    """Compute inverse-frequency class weights for focal/CE loss (primary label)."""
    labels = np.array([dataset._index_map[i][2] for i in range(len(dataset))])
    counts = np.bincount(labels, minlength=3).astype(float)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (3 * counts)
    return torch.tensor(weights, dtype=torch.float32)


# ─── Label helper ────────────────────────────────────────────────────────────

def _fixed_label(ret: float, threshold: float) -> int:
    """Map a return to 0/1/2 using fixed ± threshold."""
    if ret >= threshold:
        return 2   # up
    if ret <= -threshold:
        return 0   # down
    return 1       # flat


# ─── Dataset ──────────────────────────────────────────────────────────────────

class StockSequenceDataset(Dataset):
    """Memory-efficient overlapping sequence dataset with multi-horizon labels.

    _index_map entries: (ticker_idx, end_row, label_15m, label_5m, label_30m)
    """

    def __init__(
        self,
        df: pd.DataFrame,           # must contain feature_cols + forward_return* + 'ticker'
        feature_cols: list[str],
        seq_len: int = SEQ_LEN,
        stride: int = STRIDE,
    ) -> None:
        self.feature_cols = feature_cols
        self.seq_len      = seq_len
        self.seq_5m       = max(1, seq_len // 5)
        self.stride       = stride

        self._feats:      list[np.ndarray] = []
        self._targets_15: list[np.ndarray] = []
        self._targets_5:  list[np.ndarray] = []
        self._targets_30: list[np.ndarray] = []

        # (ticker_idx, end_row, label_15m, label_5m, label_30m)
        self._index_map: list[tuple[int, int, int, int, int]] = []

        max_forward = max(FORWARD_N, FORWARD_5, FORWARD_30)

        for t_idx, (ticker, grp) in enumerate(df.groupby("ticker")):
            grp = grp.sort_values("time").reset_index(drop=True)
            feats = grp[feature_cols].values.astype(np.float32)

            # All three forward return columns
            fwd_15 = grp["forward_return"].values.astype(np.float32)
            fwd_5  = grp.get("forward_return_5m",  pd.Series(np.nan, index=grp.index)).values.astype(np.float32)
            fwd_30 = grp.get("forward_return_30m", pd.Series(np.nan, index=grp.index)).values.astype(np.float32)

            self._feats.append(feats)
            self._targets_15.append(fwd_15)
            self._targets_5.append(fwd_5)
            self._targets_30.append(fwd_30)

            n = len(grp)
            for end in range(seq_len, n - max_forward, stride):
                r15 = fwd_15[end]
                r5  = fwd_5[end]
                r30 = fwd_30[end]

                # Skip if primary label is missing
                if np.isnan(r15):
                    continue
                window = feats[end - seq_len : end]
                if np.isnan(window).mean() > 0.20:
                    continue

                lbl_15 = _fixed_label(float(r15), THRESH_15M)
                lbl_5  = _fixed_label(float(r5),  THRESH_5M)  if not np.isnan(r5)  else 1
                lbl_30 = _fixed_label(float(r30), THRESH_30M) if not np.isnan(r30) else 1
                self._index_map.append((t_idx, end, lbl_15, lbl_5, lbl_30))

        labels_15 = np.array([m[2] for m in self._index_map])
        counts = np.bincount(labels_15, minlength=3)
        logger.info(
            "dataset_built: %d seqs | tickers=%d | 15m: down=%d flat=%d up=%d | stride=%d",
            len(self._index_map),
            len(self._feats),
            counts[0], counts[1], counts[2],
            stride,
        )

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
        t_idx, end, lbl_15, lbl_5, lbl_30 = self._index_map[idx]
        seq = self._feats[t_idx][end - self.seq_len : end].copy()
        np.nan_to_num(seq, copy=False, nan=0.0)

        seq_5m = seq[4::5][-self.seq_5m :]
        if len(seq_5m) < self.seq_5m:
            pad = np.zeros((self.seq_5m - len(seq_5m), seq.shape[1]), dtype=np.float32)
            seq_5m = np.vstack([pad, seq_5m])

        x_1m = torch.from_numpy(seq)
        x_5m = torch.from_numpy(seq_5m)
        return x_1m, x_5m, lbl_15, lbl_5, lbl_30
