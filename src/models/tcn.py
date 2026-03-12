"""Temporal Convolutional Network (TCN) signal model.

Architecture:
  - Dilated causal convolutions with residual connections
  - Dilations: 1, 2, 4, 8, 16 (receptive field: ~62 bars)
  - Dual-stream input: 1m features + 5m features
  - Output: next-bar return regression + direction probability

Faster inference than Transformer — complements it on short-term momentum.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

CHECKPOINT_DIR = Path("models/tcn")
N_FEATURES_1M = 30   # FFSA top-30 at 1m
N_FEATURES_5M = 30   # Same features at 5m resolution
N_CHANNELS = 128      # was 64 — doubles feature width
KERNEL_SIZE = 3
DILATIONS = [1, 2, 4, 8, 16, 32]   # was [1,2,4,8,16] — receptive field: ~126 bars (was ~62)
DROPOUT = 0.1


# ─── Temporal block ───────────────────────────────────────────────────────────

class TemporalBlock(nn.Module):
    """Causal dilated conv block with residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation   # causal: only pad left

        self.conv1 = nn.utils.parametrize.register_parametrization if False else nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation,
        )
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.GELU()

        # Residual projection if channel dims differ
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self._padding = padding

    def _causal_trim(self, x: torch.Tensor, length: int) -> torch.Tensor:
        """Remove right-padding to preserve causality."""
        return x[:, :, :length]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(2)
        residual = self.downsample(x)

        out = self._causal_trim(self.conv1(x), T)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self._causal_trim(self.conv2(out), T)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.dropout(out)

        return self.relu(out + residual)


# ─── TCN model ────────────────────────────────────────────────────────────────

class TCNSignalModel(nn.Module):
    """Dual-stream TCN: 1m and 5m features fused before output heads."""

    def __init__(
        self,
        n_features_1m: int = N_FEATURES_1M,
        n_features_5m: int = N_FEATURES_5M,
        n_channels: int = N_CHANNELS,
        kernel_size: int = KERNEL_SIZE,
        dilations: list[int] | None = None,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        dilations = dilations or DILATIONS

        # Input normalization — LayerNorm over feature dim before 1x1 conv
        # prevents raw financial feature scales (OBV ~10^7, returns ~0.001) from
        # blowing up the first TemporalBlock and causing NaN losses.
        self.input_norm_1m = nn.LayerNorm(n_features_1m)
        self.input_norm_5m = nn.LayerNorm(n_features_5m)

        # ── 1m stream ─────────────────────────────────────────────────────────
        layers_1m: list[nn.Module] = [
            TemporalBlock(n_features_1m, n_channels, kernel_size, dilation=1, dropout=dropout)
        ]
        for d in dilations[1:]:
            layers_1m.append(
                TemporalBlock(n_channels, n_channels, kernel_size, dilation=d, dropout=dropout)
            )
        self.tcn_1m = nn.Sequential(*layers_1m)

        # ── 5m stream ─────────────────────────────────────────────────────────
        layers_5m: list[nn.Module] = [
            TemporalBlock(n_features_5m, n_channels, kernel_size, dilation=1, dropout=dropout)
        ]
        for d in dilations[1:]:
            layers_5m.append(
                TemporalBlock(n_channels, n_channels, kernel_size, dilation=d, dropout=dropout)
            )
        self.tcn_5m = nn.Sequential(*layers_5m)

        # ── Fusion & heads ─────────────────────────────────────────────────────
        fused = n_channels * 2
        self.fusion = nn.Sequential(
            nn.Linear(fused, n_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.return_head = nn.Linear(n_channels, 1)       # next-bar return

        # Multi-task direction heads: PRIMARY 15m + AUX 5m, 30m
        self.direction_head    = nn.Linear(n_channels, 3)  # PRIMARY: 15m
        self.direction_head_5m  = nn.Linear(n_channels, 3) # AUX: 5m
        self.direction_head_30m = nn.Linear(n_channels, 3) # AUX: 30m

        self.confidence_head = nn.Sequential(
            nn.Linear(n_channels, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x_1m: torch.Tensor,   # (B, n_features_1m, seq_len)
        x_5m: torch.Tensor,   # (B, n_features_5m, seq_len_5m)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            return_pred:    (B, 1) scalar return prediction
            dir_logits_15m: (B, 3) PRIMARY — 15m direction logits
            dir_logits_5m:  (B, 3) AUX — 5m direction logits
            dir_logits_30m: (B, 3) AUX — 30m direction logits
            confidence:     (B, 1) scalar in [0, 1]
        """
        x_1m = self.input_norm_1m(x_1m.permute(0, 2, 1)).permute(0, 2, 1)
        x_5m = self.input_norm_5m(x_5m.permute(0, 2, 1)).permute(0, 2, 1)

        h_1m = self.tcn_1m(x_1m)[:, :, -1]
        h_5m = self.tcn_5m(x_5m)[:, :, -1]

        fused = torch.cat([h_1m, h_5m], dim=-1)
        h = self.fusion(fused)

        return_pred     = self.return_head(h)
        dir_logits_15m  = self.direction_head(h)
        dir_logits_5m   = self.direction_head_5m(h)
        dir_logits_30m  = self.direction_head_30m(h)
        confidence      = self.confidence_head(h)
        return return_pred, dir_logits_15m, dir_logits_5m, dir_logits_30m, confidence

    def predict(
        self,
        x_1m: torch.Tensor,
        x_5m: torch.Tensor,
    ) -> tuple[float, float]:
        """Single-sample inference using PRIMARY 15m head. Returns (direction, confidence)."""
        self.eval()
        with torch.no_grad():
            _, dir_logits_15m, _, _, conf = self.forward(
                x_1m.unsqueeze(0), x_5m.unsqueeze(0)
            )
            probs = F.softmax(dir_logits_15m, dim=-1).squeeze(0)
            class_idx = probs.argmax().item()
            direction_map = {0: -1.0, 1: 0.0, 2: 1.0}
            return direction_map[class_idx], float(conf.squeeze())


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def save_checkpoint(model: TCNSignalModel, step: int, sharpe: float) -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f"step_{step:06d}_sharpe_{sharpe:.3f}.pt"
    torch.save({"step": step, "sharpe": sharpe, "model_state": model.state_dict()}, path)
    return path


def load_best_checkpoint(checkpoint_dir: Path | None = None) -> TCNSignalModel | None:
    ckpt_dir = checkpoint_dir or CHECKPOINT_DIR
    checkpoints = sorted(ckpt_dir.glob("step_*_sharpe_*.pt"))
    if not checkpoints:
        return None
    best = max(checkpoints, key=lambda p: float(p.stem.split("sharpe_")[1]))
    state = torch.load(best, map_location="cpu")
    model = TCNSignalModel()
    model.load_state_dict(state["model_state"])
    return model
