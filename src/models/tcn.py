"""TCN signal model (v3 — Hybrid: Regression + Binary Direction).

Architecture:
  - Dilated causal convolutions with residual connections
  - Dilations: 1, 2, 4, 8, 16, 32 (receptive field: ~126 bars)
  - Dual-stream input: 1m features + 5m features
  - Learned ticker embedding fused before output heads
  - Dual output heads:
      Head A (Regression): predicts raw 15m forward return
      Head B (Direction):  binary logit — sign(return) > epsilon → 1, else 0
  - Dropout raised to 0.3 to combat memorization.

Training: HuberLoss(Head A) + λ * BCEWithLogitsLoss(Head B).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.dataset import TRADING_THRESHOLD

CHECKPOINT_DIR = Path("models/tcn")
N_FEATURES_1M = 30
N_FEATURES_5M = 30
N_CHANNELS = 128
KERNEL_SIZE = 3
DILATIONS = [1, 2, 4, 8, 16, 32]
DROPOUT = 0.3          # raised from 0.1
N_TICKERS = 64


# ─── Temporal block ───────────────────────────────────────────────────────────

class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
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

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self._padding = padding

    def _causal_trim(self, x: torch.Tensor, length: int) -> torch.Tensor:
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
    """Dual-stream TCN with dual heads: regression + binary direction."""

    def __init__(
        self,
        n_features_1m: int = N_FEATURES_1M,
        n_features_5m: int = N_FEATURES_5M,
        n_channels: int = N_CHANNELS,
        kernel_size: int = KERNEL_SIZE,
        dilations: list[int] | None = None,
        dropout: float = DROPOUT,
        n_tickers: int = N_TICKERS,
    ) -> None:
        super().__init__()
        dilations = dilations or DILATIONS

        self.input_norm_1m = nn.LayerNorm(n_features_1m)
        self.input_norm_5m = nn.LayerNorm(n_features_5m)
        self.ticker_embed = nn.Embedding(n_tickers, n_channels)

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

        # ── Fusion ─────────────────────────────────────────────────────────────
        fused = n_channels * 3   # 1m + 5m + ticker
        self.fusion = nn.Sequential(
            nn.Linear(fused, n_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── Head A: Regression (15m return) ────────────────────────────────────
        self.return_head = nn.Linear(n_channels, 1)

        # ── Head B: Binary direction (forcing function) ────────────────────────
        self.direction_head = nn.Sequential(
            nn.Linear(n_channels, n_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_channels // 2, 1),
        )

        # Confidence
        self.confidence_head = nn.Sequential(
            nn.Linear(n_channels, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x_1m: torch.Tensor,                      # (B, n_features_1m, seq_len)
        x_5m: torch.Tensor,                      # (B, n_features_5m, seq_len_5m)
        ticker_ids: torch.Tensor | None = None,   # (B,) int tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            pred_return:  (B, 1) — Head A: predicted 15m return
            dir_logit:    (B, 1) — Head B: binary direction logit (pre-sigmoid)
            confidence:   (B, 1) — scalar in [0, 1]
        """
        x_1m = self.input_norm_1m(x_1m.permute(0, 2, 1)).permute(0, 2, 1)
        x_5m = self.input_norm_5m(x_5m.permute(0, 2, 1)).permute(0, 2, 1)

        h_1m = self.tcn_1m(x_1m)[:, :, -1]
        h_5m = self.tcn_5m(x_5m)[:, :, -1]

        if ticker_ids is not None:
            tick_emb = self.ticker_embed(ticker_ids)
        else:
            tick_emb = torch.zeros_like(h_1m)

        fused = torch.cat([h_1m, h_5m, tick_emb], dim=-1)
        h = self.fusion(fused)

        pred_return = self.return_head(h)        # Head A
        dir_logit   = self.direction_head(h)      # Head B
        confidence  = self.confidence_head(h)
        return pred_return, dir_logit, confidence

    def predict(
        self,
        x_1m: torch.Tensor,
        x_5m: torch.Tensor,
        ticker_id: int | None = None,
    ) -> tuple[float, float]:
        """Single-sample inference. Returns (direction, confidence).

        Uses binary head for direction, regression head for threshold gate.
        """
        self.eval()
        with torch.no_grad():
            tid = None
            if ticker_id is not None:
                tid = torch.tensor([ticker_id], device=x_1m.device)
            pred_return, dir_logit, conf = self.forward(
                x_1m.unsqueeze(0), x_5m.unsqueeze(0), ticker_ids=tid
            )
            pred_ret = pred_return.squeeze().item()
            dir_prob = torch.sigmoid(dir_logit).squeeze().item()
            confidence = float(conf.squeeze())

            if abs(pred_ret) < TRADING_THRESHOLD:
                return 0.0, confidence
            direction = 1.0 if dir_prob > 0.5 else -1.0
            return direction, confidence


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def save_checkpoint(model: TCNSignalModel, step: int, sharpe: float) -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f"step_{step:06d}_sharpe_{sharpe:.3f}.pt"
    torch.save({
        "step": step,
        "sharpe": sharpe,
        "model_state": model.state_dict(),
        "config": {
            "n_features_1m": model.input_norm_1m.normalized_shape[0],
            "n_features_5m": model.input_norm_5m.normalized_shape[0],
            "n_channels": model.return_head.in_features,
            "n_tickers": model.ticker_embed.num_embeddings,
        },
    }, path)
    return path


def load_best_checkpoint(checkpoint_dir: Path | None = None) -> TCNSignalModel | None:
    ckpt_dir = checkpoint_dir or CHECKPOINT_DIR
    checkpoints = sorted(ckpt_dir.glob("step_*_sharpe_*.pt"))
    if not checkpoints:
        return None
    best = max(checkpoints, key=lambda p: float(p.stem.split("sharpe_")[1]))
    state = torch.load(best, map_location="cpu")
    cfg = state["config"]
    model = TCNSignalModel(**cfg)
    model.load_state_dict(state["model_state"])
    return model
