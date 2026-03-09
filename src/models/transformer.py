"""Transformer + Options Flow signal model.

Architecture:
  - Main encoder: TransformerEncoder (4 heads, 3 layers) on FFSA top-30 price/volume features.
  - Options cross-attention: separate encoder processes options flow features as
    key/value stream, injected into the main encoder via cross-attention.
  - Output: 3-class softmax (up/flat/down) + confidence logit.

Training: 2 years of 1m bars, focal loss for class imbalance, walk-forward splits.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Constants ────────────────────────────────────────────────────────────────

SEQ_LEN = 60          # 60 bars of history (1-hour at 1m resolution)
N_FEATURE_COLS = 30   # FFSA top-30
N_OPTIONS_COLS = 6    # iv_rank, put_call_ratio, unusual_flow_flag, net_gex, smart_money_score, vol_oi
N_CLASSES = 3         # 0=down, 1=flat, 2=up
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 3
D_FF = 256
DROPOUT = 0.1

CHECKPOINT_DIR = Path("models/transformer")


# ─── Positional encoding ──────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ─── Model ────────────────────────────────────────────────────────────────────

class TransformerSignalModel(nn.Module):
    """Transformer encoder with cross-attention for options flow injection."""

    def __init__(
        self,
        n_features: int = N_FEATURE_COLS,
        n_options: int = N_OPTIONS_COLS,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        d_ff: int = D_FF,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()

        self.input_norm = nn.LayerNorm(n_features)
        self.feature_proj = nn.Linear(n_features, d_model)
        self.options_proj = nn.Linear(n_options, d_model)

        self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout=dropout)

        # Self-attention encoder for price/volume features
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Cross-attention: price features (query) attend to options flow (key/value)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(d_model)

        # Classification head
        self.pool = nn.AdaptiveAvgPool1d(1)   # mean over time dimension
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, N_CLASSES),
        )
        # Separate confidence head (single logit → sigmoid)
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        features: torch.Tensor,      # (B, seq_len, n_features)
        options: torch.Tensor | None = None,  # (B, seq_len, n_options) or None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: (B, 3) raw class logits
            confidence: (B, 1) scalar in [0, 1]
        """
        B, T, _ = features.shape

        x = self.pos_enc(self.feature_proj(self.input_norm(features)))   # (B, T, d_model)
        x = self.encoder(x)                              # (B, T, d_model)

        # Cross-attention with options flow
        if options is not None:
            o = self.pos_enc(self.options_proj(options))  # (B, T, d_model)
            attn_out, _ = self.cross_attn(query=x, key=o, value=o)
            x = self.cross_attn_norm(x + attn_out)

        # Pool over time → (B, d_model)
        pooled = self.pool(x.transpose(1, 2)).squeeze(-1)

        logits = self.classifier(pooled)         # (B, 3)
        confidence = self.confidence_head(pooled)  # (B, 1)
        return logits, confidence

    def predict(
        self,
        features: torch.Tensor,
        options: torch.Tensor | None = None,
    ) -> tuple[float, float]:
        """Single-sample inference. Returns (direction, confidence).

        direction: +1 (up), 0 (flat), -1 (down)
        confidence: float in [0, 1]
        """
        self.eval()
        with torch.no_grad():
            logits, conf = self.forward(features.unsqueeze(0), options)
            probs = F.softmax(logits, dim=-1).squeeze(0)
            class_idx = probs.argmax().item()
            direction_map = {0: -1.0, 1: 0.0, 2: 1.0}
            return direction_map[class_idx], float(conf.squeeze())


# ─── Focal loss ───────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal loss for class imbalance (down/flat/up often skewed)."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def save_checkpoint(model: TransformerSignalModel, step: int, sharpe: float) -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f"step_{step:06d}_sharpe_{sharpe:.3f}.pt"
    torch.save(
        {
            "step": step,
            "sharpe": sharpe,
            "model_state": model.state_dict(),
            "config": {
                "n_features": N_FEATURE_COLS,
                "n_options": N_OPTIONS_COLS,
                "d_model": D_MODEL,
                "n_heads": N_HEADS,
                "n_layers": N_LAYERS,
            },
        },
        path,
    )
    return path


def load_best_checkpoint(checkpoint_dir: Path | None = None) -> TransformerSignalModel | None:
    """Load the checkpoint with the highest Sharpe ratio from the directory."""
    ckpt_dir = checkpoint_dir or CHECKPOINT_DIR
    checkpoints = sorted(ckpt_dir.glob("step_*_sharpe_*.pt"))
    if not checkpoints:
        return None

    best = max(checkpoints, key=lambda p: float(p.stem.split("sharpe_")[1]))
    state = torch.load(best, map_location="cpu")
    cfg = state["config"]
    model = TransformerSignalModel(**cfg)
    model.load_state_dict(state["model_state"])
    return model
