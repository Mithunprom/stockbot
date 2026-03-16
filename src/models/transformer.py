"""Transformer signal model (v3 — Hybrid: Regression + Binary Direction).

Architecture:
  - Learned ticker embedding: added to temporal features before attention.
  - Main encoder: TransformerEncoder (8 heads, 5 layers) on FFSA top-N features.
  - Options cross-attention: key/value stream for options flow injection.
  - Dual output heads:
      Head A (Regression): predicts raw 15m forward return
      Head B (Direction):  binary logit — sign(return) > epsilon → 1, else 0
  - Dropout raised to 0.3 to combat ticker embedding memorization.

Training: HuberLoss(Head A) + λ * BCEWithLogitsLoss(Head B).
The binary head prevents prediction collapse to zero.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.dataset import TRADING_THRESHOLD

# ─── Constants ────────────────────────────────────────────────────────────────

SEQ_LEN = 60
N_FEATURE_COLS = 30
N_OPTIONS_COLS = 6
D_MODEL = 128
N_HEADS = 8
N_LAYERS = 5
D_FF = 512
DROPOUT = 0.3          # raised from 0.1 to combat memorization
N_TICKERS = 64

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
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ─── Model ────────────────────────────────────────────────────────────────────

class TransformerSignalModel(nn.Module):
    """Transformer with dual heads: regression + binary direction."""

    def __init__(
        self,
        n_features: int = N_FEATURE_COLS,
        n_options: int = N_OPTIONS_COLS,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        d_ff: int = D_FF,
        dropout: float = DROPOUT,
        n_tickers: int = N_TICKERS,
    ) -> None:
        super().__init__()

        self.ticker_embed = nn.Embedding(n_tickers, d_model)
        self.input_norm = nn.LayerNorm(n_features)
        self.feature_proj = nn.Linear(n_features, d_model)
        self.options_proj = nn.Linear(n_options, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(d_model)

        self.pool = nn.AdaptiveAvgPool1d(1)

        # ── Head A: Regression (15m return prediction) ─────────────────────────
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        # ── Head B: Binary direction (forcing function — prevents collapse) ────
        # Raw logit output — use BCEWithLogitsLoss (numerically stable)
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        features: torch.Tensor,                  # (B, seq_len, n_features)
        options: torch.Tensor | None = None,      # (B, seq_len, n_options) or None
        ticker_ids: torch.Tensor | None = None,   # (B,) int tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            pred_return:  (B, 1) — Head A: predicted 15m return
            dir_logit:    (B, 1) — Head B: binary direction logit (pre-sigmoid)
            confidence:   (B, 1) — scalar in [0, 1]
        """
        B, T, _ = features.shape

        x = self.pos_enc(self.feature_proj(self.input_norm(features)))

        if ticker_ids is not None:
            tick_emb = self.ticker_embed(ticker_ids)
            x = x + tick_emb.unsqueeze(1)

        x = self.encoder(x)

        if options is not None:
            o = self.pos_enc(self.options_proj(options))
            attn_out, _ = self.cross_attn(query=x, key=o, value=o)
            x = self.cross_attn_norm(x + attn_out)

        pooled = self.pool(x.transpose(1, 2)).squeeze(-1)  # (B, d_model)

        pred_return = self.regressor(pooled)        # Head A
        dir_logit   = self.direction_head(pooled)    # Head B
        confidence  = self.confidence_head(pooled)
        return pred_return, dir_logit, confidence

    def predict(
        self,
        features: torch.Tensor,
        options: torch.Tensor | None = None,
        ticker_id: int | None = None,
    ) -> tuple[float, float]:
        """Single-sample inference. Returns (direction, confidence).

        Uses the BINARY HEAD for direction (sigmoid > 0.5 → up, else down).
        Uses confidence head for signal strength.
        Applies trading threshold on regression head magnitude.
        """
        self.eval()
        with torch.no_grad():
            tid = None
            if ticker_id is not None:
                tid = torch.tensor([ticker_id], device=features.device)
            pred_return, dir_logit, conf = self.forward(
                features.unsqueeze(0), options, ticker_ids=tid
            )
            pred_ret = pred_return.squeeze().item()
            dir_prob = torch.sigmoid(dir_logit).squeeze().item()
            confidence = float(conf.squeeze())

            # Use regression magnitude for threshold gate,
            # binary head for direction commitment
            if abs(pred_ret) < TRADING_THRESHOLD:
                return 0.0, confidence
            direction = 1.0 if dir_prob > 0.5 else -1.0
            return direction, confidence


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
                "n_features": model.feature_proj.in_features,
                "n_options": model.options_proj.in_features,
                "d_model": model.feature_proj.out_features,
                "n_heads": N_HEADS,
                "n_layers": N_LAYERS,
                "d_ff": D_FF,
                "n_tickers": model.ticker_embed.num_embeddings,
            },
        },
        path,
    )
    return path


def load_best_checkpoint(checkpoint_dir: Path | None = None) -> TransformerSignalModel | None:
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
