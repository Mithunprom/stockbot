"""Ensemble layer: combines Transformer, TCN, and FinBERT signals.

ensemble_signal = (
    0.45 * transformer_confidence * transformer_direction +
    0.35 * tcn_confidence * tcn_direction +
    0.20 * sentiment_index
)

Initial weights as above. Profit Sub-Agent re-optimizes quarterly.
Direction encoding: long=+1, neutral=0, short=-1.

All signal computation is logged with model confidence for explainability.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
import torch

from src.models.sentiment import SentimentScorer
from src.models.tcn import TCNSignalModel, load_best_checkpoint as load_tcn
from src.models.transformer import TransformerSignalModel
from src.models.transformer import load_best_checkpoint as load_transformer

logger = structlog.get_logger(__name__)


# ─── Signal dataclass ─────────────────────────────────────────────────────────

@dataclass
class EnsembleSignal:
    """Fully attributed signal for one ticker at one bar."""

    ticker: str
    timestamp: datetime

    # Individual model outputs
    transformer_direction: float    # +1 / 0 / -1
    transformer_confidence: float   # [0, 1]
    tcn_direction: float
    tcn_confidence: float
    sentiment_index: float          # rolling weighted SI

    # Ensemble
    ensemble_signal: float          # weighted combination

    # Ensemble weights used (may be updated by Profit Agent)
    w_transformer: float = 0.45
    w_tcn: float = 0.35
    w_sentiment: float = 0.20

    # Derived
    strength: str = field(init=False)   # "strong" / "moderate" / "weak" / "flat"

    def __post_init__(self) -> None:
        abs_sig = abs(self.ensemble_signal)
        if abs_sig >= 0.60:
            self.strength = "strong"
        elif abs_sig >= 0.40:
            self.strength = "moderate"
        elif abs_sig >= 0.20:
            self.strength = "weak"
        else:
            self.strength = "flat"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp.isoformat(),
            "transformer_direction": self.transformer_direction,
            "transformer_confidence": self.transformer_confidence,
            "tcn_direction": self.tcn_direction,
            "tcn_confidence": self.tcn_confidence,
            "sentiment_index": self.sentiment_index,
            "ensemble_signal": round(self.ensemble_signal, 4),
            "strength": self.strength,
            "weights": {
                "transformer": self.w_transformer,
                "tcn": self.w_tcn,
                "sentiment": self.w_sentiment,
            },
        }

    def plain_english(self) -> str:
        """Human-readable signal summary for logs and dashboard."""
        direction = "bullish" if self.ensemble_signal > 0 else ("bearish" if self.ensemble_signal < 0 else "neutral")
        return (
            f"{self.ticker}: {self.strength.upper()} {direction} signal "
            f"(ensemble={self.ensemble_signal:+.3f}) — "
            f"Transformer [{self.transformer_confidence:.0%} conf, dir={self.transformer_direction:+.0f}], "
            f"TCN [{self.tcn_confidence:.0%} conf, dir={self.tcn_direction:+.0f}], "
            f"Sentiment SI={self.sentiment_index:+.3f}"
        )


# ─── Ensemble weights (updatable by Profit Agent) ────────────────────────────

@dataclass
class EnsembleWeights:
    transformer: float = 0.45
    tcn: float = 0.35
    sentiment: float = 0.20

    def validate(self) -> None:
        total = self.transformer + self.tcn + self.sentiment
        if abs(total - 1.0) > 1e-3:
            raise ValueError(f"Ensemble weights must sum to 1.0, got {total:.3f}")

    @classmethod
    def from_staging(cls, staging_path: Path) -> "EnsembleWeights":
        """Load weights proposed by Profit Agent from staging file."""
        import json

        with open(staging_path) as f:
            data = json.load(f)
        weights = data.get("ensemble_weights", {})
        obj = cls(
            transformer=weights.get("transformer", 0.45),
            tcn=weights.get("tcn", 0.35),
            sentiment=weights.get("sentiment", 0.20),
        )
        obj.validate()
        return obj


# ─── Ensemble engine ─────────────────────────────────────────────────────────

class EnsembleEngine:
    """Loads all three models and produces ensemble signals.

    Thread-safe: model inference is done in thread pools.
    """

    def __init__(
        self,
        weights: EnsembleWeights | None = None,
        device: str | None = None,
    ) -> None:
        self.weights = weights or EnsembleWeights()
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._transformer: TransformerSignalModel | None = None
        self._tcn: TCNSignalModel | None = None
        self._sentiment: SentimentScorer = SentimentScorer()

    async def load(self) -> None:
        """Load all models concurrently."""
        loop = asyncio.get_event_loop()
        t_load = loop.run_in_executor(None, self._load_transformer)
        tcn_load = loop.run_in_executor(None, self._load_tcn)
        await asyncio.gather(t_load, tcn_load, self._sentiment.load())

    def _load_transformer(self) -> None:
        self._transformer = load_transformer()
        if self._transformer:
            self._transformer.eval()
            logger.info("transformer_loaded")
        else:
            logger.warning("transformer_checkpoint_not_found")

    def _load_tcn(self) -> None:
        self._tcn = load_tcn()
        if self._tcn:
            self._tcn.eval()
            logger.info("tcn_loaded")
        else:
            logger.warning("tcn_checkpoint_not_found")

    def update_weights(self, new_weights: EnsembleWeights) -> None:
        """Hot-swap ensemble weights (called by Profit Agent)."""
        new_weights.validate()
        self.weights = new_weights
        logger.info(
            "ensemble_weights_updated",
            transformer=new_weights.transformer,
            tcn=new_weights.tcn,
            sentiment=new_weights.sentiment,
        )

    async def compute_signal(
        self,
        ticker: str,
        features_1m: torch.Tensor,        # (seq_len, n_features)
        features_5m: torch.Tensor,        # (seq_len, n_features)
        options_flow: torch.Tensor | None = None,  # (seq_len, n_options)
    ) -> EnsembleSignal:
        """Compute the full ensemble signal for one ticker.

        Args:
            ticker: Ticker symbol.
            features_1m: Feature tensor at 1m resolution.
            features_5m: Feature tensor at 5m resolution.
            options_flow: Options flow feature tensor (optional).

        Returns:
            EnsembleSignal with full attribution.
        """
        loop = asyncio.get_event_loop()

        # Transformer inference
        if self._transformer is not None:
            t_dir, t_conf = await loop.run_in_executor(
                None, self._transformer.predict, features_1m, options_flow
            )
        else:
            t_dir, t_conf = 0.0, 0.0

        # TCN inference
        if self._tcn is not None:
            tcn_dir, tcn_conf = await loop.run_in_executor(
                None, self._tcn.predict,
                features_1m.T.unsqueeze(0).squeeze(0),   # (n_features, seq_len)
                features_5m.T.unsqueeze(0).squeeze(0),
            )
        else:
            tcn_dir, tcn_conf = 0.0, 0.0

        # Sentiment rolling index
        si = await self._sentiment.rolling_sentiment_index(ticker, lookback_hours=24)

        # Weighted ensemble
        w = self.weights
        ensemble = (
            w.transformer * t_conf * t_dir
            + w.tcn * tcn_conf * tcn_dir
            + w.sentiment * si
        )
        # Clip to [-1, 1]
        ensemble = max(-1.0, min(1.0, ensemble))

        signal = EnsembleSignal(
            ticker=ticker,
            timestamp=datetime.now(timezone.utc),
            transformer_direction=t_dir,
            transformer_confidence=t_conf,
            tcn_direction=tcn_dir,
            tcn_confidence=tcn_conf,
            sentiment_index=si,
            ensemble_signal=ensemble,
            w_transformer=w.transformer,
            w_tcn=w.tcn,
            w_sentiment=w.sentiment,
        )
        logger.info("signal_computed", **signal.to_dict())
        return signal

    async def compute_universe(
        self,
        universe_features: dict[str, dict[str, torch.Tensor]],
    ) -> list[EnsembleSignal]:
        """Compute signals for the full trading universe.

        Args:
            universe_features: {ticker: {"1m": Tensor, "5m": Tensor, "options": Tensor|None}}

        Returns:
            List of EnsembleSignal sorted by |ensemble_signal| desc.
        """
        tasks = [
            self.compute_signal(
                ticker=ticker,
                features_1m=data["1m"],
                features_5m=data["5m"],
                options_flow=data.get("options"),
            )
            for ticker, data in universe_features.items()
        ]
        signals = await asyncio.gather(*tasks)
        return sorted(signals, key=lambda s: abs(s.ensemble_signal), reverse=True)
