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

# torch and ML models are optional — server runs without them if not installed
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

try:
    from src.models.sentiment import SentimentScorer
    from src.models.tcn import TCNSignalModel, load_best_checkpoint as load_tcn
    from src.models.transformer import TransformerSignalModel
    from src.models.transformer import load_best_checkpoint as load_transformer
    _MODELS_AVAILABLE = True
except (ImportError, Exception):
    _MODELS_AVAILABLE = False

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
        self._device = device or ("cuda" if (_TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu")
        self._transformer: Any | None = None
        self._tcn: Any | None = None
        self._sentiment: Any | None = SentimentScorer() if _MODELS_AVAILABLE else None

    async def load(self) -> None:
        """Load all models concurrently. No-op if torch is not installed."""
        if not _MODELS_AVAILABLE:
            logger.warning("ml_models_unavailable: torch not installed — running in signal-free mode")
            return
        loop = asyncio.get_event_loop()
        t_load = loop.run_in_executor(None, self._load_transformer)
        tcn_load = loop.run_in_executor(None, self._load_tcn)
        sentiment_load = self._sentiment.load() if self._sentiment else asyncio.sleep(0)
        await asyncio.gather(t_load, tcn_load, sentiment_load)

    def _load_transformer(self) -> None:
        if not _MODELS_AVAILABLE:
            return
        self._transformer = load_transformer()
        if self._transformer:
            self._transformer.eval()
            logger.info("transformer_loaded")
        else:
            logger.warning("transformer_checkpoint_not_found")

    def _load_tcn(self) -> None:
        if not _MODELS_AVAILABLE:
            return
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

    # ── Rule-based signal fallback ────────────────────────────────────────────

    @staticmethod
    def compute_signal_rule_based(
        ticker: str,
        closes: "list[float] | Any",
    ) -> "EnsembleSignal":
        """Compute a rule-based signal from a sequence of close prices.

        Uses MACD crossover + RSI(14) to derive a directional signal when
        ML models are unavailable or feature_cols is empty.

        Signal rules:
          RSI < 35 AND MACD line just crossed above signal line  → +0.50 (buy)
          RSI > 65 AND MACD line just crossed below signal line  → -0.50 (sell)
          Otherwise                                              →  0.00 (hold)

        Args:
            ticker: Ticker symbol.
            closes: Sequence of close prices (at least 40 values recommended).

        Returns:
            EnsembleSignal with transformer/tcn confidences=0 and rule-based
            ensemble_signal. Transformer/TCN weights are 0; full weight on
            sentiment_index field (used to carry the rule signal).
        """
        import numpy as np

        closes_arr = list(closes)
        if len(closes_arr) < 27:
            # Not enough data — flat signal
            return EnsembleSignal(
                ticker=ticker,
                timestamp=datetime.now(timezone.utc),
                transformer_direction=0.0,
                transformer_confidence=0.0,
                tcn_direction=0.0,
                tcn_confidence=0.0,
                sentiment_index=0.0,
                ensemble_signal=0.0,
                w_transformer=0.0,
                w_tcn=0.0,
                w_sentiment=1.0,
            )

        import pandas as pd

        s = pd.Series(closes_arr, dtype=float)

        # RSI(14)
        delta = s.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
        rs = gain / loss.replace(0, float("nan"))
        rsi = (100 - (100 / (1 + rs))).iloc[-1]

        # MACD (12, 26, 9)
        ema12 = s.ewm(span=12, adjust=False).mean()
        ema26 = s.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()

        # Crossover: current bar vs previous bar
        macd_now = macd_line.iloc[-1]
        macd_prev = macd_line.iloc[-2] if len(macd_line) >= 2 else macd_now
        sig_now = signal_line.iloc[-1]
        sig_prev = signal_line.iloc[-2] if len(signal_line) >= 2 else sig_now

        bullish_cross = (macd_now > sig_now) and (macd_prev <= sig_prev)
        bearish_cross = (macd_now < sig_now) and (macd_prev >= sig_prev)

        if rsi < 35 and bullish_cross:
            rule_signal = 0.50
        elif rsi > 65 and bearish_cross:
            rule_signal = -0.50
        else:
            rule_signal = 0.0

        logger.info(
            "rule_based_signal",
            ticker=ticker,
            rsi=round(float(rsi), 2) if not (rsi != rsi) else None,
            bullish_cross=bullish_cross,
            bearish_cross=bearish_cross,
            rule_signal=rule_signal,
        )

        return EnsembleSignal(
            ticker=ticker,
            timestamp=datetime.now(timezone.utc),
            transformer_direction=0.0,
            transformer_confidence=0.0,
            tcn_direction=0.0,
            tcn_confidence=0.0,
            sentiment_index=rule_signal,    # carries the rule signal
            ensemble_signal=rule_signal,
            w_transformer=0.0,
            w_tcn=0.0,
            w_sentiment=1.0,
        )
