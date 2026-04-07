"""Pipeline B — Rules-based signal engine.

Combines five signal dimensions into an EnsembleSignal-compatible output:
  1. Technical score (30%) — Rule-based scoring from existing indicators
  2. Fundamental score (25%) — P/E, earnings surprise, revenue growth
  3. Regime score (20%) — VIX + SPY/QQQ momentum (macro bias)
  4. Sentiment (20%) — Reuses existing FinBERT pipeline
  5. Social (5%) — Tweet stub (placeholder until Twitter API budget)

Produces the same EnsembleSignal dataclass as Pipeline A, so the entire
execution stack (SmartPositionSizer, PositionManager, _act_on_signal)
works unchanged. The lgbm_dir_prob/lgbm_pred_return fields carry
Pipeline B's composite scores for the sizer.
"""

from __future__ import annotations

import asyncio
import structlog
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

from src.models.ensemble import EnsembleSignal

logger = structlog.get_logger(__name__)


# ─── Pipeline B weights ─────────────────────────────────────────────────────

@dataclass
class PipelineBWeights:
    """Weight allocation across Pipeline B's signal dimensions."""

    regime: float = 0.20
    fundamentals: float = 0.25
    technicals: float = 0.30
    sentiment: float = 0.20
    social: float = 0.05

    def validate(self) -> None:
        total = self.regime + self.fundamentals + self.technicals + self.sentiment + self.social
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Pipeline B weights must sum to 1.0, got {total:.3f}")


# ─── Technical scoring (rule-based) ─────────────────────────────────────────

def _score_technicals(row: dict[str, float]) -> float:
    """Score a ticker from its latest indicator values.

    Uses RSI, MACD, Bollinger %B, VWAP deviation, ADX, MFI, Stochastic,
    OBV, candlestick patterns, divergence signals, and multi-timeframe
    confluence — all from the existing feature_matrix.

    Returns a score in [-1, +1].
    """
    score = 0.0

    # ── RSI mean reversion ───────────────────────────────────────────────────
    rsi = row.get("rsi_14", 50.0)
    if rsi < 30:
        score += 0.40
    elif rsi < 40:
        score += 0.15
    elif rsi > 70:
        score -= 0.40
    elif rsi > 60:
        score -= 0.15

    # ── MACD momentum ────────────────────────────────────────────────────────
    macd_hist = row.get("macd_hist", 0.0)
    macd = row.get("macd", 0.0)
    macd_signal = row.get("macd_signal", 0.0)
    # Bullish: MACD crosses above signal (hist turns positive)
    if macd_hist > 0 and macd > macd_signal:
        score += 0.25
    elif macd_hist < 0 and macd < macd_signal:
        score -= 0.25

    # ── Bollinger Band position ──────────────────────────────────────────────
    bb_pct = row.get("bb_pct", 0.5)
    if bb_pct < 0.10:
        score += 0.20       # near lower band — oversold
    elif bb_pct > 0.90:
        score -= 0.20       # near upper band — overbought

    # ── VWAP deviation (intraday mean reversion) ─────────────────────────────
    vwap_dev = row.get("vwap_dev", 0.0)
    if vwap_dev < -0.015:
        score += 0.20       # significantly below VWAP
    elif vwap_dev > 0.015:
        score -= 0.20       # significantly above VWAP

    # ── ADX trend strength filter ────────────────────────────────────────────
    # Low ADX = choppy market — reduce signal magnitude
    # Damping softened so Pipeline B signals can reach entry-gate threshold
    adx = row.get("adx", 20.0)
    if adx < 15:
        score *= 0.60       # very weak trend — moderate damping
    elif adx < 20:
        score *= 0.85       # weak trend — light damping

    # ── MFI (volume-weighted RSI) ────────────────────────────────────────────
    mfi = row.get("mfi_14", 50.0)
    if mfi < 20:
        score += 0.15       # extreme selling pressure exhaustion
    elif mfi > 80:
        score -= 0.15       # extreme buying pressure exhaustion

    # ── Stochastic K ─────────────────────────────────────────────────────────
    stoch_k = row.get("stoch_k", 50.0)
    if stoch_k < 20:
        score += 0.15
    elif stoch_k > 80:
        score -= 0.15

    # ── OBV trend (volume confirmation) ──────────────────────────────────────
    obv_pct = row.get("obv_pct", 0.0)
    if obv_pct > 0.02:
        score += 0.10       # rising volume trend
    elif obv_pct < -0.02:
        score -= 0.10       # falling volume trend

    # ── Multi-timeframe confluence ───────────────────────────────────────────
    mtf_confluence = row.get("mtf_confluence", 0.0)
    mtf_aligned = row.get("mtf_aligned", 0.0)
    if mtf_aligned == 1.0:
        # All timeframes agree — strong trend confirmation
        if mtf_confluence > 0:
            score += 0.20
        elif mtf_confluence < 0:
            score -= 0.20
    elif abs(mtf_confluence) >= 2:
        # 2 of 3 timeframes agree
        score += 0.10 * np.sign(mtf_confluence)

    # ── Divergence signals ───────────────────────────────────────────────────
    div_strength = row.get("div_strength", 0.0)
    if abs(div_strength) >= 1.0:
        # Divergence detected — contrarian signal
        score += 0.15 * div_strength / 2.0  # normalize [-2,2] → [-0.15,+0.15]

    # ── Candlestick patterns ─────────────────────────────────────────────────
    candle_engulfing = row.get("candle_engulfing", 0.0)
    candle_hammer = row.get("candle_hammer", 0.0)
    candle_morning_eve = row.get("candle_morning_eve_star", 0.0)
    candle_score = (candle_engulfing * 0.10 + candle_hammer * 0.05 + candle_morning_eve * 0.10)
    score += candle_score

    # ── Supply & demand zones ────────────────────────────────────────────────
    in_demand = row.get("sd_in_demand", 0.0)
    in_supply = row.get("sd_in_supply", 0.0)
    if in_demand == 1.0:
        score += 0.15       # at demand zone — support bounce
    elif in_supply == 1.0:
        score -= 0.15       # at supply zone — resistance rejection

    # ── Parabolic SAR ────────────────────────────────────────────────────────
    psar = row.get("psar_signal", 0.0)
    score += psar * 0.05

    # ── Donchian breakout ────────────────────────────────────────────────────
    donchian_breakout = row.get("donchian_breakout", 0.0)
    score += donchian_breakout * 0.10

    return max(-1.0, min(1.0, score))


# ─── Fundamental scoring ────────────────────────────────────────────────────

def _score_fundamentals(data: Any) -> float:
    """Score a ticker based on its fundamental data.

    Args:
        data: FundamentalData instance from src/data/fundamentals.py

    Returns score in [-1, +1]: positive = undervalued/growth, negative = overvalued.
    """
    if data is None:
        return 0.0

    score = 0.0

    # P/E value scoring (lower trailing P/E = more value)
    pe = data.pe_ratio
    if pe is not None and pe > 0:
        if pe < 12:
            score += 0.30       # deep value
        elif pe < 18:
            score += 0.15       # fair value
        elif pe < 25:
            score += 0.0        # neutral
        elif pe < 40:
            score -= 0.10       # somewhat expensive
        else:
            score -= 0.20       # very expensive

    # Forward P/E vs trailing P/E (growth expectation)
    fwd_pe = data.forward_pe
    if pe is not None and fwd_pe is not None and pe > 0 and fwd_pe > 0:
        pe_compression = (pe - fwd_pe) / pe  # positive = earnings expected to grow
        if pe_compression > 0.20:
            score += 0.15       # strong growth expected
        elif pe_compression > 0.05:
            score += 0.05
        elif pe_compression < -0.10:
            score -= 0.10       # earnings expected to shrink

    # Earnings surprise
    surprise = data.earnings_surprise_pct
    if surprise is not None:
        if surprise > 10:
            score += 0.30       # massive beat
        elif surprise > 5:
            score += 0.20       # solid beat
        elif surprise > 0:
            score += 0.10       # small beat
        elif surprise < -10:
            score -= 0.30       # massive miss
        elif surprise < -5:
            score -= 0.20       # solid miss
        elif surprise < 0:
            score -= 0.10       # small miss

    # Revenue growth (YoY)
    rev_growth = data.revenue_growth_pct
    if rev_growth is not None:
        if rev_growth > 25:
            score += 0.20       # hypergrowth
        elif rev_growth > 10:
            score += 0.10       # solid growth
        elif rev_growth > 0:
            score += 0.05       # growing
        elif rev_growth < -10:
            score -= 0.20       # shrinking fast
        elif rev_growth < 0:
            score -= 0.10       # shrinking

    return max(-1.0, min(1.0, score))


# ─── Pipeline B Engine ───────────────────────────────────────────────────────

class PipelineBEngine:
    """Rules-based signal engine for Pipeline B.

    Combines technicals, fundamentals, regime, sentiment, and social
    scores into an EnsembleSignal-compatible output.

    Usage:
        engine = PipelineBEngine(fundamentals_cache, regime_monitor, social_feed)
        await engine.load()
        signal = await engine.compute_signal("AAPL", feature_row)
    """

    def __init__(
        self,
        fundamentals_cache: Any,       # FundamentalsCache
        market_regime: Any,            # MarketRegimeMonitor
        social_feed: Any,              # SocialFeedStub
        sentiment_scorer: Any = None,  # SentimentScorer (reuse from Pipeline A)
        weights: PipelineBWeights | None = None,
    ) -> None:
        self._fundamentals = fundamentals_cache
        self._regime = market_regime
        self._social = social_feed
        self._sentiment = sentiment_scorer
        self._weights = weights or PipelineBWeights()
        self._weights.validate()
        self._loaded = False

    async def load(self) -> None:
        """Initialize data sources. Called once at startup."""
        logger.info(
            "pipeline_b_loading",
            weights={
                "regime": self._weights.regime,
                "fundamentals": self._weights.fundamentals,
                "technicals": self._weights.technicals,
                "sentiment": self._weights.sentiment,
                "social": self._weights.social,
            },
        )
        self._loaded = True
        logger.info("pipeline_b_loaded")

    async def compute_signal(
        self,
        ticker: str,
        feature_row: dict[str, float],
    ) -> EnsembleSignal:
        """Compute a rules-based signal for a single ticker.

        Args:
            ticker: Stock symbol.
            feature_row: Dict of indicator values from the latest feature_matrix row.

        Returns:
            EnsembleSignal with Pipeline B's composite scores mapped into
            the standard fields for downstream execution compatibility.
        """
        w = self._weights

        # 1. Technical score
        tech_score = _score_technicals(feature_row)

        # 2. Fundamental score
        from src.data.fundamentals import get_fundamentals
        fund_data = get_fundamentals(ticker)
        fund_score = _score_fundamentals(fund_data)

        # 3. Regime score
        from src.data.market_regime import get_market_regime
        regime_snapshot = get_market_regime()
        regime_score = regime_snapshot.regime_score

        # 4. Sentiment score (reuse FinBERT)
        sentiment_score = 0.0
        if self._sentiment is not None:
            try:
                sentiment_score = await self._sentiment.rolling_sentiment_index(ticker)
            except Exception:
                pass

        # 5. Social score (stub → 0.0)
        social_score = await self._social.get_social_score(ticker)

        # ── Weighted ensemble ────────────────────────────────────────────────
        ensemble = (
            w.technicals * tech_score
            + w.fundamentals * fund_score
            + w.regime * regime_score
            + w.sentiment * sentiment_score
            + w.social * social_score
        )
        ensemble = max(-1.0, min(1.0, ensemble))

        # ── Map to EnsembleSignal fields ─────────────────────────────────────
        # The SmartPositionSizer reads lgbm_dir_prob and lgbm_pred_return
        # to determine conviction and expected return. We map Pipeline B's
        # composite scores into these fields for execution compatibility.
        #
        # lgbm_dir_prob: map ensemble [-1,+1] → [0.2, 0.8]
        #   This keeps values within the sizer's conviction buckets:
        #   [0.55, 0.65) → 2% cap, [0.65, 0.80) → 4% cap, [0.80+] → 6% cap
        dir_prob = 0.5 + ensemble * 0.30
        dir_prob = max(0.05, min(0.95, dir_prob))

        # lgbm_pred_return: synthetic expected return proportional to signal
        # The entry gate requires abs(pred_return) > 0.003 (SIZING_COST_THRESHOLD).
        # Scale so a moderate signal (±0.125) maps to ±0.003 (passes gate).
        # Strong signal (±1.0) → ±0.024 expected return.
        pred_return = ensemble * 0.024

        # Direction: +1 bullish, -1 bearish, 0 neutral
        if abs(ensemble) < 0.10:
            direction = 0.0
            confidence = 0.0
        else:
            direction = 1.0 if ensemble > 0 else -1.0
            confidence = min(abs(ensemble), 1.0)

        return EnsembleSignal(
            ticker=ticker,
            timestamp=datetime.now(timezone.utc),
            # Technical score mapped to lgbm fields (execution layer reads these)
            lgbm_direction=direction,
            lgbm_confidence=confidence,
            lgbm_pred_return=pred_return,
            lgbm_dir_prob=dir_prob,
            # Fundamental score in transformer fields
            transformer_direction=1.0 if fund_score > 0.1 else (-1.0 if fund_score < -0.1 else 0.0),
            transformer_confidence=abs(fund_score),
            # Regime score in TCN fields
            tcn_direction=1.0 if regime_score > 0.1 else (-1.0 if regime_score < -0.1 else 0.0),
            tcn_confidence=abs(regime_score),
            # Sentiment
            sentiment_index=sentiment_score,
            # Ensemble
            ensemble_signal=ensemble,
            # Weights reflect Pipeline B allocation (for logging/dashboard)
            w_lgbm=w.technicals,
            w_transformer=w.fundamentals,
            w_tcn=w.regime,
            w_sentiment=w.sentiment,
        )

    async def compute_universe(
        self,
        universe_features: dict[str, dict[str, float]],
    ) -> list[EnsembleSignal]:
        """Compute signals for all tickers in the universe.

        Args:
            universe_features: {ticker: feature_row_dict} from the latest
                feature_matrix row per ticker.

        Returns:
            List of EnsembleSignal sorted by |ensemble_signal| descending.
        """
        signals: list[EnsembleSignal] = []

        for ticker, feature_row in universe_features.items():
            try:
                sig = await self.compute_signal(ticker, feature_row)
                signals.append(sig)
            except Exception as exc:
                logger.warning(
                    "pipeline_b_signal_error",
                    ticker=ticker,
                    error=str(exc),
                )

        # Sort by signal strength (strongest first)
        signals.sort(key=lambda s: abs(s.ensemble_signal), reverse=True)
        return signals
