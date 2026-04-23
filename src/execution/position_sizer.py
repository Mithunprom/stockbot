"""Smart Position Sizer — 6-stage pipeline for position sizing.

Replaces the scattered Kelly / RL / flat-percentage sizing logic with a clean,
auditable pipeline that addresses:
  1. No signal-proportional sizing (was flat %)
  2. No per-ticker ATR normalization (same % regardless of volatility)
  3. No sector/correlation limits (3 energy stocks could each take 25%)

Pipeline stages:
  Signal (dir_prob, pred_ret)
    → [1] Signal-Proportional Base (1–6% based on conviction distance from 0.5)
    → [2] ATR Volatility Normalization (equal dollar-risk per trade)
    → [3] Kelly Fraction Cap (per-bucket: low/mid/high conviction)
    → [4] Portfolio Constraints (graduated heat 50–80%, sector cap 40%)
    → [5] Minimum Viable Check ($100 min, 1 share min)
    → [6] Share Conversion (fractional for paper, whole for live)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


# ─── Sector mapping for the trading universe ────────────────────────────────

SECTOR_MAP: dict[str, str] = {
    # Tech
    "AAPL": "tech",
    "MSFT": "tech",
    "GOOGL": "tech",
    "PLTR": "tech",
    "MSTR": "tech",
    # Semiconductors
    "NVDA": "semis",
    "AVGO": "semis",
    "AMD": "semis",
    "ARM": "semis",
    "SNDK": "semis",
    # Financials
    "JPM": "financials",
    "V": "financials",
    "MA": "financials",
    # Consumer
    "AMZN": "consumer",
    "TSLA": "consumer",
    "COST": "consumer",
    "NFLX": "consumer",
    # Energy
    "XOM": "energy",
    "CVX": "energy",
    # Healthcare
    "LLY": "healthcare",
    "UNH": "healthcare",
}

# ─── Pipeline configuration ─────────────────────────────────────────────────

# Stage 1: Signal-proportional base sizing
_BASE_MIN_PCT = 0.02       # 2% at weakest conviction (dir_prob ≈ 0.55)
_BASE_MAX_PCT = 0.08       # 8% at strongest conviction (dir_prob ≈ 1.0)

# Stage 2: ATR volatility normalization
# Scale positions so each trade risks roughly the same $ amount.
# High-ATR tickers get smaller positions; low-ATR tickers get larger.
_TARGET_ATR_PCT = 0.012    # Universe median ATR_pct (~1.2% for mega-cap equities)
_ATR_FLOOR = 0.001         # Floor ATR_pct to prevent divide-by-zero / huge sizes
_ATR_CEIL = 0.05           # Ceil ATR_pct to prevent tiny sizes on flash-crash bars
_ATR_UPSCALE_CAP = 2.0     # Max upscale for low-vol stocks (prevent oversizing)

# Stage 3: Kelly fraction caps per conviction bucket
_KELLY_BUCKETS: list[tuple[float, float, float]] = [
    # (dir_prob_lo, dir_prob_hi, max_pct)
    (0.55, 0.65, 0.04),    # low conviction → cap at 4%
    (0.65, 0.80, 0.06),    # mid conviction → cap at 6%
    (0.80, 1.01, 0.08),    # high conviction → cap at 8%
]

# Stage 4: Portfolio constraints
_HEAT_TIERS: list[tuple[float, float]] = [
    # (heat_threshold, size_multiplier)
    (0.50, 1.00),   # heat < 50%: full size
    (0.65, 0.50),   # 50% ≤ heat < 65%: half size
    (0.80, 0.25),   # 65% ≤ heat < 80%: quarter size
    (1.00, 0.00),   # heat ≥ 80%: no new entries
]
_SECTOR_CAP_PCT = 0.40     # Max 40% of portfolio in any single sector

# Stage 5: Minimum viable trade
_MIN_NOTIONAL = 100.0      # $100 minimum trade
_MAX_NOTIONAL = 8000.0     # $8k hard cap per position (matches small account)
_MIN_SHARES_FRACTIONAL = 0.01   # Paper mode minimum
_MIN_SHARES_WHOLE = 1.0         # Live mode minimum


# ─── Sizing result ──────────────────────────────────────────────────────────

@dataclass
class SizingResult:
    """Output of the 6-stage position sizing pipeline."""
    ticker: str
    side: str                # "buy" or "sell"
    shares: float            # Final share count
    notional: float          # Final dollar amount
    size_pct: float          # Final % of portfolio

    # Audit trail — what each stage computed
    stage1_base_pct: float
    stage2_atr_pct: float
    stage3_kelly_pct: float
    stage4_constraint_pct: float
    stage5_viable: bool
    stage6_mode: str         # "fractional" or "whole"

    # Inputs for logging
    dir_prob: float
    pred_return: float
    atr_pct: float
    kelly_fraction: float
    portfolio_heat: float
    sector_heat: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "side": self.side,
            "shares": self.shares,
            "notional": round(self.notional, 2),
            "size_pct": round(self.size_pct, 4),
            "stages": {
                "1_base": round(self.stage1_base_pct, 4),
                "2_atr": round(self.stage2_atr_pct, 4),
                "3_kelly": round(self.stage3_kelly_pct, 4),
                "4_constraint": round(self.stage4_constraint_pct, 4),
                "5_viable": self.stage5_viable,
                "6_mode": self.stage6_mode,
            },
            "inputs": {
                "dir_prob": round(self.dir_prob, 4),
                "pred_return": round(self.pred_return, 6),
                "atr_pct": round(self.atr_pct, 4),
                "kelly": round(self.kelly_fraction, 4),
                "heat": round(self.portfolio_heat, 4),
                "sector_heat": round(self.sector_heat, 4),
            },
        }


# ─── Smart Position Sizer ──────────────────────────────────────────────────

class SmartPositionSizer:
    """6-stage position sizing pipeline.

    Usage:
        sizer = SmartPositionSizer(mode="paper")
        result = sizer.compute(
            ticker="AAPL", dir_prob=0.72, pred_return=0.003,
            atr_pct=0.012, price=185.0,
            portfolio_value=100000, portfolio_heat=0.35,
            sector_notionals={"tech": 15000, "semis": 8000},
            kelly_fraction=0.15,
        )
        if result is not None:
            # execute order for result.shares at result.side
    """

    def __init__(self, mode: str = "paper") -> None:
        self._mode = mode   # "paper" or "live"

    def compute(
        self,
        ticker: str,
        dir_prob: float,
        pred_return: float,
        atr_pct: float,
        price: float,
        portfolio_value: float,
        portfolio_heat: float,
        sector_notionals: dict[str, float],
        kelly_fraction: float,
    ) -> SizingResult | None:
        """Run the full 6-stage sizing pipeline.

        Args:
            ticker: Ticker symbol.
            dir_prob: LightGBM P(up) in [0, 1].
            pred_return: LightGBM predicted forward return.
            atr_pct: ATR(14) / close for this ticker (from feature_matrix).
            price: Current price.
            portfolio_value: Total portfolio value ($).
            portfolio_heat: Current fraction of portfolio deployed.
            sector_notionals: Sector -> total $ currently deployed in that sector.
            kelly_fraction: Current rolling Kelly fraction from trade history.

        Returns:
            SizingResult with final shares/notional, or None if trade is rejected.
        """
        direction = 1 if pred_return > 0 else -1
        side = "buy" if direction > 0 else "sell"
        conviction = abs(dir_prob - 0.5) * 2.0  # 0.0 (no edge) to 1.0 (max conviction)

        # ── Stage 1: Signal-Proportional Base ────────────────────────────────
        stage1 = _BASE_MIN_PCT + (_BASE_MAX_PCT - _BASE_MIN_PCT) * conviction
        # Scale by prediction magnitude: stronger predicted return → larger base
        pred_scale = min(abs(pred_return) / 0.005, 2.0)  # normalize to ~0.5% expected
        stage1 *= max(pred_scale, 0.5)  # floor at 0.5× to avoid near-zero sizing
        stage1 = min(stage1, _BASE_MAX_PCT)  # hard cap at max

        # ── Stage 2: ATR Volatility Normalization ────────────────────────────
        # Scale so each trade risks equal $ regardless of ticker volatility.
        # vol_scalar > 1 for calm stocks (bigger position ok),
        # vol_scalar < 1 for volatile stocks (reduce position).
        clamped_atr = max(min(atr_pct, _ATR_CEIL), _ATR_FLOOR)
        vol_scalar = min(_TARGET_ATR_PCT / clamped_atr, _ATR_UPSCALE_CAP)
        stage2 = stage1 * vol_scalar

        # ── Stage 3: Kelly Fraction Cap ──────────────────────────────────────
        # Determine conviction bucket cap
        bucket_cap = _KELLY_BUCKETS[-1][2]  # default to highest bucket
        abs_dir_prob = max(dir_prob, 1.0 - dir_prob)  # symmetric: use distance from 0.5
        for lo, hi, cap in _KELLY_BUCKETS:
            if lo <= abs_dir_prob < hi:
                bucket_cap = cap
                break

        # If Kelly fraction is positive (proven edge), allow up to half-Kelly
        # but never exceed the conviction bucket cap
        if kelly_fraction > 0:
            kelly_limit = kelly_fraction / 2.0  # half-Kelly for safety
            stage3 = min(stage2, bucket_cap, kelly_limit)
        else:
            # No proven edge yet — use bucket cap for data collection.
            # Pipeline B needs meaningful positions to validate signals.
            stage3 = min(stage2, bucket_cap)

        # ── Stage 4: Portfolio Constraints ───────────────────────────────────
        # 4a: Graduated heat limit
        heat_multiplier = 0.0
        for threshold, multiplier in _HEAT_TIERS:
            if portfolio_heat < threshold:
                heat_multiplier = multiplier
                break

        if heat_multiplier <= 0:
            logger.debug(
                "sizing_blocked_heat",
                ticker=ticker,
                heat=round(portfolio_heat, 3),
            )
            return None

        stage4 = stage3 * heat_multiplier

        # 4b: Sector cap — prevent concentration in a single sector
        sector = SECTOR_MAP.get(ticker, "other")
        current_sector_notional = sector_notionals.get(sector, 0.0)
        sector_heat = current_sector_notional / max(portfolio_value, 1.0)
        max_sector_room = max(_SECTOR_CAP_PCT - sector_heat, 0.0)

        if max_sector_room <= 0:
            logger.info(
                "sizing_blocked_sector_cap",
                ticker=ticker,
                sector=sector,
                sector_heat=round(sector_heat, 3),
                cap=_SECTOR_CAP_PCT,
            )
            return None

        stage4 = min(stage4, max_sector_room)

        # ── Stage 5: Minimum Viable Check ────────────────────────────────────
        notional = stage4 * portfolio_value
        # Hard cap: never exceed _MAX_NOTIONAL per position
        notional = min(notional, _MAX_NOTIONAL)
        shares = notional / max(price, 0.01)

        min_shares = (
            _MIN_SHARES_FRACTIONAL if self._mode == "paper"
            else _MIN_SHARES_WHOLE
        )

        viable = notional >= _MIN_NOTIONAL and shares >= min_shares
        if not viable:
            logger.debug(
                "sizing_below_minimum",
                ticker=ticker,
                notional=round(notional, 2),
                shares=round(shares, 4),
            )
            return None

        # ── Stage 6: Share Conversion ────────────────────────────────────────
        if self._mode == "paper":
            # Fractional shares allowed in Alpaca paper trading
            shares = round(shares, 2)
            mode_label = "fractional"
        else:
            # Live: whole shares only (floor to avoid exceeding budget)
            shares = math.floor(shares)
            mode_label = "whole"
            if shares < 1:
                return None

        # Recalculate final notional from actual shares
        final_notional = shares * price
        final_pct = final_notional / max(portfolio_value, 1.0)

        result = SizingResult(
            ticker=ticker,
            side=side,
            shares=shares,
            notional=final_notional,
            size_pct=final_pct,
            stage1_base_pct=stage1,
            stage2_atr_pct=stage2,
            stage3_kelly_pct=stage3,
            stage4_constraint_pct=stage4,
            stage5_viable=viable,
            stage6_mode=mode_label,
            dir_prob=dir_prob,
            pred_return=pred_return,
            atr_pct=atr_pct,
            kelly_fraction=kelly_fraction,
            portfolio_heat=portfolio_heat,
            sector_heat=sector_heat,
        )

        logger.info(
            "sizing_computed",
            ticker=ticker,
            side=side,
            pct=round(final_pct, 4),
            shares=shares,
            notional=round(final_notional, 2),
            stages=f"{stage1:.3f}→{stage2:.3f}→{stage3:.3f}→{stage4:.3f}",
            dir_prob=round(dir_prob, 3),
            atr=round(atr_pct, 4),
            kelly=round(kelly_fraction, 3),
            heat=round(portfolio_heat, 3),
            sector=sector,
        )

        return result
