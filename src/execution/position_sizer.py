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
#
# 2026-09-15 — FAIL-CLOSED REWRITE. This table used to cover 24 tickers and
# every miss fell through to a single shared `"other"` bucket. The trading
# universe is SCREENED DYNAMICALLY into the DB (see
# screener_agent.load_universe_from_db), so it rotates faster than any
# hand-maintained table can track: across the live ledger the bot traded 57
# distinct names and 35 of them were absent here.
#
# The consequence was not a missing limit, it was a SILENTLY INVERTED one.
# `MAX_POSITIONS_PER_SECTOR = 2` counts positions per bucket, so 34 unmapped
# names sharing one bucket meant the correlation guard read a basket of four
# simultaneous semiconductor longs as "2 semis + 2 other" and passed it.
# Measured on the ledger (`python scripts/replay_ledger.py sectors`):
#   2026-08-19  KLAC, INTC, MU, WDC  open together   -> -$1,237 in 31 minutes
#   2026-09-11  SNDK, LITE, LRCX, AMD open together  ->   -$621
# Net on unmapped tickers -$1,203 vs +$408 on mapped ones (M2, n=111).
#
# Two changes, together:
#   1. The table is completed for every name the bot has actually traded.
#   2. `sector_of()` NEVER returns a shared permissive bucket. An unknown
#      ticker resolves to UNMAPPED_SECTOR, which carries the STRICTEST caps in
#      the system (one position, one position's worth of notional). An
#      unrecognized name is assumed to be correlated with every other
#      unrecognized name, because we have no evidence that it is not.
#
# Adding a ticker here LOOSENS a constraint, so it is a deliberate act. Leaving
# one out now costs opportunity, not risk — which is the correct direction for
# a table that will always lag a dynamic universe.

SECTOR_MAP: dict[str, str] = {
    # Software / internet / platform tech
    "AAPL": "tech",
    "MSFT": "tech",
    "GOOGL": "tech",
    "PLTR": "tech",
    "MSTR": "tech",
    "ORCL": "tech",
    "NOW": "tech",
    "SNOW": "tech",
    "DDOG": "tech",
    "WDAY": "tech",
    "TTD": "tech",
    "ANET": "tech",
    "DELL": "tech",
    # Semiconductors / semicap / memory & storage / photonics.
    # Deliberately ONE bucket: these names share the same demand cycle and
    # trade as a single factor intraday. Splitting "semicap" or "photonics"
    # out would re-create the 2026-08-19 basket under prettier labels.
    "NVDA": "semis",
    "AVGO": "semis",
    "AMD": "semis",
    "ARM": "semis",
    "SNDK": "semis",
    "MU": "semis",
    "SMCI": "semis",
    "WDC": "semis",
    "INTC": "semis",
    "QCOM": "semis",
    "MRVL": "semis",
    "KLAC": "semis",
    "LRCX": "semis",
    "AMAT": "semis",
    "TER": "semis",
    "STX": "semis",
    "LITE": "semis",
    "COHR": "semis",
    "CIEN": "semis",
    "FLEX": "semis",
    # Financials
    "JPM": "financials",
    "V": "financials",
    "MA": "financials",
    "GS": "financials",
    "WFC": "financials",
    "MSCI": "financials",
    "HOOD": "financials",
    # Consumer
    "AMZN": "consumer",
    "TSLA": "consumer",
    "COST": "consumer",
    "NFLX": "consumer",
    "APTV": "consumer",
    "GRMN": "consumer",
    # Energy
    "XOM": "energy",
    "CVX": "energy",
    "COP": "energy",
    # Healthcare / pharma
    "LLY": "healthcare",
    "UNH": "healthcare",
    "JNJ": "healthcare",
    "PFE": "healthcare",
    "ABBV": "healthcare",
    "MRNA": "healthcare",
    # Industrials / defense
    "LMT": "industrials",
    "LDOS": "industrials",
    "LII": "industrials",
    "ZBRA": "industrials",
}

# Bucket for any ticker absent from SECTOR_MAP. It is a real bucket name so it
# shows up in diagnostics and logs, and it is subject to the strictest caps in
# the system — never a permissive default.
UNMAPPED_SECTOR = "unmapped"


def sector_of(ticker: str) -> str:
    """Resolve a ticker to its correlation bucket, failing CLOSED.

    Args:
        ticker: Ticker symbol.

    Returns:
        The mapped sector, or `UNMAPPED_SECTOR` for anything unrecognized.
        Never a shared permissive bucket: callers must pair this with
        `max_positions_for_sector` / `sector_cap_pct`, which treat
        `UNMAPPED_SECTOR` as the most restrictive bucket in the system.
    """
    return SECTOR_MAP.get(ticker.upper(), UNMAPPED_SECTOR)


def max_positions_for_sector(sector: str) -> int:
    """Concurrent-position cap for a correlation bucket.

    Args:
        sector: Bucket name, as returned by `sector_of`.

    Returns:
        `MAX_POSITIONS_UNMAPPED` for the unmapped bucket (an unknown name is
        assumed correlated with every other unknown name), otherwise
        `MAX_POSITIONS_PER_SECTOR_DEFAULT`.
    """
    return (
        MAX_POSITIONS_UNMAPPED if sector == UNMAPPED_SECTOR
        else MAX_POSITIONS_PER_SECTOR_DEFAULT
    )


def sector_cap_pct(sector: str) -> float:
    """Maximum share of the portfolio allowed in a correlation bucket.

    Args:
        sector: Bucket name, as returned by `sector_of`.

    Returns:
        The notional cap as a fraction of portfolio value. The unmapped bucket
        gets one position's worth, mirroring its one-position count cap.
    """
    return (
        _UNMAPPED_SECTOR_CAP_PCT if sector == UNMAPPED_SECTOR
        else _SECTOR_CAP_PCT
    )

# ─── Pipeline configuration ─────────────────────────────────────────────────

# Stage 1: Signal-proportional base sizing
# Bigger positions (backtest-validated 2026-06-24 on the ~$98k account, no PDT):
# strong signals target the 10% per-position cap; the conviction floor was
# raised so high-conviction entries actually reach ~10% instead of ~3%.
# Backtest: avg position 2.4%→~5%, return ~doubled both legs, max DD still ~0.2%.
_BASE_MIN_PCT = 0.24       # weakest qualifying entry
_BASE_MAX_PCT = 0.34       # strongest conviction

# Stage 2: ATR volatility normalization
# Scale positions so each trade risks roughly the same $ amount.
# atr_pct arrives as a 1-MINUTE ATR ratio; it is converted to a daily-sigma
# estimate (×sqrt(390)) before comparing against the daily target. Target +
# upscale were raised so even volatile names can approach the 10% cap (the
# 10% notional cap + 60% heat ceiling + 25% breaker remain the safety net).
_DAILY_VOL_SQRT_BARS = 19.75  # sqrt(390 one-minute bars)
_TARGET_ATR_PCT = 0.030    # Target daily sigma (raised from 1.5% → bigger positions)
_ATR_FLOOR = 0.0002        # Floor 1m ATR_pct to prevent divide-by-zero / huge sizes
_ATR_CEIL = 0.01           # Ceil 1m ATR_pct to prevent tiny sizes on flash-crash bars
_ATR_UPSCALE_CAP = 2.0     # Max upscale for low-vol stocks (prevent oversizing)

# Stage 3: Kelly fraction caps per conviction bucket
_KELLY_BUCKETS: list[tuple[float, float, float]] = [
    # (dir_prob_lo, dir_prob_hi, max_pct) — all well below the 25% breaker cap;
    # the 10% _MAX_NOTIONAL_PCT is the real binding cap on a large account
    (0.55, 0.65, 0.30),    # low conviction
    (0.65, 0.80, 0.33),    # mid conviction
    (0.80, 1.01, 0.36),    # high conviction
]
_KELLY_POSITIVE_FLOOR = 0.15  # small positive Kelly shouldn't zero out sizing

# Stage 4: Portfolio constraints
_HEAT_TIERS: list[tuple[float, float]] = [
    # (heat_threshold, size_multiplier) — aligned with the signal loop's
    # 75% portfolio heat ceiling (2026-07-10 deployment-target raise)
    (0.60, 1.00),   # heat < 60%: full size
    (0.75, 0.50),   # 60% ≤ heat < 75%: half size
    (1.00, 0.00),   # heat ≥ 75%: no new entries
]
_SECTOR_CAP_PCT = 0.40     # Max 40% of portfolio in any single KNOWN sector
# The unmapped bucket gets one position's worth of notional — the same
# restriction its one-position count cap expresses, so the two agree. This had
# the identical fail-open defect as the count guard: every unrecognized ticker
# shared one "other" bucket and 40% of the book could pile into names the
# correlation model knew nothing about.
_UNMAPPED_SECTOR_CAP_PCT = 0.125   # == _MAX_NOTIONAL_PCT: one position

# Concurrent-position caps per correlation bucket. Consumed by
# `max_positions_for_sector`; the signal loop's entry gate enforces them.
MAX_POSITIONS_PER_SECTOR_DEFAULT = 2   # max concurrent positions per known sector
MAX_POSITIONS_UNMAPPED = 1             # an unknown name gets no diversification credit

# Stage 5: Minimum viable trade
_MIN_NOTIONAL = 1000.0     # $1k minimum trade (probation probes stay viable)
_MAX_NOTIONAL = 2500.0     # hard $ cap per position on small accounts
# Per-position cap on larger accounts (2026-07-01: 0.10→0.15 — "fewer, bigger
# bets"). With MAX_OPEN_POSITIONS=4 and the 60% heat ceiling, 4 × 15% = 60%
# fully deploys at max conviction. Backtest-validated before deploy.
#
# 2026-07-24: 0.15→0.125. The 15% cap was derived against 4 slots / 60% heat,
# but v0.3.5 moved production to 6 slots / 75% heat and this constant was never
# re-derived — 6 × 15% = 90% overshoots the 75% ceiling by construction, which
# is how the book reached 87.2% heat on 07-24. 6 × 12.5% = 75% exactly, so max
# conviction now lands ON the ceiling instead of through it.
# Second reason: over n=108, size was ANTI-predictive — above-median-size
# trades won 40.7% (net −$527) vs 51.9% (net −$278) below-median. Until the
# sizer earns its dispersion back, compressing the cap is the conservative
# direction: it reduces the amount of capital steered by the weaker signal.
_MAX_NOTIONAL_PCT = 0.125  # on larger accounts: cap at 12.5% of portfolio
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
        daily_vol = clamped_atr * _DAILY_VOL_SQRT_BARS
        vol_scalar = min(_TARGET_ATR_PCT / daily_vol, _ATR_UPSCALE_CAP)
        stage2 = stage1 * vol_scalar

        # ── Stage 3: Kelly Fraction Cap ──────────────────────────────────────
        # Determine conviction bucket cap
        bucket_cap = _KELLY_BUCKETS[-1][2]  # default to highest bucket
        abs_dir_prob = max(dir_prob, 1.0 - dir_prob)  # symmetric: use distance from 0.5
        for lo, hi, cap in _KELLY_BUCKETS:
            if lo <= abs_dir_prob < hi:
                bucket_cap = cap
                break

        # Positive Kelly (proven edge) caps size, floored so a barely-positive
        # Kelly doesn't zero out trades. Non-positive Kelly is handled by the
        # signal loop's probation logic (probe-sized entries), so the bucket
        # cap applies unchanged here.
        if kelly_fraction > 0:
            stage3 = min(stage2, bucket_cap, max(kelly_fraction, _KELLY_POSITIVE_FLOOR))
        else:
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

        # 4b: Sector cap — prevent concentration in a single correlation bucket.
        # `sector_of` fails closed: an unrecognized ticker lands in
        # UNMAPPED_SECTOR, which carries the tighter _UNMAPPED_SECTOR_CAP_PCT.
        sector = sector_of(ticker)
        cap_pct = sector_cap_pct(sector)
        current_sector_notional = sector_notionals.get(sector, 0.0)
        sector_heat = current_sector_notional / max(portfolio_value, 1.0)
        max_sector_room = max(cap_pct - sector_heat, 0.0)

        if max_sector_room <= 0:
            logger.info(
                "sizing_blocked_sector_cap",
                ticker=ticker,
                sector=sector,
                sector_heat=round(sector_heat, 3),
                cap=cap_pct,
            )
            return None

        stage4 = min(stage4, max_sector_room)

        # ── Stage 5: Minimum Viable Check ────────────────────────────────────
        notional = stage4 * portfolio_value
        # Hard cap: $ cap on small accounts, % cap on larger ones
        notional = min(notional, max(_MAX_NOTIONAL, _MAX_NOTIONAL_PCT * portfolio_value))
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
