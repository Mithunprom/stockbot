"""
Kelly fraction sensitivity analysis for multi-trade-per-day intraday strategies.

For 30-bar intraday holds with up to 6 trades/day, the standard per-trade Kelly
fraction over a 10-calendar-day lookback is sensitive to trade clustering: a
single macro bad day can fire 6 correlated losses, dominating the fraction.
Observed: Aug 31 2026 (12 trades in 2 batches) drove fraction +0.23 → -5.37;
Sep 11 2026 (4 losses including LITE -3.5% with dead stops) drove to -0.67.

This module provides three diagnostic views (diagnostics-only, no live impact):

  per_trade       Standard production computation.
  per_day_avg     One outcome per trading day (average of all trades that day).
                  For a single-trade-per-day strategy this equals per_trade;
                  for multi-slot intraday it removes within-day correlation.
  window_comparison  Per-trade fractions at multiple lookback windows (10/20/30d).
  worst_trade_impact Kelly with and without the single worst pnl_pct trade removed,
                  showing how much one structural outlier (dead stops) moves the
                  fraction.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

# Must match KELLY_MIN_TRADES in signal_loop.py — kept explicit here to avoid
# a circular import (this module is imported inside SignalLoop methods).
_KELLY_MIN_TRADES = 10


def _kelly_fraction(outcomes: list[float]) -> float | None:
    """Compute Kelly f* = (p*b - q) / b from a list of pnl_pct outcomes.

    Returns None when the sample is too small or degenerate (all-win/all-loss).
    """
    if len(outcomes) < _KELLY_MIN_TRADES:
        return None
    wins = [o for o in outcomes if o > 0]
    losses = [o for o in outcomes if o < 0]
    if not wins or not losses:
        return None
    p = len(wins) / len(outcomes)
    avg_win = statistics.mean(wins)
    avg_loss = abs(statistics.mean(losses))
    b = avg_win / max(avg_loss, 1e-9)
    q = 1.0 - p
    return (p * b - q) / max(b, 1e-9)


def kelly_sensitivity(
    outcomes: list[tuple[datetime, float]],
    now: datetime | None = None,
    window_days: tuple[int, ...] = (10, 20, 30),
) -> dict[str, Any]:
    """Analyze Kelly fraction sensitivity to window length and trade clustering.

    Args:
        outcomes: (exit_time_utc, pnl_pct) pairs already filtered to the
            production window (i.e. output of _sizing_recent_outcomes after
            _prune_kelly_window).
        now: reference time for window boundary calculations; defaults to UTC now.
        window_days: lookback windows to compare in window_comparison.

    Returns a dict suitable for embedding in the /diagnostics response.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    # ── 1. Per-trade (current production) ────────────────────────────────────
    trade_pnl = [p for _, p in outcomes]
    current_kelly = _kelly_fraction(trade_pnl)
    n_trades = len(trade_pnl)
    unique_days = len({ts.date() for ts, _ in outcomes if ts is not None})

    # ── 2. Per-day-average ───────────────────────────────────────────────────
    by_day: dict[Any, list[float]] = defaultdict(list)
    for ts, pnl in outcomes:
        if ts is not None:
            by_day[ts.date()].append(pnl)
    day_avgs = [statistics.mean(v) for v in by_day.values()]
    n_days = len(day_avgs)
    # Require ≥10 unique trading days, mirroring the ≥10-trade threshold.
    day_kelly = _kelly_fraction(day_avgs) if n_days >= _KELLY_MIN_TRADES else None

    # ── 3. Multi-window comparison (per-trade) ───────────────────────────────
    window_results: dict[str, Any] = {}
    for days in window_days:
        cutoff = now - timedelta(days=days)
        w_pnl = [p for ts, p in outcomes if ts is not None and ts >= cutoff]
        window_results[f"{days}d"] = {
            "n_trades": len(w_pnl),
            "kelly": _kelly_fraction(w_pnl),
        }

    # ── 4. Worst-trade impact ────────────────────────────────────────────────
    worst_trade_pct: float | None = None
    kelly_ex_worst: float | None = None
    if n_trades >= _KELLY_MIN_TRADES + 1:
        worst_trade_pct = min(trade_pnl)
        idx = trade_pnl.index(worst_trade_pct)
        without_worst = trade_pnl[:idx] + trade_pnl[idx + 1:]
        kelly_ex_worst = _kelly_fraction(without_worst)

    return {
        "per_trade": {
            "kelly": round(current_kelly, 4) if current_kelly is not None else None,
            "n_trades": n_trades,
            "n_unique_days": unique_days,
        },
        "per_day_avg": {
            "kelly": round(day_kelly, 4) if day_kelly is not None else None,
            "n_days": n_days,
            "note": (
                "inactive — fewer than 10 trading days in window"
                if n_days < _KELLY_MIN_TRADES
                else "active"
            ),
        },
        "window_comparison": window_results,
        "worst_trade_impact": {
            "worst_trade_pct": round(worst_trade_pct, 4) if worst_trade_pct is not None else None,
            "kelly_with_worst": round(current_kelly, 4) if current_kelly is not None else None,
            "kelly_without_worst": round(kelly_ex_worst, 4) if kelly_ex_worst is not None else None,
        },
    }
