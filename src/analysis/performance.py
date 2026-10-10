"""Performance analytics — exit-reason stratified PnL.

Computes PF / WR / expectancy per exit_reason and surfaces concentration
risk (top-N wins as share of gross profit). Pure function — no DB/network
calls — so it can be unit-tested against synthetic trade lists and re-run
from any /trades snapshot export.

The motivating observation (Oct 9 2026, n=99, v0.9.x):

    Overall PF = 1.16, but three integrity_broker_reconcile exits
    (COHR +$1,451, MSTR +$1,008, LITE +$968) account for ~48 % of gross
    profit. Excluding those, the clean PF ≈ 0.60 on 96 regular exits. The
    strategy-only bucket (stop_loss + trailing_stop + take_profit) is even
    smaller. This function makes that stratification explicit and permanent.

    Principal Skeptic standing rule: a losing streak is FIRST a measurement
    question, THEN a strategy question. This function answers the measurement
    question by separating "stranded-position recovery" PnL from "designed
    exit" PnL.

Added: 2026-10-10 (H28)
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any


# Exit reasons that represent designed strategy outcomes (not recovery ops).
STRATEGY_EXIT_REASONS: frozenset[str] = frozenset({
    "stop_loss",
    "trailing_stop",
    "take_profit",
    "max_hold",
    "signal_reversal",
})

# Exit reason that marks halt-stranded position recovery.
# Introduced by v0.8.6: positions orphaned by the Sep-30 CB halt were closed
# from broker fill records rather than a live strategy signal. Their PnL is
# real but their hold duration is uncontrolled (potentially multi-day), which
# inflates both the gain and the measured edge if pooled with designed exits.
RECONCILE_REASON = "integrity_broker_reconcile"


def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def _hold_minutes(entry_time: str | None, exit_time: str | None) -> float | None:
    entry = _parse_dt(entry_time)
    exit_ = _parse_dt(exit_time)
    if entry is None or exit_ is None:
        return None
    delta = (exit_ - entry).total_seconds()
    if delta < 0:
        return None
    return round(delta / 60.0, 1)


def _bucket_stats(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate stats for one bucket of closed trades."""
    n = len(trades)
    if n == 0:
        return {
            "n": 0, "wins": 0, "losses": 0, "ties": 0,
            "gross_profit": 0.0, "gross_loss": 0.0, "net_pnl": 0.0,
            "pf": None, "win_rate": None, "expectancy": None,
            "avg_hold_minutes": None,
        }

    wins = [t for t in trades if (t.get("pnl") or 0.0) > 0]
    losses = [t for t in trades if (t.get("pnl") or 0.0) < 0]
    ties = [t for t in trades if (t.get("pnl") or 0.0) == 0]

    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    net_pnl = gross_profit - gross_loss

    pf = round(gross_profit / gross_loss, 3) if gross_loss > 0 else None
    win_rate = round(len(wins) / n, 4)
    expectancy = round(net_pnl / n, 2)

    hold_mins = [
        _hold_minutes(t.get("entry_time"), t.get("exit_time"))
        for t in trades
    ]
    valid = [h for h in hold_mins if h is not None]
    avg_hold = round(sum(valid) / len(valid), 1) if valid else None

    return {
        "n": n,
        "wins": len(wins),
        "losses": len(losses),
        "ties": len(ties),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "net_pnl": round(net_pnl, 2),
        "pf": pf,
        "win_rate": win_rate,
        "expectancy": expectancy,
        "avg_hold_minutes": avg_hold,
    }


def compute_exit_stratification(
    trades: list[dict[str, Any]],
    top_n_concentration: int = 3,
) -> dict[str, Any]:
    """Stratify closed-trade performance by exit_reason.

    Args:
        trades: Trade dicts from the /trades API response. Each dict must
            contain at minimum: ``pnl`` (float or None), ``exit_reason``
            (str or None), ``exit_time`` (str ISO or None), and optionally
            ``entry_time``, ``ticker``. Open trades (exit_time or pnl is
            None) are silently skipped and counted in n_open_skipped.
        top_n_concentration: How many top winning trades to measure for
            gross-profit concentration (default: 3).

    Returns:
        Dict with keys:
          ``overall``
              Stats across all closed trades (all exit reasons pooled).
          ``by_exit_reason``
              Per-reason dicts with identical stat keys.
          ``clean``
              Stats *excluding* ``integrity_broker_reconcile`` exits.
              This is the most representative view of normal strategy
              performance — stranded-position recovery PnL is excluded.
          ``strategy_only``
              Stats for STRATEGY_EXIT_REASONS only (stop_loss, trailing_stop,
              take_profit, max_hold, signal_reversal). Excludes any
              operational/recovery exits.
          ``concentration``
              Top-N winner analysis: their sum as % of gross profit, plus
              a list of {ticker, pnl, exit_reason} for each.
          ``n_open_skipped``
              Number of trades that were omitted (still open).
          ``n_exit_reasons``
              Count of distinct exit reasons seen.
          ``exit_reasons_seen``
              Sorted list of distinct exit reasons.
    """
    closed = [
        t for t in trades
        if t.get("exit_time") is not None and t.get("pnl") is not None
    ]
    n_open_skipped = len(trades) - len(closed)

    by_reason: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in closed:
        reason = t.get("exit_reason") or "unknown"
        by_reason[reason].append(t)

    by_reason_stats = {
        reason: _bucket_stats(bucket)
        for reason, bucket in sorted(by_reason.items())
    }

    clean = [t for t in closed if t.get("exit_reason") != RECONCILE_REASON]
    strategy = [
        t for t in closed
        if t.get("exit_reason") in STRATEGY_EXIT_REASONS
    ]

    # Concentration: top-N winners as % of total gross profit.
    all_wins = sorted(
        [t for t in closed if (t.get("pnl") or 0.0) > 0],
        key=lambda t: t["pnl"],
        reverse=True,
    )
    top_wins = all_wins[:top_n_concentration]
    overall_stats = _bucket_stats(closed)
    overall_gross = overall_stats["gross_profit"]
    top_wins_sum = round(sum(t["pnl"] for t in top_wins), 2)

    concentration = {
        "top_n": top_n_concentration,
        "top_n_gross_profit": top_wins_sum,
        "total_gross_profit": overall_gross,
        "top_n_pct_of_gross": (
            round(top_wins_sum / overall_gross * 100, 1)
            if overall_gross > 0 else None
        ),
        "tickers": [
            {
                "ticker": t.get("ticker"),
                "pnl": round(t["pnl"], 2),
                "exit_reason": t.get("exit_reason"),
            }
            for t in top_wins
        ],
    }

    return {
        "overall": overall_stats,
        "by_exit_reason": by_reason_stats,
        "clean": _bucket_stats(clean),
        "strategy_only": _bucket_stats(strategy),
        "concentration": concentration,
        "n_open_skipped": n_open_skipped,
        "n_exit_reasons": len(by_reason),
        "exit_reasons_seen": sorted(by_reason.keys()),
    }
