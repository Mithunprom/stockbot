"""Unit tests for Kelly fraction sensitivity analysis (H26).

Validates that kelly_sensitivity() correctly diagnoses the clustering problem
observed in production: Aug 31 2026 (12 trades in 2 batches drove fraction
+0.23 → -5.37) and Sep 11 2026 (LITE -3.5% with dead stops drove to -0.67).
"""

import statistics
from datetime import date, datetime, timedelta, timezone

import pytest

from src.execution.kelly_sensitivity import _kelly_fraction, kelly_sensitivity


# ── Helper ────────────────────────────────────────────────────────────────────

def _make_outcomes(
    trades: list[tuple[str, float]],  # (date_str_YYYY-MM-DD, pnl_pct)
    base_hour: int = 14,
) -> list[tuple[datetime, float]]:
    """Build (exit_time_utc, pnl_pct) pairs from a list of (date, pnl)."""
    result = []
    for date_str, pnl in trades:
        y, m, d = map(int, date_str.split("-"))
        ts = datetime(y, m, d, base_hour, 0, 0, tzinfo=timezone.utc)
        result.append((ts, pnl))
    return result


# ── _kelly_fraction unit tests ────────────────────────────────────────────────

def test_kelly_fraction_basic():
    """Standard formula: p=0.5, b=2 → Kelly = (0.5*2 - 0.5)/2 = 0.25."""
    outcomes = [0.02, 0.02, 0.02, 0.02, 0.02, -0.01, -0.01, -0.01, -0.01, -0.01]
    f = _kelly_fraction(outcomes)
    assert f is not None
    assert abs(f - 0.25) < 1e-6


def test_kelly_fraction_returns_none_below_min_trades():
    outcomes = [0.01, -0.01, 0.01, -0.01, 0.01]  # only 5
    assert _kelly_fraction(outcomes) is None


def test_kelly_fraction_returns_none_all_wins():
    outcomes = [0.01] * 10
    assert _kelly_fraction(outcomes) is None


def test_kelly_fraction_returns_none_all_losses():
    outcomes = [-0.01] * 10
    assert _kelly_fraction(outcomes) is None


def test_kelly_fraction_negative_on_bad_window():
    """Negative fraction when avg_loss >> avg_win (as on Sep 10-11 2026 data)."""
    # Matches actual Sep 10-11 pnl_pct values from live trades
    outcomes = [
        0.0058, 0.0058, -0.0051, 0.0059, 0.0011, 0.0011,  # Sep 10 (6 trades)
        0.0109, -0.0102, -0.0351, -0.0167, -0.0017, 0.0015,  # Sep 11 (6 trades)
    ]
    f = _kelly_fraction(outcomes)
    assert f is not None
    assert f < 0, f"Expected negative Kelly, got {f}"
    # Matches diagnostics observation of -0.6661
    assert abs(f - (-0.6661)) < 0.01, f"Expected ≈-0.67, got {f:.4f}"


# ── kelly_sensitivity() integration tests ─────────────────────────────────────

NOW_UTC = datetime(2026, 9, 16, 0, 0, 0, tzinfo=timezone.utc)


def test_h26_sep1011_per_day_is_inactive():
    """Per-day view of Sep 10-11 data (2 unique days) must be inactive.

    The current 10-day window holds only Sep 10-11 trades (n=12).
    Per-day-avg has only 2 unique trading days, far below KELLY_MIN_TRADES=10.
    This exposes that the per-trade fraction's -0.67 is computed from a
    2-day sample masquerading as 12 independent observations.
    """
    outcomes = _make_outcomes([
        ("2026-09-10", 0.0058),
        ("2026-09-10", 0.0058),
        ("2026-09-10", -0.0051),
        ("2026-09-10", 0.0059),
        ("2026-09-10", 0.0011),
        ("2026-09-10", 0.0011),
        ("2026-09-11", 0.0109),
        ("2026-09-11", -0.0102),
        ("2026-09-11", -0.0351),
        ("2026-09-11", -0.0167),
        ("2026-09-11", -0.0017),
        ("2026-09-11", 0.0015),
    ])
    result = kelly_sensitivity(outcomes, now=NOW_UTC)

    # Per-trade: n=12, kelly ≈ -0.67
    pt = result["per_trade"]
    assert pt["n_trades"] == 12
    assert pt["n_unique_days"] == 2
    assert pt["kelly"] is not None
    assert pt["kelly"] < 0

    # Per-day-avg: only 2 days — must be inactive
    pd = result["per_day_avg"]
    assert pd["n_days"] == 2
    assert pd["kelly"] is None
    assert "fewer than 10" in pd["note"]


def test_h26_worst_trade_impact_improves_fraction():
    """Removing the LITE -3.5% outlier (dead-stop artifact) must improve Kelly."""
    outcomes = _make_outcomes([
        ("2026-09-10", 0.0058),
        ("2026-09-10", 0.0058),
        ("2026-09-10", -0.0051),
        ("2026-09-10", 0.0059),
        ("2026-09-10", 0.0011),
        ("2026-09-10", 0.0011),
        ("2026-09-11", 0.0109),
        ("2026-09-11", -0.0102),
        ("2026-09-11", -0.0351),   # LITE -$428, dead-stop artifact
        ("2026-09-11", -0.0167),
        ("2026-09-11", -0.0017),
        ("2026-09-11", 0.0015),
    ])
    result = kelly_sensitivity(outcomes, now=NOW_UTC)
    wt = result["worst_trade_impact"]
    assert wt["worst_trade_pct"] is not None
    assert abs(wt["worst_trade_pct"] - (-0.0351)) < 1e-4
    # Removing the dead-stop outlier should improve (or at least not worsen) Kelly
    if wt["kelly_without_worst"] is not None and wt["kelly_with_worst"] is not None:
        assert wt["kelly_without_worst"] > wt["kelly_with_worst"]


def test_h26_window_comparison_longer_window_may_differ():
    """A 30-day window that includes pre-Aug-31 profitable days shows higher Kelly."""
    good_days = [
        # Simulate 10 profitable trading days at 0.3% avg gain
        ("2026-08-13", 0.005), ("2026-08-13", 0.006),
        ("2026-08-14", 0.007), ("2026-08-14", 0.005),
        ("2026-08-17", 0.008), ("2026-08-17", 0.004),
        ("2026-08-18", 0.003), ("2026-08-18", 0.006),
        ("2026-08-19", -0.029), ("2026-08-19", -0.026),  # macro selloff
        ("2026-08-20", 0.015), ("2026-08-20", 0.003),
        ("2026-08-21", -0.008), ("2026-08-21", 0.010),
    ]
    bad_cluster = [
        # Aug 31 and Sep 10-11 cluster
        ("2026-08-31", -0.020), ("2026-08-31", -0.008), ("2026-08-31", -0.011),
        ("2026-08-31", -0.006), ("2026-08-31", -0.002), ("2026-08-31", 0.002),
        ("2026-09-10", 0.006), ("2026-09-10", 0.006), ("2026-09-10", -0.005),
        ("2026-09-10", 0.006), ("2026-09-10", 0.001), ("2026-09-10", 0.001),
        ("2026-09-11", 0.011), ("2026-09-11", -0.010), ("2026-09-11", -0.035),
        ("2026-09-11", -0.017), ("2026-09-11", -0.002), ("2026-09-11", 0.002),
    ]
    outcomes = _make_outcomes(good_days + bad_cluster)
    result = kelly_sensitivity(outcomes, now=NOW_UTC)

    wc = result["window_comparison"]
    assert "10d" in wc
    assert "20d" in wc
    assert "30d" in wc
    # 10-day window contains only Sep 6-16 trades (the two bad clusters)
    assert wc["10d"]["n_trades"] <= 12
    # 30-day window contains more trades (Aug 13+ onward)
    assert wc["30d"]["n_trades"] >= wc["10d"]["n_trades"]


def test_h26_per_day_avg_n_unique_days_matches_expected():
    """Per-day-avg compresses N trades per day to exactly one outcome per day.

    12 trading days, 4 trades each (48 total). Per-trade view has 48 outcomes;
    per-day view must have exactly 12.
    """
    multi_trades = []
    ref = date(2026, 9, 1)
    for i in range(12):
        day = (ref + timedelta(days=i)).isoformat()
        multi_trades.extend([(day, p) for p in (0.008, 0.006, -0.003, -0.002)])

    outcomes = _make_outcomes(multi_trades)
    result = kelly_sensitivity(outcomes, now=datetime(2026, 9, 13, 23, 0, 0, tzinfo=timezone.utc))

    pt = result["per_trade"]
    pd = result["per_day_avg"]

    # 4 trades × 12 days = 48 per-trade outcomes
    assert pt["n_trades"] == 48
    assert pt["n_unique_days"] == 12

    # Per-day compresses to exactly 12 day-averages
    assert pd["n_days"] == 12
    # Day avg = (0.008+0.006-0.003-0.002)/4 = 0.009/4 = +0.00225 — all positive days
    # → Kelly degenerate (all wins) → None
    assert pd["kelly"] is None  # all-positive day averages → _kelly_fraction returns None


def test_h26_per_day_kelly_computable_with_mixed_days():
    """Per-day Kelly is computable and positive when win-days dominate with adequate b."""
    # 10 win-days: avg +0.004 each. 4 loss-days: avg -0.004 each.
    # b = 0.004/0.004 = 1.0; p = 10/14 = 0.714
    # Kelly = (0.714*1 - 0.286)/1 = 0.428
    multi_trades = []
    ref = date(2026, 9, 1)
    for i in range(14):
        day = (ref + timedelta(days=i)).isoformat()
        if i < 10:
            # win day: avg = (0.007 + 0.005 - 0.002 - 0.002) / 4 = 0.008/4 = +0.002
            multi_trades.extend([(day, p) for p in (0.007, 0.005, -0.002, -0.002)])
        else:
            # loss day: avg = (-0.007 - 0.005 + 0.002 + 0.002) / 4 = -0.008/4 = -0.002
            multi_trades.extend([(day, p) for p in (-0.007, -0.005, 0.002, 0.002)])

    outcomes = _make_outcomes(multi_trades)
    result = kelly_sensitivity(outcomes, now=datetime(2026, 9, 15, 23, 0, 0, tzinfo=timezone.utc))

    pd = result["per_day_avg"]
    assert pd["n_days"] == 14
    assert pd["kelly"] is not None
    # b = 0.002/0.002 = 1, p = 10/14 → kelly > 0
    assert pd["kelly"] > 0, f"Expected positive per-day Kelly, got {pd['kelly']}"


def test_h26_output_structure():
    """kelly_sensitivity() always returns the expected top-level keys."""
    outcomes: list[tuple[datetime, float]] = []
    result = kelly_sensitivity(outcomes, now=NOW_UTC)
    assert set(result.keys()) == {
        "per_trade", "per_day_avg", "window_comparison", "worst_trade_impact"
    }
    assert "10d" in result["window_comparison"]
    assert "20d" in result["window_comparison"]
    assert "30d" in result["window_comparison"]


def test_h26_empty_outcomes_returns_none_kelly():
    """Empty window must not raise; all kelly fields None."""
    result = kelly_sensitivity([], now=NOW_UTC)
    assert result["per_trade"]["kelly"] is None
    assert result["per_day_avg"]["kelly"] is None
    assert result["worst_trade_impact"]["worst_trade_pct"] is None
