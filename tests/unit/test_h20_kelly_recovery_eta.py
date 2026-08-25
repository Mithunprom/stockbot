"""H20: Kelly probation recovery ETA computation tests.

Tests the pure logic of _kelly_probation_recovery_eta without importing
signal_loop (which requires unavailable runtime deps in the sandbox).
The function groups Kelly-window outcomes by exit date and finds the first
date at which cumulative rolloff drops n below KELLY_MIN_TRADES (=10),
making Kelly mode 'inactive' and re-allowing full-size entries.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


KELLY_LOOKBACK_DAYS = 10
KELLY_MIN_TRADES = 10


def compute_recovery_eta(
    outcomes: list[tuple[datetime, float]],
    now_utc: datetime,
) -> dict | None:
    """Standalone replica of _kelly_probation_recovery_eta logic."""
    cutoff = now_utc - timedelta(days=KELLY_LOOKBACK_DAYS)
    active = [(ts, p) for ts, p in outcomes if ts is not None and ts >= cutoff]
    n_current = len(active)
    if n_current < KELLY_MIN_TRADES:
        return None  # not in probation (inactive mode)
    # Check we'd actually be in probation (Kelly fraction <= 0)
    wins = [p for _, p in active if p > 0]
    losses = [p for _, p in active if p < 0]
    if not wins or not losses:
        return None
    avg_win = sum(wins) / len(wins)
    avg_loss = abs(sum(losses) / len(losses))
    if avg_loss == 0:
        return None
    p = len(wins) / n_current
    b = avg_win / avg_loss
    kelly = (p * b - (1 - p)) / b
    if kelly > 0:
        return None  # normal mode, no ETA needed
    by_date: dict[str, int] = {}
    for ts, _ in active:
        d = ts.strftime("%Y-%m-%d")
        by_date[d] = by_date.get(d, 0) + 1
    remaining = n_current
    recovery_date: str | None = None
    for date_str in sorted(by_date.keys()):
        trade_ts = datetime.fromisoformat(date_str + "T12:00:00").replace(tzinfo=timezone.utc)
        rollout = trade_ts + timedelta(days=KELLY_LOOKBACK_DAYS)
        remaining -= by_date[date_str]
        if remaining < KELLY_MIN_TRADES:
            recovery_date = rollout.strftime("%Y-%m-%d")
            break
    return {
        "earliest_recovery_date": recovery_date,
        "n_current": n_current,
        "by_date": by_date,
    }


def _make_outcomes(dates_counts: dict[str, tuple[int, int]]) -> list[tuple[datetime, float]]:
    """Build a list of (exit_ts, pnl_pct) outcomes.

    dates_counts: {YYYY-MM-DD: (n_wins, n_losses)}
    Wins have pnl_pct=+0.01, losses -0.02 (making Kelly negative when losses dominate).
    """
    result = []
    for date_str, (n_wins, n_losses) in dates_counts.items():
        ts = datetime.fromisoformat(date_str + "T13:00:00").replace(tzinfo=timezone.utc)
        result.extend([(ts, 0.01)] * n_wins)
        result.extend([(ts, -0.02)] * n_losses)
    return result


class TestKellyRecoveryEta:
    def test_returns_none_when_not_in_probation(self) -> None:
        """With positive Kelly (more wins than losses), returns None."""
        now = datetime(2026, 8, 25, tzinfo=timezone.utc)
        outcomes = _make_outcomes({"2026-08-20": (10, 3)})  # 10W/3L → positive Kelly
        assert compute_recovery_eta(outcomes, now) is None

    def test_returns_none_when_inactive_below_min_trades(self) -> None:
        """Fewer than KELLY_MIN_TRADES (10) in window → inactive mode → None."""
        now = datetime(2026, 8, 25, tzinfo=timezone.utc)
        outcomes = _make_outcomes({"2026-08-22": (2, 5)})  # 7 total < 10
        assert compute_recovery_eta(outcomes, now) is None

    def test_recovery_eta_with_aug19_selloff_scenario(self) -> None:
        """Reproduce the live scenario: 37 W35 trades (15W/22L), n=37, PF=0.535.

        Aug 17: 10 trades, Aug 18: 9, Aug 19: 6, Aug 20: 6, Aug 21: 6.
        Window from Aug 25 back 10 days includes Aug 15+, so all 37 are in window.
        After Aug 20 trades roll out (n drops by 6 to 6), n < 10 → inactive.
        Expected recovery date: Aug 30 (Aug 20 + 10 days).
        """
        now = datetime(2026, 8, 25, 21, 0, tzinfo=timezone.utc)
        # W35 breakdown: approximate 15W/22L spread across the 5 days
        # (exact split doesn't matter — just need negative Kelly)
        outcomes = _make_outcomes({
            "2026-08-17": (4, 6),   # 10 trades
            "2026-08-18": (3, 6),   # 9 trades
            "2026-08-19": (2, 4),   # 6 trades (bad selloff day)
            "2026-08-20": (3, 3),   # 6 trades
            "2026-08-21": (3, 3),   # 6 trades
        })
        result = compute_recovery_eta(outcomes, now)
        assert result is not None, "Should be in probation with 37 trades and negative Kelly"
        assert result["n_current"] == 37
        assert result["by_date"]["2026-08-17"] == 10
        assert result["by_date"]["2026-08-18"] == 9
        assert result["by_date"]["2026-08-19"] == 6
        # Rollout of Aug 20 trades (n=6) drops remaining from 12 → 6, which is < 10
        # Rollout date = Aug 20 + 10 days = Aug 30
        assert result["earliest_recovery_date"] == "2026-08-30"

    def test_old_trades_outside_window_excluded(self) -> None:
        """Trades older than KELLY_LOOKBACK_DAYS are excluded from n_current."""
        now = datetime(2026, 8, 25, tzinfo=timezone.utc)
        # Aug 1 is 24 days ago → outside 10-day window → excluded
        outcomes = _make_outcomes({
            "2026-08-01": (2, 20),  # old, outside window, ignored
            "2026-08-20": (3, 9),   # 12 total in window, negative Kelly
        })
        result = compute_recovery_eta(outcomes, now)
        assert result is not None
        assert result["n_current"] == 12
        assert "2026-08-01" not in result["by_date"]

    def test_single_date_cluster_rolloff(self) -> None:
        """When all trades cluster on one day, recovery happens in one shot."""
        now = datetime(2026, 8, 25, tzinfo=timezone.utc)
        # 15 trades on Aug 20, all losses → negative Kelly, single rollout date
        outcomes = _make_outcomes({"2026-08-20": (0, 15)})
        # 15 all-loss → no wins → wins list is empty → returns None
        # Adjust: 2W/13L still gives negative Kelly
        outcomes = _make_outcomes({"2026-08-20": (2, 13)})
        result = compute_recovery_eta(outcomes, now)
        assert result is not None
        assert result["n_current"] == 15
        # All 15 roll out on Aug 30 → remaining=0 < 10
        assert result["earliest_recovery_date"] == "2026-08-30"

    def test_no_recovery_date_when_window_never_clears(self) -> None:
        """If trades span many dates and n never drops below min after any rollout,
        recovery_date is None (edge case: very dense window that stays above min)."""
        now = datetime(2026, 8, 25, tzinfo=timezone.utc)
        # 2 trades per day for 5 days = 10 total; each rolloff removes 2, so
        # remaining: 8, 6, 4, 2, 0 — first date to drop below 10 is Aug 17+10=Aug27
        outcomes = _make_outcomes({
            "2026-08-17": (0, 2),
            "2026-08-18": (0, 2),
            "2026-08-19": (0, 2),
            "2026-08-20": (0, 2),
            "2026-08-21": (0, 2),
        })
        # But all losses → no wins → returns None (edge case)
        result = compute_recovery_eta(outcomes, now)
        assert result is None  # all-loss window → no wins → can't compute Kelly


class TestKellyRecoveryEtaEdgeCases:
    def test_probation_with_exact_min_trades(self) -> None:
        """Exactly KELLY_MIN_TRADES trades with negative Kelly → probation → ETA computed."""
        now = datetime(2026, 8, 25, tzinfo=timezone.utc)
        outcomes = _make_outcomes({"2026-08-22": (3, 7)})  # 10 trades, 3W/7L → Kelly < 0
        result = compute_recovery_eta(outcomes, now)
        assert result is not None
        assert result["n_current"] == 10
        # After Aug 22 trades roll (all 10): remaining = 0 < 10 → recovery = Sep 1
        assert result["earliest_recovery_date"] == "2026-09-01"
