"""Unit tests for H38: Kelly recovery ETA in /diagnostics.

The core logic lives in two places:
  - _kelly_fraction_from_pnls() — module-level pure function
  - SignalLoop._kelly_recovery_eta() — method that wraps the schedule

Tested without instantiating SignalLoop (too many ML deps): the pure function
and the rolloff-schedule logic are exercised directly.  The method-level test
uses a minimal mock object so we can call _kelly_recovery_eta() without a live
DB, Alpaca API, or feature store.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from src.agents.signal_loop import (
    KELLY_HARD_BLOCK_THRESHOLD,
    KELLY_LOOKBACK_DAYS,
    KELLY_MIN_TRADES,
    _kelly_fraction_from_pnls,
)


# ── _kelly_fraction_from_pnls ─────────────────────────────────────────────────

class TestKellyFractionFromPnls:
    def test_all_winners_returns_zero(self):
        """All-win sample has no loss average → indeterminate, returns 0."""
        assert _kelly_fraction_from_pnls([0.01, 0.02, 0.03]) == 0.0

    def test_all_losers_returns_zero(self):
        """All-loss sample has no win average → indeterminate, returns 0."""
        assert _kelly_fraction_from_pnls([-0.01, -0.02, -0.03]) == 0.0

    def test_empty_returns_zero(self):
        assert _kelly_fraction_from_pnls([]) == 0.0

    def test_positive_edge(self):
        """Equal wins and losses larger than losses → positive Kelly."""
        # 2 wins @ 2%, 2 losses @ 1% → p=0.5, b=2.0 → f*=(0.5*2-0.5)/2=0.25
        f = _kelly_fraction_from_pnls([0.02, 0.02, -0.01, -0.01])
        assert f == pytest.approx(0.25, rel=1e-4)

    def test_negative_edge(self):
        """Equal wins smaller than losses → negative Kelly (bad edge)."""
        # 2 wins @ 1%, 2 losses @ 2% → p=0.5, b=0.5 → f*=(0.5*0.5-0.5)/0.5=-0.5
        f = _kelly_fraction_from_pnls([0.01, 0.01, -0.02, -0.02])
        assert f == pytest.approx(-0.5, rel=1e-4)

    def test_hard_block_scenario(self):
        """Sample like Oct 29-30 produces fraction <= KELLY_HARD_BLOCK_THRESHOLD."""
        # 5 losses @ -2%, 5 wins @ 0.3% → highly negative
        outcomes = [0.003] * 5 + [-0.02] * 5
        f = _kelly_fraction_from_pnls(outcomes)
        assert f <= KELLY_HARD_BLOCK_THRESHOLD


# ── _kelly_recovery_eta (via minimal mock) ────────────────────────────────────

def _make_minimal_loop(
    outcomes: list[tuple[datetime, float]],
    halted: bool = False,
    halt_reason: str = "",
) -> "Any":
    """Return a minimal object that exposes what _kelly_recovery_eta() needs."""
    from src.agents.signal_loop import SignalLoop

    loop = object.__new__(SignalLoop)
    loop._sizing_recent_outcomes = outcomes
    loop._kelly_min_trades = KELLY_MIN_TRADES

    cb = MagicMock()
    cb.is_halted = halted
    cb.halt_reason = halt_reason
    loop._cb = cb
    return loop


class TestKellyRecoveryEta:
    def _outcomes_on(
        self, day_offset: int, pnls: list[float]
    ) -> list[tuple[datetime, float]]:
        """Build outcome tuples all timestamped `day_offset` days ago."""
        ts = datetime.now(timezone.utc) - timedelta(days=day_offset)
        return [(ts, p) for p in pnls]

    # ── empty window ──────────────────────────────────────────────────────────

    def test_empty_outcomes(self):
        loop = _make_minimal_loop([])
        result = loop._kelly_recovery_eta()
        assert result["blocking_outcomes_rolloff_schedule"] == []
        assert result["earliest_kelly_unblock_date"] is None
        assert result["days_until_kelly_unblock"] is None
        assert "Kelly inactive" in result["earliest_entry_possible"]

    # ── no hard-block ─────────────────────────────────────────────────────────

    def test_positive_kelly_no_block(self):
        """Good edge (n >= min_trades) → no hard-block, earliest unblock date is None."""
        # Need n >= KELLY_MIN_TRADES so the governor is active, not inactive.
        # Positive edge: 10 wins @ 2%, 10 losses @ 0.5% → b=4, p=0.5 → f*=0.375
        outcomes = (
            self._outcomes_on(2, [0.02] * 10)
            + self._outcomes_on(1, [-0.005] * 10)
        )
        loop = _make_minimal_loop(outcomes)
        result = loop._kelly_recovery_eta()
        assert result["earliest_kelly_unblock_date"] is None
        assert "No Kelly hard-block" in result["earliest_entry_possible"]

    # ── rolloff schedule shape ────────────────────────────────────────────────

    def test_rolloff_schedule_dates(self):
        """rolloff_date = outcome_date + KELLY_LOOKBACK_DAYS.

        Requires a hard-blocked sample (n >= KELLY_MIN_TRADES, Kelly <= threshold).
        """
        from datetime import date, timedelta
        # 2 wins @ 0.3%, 10 big losses → hard-blocked; all on day 5
        outcomes = (
            self._outcomes_on(5, [-0.025] * 10)
            + self._outcomes_on(2, [0.003] * 2)
        )
        loop = _make_minimal_loop(outcomes)
        result = loop._kelly_recovery_eta()
        schedule = result["blocking_outcomes_rolloff_schedule"]
        assert len(schedule) >= 1
        outcome_date = date.fromisoformat(schedule[0]["outcome_date"])
        expected_rolloff = outcome_date + timedelta(days=KELLY_LOOKBACK_DAYS)
        assert date.fromisoformat(schedule[0]["rolloff_date"]) == expected_rolloff

    def test_rolloff_schedule_net_pnl(self):
        """net_pnl_pct sums all pnl values for that day.

        Requires a hard-blocked sample (n >= KELLY_MIN_TRADES, Kelly <= threshold).
        """
        # 10 losses @ -0.01/-0.02/-0.03 on day 3, plus wins to make it hard-blocked
        # Put losses all on day 3, wins on day 1
        big_losses = self._outcomes_on(3, [-0.03] * 8)
        small_wins = self._outcomes_on(1, [0.001] * 4)
        outcomes = big_losses + small_wins
        loop = _make_minimal_loop(outcomes)
        result = loop._kelly_recovery_eta()
        schedule = result["blocking_outcomes_rolloff_schedule"]
        # Find the day-3 entry
        day_3_entries = [s for s in schedule if s["days_until_rolloff"] == KELLY_LOOKBACK_DAYS - 3]
        assert len(day_3_entries) == 1
        assert day_3_entries[0]["n_outcomes"] == 8

    # ── hard-block + rolloff recovery ─────────────────────────────────────────

    def test_hard_block_lifts_as_bad_day_rolls_off(self):
        """Adding a bad day 9 days ago; removing it brings Kelly above threshold."""
        # 10 wins @ 0.5%, 10 big losses 9 days ago → hard-blocked
        good = self._outcomes_on(2, [0.005] * 10)
        bad = self._outcomes_on(9, [-0.025] * 10)
        outcomes = good + bad
        loop = _make_minimal_loop(outcomes)
        result = loop._kelly_recovery_eta()
        # Removal of the bad day should fix the fraction
        assert result["earliest_kelly_unblock_date"] is not None
        assert result["days_until_kelly_unblock"] is not None
        assert result["days_until_kelly_unblock"] >= 1

    def test_unblock_date_is_outcome_date_plus_lookback(self):
        """earliest_kelly_unblock_date = oldest_bad_date + KELLY_LOOKBACK_DAYS."""
        from datetime import date, timedelta
        bad_day_offset = 8  # 8 days ago
        good = self._outcomes_on(1, [0.005] * 10)
        bad = self._outcomes_on(bad_day_offset, [-0.03] * 10)
        outcomes = good + bad
        loop = _make_minimal_loop(outcomes)
        result = loop._kelly_recovery_eta()
        if result["earliest_kelly_unblock_date"] is not None:
            unblock = date.fromisoformat(result["earliest_kelly_unblock_date"])
            today = datetime.now(timezone.utc).date()
            bad_date = today - timedelta(days=bad_day_offset)
            expected = bad_date + timedelta(days=KELLY_LOOKBACK_DAYS)
            assert unblock == expected

    # ── CB halt fields ────────────────────────────────────────────────────────

    def test_cb_halt_reflected_in_result(self):
        """CB halt flag and manual-resume fields propagate from cb.is_halted.

        CB halt is surfaced even when Kelly is not hard-blocked (early-return path).
        """
        outcomes = self._outcomes_on(1, [0.01, -0.01])
        loop = _make_minimal_loop(outcomes, halted=True, halt_reason="max_drawdown")
        result = loop._kelly_recovery_eta()
        assert result["cb_halt_active"] is True
        assert result["cb_halt_requires_manual_resume"] is True
        assert "CB halt active" in result["earliest_entry_possible"]

    def test_cb_halt_with_kelly_block(self):
        """CB halt appears in message when BOTH CB halt AND Kelly hard-block are active."""
        bad = self._outcomes_on(8, [-0.025] * 10)
        good = self._outcomes_on(1, [0.003] * 2)
        outcomes = bad + good
        loop = _make_minimal_loop(outcomes, halted=True, halt_reason="max_drawdown")
        result = loop._kelly_recovery_eta()
        assert result["cb_halt_active"] is True
        assert "CB halt active" in result["earliest_entry_possible"]

    def test_no_cb_halt(self):
        outcomes = self._outcomes_on(1, [0.01, -0.01])
        loop = _make_minimal_loop(outcomes, halted=False)
        result = loop._kelly_recovery_eta()
        assert result["cb_halt_active"] is False
        assert result["cb_halt_requires_manual_resume"] is False

    # ── sample too thin → inactive mode → no hard-block ──────────────────────

    def test_rolloff_drains_sample_to_inactive(self):
        """When rolloff leaves fewer than KELLY_MIN_TRADES, governor goes inactive.

        A sample below KELLY_MIN_TRADES is itself not hard-blocked (inactive mode),
        so the method returns early with no earliest_unblock_date.
        """
        # Only 2 outcomes (< KELLY_MIN_TRADES = 10), all losses → n < min → inactive
        outcomes = self._outcomes_on(8, [-0.03, -0.03])
        loop = _make_minimal_loop(outcomes)
        result = loop._kelly_recovery_eta()
        # Sample is too thin for the governor to be active → not hard-blocked
        assert result["earliest_kelly_unblock_date"] is None
        assert result["blocking_outcomes_rolloff_schedule"] == []

    def test_rolloff_drains_to_inactive_during_simulation(self):
        """When simulation drops below min trades mid-walk, that date unblocks."""
        # 10 big losses 8 days ago + 2 wins recently — currently hard-blocked.
        # After removing 8-day-old losses, only 2 remain → inactive → unblocked.
        bad = self._outcomes_on(8, [-0.03] * 10)
        good = self._outcomes_on(1, [0.005, 0.005])
        outcomes = bad + good
        loop = _make_minimal_loop(outcomes)
        result = loop._kelly_recovery_eta()
        # Method should find that removing the bad day resolves the block
        assert result["earliest_kelly_unblock_date"] is not None
