"""Unit tests for H34 — extended post-stop-loss ticker cooldown.

After a stop_loss exit, the ticker is barred from re-entry for a full trading
day (SIZING_STOP_LOSS_COOLDOWN_BARS = 390 bars) rather than the standard
1-hour churn guard (SIZING_TICKER_COOLDOWN_BARS = 60 bars).

Evidence: NOW triggered stop_loss on Oct 16 ($-395) and again on Oct 29
($-576) in the same M3 window — the 1-hour cooldown cleared long before the
next session opened, allowing a repeat entry on a chronically failing ticker.

These are money-handling constants and are covered as such.
"""

from __future__ import annotations

import pytest

from src.agents.signal_loop import (
    SIZING_STOP_LOSS_COOLDOWN_BARS,
    SIZING_TICKER_COOLDOWN_BARS,
)

TRADING_BARS_PER_DAY = 390  # 6.5 h × 60 min


class TestH34StopLossCooldownConstants:
    def test_stop_loss_cooldown_longer_than_standard(self):
        """Stop-loss cooldown must be strictly longer than the churn guard."""
        assert SIZING_STOP_LOSS_COOLDOWN_BARS > SIZING_TICKER_COOLDOWN_BARS

    def test_stop_loss_cooldown_at_least_one_trading_day(self):
        """One full trading day prevents same-next-session re-entry."""
        assert SIZING_STOP_LOSS_COOLDOWN_BARS >= TRADING_BARS_PER_DAY

    def test_stop_loss_cooldown_not_excessive(self):
        """Cap at 5 trading days to avoid starving the Kelly window."""
        assert SIZING_STOP_LOSS_COOLDOWN_BARS <= 5 * TRADING_BARS_PER_DAY

    def test_standard_cooldown_unchanged(self):
        """Churn guard must remain 60 bars (1 hour)."""
        assert SIZING_TICKER_COOLDOWN_BARS == 60


def _pick_cooldown(exit_reason: str) -> int:
    """Mirror of the branching logic in signal_loop._act_on_signal."""
    return (
        SIZING_STOP_LOSS_COOLDOWN_BARS
        if exit_reason == "stop_loss"
        else SIZING_TICKER_COOLDOWN_BARS
    )


class TestH34CooldownBranchingLogic:
    def test_stop_loss_uses_extended_cooldown(self):
        assert _pick_cooldown("stop_loss") == SIZING_STOP_LOSS_COOLDOWN_BARS

    def test_session_close_uses_standard_cooldown(self):
        assert _pick_cooldown("session_close") == SIZING_TICKER_COOLDOWN_BARS

    def test_trailing_stop_uses_standard_cooldown(self):
        assert _pick_cooldown("trailing_stop") == SIZING_TICKER_COOLDOWN_BARS

    def test_take_profit_uses_standard_cooldown(self):
        assert _pick_cooldown("take_profit") == SIZING_TICKER_COOLDOWN_BARS

    def test_max_hold_uses_standard_cooldown(self):
        assert _pick_cooldown("max_hold") == SIZING_TICKER_COOLDOWN_BARS

    @pytest.mark.parametrize("reason", ["stop_loss", "session_close", "trailing_stop",
                                         "take_profit", "max_hold"])
    def test_all_reasons_return_positive_cooldown(self, reason: str):
        assert _pick_cooldown(reason) > 0
