"""`record_forced_exit` must do the bookkeeping the force-exit path used to skip.

The watchdog's zombie heal bypasses the exit DECISION logic on purpose — that
logic may be the broken component. But it also skipped every piece of
bookkeeping, which is what made the 2026-10-01/05 failure invisible and left
stale per-ticker state behind for the next entry to inherit.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.agents import signal_loop as sl


def _loop():
    """A SignalLoop stub exposing only what `record_forced_exit` touches."""

    class _L:
        pass

    loop = _L()
    loop._pm = MagicMock()
    pos = MagicMock()
    pos.avg_entry_price = 100.0
    pos.qty = 10.0
    loop._pm._positions = {"COHR": pos}
    loop._pm.close_position = MagicMock(return_value=-50.0)   # 10sh @ 100 → 95
    loop._sizing_mode = True
    loop._consecutive_losses = 0
    loop._entry_prices = {"COHR": 100.0}
    loop._entry_directions = {"COHR": 1}
    loop._entry_dates = {"COHR": None}
    loop._peak_prices = {"COHR": 108.0}
    loop._bars_held = {"COHR": 95}
    loop._reversal_counts = {"COHR": 2}
    loop._hold_extension_count = {"COHR": 1}
    loop._pdt_deferred_logged = {"COHR"}
    loop._sizing_recent_outcomes = []
    loop._kelly_fraction = 0.0
    loop._kelly_min_trades = sl.KELLY_MIN_TRADES
    loop.last_exit_at = None
    loop.written = {}

    async def _write(**kwargs):
        loop.written = kwargs

    loop._write_trade_exit = _write
    loop._clear_sizing_state = sl.SignalLoop._clear_sizing_state.__get__(loop)
    loop._update_kelly = sl.SignalLoop._update_kelly.__get__(loop)
    loop._prune_kelly_window = sl.SignalLoop._prune_kelly_window.__get__(loop)
    loop._kelly_mode = sl.SignalLoop._kelly_mode.__get__(loop)
    loop.record_forced_exit = sl.SignalLoop.record_forced_exit.__get__(loop)
    return loop


def _run(loop, price=95.0):
    return asyncio.run(
        loop.record_forced_exit(
            ticker="COHR",
            fill_price=price,
            exit_qty=10.0,
            exit_time=datetime(2026, 10, 5, 15, 10, tzinfo=timezone.utc),
            exit_reason="watchdog_force_exit",
        )
    )


def test_the_ledger_gets_the_exit_with_its_own_reason():
    loop = _loop()
    _run(loop)
    assert loop.written["exit_reason"] == "watchdog_force_exit"
    assert loop.written["fill_price"] == 95.0
    assert loop.written["pnl"] == -50.0
    assert loop.written["pnl_pct"] == pytest.approx(-50.0 / 1000.0)


def test_stale_per_ticker_state_is_cleared():
    """The live hazard: a surviving peak and bar count poison the NEXT entry.

    Before this, `_bars_held` and `_peak_prices` outlived the close, so a
    re-entry inherited the previous trade's trailing peak — and was measured
    against a bar count it never accumulated.
    """
    loop = _loop()
    _run(loop)
    assert "COHR" not in loop._bars_held
    assert "COHR" not in loop._peak_prices
    assert "COHR" not in loop._entry_prices
    assert "COHR" not in loop._reversal_counts
    assert "COHR" not in loop._hold_extension_count


def test_a_forced_loss_counts_toward_the_consecutive_loss_breaker():
    """Forced losses were invisible to this circuit breaker."""
    loop = _loop()
    _run(loop)
    assert loop._consecutive_losses == 1


def test_a_forced_win_resets_the_loss_streak():
    loop = _loop()
    loop._pm.close_position = MagicMock(return_value=+25.0)
    loop._consecutive_losses = 3
    _run(loop, price=102.5)
    assert loop._consecutive_losses == 0


def test_the_outcome_reaches_the_kelly_window():
    loop = _loop()
    _run(loop)
    assert len(loop._sizing_recent_outcomes) == 1
    ts, pct = loop._sizing_recent_outcomes[0]
    assert ts.tzinfo is not None          # must be tz-aware or pruning breaks
    assert pct == pytest.approx(-0.05)


def test_the_watchdog_beacon_is_refreshed():
    """`last_exit_at` is how the watchdog knows the exit path is alive.

    A force-exit that left it stale would make the watchdog accuse the exit path
    of being dead on the strength of its own intervention.
    """
    loop = _loop()
    _run(loop)
    assert loop.last_exit_at is not None


def test_position_is_closed_at_the_fill_price():
    loop = _loop()
    _run(loop)
    loop._pm.close_position.assert_called_once_with("COHR", 95.0)
    loop._pm.record_return.assert_called_once()
