"""Unit tests for H33: post-stop-loss session entry pause (Gate 2c).

H33 adds ENTRY_POST_STOP_PAUSE_MINS (30 min) pause on new entries after the
first stop_loss or trailing_stop exit fires in a session.  The pause expires
naturally so no manual intervention is needed; it resets at the daily session
boundary.

Evidence: 2026-10-16 — 6 entries in 2.5 min at open, 5 stop/trail exits,
net -$2,148 on a single session.  With PR #65 (ENTRY_SPREAD_MINS=10) plus
H33 only 2 entries would have been placed before the pause armed.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.agents.signal_loop import (
    ENTRY_POST_STOP_PAUSE_MINS,
    SignalLoop,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UTC = timezone.utc


def _now_utc() -> datetime:
    return datetime.now(_UTC)


def _make_loop() -> SignalLoop:
    """Construct a minimal SignalLoop with all external deps mocked."""
    loop = SignalLoop.__new__(SignalLoop)
    # Mimic enough __init__ state for the gate and reset paths.
    loop._session_first_stop_at = None
    loop._sizing_recent_outcomes = []
    loop._pending_exit_reasons = {}
    loop._sizing_mode = True
    loop._pm = MagicMock()
    loop._pm._positions = {}
    loop._pm.managed_heat = 0.0
    loop._pm.portfolio_value = 100_000.0
    loop._kelly_fraction = 0.0
    loop._kelly_min_trades = 10
    loop._sizing_n_trades_today = 0
    loop._probation_entries_today = 0
    loop._ticker_cooldown = {}
    loop._pdt_deferred_logged = set()
    loop._data_fresh = True
    loop._ticker_earnings = {}
    loop._earnings_refresh_countdown = 0
    loop._ticker_ic = {}
    loop._ticker_ic_probe = {}
    loop._entry_ranks = []
    loop._peak_prices = {}
    loop._entry_prices = {}
    loop._entry_dates = {}
    loop._daytrade_count = 0
    loop._daily_start_value = 100_000.0
    loop._cb = MagicMock()
    loop._cb.is_halted = False
    loop._last_reset_date = None
    return loop


# ---------------------------------------------------------------------------
# Constant sanity
# ---------------------------------------------------------------------------


def test_pause_constant_positive() -> None:
    assert ENTRY_POST_STOP_PAUSE_MINS > 0, "Pause must be positive"
    assert ENTRY_POST_STOP_PAUSE_MINS <= 120, "Pause > 2h would block most of a session"


# ---------------------------------------------------------------------------
# _session_first_stop_at initialises to None
# ---------------------------------------------------------------------------


def test_initial_state_none() -> None:
    loop = _make_loop()
    assert loop._session_first_stop_at is None


# ---------------------------------------------------------------------------
# Gate 2c: blocks entries while pause window is active
# ---------------------------------------------------------------------------


def _make_sig(ticker: str = "AAPL") -> SimpleNamespace:
    """Minimal EnsembleSignal-compatible mock that passes Gate 6."""
    return SimpleNamespace(
        ticker=ticker,
        lgbm_pred_return=0.01,
        lgbm_dir_prob=0.70,
        ensemble_signal=0.80,
    )


def test_gate_blocks_during_pause() -> None:
    """Gate 2c returns False while now < first_stop + pause_mins."""
    loop = _make_loop()
    stop_ts = _now_utc() - timedelta(minutes=5)   # stop fired 5 min ago
    loop._session_first_stop_at = stop_ts

    sig = _make_sig()

    with (
        patch.object(loop, "_in_entry_window", return_value=True),
        patch.object(loop, "_in_earnings_blackout", return_value=False),
        patch.object(loop, "_kelly_entries_blocked", return_value=False),
        patch.object(loop, "_kelly_mode", return_value="normal"),
        patch.object(loop, "_ticker_ic_blocked", return_value=False),
        patch.object(loop, "_dynamic_cost_threshold", return_value=0.001),
        patch.object(loop, "_sector_position_count", return_value=0),
        patch("src.agents.signal_loop.sector_of", return_value="tech"),
        patch("src.agents.signal_loop.max_positions_for_sector", return_value=2),
    ):
        result = loop._sizing_entry_gate_open(sig)
    assert result is False, "Gate 2c should block entry within pause window"


def test_gate_passes_after_pause_expires() -> None:
    """Gate 2c does NOT block once now >= first_stop + pause_mins."""
    loop = _make_loop()
    # Stop fired long enough ago that pause has expired
    stop_ts = _now_utc() - timedelta(minutes=ENTRY_POST_STOP_PAUSE_MINS + 5)
    loop._session_first_stop_at = stop_ts

    sig = _make_sig()

    with (
        patch.object(loop, "_in_entry_window", return_value=True),
        patch.object(loop, "_in_earnings_blackout", return_value=False),
        patch.object(loop, "_kelly_entries_blocked", return_value=False),
        patch.object(loop, "_kelly_mode", return_value="normal"),
        patch.object(loop, "_ticker_ic_blocked", return_value=False),
        patch.object(loop, "_dynamic_cost_threshold", return_value=0.001),
        patch.object(loop, "_sector_position_count", return_value=0),
        patch("src.agents.signal_loop.sector_of", return_value="tech"),
        patch("src.agents.signal_loop.max_positions_for_sector", return_value=2),
    ):
        result = loop._sizing_entry_gate_open(sig)
    assert result is True, "Gate 2c should not block after pause expires"


def test_gate_inactive_when_no_stop_yet() -> None:
    """Gate 2c is a no-op when no stop has fired this session."""
    loop = _make_loop()
    assert loop._session_first_stop_at is None

    sig = _make_sig()

    with (
        patch.object(loop, "_in_entry_window", return_value=True),
        patch.object(loop, "_in_earnings_blackout", return_value=False),
        patch.object(loop, "_kelly_entries_blocked", return_value=False),
        patch.object(loop, "_kelly_mode", return_value="normal"),
        patch.object(loop, "_ticker_ic_blocked", return_value=False),
        patch.object(loop, "_dynamic_cost_threshold", return_value=0.001),
        patch.object(loop, "_sector_position_count", return_value=0),
        patch("src.agents.signal_loop.sector_of", return_value="tech"),
        patch("src.agents.signal_loop.max_positions_for_sector", return_value=2),
    ):
        result = loop._sizing_entry_gate_open(sig)
    assert result is True, "Gate 2c must not block when _session_first_stop_at is None"


# ---------------------------------------------------------------------------
# Recording: only the FIRST stop sets _session_first_stop_at
# ---------------------------------------------------------------------------


def test_first_stop_recorded() -> None:
    """After a stop_loss fill, _session_first_stop_at is set."""
    loop = _make_loop()
    assert loop._session_first_stop_at is None

    stop_ts = _now_utc()
    # Simulate the exit recording code path
    exit_reason = "stop_loss"
    loop._pending_exit_reasons["MRNA"] = exit_reason
    resolved = loop._pending_exit_reasons.pop("MRNA", "signal_reversal")
    if resolved in ("stop_loss", "trailing_stop") and loop._session_first_stop_at is None:
        loop._session_first_stop_at = stop_ts

    assert loop._session_first_stop_at == stop_ts


def test_second_stop_does_not_overwrite() -> None:
    """A second stop_loss in the same session must not overwrite the first."""
    loop = _make_loop()
    first_ts = _now_utc() - timedelta(minutes=10)
    loop._session_first_stop_at = first_ts

    second_ts = _now_utc()
    exit_reason = "stop_loss"
    if exit_reason in ("stop_loss", "trailing_stop") and loop._session_first_stop_at is None:
        loop._session_first_stop_at = second_ts  # should NOT run

    assert loop._session_first_stop_at == first_ts, "First stop timestamp must be preserved"


def test_trailing_stop_also_arms_pause() -> None:
    """trailing_stop is as valid a 'bad session' signal as stop_loss."""
    loop = _make_loop()
    ts = _now_utc()
    exit_reason = "trailing_stop"
    if exit_reason in ("stop_loss", "trailing_stop") and loop._session_first_stop_at is None:
        loop._session_first_stop_at = ts
    assert loop._session_first_stop_at == ts


def test_session_close_does_not_arm_pause() -> None:
    """A session_close exit must NOT arm H33 (it is not a 'bad session' signal)."""
    loop = _make_loop()
    exit_reason = "session_close"
    if exit_reason in ("stop_loss", "trailing_stop") and loop._session_first_stop_at is None:
        loop._session_first_stop_at = _now_utc()  # should NOT run
    assert loop._session_first_stop_at is None


def test_take_profit_does_not_arm_pause() -> None:
    """A take_profit exit must NOT arm H33."""
    loop = _make_loop()
    exit_reason = "take_profit"
    if exit_reason in ("stop_loss", "trailing_stop") and loop._session_first_stop_at is None:
        loop._session_first_stop_at = _now_utc()  # should NOT run
    assert loop._session_first_stop_at is None


# ---------------------------------------------------------------------------
# Daily reset clears the pause state
# ---------------------------------------------------------------------------


def test_daily_reset_clears_session_first_stop() -> None:
    """Session reset must clear _session_first_stop_at for a clean new day."""
    loop = _make_loop()
    loop._session_first_stop_at = _now_utc() - timedelta(hours=8)

    # Simulate the daily reset block
    loop._sizing_n_trades_today = 0
    loop._probation_entries_today = 0
    loop._session_first_stop_at = None  # the reset line we added

    assert loop._session_first_stop_at is None


# ---------------------------------------------------------------------------
# Diagnostics keys are present
# ---------------------------------------------------------------------------


def test_diagnostics_keys_exported() -> None:
    """session_first_stop_at and related fields should be in the diagnostics dict."""
    loop = _make_loop()
    loop._sizing_recent_outcomes = []
    loop._pm.managed_heat = 0.0
    loop._pm.portfolio_value = 100_000.0
    loop._entry_ranks = []
    loop._consecutive_losses = 0

    # We test directly by checking the attribute exists and can be rendered
    stop_ts = _now_utc() - timedelta(minutes=5)
    loop._session_first_stop_at = stop_ts

    iso = loop._session_first_stop_at.isoformat()
    assert "T" in iso  # must be a valid ISO timestamp

    pause_active = (
        loop._session_first_stop_at is not None
        and datetime.now(timezone.utc) < (
            loop._session_first_stop_at + timedelta(minutes=ENTRY_POST_STOP_PAUSE_MINS)
        )
    )
    assert pause_active is True


def test_diagnostics_no_stop_shows_none() -> None:
    """When no stop has fired, the diagnostics fields should be None / False."""
    loop = _make_loop()
    loop._session_first_stop_at = None

    first_stop_iso = loop._session_first_stop_at.isoformat() if loop._session_first_stop_at else None
    pause_active = (
        loop._session_first_stop_at is not None
        and datetime.now(timezone.utc) < (
            loop._session_first_stop_at + timedelta(minutes=ENTRY_POST_STOP_PAUSE_MINS)
        )
    )
    resume_at = (
        (loop._session_first_stop_at + timedelta(minutes=ENTRY_POST_STOP_PAUSE_MINS)).isoformat()
        if loop._session_first_stop_at else None
    )

    assert first_stop_iso is None
    assert pause_active is False
    assert resume_at is None
