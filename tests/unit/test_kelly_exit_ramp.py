"""Unit tests for H34 — Kelly probation-exit ramp (soft landing).

When Kelly transitions from probation/hard-block back to "normal" (fraction
turns positive), the first ET session is capped at KELLY_EXIT_RAMP_FACTOR of
normal notional. The second session has no restriction.

Motivation: Oct 16 2026 bad batch showed 6 full-size entries firing immediately
after a constrained period. With H24/H25 structural guards unmerged, any clift
transition from probation→normal can fire a large batch without warning. H34
closes this gap with a lightweight one-session ramp.
"""

from __future__ import annotations

from datetime import datetime, date, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.agents.signal_loop import (
    KELLY_EXIT_RAMP_FACTOR,
    KELLY_HARD_BLOCK_THRESHOLD,
    KELLY_MIN_TRADES,
    SignalLoop,
)
from src.execution.position_manager import PositionManager
from src.risk.circuit_breakers import CircuitBreakers


def _make_loop() -> SignalLoop:
    loop = SignalLoop(
        universe=["AAPL", "XOM"],
        ensemble=MagicMock(),
        alpaca=MagicMock(),
        circuit_breakers=CircuitBreakers(),
        pos_manager=PositionManager(initial_portfolio=100_000.0),
        session_factory=MagicMock(),
        feature_cols=[f"feat_{i}" for i in range(30)],
    )
    loop._in_entry_window = lambda: True
    loop._data_fresh = True
    return loop


def _seed_losing(loop: SignalLoop, age_days: float = 1.0) -> None:
    stamp = datetime.now(timezone.utc) - timedelta(days=age_days)
    loop._sizing_recent_outcomes = [(stamp, o) for o in ([0.005] * 3 + [-0.03] * 9)]


def _seed_winning(loop: SignalLoop, age_days: float = 1.0) -> None:
    stamp = datetime.now(timezone.utc) - timedelta(days=age_days)
    loop._sizing_recent_outcomes = [(stamp, o) for o in ([0.02] * 8 + [-0.004] * 4)]


# ── Constant sanity ─────────────────────────────────────────────────────────

def test_exit_ramp_factor_is_between_zero_and_one():
    """Ramp must be strictly conservative: less than full size, positive."""
    assert 0.0 < KELLY_EXIT_RAMP_FACTOR < 1.0


# ── Ramp not active in steady-state ─────────────────────────────────────────

def test_ramp_not_active_on_fresh_loop():
    """A freshly constructed loop has no ramp state."""
    loop = _make_loop()
    assert loop._kelly_exit_ramp_active() is False


def test_ramp_not_active_when_kelly_always_normal():
    """No ramp fires if Kelly was never in a non-normal state."""
    loop = _make_loop()
    _seed_winning(loop)
    loop._update_kelly()
    assert loop._kelly_mode() == "normal"
    assert loop._kelly_exit_ramp_active() is False


def test_ramp_not_active_in_probation():
    """Ramp is irrelevant while still in probation — probation cap applies."""
    loop = _make_loop()
    _seed_losing(loop)
    loop._update_kelly()
    assert loop._kelly_mode() == "probation"
    assert loop._kelly_exit_ramp_active() is False


# ── Ramp arms on probation→normal transition ─────────────────────────────────

def test_ramp_arms_when_probation_flips_to_normal():
    """After a losing then winning window, the ramp activates for today."""
    loop = _make_loop()
    # Phase 1: push into probation
    _seed_losing(loop)
    loop._update_kelly()
    assert loop._kelly_mode() == "probation"

    # Phase 2: window recovers with winning trades
    _seed_winning(loop)
    loop._update_kelly()
    assert loop._kelly_mode() == "normal"
    assert loop._kelly_exit_ramp_active() is True


def test_ramp_session_matches_today_et():
    """The recorded ramp session is today's ET date."""
    loop = _make_loop()
    _seed_losing(loop)
    loop._update_kelly()
    _seed_winning(loop)

    from zoneinfo import ZoneInfo
    today_et = datetime.now(ZoneInfo("America/New_York")).date()
    loop._update_kelly()
    assert loop._kelly_first_normal_date == today_et


# ── Ramp clears the next session ─────────────────────────────────────────────

def test_ramp_inactive_when_first_normal_date_is_old():
    """The ramp fires ONLY on the transition day; a prior date means it is gone."""
    loop = _make_loop()
    _seed_losing(loop)
    loop._update_kelly()
    _seed_winning(loop)
    loop._update_kelly()
    assert loop._kelly_exit_ramp_active() is True

    # Simulate the ramp session being two days ago — never today regardless of timezone.
    two_days_ago = date.today() - timedelta(days=2)
    loop._kelly_first_normal_date = two_days_ago
    assert loop._kelly_exit_ramp_active() is False


# ── Ramp does not fire from inactive→normal (H24 owns that) ─────────────────

def test_ramp_does_not_arm_when_transitioning_from_inactive():
    """Inactive→normal is handled by H24 guards; H34 only arms on probation→normal."""
    loop = _make_loop()
    # inactive: window is empty
    assert loop._kelly_mode() == "inactive"
    assert loop._kelly_prev_mode == "inactive"
    _seed_winning(loop)
    loop._update_kelly()
    # This is inactive→normal, not probation→normal
    # H34 should NOT arm — prev_mode was "inactive", not "probation"
    assert loop._kelly_first_normal_date is None
    assert loop._kelly_exit_ramp_active() is False


# ── Diagnostics expose ramp state ───────────────────────────────────────────

def test_diagnostics_includes_ramp_fields():
    """Both ramp fields must appear in the get_portfolio_summary() payload."""
    loop = _make_loop()
    diag = loop.get_portfolio_summary()
    assert "kelly_exit_ramp_active" in diag
    assert "kelly_exit_ramp_session" in diag


def test_diagnostics_ramp_active_false_by_default():
    loop = _make_loop()
    diag = loop.get_portfolio_summary()
    assert diag["kelly_exit_ramp_active"] is False
    assert diag["kelly_exit_ramp_session"] is None


def test_diagnostics_ramp_active_true_after_transition():
    loop = _make_loop()
    _seed_losing(loop)
    loop._update_kelly()
    _seed_winning(loop)
    loop._update_kelly()
    diag = loop.get_portfolio_summary()
    assert diag["kelly_exit_ramp_active"] is True
    assert diag["kelly_exit_ramp_session"] is not None
