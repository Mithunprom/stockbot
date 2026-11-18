"""DRAWDOWN_PEAK_EPOCH — an audited, bounded re-anchor of the high-water mark.

Owner-authorised 2026-11-17. The drawdown breaker halted correctly at 11.01%
against its 8% limit on 2026-11-02, and that halt cannot be traded out of:
clearing it needs equity at $99,901, which needs trading, which the halt forbids.
A resume on 2026-11-16 was re-halted by the next risk evaluation without placing
a single order.

The property that makes this acceptable rather than reckless is the set of tests
below the first section: the 8% limit is untouched, the breaker still fires on an
8% fall from the epoch forward, and the default — unset — is the all-time peak.
An epoch that disabled the breaker would be removing a risk control, which
CLAUDE.md forbids; this moves where the measurement STARTS, nothing else.
"""
from __future__ import annotations

import inspect
from datetime import datetime, timezone

import pytest

from src.agents import signal_loop as sl
from src.risk import circuit_breakers as cb


# ── Parsing, and the safe default ────────────────────────────────────────────

def test_unset_means_all_time_peak(monkeypatch):
    """The default must be the conservative behaviour, not the bypass."""
    monkeypatch.delenv("DRAWDOWN_PEAK_EPOCH", raising=False)
    assert sl.SignalLoop._drawdown_peak_epoch() is None


def test_empty_and_whitespace_are_treated_as_unset(monkeypatch):
    for raw in ("", "   "):
        monkeypatch.setenv("DRAWDOWN_PEAK_EPOCH", raw)
        assert sl.SignalLoop._drawdown_peak_epoch() is None


def test_unparseable_fails_open_to_all_time(monkeypatch):
    """A typo must not silently re-anchor the peak to something arbitrary."""
    monkeypatch.setenv("DRAWDOWN_PEAK_EPOCH", "not-a-date")
    assert sl.SignalLoop._drawdown_peak_epoch() is None


def test_naive_timestamps_are_read_as_utc(monkeypatch):
    monkeypatch.setenv("DRAWDOWN_PEAK_EPOCH", "2026-11-17")
    got = sl.SignalLoop._drawdown_peak_epoch()
    assert got == datetime(2026, 11, 17, tzinfo=timezone.utc)


def test_tz_aware_is_preserved(monkeypatch):
    monkeypatch.setenv("DRAWDOWN_PEAK_EPOCH", "2026-11-17T00:00:00+00:00")
    assert sl.SignalLoop._drawdown_peak_epoch().tzinfo is not None


# ── The control itself must be untouched ─────────────────────────────────────

def test_the_eight_percent_limit_is_not_changed():
    """The epoch moves the measurement START, never the threshold."""
    src = inspect.getsource(cb.CircuitBreakers._check_max_drawdown)
    assert "threshold = 0.08" in src
    assert "DRAWDOWN_PEAK_EPOCH" not in src
    assert "epoch" not in src.lower()


def test_the_breaker_still_halts_on_an_eight_percent_fall():
    """A fall from a RE-ANCHORED peak must still halt — the control survives."""
    breaker = cb.CircuitBreakers(pipeline_id="test")
    peak = 96_638.04                       # peak re-anchored to the epoch value
    state = cb.RiskState(
        portfolio_value=peak * 0.90,       # 10% below the NEW peak
        peak_portfolio=peak,
        daily_start_value=peak,
        vix=20.0,
        consecutive_losses=0,
    )
    # drawdown_pct is derived, so this exercises the real arithmetic.
    assert state.drawdown_pct == pytest.approx(0.10)
    res = breaker._check_max_drawdown(state)
    assert res.triggered is True
    assert breaker.is_halted is True


def test_resume_still_requires_human_authorization():
    """The epoch must not become a route around the human-resume rule."""
    breaker = cb.CircuitBreakers(pipeline_id="test")
    breaker._halt("max_drawdown", "test")
    with pytest.raises(ValueError):
        breaker.resume_trading(authorized_by="")
    assert breaker.is_halted is True


# ── It must actually defeat BOTH sources of the stale peak ───────────────────

def test_the_stored_snapshot_peak_is_also_discarded():
    """Filtering only the broker lookup would leave the stored peak to win.

    `_restore_risk_state` max()es (current, broker, stored). The stored value is
    $108,588 from 2026-04-03; if it survived the epoch it would re-halt
    immediately and the re-anchor would appear to do nothing.
    """
    src = inspect.getsource(sl.SignalLoop._restore_risk_state)
    assert "_drawdown_peak_epoch()" in src
    assert "epoch is None" in src
    i_guard = src.index("_drawdown_peak_epoch()")
    i_max = src.index("max(peak_candidates)")
    assert i_guard < i_max, "epoch must be consulted before the peak is chosen"


def test_the_broker_lookup_filters_by_timestamp():
    """Equity points before the epoch must be excluded from the max."""
    src = inspect.getsource(sl.SignalLoop._peak_from_broker_history)
    assert "_drawdown_peak_epoch()" in src
    assert "cutoff" in src
    assert "timestamp" in src


def test_the_peak_can_still_only_climb_within_an_epoch():
    """Inside an epoch the high-water mark must remain monotone upward."""
    src = inspect.getsource(sl.SignalLoop._restore_risk_state)
    assert "max(peak_candidates)" in src


# ── Observability: a silent re-anchor would be indefensible ──────────────────

def test_applying_the_epoch_is_logged_loudly():
    """A bypass of a risk control must be visible in the logs, at WARNING."""
    a = inspect.getsource(sl.SignalLoop._peak_from_broker_history)
    b = inspect.getsource(sl.SignalLoop._restore_risk_state)
    assert "drawdown_peak_epoch_applied" in a
    assert "logger.warning" in a
    assert "drawdown_peak_epoch_discards_stored_peak" in b
