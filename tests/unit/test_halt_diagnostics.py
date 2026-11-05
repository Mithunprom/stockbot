"""H37 — circuit-breaker halt timestamp and duration in diagnostics.

After 2026-11-02 the bot halted at 11.01% drawdown (max_drawdown CB fires).
EMAIL_ALERTS_ENABLED was false, so no email reached the owner. The diagnostics
endpoint showed `circuit_breaker_halted: true` but `halt_since: null` and
`halt_duration_hours: null` — the owner had no way to see from the dashboard
how long the bot had been silent without reading log files.

H37 adds three fields to the diagnostics export:
  - halt_since           ISO timestamp (UTC) when the halt fired
  - halt_duration_hours  wall-clock hours since halt (rounded to 1 decimal)
  - halt_lift_requires_human  True for max_drawdown, False/None for auto-clear halts

The fields are null when the bot is not halted, so they add no noise in
normal operation.

WHITELIST NOTE: main.py re-exports an explicit whitelist of diagnostics keys.
Adding to signal_loop is not enough — both files must be changed. That was the
lesson from entry_rank_* (v0.7.1→v0.7.2) and ticker_daily_vol (v0.8.6→v0.8.7).
Tests here verify BOTH sides of the wire.
"""
from __future__ import annotations

import inspect
from datetime import datetime, timedelta, timezone

import pytest


# ---------------------------------------------------------------------------
# signal_loop.py — the source dict
# ---------------------------------------------------------------------------

def _get_summary_source() -> str:
    from src.agents import signal_loop as sl
    return inspect.getsource(sl.SignalLoop.get_diagnostics_summary)


def test_halt_since_emitted_in_summary():
    """get_diagnostics_summary must include halt_since key."""
    src = _get_summary_source()
    assert '"halt_since"' in src, (
        "halt_since missing from get_diagnostics_summary — "
        "the nightly agent cannot determine how long the bot has been halted"
    )


def test_halt_duration_hours_emitted_in_summary():
    """get_diagnostics_summary must include halt_duration_hours key."""
    src = _get_summary_source()
    assert '"halt_duration_hours"' in src, (
        "halt_duration_hours missing from get_diagnostics_summary"
    )


def test_halt_lift_requires_human_emitted_in_summary():
    """get_diagnostics_summary must include halt_lift_requires_human key."""
    src = _get_summary_source()
    assert '"halt_lift_requires_human"' in src, (
        "halt_lift_requires_human missing from get_diagnostics_summary"
    )


# ---------------------------------------------------------------------------
# main.py — the whitelist
# ---------------------------------------------------------------------------

def _get_pipeline_diagnostics_source() -> str:
    import main as m
    return inspect.getsource(m._pipeline_diagnostics)


def test_halt_since_in_whitelist():
    """main.py _pipeline_diagnostics must pass halt_since through the whitelist."""
    src = _get_pipeline_diagnostics_source()
    assert '"halt_since"' in src, (
        "halt_since not in _pipeline_diagnostics whitelist. "
        "The field exists in signal_loop but will not reach /diagnostics. "
        "Lesson from entry_rank_* and ticker_daily_vol: always update BOTH files."
    )


def test_halt_duration_hours_in_whitelist():
    """main.py _pipeline_diagnostics must pass halt_duration_hours through."""
    src = _get_pipeline_diagnostics_source()
    assert '"halt_duration_hours"' in src, (
        "halt_duration_hours not in _pipeline_diagnostics whitelist"
    )


def test_halt_lift_requires_human_in_whitelist():
    """main.py _pipeline_diagnostics must pass halt_lift_requires_human through."""
    src = _get_pipeline_diagnostics_source()
    assert '"halt_lift_requires_human"' in src, (
        "halt_lift_requires_human not in _pipeline_diagnostics whitelist"
    )


# ---------------------------------------------------------------------------
# Logic unit tests for the halt timestamp values
# ---------------------------------------------------------------------------

class _FakeCB:
    """Minimal stand-in for CircuitBreakers."""
    def __init__(self, halted: bool, reason: str, halt_time: datetime | None):
        self._halted = halted
        self._halt_reason = reason
        self._halt_time = halt_time

    @property
    def is_halted(self) -> bool:
        return self._halted

    @property
    def halt_reason(self) -> str:
        return self._halt_reason


def _eval_halt_since(cb: _FakeCB) -> str | None:
    """Reproduce the halt_since expression from get_diagnostics_summary."""
    return (
        cb._halt_time.isoformat()
        if getattr(cb, "_halt_time", None) else None
    )


def _eval_halt_duration(cb: _FakeCB, now: datetime) -> float | None:
    """Reproduce the halt_duration_hours expression (injecting 'now')."""
    halt_time = getattr(cb, "_halt_time", None)
    if halt_time is None:
        return None
    return round((now - halt_time).total_seconds() / 3600, 1)


def _eval_halt_lift_requires_human(cb: _FakeCB) -> bool | None:
    return cb.halt_reason == "max_drawdown" if cb.is_halted else None


def test_halt_since_is_null_when_not_halted():
    cb = _FakeCB(halted=False, reason="", halt_time=None)
    assert _eval_halt_since(cb) is None


def test_halt_since_is_iso_string_when_halted():
    t = datetime(2026, 11, 2, 22, 20, 0, tzinfo=timezone.utc)
    cb = _FakeCB(halted=True, reason="max_drawdown", halt_time=t)
    result = _eval_halt_since(cb)
    assert result == "2026-11-02T22:20:00+00:00"


def test_halt_duration_null_when_no_halt_time():
    cb = _FakeCB(halted=True, reason="max_drawdown", halt_time=None)
    now = datetime(2026, 11, 5, 1, 0, 0, tzinfo=timezone.utc)
    assert _eval_halt_duration(cb, now) is None


def test_halt_duration_computes_correctly():
    halt_at = datetime(2026, 11, 2, 22, 20, 0, tzinfo=timezone.utc)
    # ~50.7 hours later
    now = datetime(2026, 11, 5, 1, 2, 0, tzinfo=timezone.utc)
    cb = _FakeCB(halted=True, reason="max_drawdown", halt_time=halt_at)
    hours = _eval_halt_duration(cb, now)
    assert hours == pytest.approx(50.7, abs=0.1)


def test_halt_lift_requires_human_true_for_max_drawdown():
    cb = _FakeCB(halted=True, reason="max_drawdown", halt_time=None)
    assert _eval_halt_lift_requires_human(cb) is True


def test_halt_lift_requires_human_false_for_daily_loss():
    cb = _FakeCB(halted=True, reason="daily_loss", halt_time=None)
    assert _eval_halt_lift_requires_human(cb) is False


def test_halt_lift_requires_human_null_when_not_halted():
    cb = _FakeCB(halted=False, reason="", halt_time=None)
    assert _eval_halt_lift_requires_human(cb) is None
