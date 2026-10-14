"""Nothing may carry overnight — enforced by the clock, not a bar counter.

`_effective_hold_bars()` has always promised "nothing carries overnight", but it
counts BARS. In profit-target mode the cap is 390 and `_bars_held` is never
reset at a session boundary, so a position entered at 09:40 can only reach ~380
bars before the close and NECESSARILY survives into the next session, tripping
at ~09:42 the following trading day. That is systematic, not a rare stall: every
position that misses its stop, trail and target carries.

SWKS proved it on 2026-10-09 — entered Friday 09:40 ET, closed Monday 09:42 ET
across a weekend gap. It profited (+$22.96), exactly as the six positions
stranded on 2026-09-30 profited (+$3,916). Being lucky with unhedged gap risk
twice is not a control.

It only surfaced after v0.9.2, because the watchdog had been force-selling
everything at 90 bars — accidentally preventing carries while corrupting the
ledger.
"""
from __future__ import annotations

from datetime import date, datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from src.agents import signal_loop as sl

ET = ZoneInfo("America/New_York")


def _loop(entry_date: date | None = None, bars: int = 5):
    """A SignalLoop stub exposing only what the backstop reads."""

    class _L:
        pass

    loop = _L()
    loop._entry_dates = {"SWKS": entry_date} if entry_date is not None else {}
    loop._bars_held = {"SWKS": bars}
    loop._session_boundary_crossed = sl.SignalLoop._session_boundary_crossed.__get__(loop)
    return loop


def _at(y, m, d, hh, mm):
    """Patch the module clock to a fixed ET wall-clock time."""
    fixed = datetime(y, m, d, hh, mm, tzinfo=ET)
    mock_dt = MagicMock(wraps=datetime)
    mock_dt.now.return_value = fixed
    return patch.object(sl, "datetime", mock_dt)


# ── The clock arm ─────────────────────────────────────────────────────────────

def test_flattens_at_the_session_flat_time():
    loop = _loop(entry_date=date(2026, 10, 9))
    with _at(2026, 10, 9, 15, 55):
        assert loop._session_boundary_crossed("SWKS") is True


def test_does_not_flatten_mid_session():
    """09:41 on the entry day is an ordinary hold, not a boundary."""
    loop = _loop(entry_date=date(2026, 10, 9))
    with _at(2026, 10, 9, 9, 41):
        assert loop._session_boundary_crossed("SWKS") is False


def test_does_not_flatten_one_minute_early():
    loop = _loop(entry_date=date(2026, 10, 9))
    with _at(2026, 10, 9, 15, 54):
        assert loop._session_boundary_crossed("SWKS") is False


def test_flattens_after_the_flat_time_too():
    """A missed tick at 15:55 must not mean the position survives the day."""
    loop = _loop(entry_date=date(2026, 10, 9))
    with _at(2026, 10, 9, 15, 59):
        assert loop._session_boundary_crossed("SWKS") is True


def test_flat_time_leaves_room_to_fill():
    """15:55, not 16:00 — a market order at 15:59:30 may not fill."""
    assert sl.SESSION_FLAT_ET < (16, 0)


# ── The catch-up arm ─────────────────────────────────────────────────────────

def test_a_position_entered_on_an_earlier_date_exits_immediately():
    """Closes an already-carried position on the NEXT tick, at any hour.

    This is what rescues a position that slipped through a restart, a halt or a
    sparse feed, instead of waiting on a bar count that may never arrive.
    """
    loop = _loop(entry_date=date(2026, 10, 9))
    with _at(2026, 10, 12, 9, 42):   # Monday open
        assert loop._session_boundary_crossed("SWKS") is True


def test_unknown_entry_date_falls_back_to_the_clock():
    """A recovered position with no entry date is still flattened at 15:55."""
    loop = _loop(entry_date=None)
    with _at(2026, 10, 9, 15, 55):
        assert loop._session_boundary_crossed("SWKS") is True
    with _at(2026, 10, 9, 10, 0):
        assert loop._session_boundary_crossed("SWKS") is False


# ── Bars must be irrelevant ──────────────────────────────────────────────────

@pytest.mark.parametrize("bars", [0, 5, 100, 389, 10_000])
def test_the_bar_count_cannot_defeat_the_backstop(bars):
    """The whole point: a stalled or sparse feed must not keep a position open."""
    loop = _loop(entry_date=date(2026, 10, 9), bars=bars)
    with _at(2026, 10, 9, 15, 55):
        assert loop._session_boundary_crossed("SWKS") is True


def test_the_swks_regression_would_have_closed_on_friday():
    """Friday 09:40 entry → flat Friday 15:55, not Monday 09:42.

    At 380 bars the old cap of 390 was still unreachable, which is precisely why
    the position survived the weekend.
    """
    loop = _loop(entry_date=date(2026, 10, 9), bars=380)
    assert 380 < sl._effective_hold_bars() or not sl.EXIT_PROFIT_TARGET_MODE
    with _at(2026, 10, 9, 15, 55):
        assert loop._session_boundary_crossed("SWKS") is True


# ── Mode independence ────────────────────────────────────────────────────────

@pytest.mark.parametrize("profit_target", [True, False])
def test_the_backstop_applies_in_both_exit_modes(monkeypatch, profit_target):
    """It can only ever exit EARLIER than the bar cap, so it is safe in both.

    Timer mode's 30-bar cap normally fires long before 15:55, making this
    near-inert there — but "nothing carries overnight" is a global invariant,
    not a profit-target-mode feature.
    """
    monkeypatch.setattr(sl, "EXIT_PROFIT_TARGET_MODE", profit_target)
    loop = _loop(entry_date=date(2026, 10, 9))
    with _at(2026, 10, 9, 15, 55):
        assert loop._session_boundary_crossed("SWKS") is True


# ── It must not pre-empt a real exit reason ──────────────────────────────────

def test_h5_keeps_its_extension_attempt_before_the_flatten():
    """The backstop is OR'd with the bar cap, not checked before it.

    H5 (`_maybe_extend_hold`) grants "one extra trading DAY" up to a 3-day cap —
    it is the one sanctioned multi-day path. Checking the backstop first would
    have made it dead code: a carried position would be flattened before the
    extension logic ever ran. Inert today at MAX_HOLD_EXTENSIONS = 0, so this
    ordering costs nothing now and keeps H5 working if re-enabled.
    """
    import inspect

    body = inspect.getsource(sl.SignalLoop._check_sizing_exit)
    assert "bars >= hold_cap or must_flatten" in body
    # The extension attempt must gate the reason assignment, not follow it.
    extend_at = body.index("_maybe_extend_hold")
    reason_at = body.index('"session_close"\n', extend_at)
    assert extend_at < reason_at


def test_a_real_stop_loss_still_logs_as_stop_loss():
    """Ordering matters: a stop at 15:56 must not be relabelled session_close.

    The backstop only assigns a reason when none has been set, so the exit mix
    stays readable — if `session_close` starts dominating, the target is set too
    far away, which is exactly the signal reading the mix is meant to give.
    """
    import inspect

    body = inspect.getsource(sl.SignalLoop._check_sizing_exit)
    assert body.index('reason = "stop_loss"') < body.index("must_flatten")
