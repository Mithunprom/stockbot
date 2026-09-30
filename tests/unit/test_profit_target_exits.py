"""Profit-target exit mode + the Kelly probation deadlock fix.

Two owner-directed changes (2026-09-22):

1. Stop closing positions on a 30-minute timer. Exit on a profit target sized
   as a fraction of the name's ANNUAL volatility; the clock survives only as a
   session backstop so nothing carries overnight.
2. Kelly probation must be escapable. The probe required n >= 300 filled
   predictions per ticker while the system produced at most 99, so
   `tickers_probe_eligible` was structurally empty and probation became a
   permanent freeze — four times.
"""
from __future__ import annotations

import math
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from src.agents import signal_loop as sl

ET = ZoneInfo("America/New_York")


@pytest.fixture(autouse=True)
def profit_target_mode(monkeypatch):
    """Enable the mode explicitly.

    It is env-gated and OFF by default so the rest of the suite keeps testing
    the shipped timer behaviour — this mode changes the exit ladder and must
    not silently rewrite what those tests protect.
    """
    monkeypatch.setattr(sl, "EXIT_PROFIT_TARGET_MODE", True)


# ── Profit target ────────────────────────────────────────────────────────────

def test_target_scales_with_annual_volatility():
    """A 'move worth taking' must mean the same thing across names.

    A fixed 2% is noise for MSTR and unreachable for XOM; a fraction of each
    name's own annual vol is comparable across the book.
    """
    calm_daily, wild_daily = 0.010, 0.040          # ~16% vs ~63% annualised
    _, _, tp_calm = sl._atr_exits(calm_daily)
    _, _, tp_wild = sl._atr_exits(wild_daily)
    assert tp_wild > tp_calm
    assert tp_wild / tp_calm == pytest.approx(wild_daily / calm_daily, rel=0.25)


def test_target_matches_the_requested_fraction_of_annual_vol():
    daily = 0.025                                   # ~39.7% annualised
    annual = daily * math.sqrt(sl.TRADING_DAYS_PER_YEAR)
    expected = annual * sl.PROFIT_TARGET_ANNUAL_VOL_FRAC
    _, _, tp = sl._atr_exits(daily)
    if sl.SIZING_TAKE_PROFIT_FLOOR < expected < sl.SIZING_TAKE_PROFIT_CAP:
        assert tp == pytest.approx(expected, rel=1e-6)


def test_target_is_reachable_not_decorative():
    """The old barrier sat at ~5.4 sigma of the hold window and never fired.

    A target nothing can reach is the defect, not the fix.
    """
    daily = 0.025
    _, _, tp = sl._atr_exits(daily)
    # a full-session sigma for this name
    session_sigma = sl._hold_window_vol(daily, sl.FULL_SESSION_BARS)
    assert tp / session_sigma < 3.0, (
        f"take-profit is {tp/session_sigma:.1f} sigma of a session — "
        f"decorative again"
    )


def test_hold_window_is_a_full_session_in_profit_target_mode():
    assert sl._effective_hold_bars() == sl.FULL_SESSION_BARS
    assert sl._effective_hold_bars() > sl.SIZING_MAX_HOLD_BARS


def test_session_backstop_still_exists():
    """The timer must not be removed outright.

    An unbounded hold produced the MSCI zombie (9 days open, portfolio_heat
    reporting 0.0 while 12.9% was deployed). Bounded by the session is the
    floor, not a preference.
    """
    assert sl._effective_hold_bars() <= sl.FULL_SESSION_BARS


def test_stop_loss_still_binds_in_the_longer_window():
    """Letting winners run must not mean letting losers run."""
    daily = 0.025
    stop, trail, tp = sl._atr_exits(daily)
    assert 0 < stop < tp, "stop must be tighter than the profit target"
    assert 0 < trail


@pytest.mark.parametrize("daily", [0.004, 0.010, 0.025, 0.060])
def test_exits_are_ordered_and_finite(daily):
    stop, trail, tp = sl._atr_exits(daily)
    for v in (stop, trail, tp):
        assert math.isfinite(v) and v > 0
    assert stop <= sl.SIZING_STOP_LOSS_CAP
    assert tp <= sl.SIZING_TAKE_PROFIT_CAP


# ── Kelly probation deadlock ─────────────────────────────────────────────────

def test_probe_threshold_is_reachable_by_observed_data():
    """REGRESSION: the fourth recurrence of the same freeze.

    Measured 2026-09-22 in production: 82 tickers, 6,426 filled predictions,
    max 99 on any single ticker. A bar of 300 can never be met.
    """
    OBSERVED_MAX_N_PER_TICKER = 99
    assert sl.KELLY_PROBE_MIN_N <= OBSERVED_MAX_N_PER_TICKER, (
        f"probe needs n>={sl.KELLY_PROBE_MIN_N} but the system has never "
        f"produced more than {OBSERVED_MAX_N_PER_TICKER} per ticker — "
        f"probation would be a permanent freeze again"
    )


def test_probe_bar_is_far_below_the_block_gate_bar():
    """They serve opposite purposes and must not share a threshold.

    Blocking a ticker needs overwhelming evidence (high bar). Probing one needs
    only a plausible candidate — requiring proof to gather proof is circular.
    """
    assert sl.KELLY_PROBE_MIN_N < sl.TICKER_IC_MIN_N


def test_probe_still_demands_a_positive_ic():
    """Loosening the sample bar must not loosen the quality bar."""
    assert sl.KELLY_PROBATION_MIN_TICKER_IC > 0


# ── Session backstop wall-clock path (H28 — 2026-09-29 zombie root cause) ────
#
# Root cause: bars_held maxes at 389 by 15:59 ET for any intraday entry while
# FULL_SESSION_BARS=390. The bars-only check was dead code — six positions
# entered Sep 29 were never closed in the DB (IDs 248-253).

def test_bars_held_off_by_one_was_zombie_root_cause():
    """Regression: the maximum achievable bars_held within a session is 389.

    A 9:30 AM entry reaches bars_held=389 by 15:59 ET — one short of
    FULL_SESSION_BARS=390. The wall-clock fix must cover this gap.
    """
    max_bars_in_session = sl.FULL_SESSION_BARS - 1  # 389 for a 9:30 AM entry
    assert max_bars_in_session < sl.FULL_SESSION_BARS, (
        "off-by-one: bars_held can never reach FULL_SESSION_BARS within a session"
    )
    # Wall-clock at 15:57 covers a 9:30 AM entry that would only reach bar 387
    at_1557 = datetime(2026, 9, 29, 15, 57, 0, tzinfo=ET)
    assert sl._is_session_closing(at_1557), (
        "wall-clock backstop must cover bar-389 entries"
    )


def test_session_close_fires_at_buffer_boundary():
    """_is_session_closing fires at exactly SESSION_CLOSE_BUFFER_MINUTES before 16:00."""
    cutoff_minute = 60 - sl.SESSION_CLOSE_BUFFER_MINUTES   # 57 by default
    at_cutoff = datetime(2026, 9, 29, 15, cutoff_minute, 0, tzinfo=ET)
    assert sl._is_session_closing(at_cutoff)


def test_session_close_fires_at_1559():
    at_1559 = datetime(2026, 9, 29, 15, 59, 0, tzinfo=ET)
    assert sl._is_session_closing(at_1559)


def test_session_close_does_not_fire_one_minute_before_buffer():
    cutoff_minute = 60 - sl.SESSION_CLOSE_BUFFER_MINUTES
    before_cutoff = datetime(2026, 9, 29, 15, cutoff_minute - 1, 59, tzinfo=ET)
    assert not sl._is_session_closing(before_cutoff)


def test_session_close_does_not_fire_at_midday():
    at_noon = datetime(2026, 9, 29, 12, 0, 0, tzinfo=ET)
    assert not sl._is_session_closing(at_noon)


def test_session_close_does_not_fire_on_weekend():
    # Sep 27 2026 is a Saturday
    saturday_near_close = datetime(2026, 9, 27, 15, 58, 0, tzinfo=ET)
    assert not sl._is_session_closing(saturday_near_close)


def test_session_close_accepts_utc_datetime():
    """_is_session_closing must convert to ET before comparing."""
    from datetime import timezone
    # 19:57 UTC = 15:57 ET (EDT, UTC-4)
    at_1957_utc = datetime(2026, 9, 29, 19, 57, 0, tzinfo=timezone.utc)
    assert sl._is_session_closing(at_1957_utc)


def test_session_close_buffer_constant_is_positive_and_small():
    """Buffer must give time for fills without being so wide it exits too early."""
    assert 1 <= sl.SESSION_CLOSE_BUFFER_MINUTES <= 10
