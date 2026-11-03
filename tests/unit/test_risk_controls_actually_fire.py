"""Two risk controls that reported protection they were not providing.

Both found on 2026-11-02 after the bot lost $2,443 across 2026-10-29/30 at full
size (~$12,300 x 12 entries, 9 losers) and neither control stopped it.

1. MAX DRAWDOWN measured from a peak found by a **3-month** broker-history
   window. A high-water mark is by definition all-time, so the real peak
   ($108,588.50, 2026-04-03) aged out of the window. The breaker ran on
   ~$104,278 and reported 7.33% against its 8% limit while the true drawdown
   was 11.01%. It should have halted days before those sessions.

2. CONSECUTIVE LOSSES returned action="reduce_sizing_50pct" and **nothing read
   it**. It triggered at 6 losses on 2026-10-30; both subsequent entries still
   went out at full size. A breaker whose action is discarded is worse than no
   breaker — it reads as cover on the dashboard.
"""
from __future__ import annotations

import inspect

import pytest

from src.agents import signal_loop as sl
from src.risk import circuit_breakers as cb


# ── 1. The peak window ───────────────────────────────────────────────────────

def test_broker_peak_window_covers_more_than_a_quarter():
    """A rolling window silently lowers an all-time high-water mark."""
    sig = inspect.signature(sl.SignalLoop._peak_from_broker_history)
    period = sig.parameters["period"].default
    assert period != "3M", "the 3-month window is the bug"
    assert period in ("1A", "2A", "5A", "all"), f"unexpected period {period!r}"


def test_the_oct_peak_would_now_be_visible():
    """2026-04-03 is ~7 months before 2026-11-02 — inside 1A, outside 3M."""
    months = {"3M": 3, "6M": 6, "1A": 12, "2A": 24}
    period = inspect.signature(
        sl.SignalLoop._peak_from_broker_history).parameters["period"].default
    assert months.get(period, 99) >= 8, (
        f"{period} cannot reach a peak set 7 months ago")


def test_drawdown_from_the_true_peak_breaches_the_limit():
    """Pins the arithmetic that makes this consequential, not cosmetic."""
    true_peak, equity, limit = 108_588.50, 96_638.04, 0.08
    true_dd = (true_peak - equity) / true_peak
    assert true_dd > limit
    stale_peak = 104_278.53
    stale_dd = (stale_peak - equity) / stale_peak
    assert stale_dd < limit, "the stale peak is what kept the breaker quiet"


def test_the_peak_can_only_be_corrected_upward():
    """`_restore_risk_state` max()es stored/broker/current.

    If a future edit lets the peak move DOWN, drawdown silently resets to zero
    and the breaker goes blind — the original bug this module was written for.
    """
    src = inspect.getsource(sl.SignalLoop._restore_risk_state)
    assert "max(peak_candidates)" in src
    assert "broker_peak" in src


# ── 2. The consecutive-loss brake ────────────────────────────────────────────

def test_the_scalar_only_ever_reduces():
    for n in range(0, 20):
        assert 0.0 < cb.consecutive_loss_size_scalar(n) <= 1.0


def test_no_reduction_below_the_limit():
    for n in range(0, cb.CONSECUTIVE_LOSS_LIMIT):
        assert cb.consecutive_loss_size_scalar(n) == 1.0


def test_reduction_applies_at_and_above_the_limit():
    for n in (cb.CONSECUTIVE_LOSS_LIMIT, cb.CONSECUTIVE_LOSS_LIMIT + 1, 20):
        assert cb.consecutive_loss_size_scalar(n) == cb.CONSECUTIVE_LOSS_SIZE_SCALAR


def test_the_oct30_streak_would_have_been_halved():
    """6 consecutive losses, the state on 2026-10-30."""
    assert cb.consecutive_loss_size_scalar(6) == 0.5
    full = 12_300.0
    assert full * cb.consecutive_loss_size_scalar(6) == pytest.approx(6_150.0)


def test_the_breaker_and_the_scalar_share_one_threshold():
    """Duplicating 5 at the call site is how the two drift apart."""
    src = inspect.getsource(cb.CircuitBreakers._check_consecutive_losses)
    assert "CONSECUTIVE_LOSS_LIMIT" in src
    assert "threshold = 5" not in src


def test_the_entry_path_actually_applies_the_scalar():
    """The whole point: the action must reach sizing, not just the log.

    Asserts the call plus the multiplication, since importing the symbol
    without using it would satisfy a name-only check.
    """
    src = inspect.getsource(sl.SignalLoop._act_on_signal)
    assert "consecutive_loss_size_scalar" in src
    assert "notional *= _loss_scalar" in src


def test_the_brake_composes_with_the_kelly_cap_rather_than_replacing_it():
    """Both are haircuts; the scalar must be applied BEFORE the Kelly min()
    so the tighter of the two always wins."""
    src = inspect.getsource(sl.SignalLoop._act_on_signal)
    assert src.index("notional *= _loss_scalar") < src.index("_kelly_size_cap()")
