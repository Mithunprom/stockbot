"""The per-ticker IC gate must be able to arm at all.

It never had. `TICKER_IC_MIN_N = 300` was asked of a 7-day window over a table
pruned at 7 days, so `tickers_ic_blocked` is empty in every production snapshot
on record and the positive-IC entry requirement from the 2026-07-07 diagnosis
was never in force either.

The same cause defeated the Kelly probe's 30-day widening in June: the premise
at signal_loop.py:455 was "~600+/ticker (reachable)", the measurement at :478
found 99, and the bar was lowered 300 → 30 to compensate without anyone finding
that retention was deleting the rows.

These tests pin the RELATIONSHIP between retention, window and threshold rather
than any single literal, because every past fix moved one of the three and was
silently cancelled by another.
"""
from __future__ import annotations

import pytest

from src.agents import signal_loop as sl
from src.data import db

# Measured in production 2026-10-07 from /admin/ic/report: 4,839 filled
# predictions over 7 calendar days across 75 tickers.
OBSERVED_PER_TICKER_PER_TRADING_DAY = 4839 / 75 / 5
TRADING_DAYS_PER_CALENDAR_DAY = 21 / 30


def _retention_days(table: str) -> int:
    for name, _col, days in db._RETENTION_POLICIES:
        if name == table:
            return days
    raise AssertionError(f"{table} has no retention policy")


def _reachable_sample(window_days: int) -> float:
    """Filled predictions per ticker available to a window, after retention."""
    effective = min(window_days, _retention_days("prediction_outcomes"))
    return (
        effective * TRADING_DAYS_PER_CALENDAR_DAY
        * OBSERVED_PER_TICKER_PER_TRADING_DAY
    )


# ── The regression ────────────────────────────────────────────────────────────

def test_retention_does_not_truncate_the_ic_window():
    """Retention must outlast the longest window that reads this table.

    This is the check whose absence let a 7-day prune silently cancel a 30-day
    window for four months.
    """
    longest_window = max(sl.TICKER_IC_WINDOW_DAYS, sl.KELLY_PROBE_IC_WINDOW_DAYS)
    assert _retention_days("prediction_outcomes") >= longest_window


def test_the_block_gate_threshold_is_attainable():
    """TICKER_IC_MIN_N must be within reach of its own window.

    Deliberately allows a margin below 300: at the observed rate a 30-day
    window yields ~271/ticker, so the gate arms on better-covered names first
    and fully as 120-day retention accumulates. The failure this guards is the
    old state — ~67 available against a bar of 300, unreachable by a factor of
    four.
    """
    available = _reachable_sample(sl.TICKER_IC_WINDOW_DAYS)
    assert available >= 0.75 * sl.TICKER_IC_MIN_N, (
        f"{available:.0f} predictions/ticker reachable in "
        f"{sl.TICKER_IC_WINDOW_DAYS}d vs a bar of {sl.TICKER_IC_MIN_N}"
    )


def test_the_old_configuration_would_fail_this():
    """Proof the test has teeth: 7d window + 7d retention cannot reach 300."""
    old_available = 7 * TRADING_DAYS_PER_CALENDAR_DAY * OBSERVED_PER_TICKER_PER_TRADING_DAY
    assert old_available < 0.75 * 300


def test_probe_bar_remains_reachable():
    """The probe deadlock must not come back — it has frozen the bot twice."""
    assert _reachable_sample(sl.KELLY_PROBE_IC_WINDOW_DAYS) >= sl.KELLY_PROBE_MIN_N


# ── Noise protection must not be quietly traded away ─────────────────────────

def test_min_n_was_not_lowered_to_force_the_gate_to_arm():
    """300 is the noise protection the May/June pattern study paid for.

    One-week per-ticker ICs flipped sign in 15 of 20 names. At n=300 the
    Spearman standard error is still ~0.058 against a -0.05 threshold, so
    lowering the bar to make the gate arm sooner would make it fire on noise —
    the exact failure that study identified. Fix the sample, not the bar.
    """
    assert sl.TICKER_IC_MIN_N >= 300


def test_block_threshold_unchanged():
    """Reachability work must not drift the gate's trigger level."""
    assert sl.TICKER_IC_BLOCK_THRESHOLD == -0.05
    assert sl.TICKER_IC_MIN_ENTRY == 0.0


# ── The restart reset ────────────────────────────────────────────────────────

def test_block_gate_no_longer_filters_by_process_start(monkeypatch):
    """`since=_loop_started_at` zeroed the sample on every deploy.

    That is why ticker_ic_tracked read 72 on 2026-10-02 and 0 on 10-06: with
    _compute_per_ticker_ic's n>=20 floor and ~13 predictions/ticker/day, a
    restart blinded the gate for ~2 trading days. The probe omitted `since` for
    this reason in June; the block gate must match.
    """
    calls: list[dict] = []

    class _Tracker:
        async def _compute_per_ticker_ic(self, **kwargs):
            calls.append(kwargs)
            return {}

    class _L:
        _ic_tracker = _Tracker()
        _ic_refresh_countdown = 1
        _ticker_ic: dict = {}
        _ticker_ic_probe: dict = {}
        _loop_started_at = None
        _kelly_mode = lambda self: "normal"  # noqa: E731

    loop = _L()
    import asyncio
    asyncio.run(sl.SignalLoop._maybe_refresh_ticker_ic(loop))

    assert calls, "the gate never queried the tracker"
    assert "since" not in calls[0]
    assert calls[0]["window_days"] == sl.TICKER_IC_WINDOW_DAYS
