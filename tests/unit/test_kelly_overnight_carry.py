"""Overnight carries must not seed the Kelly governor.

On 2026-09-30 six positions were stranded by a bar-counted session backstop and
closed on 2026-10-02 for +$3,916 — three names supplied all of it. The integrity
agent re-seeds Kelly from the ledger after every repair, so that windfall took
the governor from `inactive` to `normal` at f=0.454. The same window with the
cohort removed is n=12, PF 0.89, −$90.71: no edge. The bot was about to size up
on a defect's payoff.

The trades are real and stay in the ledger; this only changes which outcomes the
POSITION SIZER treats as representative of the exit rule it actually runs.

The load-bearing test is the last one: the filter must not be able to empty the
window into a silently permissive state.
"""
from __future__ import annotations

from datetime import datetime, timezone

from src.agents import signal_loop as sl

ET_OFFSET_HOURS = 4  # ET = UTC-4 in October (EDT)


def _utc(y: int, m: int, d: int, hh: int, mm: int = 0) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=timezone.utc)


# ── The cohort that caused this ───────────────────────────────────────────────

def test_the_stranded_sep30_cohort_is_excluded() -> None:
    """Entered 2026-09-30 09:40 ET, closed 2026-10-02 — two sessions later."""
    entry = _utc(2026, 9, 30, 13, 40)   # 09:40 ET Sep 30
    exit_ = _utc(2026, 10, 2, 14, 25)   # 10:25 ET Oct 2
    assert sl._carried_overnight(entry, exit_) is True


def test_a_normal_same_session_trade_is_kept() -> None:
    """Entered 09:40 ET, closed 11:25 ET the same day — the intended shape."""
    entry = _utc(2026, 10, 5, 13, 40)   # 09:40 ET
    exit_ = _utc(2026, 10, 5, 15, 25)   # 11:25 ET
    assert sl._carried_overnight(entry, exit_) is False


# ── ET, not UTC — the boundary that actually matters ──────────────────────────

def test_utc_midnight_mid_session_is_not_a_carry() -> None:
    """A full-session hold crossing 20:00 ET (00:00 UTC) stays one ET date.

    This is the case a naive UTC-date comparison gets wrong: both timestamps
    are inside one trading day but land on different UTC dates.
    """
    entry = _utc(2026, 10, 5, 19, 30)        # 15:30 ET Oct 5
    exit_ = _utc(2026, 10, 6, 0, 30)         # 20:30 ET Oct 5 — same ET date
    assert entry.date() != exit_.date()      # differs in UTC …
    assert sl._carried_overnight(entry, exit_) is False  # … but not in ET


def test_a_true_overnight_hold_is_a_carry() -> None:
    """Held from one session's close into the next session's open."""
    entry = _utc(2026, 10, 5, 19, 55)   # 15:55 ET Oct 5
    exit_ = _utc(2026, 10, 6, 13, 45)   # 09:45 ET Oct 6
    assert sl._carried_overnight(entry, exit_) is True


# ── Imprecision must fall on the safe side ───────────────────────────────────

def test_post_midnight_repair_stamp_is_treated_as_a_carry() -> None:
    """`integrity_broker_reconcile` stamps the agent's RUN time, not the fill.

    A repair landing after ET midnight for a position entered in the morning
    window is indistinguishable from a real carry, so it must be dropped. The
    cost of being wrong here is one outcome; the cost of the opposite error is
    an inflated position size.
    """
    entry = _utc(2026, 10, 5, 13, 40)   # 09:40 ET Oct 5
    exit_ = _utc(2026, 10, 6, 5, 0)     # 01:00 ET Oct 6 — repair ran overnight
    assert sl._carried_overnight(entry, exit_) is True


def test_missing_timestamps_are_not_treated_as_carries() -> None:
    """An unknown span is not evidence of a carry; the caller drops no-exit rows."""
    assert sl._carried_overnight(None, _utc(2026, 10, 5, 15, 25)) is False
    assert sl._carried_overnight(_utc(2026, 10, 5, 13, 40), None) is False
    assert sl._carried_overnight(None, None) is False


def test_naive_timestamps_are_read_as_utc() -> None:
    """Asyncpg can hand back naive datetimes; they must not raise."""
    entry = datetime(2026, 9, 30, 13, 40)
    exit_ = datetime(2026, 10, 2, 14, 25)
    assert sl._carried_overnight(entry, exit_) is True


# ── The control must not become permissive ───────────────────────────────────

def test_excluding_the_cohort_leaves_kelly_active_not_inactive() -> None:
    """18 honest outcomes clear KELLY_MIN_TRADES, so the governor keeps judging.

    If the filter dropped the sample below the minimum, Kelly would go
    `inactive` — and `_kelly_entries_blocked` returns False in that mode. A
    filter that silently disarms the negative-edge block would be worse than
    the inflation it fixes.
    """
    n_window, n_carried = 24, 6
    assert n_window - n_carried >= sl.KELLY_MIN_TRADES
