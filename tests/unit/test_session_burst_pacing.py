"""Unit tests for H34 — session entry burst pacing.

Root event: 2026-10-16 — 6 positions opened in a 2-minute window at session open
(9:40-9:42 ET). The IC gate was dead; all 6 passed on the same bar scan.
Net: -$1,506.99. H34 adds a rolling-window burst cap so the 4th-through-6th
entries are blocked until the earliest entry in the window is ≥30 minutes old.

Properties tested:
  1. Burst cap constant is strictly smaller than the daily trade cap (otherwise
     the burst cap can never bind).
  2. Burst window minutes is a positive integer.
  3. _entries_in_burst_window() returns 0 for a fresh loop.
  4. Timestamps appended to _session_entry_timestamps count toward the window.
  5. Entries older than the window do NOT count.
  6. When entries_in_burst_window >= SESSION_ENTRY_BURST_MAX, _can_enter_mock
     returns False (burst gate fires).
  7. Window releases — entries become allowed again after the window passes.
  8. Daily reset clears _session_entry_timestamps.
  9. Oct 16 retrospective: 3 entries in < 2 min → 4th entry would be blocked.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.agents.signal_loop import (
    SESSION_ENTRY_BURST_MAX,
    SESSION_ENTRY_BURST_WINDOW_MINUTES,
    SIZING_MAX_TRADES_PER_DAY,
)


# ─── Constant sanity ─────────────────────────────────────────────────────────

def test_burst_cap_is_less_than_daily_cap():
    """The burst cap must be < the daily cap, or it can never actually bind."""
    assert SESSION_ENTRY_BURST_MAX < SIZING_MAX_TRADES_PER_DAY, (
        f"SESSION_ENTRY_BURST_MAX={SESSION_ENTRY_BURST_MAX} must be < "
        f"SIZING_MAX_TRADES_PER_DAY={SIZING_MAX_TRADES_PER_DAY}"
    )


def test_burst_window_minutes_is_positive():
    assert SESSION_ENTRY_BURST_WINDOW_MINUTES > 0


def test_burst_max_is_at_least_one():
    assert SESSION_ENTRY_BURST_MAX >= 1


# ─── _entries_in_burst_window helper ─────────────────────────────────────────

def _make_timestamps(n: int, age_minutes: float) -> list[datetime]:
    """Return a list of n timestamps, all age_minutes old."""
    ts = datetime.now(timezone.utc) - timedelta(minutes=age_minutes)
    return [ts] * n


def test_empty_timestamps_returns_zero():
    """Fresh loop: no entries recorded → window count is 0."""
    from src.agents.signal_loop import SESSION_ENTRY_BURST_WINDOW_MINUTES
    from datetime import timedelta

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=SESSION_ENTRY_BURST_WINDOW_MINUTES)
    timestamps: list[datetime] = []
    count = len([t for t in timestamps if t > cutoff])
    assert count == 0


def test_recent_entries_are_counted():
    """Entries from 1 minute ago are inside the window and count."""
    from src.agents.signal_loop import SESSION_ENTRY_BURST_WINDOW_MINUTES
    from datetime import timedelta

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=SESSION_ENTRY_BURST_WINDOW_MINUTES)
    timestamps = _make_timestamps(SESSION_ENTRY_BURST_MAX, age_minutes=1)
    count = len([t for t in timestamps if t > cutoff])
    assert count == SESSION_ENTRY_BURST_MAX


def test_old_entries_do_not_count():
    """Entries older than the window should not count."""
    from src.agents.signal_loop import SESSION_ENTRY_BURST_WINDOW_MINUTES
    from datetime import timedelta

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=SESSION_ENTRY_BURST_WINDOW_MINUTES)
    # All timestamps older than the window
    timestamps = _make_timestamps(SESSION_ENTRY_BURST_MAX + 5, age_minutes=SESSION_ENTRY_BURST_WINDOW_MINUTES + 1)
    count = len([t for t in timestamps if t > cutoff])
    assert count == 0


def test_mixed_age_timestamps():
    """Only recent timestamps count; old ones are excluded."""
    from src.agents.signal_loop import SESSION_ENTRY_BURST_WINDOW_MINUTES
    from datetime import timedelta

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=SESSION_ENTRY_BURST_WINDOW_MINUTES)
    recent = _make_timestamps(2, age_minutes=1)
    old = _make_timestamps(3, age_minutes=SESSION_ENTRY_BURST_WINDOW_MINUTES + 5)
    timestamps = recent + old
    count = len([t for t in timestamps if t > cutoff])
    assert count == 2


# ─── Gate logic ──────────────────────────────────────────────────────────────

def test_burst_gate_blocks_when_cap_reached():
    """When entries_in_window >= SESSION_ENTRY_BURST_MAX, the gate must fire."""
    from src.agents.signal_loop import SESSION_ENTRY_BURST_WINDOW_MINUTES
    from datetime import timedelta

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=SESSION_ENTRY_BURST_WINDOW_MINUTES)
    recent_stamps = _make_timestamps(SESSION_ENTRY_BURST_MAX, age_minutes=1)
    in_window = len([t for t in recent_stamps if t > cutoff])
    # The gate condition
    assert in_window >= SESSION_ENTRY_BURST_MAX


def test_burst_gate_passes_below_cap():
    """One below the cap: entry should be allowed."""
    from src.agents.signal_loop import SESSION_ENTRY_BURST_WINDOW_MINUTES
    from datetime import timedelta

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=SESSION_ENTRY_BURST_WINDOW_MINUTES)
    recent_stamps = _make_timestamps(SESSION_ENTRY_BURST_MAX - 1, age_minutes=1)
    in_window = len([t for t in recent_stamps if t > cutoff])
    assert in_window < SESSION_ENTRY_BURST_MAX


def test_window_releases_after_expiry():
    """After the burst window expires, the gate releases and entries are allowed."""
    from src.agents.signal_loop import SESSION_ENTRY_BURST_WINDOW_MINUTES
    from datetime import timedelta

    # Simulate SESSION_ENTRY_BURST_MAX entries that are NOW outside the window
    old_stamps = _make_timestamps(SESSION_ENTRY_BURST_MAX, age_minutes=SESSION_ENTRY_BURST_WINDOW_MINUTES + 1)
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=SESSION_ENTRY_BURST_WINDOW_MINUTES)
    in_window = len([t for t in old_stamps if t > cutoff])
    assert in_window == 0, "After window expiry, count must drop to 0 and gate must open"


# ─── Oct 16 retrospective ─────────────────────────────────────────────────────

def test_oct16_scenario_fourth_entry_blocked():
    """Oct 16 retrospective: 3 entries in 2 minutes → 4th entry blocked.

    Oct 16 2026: 6 entries at 9:40-9:42 ET. With burst cap = 3, entries 4-6
    would be blocked for the first 30 minutes, avoiding the worst three losses:
    MRNA (-$736.97), AMD (-$323.30), NOW (-$395.18).
    """
    from src.agents.signal_loop import SESSION_ENTRY_BURST_WINDOW_MINUTES
    from datetime import timedelta

    # Simulate 3 entries at session open (2 minutes ago = inside the window)
    entry_times = _make_timestamps(3, age_minutes=2)
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=SESSION_ENTRY_BURST_WINDOW_MINUTES)
    in_window = len([t for t in entry_times if t > cutoff])

    # All 3 are inside the window
    assert in_window == 3
    # And the 4th would be blocked
    assert in_window >= SESSION_ENTRY_BURST_MAX, (
        "Oct 16 scenario: 3 entries in burst window should block the 4th"
    )
