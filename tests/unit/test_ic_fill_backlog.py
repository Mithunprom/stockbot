"""The outcome-fill job must make progress, and must say so when it cannot.

Measured in production on 2026-10-14: three consecutive `POST /admin/ic/fill`
runs returned `{"filled_15m": 0, "filled_30m": 0, "errors": 0}`. The job was not
slow, it was DEAD. `.limit(500)` with no ORDER BY returned the same oldest
unfilled rows every run; `_lookup_forward_return` returns None when the
ohlcv_1m bars are missing, which left each row untouched to be re-selected
forever. New predictions were never reached and retention deleted them
unmeasured.

That is why live IC read -0.024 against a backtest +0.173: the figure was a
FROZEN set filled at some point in the past, not a sample of the current model.
It was never evidence about the signal.

`errors: 0` is what made it invisible — nothing threw, so nothing alerted, and a
dead pipeline looked exactly like an idle one.
"""
from __future__ import annotations

import inspect

import pytest

from src.agents import live_ic_tracker as t


def _body() -> str:
    """Source of `fill_actual_returns` with the docstring removed.

    The docstring quotes the old `.limit(500)` to explain the bug, so a naive
    substring search over the whole function matches the very text documenting
    the fix.
    """
    src = inspect.getsource(t.LiveICTracker.fill_actual_returns)
    # Slice from the first statement after the docstring. Removing `__doc__` by
    # substring does not work — the raw source and the parsed docstring differ
    # in leading whitespace, so the replace silently no-ops and every assertion
    # below matches the prose instead of the code.
    marker = "now = datetime.now(timezone.utc)"
    assert marker in src, "docstring boundary marker moved; update _body()"
    return src[src.index(marker):]


# ── The three defects, pinned at the source ──────────────────────────────────

def test_the_batch_is_ordered_newest_first():
    """Without ORDER BY, Postgres is free to return the same rows forever.

    Newest-first is also the right priority on the merits: recent predictions
    are the ones whose bars exist and the ones worth measuring.
    """
    src = _body()
    assert "order_by" in src
    assert "timestamp.desc()" in src
    assert ".limit(500)" not in src, "hardcoded batch size is back"


def test_retired_rows_are_excluded_from_the_batch():
    """The query must skip rows already resolved, or the backlog never drains."""
    src = _body()
    assert "filled_at.is_(None)" in src


def test_unfillable_rows_are_retired_rather_than_retried():
    src = _body()
    assert "retire_before" in src
    assert "retired_15m" in src


# ── The retirement sentinel must not corrupt any IC measurement ──────────────

def test_retiring_a_row_never_invents_an_outcome():
    """Retirement stamps `filled_at` and leaves `actual_return_15m` NULL.

    This is load-bearing. Every IC query filters on
    `actual_return_15m IS NOT NULL`, so a retired row is excluded from the
    measurement entirely. Writing a placeholder return — 0.0, say — would
    silently drag measured IC toward zero, which is far worse than a missing
    sample.
    """
    src = _body()
    retire_block = src[src.index("elif pred.timestamp <= retire_before"):]
    retire_block = retire_block[: retire_block.index("except")]
    assert "pred.filled_at = now" in retire_block
    assert "actual_return_15m" not in retire_block, (
        "retirement must not touch the outcome column"
    )


def test_the_30m_leg_cannot_pick_up_retired_rows():
    """It requires actual_return_15m to be set, which retired rows never have."""
    src = _body()
    assert "PredictionOutcome.actual_return_15m.isnot(None)" in src


def test_retirement_window_allows_same_session_backfills():
    """Generous on purpose — a late bar within the session must still land."""
    assert t.UNFILLABLE_AFTER_HOURS >= 12
    assert t.UNFILLABLE_AFTER_HOURS <= 72


# ── Capacity ─────────────────────────────────────────────────────────────────

def test_the_batch_can_keep_pace_with_recording():
    """28 runs/session must cover ~29k predictions (75 tickers × ~390 ticks).

    The job is scheduled 4×/hour, 10:00–16:00 ET. At the old 500 it could cover
    at most 14k — permanently behind even with perfect ordering.
    """
    runs_per_session = 4 * 7
    recorded_per_session = 75 * 390
    assert t.FILL_BATCH_SIZE * runs_per_session >= recorded_per_session


# ── The silence that hid it ──────────────────────────────────────────────────

def test_a_full_batch_yielding_nothing_warns():
    """"0 filled of 500 attempted" must never again read as "nothing to do"."""
    src = _body()
    assert "ic_fill_yielded_nothing" in src
    assert "attempted_15m > 0 and filled_15m == 0" in src


def test_the_summary_reports_what_was_attempted():
    """Without `attempted`, a dead pipeline and an idle one are identical."""
    src = _body()
    summary = src[src.index("summary = {"):]
    for key in ('"attempted_15m"', '"filled_15m"', '"retired_15m"'):
        assert key in summary
