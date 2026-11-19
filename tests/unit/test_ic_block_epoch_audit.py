"""Unit tests for _ic_block_epoch_audit (H43).

Tests the pure async helper that checks which IC-blocked tickers would unblock
if IC_OUTCOMES_EPOCH were set to the IC fill job repair date (2026-10-16).
All tests use a fake tracker to avoid any DB or settings dependencies.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from src.agents.signal_loop import _ic_block_epoch_audit


REF_EPOCH = "2026-10-16T00:00:00+00:00"
MIN_N = 300
BLOCK_THRESHOLD = -0.05


class FakeTracker:
    """Minimal tracker stand-in; _compute_per_ticker_ic is replaced per test."""

    async def _compute_per_ticker_ic(
        self, window_days: int, since=None
    ) -> dict:  # pragma: no cover
        return {}


# ── helpers ──────────────────────────────────────────────────────────────────


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── tests ─────────────────────────────────────────────────────────────────────


def test_all_unblock_when_post_epoch_data_missing():
    """Tickers with no post-epoch data at all count as would_unblock."""
    tracker = FakeTracker()
    tracker._compute_per_ticker_ic = AsyncMock(return_value={})

    result = run(
        _ic_block_epoch_audit(
            tracker, ["AAPL", "TSLA"], 30, REF_EPOCH, MIN_N, BLOCK_THRESHOLD
        )
    )

    assert result["would_unblock"] == ["AAPL", "TSLA"]
    assert result["would_unblock_count"] == 2
    assert result["would_remain_blocked"] == []
    assert result["currently_blocked_count"] == 2
    assert result["reference_epoch"] == REF_EPOCH


def test_ticker_unblocks_when_post_epoch_ic_above_threshold():
    """A ticker whose post-epoch IC is >= block_threshold should unblock."""
    tracker = FakeTracker()
    # AAPL: post-epoch IC is +0.03 (above -0.05 threshold) → unblocks
    # TSLA: post-epoch IC is -0.08 with n=400 → stays blocked
    tracker._compute_per_ticker_ic = AsyncMock(
        return_value={
            "AAPL": {"ic": 0.03, "n": 350},
            "TSLA": {"ic": -0.08, "n": 400},
        }
    )

    result = run(
        _ic_block_epoch_audit(
            tracker, ["AAPL", "TSLA"], 30, REF_EPOCH, MIN_N, BLOCK_THRESHOLD
        )
    )

    assert result["would_unblock"] == ["AAPL"]
    assert result["would_remain_blocked"] == ["TSLA"]


def test_ticker_stays_blocked_when_insufficient_post_epoch_n():
    """Post-epoch data with n < MIN_N means unblock (gate never fires without n)."""
    tracker = FakeTracker()
    # NVDA: post-epoch IC = -0.10 but only n=50 — not enough to gate → unblocks
    tracker._compute_per_ticker_ic = AsyncMock(
        return_value={"NVDA": {"ic": -0.10, "n": 50}}
    )

    result = run(
        _ic_block_epoch_audit(
            tracker, ["NVDA"], 30, REF_EPOCH, MIN_N, BLOCK_THRESHOLD
        )
    )

    assert result["would_unblock"] == ["NVDA"]
    assert result["would_remain_blocked"] == []


def test_empty_blocked_list_returns_empty_audit():
    """With no blocked tickers the function should still return a valid dict."""
    tracker = FakeTracker()
    tracker._compute_per_ticker_ic = AsyncMock(return_value={})

    result = run(
        _ic_block_epoch_audit(
            tracker, [], 30, REF_EPOCH, MIN_N, BLOCK_THRESHOLD
        )
    )

    assert result["would_unblock"] == []
    assert result["would_remain_blocked"] == []
    assert result["currently_blocked_count"] == 0


def test_tracker_exception_returns_empty_dict():
    """If the tracker raises, the function must return {} without propagating."""
    tracker = FakeTracker()
    tracker._compute_per_ticker_ic = AsyncMock(side_effect=RuntimeError("db down"))

    result = run(
        _ic_block_epoch_audit(
            tracker, ["AMZN"], 30, REF_EPOCH, MIN_N, BLOCK_THRESHOLD
        )
    )

    assert result == {}
