"""Kelly epoch — an audited, bounded bypass of the negative-edge entry block.

Authorised by the owner on 2026-09-29 (see KELLY_EPOCH_ISO for the full
rationale). kelly_fraction sat at -0.2699 against a -0.25 hard block, and every
outcome in the window came from one session whose trades ran with `rs_vwap_dev`
zero-filled — a serving defect fixed in v0.8.2. The governor was judging a
configuration that no longer existed.

The property that makes this acceptable rather than reckless is the LAST test
here: the control must still re-arm on post-epoch outcomes. An epoch that
permanently defanged the block would be removing the control, not re-seeding it.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.agents import signal_loop as sl


def _loop():
    """A bare object exposing only the window helpers under test."""

    class _L:
        _sizing_recent_outcomes: list = []
        _kelly_fraction = 0.0
        _kelly_min_trades = sl.KELLY_MIN_TRADES
        _prune_kelly_window = sl.SignalLoop._prune_kelly_window
        _kelly_mode = sl.SignalLoop._kelly_mode
        _kelly_entries_blocked = sl.SignalLoop._kelly_entries_blocked

    return _L()


def test_pre_epoch_outcomes_are_excluded(monkeypatch):
    monkeypatch.setattr(sl, "KELLY_EPOCH_ISO", "2026-09-29T00:00:00+00:00")
    loop = _loop()
    stale = datetime(2026, 9, 21, 14, 0, tzinfo=timezone.utc)   # the 09-21 session
    loop._sizing_recent_outcomes = [(stale, -0.01) for _ in range(11)]

    loop._prune_kelly_window()
    assert loop._sizing_recent_outcomes == [], (
        "outcomes from before the epoch still count toward the governor"
    )


def test_post_epoch_outcomes_are_kept(monkeypatch):
    monkeypatch.setattr(sl, "KELLY_EPOCH_ISO", "2026-09-29T00:00:00+00:00")
    loop = _loop()
    fresh = datetime.now(timezone.utc) - timedelta(hours=2)
    loop._sizing_recent_outcomes = [(fresh, -0.01) for _ in range(11)]

    loop._prune_kelly_window()
    assert len(loop._sizing_recent_outcomes) == 11


def test_lookback_still_applies_after_the_epoch(monkeypatch):
    """The epoch tightens the window; it must never widen it.

    An outcome older than KELLY_LOOKBACK_DAYS stays excluded even when it
    falls after the epoch.
    """
    monkeypatch.setattr(sl, "KELLY_EPOCH_ISO", "2020-01-01T00:00:00+00:00")
    loop = _loop()
    ancient = datetime.now(timezone.utc) - timedelta(days=sl.KELLY_LOOKBACK_DAYS + 5)
    loop._sizing_recent_outcomes = [(ancient, 0.01)]

    loop._prune_kelly_window()
    assert loop._sizing_recent_outcomes == []


def test_unset_epoch_disables_filtering(monkeypatch):
    monkeypatch.setattr(sl, "KELLY_EPOCH_ISO", "")
    assert sl._kelly_epoch() is None

    loop = _loop()
    recent = datetime.now(timezone.utc) - timedelta(hours=1)
    loop._sizing_recent_outcomes = [(recent, -0.01)]
    loop._prune_kelly_window()
    assert len(loop._sizing_recent_outcomes) == 1


def test_unparseable_epoch_fails_open_to_no_filtering(monkeypatch):
    """A typo must not silently discard the whole Kelly window."""
    monkeypatch.setattr(sl, "KELLY_EPOCH_ISO", "not-a-timestamp")
    assert sl._kelly_epoch() is None


def test_naive_epoch_is_treated_as_utc(monkeypatch):
    monkeypatch.setattr(sl, "KELLY_EPOCH_ISO", "2026-09-29T00:00:00")
    parsed = sl._kelly_epoch()
    assert parsed is not None and parsed.tzinfo is not None


def test_the_hard_block_still_rearms_on_post_epoch_losses(monkeypatch):
    """THE test that makes this a re-seed and not a removal.

    If a fresh, post-epoch sample shows a genuinely negative edge, entries must
    stop again. Otherwise the epoch would have deleted the control rather than
    refreshed its input.
    """
    monkeypatch.setattr(sl, "KELLY_EPOCH_ISO", "2026-09-29T00:00:00+00:00")
    loop = _loop()
    fresh = datetime.now(timezone.utc) - timedelta(hours=1)
    loop._sizing_recent_outcomes = [(fresh, -0.02) for _ in range(sl.KELLY_MIN_TRADES)]
    loop._kelly_fraction = sl.KELLY_HARD_BLOCK_THRESHOLD - 0.01

    assert loop._kelly_mode() != "inactive"
    assert loop._kelly_entries_blocked() is True, (
        "the negative-edge block did not re-arm on a fresh sample — the epoch "
        "removed the control instead of re-seeding it"
    )


def test_a_healthy_post_epoch_sample_does_not_block(monkeypatch):
    monkeypatch.setattr(sl, "KELLY_EPOCH_ISO", "2026-09-29T00:00:00+00:00")
    loop = _loop()
    fresh = datetime.now(timezone.utc) - timedelta(hours=1)
    loop._sizing_recent_outcomes = [(fresh, 0.01) for _ in range(sl.KELLY_MIN_TRADES)]
    loop._kelly_fraction = 0.15

    assert loop._kelly_entries_blocked() is False
