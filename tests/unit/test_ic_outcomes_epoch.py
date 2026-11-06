"""IC_OUTCOMES_EPOCH — bounded bypass of the contaminated-prediction IC block.

19 tickers on the block list in the Nov 6 snapshot all accumulated n ≥ 300 from
the contaminated serving window (H36 not yet deployed; serving IC ≈ −0.010 vs
full-history IC ≈ +0.181 — the train/serve IC skew from mtf_trend_60m EMA
initial contamination). Setting IC_OUTCOMES_EPOCH to the H36 deploy date lets
the IC-block gate re-arm cleanly on post-fix predictions without touching data.

The property that makes this acceptable: the LAST test here verifies that the
IC block still re-arms on fresh contaminated outcomes after the epoch. An epoch
that permanently defanged the block would be removing the control, not re-seeding
it.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents import signal_loop as sl


# ---------------------------------------------------------------------------
# _ic_outcomes_epoch() parsing
# ---------------------------------------------------------------------------


def test_unset_epoch_disables_filtering(monkeypatch):
    monkeypatch.setattr(sl, "IC_OUTCOMES_EPOCH_ISO", "")
    assert sl._ic_outcomes_epoch() is None


def test_unparseable_epoch_fails_open(monkeypatch):
    """A typo must not silently change the IC window."""
    monkeypatch.setattr(sl, "IC_OUTCOMES_EPOCH_ISO", "not-a-timestamp")
    assert sl._ic_outcomes_epoch() is None


def test_naive_epoch_is_treated_as_utc(monkeypatch):
    monkeypatch.setattr(sl, "IC_OUTCOMES_EPOCH_ISO", "2026-11-01T00:00:00")
    parsed = sl._ic_outcomes_epoch()
    assert parsed is not None and parsed.tzinfo is not None


def test_epoch_parsed_correctly(monkeypatch):
    monkeypatch.setattr(sl, "IC_OUTCOMES_EPOCH_ISO", "2026-11-10T00:00:00+00:00")
    parsed = sl._ic_outcomes_epoch()
    assert parsed == datetime(2026, 11, 10, 0, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# _maybe_refresh_ticker_ic — gate receives epoch filter, probe does not
# ---------------------------------------------------------------------------


def _make_loop(kelly_mode: str = "healthy") -> object:
    """Minimal fake SignalLoop exposing only _maybe_refresh_ticker_ic."""

    class _L:
        _ic_refresh_countdown: int = 1  # decrements to 0 → triggers on first call
        _ticker_ic: dict = {}
        _ticker_ic_probe: dict = {}
        _mock_tracker: object = None

        @property
        def _ic_tracker(self):  # type: ignore[override]
            return self._mock_tracker

        def _kelly_mode(self) -> str:
            return kelly_mode

        _maybe_refresh_ticker_ic = sl.SignalLoop._maybe_refresh_ticker_ic

    return _L()


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_ic_block_gate_receives_epoch_as_since(monkeypatch):
    epoch_str = "2026-11-01T00:00:00+00:00"
    monkeypatch.setattr(sl, "IC_OUTCOMES_EPOCH_ISO", epoch_str)
    expected = datetime(2026, 11, 1, 0, 0, 0, tzinfo=timezone.utc)

    tracker = MagicMock()
    tracker._compute_per_ticker_ic = AsyncMock(return_value={})

    loop = _make_loop()
    loop._mock_tracker = tracker

    _run(loop._maybe_refresh_ticker_ic())

    tracker._compute_per_ticker_ic.assert_called_once_with(
        window_days=sl.TICKER_IC_WINDOW_DAYS,
        since=expected,
    )


def test_probe_uses_no_epoch_filter_when_epoch_set(monkeypatch):
    """The probe must NOT receive IC_OUTCOMES_EPOCH — it needs the full window.

    If the probe inherits the epoch filter, its eligible-ticker count shrinks,
    potentially re-deadlocking the Kelly-probation governor (the same deadlock
    that required KELLY_PROBE_MIN_N to be lowered in September).
    """
    monkeypatch.setattr(sl, "IC_OUTCOMES_EPOCH_ISO", "2026-11-01T00:00:00+00:00")

    tracker = MagicMock()
    tracker._compute_per_ticker_ic = AsyncMock(return_value={})

    loop = _make_loop(kelly_mode="probation")
    loop._mock_tracker = tracker

    _run(loop._maybe_refresh_ticker_ic())

    # Two calls: IC-block gate (with since), then probe (no since).
    assert tracker._compute_per_ticker_ic.call_count == 2, (
        "expected two separate queries when IC_OUTCOMES_EPOCH is set in probation"
    )
    probe_kwargs = tracker._compute_per_ticker_ic.call_args_list[1].kwargs
    assert "since" not in probe_kwargs, (
        "IC probe received an epoch filter — it must use the full window to avoid "
        "re-deadlocking the Kelly-probation governor"
    )


def test_probe_reuses_gate_query_when_no_epoch(monkeypatch):
    """When IC_OUTCOMES_EPOCH is unset and both windows are equal, one DB query fires."""
    monkeypatch.setattr(sl, "IC_OUTCOMES_EPOCH_ISO", "")

    tracker = MagicMock()
    tracker._compute_per_ticker_ic = AsyncMock(return_value={})

    loop = _make_loop(kelly_mode="probation")
    loop._mock_tracker = tracker

    _run(loop._maybe_refresh_ticker_ic())

    # Same window + no epoch → both gates share one round-trip.
    assert tracker._compute_per_ticker_ic.call_count == 1, (
        "expected probe to reuse the gate query when IC_OUTCOMES_EPOCH is unset"
    )


# ---------------------------------------------------------------------------
# Control must re-arm on post-epoch contaminated outcomes (key safety property)
# ---------------------------------------------------------------------------


def test_ic_block_still_rearms_on_fresh_contaminated_outcomes(monkeypatch):
    """THE test: IC_OUTCOMES_EPOCH must re-seed the control, not remove it.

    Post-epoch predictions that happen to be near-random must still arm the block
    once n ≥ TICKER_IC_MIN_N and ic < TICKER_IC_BLOCK_THRESHOLD.
    """
    monkeypatch.setattr(sl, "IC_OUTCOMES_EPOCH_ISO", "2026-11-01T00:00:00+00:00")

    fresh_bad_ic = sl.TICKER_IC_BLOCK_THRESHOLD - 0.01   # e.g. −0.06, below threshold
    loop = _make_loop()
    loop._ticker_ic = {"AAPL": (fresh_bad_ic, sl.TICKER_IC_MIN_N + 10)}

    assert sl.SignalLoop._ticker_ic_blocked(loop, "AAPL") is True, (
        "IC block did not re-arm on fresh contaminated predictions — epoch removed "
        "the control instead of re-seeding it"
    )


def test_ic_block_does_not_fire_on_post_epoch_healthy_ic(monkeypatch):
    monkeypatch.setattr(sl, "IC_OUTCOMES_EPOCH_ISO", "2026-11-01T00:00:00+00:00")

    healthy_ic = sl.TICKER_IC_MIN_ENTRY + 0.05   # positive IC, above entry threshold
    loop = _make_loop()
    loop._ticker_ic = {"AAPL": (healthy_ic, sl.TICKER_IC_MIN_N + 10)}

    assert sl.SignalLoop._ticker_ic_blocked(loop, "AAPL") is False
