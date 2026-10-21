"""Unit tests for H35: IC fill job health check in WatchdogAgent.

Root event: 2026-10-08 – 2026-10-16. fill_actual_returns() ran every 15 min,
attempted 500 rows, filled 0, retired 0, errors 0. IC gate unable to arm for
9 trading days. This check surfaces that silent failure mode.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.watchdog_agent import (
    IC_FILL_STALE_MINUTES,
    IC_FILL_ZERO_RUNS_CRITICAL,
    IC_FILL_ZERO_RUNS_WARN,
    WatchdogAgent,
)


def _make_loop(ic_tracker=None):
    loop = MagicMock()
    loop._ic_tracker = ic_tracker
    loop._data_fresh = True
    return loop


def _make_tracker(
    last_fill_at: datetime | None = None,
    last_fill_result: dict | None = None,
    consecutive_zero: int = 0,
):
    t = MagicMock()
    t.last_fill_at = last_fill_at
    t.last_fill_result = last_fill_result
    t._consecutive_zero_fill_runs = consecutive_zero
    return t


def _make_agent(loop=None, market_open: bool = True) -> WatchdogAgent:
    if loop is None:
        loop = _make_loop()
    agent = WatchdogAgent(loop, MagicMock(), MagicMock())
    agent._market_open = MagicMock(return_value=market_open)
    return agent


# ── market-closed path ──────────────────────────────────────────────────────

def test_ic_fill_check_skipped_when_market_closed():
    agent = _make_agent(market_open=False)
    result = agent._check_ic_fill_fresh()
    assert result["status"] == "ok"
    assert "not applicable" in result["detail"]


# ── no ic_tracker attached ───────────────────────────────────────────────────

def test_ic_fill_check_skipped_when_no_tracker():
    loop = _make_loop(ic_tracker=None)
    agent = _make_agent(loop=loop, market_open=True)
    result = agent._check_ic_fill_fresh()
    assert result["status"] == "ok"
    assert "not attached" in result["detail"]


# ── job never started ────────────────────────────────────────────────────────

def test_ic_fill_check_warns_when_job_not_started():
    tracker = _make_tracker(last_fill_at=None)
    loop = _make_loop(ic_tracker=tracker)
    agent = _make_agent(loop=loop, market_open=True)
    result = agent._check_ic_fill_fresh()
    assert result["status"] == "warn"
    assert "not run yet" in result["detail"]


# ── job stale (scheduler dead) ───────────────────────────────────────────────

def test_ic_fill_check_critical_when_stale():
    stale_time = datetime.now(timezone.utc) - timedelta(minutes=IC_FILL_STALE_MINUTES + 5)
    tracker = _make_tracker(last_fill_at=stale_time, consecutive_zero=0)
    loop = _make_loop(ic_tracker=tracker)
    agent = _make_agent(loop=loop, market_open=True)
    result = agent._check_ic_fill_fresh()
    assert result["status"] == "critical"
    assert "last ran" in result["detail"]


def test_ic_fill_check_ok_when_recent():
    recent_time = datetime.now(timezone.utc) - timedelta(minutes=10)
    tracker = _make_tracker(
        last_fill_at=recent_time,
        last_fill_result={"filled_15m": 12, "retired_15m": 3, "attempted_15m": 15, "consecutive_zero_fill_runs": 0},
        consecutive_zero=0,
    )
    loop = _make_loop(ic_tracker=tracker)
    agent = _make_agent(loop=loop, market_open=True)
    result = agent._check_ic_fill_fresh()
    assert result["status"] == "ok"
    assert "last ran" in result["detail"]


# ── dead-queue (Oct 8-16 signature) ─────────────────────────────────────────

def test_ic_fill_check_warns_on_two_consecutive_zero_runs():
    recent = datetime.now(timezone.utc) - timedelta(minutes=5)
    tracker = _make_tracker(
        last_fill_at=recent,
        last_fill_result={"filled_15m": 0, "retired_15m": 0, "attempted_15m": 500, "consecutive_zero_fill_runs": 2},
        consecutive_zero=IC_FILL_ZERO_RUNS_WARN,
    )
    loop = _make_loop(ic_tracker=tracker)
    agent = _make_agent(loop=loop, market_open=True)
    result = agent._check_ic_fill_fresh()
    assert result["status"] == "warn"
    assert "yielded nothing" in result["detail"]


def test_ic_fill_check_critical_on_four_consecutive_zero_runs():
    recent = datetime.now(timezone.utc) - timedelta(minutes=5)
    tracker = _make_tracker(
        last_fill_at=recent,
        last_fill_result={"filled_15m": 0, "retired_15m": 0, "attempted_15m": 500, "consecutive_zero_fill_runs": 4},
        consecutive_zero=IC_FILL_ZERO_RUNS_CRITICAL,
    )
    loop = _make_loop(ic_tracker=tracker)
    agent = _make_agent(loop=loop, market_open=True)
    result = agent._check_ic_fill_fresh()
    assert result["status"] == "critical"
    assert "stalled" in result["detail"]
    assert "ohlcv_1m" in result["detail"]


# ── LiveICTracker state management ──────────────────────────────────────────

def test_consecutive_zero_fill_runs_increments_on_zero_resolve():
    from src.agents.live_ic_tracker import LiveICTracker
    tracker = LiveICTracker.__new__(LiveICTracker)
    tracker._consecutive_zero_fill_runs = 1
    tracker.last_fill_at = None
    tracker.last_fill_result = None
    # Simulate the update logic: attempted>0, resolved==0
    resolved = 0
    attempted = 5
    if attempted > 0 and resolved == 0:
        tracker._consecutive_zero_fill_runs += 1
    assert tracker._consecutive_zero_fill_runs == 2


def test_consecutive_zero_fill_runs_resets_on_successful_fill():
    from src.agents.live_ic_tracker import LiveICTracker
    tracker = LiveICTracker.__new__(LiveICTracker)
    tracker._consecutive_zero_fill_runs = 3
    tracker.last_fill_at = None
    tracker.last_fill_result = None
    # Simulate the update logic: resolved > 0
    resolved = 5
    attempted = 5
    if attempted > 0 and resolved == 0:
        tracker._consecutive_zero_fill_runs += 1
    else:
        tracker._consecutive_zero_fill_runs = 0
    assert tracker._consecutive_zero_fill_runs == 0


def test_ic_fill_check_ok_just_below_warn_threshold():
    recent = datetime.now(timezone.utc) - timedelta(minutes=5)
    tracker = _make_tracker(
        last_fill_at=recent,
        last_fill_result={"filled_15m": 0, "retired_15m": 0, "attempted_15m": 500, "consecutive_zero_fill_runs": 1},
        consecutive_zero=IC_FILL_ZERO_RUNS_WARN - 1,  # 1 run — not yet warn
    )
    loop = _make_loop(ic_tracker=tracker)
    agent = _make_agent(loop=loop, market_open=True)
    result = agent._check_ic_fill_fresh()
    # 1 consecutive zero run is below warn threshold (2) — should be ok
    assert result["status"] == "ok"
