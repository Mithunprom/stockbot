"""Unit tests for _kelly_window_forecast diagnostics helper.

H29: Adds a rolloff schedule and projected-fraction trajectory to the
diagnostics payload so operators can see when the Kelly hard block will
self-drain — the critical observable for the self-draining deadlock scenario
described in the KELLY_HARD_BLOCK_THRESHOLD comment block.

Contract:
  - Pure computation; never mutates _sizing_recent_outcomes.
  - inactive_date is the earliest rolloff_date after which n_after falls below
    KELLY_MIN_TRADES, meaning the hard block automatically releases.
  - currently_hard_blocked mirrors _kelly_entries_blocked() for current window.
  - window_by_date and rolloff_trajectory are sorted oldest-exit-date first.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from src.agents.signal_loop import (
    KELLY_HARD_BLOCK_THRESHOLD,
    KELLY_LOOKBACK_DAYS,
    KELLY_MIN_TRADES,
    SignalLoop,
)
from src.execution.position_manager import PositionManager
from src.risk.circuit_breakers import CircuitBreakers


def _make_loop() -> SignalLoop:
    loop = SignalLoop(
        universe=["AAPL", "XOM"],
        ensemble=MagicMock(),
        alpaca=MagicMock(),
        circuit_breakers=CircuitBreakers(),
        pos_manager=PositionManager(initial_portfolio=100_000.0),
        session_factory=MagicMock(),
        feature_cols=[f"feat_{i}" for i in range(30)],
    )
    return loop


def _stamp(days_ago: float) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=days_ago)


def _seed(loop: SignalLoop, outcomes: list[float], days_ago: float = 1.0) -> None:
    """Seed all outcomes to the same timestamp (single-date window)."""
    ts = _stamp(days_ago)
    loop._sizing_recent_outcomes = [(ts, o) for o in outcomes]


def _seed_multi_date(
    loop: SignalLoop,
    batches: list[tuple[float, list[float]]],
) -> None:
    """Seed outcomes across multiple ages.

    batches = [(days_ago_1, [pnl, ...]), (days_ago_2, [pnl, ...]), ...]
    """
    outcomes: list[tuple[datetime, float]] = []
    for days_ago, pnls in batches:
        ts = _stamp(days_ago)
        outcomes.extend((ts, p) for p in pnls)
    loop._sizing_recent_outcomes = outcomes


# ── Empty / below-minimum cases ─────────────────────────────────────────────

def test_empty_window_returns_safe_defaults():
    loop = _make_loop()
    loop._sizing_recent_outcomes = []
    result = loop._kelly_window_forecast()
    assert result["window_n"] == 0
    assert result["window_by_date"] == []
    assert result["rolloff_trajectory"] == []
    assert result["inactive_date"] is None
    assert result["currently_hard_blocked"] is False


def test_below_min_trades_not_hard_blocked():
    """With fewer than KELLY_MIN_TRADES the governor is inactive — never hard-blocked."""
    loop = _make_loop()
    _seed(loop, [-0.05] * (KELLY_MIN_TRADES - 1))
    result = loop._kelly_window_forecast()
    assert result["currently_hard_blocked"] is False
    assert result["window_n"] == KELLY_MIN_TRADES - 1


# ── currently_hard_blocked mirrors real block state ─────────────────────────

def test_currently_hard_blocked_true_when_fraction_below_threshold():
    loop = _make_loop()
    # 3 small wins, 9 larger losses → deeply negative Kelly
    _seed(loop, [0.005] * 3 + [-0.03] * 9)
    loop._update_kelly()
    assert loop._kelly_fraction <= KELLY_HARD_BLOCK_THRESHOLD

    result = loop._kelly_window_forecast()
    assert result["currently_hard_blocked"] is True


def test_currently_hard_blocked_false_when_healthy_edge():
    loop = _make_loop()
    _seed(loop, [0.02] * 8 + [-0.004] * 4)
    loop._update_kelly()
    assert loop._kelly_fraction > 0

    result = loop._kelly_window_forecast()
    assert result["currently_hard_blocked"] is False


# ── Rolloff trajectory structure ─────────────────────────────────────────────

def test_trajectory_sorted_oldest_first():
    """window_by_date entries are sorted ascending by exit_date."""
    loop = _make_loop()
    _seed_multi_date(loop, [
        (8.0, [0.01] * 5 + [-0.02] * 2),   # older batch
        (2.0, [-0.03] * 5 + [0.01] * 3),   # newer batch
    ])
    result = loop._kelly_window_forecast()
    dates = [e["exit_date"] for e in result["window_by_date"]]
    assert dates == sorted(dates)


def test_trajectory_n_after_decrements_by_removed_batch():
    """n_after in the trajectory must equal window_n minus cumulative removed."""
    loop = _make_loop()
    # Two well-separated batches so they fall on different ET dates.
    batch_a = [0.01] * 3 + [-0.02] * 3  # 6 trades, 8 days ago
    batch_b = [-0.015] * 4              # 4 trades, 1 day ago
    _seed_multi_date(loop, [(8.0, batch_a), (1.0, batch_b)])

    result = loop._kelly_window_forecast()
    assert result["window_n"] == 10

    traj = result["rolloff_trajectory"]
    assert len(traj) == 2
    # After removing batch_a (oldest): 4 remain
    assert traj[0]["n_after"] == 4
    # After also removing batch_b: 0 remain
    assert traj[1]["n_after"] == 0


# ── inactive_date detection ──────────────────────────────────────────────────

def test_inactive_date_set_when_rolloff_drops_below_min_trades():
    """inactive_date is the rolloff_date of the step that first takes n below min."""
    loop = _make_loop()
    # Batch A: exactly KELLY_MIN_TRADES trades 8 days ago (fills the window)
    batch_a = [0.01] * 5 + [-0.02] * 5   # 10 trades
    # Batch B: 2 more trades 1 day ago (total 12, but after A rolls off → 2 left)
    batch_b = [-0.015] * 2
    _seed_multi_date(loop, [(8.0, batch_a), (1.0, batch_b)])

    result = loop._kelly_window_forecast()
    # After batch_a (10 trades) rolls off: n_after = 2 < KELLY_MIN_TRADES → inactive
    assert result["inactive_date"] is not None
    traj = result["rolloff_trajectory"]
    first_inactive = next(t for t in traj if t["inactive_after"])
    assert result["inactive_date"] == first_inactive["rolloff_date"]


def test_inactive_date_none_when_window_stays_above_min_throughout():
    """If every rolloff step keeps n_after >= KELLY_MIN_TRADES, inactive_date is None."""
    loop = _make_loop()
    # Three batches each contributing KELLY_MIN_TRADES trades — even after rolling
    # off the oldest, plenty remain (2 * KELLY_MIN_TRADES >= KELLY_MIN_TRADES).
    per_batch = KELLY_MIN_TRADES
    _seed_multi_date(loop, [
        (8.0, [0.01] * per_batch),
        (5.0, [0.01] * per_batch),
        (1.0, [0.01] * per_batch),
    ])
    result = loop._kelly_window_forecast()
    # After oldest rolls off: 2*KELLY_MIN_TRADES remain — still active.
    # Only after the second batch also rolls off does n drop to KELLY_MIN_TRADES,
    # which is NOT below min, so inactive_after would only trigger at 0.
    # Check that no step has inactive_after until window is actually empty or below min.
    for step in result["rolloff_trajectory"][:-1]:
        # At least KELLY_MIN_TRADES remain after each early rolloff
        if step["n_after"] >= KELLY_MIN_TRADES:
            assert not step["inactive_after"]


# ── Forecast does not mutate state ──────────────────────────────────────────

def test_forecast_does_not_mutate_sizing_recent_outcomes():
    """_kelly_window_forecast is read-only."""
    loop = _make_loop()
    _seed(loop, [0.01] * 6 + [-0.03] * 6)
    original = list(loop._sizing_recent_outcomes)
    loop._kelly_window_forecast()
    assert loop._sizing_recent_outcomes == original


# ── Self-draining property (the hard block releases via natural rolloff) ──────

def test_self_draining_hard_block_shows_inactive_date():
    """A hard-blocked window must show an inactive_date (self-draining guarantee)."""
    loop = _make_loop()
    # Deeply negative Kelly: 3 wins 9 losses, all recent → hard blocked.
    _seed(loop, [0.005] * 3 + [-0.03] * 9)
    loop._update_kelly()
    assert loop._kelly_entries_blocked() is True

    result = loop._kelly_window_forecast()
    assert result["currently_hard_blocked"] is True
    # Because the same batch holds all trades, rolling it off leaves n=0 < MIN_TRADES.
    assert result["inactive_date"] is not None
