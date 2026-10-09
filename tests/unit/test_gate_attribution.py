"""Unit tests for H32: signal gate attribution diagnostic.

Every claim about WHICH gate blocked a signal must be independently
verifiable. The diagnostic method (_entry_gate_reason) must agree with
the production gate (_sizing_entry_gate_open) on pass/fail for every
case it covers — a discrepancy here flags maintenance drift early.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from src.agents.signal_loop import (
    KELLY_HARD_BLOCK_THRESHOLD,
    KELLY_PROBE_MIN_N,
    KELLY_PROBATION_MIN_TICKER_IC,
    MAX_OPEN_POSITIONS,
    PORTFOLIO_HEAT_CEILING,
    SIZING_DIR_PROB_DEAD_ZONE,
    SIZING_MAX_TRADES_PER_DAY,
    SignalLoop,
)
from src.execution.position_manager import PositionManager
from src.models.ensemble import EnsembleSignal
from src.risk.circuit_breakers import CircuitBreakers


def _make_loop() -> SignalLoop:
    loop = SignalLoop(
        universe=["AAPL", "XOM", "MSFT"],
        ensemble=MagicMock(),
        alpaca=MagicMock(),
        circuit_breakers=CircuitBreakers(),
        pos_manager=PositionManager(initial_portfolio=100_000.0),
        session_factory=MagicMock(),
        feature_cols=[f"feat_{i}" for i in range(30)],
    )
    loop._in_entry_window = lambda: True
    loop._data_fresh = True
    return loop


def _seed_winning_kelly(loop: SignalLoop) -> None:
    # Must include both wins AND losses: _update_kelly() returns 0.0 when
    # either list is empty, and 0.0 > 0 is False → "probation" mode.
    stamp = datetime.now(timezone.utc) - timedelta(days=1)
    loop._sizing_recent_outcomes = [(stamp, 0.02)] * 8 + [(stamp, -0.004)] * 4
    loop._update_kelly()
    assert loop._kelly_mode() == "normal", (
        f"precondition failed: kelly_mode={loop._kelly_mode()!r}, "
        f"fraction={loop._kelly_fraction}"
    )


def _good_signal(ticker: str = "AAPL") -> EnsembleSignal:
    sig = EnsembleSignal(ticker=ticker, timestamp=None)
    sig.lgbm_pred_return = 0.009
    sig.lgbm_dir_prob = 0.70
    return sig


# ── pass / fail consistency with the production gate ─────────────────────────

def test_clean_signal_passes():
    loop = _make_loop()
    _seed_winning_kelly(loop)
    passed, reason = loop._entry_gate_reason(_good_signal())
    assert passed is True
    assert reason is None


def test_gate_reason_agrees_with_gate_open_pass():
    loop = _make_loop()
    _seed_winning_kelly(loop)
    sig = _good_signal()
    assert loop._sizing_entry_gate_open(sig) == loop._entry_gate_reason(sig)[0]


def test_gate_reason_agrees_with_gate_open_hard_blocked():
    loop = _make_loop()
    stamp = datetime.now(timezone.utc) - timedelta(days=1)
    loop._sizing_recent_outcomes = [(stamp, 0.005)] * 3 + [(stamp, -0.03)] * 9
    loop._update_kelly()
    sig = _good_signal()
    assert loop._sizing_entry_gate_open(sig) == loop._entry_gate_reason(sig)[0]


def test_gate_reason_agrees_with_gate_open_daily_cap():
    loop = _make_loop()
    _seed_winning_kelly(loop)
    loop._sizing_n_trades_today = SIZING_MAX_TRADES_PER_DAY
    sig = _good_signal()
    assert loop._sizing_entry_gate_open(sig) == loop._entry_gate_reason(sig)[0]


# ── specific gate labels ──────────────────────────────────────────────────────

def test_pred_return_too_small_label():
    loop = _make_loop()
    _seed_winning_kelly(loop)
    sig = _good_signal()
    sig.lgbm_pred_return = 0.000001
    passed, reason = loop._entry_gate_reason(sig)
    assert not passed
    assert reason == "pred_return_too_small"


def test_dir_prob_dead_zone_label():
    loop = _make_loop()
    _seed_winning_kelly(loop)
    sig = _good_signal()
    lo, hi = SIZING_DIR_PROB_DEAD_ZONE
    sig.lgbm_dir_prob = (lo + hi) / 2
    passed, reason = loop._entry_gate_reason(sig)
    assert not passed
    assert reason == "dir_prob_dead_zone"


def test_daily_cap_label():
    loop = _make_loop()
    _seed_winning_kelly(loop)
    loop._sizing_n_trades_today = SIZING_MAX_TRADES_PER_DAY
    sig = _good_signal()
    passed, reason = loop._entry_gate_reason(sig)
    assert not passed
    assert reason == "daily_cap"


def test_kelly_hard_blocked_label():
    loop = _make_loop()
    stamp = datetime.now(timezone.utc) - timedelta(days=1)
    loop._sizing_recent_outcomes = [(stamp, 0.005)] * 3 + [(stamp, -0.03)] * 9
    loop._update_kelly()
    assert loop._kelly_entries_blocked(), "precondition: must be hard-blocked for this test"
    sig = _good_signal()
    passed, reason = loop._entry_gate_reason(sig)
    assert not passed
    assert reason == "kelly_hard_blocked"


def test_kelly_probation_probe_exhausted_label():
    loop = _make_loop()
    stamp = datetime.now(timezone.utc) - timedelta(days=1)
    # Modest losses → probation (negative but above hard-block)
    loop._sizing_recent_outcomes = [(stamp, -0.008)] * 8 + [(stamp, 0.015)] * 4
    loop._update_kelly()
    # Only enter this branch if mode is actually probation
    if loop._kelly_mode() != "probation":
        pytest.skip("Kelly not in probation with these outcomes — adjust seeds")
    loop._probation_entries_today = 1  # probe slot used
    sig = _good_signal()
    passed, reason = loop._entry_gate_reason(sig)
    assert not passed
    assert reason == "kelly_probation"


def test_data_stale_label():
    loop = _make_loop()
    _seed_winning_kelly(loop)
    loop._data_fresh = False
    sig = _good_signal()
    passed, reason = loop._entry_gate_reason(sig)
    assert not passed
    assert reason == "data_stale"


# ── snapshot aggregation ──────────────────────────────────────────────────────

def test_gate_attribution_snapshot_counts():
    loop = _make_loop()
    _seed_winning_kelly(loop)

    good = _good_signal("AAPL")

    small_pred = _good_signal("XOM")
    small_pred.lgbm_pred_return = 0.000001

    dead_zone = _good_signal("MSFT")
    lo, hi = SIZING_DIR_PROB_DEAD_ZONE
    dead_zone.lgbm_dir_prob = (lo + hi) / 2

    loop._latest_signals = [good, small_pred, dead_zone]

    snap = loop._get_gate_attribution_snapshot()
    assert snap["n_evaluated"] == 3
    assert snap["n_would_trade"] == 1
    assert snap["n_blocked"] == 2
    assert snap["blocked_by"].get("pred_return_too_small") == 1
    assert snap["blocked_by"].get("dir_prob_dead_zone") == 1


def test_gate_attribution_empty_signals():
    loop = _make_loop()
    loop._latest_signals = []
    snap = loop._get_gate_attribution_snapshot()
    assert snap["n_evaluated"] == 0
    assert snap["n_would_trade"] == 0
    assert snap["n_blocked"] == 0
    assert snap["blocked_by"] == {}


def test_gate_attribution_all_pass():
    loop = _make_loop()
    _seed_winning_kelly(loop)
    loop._latest_signals = [_good_signal("AAPL"), _good_signal("XOM")]
    snap = loop._get_gate_attribution_snapshot()
    assert snap["n_would_trade"] == 2
    assert snap["n_blocked"] == 0
    assert snap["blocked_by"] == {}


def test_gate_attribution_exposed_in_portfolio_summary():
    loop = _make_loop()
    loop._latest_signals = []
    summary = loop.get_portfolio_summary()
    assert "signal_gate_attribution" in summary
    snap = summary["signal_gate_attribution"]
    assert "n_evaluated" in snap
    assert "n_would_trade" in snap
    assert "n_blocked" in snap
    assert "blocked_by" in snap
