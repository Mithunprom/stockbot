"""Unit tests for H28 — exit-path health monitor.

The defect these guard against: under EXIT_PROFIT_TARGET_MODE=ON (v0.8.0+)
most positions run to EOD without hitting a stop or target. Alpaca paper
closes them at 4pm ET; the Integrity Sentinel (hourly) discovers the
broker-closed positions BEFORE the strategy loop fires session_close, and
stamps them integrity_broker_reconcile. In M3 (n=25), 20/25 exits (80%)
carry that reason, meaning the strategy's own exit path is nearly inactive.

compute_exit_health() surfaces this ratio as a diagnostic metric in
/diagnostics so the operator can see the problem rather than only infer it
from the raw exit_reason distribution.
"""

from __future__ import annotations

import pytest

from src.agents.signal_loop import compute_exit_health


# ── status boundaries ────────────────────────────────────────────────────────

def test_empty_input_returns_no_data():
    health = compute_exit_health([])
    assert health["status"] == "no_data"
    assert health["strategy_exit_rate"] is None
    assert health["n"] == 0


def test_all_strategy_exits_returns_ok():
    reasons = ["stop_loss"] * 5 + ["take_profit"] * 5
    health = compute_exit_health(reasons)
    assert health["status"] == "ok"
    assert health["strategy_exit_rate"] == 1.0
    assert health["reconcile_exits"] == 0
    assert health["strategy_exits"] == 10


def test_all_reconcile_returns_warning():
    reasons = ["integrity_broker_reconcile"] * 15
    health = compute_exit_health(reasons)
    assert health["status"] == "warning"
    assert health["strategy_exit_rate"] == 0.0
    assert health["reconcile_exits"] == 15
    assert health["strategy_exits"] == 0


def test_small_sample_below_threshold_is_insufficient_data():
    # n=5 < 10, rate=0.0 — not enough data to call WARNING
    reasons = ["integrity_broker_reconcile"] * 5
    health = compute_exit_health(reasons)
    assert health["status"] == "insufficient_data"
    assert health["strategy_exit_rate"] == 0.0


def test_small_sample_with_good_rate_is_insufficient_data():
    # n < 10 regardless of rate — sample too small to trust
    reasons = ["stop_loss"] * 9
    health = compute_exit_health(reasons)
    assert health["status"] == "insufficient_data"


def test_boundary_exactly_at_warn_rate_is_ok():
    # 30% strategy rate is the boundary → should be "ok" (>=)
    n_total = 10
    strategy_n = 3   # exactly 0.30
    reasons = (
        ["stop_loss"] * strategy_n
        + ["integrity_broker_reconcile"] * (n_total - strategy_n)
    )
    health = compute_exit_health(reasons)
    assert health["strategy_exit_rate"] == pytest.approx(0.30)
    assert health["status"] == "ok"


def test_just_below_warn_rate_is_warning():
    # 29% strategy rate → WARNING
    n_total = 100
    strategy_n = 29
    reasons = (
        ["stop_loss"] * strategy_n
        + ["integrity_broker_reconcile"] * (n_total - strategy_n)
    )
    health = compute_exit_health(reasons)
    assert health["strategy_exit_rate"] == pytest.approx(0.29)
    assert health["status"] == "warning"


# ── M3 realistic case ────────────────────────────────────────────────────────

def test_realistic_m3_distribution_is_warning():
    """M3 actual: 20 reconcile, 2 take_profit, 1 stop_loss, 1 session_close, 1 max_hold."""
    reasons = (
        ["integrity_broker_reconcile"] * 20
        + ["take_profit"] * 2
        + ["stop_loss"]
        + ["session_close"]
        + ["max_hold"]
    )
    health = compute_exit_health(reasons)
    assert health["n"] == 25
    assert health["strategy_exit_rate"] == pytest.approx(5 / 25)
    assert health["status"] == "warning"
    assert health["reconcile_exits"] == 20
    assert health["strategy_exits"] == 5


# ── distribution accuracy ────────────────────────────────────────────────────

def test_distribution_counts_every_reason():
    reasons = ["stop_loss"] * 3 + ["take_profit"] * 2 + ["integrity_broker_reconcile"] * 7
    health = compute_exit_health(reasons)
    assert health["distribution"]["stop_loss"] == 3
    assert health["distribution"]["take_profit"] == 2
    assert health["distribution"]["integrity_broker_reconcile"] == 7
    assert health["n"] == 12


def test_unknown_reasons_count_as_strategy():
    """Unrecognised exit reasons are not reconcile — they count toward strategy."""
    reasons = ["unknown_reason"] * 10 + ["integrity_broker_reconcile"] * 5
    health = compute_exit_health(reasons)
    assert health["reconcile_exits"] == 5
    assert health["strategy_exits"] == 10
    assert health["strategy_exit_rate"] == pytest.approx(10 / 15, abs=0.001)


# ── return structure completeness ─────────────────────────────────────────────

def test_return_dict_has_all_required_keys():
    required = {"n", "strategy_exit_rate", "strategy_exits", "reconcile_exits",
                "status", "distribution"}
    for sample in ([], ["stop_loss"], ["integrity_broker_reconcile"] * 20):
        health = compute_exit_health(sample)
        assert required.issubset(health.keys()), (
            f"Missing keys for sample len={len(sample)}: {required - health.keys()}"
        )
