"""H30: Post-halt recovery size ramp.

After the circuit breaker is lifted, the first HALT_RECOVERY_N_TRADES entries
are sized at HALT_RECOVERY_SIZE_MULT of normal to prevent immediately
re-triggering the halt on a still-impaired account.

Root event: 2026-09-30 — Kelly rolled off after Sep 21 losses, bot resumed at
full size, triggered max_drawdown halt within the same trading session (6 new
full-size positions opened before halt fired).  H30 makes that scenario cost
half as much capital.
"""

from __future__ import annotations

import pytest

from src.execution.position_sizer import SmartPositionSizer
from src.risk.circuit_breakers import (
    CircuitBreakers,
    HALT_RECOVERY_N_TRADES,
    HALT_RECOVERY_SIZE_MULT,
)
from src.risk.risk_state_store import RiskStateSnapshot


# ─── CircuitBreakers behaviour ───────────────────────────────────────────────


def test_post_halt_mult_is_1_by_default():
    cb = CircuitBreakers()
    assert cb.post_halt_size_mult == 1.0


def test_resume_trading_arms_recovery_counter():
    cb = CircuitBreakers()
    cb.resume_trading(authorized_by="owner")
    assert cb.post_halt_size_mult == HALT_RECOVERY_SIZE_MULT


def test_recovery_counter_decrements_to_full_size():
    cb = CircuitBreakers()
    cb.resume_trading(authorized_by="owner")
    for i in range(HALT_RECOVERY_N_TRADES):
        assert cb.post_halt_size_mult == HALT_RECOVERY_SIZE_MULT, (
            f"Expected reduced mult at trade {i}"
        )
        cb.record_post_halt_entry()
    assert cb.post_halt_size_mult == 1.0


def test_record_entry_noop_outside_recovery():
    cb = CircuitBreakers()
    cb.record_post_halt_entry()   # no recovery counter → no-op, no raise
    assert cb.post_halt_size_mult == 1.0


def test_restore_post_halt_counter_roundtrip():
    cb = CircuitBreakers()
    cb.restore_post_halt_counter(3)
    assert cb.post_halt_size_mult == HALT_RECOVERY_SIZE_MULT
    cb.record_post_halt_entry()
    cb.record_post_halt_entry()
    cb.record_post_halt_entry()
    assert cb.post_halt_size_mult == 1.0


def test_restore_negative_clamps_to_zero():
    cb = CircuitBreakers()
    cb.restore_post_halt_counter(-5)
    assert cb.post_halt_size_mult == 1.0


# ─── SmartPositionSizer integration ──────────────────────────────────────────


def _size(halt_recovery_mult: float = 1.0, portfolio_value: float = 98_000.0):
    return SmartPositionSizer(mode="paper").compute(
        ticker="AAPL",
        dir_prob=0.72,
        pred_return=0.009,
        atr_pct=0.0006,
        price=150.0,
        portfolio_value=portfolio_value,
        portfolio_heat=0.0,
        sector_notionals={},
        kelly_fraction=0.0,
        halt_recovery_mult=halt_recovery_mult,
    )


def test_sizing_halved_during_recovery():
    # Use a small portfolio ($7 000) so pre-cap notional stays below the hard
    # $2 500 per-position cap, meaning the halt_recovery_mult is the binding
    # constraint and both sizes remain above the $1 000 minimum viable floor.
    r_normal = _size(halt_recovery_mult=1.0, portfolio_value=7_000.0)
    r_recovery = _size(halt_recovery_mult=HALT_RECOVERY_SIZE_MULT, portfolio_value=7_000.0)
    assert r_normal is not None and r_recovery is not None
    # stage4_constraint_pct is recorded before the hard-dollar cap, so the
    # 50% ramp is exactly visible there regardless of account size.
    ratio_stage4 = r_recovery.stage4_constraint_pct / r_normal.stage4_constraint_pct
    assert abs(ratio_stage4 - HALT_RECOVERY_SIZE_MULT) < 0.001
    # Final notional should also be roughly halved (allow ±3% for share rounding)
    ratio_notional = r_recovery.notional / r_normal.notional
    assert abs(ratio_notional - HALT_RECOVERY_SIZE_MULT) < 0.03


def test_halt_recovery_mult_in_audit_dict():
    r = _size(halt_recovery_mult=HALT_RECOVERY_SIZE_MULT)
    assert r is not None
    d = r.to_dict()
    assert d["stages"]["4c_halt_ramp"] == pytest.approx(HALT_RECOVERY_SIZE_MULT, abs=0.001)


# ─── RiskStateSnapshot persistence ───────────────────────────────────────────


def test_snapshot_persists_post_halt_trades_remaining():
    snap = RiskStateSnapshot(
        peak_value=100_000.0,
        daily_start_value=100_000.0,
        daily_start_date="2026-10-01",
        consecutive_losses=0,
        post_halt_trades_remaining=4,
    )
    back = RiskStateSnapshot.from_json(snap.to_json())
    assert back is not None
    assert back.post_halt_trades_remaining == 4


def test_snapshot_backward_compat_default_zero():
    """Old snapshots without the field deserialize to post_halt_trades_remaining=0."""
    raw = (
        '{"version": 1, "peak_value": 100000.0, "daily_start_value": 100000.0, '
        '"daily_start_date": "2026-10-01", "consecutive_losses": 0, '
        '"halted": false, "halt_reason": "", "halt_time": null}'
    )
    snap = RiskStateSnapshot.from_json(raw)
    assert snap is not None
    assert snap.post_halt_trades_remaining == 0
