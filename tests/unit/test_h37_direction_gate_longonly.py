"""H37: Long-only direction gate — _sizing_entry_gate_open requires dir_prob >= 0.60.

The prior implementation used `not (0.40 < dir_prob < 0.60)`, which passed
both bullish (dir_prob > 0.60) and strongly bearish (dir_prob < 0.40) signals.
Bearish signals were blocked downstream by the Phase-5 short-disable check, but:
  - The gate comment said "long entries need P(up) >= 0.60" — wrong in code
  - /diagnostics `signals_passing` counted bearish signals as candidates
  - `would_trade=true` appeared for dir_prob=0.27 tickers in Nov 4 diagnostics

Fix: `dir_prob >= hi` (>= 0.60) in Gate 6 of `_sizing_entry_gate_open`.
Freeze-exempt: no change to actual trades taken (same downstream direction block).
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.agents.signal_loop import SIZING_DIR_PROB_DEAD_ZONE, SignalLoop
from src.execution.position_manager import PositionManager
from src.models.ensemble import EnsembleSignal
from src.risk.circuit_breakers import CircuitBreakers


def _stamp(offset_min: int = 0) -> datetime:
    return datetime(2026, 11, 4, 14, 30 + offset_min, tzinfo=timezone.utc)


def _make_loop() -> SignalLoop:
    return SignalLoop(
        universe=["AAPL", "MSFT"],
        ensemble=MagicMock(),
        alpaca=MagicMock(),
        circuit_breakers=CircuitBreakers(),
        pos_manager=PositionManager(initial_portfolio=100_000.0),
        session_factory=MagicMock(),
        feature_cols=[f"feat_{i}" for i in range(30)],
    )


def _sig(pred_ret: float, dir_prob: float, ticker: str = "AAPL") -> EnsembleSignal:
    s = EnsembleSignal(ticker=ticker, timestamp=_stamp())
    s.lgbm_pred_return = pred_ret
    s.lgbm_dir_prob = dir_prob
    return s


# ── H37 tests ─────────────────────────────────────────────────────────────────

def test_h37_bearish_signal_blocked_at_gate():
    """Strongly bearish signal (dir_prob=0.27) must NOT pass the entry gate.

    Before H37: `not (0.40 < 0.27 < 0.60)` = not False = True  → gate OPEN (bug)
    After  H37: `0.27 >= 0.60`              = False             → gate CLOSED

    Observed Nov 4: IDXX dir_prob=0.2715 showed would_trade=true in /diagnostics.
    """
    loop = _make_loop()
    loop._in_entry_window = lambda: True
    loop._data_fresh = True
    sig = _sig(pred_ret=-0.0147, dir_prob=0.2715)  # IDXX Nov-4 actual values
    assert not loop._sizing_entry_gate_open(sig), (
        "Bearish signal (dir_prob=0.27) must be blocked by the long-only entry gate"
    )


def test_h37_strongly_bearish_blocked():
    """TWLO Nov 4: dir_prob=0.2136 — must not pass the gate."""
    loop = _make_loop()
    loop._in_entry_window = lambda: True
    loop._data_fresh = True
    assert not loop._sizing_entry_gate_open(_sig(pred_ret=-0.009, dir_prob=0.2136))


def test_h37_ambiguous_signal_blocked():
    """dir_prob=0.50 is inside the dead zone — blocked in both old and new code."""
    loop = _make_loop()
    loop._in_entry_window = lambda: True
    loop._data_fresh = True
    assert not loop._sizing_entry_gate_open(_sig(pred_ret=0.009, dir_prob=0.50))


def test_h37_bullish_signal_passes():
    """Bullish signal with dir_prob=0.70 must still pass the gate."""
    loop = _make_loop()
    loop._in_entry_window = lambda: True
    loop._data_fresh = True
    assert loop._sizing_entry_gate_open(_sig(pred_ret=0.009, dir_prob=0.70))


def test_h37_exactly_at_threshold_passes():
    """dir_prob == 0.60 is the boundary — must pass (>= not >)."""
    _lo, hi = SIZING_DIR_PROB_DEAD_ZONE
    loop = _make_loop()
    loop._in_entry_window = lambda: True
    loop._data_fresh = True
    assert loop._sizing_entry_gate_open(_sig(pred_ret=0.009, dir_prob=hi)), (
        f"dir_prob exactly at dead-zone ceiling ({hi}) must pass the long-only gate"
    )


def test_h37_reversal_detection_unaffected():
    """_confirmed_opposite_signal must still accept bearish signals as reversals.

    Reversal detection for a long position legitimately needs to identify
    strongly bearish signals — dir_prob=0.20 qualifies.  H37 must not touch
    this path.
    """
    loop = _make_loop()
    sig = _sig(pred_ret=-0.009, dir_prob=0.20)
    assert loop._confirmed_opposite_signal(sig, entry_dir=1), (
        "Reversal detection must still accept bearish signals (dir_prob < 0.40) "
        "for long positions — H37 fix applies only to _sizing_entry_gate_open"
    )
