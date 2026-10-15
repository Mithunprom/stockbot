"""Unit tests for H29 — long-only signal gate consistency (Phase 5).

Gate 6 of _sizing_entry_gate_open formerly used abs(pred_ret) and
`not (0.40 < dir_prob < 0.60)`. In Phase 5 (long-only) this admitted two
classes of contradictory signals that should never enter:

  Case A: pred_ret < 0, dir_prob >= 0.60
    Regression says DOWN, classification says UP. Blocked at execution
    (_act_on_signal: direction = -1 → return False), but the gate reported
    "passes both gates" in diagnostics — misleading, and a latent risk if
    Phase 5 short-blocking were ever relaxed.

  Case B: pred_ret > 0, dir_prob <= 0.40
    Regression says UP → direction = +1 → LONG entry fires, despite the
    classification head saying P(up) < 40% (bearish). This is the most
    dangerous case: an entry could actually execute on a contradictory signal.

Fix (H29): Gate 6 now requires BOTH heads to agree bullish —
  pred_ret > threshold  (positive regression, never |pred_ret|)
  dir_prob >= 0.60      (bullish classification, not just outside dead zone)

These tests confirm the four-quadrant behaviour after the fix:
  Q1 (both bullish)   : pred_ret > 0, dir_prob >= 0.60 → PASSES (valid long)
  Q2 (case A)         : pred_ret < 0, dir_prob >= 0.60 → BLOCKED at gate
  Q3 (both bearish)   : pred_ret < 0, dir_prob <= 0.40 → BLOCKED at gate
  Q4 (case B)         : pred_ret > 0, dir_prob <= 0.40 → BLOCKED at gate
And the dead zone:
  Q5 (uncertain)      : pred_ret > 0, 0.40 < dir_prob < 0.60 → BLOCKED at gate
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agents.signal_loop import (
    SIZING_DIR_PROB_DEAD_ZONE,
    SignalLoop,
)
from src.execution.position_manager import PositionManager
from src.models.ensemble import EnsembleSignal
from src.risk.circuit_breakers import CircuitBreakers


def _mock_settings() -> MagicMock:
    s = MagicMock()
    s.alpaca_mode = "paper"
    return s


@pytest.fixture(autouse=True)
def _patch_settings():
    """Provide a fake Settings object so SignalLoop can be instantiated
    without real API keys in the environment.

    get_settings() is imported locally inside SignalLoop.__init__ at each
    call (`from src.config import get_settings`), so replacing the name on
    the module is enough — the local re-import picks up the mock on every
    construction. No lru_cache manipulation is needed: we only replace the
    attribute while our tests run, then restore it, leaving any cached real
    Settings produced by other test modules untouched.
    """
    import src.config as _cfg
    with patch.object(_cfg, "get_settings", return_value=_mock_settings()):
        yield


def _make_loop() -> SignalLoop:
    loop = SignalLoop(
        universe=["AAPL", "HUM", "AKAM"],
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


def _sig(ticker: str = "AAPL", pred_ret: float = 0.009,
         dir_prob: float = 0.70) -> EnsembleSignal:
    sig = EnsembleSignal(ticker=ticker, timestamp=None)
    sig.lgbm_pred_return = pred_ret
    sig.lgbm_dir_prob = dir_prob
    return sig


# ─── Gate 6 constant sanity ──────────────────────────────────────────────────

def test_dir_prob_dead_zone_upper_bound_is_sixty_pct():
    """The bullish threshold is exactly 0.60 — any change would widen or
    narrow the conviction bar globally."""
    _, hi = SIZING_DIR_PROB_DEAD_ZONE
    assert hi == 0.60


# ─── Q1: valid long signal (both heads bullish) ───────────────────────────────

def test_q1_bullish_regression_and_bullish_classification_passes():
    """Standard high-conviction long: regression positive, classification bullish."""
    loop = _make_loop()
    # pred_ret >> threshold (0.003 fallback), dir_prob well above 0.60
    assert loop._sizing_entry_gate_open(_sig(pred_ret=0.008, dir_prob=0.70))


def test_q1_passes_at_exact_dir_prob_threshold():
    """Exactly at the 0.60 boundary must be admitted (>= not >)."""
    loop = _make_loop()
    assert loop._sizing_entry_gate_open(_sig(pred_ret=0.009, dir_prob=0.60))


def test_q1_pred_return_just_above_fallback_threshold():
    """pred_ret just above DYN_THRESH_FALLBACK with fallback active passes."""
    loop = _make_loop()
    # pred_magnitudes is empty → threshold is DYN_THRESH_FALLBACK (0.003)
    assert loop._sizing_entry_gate_open(_sig(pred_ret=0.0031, dir_prob=0.65))


# ─── Case A: negative pred_ret regardless of dir_prob ────────────────────────

def test_case_a_negative_regression_with_bullish_classification_blocked():
    """H29 primary fix: pred_ret < 0 must not pass the gate even if dir_prob >= 0.60.

    The Oct 14 diagnostics showed HUM (pred_ret=-0.39, ensemble=-0.39) and
    AKAM (pred_ret=-0.31) reported as 'passes both gates'. This test ensures
    that is no longer true after the fix.
    """
    loop = _make_loop()
    # Mimics HUM from Oct 14 diagnostics
    assert not loop._sizing_entry_gate_open(_sig(ticker="HUM", pred_ret=-0.3899, dir_prob=0.65))
    # Mimics AKAM from Oct 14 diagnostics
    assert not loop._sizing_entry_gate_open(_sig(ticker="AKAM", pred_ret=-0.3121, dir_prob=0.62))


def test_case_a_small_negative_regression_blocked():
    """Small negative pred_ret (just below zero) is also blocked."""
    loop = _make_loop()
    assert not loop._sizing_entry_gate_open(_sig(pred_ret=-0.0001, dir_prob=0.70))


# ─── Q3: both bearish ─────────────────────────────────────────────────────────

def test_q3_negative_regression_and_bearish_classification_blocked():
    """Short signal (pred_ret < 0, dir_prob <= 0.40): blocked at gate."""
    loop = _make_loop()
    assert not loop._sizing_entry_gate_open(_sig(pred_ret=-0.009, dir_prob=0.30))


# ─── Case B: dangerous false positive — was entering, now blocked ─────────────

def test_case_b_positive_regression_with_bearish_classification_blocked():
    """Previously: pred_ret > 0, dir_prob <= 0.40 → gate passed AND long entry
    fired (direction = +1 from pred_ret > 0). This is contradictory: the
    regression predicts UP but the classifier is <40% confident of UP.

    H29 blocks this at Gate 6 — a long-only entry must not fire when the
    classification head is bearish.
    """
    loop = _make_loop()
    # Regression says +0.5%, classification says 30% chance of going up
    assert not loop._sizing_entry_gate_open(_sig(pred_ret=0.005, dir_prob=0.30))
    assert not loop._sizing_entry_gate_open(_sig(pred_ret=0.009, dir_prob=0.40))


def test_case_b_blocked_at_dir_prob_just_below_threshold():
    """dir_prob = 0.599 (one tick below 0.60) is blocked."""
    loop = _make_loop()
    assert not loop._sizing_entry_gate_open(_sig(pred_ret=0.009, dir_prob=0.599))


# ─── Q5: dir_prob in dead zone ────────────────────────────────────────────────

def test_uncertain_dir_prob_blocked():
    """A signal in the dead zone (0.40, 0.60) exclusive is uncertain: blocked."""
    loop = _make_loop()
    assert not loop._sizing_entry_gate_open(_sig(pred_ret=0.009, dir_prob=0.50))
    assert not loop._sizing_entry_gate_open(_sig(pred_ret=0.009, dir_prob=0.55))
    assert not loop._sizing_entry_gate_open(_sig(pred_ret=0.009, dir_prob=0.41))


# ─── pred_ret below magnitude threshold ──────────────────────────────────────

def test_pred_ret_at_zero_blocked():
    """pred_ret = 0.0 does not clear any positive threshold."""
    loop = _make_loop()
    assert not loop._sizing_entry_gate_open(_sig(pred_ret=0.0, dir_prob=0.70))


def test_pred_ret_below_fallback_threshold_blocked():
    """pred_ret below DYN_THRESH_FALLBACK (0.003) is blocked even if bullish."""
    loop = _make_loop()
    assert not loop._sizing_entry_gate_open(_sig(pred_ret=0.002, dir_prob=0.70))


# ─── Regression: gate no longer uses abs(pred_ret) ───────────────────────────

def test_gate_uses_pred_ret_not_abs_pred_ret():
    """The defining property of H29: a large NEGATIVE pred_ret must be blocked,
    whereas abs() would pass it. If this test fails, the gate regressed to abs().
    """
    loop = _make_loop()
    large_negative = _sig(pred_ret=-0.5, dir_prob=0.80)
    large_positive = _sig(pred_ret=+0.5, dir_prob=0.80)
    assert not loop._sizing_entry_gate_open(large_negative), (
        "Negative pred_ret passed Gate 6 — gate may be using abs(pred_ret)"
    )
    assert loop._sizing_entry_gate_open(large_positive)
