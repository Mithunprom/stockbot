"""Unit tests for the measured-negative-edge entry block.

The gap being closed: production reported `kelly_fraction = -0.6661` and
`kelly_mode = "probation"` while the `kelly_entries_blocked` diagnostic was a
hardcoded `False`. The field never reflected reality, and nothing in the entry
path acted on a measured negative edge once `_kelly_mode()` fell back to
"inactive".

Two properties matter as much as the block itself and are tested here:
  1. EXITS must remain possible while entries are blocked (H13 — a halt that
     strands open positions is a known open issue and must not get worse).
  2. The block must be SELF-DRAINING. This subsystem froze the bot on
     2026-05-27 and again on 2026-06-26; a block that cannot release itself
     is a regression, not a fix.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from src.agents.signal_loop import (
    KELLY_HARD_BLOCK_THRESHOLD,
    KELLY_LOOKBACK_DAYS,
    KELLY_MIN_TRADES,
    SIZING_MAX_HOLD_BARS,
    SignalLoop,
)
from src.execution.position_manager import PositionManager
from src.models.ensemble import EnsembleSignal
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
    loop._in_entry_window = lambda: True
    loop._data_fresh = True
    return loop


def _seed_outcomes(loop: SignalLoop, outcomes: list[float],
                   age_days: float = 1.0) -> None:
    """Fill the rolling Kelly window with outcomes of a given age."""
    stamp = datetime.now(timezone.utc) - timedelta(days=age_days)
    loop._sizing_recent_outcomes = [(stamp, o) for o in outcomes]


def _losing_window() -> list[float]:
    """Outcomes that produce a Kelly well below the hard-block threshold.

    Frequent small wins against rare large losses: win rate 0.25 and
    b = 0.5/3.0, giving f* = (0.25*0.1667 - 0.75)/0.1667 ~= -4.3.
    """
    return [0.005] * 3 + [-0.03] * 9


def _signal(ticker: str = "AAPL") -> EnsembleSignal:
    sig = EnsembleSignal(ticker=ticker, timestamp=None)
    sig.lgbm_pred_return = 0.009
    sig.lgbm_dir_prob = 0.65
    return sig


# ─── The predicate ──────────────────────────────────────────────────────────

def test_negative_measured_edge_blocks_entries():
    """REGRESSION: kelly -0.67 must not coexist with entries flowing."""
    loop = _make_loop()
    _seed_outcomes(loop, _losing_window())
    loop._update_kelly()
    assert loop._kelly_fraction <= KELLY_HARD_BLOCK_THRESHOLD
    assert loop._kelly_entries_blocked() is True


def test_healthy_edge_does_not_block():
    loop = _make_loop()
    _seed_outcomes(loop, [0.02] * 7 + [-0.005] * 5)
    loop._update_kelly()
    assert loop._kelly_fraction > 0
    assert loop._kelly_entries_blocked() is False


def test_mildly_negative_edge_stays_on_probation_not_a_hard_block():
    """Probation still handles an unlucky stretch; the hard block is for worse."""
    loop = _make_loop()
    loop._sizing_recent_outcomes = [
        (datetime.now(timezone.utc), o) for o in ([0.01] * 5 + [-0.011] * 6)
    ]
    loop._update_kelly()
    assert KELLY_HARD_BLOCK_THRESHOLD < loop._kelly_fraction <= 0.0
    assert loop._kelly_mode() == "probation"
    assert loop._kelly_entries_blocked() is False


def test_stale_fraction_cannot_block_when_the_governor_is_inactive():
    """DEADLOCK SAFETY: a value with no live sample behind it gates nothing."""
    loop = _make_loop()
    loop._kelly_fraction = -0.9
    loop._sizing_recent_outcomes = []
    assert loop._kelly_mode() == "inactive"
    assert loop._kelly_entries_blocked() is False


def test_update_kelly_retires_a_stale_fraction():
    """An expired window must not keep publishing its last reading."""
    loop = _make_loop()
    loop._kelly_fraction = -0.6661
    loop._sizing_recent_outcomes = [(datetime.now(timezone.utc), -0.01)]
    loop._update_kelly()
    assert loop._kelly_fraction == 0.0


def test_block_is_self_draining_as_the_window_expires():
    """The window ages out, the sample falls under the minimum, block releases.

    This is the property that makes the block bounded rather than a repeat of
    the 2026-05-27 / 2026-06-26 freezes: no human action and no new trades are
    required for it to lift.
    """
    loop = _make_loop()
    _seed_outcomes(loop, _losing_window(), age_days=1.0)
    loop._update_kelly()
    assert loop._kelly_entries_blocked() is True

    # Age every outcome past the lookback — exactly what happens if entries
    # stop and nothing new closes.
    _seed_outcomes(loop, _losing_window(), age_days=KELLY_LOOKBACK_DAYS + 1)
    assert loop._kelly_mode() == "inactive"
    assert loop._kelly_entries_blocked() is False


def test_block_needs_a_real_sample():
    """Below KELLY_MIN_TRADES the governor has nothing to act on."""
    loop = _make_loop()
    _seed_outcomes(loop, [-0.05] * (KELLY_MIN_TRADES - 1))
    loop._update_kelly()
    assert loop._kelly_entries_blocked() is False


# ─── Entry gate ─────────────────────────────────────────────────────────────

def test_entry_gate_refuses_every_ticker_while_blocked():
    loop = _make_loop()
    _seed_outcomes(loop, _losing_window())
    loop._update_kelly()
    assert not loop._sizing_entry_gate_open(_signal("AAPL"))
    assert not loop._sizing_entry_gate_open(_signal("XOM"))


def test_probation_probe_cannot_bypass_the_hard_block():
    """A probe is exactly what must not happen at a measured edge this bad."""
    loop = _make_loop()
    _seed_outcomes(loop, _losing_window())
    loop._update_kelly()
    # Make the ticker maximally probe-eligible: large sample, strong live IC.
    loop._ticker_ic_probe["AAPL"] = (0.50, 10_000)
    loop._ticker_ic["AAPL"] = (0.50, 10_000)
    loop._probation_entries_today = 0
    assert not loop._sizing_entry_gate_open(_signal("AAPL"))


def test_entries_resume_once_the_measured_edge_recovers():
    loop = _make_loop()
    _seed_outcomes(loop, _losing_window())
    loop._update_kelly()
    assert not loop._sizing_entry_gate_open(_signal("AAPL"))

    _seed_outcomes(loop, [0.02] * 8 + [-0.004] * 4)
    loop._update_kelly()
    assert loop._sizing_entry_gate_open(_signal("AAPL"))


# ─── Exits must survive the block (H13) ────────────────────────────────────

def test_exits_still_fire_while_entries_are_blocked():
    """CRITICAL: blocking entries must never strand an open position.

    H13 (a halt stranding positions) is a known open issue. This gate lives
    only in the entry path, so every exit reason must still be reachable while
    it is active.
    """
    loop = _make_loop()
    _seed_outcomes(loop, _losing_window())
    loop._update_kelly()
    assert loop._kelly_entries_blocked() is True

    loop._pm.portfolio_value = 100_000.0
    loop._entry_prices["AAPL"] = 100.0
    loop._entry_directions["AAPL"] = 1
    loop._peak_prices["AAPL"] = 100.0
    loop._ticker_daily_vol["AAPL"] = 0.02

    flat = EnsembleSignal(ticker="AAPL", timestamp=None)
    flat.lgbm_pred_return = 0.0
    flat.lgbm_dir_prob = 0.5

    loop._bars_held["AAPL"] = 1
    assert loop._check_sizing_exit("AAPL", 98.5, flat) == "stop_loss"
    assert loop._check_sizing_exit("AAPL", 102.0, flat) == "take_profit"

    loop._bars_held["AAPL"] = SIZING_MAX_HOLD_BARS
    assert loop._check_sizing_exit("AAPL", 100.0, flat) == "max_hold"


def test_block_does_not_raise_a_circuit_breaker_halt():
    """The block is an entry gate, not a halt — halts are the human's call."""
    loop = _make_loop()
    _seed_outcomes(loop, _losing_window())
    loop._update_kelly()
    loop._sizing_entry_gate_open(_signal("AAPL"))
    assert loop._cb.is_halted is False


# ─── Diagnostics must tell the truth ───────────────────────────────────────

def test_diagnostics_reports_the_real_block_state():
    """REGRESSION: this field was a hardcoded `False`."""
    loop = _make_loop()
    _seed_outcomes(loop, _losing_window())
    loop._update_kelly()
    assert loop._kelly_entries_blocked() is True

    _seed_outcomes(loop, [0.02] * 8 + [-0.004] * 4)
    loop._update_kelly()
    assert loop._kelly_entries_blocked() is False


def test_hard_block_threshold_is_strictly_negative():
    """A non-negative threshold would block on a merely flat edge."""
    assert KELLY_HARD_BLOCK_THRESHOLD < 0.0
