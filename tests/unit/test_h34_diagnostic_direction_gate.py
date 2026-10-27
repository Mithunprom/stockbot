"""H34: /diagnostics signal_gate_analysis must include direction and IC gates.

Before H34, would_trade=true was returned for bearish signals (negative
lgbm_pred_return) because _sizing_entry_gate_open uses abs(pred_return) for
Phase 5 symmetry.  In execution, _act_on_signal blocks direction < 0 for the
current long-only Phase 4 bot — so the diagnostic falsely reported signals as
actionable that would never enter.

Example from 2026-10-26 live diagnostics (post-market):
  GLW  pred_return=-0.004248  dir_prob=0.331  → would_trade=True (FALSE POSITIVE)
  CIEN pred_return=-0.002982  dir_prob=0.389  → would_trade=True (FALSE POSITIVE)
  INTC pred_return=+0.007194  dir_prob=0.358  → would_trade=True (correct)

After H34, GLW and CIEN show would_trade=False, blocked_by=["bearish_direction"].

Similarly, tickers in tickers_ic_blocked were not surfaced per-row; now
ic_blocked appears in blocked_by so the operator can see why each row fails.
"""
from __future__ import annotations

import pytest

# ── Direction gate properties ─────────────────────────────────────────────────

# These tests verify the relationship between the pred_return sign and
# direction-gate outcome rather than importing main.py (which has FastAPI deps).
# They use the same logic as the updated _pipeline_diagnostics in main.py.


def _gate_row(
    pred_ret: float,
    dir_prob: float,
    cost_thr: float = 0.002,
    dead_zone: tuple[float, float] = (0.4, 0.6),
    on_cooldown: bool = False,
    in_probation: bool = False,
    managed_heat: float = 0.0,
    ic_blocked: bool = False,
) -> dict:
    """Replicate the gate analysis logic from main.py _pipeline_diagnostics."""
    lo, hi = dead_zone
    passes_pred = abs(pred_ret) > cost_thr
    passes_dir = not (lo < dir_prob < hi)
    passes_both = passes_pred and passes_dir
    passes_direction = pred_ret > 0
    would_trade = (
        passes_both
        and not on_cooldown
        and not in_probation
        and passes_direction
        and not ic_blocked
    )
    blocked_by = (
        (["pred_return_too_small"] if not passes_pred else [])
        + (["dir_prob_in_dead_zone"] if not passes_dir else [])
        + (["ticker_cooldown"] if on_cooldown else [])
        + (["kelly_probation"] if in_probation else [])
        + (["managed_heat_too_high"] if managed_heat >= 0.80 else [])
        + (["bearish_direction"] if not passes_direction else [])
        + (["ic_blocked"] if ic_blocked else [])
    )
    return {
        "passes_pred_return_gate": passes_pred,
        "passes_dir_prob_gate": passes_dir,
        "passes_both_gates": passes_both,
        "would_trade": would_trade,
        "blocked_by": blocked_by,
    }


class TestDirectionGate:
    """H34: bearish (negative pred_return) signals must not show would_trade=True."""

    def test_glw_live_example_blocked(self):
        """GLW 2026-10-26: pred=-0.004248, dir_prob=0.331 → should NOT would_trade."""
        row = _gate_row(pred_ret=-0.004248, dir_prob=0.331)
        assert row["would_trade"] is False
        assert "bearish_direction" in row["blocked_by"]

    def test_cien_live_example_blocked(self):
        """CIEN 2026-10-26: pred=-0.002982, dir_prob=0.389 → should NOT would_trade."""
        row = _gate_row(pred_ret=-0.002982, dir_prob=0.389)
        assert row["would_trade"] is False
        assert "bearish_direction" in row["blocked_by"]

    def test_intc_live_example_passes(self):
        """INTC 2026-10-26: pred=+0.007194, dir_prob=0.358 → should would_trade."""
        row = _gate_row(pred_ret=0.007194, dir_prob=0.358)
        assert row["would_trade"] is True
        assert "bearish_direction" not in row["blocked_by"]

    def test_zero_pred_return_blocked(self):
        """A pred_return of exactly 0.0 is not bullish — must not would_trade."""
        row = _gate_row(pred_ret=0.0, dir_prob=0.7)
        assert row["would_trade"] is False
        # pred=0.0 also fails the magnitude gate AND the direction gate
        assert "pred_return_too_small" in row["blocked_by"]
        assert "bearish_direction" in row["blocked_by"]

    def test_bearish_high_magnitude_still_blocked(self):
        """A large negative pred_return should NOT would_trade even if dir_prob is strong."""
        row = _gate_row(pred_ret=-0.05, dir_prob=0.65)
        assert row["would_trade"] is False
        assert "bearish_direction" in row["blocked_by"]
        # passes_both was True under old logic — the bug
        assert row["passes_both_gates"] is True

    def test_bullish_below_threshold_correctly_blocked(self):
        """A positive but tiny pred_return fails the magnitude gate (not direction gate)."""
        row = _gate_row(pred_ret=0.001, dir_prob=0.7)
        assert row["would_trade"] is False
        assert "pred_return_too_small" in row["blocked_by"]
        assert "bearish_direction" not in row["blocked_by"]


class TestIcBlockGate:
    """H34: IC-blocked tickers must appear in blocked_by."""

    def test_ic_blocked_ticker_shows_in_blocked_by(self):
        row = _gate_row(pred_ret=0.007, dir_prob=0.7, ic_blocked=True)
        assert row["would_trade"] is False
        assert "ic_blocked" in row["blocked_by"]

    def test_non_ic_blocked_ticker_not_labelled(self):
        row = _gate_row(pred_ret=0.007, dir_prob=0.7, ic_blocked=False)
        assert "ic_blocked" not in row["blocked_by"]

    def test_ic_blocked_bearish_shows_both(self):
        """A bearish IC-blocked ticker gets both labels."""
        row = _gate_row(pred_ret=-0.005, dir_prob=0.7, ic_blocked=True)
        assert "bearish_direction" in row["blocked_by"]
        assert "ic_blocked" in row["blocked_by"]


class TestOldBugEvidence:
    """Document the pre-H34 false positive to prove the fix has teeth."""

    def test_old_logic_would_have_returned_true_for_glw(self):
        """Under old logic (abs check, no direction gate), GLW would_trade=True."""
        # Replicate OLD logic without direction gate
        pred_ret = -0.004248
        dir_prob = 0.331
        cost_thr = 0.002
        lo, hi = 0.4, 0.6
        passes_pred_old = abs(pred_ret) > cost_thr   # True: 0.004248 > 0.002
        passes_dir_old = not (lo < dir_prob < hi)     # True: 0.331 not in (0.4,0.6)
        old_would_trade = passes_pred_old and passes_dir_old
        assert old_would_trade is True, "test is only useful if old logic fires True"

    def test_new_logic_returns_false_for_glw(self):
        """Under new logic, GLW would_trade=False (direction gate blocks it)."""
        row = _gate_row(pred_ret=-0.004248, dir_prob=0.331)
        assert row["would_trade"] is False


class TestSignalCountConsistency:
    """signals_passing must count would_trade rows, not passes_both_gates rows."""

    def test_n_pass_uses_would_trade_not_passes_both(self):
        """Two bearish + one bullish: n_pass=1 not 3."""
        rows = [
            _gate_row(pred_ret=-0.004248, dir_prob=0.331),  # GLW — bearish
            _gate_row(pred_ret=-0.002982, dir_prob=0.389),  # CIEN — bearish
            _gate_row(pred_ret=0.007194, dir_prob=0.358),   # INTC — bullish
        ]
        n_pass_old = sum(1 for r in rows if r["passes_both_gates"])
        n_pass_new = sum(1 for r in rows if r["would_trade"])
        assert n_pass_old == 3, "old count was inflated by bearish signals"
        assert n_pass_new == 1, "only INTC should count"
