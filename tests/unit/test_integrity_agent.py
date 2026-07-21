"""Unit tests for the Integrity Sentinel's audit math.

The 2026-07-20 corruption signature: a partially filled exit passed
filled_qty=6 on a 27-share GOOGL position, storing pnl_pct=-15.7% on a
-4.0% trade. These tests pin the recomputation + divergence logic that
catches (and repairs) that class of defect.
"""

from __future__ import annotations

import pytest

from src.agents.integrity_agent import (
    KELLY_SANE_ABS_PNL_PCT,
    PNL_PCT_TOLERANCE,
    classify_row,
    expected_pnl_pct,
    is_divergent,
)


class TestExpectedPnlPct:
    def test_simple_long(self):
        # 27 shares @ 362.77, pnl -394.91 → -4.03%
        exp = expected_pnl_pct(pnl=-394.91, entry_price=362.768889, shares=27.0)
        assert exp == pytest.approx(-0.0403, abs=1e-3)

    def test_winning_trade(self):
        exp = expected_pnl_pct(pnl=154.16, entry_price=482.82, shares=7.0)
        assert exp == pytest.approx(0.0456, abs=1e-3)

    def test_missing_inputs_return_none(self):
        assert expected_pnl_pct(None, 100.0, 10.0) is None
        assert expected_pnl_pct(50.0, None, 10.0) is None
        assert expected_pnl_pct(50.0, 100.0, None) is None
        assert expected_pnl_pct(50.0, 0.0, 10.0) is None
        assert expected_pnl_pct(50.0, 100.0, 0.0) is None

    def test_dust_notional_returns_none(self):
        # Sub-$1 notional would explode the ratio — refuse to compute
        assert expected_pnl_pct(0.5, 0.01, 10.0) is None


class TestIsDivergent:
    def test_googl_partial_fill_signature_flagged(self):
        # Stored -15.7% vs true -4.0% — the exact 2026-07-20 defect
        assert is_divergent(stored=-0.1570, expected=-0.0403)

    def test_extreme_corruption_flagged(self):
        assert is_divergent(stored=9.9295, expected=0.0412)

    def test_matching_values_pass(self):
        assert not is_divergent(stored=-0.0403, expected=-0.0403)

    def test_float_noise_within_tolerance_passes(self):
        assert not is_divergent(stored=0.0210, expected=0.0210 + PNL_PCT_TOLERANCE / 2)

    def test_none_never_diverges(self):
        # Rows lacking inputs are not flagged (no guessing)
        assert not is_divergent(stored=None, expected=0.05)
        assert not is_divergent(stored=0.05, expected=None)


class TestClassifyRow:
    def test_exit_side_corruption_rewritten(self):
        # GOOGL: stored -15.7%, true -4.0% → rewrite to the recompute
        assert classify_row(stored=-0.1570, expected=-0.0403) == "rewrite"

    def test_entry_side_corruption_nullified(self):
        # SMCI row 83: shares column under-recorded (partial ENTRY fill),
        # so the recompute itself is impossible (-31.7%). If stored is also
        # insane, no derived pct is trustworthy → NULL.
        assert classify_row(stored=-0.3171, expected=-0.3171) == "nullify"

    def test_sane_stored_with_insane_recompute_kept(self):
        # Same row BEFORE the v0.4.5 repair: stored -6.9% (from true PM
        # entry notional) vs insane recompute -31.7% → keep stored.
        assert classify_row(stored=-0.0691, expected=-0.3171) is None

    def test_consistent_row_untouched(self):
        assert classify_row(stored=-0.0403, expected=-0.0403) is None

    def test_missing_recompute_untouched(self):
        assert classify_row(stored=-0.0403, expected=None) is None


class TestKellySanityBound:
    def test_bound_catches_corrupt_rows_not_real_trades(self):
        # Real swing-trade returns in this system are low single digits
        real_trades = [-0.0403, 0.0456, -0.0022, 0.0267, -0.0218]
        corrupt_rows = [-0.2418, 9.9295, 0.2676, -0.1570]
        assert all(abs(p) <= KELLY_SANE_ABS_PNL_PCT for p in real_trades)
        # The coarse bound catches the grossly impossible rows; milder
        # corruption (-24%, -15.7%) is caught by the recomputation check,
        # which is why both checks exist.
        assert sum(1 for p in corrupt_rows if abs(p) > KELLY_SANE_ABS_PNL_PCT) == 2


class TestSharesReconciliation:
    def test_implied_shares_math_matches_mstr_row_99(self):
        # stored pnl_pct trusted (post-v0.4.5); shares column indicted:
        # 51 recorded but pnl/(stored*entry) implies the true 156.5
        pnl, stored, entry = 777.0, 0.05172, 96.047255
        implied = pnl / (stored * entry)
        assert abs(implied - 156.4) < 1.0
