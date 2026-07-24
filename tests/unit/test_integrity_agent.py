"""Unit tests for the Integrity Sentinel's audit math.

The 2026-07-20 corruption signature: a partially filled exit passed
filled_qty=6 on a 27-share GOOGL position, storing pnl_pct=-15.7% on a
-4.0% trade. These tests pin the recomputation + divergence logic that
catches (and repairs) that class of defect.
"""

from __future__ import annotations

import asyncio

import pytest

from src.agents.integrity_agent import (
    FILL_PNL_TOLERANCE,
    KELLY_SANE_ABS_PNL_PCT,
    PNL_PCT_TOLERANCE,
    classify_row,
    corrected_fill_pnl,
    expected_pnl_pct,
    fill_price_pnl_deviation,
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


class TestFillPricePnlDeviation:
    """Pin the fill-price vs stored-pnl reconciliation helper.

    The existing pnl_pct check verifies pnl_pct = pnl/(entry*shares) which is
    internally consistent even when pnl itself is corrupted. This helper catches
    the complementary case: pnl doesn't match (exit-entry)*shares.
    """

    def test_clean_trade_near_zero_deviation(self):
        # MA #98: entry 540.12, exit 542.84, 28.22 shares, pnl 76.76
        # fill_pnl = (542.84-540.12)*28.22 = 76.76 → deviation ≈ 0
        dev = fill_price_pnl_deviation(
            pnl=76.76, exit_price=542.84, entry_price=540.12, shares=28.22
        )
        assert dev is not None
        assert dev < FILL_PNL_TOLERANCE

    def test_mu_93_corruption_flagged(self):
        # MU #93 (pre-v0.4.5): price moved from 860.46→822.58 on 3 shares
        # → fill_pnl ≈ -113.64, but stored pnl = -512.48 (15.45% deviation)
        dev = fill_price_pnl_deviation(
            pnl=-512.48, exit_price=822.58, entry_price=860.46, shares=3.0
        )
        assert dev is not None
        assert dev > FILL_PNL_TOLERANCE
        assert dev == pytest.approx(0.1545, abs=0.002)

    def test_smci_83_corruption_flagged(self):
        # SMCI #83 (pre-v0.4.5): price moved from 27.95→26.02 on 72 shares
        # → fill_pnl ≈ -138.96, but stored pnl = -638.06 (24.8% deviation)
        dev = fill_price_pnl_deviation(
            pnl=-638.06, exit_price=26.02, entry_price=27.95, shares=72.0
        )
        assert dev is not None
        assert dev > FILL_PNL_TOLERANCE
        assert dev == pytest.approx(0.2481, abs=0.002)

    def test_multi_fill_tolerance_not_exceeded(self):
        # WDC #97: entry 482.82, exit 494.44, 7 shares, pnl 154.16
        # fill_pnl = 81.34 → deviation 2.15%, within 5% tolerance
        dev = fill_price_pnl_deviation(
            pnl=154.16, exit_price=494.44, entry_price=482.82, shares=7.0
        )
        assert dev is not None
        assert dev < FILL_PNL_TOLERANCE

    def test_missing_inputs_return_none(self):
        assert fill_price_pnl_deviation(None, 100.0, 90.0, 10.0) is None
        assert fill_price_pnl_deviation(50.0, None, 90.0, 10.0) is None
        assert fill_price_pnl_deviation(50.0, 100.0, None, 10.0) is None
        assert fill_price_pnl_deviation(50.0, 100.0, 90.0, None) is None
        assert fill_price_pnl_deviation(50.0, 100.0, 0.0, 10.0) is None

    def test_dust_notional_returns_none(self):
        # entry_price=0.09 * shares=10 = notional=0.9 < 1.0 → skip
        assert fill_price_pnl_deviation(0.5, 0.11, 0.09, 10.0) is None

    def test_zero_deviation_perfect_fill(self):
        # Sanity: (exit-entry)*shares exactly matches pnl → deviation = 0
        dev = fill_price_pnl_deviation(
            pnl=100.0, exit_price=110.0, entry_price=100.0, shares=10.0
        )
        assert dev == pytest.approx(0.0)

    def test_fill_pnl_tolerance_constant_is_five_pct(self):
        # Ensures calibration hasn't drifted from the design spec
        assert FILL_PNL_TOLERANCE == pytest.approx(0.05)


class TestCorrectedFillPnl:
    """The healer that rewrites corrupt dollar pnl from fill prices + shares."""

    def test_mu_93_heals_to_fill_truth(self):
        # MU #93: stored pnl -512.48 is corrupt; true = (822.58-860.46)*3
        out = corrected_fill_pnl(exit_price=822.58, entry_price=860.46, shares=3.0)
        assert out is not None
        pnl, pnl_pct = out
        assert pnl == pytest.approx(-113.64, abs=0.01)
        # true pnl_pct -4.4%, not the corrupt -19.85% that poisoned Kelly
        assert pnl_pct == pytest.approx(-0.0440, abs=1e-3)

    def test_smci_83_heals_to_fill_truth(self):
        out = corrected_fill_pnl(exit_price=26.02, entry_price=27.95, shares=72.0)
        assert out is not None
        pnl, pnl_pct = out
        assert pnl == pytest.approx(-138.96, abs=0.01)
        assert pnl_pct == pytest.approx(-0.0690, abs=1e-3)

    def test_xom_68_heals_to_fill_truth(self):
        out = corrected_fill_pnl(exit_price=137.9, entry_price=141.2, shares=13.0)
        assert out is not None
        pnl, _ = out
        assert pnl == pytest.approx(-42.90, abs=0.01)

    def test_missing_inputs_return_none(self):
        assert corrected_fill_pnl(None, 100.0, 10.0) is None
        assert corrected_fill_pnl(100.0, None, 10.0) is None
        assert corrected_fill_pnl(100.0, 90.0, None) is None
        assert corrected_fill_pnl(100.0, 0.0, 10.0) is None

    def test_dust_notional_returns_none(self):
        assert corrected_fill_pnl(0.11, 0.09, 10.0) is None

    def test_corrected_pct_is_sane_for_all_known_corrupt_rows(self):
        # Every real corrupt row heals to a |pnl_pct| inside the Kelly bound,
        # so the audit will mark it action="rewrite_pnl" (auto-healable).
        for exit_p, entry_p, sh in [
            (822.58, 860.46, 3.0),   # MU #93
            (26.02, 27.95, 72.0),    # SMCI #83
            (137.9, 141.2, 13.0),    # XOM #68
            (380.17, 389.27, 7.0),   # MSFT #82
        ]:
            out = corrected_fill_pnl(exit_p, entry_p, sh)
            assert out is not None
            assert abs(out[1]) <= KELLY_SANE_ABS_PNL_PCT


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _FakeSession:
    """Minimal async session that records UPDATE executions for _repair tests."""

    def __init__(self):
        self.updates = 0
        self.committed = False

    async def execute(self, _stmt):
        self.updates += 1
        return _FakeResult([])

    async def commit(self):
        self.committed = True


class TestRepairHealsFillCorruption:
    """_repair rewrites the corrupt dollar pnl for healable fill rows."""

    def _run(self, coro):
        return asyncio.run(coro)

    def test_repair_issues_update_for_healable_fill_row(self):
        from src.agents.integrity_agent import IntegrityAgent

        agent = IntegrityAgent(session_factory=None, signal_loop=None)
        session = _FakeSession()
        fill_rows = [{
            "id": 93, "ticker": "MU", "stored_pnl": -512.48,
            "corrected_pnl": -113.64, "corrected_pnl_pct": -0.0440,
            "action": "rewrite_pnl",
        }]
        result = self._run(agent._repair(session, [], [], fill_rows))
        assert session.updates == 1          # exactly the one MU row
        assert session.committed
        assert result["fill_pnl_rewritten"] == 1

    def test_repair_skips_unhealable_fill_row(self):
        from src.agents.integrity_agent import IntegrityAgent

        agent = IntegrityAgent(session_factory=None, signal_loop=None)
        session = _FakeSession()
        # action=None → recompute itself was insane; leave for a human
        fill_rows = [{"id": 7, "ticker": "ZZZ", "action": None,
                      "corrected_pnl": None, "corrected_pnl_pct": None}]
        result = self._run(agent._repair(session, [], [], fill_rows))
        assert session.updates == 0
        assert result["fill_pnl_rewritten"] == 0
