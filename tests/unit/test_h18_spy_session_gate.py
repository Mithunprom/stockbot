"""Unit tests for H18: SPY intraday session-return gate.

Tests _compute_spy_session_block() in isolation (pure functions, no DB/sqlalchemy).
"""
import math
import pytest
from src.data.market_regime import (
    _compute_spy_session_block,
    SPY_SESSION_BLOCK_THRESHOLD,
    MarketRegimeSnapshot,
)


class TestComputeSpySessionBlock:
    def test_flat_session_no_block(self):
        assert _compute_spy_session_block(0.0) is False

    def test_mild_up_session_no_block(self):
        assert _compute_spy_session_block(0.003) is False

    def test_mild_down_below_threshold_no_block(self):
        # -0.4% < 0.5% threshold → no block
        assert _compute_spy_session_block(-0.004) is False

    def test_exactly_at_threshold_no_block(self):
        # -0.5% == threshold, strict less-than → no block
        assert _compute_spy_session_block(-SPY_SESSION_BLOCK_THRESHOLD) is False

    def test_just_below_threshold_blocks(self):
        # -0.501% > threshold → block
        assert _compute_spy_session_block(-0.00501) is True

    def test_severe_down_blocks(self):
        # -2% down day → block
        assert _compute_spy_session_block(-0.02) is True

    def test_nan_fails_open(self):
        # NaN is not finite → fail open (no block)
        assert _compute_spy_session_block(float("nan")) is False

    def test_positive_inf_fails_open(self):
        assert _compute_spy_session_block(math.inf) is False

    def test_negative_inf_fails_open(self):
        # -inf is finite=False → fail open, NOT a block
        assert _compute_spy_session_block(-math.inf) is False

    def test_custom_threshold_respected(self):
        # custom 1% threshold: -0.9% should not block
        assert _compute_spy_session_block(-0.009, threshold=0.01) is False
        # -1.1% should block
        assert _compute_spy_session_block(-0.011, threshold=0.01) is True


class TestMarketRegimeSnapshotDefaults:
    def test_default_spy_session_fields(self):
        snap = MarketRegimeSnapshot()
        assert snap.spy_session_return == 0.0
        assert snap.spy_session_block is False

    def test_spy_session_return_in_to_dict(self):
        snap = MarketRegimeSnapshot(spy_session_return=-0.007, spy_session_block=True)
        d = snap.to_dict()
        assert "spy_session_return" in d
        assert d["spy_session_return"] == pytest.approx(-0.007, abs=1e-6)
        assert d["spy_session_block"] is True

    def test_spy_session_block_false_when_return_zero(self):
        snap = MarketRegimeSnapshot(spy_session_return=0.0, spy_session_block=False)
        assert snap.spy_session_block is False
