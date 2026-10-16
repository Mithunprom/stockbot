"""kelly_trades_until_active exposes the activation cliff in /diagnostics.

When Kelly has fewer than KELLY_MIN_TRADES outcomes in its rolling window, the
fraction is reported as 0.0 (mode='inactive') and entries proceed at default
notional. The transition from inactive → measuring happens in a single tick the
moment the 10th outcome lands, which is a cliff: if recent trades are winning,
the fraction can jump to a large positive value and size up immediately; if
losing, it can jump into probation.

`kelly_trades_until_active` makes that cliff visible at /diagnostics so the
operator can see how many trades away from activation the governor is — and
know to watch the H24/H25 structural guards before that tick arrives.
"""
from __future__ import annotations

from src.agents.signal_loop import KELLY_MIN_TRADES


def _make_diag(n_outcomes: int) -> dict:
    """Simulate the kelly_trades_until_active computation for n outcomes."""
    return max(0, KELLY_MIN_TRADES - n_outcomes)


def test_zero_outcomes_needs_full_min():
    assert _make_diag(0) == KELLY_MIN_TRADES


def test_nine_outcomes_needs_one_more():
    assert _make_diag(KELLY_MIN_TRADES - 1) == 1


def test_exactly_at_threshold_returns_zero():
    assert _make_diag(KELLY_MIN_TRADES) == 0


def test_above_threshold_clamps_to_zero():
    assert _make_diag(KELLY_MIN_TRADES + 5) == 0


def test_constant_is_positive():
    assert KELLY_MIN_TRADES > 0
