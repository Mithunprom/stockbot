"""H39 — Kelly Window Composition + IC Block Detail diagnostics.

Stdlib-only. These tests verify the _kelly_window_summary() logic and the
ic_block_detail construction without importing numpy/pandas/torch.

The helper under test lives in signal_loop, which imports many heavy deps.
We test the pure logic by extracting it into a local shim that mirrors the
exact computation in _kelly_window_summary().
"""

from __future__ import annotations

import statistics
from typing import Any

# ---------------------------------------------------------------------------
# Shim — mirrors _kelly_window_summary() logic exactly.
# If the production code changes, update both.
# ---------------------------------------------------------------------------

KELLY_HARD_BLOCK_THRESHOLD = -0.25
KELLY_MIN_TRADES = 10


def _kelly_window_summary(
    outcomes: list[float],
    *,
    hard_block_threshold: float = KELLY_HARD_BLOCK_THRESHOLD,
    min_trades: int = KELLY_MIN_TRADES,
) -> dict[str, Any]:
    """Pure-logic copy of SignalLoop._kelly_window_summary()."""
    n = len(outcomes)
    if n == 0:
        return {
            "n": 0,
            "n_wins": 0,
            "n_losses": 0,
            "win_rate": None,
            "projected_fraction": None,
            "activation_cliff_risk": None,
            "trades_until_activation": min_trades,
            "window_fully_drained": True,
        }
    wins = [o for o in outcomes if o > 0]
    losses = [o for o in outcomes if o < 0]
    win_rate: float | None = round(len(wins) / n, 3) if n >= 2 else None
    projected: float | None = None
    if wins and losses:
        p = len(wins) / n
        avg_win = statistics.mean(wins)
        avg_loss = abs(statistics.mean(losses))
        b = avg_win / max(avg_loss, 1e-9)
        raw = (p * b - (1 - p)) / max(b, 1e-9)
        projected = round(raw, 4)
    cliff_risk = (
        projected is not None and projected < hard_block_threshold
    )
    return {
        "n": n,
        "n_wins": len(wins),
        "n_losses": len(losses),
        "win_rate": win_rate,
        "projected_fraction": projected,
        "activation_cliff_risk": cliff_risk,
        "trades_until_activation": max(0, min_trades - n),
        "window_fully_drained": False,
    }


def _ic_block_detail(
    ticker_ic: dict[str, tuple[float, int]],
    *,
    min_n: int = 300,
    threshold: float = -0.05,
) -> list[dict[str, Any]]:
    """Pure-logic copy of the ic_block_detail comprehension."""
    return sorted(
        [
            {"ticker": t, "ic": round(ic, 4), "n": n}
            for t, (ic, n) in ticker_ic.items()
            if n >= min_n and ic < threshold
        ],
        key=lambda x: x["ic"],
    )


# ---------------------------------------------------------------------------
# Tests: empty window
# ---------------------------------------------------------------------------


def test_empty_window_structure():
    r = _kelly_window_summary([])
    assert r["n"] == 0
    assert r["n_wins"] == 0
    assert r["n_losses"] == 0
    assert r["win_rate"] is None
    assert r["projected_fraction"] is None
    assert r["activation_cliff_risk"] is None
    assert r["trades_until_activation"] == KELLY_MIN_TRADES
    assert r["window_fully_drained"] is True


def test_empty_window_fully_drained_true():
    r = _kelly_window_summary([])
    assert r["window_fully_drained"] is True


# ---------------------------------------------------------------------------
# Tests: partial window (n < KELLY_MIN_TRADES)
# ---------------------------------------------------------------------------


def test_partial_window_counts():
    outcomes = [0.01, -0.02, 0.015, -0.01, 0.008]
    r = _kelly_window_summary(outcomes)
    assert r["n"] == 5
    assert r["n_wins"] == 3
    assert r["n_losses"] == 2
    assert r["window_fully_drained"] is False
    assert r["trades_until_activation"] == 5  # 10 - 5


def test_partial_window_win_rate():
    outcomes = [0.01, -0.02, 0.015, -0.01, 0.008]
    r = _kelly_window_summary(outcomes)
    assert r["win_rate"] == round(3 / 5, 3)


def test_partial_window_projected_fraction_positive_edge():
    # Wins larger than losses → positive Kelly → no cliff risk
    outcomes = [0.03, -0.01, 0.03, -0.01, 0.03]
    r = _kelly_window_summary(outcomes)
    assert r["projected_fraction"] is not None
    assert r["projected_fraction"] > 0
    assert r["activation_cliff_risk"] is False


def test_partial_window_projected_fraction_negative_edge():
    # Losses larger than wins → negative projected fraction
    outcomes = [-0.05, 0.005, -0.05, 0.005, -0.05]
    r = _kelly_window_summary(outcomes)
    assert r["projected_fraction"] is not None
    assert r["projected_fraction"] < 0


def test_cliff_risk_true_when_projected_below_threshold():
    # Very bad edge: win 20%, avg_win=0.005, avg_loss=0.05 → f* << -0.25
    outcomes = [-0.05, -0.05, -0.05, -0.05, 0.005]
    r = _kelly_window_summary(outcomes)
    assert r["activation_cliff_risk"] is True


def test_cliff_risk_false_when_projected_above_threshold():
    outcomes = [0.03, 0.03, 0.03, -0.01, -0.01]
    r = _kelly_window_summary(outcomes)
    assert r["activation_cliff_risk"] is False


def test_cliff_risk_exactly_at_threshold():
    # projected fraction exactly equal to threshold → not less than → False
    # Construct: p=0.5, b=1 → f* = (0.5*1 - 0.5)/1 = 0.0 → not cliff
    outcomes = [0.01, -0.01, 0.01, -0.01]
    r = _kelly_window_summary(outcomes)
    # f* = 0 which is > -0.25, so no cliff
    assert r["activation_cliff_risk"] is False


# ---------------------------------------------------------------------------
# Tests: full window (n >= KELLY_MIN_TRADES)
# ---------------------------------------------------------------------------


def test_full_window_trades_until_activation_zero():
    outcomes = [0.01] * 5 + [-0.005] * 5  # exactly 10
    r = _kelly_window_summary(outcomes)
    assert r["trades_until_activation"] == 0


def test_full_window_win_rate_calculation():
    outcomes = [0.01] * 7 + [-0.005] * 3
    r = _kelly_window_summary(outcomes)
    assert r["win_rate"] == round(7 / 10, 3)


# ---------------------------------------------------------------------------
# Tests: single-outcome edge cases
# ---------------------------------------------------------------------------


def test_single_win_no_loss():
    outcomes = [0.02]
    r = _kelly_window_summary(outcomes)
    assert r["n"] == 1
    assert r["n_wins"] == 1
    assert r["n_losses"] == 0
    assert r["projected_fraction"] is None  # needs both wins and losses
    assert r["activation_cliff_risk"] is False
    assert r["win_rate"] is None  # n < 2


def test_single_loss_no_win():
    outcomes = [-0.02]
    r = _kelly_window_summary(outcomes)
    assert r["n"] == 1
    assert r["n_wins"] == 0
    assert r["n_losses"] == 1
    assert r["projected_fraction"] is None
    assert r["activation_cliff_risk"] is False


# ---------------------------------------------------------------------------
# Tests: ic_block_detail
# ---------------------------------------------------------------------------


def test_ic_block_detail_empty_when_no_tickers():
    assert _ic_block_detail({}) == []


def test_ic_block_detail_excludes_thin_samples():
    ticker_ic = {"AAPL": (-0.10, 50)}  # n=50 < min_n=300
    assert _ic_block_detail(ticker_ic) == []


def test_ic_block_detail_excludes_above_threshold():
    ticker_ic = {"AAPL": (0.03, 400)}  # ic=0.03 > -0.05
    assert _ic_block_detail(ticker_ic) == []


def test_ic_block_detail_includes_blocked_ticker():
    ticker_ic = {"AAPL": (-0.08, 400)}
    result = _ic_block_detail(ticker_ic)
    assert len(result) == 1
    assert result[0]["ticker"] == "AAPL"
    assert result[0]["ic"] == -0.08
    assert result[0]["n"] == 400


def test_ic_block_detail_sorted_worst_ic_first():
    ticker_ic = {
        "AAPL": (-0.06, 500),
        "TSLA": (-0.12, 600),
        "SPY": (-0.08, 400),
    }
    result = _ic_block_detail(ticker_ic)
    ics = [r["ic"] for r in result]
    assert ics == sorted(ics)  # ascending = worst first


def test_ic_block_detail_mixed_eligible_ineligible():
    ticker_ic = {
        "AAPL": (-0.08, 400),   # blocked
        "TSLA": (0.05, 500),    # above threshold
        "NVDA": (-0.09, 100),   # thin sample
        "SPY": (-0.11, 350),    # blocked
    }
    result = _ic_block_detail(ticker_ic)
    tickers = {r["ticker"] for r in result}
    assert tickers == {"AAPL", "SPY"}
    # SPY has worse IC, should be first
    assert result[0]["ticker"] == "SPY"
