"""H21: probe_ic_debug diagnostics field — shows why probes are blocked.

When Kelly is in probation and tickers_probe_eligible is empty, the owner
needs to know whether the issue is:
  (a) insufficient_predictions: n < TICKER_IC_MIN_N (300) → wait for accumulation
  (b) ic_below_threshold: n >= 300 but ic < 0.05 → model underperforming

Without this field you have to read source code to find out.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.agents.signal_loop import (
    KELLY_MIN_TRADES,
    KELLY_PROBATION_MIN_TICKER_IC,
    TICKER_IC_MIN_N,
    SignalLoop,
)
from src.execution.position_manager import PositionManager
from src.risk.circuit_breakers import CircuitBreakers


def _make_loop() -> SignalLoop:
    return SignalLoop(
        universe=["AAPL", "MSFT", "MU"],
        ensemble=MagicMock(),
        alpaca=MagicMock(),
        circuit_breakers=CircuitBreakers(),
        pos_manager=PositionManager(initial_portfolio=100_000.0),
        session_factory=MagicMock(),
        feature_cols=[f"f{i}" for i in range(30)],
    )


def _fake_settings():
    s = MagicMock()
    s.alpaca_mode = "paper"
    return s


def _enter_probation(loop: SignalLoop) -> None:
    """Push enough losing trades into the Kelly window to reach probation."""
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    loop._sizing_recent_outcomes = [
        (now - timedelta(hours=1), -0.01) for _ in range(KELLY_MIN_TRADES + 2)
    ]
    loop._update_kelly()
    assert loop._kelly_mode() == "probation"


@patch("src.config.get_settings", _fake_settings)
def test_probe_ic_debug_is_none_when_not_in_probation():
    """probe_ic_debug must be None when Kelly is healthy (normal mode)."""
    loop = _make_loop()
    # No outcomes → inactive mode
    assert loop._kelly_mode() == "inactive"
    summary = loop.get_portfolio_summary()
    assert summary["probe_ic_debug"] is None


@patch("src.config.get_settings", _fake_settings)
def test_probe_ic_debug_is_none_when_probation_cache_empty():
    """probe_ic_debug is None in probation when _ticker_ic_probe is unpopulated.

    The 30d probe IC cache is only refreshed asynchronously. On the first tick
    after entering probation it may still be empty. The field should be None
    rather than an empty list so the caller knows "no data yet" vs "data says
    nothing qualifies".
    """
    loop = _make_loop()
    _enter_probation(loop)
    loop._ticker_ic_probe = {}  # cache not yet populated
    summary = loop.get_portfolio_summary()
    assert summary["probe_ic_debug"] is None


@patch("src.config.get_settings", _fake_settings)
def test_probe_ic_debug_shows_top5_by_ic_descending():
    """Returns up to 5 tickers sorted by IC descending so the best candidates
    appear first.
    """
    loop = _make_loop()
    _enter_probation(loop)
    loop._ticker_ic_probe = {
        "AAPL": (0.03, TICKER_IC_MIN_N + 100),  # n ok, ic too low
        "MSFT": (0.08, TICKER_IC_MIN_N + 50),   # eligible!
        "MU":   (-0.02, TICKER_IC_MIN_N + 200), # negative ic
        "NVDA": (0.01, TICKER_IC_MIN_N - 50),   # n too low
        "TSLA": (0.06, TICKER_IC_MIN_N + 10),   # eligible!
        "AMZN": (0.04, TICKER_IC_MIN_N + 300),  # n ok, ic too low
    }
    summary = loop.get_portfolio_summary()
    debug = summary["probe_ic_debug"]

    assert debug is not None
    assert len(debug) == 5  # capped at 5

    # Top entry is highest IC
    assert debug[0]["ticker"] == "MSFT"
    assert debug[0]["ic"] == 0.08
    assert debug[0]["blocked_by"] == []  # eligible — no block

    # Second is TSLA (ic=0.06, eligible)
    assert debug[1]["ticker"] == "TSLA"
    assert debug[1]["blocked_by"] == []


@patch("src.config.get_settings", _fake_settings)
def test_probe_ic_debug_blocked_by_reasons():
    """blocked_by correctly distinguishes insufficient_predictions vs ic_below_threshold."""
    loop = _make_loop()
    _enter_probation(loop)
    loop._ticker_ic_probe = {
        "LOW_N":  (0.10, TICKER_IC_MIN_N - 1),  # fails on n
        "LOW_IC": (KELLY_PROBATION_MIN_TICKER_IC - 0.01, TICKER_IC_MIN_N + 100),  # fails on ic
        "OK":     (KELLY_PROBATION_MIN_TICKER_IC + 0.01, TICKER_IC_MIN_N + 50),   # eligible
    }
    summary = loop.get_portfolio_summary()
    debug = {row["ticker"]: row for row in summary["probe_ic_debug"]}

    assert debug["LOW_N"]["blocked_by"] == ["insufficient_predictions"]
    assert debug["LOW_IC"]["blocked_by"] == ["ic_below_threshold"]
    assert debug["OK"]["blocked_by"] == []
