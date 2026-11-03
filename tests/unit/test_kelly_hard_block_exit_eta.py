"""Unit tests for _kelly_hard_block_exit_eta() and its diagnostics surface.

The method projects the first calendar date on which the Kelly hard block is
expected to lift purely via window rolloff (no new winning trades assumed).
It is freeze-exempt: diagnostics only, no risk controls touched.

Key invariants tested:
  1. Returns None when the bot is NOT hard-blocked.
  2. Returns a near-future ISO date when the bot IS hard-blocked.
  3. The returned date is always within [tomorrow, lookback_days+2].
  4. The field appears in the diagnostics dict (signal_loop.py side).
  5. The field appears in the /diagnostics endpoint whitelist (main.py side).
  6. The field appears in main.py's /diagnostics endpoint whitelist.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from src.agents.signal_loop import (
    KELLY_HARD_BLOCK_THRESHOLD,
    KELLY_LOOKBACK_DAYS,
    KELLY_MIN_TRADES,
    SignalLoop,
)
from src.execution.position_manager import PositionManager
from src.risk.circuit_breakers import CircuitBreakers


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _make_loop() -> SignalLoop:
    loop = SignalLoop(
        universe=["AAPL"],
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


def _seed(loop: SignalLoop, outcomes: list[float], age_days: float = 1.0) -> None:
    stamp = datetime.now(timezone.utc) - timedelta(days=age_days)
    loop._sizing_recent_outcomes = [(stamp, o) for o in outcomes]


def _hard_blocking_window() -> list[float]:
    """Outcomes whose Kelly fraction lands well below KELLY_HARD_BLOCK_THRESHOLD.

    3 small wins + 9 larger losses  →  f* ≈ -4.3, far below -0.25.
    """
    return [0.005] * 3 + [-0.03] * 9


def _iso_date_re() -> re.Pattern:
    return re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ─── None path (not hard-blocked) ────────────────────────────────────────────

def test_returns_none_when_not_hard_blocked():
    """No projection needed when entries are flowing."""
    loop = _make_loop()
    _seed(loop, [0.02] * 8 + [-0.004] * 4)
    loop._update_kelly()
    assert loop._kelly_entries_blocked() is False
    assert loop._kelly_hard_block_exit_eta() is None


def test_returns_none_when_kelly_inactive():
    """Below KELLY_MIN_TRADES the governor is inactive — nothing to project."""
    loop = _make_loop()
    _seed(loop, [-0.05] * (KELLY_MIN_TRADES - 1))
    loop._update_kelly()
    assert loop._kelly_mode() == "inactive"
    assert loop._kelly_hard_block_exit_eta() is None


def test_returns_none_when_outcomes_empty():
    loop = _make_loop()
    loop._sizing_recent_outcomes = []
    assert loop._kelly_hard_block_exit_eta() is None


# ─── Date path (hard-blocked) ─────────────────────────────────────────────────

def test_returns_iso_date_when_hard_blocked():
    """Must return a well-formed ISO date when the bot is in hard-block."""
    loop = _make_loop()
    _seed(loop, _hard_blocking_window())
    loop._update_kelly()
    assert loop._kelly_entries_blocked() is True

    eta = loop._kelly_hard_block_exit_eta()
    assert eta is not None
    assert _iso_date_re().match(eta), f"Not an ISO date: {eta!r}"


def test_eta_is_in_the_future():
    """The projected date must be after today."""
    loop = _make_loop()
    _seed(loop, _hard_blocking_window())
    loop._update_kelly()
    assert loop._kelly_entries_blocked() is True

    eta = loop._kelly_hard_block_exit_eta()
    assert eta is not None
    eta_date = date.fromisoformat(eta)
    today = date.today()
    assert eta_date > today, f"ETA {eta_date} is not in the future (today={today})"


def test_eta_within_lookback_bound():
    """The projection cannot reach beyond LOOKBACK + 2 days."""
    loop = _make_loop()
    _seed(loop, _hard_blocking_window(), age_days=1.0)
    loop._update_kelly()
    assert loop._kelly_entries_blocked() is True

    eta = loop._kelly_hard_block_exit_eta()
    assert eta is not None
    eta_date = date.fromisoformat(eta)
    upper = date.today() + timedelta(days=KELLY_LOOKBACK_DAYS + 2)
    assert eta_date <= upper, f"ETA {eta_date} > upper bound {upper}"


# ─── Observability: the field surfaces through diagnostics ────────────────────

def test_field_present_in_signal_loop_diagnostics():
    """signal_loop.py must emit the field in its diagnostics export.

    Both blocked (non-None value) and unblocked (None) are tested to ensure
    the key is always present, not conditionally added.
    """
    loop = _make_loop()

    # Hard-blocked: field should be a non-None ISO date
    _seed(loop, _hard_blocking_window())
    loop._update_kelly()
    assert loop._kelly_entries_blocked() is True

    diag = loop.get_portfolio_summary()
    assert "kelly_hard_block_exit_eta" in diag, (
        "kelly_hard_block_exit_eta missing from get_portfolio_summary() dict"
    )
    assert diag["kelly_hard_block_exit_eta"] is not None

    # Not blocked: field should be None but present
    _seed(loop, [0.02] * 8 + [-0.004] * 4)
    loop._update_kelly()
    assert loop._kelly_entries_blocked() is False

    diag_unblocked = loop.get_portfolio_summary()
    assert "kelly_hard_block_exit_eta" in diag_unblocked
    assert diag_unblocked["kelly_hard_block_exit_eta"] is None


def test_field_in_main_py_diagnostics_whitelist():
    """main.py /diagnostics endpoint must forward kelly_hard_block_exit_eta.

    Grep the whitelist block in main.py to ensure the field is explicitly
    present. This guards against the class of bug where signal_loop.py exports
    a field but main.py silently drops it.
    """
    import pathlib
    src = pathlib.Path("main.py").read_text()
    assert "kelly_hard_block_exit_eta" in src, (
        "kelly_hard_block_exit_eta not found in main.py diagnostics whitelist. "
        "Add: \"kelly_hard_block_exit_eta\": summary.get(\"kelly_hard_block_exit_eta\"),"
    )
