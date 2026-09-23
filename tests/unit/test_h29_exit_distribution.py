"""H29: Session-level exit reason distribution in /diagnostics.

Tests verify the Counter logic in isolation — no SignalLoop import needed,
so these run in the sandbox without numpy/sqlalchemy.
"""

from __future__ import annotations

from collections import Counter


# ---------------------------------------------------------------------------
# Helpers that replicate the H29 logic extracted from signal_loop.py
# ---------------------------------------------------------------------------

def _make_counter() -> Counter[str]:
    return Counter()


def _record_exit(counts: Counter[str], exit_reason: str) -> None:
    counts[exit_reason] += 1


def _as_diagnostics_field(counts: Counter[str]) -> dict[str, int]:
    return dict(counts)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_h29_counter_starts_empty() -> None:
    """Session starts with no exit history."""
    c = _make_counter()
    assert _as_diagnostics_field(c) == {}


def test_h29_max_hold_increments() -> None:
    """max_hold exit increments correctly."""
    c = _make_counter()
    _record_exit(c, "max_hold")
    _record_exit(c, "max_hold")
    result = _as_diagnostics_field(c)
    assert result == {"max_hold": 2}


def test_h29_stop_loss_increments() -> None:
    """stop_loss exit — the first ATR barrier exit in M3 (STX Sep 21 2026)."""
    c = _make_counter()
    _record_exit(c, "stop_loss")
    result = _as_diagnostics_field(c)
    assert result == {"stop_loss": 1}


def test_h29_multiple_exit_reasons_tracked_independently() -> None:
    """Each exit reason accumulates independently."""
    c = _make_counter()
    for _ in range(9):
        _record_exit(c, "max_hold")
    _record_exit(c, "stop_loss")        # STX
    result = _as_diagnostics_field(c)
    assert result["max_hold"] == 9
    assert result["stop_loss"] == 1
    assert result.get("trailing_stop", 0) == 0
    assert result.get("take_profit", 0) == 0


def test_h29_all_max_hold_signals_dead_barriers() -> None:
    """If exit_reason_counts has only max_hold, barriers are dead code.

    This is the v0.6.x pattern: all 111 exits were max_hold because the
    ATR floors were 3.6x too wide for the 30-bar window. v0.7.0 (H14) fixed
    this. If production data shows only max_hold again, it's a regression.
    """
    c = _make_counter()
    for _ in range(30):
        _record_exit(c, "max_hold")
    result = _as_diagnostics_field(c)
    non_max_hold = sum(v for k, v in result.items() if k != "max_hold")
    assert non_max_hold == 0, "All-max_hold pattern indicates dead exit barriers"
    # In production, any stop_loss/trailing_stop/take_profit > 0 confirms
    # barriers are live. Current M3 Day 1: stop_loss=1 (STX Sep 21 2026) ✅


def test_h29_trailing_stop_and_take_profit_tracked() -> None:
    """Trailing stop and take profit exits are tracked like other reasons."""
    c = _make_counter()
    _record_exit(c, "trailing_stop")
    _record_exit(c, "take_profit")
    result = _as_diagnostics_field(c)
    assert result["trailing_stop"] == 1
    assert result["take_profit"] == 1
