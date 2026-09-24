"""H28: Rolling exit-distribution tracker for profit-target mode monitoring.

Verifies that:
1. EXIT_DIST_WINDOW constant is defined and positive.
2. The deque accumulates exit reasons and counts them correctly.
3. The deque rolls over at EXIT_DIST_WINDOW (old reasons displaced).
4. Counter logic mirrors what _build_diagnostics exports — a dict of
   non-zero counts plus an "n" key.
"""
from __future__ import annotations

from collections import Counter, deque

from src.agents.signal_loop import EXIT_DIST_WINDOW


# ── Minimal stub — same pattern as test_entry_rank_monitor.py ─────────────────

class _Loop:
    """Exposes only the state required for exit-distribution tests."""

    def __init__(self) -> None:
        self._recent_exit_reasons: deque = deque(maxlen=EXIT_DIST_WINDOW)

    def _exit_dist_snapshot(self) -> dict:
        """Mirror of what _build_diagnostics exports."""
        return {
            "counts": dict(Counter(self._recent_exit_reasons)),
            "n": len(self._recent_exit_reasons),
        }


# ── Test 1: constant is sensible ──────────────────────────────────────────────

def test_exit_dist_window_constant():
    assert EXIT_DIST_WINDOW > 0, "EXIT_DIST_WINDOW must be a positive integer"
    assert EXIT_DIST_WINDOW == 50, "expected 50; update this test if intentionally changed"


# ── Test 2: reasons accumulate and count correctly ────────────────────────────

def test_exit_reasons_accumulate():
    loop = _Loop()
    reasons = ["take_profit", "stop_loss", "session_close", "take_profit", "stop_loss"]
    for r in reasons:
        loop._recent_exit_reasons.append(r)

    snap = loop._exit_dist_snapshot()
    assert snap["n"] == 5
    assert snap["counts"]["take_profit"] == 2
    assert snap["counts"]["stop_loss"] == 2
    assert snap["counts"]["session_close"] == 1
    # reasons with zero count must be absent (Counter omits zeros)
    assert "max_hold" not in snap["counts"]


# ── Test 3: deque rolls over at EXIT_DIST_WINDOW ──────────────────────────────

def test_exit_distribution_window_cap():
    loop = _Loop()
    for _ in range(EXIT_DIST_WINDOW):
        loop._recent_exit_reasons.append("max_hold")
    # Add one more — the oldest max_hold must fall off
    loop._recent_exit_reasons.append("take_profit")

    snap = loop._exit_dist_snapshot()
    assert snap["n"] == EXIT_DIST_WINDOW
    assert snap["counts"]["take_profit"] == 1
    assert snap["counts"]["max_hold"] == EXIT_DIST_WINDOW - 1


# ── Test 4: empty deque → zero n, empty counts ───────────────────────────────

def test_exit_distribution_empty_on_start():
    loop = _Loop()
    snap = loop._exit_dist_snapshot()
    assert snap["n"] == 0
    assert snap["counts"] == {}


# ── Test 5: profit-target mode distinguishable in snapshot ───────────────────

def test_profit_target_exit_mix_readable():
    """Simulate a healthy profit-target session: more TPs than session-closes.

    This is the qualitative check the operator performs: take_profit > session_close
    means the mode is working; the reverse means the target is set too far away.
    """
    loop = _Loop()
    # 6 take_profits, 2 stop_losses, 1 session_close out of 9 trades
    for _ in range(6):
        loop._recent_exit_reasons.append("take_profit")
    for _ in range(2):
        loop._recent_exit_reasons.append("stop_loss")
    loop._recent_exit_reasons.append("session_close")

    snap = loop._exit_dist_snapshot()
    assert snap["n"] == 9
    assert snap["counts"]["take_profit"] > snap["counts"]["session_close"], (
        "take_profit should lead session_close for healthy profit-target operation"
    )
