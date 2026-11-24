"""H43: IC gate health diagnostic.

The 30-day IC window contains rows from BEFORE a model deploy, so a 100%
block_rate_among_armed under a near-zero IC model indicates window contamination
rather than genuine negative signal.  These tests pin:

1. The structure returned by _ic_gate_health_summary().
2. Contamination-signal detection: block_rate_among_armed = 1.0 when all armed
   tickers are barely below the threshold (noise range for OOS IC = +0.0078).
3. Clean-state behaviour: block_rate_among_armed = 0 when all tickers have
   positive IC.
4. Partial-arm state: unarmed tickers (n < 300) don't affect block rate.
5. The "blocked" flag matches _ticker_ic_blocked() (entry gate ≤ 0.0), not just
   the hard-block threshold (< -0.05).  Tickers in (-0.05, 0.0] are entry-blocked
   but were previously shown as "unblocked" — Bugbot d92ca99e.
"""
from __future__ import annotations

import pytest

from src.agents.signal_loop import (
    TICKER_IC_BLOCK_THRESHOLD,
    TICKER_IC_MIN_ENTRY,
    TICKER_IC_MIN_N,
    SignalLoop,
)


# ---------------------------------------------------------------------------
# Minimal stub — only the _ticker_ic attribute is needed for the helper.
# ---------------------------------------------------------------------------

def _make_loop_with_ic(ic_data: dict[str, tuple[float, int]]) -> SignalLoop:
    """Return a SignalLoop stub with _ticker_ic pre-populated."""
    loop = object.__new__(SignalLoop)
    loop._ticker_ic = ic_data
    return loop


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

def test_summary_has_required_keys():
    loop = _make_loop_with_ic({})
    result = loop._ic_gate_health_summary()
    required = {
        "n_tickers_in_cache", "n_tickers_armed", "n_tickers_unarmed",
        "n_tickers_blocked", "block_rate_among_armed", "mean_ic_armed",
        "armed_threshold", "entry_threshold", "hard_block_threshold", "armed_tickers",
    }
    assert required.issubset(result.keys())


def test_empty_cache_returns_none_rate():
    loop = _make_loop_with_ic({})
    result = loop._ic_gate_health_summary()
    assert result["n_tickers_in_cache"] == 0
    assert result["n_tickers_armed"] == 0
    assert result["block_rate_among_armed"] is None
    assert result["mean_ic_armed"] is None
    assert result["armed_tickers"] == {}


# ---------------------------------------------------------------------------
# Contamination signal: all armed tickers barely below threshold
# ---------------------------------------------------------------------------

def test_full_block_rate_signals_contamination():
    """block_rate_among_armed = 1.0 when all armed tickers have IC at or below
    TICKER_IC_MIN_ENTRY (≤ 0.0) — the signature of a leaky-model contamination
    rather than genuine per-ticker failure on a +0.0078-IC model.
    """
    n = TICKER_IC_MIN_N + 50
    # IC slightly below 0.0 (the entry gate), consistent with noise around ~0.
    ic_data = {f"TKR{i}": (-0.01 - i * 0.001, n) for i in range(22)}
    loop = _make_loop_with_ic(ic_data)
    result = loop._ic_gate_health_summary()

    assert result["n_tickers_armed"] == 22
    assert result["n_tickers_blocked"] == 22
    assert result["block_rate_among_armed"] == 1.0


def test_entry_gate_gap_tickers_are_blocked():
    """Tickers with IC in (-0.05, 0.0] are entry-blocked but were missed by the
    old implementation that used ic < TICKER_IC_BLOCK_THRESHOLD (-0.05).

    _ticker_ic_blocked() uses ic <= TICKER_IC_MIN_ENTRY (0.0), so the diagnostic
    must match — Bugbot finding d92ca99e.
    """
    n = TICKER_IC_MIN_N + 1
    ic_data = {
        "GAP_NEG": (-0.03, n),   # -0.05 < IC ≤ 0.0 → entry-blocked (old code missed)
        "AT_ZERO": (0.0, n),     # IC == 0.0 exactly → entry-blocked (≤ 0.0)
        "POS":     (0.01, n),    # IC > 0.0 → NOT blocked
    }
    loop = _make_loop_with_ic(ic_data)
    result = loop._ic_gate_health_summary()

    assert result["n_tickers_blocked"] == 2
    assert result["armed_tickers"]["GAP_NEG"]["blocked"] is True
    assert result["armed_tickers"]["AT_ZERO"]["blocked"] is True
    assert result["armed_tickers"]["POS"]["blocked"] is False


def test_zero_block_rate_when_all_ic_positive():
    n = TICKER_IC_MIN_N + 10
    ic_data = {f"TKR{i}": (0.05 + i * 0.001, n) for i in range(10)}
    loop = _make_loop_with_ic(ic_data)
    result = loop._ic_gate_health_summary()

    assert result["n_tickers_armed"] == 10
    assert result["n_tickers_blocked"] == 0
    assert result["block_rate_among_armed"] == 0.0


# ---------------------------------------------------------------------------
# Unarmed tickers do not affect the block rate
# ---------------------------------------------------------------------------

def test_unarmed_tickers_excluded_from_block_rate():
    armed_n = TICKER_IC_MIN_N + 1
    unarmed_n = TICKER_IC_MIN_N - 1
    ic_data = {
        "ARMED_POS": (0.10, armed_n),     # armed, not blocked
        "ARMED_NEG": (-0.10, armed_n),    # armed, blocked
        "UNARMED_NEG": (-0.99, unarmed_n),  # unarmed — must not count
    }
    loop = _make_loop_with_ic(ic_data)
    result = loop._ic_gate_health_summary()

    assert result["n_tickers_in_cache"] == 3
    assert result["n_tickers_armed"] == 2
    assert result["n_tickers_unarmed"] == 1
    assert result["n_tickers_blocked"] == 1
    assert result["block_rate_among_armed"] == pytest.approx(0.5, abs=0.001)
    assert "UNARMED_NEG" not in result["armed_tickers"]


# ---------------------------------------------------------------------------
# Threshold constants are reported accurately
# ---------------------------------------------------------------------------

def test_reports_correct_thresholds():
    loop = _make_loop_with_ic({})
    result = loop._ic_gate_health_summary()
    assert result["armed_threshold"] == TICKER_IC_MIN_N
    assert result["entry_threshold"] == TICKER_IC_MIN_ENTRY
    assert result["hard_block_threshold"] == TICKER_IC_BLOCK_THRESHOLD


# ---------------------------------------------------------------------------
# armed_tickers detail sorted ascending by IC (worst first)
# ---------------------------------------------------------------------------

def test_armed_tickers_sorted_ascending_by_ic():
    n = TICKER_IC_MIN_N + 1
    ic_data = {
        "BEST": (0.20, n),
        "MID":  (0.00, n),
        "WORST": (-0.10, n),
    }
    loop = _make_loop_with_ic(ic_data)
    result = loop._ic_gate_health_summary()
    keys = list(result["armed_tickers"].keys())
    ics = [result["armed_tickers"][k]["ic"] for k in keys]
    assert ics == sorted(ics), "armed_tickers must be sorted ascending by IC"


def test_armed_tickers_detail_has_blocked_flag():
    n = TICKER_IC_MIN_N + 1
    ic_data = {
        "POS": (0.05, n),
        "NEG": (-0.10, n),
    }
    loop = _make_loop_with_ic(ic_data)
    result = loop._ic_gate_health_summary()
    assert result["armed_tickers"]["POS"]["blocked"] is False
    assert result["armed_tickers"]["NEG"]["blocked"] is True


# ---------------------------------------------------------------------------
# mean_ic_armed
# ---------------------------------------------------------------------------

def test_mean_ic_armed_is_correct():
    n = TICKER_IC_MIN_N + 1
    ic_data = {"A": (0.10, n), "B": (-0.10, n)}
    loop = _make_loop_with_ic(ic_data)
    result = loop._ic_gate_health_summary()
    assert result["mean_ic_armed"] == pytest.approx(0.0, abs=0.001)
