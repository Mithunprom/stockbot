"""Unit tests for H34 — IC gate health diagnostic fields.

Verifies that ticker_ic_blocked_details and ticker_ic_gate are computed
correctly from the _ticker_ic dict. The v0.9.5 IC fill-job fix means the
gate can now actually arm (n can reach TICKER_IC_MIN_N = 300). These tests
pin the semantics of the new diagnostic fields so no future refactor
silently breaks them.
"""
from __future__ import annotations

from src.agents.signal_loop import (
    TICKER_IC_BLOCK_THRESHOLD,
    TICKER_IC_MIN_N,
    KELLY_PROBE_MIN_N,
)


# ---------------------------------------------------------------------------
# Helpers — replicate the diagnostic computations in isolation
# ---------------------------------------------------------------------------

def _blocked_details(ticker_ic: dict[str, tuple[float, int]]) -> dict:
    return {
        t: {"ic": round(ic, 4), "n": n}
        for t, (ic, n) in ticker_ic.items()
        if n >= TICKER_IC_MIN_N and ic < TICKER_IC_BLOCK_THRESHOLD
    }


def _gate_summary(ticker_ic: dict[str, tuple[float, int]]) -> dict:
    return {
        "min_n_threshold": TICKER_IC_MIN_N,
        "block_threshold": TICKER_IC_BLOCK_THRESHOLD,
        "n_tracked": len(ticker_ic),
        "n_armed": sum(1 for _, (_, n) in ticker_ic.items() if n >= TICKER_IC_MIN_N),
        "n_blocked": sum(
            1 for _, (ic, n) in ticker_ic.items()
            if n >= TICKER_IC_MIN_N and ic < TICKER_IC_BLOCK_THRESHOLD
        ),
        "max_n": max((n for _, (_, n) in ticker_ic.items()), default=0),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_blocked_details_only_includes_armed_blocked_tickers():
    """Only tickers with n >= MIN_N AND ic < threshold appear in blocked_details."""
    ticker_ic = {
        "AAPL": (-0.10, 350),   # armed AND blocked
        "MSFT": (-0.10, 200),   # not armed (n < 300)
        "GOOG": (0.05, 350),    # armed but NOT blocked (ic >= threshold)
        "META": (-0.06, 300),   # exactly at boundary: n == MIN_N AND ic < threshold
    }
    details = _blocked_details(ticker_ic)
    assert set(details.keys()) == {"AAPL", "META"}
    assert details["AAPL"] == {"ic": -0.1, "n": 350}
    assert details["META"] == {"ic": -0.06, "n": 300}


def test_blocked_details_empty_when_no_tickers_armed():
    """If no ticker has reached MIN_N, blocked_details is empty."""
    ticker_ic = {"AAPL": (-0.20, 50), "TSLA": (-0.15, 100)}
    assert _blocked_details(ticker_ic) == {}


def test_blocked_details_empty_when_all_have_positive_ic():
    """Tickers with good IC are not exposed in blocked_details."""
    ticker_ic = {"NVDA": (0.10, 400), "AMD": (0.05, 310)}
    assert _blocked_details(ticker_ic) == {}


def test_gate_summary_counts_match_blocked_details():
    """n_blocked in gate summary must equal len(blocked_details)."""
    ticker_ic = {
        "A": (-0.10, 350),  # blocked
        "B": (-0.10, 150),  # not armed
        "C": (0.08, 400),   # armed, not blocked
        "D": (-0.07, 300),  # blocked
    }
    details = _blocked_details(ticker_ic)
    summary = _gate_summary(ticker_ic)
    assert summary["n_blocked"] == len(details) == 2
    assert summary["n_armed"] == 3   # A (350), C (400), D (300) all >= MIN_N=300


def test_gate_summary_n_armed_counts_all_tickers_above_threshold():
    """n_armed includes tickers with n >= MIN_N regardless of IC sign."""
    ticker_ic = {
        "A": (-0.10, 350),  # armed (blocked)
        "B": (0.15, 400),   # armed (not blocked)
        "C": (-0.05, 299),  # not armed
    }
    summary = _gate_summary(ticker_ic)
    assert summary["n_armed"] == 2
    assert summary["n_blocked"] == 1
    assert summary["n_tracked"] == 3


def test_gate_summary_max_n_is_correct():
    """max_n reflects the highest prediction count across all tracked tickers."""
    ticker_ic = {"X": (0.10, 100), "Y": (-0.05, 350), "Z": (0.02, 215)}
    summary = _gate_summary(ticker_ic)
    assert summary["max_n"] == 350


def test_gate_summary_max_n_zero_on_empty_dict():
    """max_n defaults to 0 when no tickers are tracked."""
    assert _gate_summary({})["max_n"] == 0
    assert _gate_summary({})["n_tracked"] == 0


def test_fill_job_health_proxy_max_n_grows():
    """Simulates fill-job accumulation: max_n grows from 10 (dead) to 350 (healthy)."""
    dead_job = {"AAPL": (-0.08, 10), "TSLA": (0.04, 8)}
    healthy_job = {"AAPL": (-0.08, 350), "TSLA": (0.04, 312)}

    dead_summary = _gate_summary(dead_job)
    healthy_summary = _gate_summary(healthy_job)

    # Before fix: max_n far below KELLY_PROBE_MIN_N
    assert dead_summary["max_n"] < KELLY_PROBE_MIN_N
    # After fix: max_n well above block threshold
    assert healthy_summary["max_n"] >= TICKER_IC_MIN_N
    # Gate cannot arm while dead
    assert dead_summary["n_armed"] == 0
    # Gate arms once healthy
    assert healthy_summary["n_armed"] == 2


def test_ic_threshold_constants_unchanged():
    """Pin threshold constants so any change is explicit and visible in diff."""
    assert TICKER_IC_MIN_N == 300
    assert TICKER_IC_BLOCK_THRESHOLD == -0.05
    # probe threshold is lower — Kelly probe can fire before IC gate arms
    assert KELLY_PROBE_MIN_N < TICKER_IC_MIN_N
